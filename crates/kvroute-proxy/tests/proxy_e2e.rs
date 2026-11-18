//! End-to-end: real proxy + real mock vLLM workers over local TCP.

use std::collections::HashSet;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::response::Response;
use axum::routing::post;
use bytes::Bytes;
use futures_util::{stream, StreamExt};
use kvroute_core::{BlockHash, CacheEvent, WorkerId, BLOCK_SIZE};
use kvroute_proxy::proxy::{request_blocks, HDR_MATCHED, HDR_WORKER};
use kvroute_proxy::{app, spawn_event_actor, AppState, ProxyConfig};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

const CHUNKS: usize = 5;

/// Mock vLLM worker: computes the same block hashes, caches them, publishes
/// `BlockStored` for new blocks, then answers (optionally as SSE).
struct Mock {
    id: u16,
    proxy: String,
    cache: Mutex<HashSet<BlockHash>>,
    client: reqwest::Client,
    /// When set, SSE streams pause after the first chunk until a permit is added.
    gate: Option<Arc<Semaphore>>,
}

async fn mock_handler(State(m): State<Arc<Mock>>, body: Bytes) -> Response {
    let req: Value = serde_json::from_slice(&body).unwrap();
    let (hashes, _) = request_blocks(&req).unwrap();
    let new: Vec<BlockHash> = {
        let mut cache = m.cache.lock().unwrap();
        hashes.into_iter().filter(|h| cache.insert(*h)).collect()
    };
    if !new.is_empty() {
        let ev = [CacheEvent::BlockStored {
            worker: WorkerId(m.id),
            hashes: new,
            now_ms: 0,
        }];
        let r = m
            .client
            .post(format!("{}/internal/kv-events", m.proxy))
            .json(&ev)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success());
    }

    if req.get("stream").and_then(Value::as_bool) != Some(true) {
        let body =
            json!({ "object": "text_completion", "worker": m.id, "choices": [{ "text": "ok" }] });
        return Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
    }

    let gate = m.gate.clone();
    let id = m.id;
    let chunks = stream::unfold(0usize, move |i| {
        let gate = gate.clone();
        async move {
            if i > CHUNKS {
                return None;
            }
            if i == 1 {
                if let Some(g) = gate {
                    g.acquire().await.unwrap().forget();
                }
            }
            let line = if i == CHUNKS {
                "data: [DONE]\n\n".to_string()
            } else {
                format!(
                    "data: {}\n\n",
                    json!({ "worker": id, "choices": [{ "text": format!("t{i}") }] })
                )
            };
            Some((Ok::<_, std::io::Error>(Bytes::from(line)), i + 1))
        }
    });
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(chunks))
        .unwrap()
}

struct Harness {
    url: String,
    state: Arc<AppState>,
    mocks: Vec<Arc<Mock>>,
    gate: Arc<Semaphore>,
    http: reqwest::Client,
}

async fn start(
    workers: usize,
    gated: bool,
    actor: bool,
    tweak: impl FnOnce(&mut ProxyConfig),
) -> Harness {
    let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", proxy_listener.local_addr().unwrap());
    let gate = Arc::new(Semaphore::new(0));

    let mut mocks = Vec::new();
    let mut urls = Vec::new();
    for id in 0..workers as u16 {
        let m = Arc::new(Mock {
            id,
            proxy: url.clone(),
            cache: Mutex::default(),
            client: reqwest::Client::new(),
            gate: gated.then(|| gate.clone()),
        });
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        urls.push(format!("http://{}", l.local_addr().unwrap()));
        let svc = axum::Router::new()
            .route("/v1/completions", post(mock_handler))
            .route("/v1/chat/completions", post(mock_handler))
            .with_state(m.clone());
        tokio::spawn(async move { axum::serve(l, svc).await.unwrap() });
        mocks.push(m);
    }

    let mut cfg = ProxyConfig {
        workers: urls,
        ..ProxyConfig::default()
    };
    tweak(&mut cfg);
    let (state, rx) = AppState::new(&cfg).unwrap();
    if actor {
        spawn_event_actor(&state, rx);
    } else {
        // Keep the receiver alive but never drain it, so the queue fills.
        std::mem::forget(rx);
    }
    let svc = app(state.clone());
    tokio::spawn(async move { axum::serve(proxy_listener, svc).await.unwrap() });
    Harness {
        url,
        state,
        mocks,
        gate,
        http: reqwest::Client::new(),
    }
}

impl Harness {
    async fn post(&self, path: &str, body: Value) -> reqwest::Response {
        self.http
            .post(format!("{}{path}", self.url))
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    async fn worker_metrics(&self, body: Value) {
        let r = self.post("/internal/worker-metrics", body).await;
        assert_eq!(r.status(), 204);
    }

    async fn metrics(&self) -> String {
        self.http
            .get(format!("{}/metrics", self.url))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }

    fn queue_depth(&self, w: usize) -> usize {
        self.state.workers[w].queue_depth.load(Relaxed)
    }

    /// Polls until `cond` holds, failing after 5s instead of hanging.
    async fn until(&self, what: &str, cond: impl Fn(&Self) -> bool) {
        for _ in 0..500 {
            if cond(self) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for {what}");
    }
}

fn header_usize(r: &reqwest::Response, name: &str) -> usize {
    r.headers()[name].to_str().unwrap().parse().unwrap()
}

/// `blocks` full blocks of prefix `seed`, followed by `extra` fresh tokens.
fn tokens(seed: u32, blocks: usize, extra: usize) -> Vec<u32> {
    let mut t: Vec<u32> = (0..(blocks * BLOCK_SIZE) as u32)
        .map(|i| seed * 1_000_000 + i)
        .collect();
    t.extend((0..extra as u32).map(|i| 999_000_000 + i));
    t
}

/// Prometheus sample value for an exact series name (with labels, if any).
fn sample(metrics: &str, series: &str) -> u64 {
    metrics
        .lines()
        .find_map(|l| l.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
        .unwrap_or_else(|| panic!("{series} not in metrics:\n{metrics}"))
}

#[tokio::test]
async fn repeated_prefix_routes_to_warm_worker() {
    let h = start(3, false, true, |_| {}).await;
    // Unreported capacity counts as full KV, so workers report it at startup.
    for w in 0..3 {
        h.worker_metrics(json!({ "worker": w, "kv_capacity_blocks": 100 }))
            .await;
    }

    // Cold: all idle, ties go to worker 0.
    let r = h
        .post("/v1/completions", json!({ "token_ids": tokens(1, 4, 10) }))
        .await;
    assert_eq!(r.status(), 200);
    assert_eq!(
        (header_usize(&r, HDR_WORKER), header_usize(&r, HDR_MATCHED)),
        (0, 0)
    );
    r.text().await.unwrap();

    // Make worker 0 look busy so prefix B lands elsewhere, then clear it.
    h.worker_metrics(json!({ "worker": 0, "kv_used_blocks": 80, "kv_capacity_blocks": 100 }))
        .await;
    let r = h
        .post("/v1/completions", json!({ "token_ids": tokens(2, 3, 0) }))
        .await;
    let b_worker = header_usize(&r, HDR_WORKER);
    assert_ne!(b_worker, 0);
    r.text().await.unwrap();
    h.worker_metrics(json!({ "worker": 0, "kv_used_blocks": 0 }))
        .await;
    h.until("KV events indexed", |h| {
        h.state.router.load().index.len() == 7
    })
    .await;

    // Warm: each prefix follows its cache, beating the id tie-break.
    // 3 * 512 + 600 tokens = 4 full blocks, 3 of them cached.
    let r = h
        .post(
            "/v1/completions",
            json!({ "token_ids": tokens(2, 3, 600), "max_tokens": 8 }),
        )
        .await;
    assert_eq!(
        (header_usize(&r, HDR_WORKER), header_usize(&r, HDR_MATCHED)),
        (b_worker, 3)
    );
    r.text().await.unwrap();
    let r = h
        .post("/v1/completions", json!({ "token_ids": tokens(1, 4, 50) }))
        .await;
    assert_eq!(
        (header_usize(&r, HDR_WORKER), header_usize(&r, HDR_MATCHED)),
        (0, 4)
    );
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["worker"], 0, "response came from the routed backend");

    // Chat requests hash role-tagged text: turn 2 reuses turn 1's blocks.
    let sys = json!({ "role": "system", "content": "You are a planner. ".repeat(600) });
    let r = h
        .post("/v1/chat/completions", json!({ "messages": [sys] }))
        .await;
    let chat_worker = header_usize(&r, HDR_WORKER);
    r.text().await.unwrap();
    let n = request_blocks(&json!({ "messages": [sys] }))
        .unwrap()
        .0
        .len();
    assert!(n >= 2);
    // 7 earlier blocks + B's new 4th block + the chat prefix.
    h.until("chat blocks indexed", |h| {
        h.state.router.load().index.len() == 8 + n
    })
    .await;
    let turn2 = json!({ "messages": [sys, { "role": "user", "content": "next step?" }] });
    let n2 = request_blocks(&turn2).unwrap().0.len();
    let r = h.post("/v1/chat/completions", turn2).await;
    assert_eq!(
        (header_usize(&r, HDR_WORKER), header_usize(&r, HDR_MATCHED)),
        (chat_worker, n)
    );

    for w in 0..3 {
        h.until("queue drained", |h| h.queue_depth(w) == 0).await;
    }
    let m = h.metrics().await;
    assert_eq!(sample(&m, "kvroute_requests_total"), 6);
    assert_eq!(
        sample(&m, "kvroute_prefix_matched_blocks_total"),
        (3 + 4 + n) as u64
    );
    assert_eq!(
        sample(&m, "kvroute_prefix_total_blocks_total"),
        (4 + 3 + 4 + 4 + n + n2) as u64
    );
    // Backends' true caches agree with where the router sent each prefix.
    let cached = |w: usize| h.mocks[w].cache.lock().unwrap().len();
    let chat_on = |w: usize| if chat_worker == w { n2 } else { 0 };
    assert_eq!(cached(0), 4 + chat_on(0));
    assert_eq!(cached(b_worker), 4 + chat_on(b_worker));
}

#[tokio::test]
async fn sse_streams_through_and_guard_releases() {
    let h = start(2, true, true, |_| {}).await;
    let req = json!({ "token_ids": tokens(3, 1, 0), "stream": true });

    // First chunk arrives while the backend is still holding the stream open,
    // so the proxy is not buffering. The request is counted in flight.
    let r = h.post("/v1/completions", req.clone()).await;
    assert_eq!(r.headers()[header::CONTENT_TYPE], "text/event-stream");
    let w = header_usize(&r, HDR_WORKER);
    let mut body = r.bytes_stream();
    let first = tokio::time::timeout(Duration::from_secs(5), body.next())
        .await
        .expect("first chunk")
        .unwrap()
        .unwrap();
    assert!(first.starts_with(b"data: {"));
    assert_eq!(h.queue_depth(w), 1);
    assert!(h.state.workers[w].in_flight_tokens.load(Relaxed) > 0);

    h.gate.add_permits(1);
    let mut all = first.to_vec();
    while let Some(c) = body.next().await {
        all.extend_from_slice(&c.unwrap());
    }
    let text = String::from_utf8(all).unwrap();
    let events: Vec<&str> = text.split("\n\n").filter(|e| !e.is_empty()).collect();
    assert_eq!(events.len(), CHUNKS + 1);
    assert!(events.iter().all(|e| e.starts_with("data: ")));
    assert_eq!(*events.last().unwrap(), "data: [DONE]");
    h.until("guard released after completion", |h| h.queue_depth(w) == 0)
        .await;
    assert_eq!(h.state.workers[w].in_flight_tokens.load(Relaxed), 0);

    // Client aborts mid-stream: the backend never finishes, yet the guard drops.
    let r = h.post("/v1/completions", req).await;
    let w = header_usize(&r, HDR_WORKER);
    let mut body = r.bytes_stream();
    body.next().await.unwrap().unwrap();
    assert_eq!(h.queue_depth(w), 1);
    drop(body);
    h.until("guard released after client abort", |h| {
        h.queue_depth(w) == 0
    })
    .await;
    assert_eq!(
        h.state.permits.available_permits(),
        ProxyConfig::default().max_concurrent
    );
    h.gate.add_permits(100);
}

#[tokio::test]
async fn admission_and_saturation_return_503_with_metrics() {
    let h = start(2, true, true, |c| c.max_concurrent = 1).await;
    for w in 0..2 {
        h.worker_metrics(json!({ "worker": w, "kv_used_blocks": 88, "kv_capacity_blocks": 100 }))
            .await;
    }

    // 88 + 4 uncached blocks = 92% > 90% on both workers.
    let r = h
        .post("/v1/completions", json!({ "token_ids": tokens(4, 4, 0) }))
        .await;
    assert_eq!(r.status(), 503);
    assert_eq!(r.headers()[header::RETRY_AFTER], "1");
    let err: Value = r.json().await.unwrap();
    assert_eq!(err["error"]["type"], "admission_deferred");

    // A 1-block request fits (89%) and is admitted.
    let r = h
        .post(
            "/v1/completions",
            json!({ "token_ids": tokens(5, 1, 0), "stream": true }),
        )
        .await;
    assert_eq!(r.status(), 200);
    let mut held = r.bytes_stream();
    held.next().await.unwrap().unwrap();

    // The open stream holds the only permit: next request is shed.
    let r = h
        .post("/v1/completions", json!({ "token_ids": tokens(6, 1, 0) }))
        .await;
    assert_eq!(r.status(), 503);
    assert_eq!(
        r.json::<Value>().await.unwrap()["error"]["type"],
        "overloaded"
    );
    h.gate.add_permits(100);
    while held.next().await.is_some() {}

    let bad = h.post("/v1/completions", json!({ "nope": 1 })).await;
    assert_eq!(bad.status(), 400);

    let m = h.metrics().await;
    assert_eq!(sample(&m, "kvroute_requests_total"), 1);
    assert_eq!(sample(&m, "kvroute_admission_deferrals_total"), 1);
    assert_eq!(sample(&m, "kvroute_saturated_total"), 1);
    assert_eq!(sample(&m, "kvroute_prefix_total_blocks_total"), 1);
    assert_eq!(sample(&m, "kvroute_events_dropped_total"), 0);
    assert_eq!(
        sample(&m, "kvroute_worker_kv_used_blocks{worker=\"1\"}"),
        88
    );
    h.until("drained", |h| h.queue_depth(0) + h.queue_depth(1) == 0)
        .await;
    assert_eq!(
        sample(
            &h.metrics().await,
            "kvroute_worker_queue_depth{worker=\"0\"}"
        ),
        0
    );

    // Health follows worker reports.
    let health = |h: &Harness| h.http.get(format!("{}/healthz", h.url)).send();
    assert_eq!(health(&h).await.unwrap().status(), 200);
    for w in 0..2 {
        h.worker_metrics(json!({ "worker": w, "healthy": false }))
            .await;
    }
    assert_eq!(health(&h).await.unwrap().status(), 503);
    let r = h
        .post("/v1/completions", json!({ "token_ids": tokens(7, 1, 0) }))
        .await;
    assert_eq!(r.status(), 503);
    h.worker_metrics(json!({ "worker": 1, "healthy": true }))
        .await;
    assert_eq!(health(&h).await.unwrap().status(), 200);
}

#[tokio::test]
async fn reset_index_and_event_backpressure() {
    let h = start(2, false, true, |_| {}).await;
    let ev = json!([{ "BlockStored": { "worker": 1, "hashes": [11, 12], "now_ms": 0 } }]);
    assert_eq!(h.post("/internal/kv-events", ev).await.status(), 202);
    h.until("indexed", |h| h.state.router.load().index.len() == 2)
        .await;
    let unknown = json!([{ "AllBlocksCleared": { "worker": 9 } }]);
    assert_eq!(h.post("/internal/kv-events", unknown).await.status(), 400);

    assert_eq!(
        h.post("/internal/reset-index", json!(null)).await.status(),
        202
    );
    h.until("index wiped", |h| h.state.router.load().index.is_empty())
        .await;

    // No actor draining a 2-slot queue: the third batch is dropped and counted.
    let h = start(1, false, false, |c| c.event_queue = 2).await;
    let ev = json!([
        { "BlockStored": { "worker": 0, "hashes": [1], "now_ms": 0 } },
        { "BlockRemoved": { "worker": 0, "hashes": [1] } }
    ]);
    for want in [202, 202, 503] {
        assert_eq!(
            h.post("/internal/kv-events", ev.clone()).await.status(),
            want
        );
    }
    assert_eq!(
        sample(&h.metrics().await, "kvroute_events_dropped_total"),
        2
    );
}
