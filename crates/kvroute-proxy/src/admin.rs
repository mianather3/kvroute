//! Control plane: KV-event ingestion, worker metrics, index reset, health,
//! and Prometheus metrics. None of these touch the routing hot path's locks
//! (there are none); index changes go through the single-writer actor.

use std::fmt::Write as _;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use kvroute_core::{CacheEvent, WorkerId};
use serde::Deserialize;
use tokio::sync::mpsc::error::TrySendError;

use crate::state::{AppState, IndexMsg, WorkerState};

fn event_worker(ev: &CacheEvent) -> WorkerId {
    match ev {
        CacheEvent::BlockStored { worker, .. }
        | CacheEvent::BlockRemoved { worker, .. }
        | CacheEvent::AllBlocksCleared { worker } => *worker,
    }
}

/// Never blocks the publisher: if the index channel is full the batch is
/// dropped and counted, and the router runs on a staler index.
pub async fn kv_events(
    State(s): State<Arc<AppState>>,
    Json(batch): Json<Vec<CacheEvent>>,
) -> Response {
    if let Some(bad) = batch
        .iter()
        .map(event_worker)
        .find(|w| s.worker(*w).is_none())
    {
        return (StatusCode::BAD_REQUEST, format!("unknown worker {}", bad.0)).into_response();
    }
    let n = batch.len() as u64;
    match s.tx.try_send(IndexMsg::Events(batch)) {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(TrySendError::Full(_) | TrySendError::Closed(_)) => {
            s.counters.events_dropped.fetch_add(n, Relaxed);
            (
                StatusCode::SERVICE_UNAVAILABLE,
                [(header::RETRY_AFTER, "1")],
                "event queue full",
            )
                .into_response()
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct WorkerUpdate {
    pub worker: u16,
    pub kv_used_blocks: Option<usize>,
    pub kv_capacity_blocks: Option<usize>,
    pub healthy: Option<bool>,
}

pub async fn worker_metrics(
    State(s): State<Arc<AppState>>,
    Json(u): Json<WorkerUpdate>,
) -> StatusCode {
    let Some(w) = s.worker(WorkerId(u.worker)) else {
        return StatusCode::NOT_FOUND;
    };
    if let Some(v) = u.kv_used_blocks {
        w.kv_used_blocks.store(v, Relaxed);
    }
    if let Some(v) = u.kv_capacity_blocks {
        w.kv_capacity_blocks.store(v, Relaxed);
    }
    if let Some(v) = u.healthy {
        w.healthy.store(v, Relaxed);
    }
    StatusCode::NO_CONTENT
}

/// Waits for channel space: a reset must not be silently dropped.
pub async fn reset_index(State(s): State<Arc<AppState>>) -> StatusCode {
    match s.tx.send(IndexMsg::ResetIndex).await {
        Ok(()) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

pub async fn healthz(State(s): State<Arc<AppState>>) -> (StatusCode, &'static str) {
    if s.workers.iter().any(|w| w.healthy.load(Relaxed)) {
        (StatusCode::OK, "ok\n")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "no healthy workers\n")
    }
}

pub async fn metrics(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    let ct = [(header::CONTENT_TYPE, "text/plain; version=0.0.4")];
    (ct, render_metrics(&s))
}

pub fn render_metrics(s: &AppState) -> String {
    let c = &s.counters;
    let mut out = String::new();
    let mut scalar = |name: &str, kind: &str, help: &str, v: u64| {
        let _ = writeln!(
            out,
            "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}"
        );
    };
    scalar(
        "kvroute_requests_total",
        "counter",
        "Requests routed to a worker.",
        c.requests.load(Relaxed),
    );
    scalar(
        "kvroute_prefix_matched_blocks_total",
        "counter",
        "Prefix blocks the router expected to hit in worker caches.",
        c.matched_blocks.load(Relaxed),
    );
    scalar(
        "kvroute_prefix_total_blocks_total",
        "counter",
        "Full prompt blocks across routed requests.",
        c.total_blocks.load(Relaxed),
    );
    scalar(
        "kvroute_admission_deferrals_total",
        "counter",
        "Requests rejected by KV-budget admission control.",
        c.deferrals.load(Relaxed),
    );
    scalar(
        "kvroute_saturated_total",
        "counter",
        "Requests rejected at the global concurrency limit.",
        c.saturated.load(Relaxed),
    );
    scalar(
        "kvroute_upstream_errors_total",
        "counter",
        "Requests whose backend could not be reached.",
        c.upstream_errors.load(Relaxed),
    );
    scalar(
        "kvroute_events_dropped_total",
        "counter",
        "KV cache events dropped because the index queue was full.",
        c.events_dropped.load(Relaxed),
    );
    scalar(
        "kvroute_index_blocks",
        "gauge",
        "Distinct block hashes in the published prefix index.",
        s.router.load().index.len() as u64,
    );

    type Gauge = fn(&WorkerState) -> u64;
    let gauges: [(&str, &str, Gauge); 5] = [
        (
            "kvroute_worker_queue_depth",
            "Requests in flight on the worker.",
            |w| w.queue_depth.load(Relaxed) as u64,
        ),
        (
            "kvroute_worker_in_flight_tokens",
            "Prompt plus max output tokens in flight.",
            |w| w.in_flight_tokens.load(Relaxed) as u64,
        ),
        (
            "kvroute_worker_kv_used_blocks",
            "Last reported committed KV blocks.",
            |w| w.kv_used_blocks.load(Relaxed) as u64,
        ),
        (
            "kvroute_worker_kv_capacity_blocks",
            "Last reported KV capacity in blocks.",
            |w| w.kv_capacity_blocks.load(Relaxed) as u64,
        ),
        (
            "kvroute_worker_healthy",
            "1 if the worker is routable.",
            |w| w.healthy.load(Relaxed) as u64,
        ),
    ];
    for (name, help, get) in gauges {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} gauge");
        for w in &s.workers {
            let _ = writeln!(out, "{name}{{worker=\"{}\"}} {}", w.id.0, get(w));
        }
    }
    out
}
