//! OpenAI-compatible request path: hash the prompt, route on a lock-free
//! snapshot, forward, and stream the backend response straight through.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use futures_util::StreamExt;
use kvroute_core::{
    hash_prefix_blocks, AdmissionDecision, BlockHash, DeferReason, WorkerId, BLOCK_SIZE,
};
use serde_json::{json, Value};
use tokio::sync::OwnedSemaphorePermit;

use crate::state::AppState;

pub const HDR_WORKER: &str = "x-kvroute-worker";
pub const HDR_MATCHED: &str = "x-kvroute-matched-blocks";

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("proxy at max concurrency")]
    Saturated,
    #[error("admission deferred: {0:?}")]
    Deferred(DeferReason),
    #[error("upstream worker {worker} failed: {source}")]
    Upstream {
        worker: u16,
        #[source]
        source: reqwest::Error,
    },
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, kind) = match &self {
            ProxyError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            ProxyError::Saturated => (StatusCode::SERVICE_UNAVAILABLE, "overloaded"),
            ProxyError::Deferred(_) => (StatusCode::SERVICE_UNAVAILABLE, "admission_deferred"),
            ProxyError::Upstream { .. } => (StatusCode::BAD_GATEWAY, "upstream_error"),
        };
        let body = Json(json!({ "error": { "message": self.to_string(), "type": kind } }));
        let mut resp = (status, body).into_response();
        if status == StatusCode::SERVICE_UNAVAILABLE {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        resp
    }
}

/// Text is packed 4 UTF-8 bytes per pseudo-token (~4 bytes per real token),
/// so a block covers ~2 KiB. Words never straddle an earlier full block, so
/// appending text keeps every previously full block hash stable.
fn text_tokens(s: &str) -> Vec<u32> {
    s.as_bytes()
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

fn token_array(v: &Value) -> Option<Vec<u32>> {
    v.as_array()?
        .iter()
        .map(|t| t.as_u64().and_then(|t| u32::try_from(t).ok()))
        .collect()
}

/// Role-tagged concatenation: each turn extends the previous serialization,
/// so multi-turn chats share block hashes with their earlier turns.
fn messages_text(msgs: &[Value]) -> Result<String, ProxyError> {
    let bad = || ProxyError::BadRequest("messages must be [{role, content}]".into());
    let mut out = String::new();
    for m in msgs {
        let role = m.get("role").and_then(Value::as_str).ok_or_else(bad)?;
        out.push_str("<|");
        out.push_str(role);
        out.push_str("|>");
        match m.get("content") {
            Some(Value::String(s)) => out.push_str(s),
            Some(Value::Array(parts)) => {
                for p in parts {
                    if let Some(t) = p.get("text").and_then(Value::as_str) {
                        out.push_str(t);
                    }
                }
            }
            Some(Value::Null) | None => {}
            Some(_) => return Err(bad()),
        }
    }
    Ok(out)
}

/// Block hashes and prompt token count for an OpenAI-style request body.
/// Precedence: `token_ids`, then `prompt` (string or token array), then `messages`.
pub fn request_blocks(body: &Value) -> Result<(Vec<BlockHash>, usize), ProxyError> {
    let tokens = if let Some(ids) = body.get("token_ids") {
        token_array(ids).ok_or_else(|| ProxyError::BadRequest("token_ids must be u32[]".into()))?
    } else if let Some(p) = body.get("prompt") {
        match p {
            Value::String(s) => text_tokens(s),
            Value::Array(_) => token_array(p)
                .ok_or_else(|| ProxyError::BadRequest("prompt must be a string or u32[]".into()))?,
            _ => {
                return Err(ProxyError::BadRequest(
                    "prompt must be a string or u32[]".into(),
                ))
            }
        }
    } else if let Some(Value::Array(msgs)) = body.get("messages") {
        text_tokens(&messages_text(msgs)?)
    } else {
        return Err(ProxyError::BadRequest(
            "expected token_ids, prompt, or messages".into(),
        ));
    };
    Ok((hash_prefix_blocks(&tokens, BLOCK_SIZE), tokens.len()))
}

/// Holds a worker's in-flight accounting and a global concurrency permit for
/// exactly as long as the response is alive. It travels inside the response
/// body stream, so completion, upstream errors, and client disconnects all
/// release it the same way: by drop.
pub struct InFlightGuard {
    state: Arc<AppState>,
    worker: WorkerId,
    tokens: usize,
    _permit: OwnedSemaphorePermit,
}

impl InFlightGuard {
    pub fn new(
        state: Arc<AppState>,
        worker: WorkerId,
        tokens: usize,
        permit: OwnedSemaphorePermit,
    ) -> Self {
        let w = &state.workers[worker.0 as usize];
        w.queue_depth.fetch_add(1, Relaxed);
        w.in_flight_tokens.fetch_add(tokens, Relaxed);
        Self {
            state,
            worker,
            tokens,
            _permit: permit,
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let w = &self.state.workers[self.worker.0 as usize];
        w.queue_depth.fetch_sub(1, Relaxed);
        w.in_flight_tokens.fetch_sub(self.tokens, Relaxed);
    }
}

pub async fn completions(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward(s, "/v1/completions", headers, body)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}

pub async fn chat_completions(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward(s, "/v1/chat/completions", headers, body)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}

async fn forward(
    state: Arc<AppState>,
    path: &'static str,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    let req: Value =
        serde_json::from_slice(&body).map_err(|e| ProxyError::BadRequest(e.to_string()))?;
    let (hashes, prompt_tokens) = request_blocks(&req)?;
    let max_tokens = req.get("max_tokens").and_then(Value::as_u64).unwrap_or(0) as usize;

    let c = &state.counters;
    let permit = state.permits.clone().try_acquire_owned().map_err(|_| {
        c.saturated.fetch_add(1, Relaxed);
        ProxyError::Saturated
    })?;

    let decision = state.router.load().route_with_admission(
        &hashes,
        &state.metrics(),
        state.now_ms(),
        state.admission.as_ref(),
    );
    let d = match decision {
        AdmissionDecision::Admit(d) => d,
        AdmissionDecision::Defer { reason } => {
            c.deferrals.fetch_add(1, Relaxed);
            return Err(ProxyError::Deferred(reason));
        }
    };
    c.requests.fetch_add(1, Relaxed);
    c.matched_blocks.fetch_add(d.matched_blocks as u64, Relaxed);
    c.total_blocks.fetch_add(d.total_blocks as u64, Relaxed);

    let guard = InFlightGuard::new(state.clone(), d.worker, prompt_tokens + max_tokens, permit);
    let url = format!("{}{}", state.workers[d.worker.0 as usize].url, path);
    let mut up = state
        .client
        .post(url)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body);
    if let Some(auth) = headers.get(header::AUTHORIZATION) {
        up = up.header(header::AUTHORIZATION, auth);
    }
    let resp = up.send().await.map_err(|source| {
        c.upstream_errors.fetch_add(1, Relaxed);
        ProxyError::Upstream {
            worker: d.worker.0,
            source,
        }
    })?;
    tracing::debug!(
        worker = d.worker.0,
        matched = d.matched_blocks,
        total = d.total_blocks,
        "routed"
    );

    let mut out = Response::builder()
        .status(resp.status())
        .header(HDR_WORKER, d.worker.0)
        .header(HDR_MATCHED, d.matched_blocks);
    for h in [header::CONTENT_TYPE, header::CACHE_CONTROL] {
        if let Some(v) = resp.headers().get(&h) {
            out = out.header(h, v);
        }
    }
    // Chunks are forwarded as they arrive; the guard lives as long as the stream.
    let stream = resp.bytes_stream().map(move |chunk| {
        let _held = &guard;
        chunk
    });
    Ok(out
        .body(Body::from_stream(stream))
        .expect("valid response parts"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks(v: Value) -> Vec<BlockHash> {
        request_blocks(&v).unwrap().0
    }

    #[test]
    fn token_ids_take_precedence() {
        let ids: Vec<u32> = (0..1100).collect();
        let (h, n) = request_blocks(&json!({ "token_ids": ids, "prompt": "x" })).unwrap();
        assert_eq!((h.len(), n), (2, 1100));
        assert_eq!(h, blocks(json!({ "prompt": ids })));
    }

    #[test]
    fn text_prompt_prefix_stable() {
        let base = "a".repeat(4 * BLOCK_SIZE * 2 + 3);
        let a = blocks(json!({ "prompt": base }));
        let b = blocks(json!({ "prompt": format!("{base}more text") }));
        assert_eq!(a.len(), 2);
        assert_eq!(a[..], b[..2]);
    }

    #[test]
    fn chat_turns_share_prefix() {
        let sys = json!({ "role": "system", "content": "s".repeat(5000) });
        let u1 = json!({ "role": "user", "content": "hello" });
        let a = blocks(json!({ "messages": [sys, u1] }));
        let parts = json!({ "role": "assistant", "content": [{ "type": "text", "text": "x".repeat(4000) }] });
        let b = blocks(json!({ "messages": [sys, u1, parts] }));
        assert!(!a.is_empty() && b.len() > a.len());
        assert_eq!(a[..], b[..a.len()]);
        // Role is part of the content: same text under another role diverges.
        let c = blocks(json!({ "messages": [{ "role": "user", "content": "s".repeat(5000) }] }));
        assert_ne!(a[0], c[0]);
    }

    #[test]
    fn rejects_malformed() {
        for v in [
            json!({}),
            json!({ "prompt": 5 }),
            json!({ "token_ids": [-1] }),
            json!({ "messages": [{ "content": "no role" }] }),
        ] {
            assert!(
                matches!(request_blocks(&v), Err(ProxyError::BadRequest(_))),
                "{v}"
            );
        }
    }
}
