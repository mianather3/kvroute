//! Async HTTP/SSE proxy that routes OpenAI-compatible inference requests
//! using the same `kvroute-core` router as the simulator.

pub mod admin;
pub mod proxy;
pub mod state;

use std::sync::Arc;

use axum::routing::{get, post};

pub use state::{spawn_event_actor, AppState, ProxyConfig};

pub fn app(state: Arc<AppState>) -> axum::Router {
    axum::Router::new()
        .route("/v1/completions", post(proxy::completions))
        .route("/v1/chat/completions", post(proxy::chat_completions))
        .route("/internal/kv-events", post(admin::kv_events))
        .route("/internal/worker-metrics", post(admin::worker_metrics))
        .route("/internal/reset-index", post(admin::reset_index))
        .route("/healthz", get(admin::healthz))
        .route("/metrics", get(admin::metrics))
        .with_state(state)
}

/// Starts the index actor and serves until `shutdown` resolves.
pub async fn serve(
    listener: tokio::net::TcpListener,
    cfg: &ProxyConfig,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let (state, rx) = AppState::new(cfg)?;
    let actor = spawn_event_actor(&state, rx);
    axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown)
        .await?;
    actor.abort();
    Ok(())
}
