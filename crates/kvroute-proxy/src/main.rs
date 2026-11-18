use std::net::SocketAddr;

use clap::Parser;
use kvroute_core::{AdmissionConfig, ScoringWeights};
use kvroute_proxy::ProxyConfig;
use tracing_subscriber::EnvFilter;

/// KV-cache-aware routing proxy for OpenAI-compatible inference backends.
#[derive(Parser, Debug)]
#[command(name = "kvroute-proxy", version, about)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    /// Backend base URLs, comma-separated; position is the worker id.
    #[arg(long, value_delimiter = ',', required = true)]
    workers: Vec<String>,
    /// Max requests in flight across all workers before returning 503.
    #[arg(long, default_value_t = 256)]
    max_concurrent: usize,
    /// KV-budget high-water mark for admission control.
    #[arg(long, default_value_t = 0.90)]
    max_kv_util: f64,
    /// Max outstanding requests per worker for admission (0 = unlimited).
    #[arg(long, default_value_t = 32)]
    max_queue_depth: usize,
    /// Disable KV-budget admission control.
    #[arg(long)]
    no_admission: bool,
    /// Reward per matched prefix block.
    #[arg(long, default_value_t = 1.0)]
    alpha: f64,
    /// Penalty per outstanding request.
    #[arg(long, default_value_t = 2.0)]
    beta: f64,
    /// Penalty on KV utilization.
    #[arg(long, default_value_t = 2.0)]
    gamma: f64,
    /// Approximate-TTL index mode: forget blocks not re-stored within this window.
    #[arg(long)]
    ttl_ms: Option<u64>,
    /// Capacity of the bounded KV-event queue.
    #[arg(long, default_value_t = 4096)]
    event_queue: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let a = Args::parse();
    let cfg = ProxyConfig {
        workers: a.workers,
        max_concurrent: a.max_concurrent,
        admission: (!a.no_admission).then_some(AdmissionConfig {
            max_kv_util: a.max_kv_util,
            max_queue_depth: (a.max_queue_depth > 0).then_some(a.max_queue_depth),
        }),
        weights: ScoringWeights {
            alpha: a.alpha,
            beta: a.beta,
            gamma: a.gamma,
        },
        ttl_ms: a.ttl_ms,
        event_queue: a.event_queue,
    };

    let listener = tokio::net::TcpListener::bind(a.listen).await?;
    tracing::info!(listen = %a.listen, workers = cfg.workers.len(), "kvroute-proxy listening");
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    kvroute_proxy::serve(listener, &cfg, shutdown).await
}
