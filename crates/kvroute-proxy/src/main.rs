use std::net::SocketAddr;

use clap::Parser;
use tracing_subscriber::EnvFilter;

/// KV-cache-aware routing proxy.
#[derive(Parser, Debug)]
#[command(name = "kvroute-proxy", version, about)]
struct Args {
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    tracing::info!(listen = %args.listen, block_size = kvroute_proxy::BLOCK_SIZE, "kvroute-proxy stub");
    Ok(())
}
