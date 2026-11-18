use clap::Parser;

/// Simulate KV-cache-aware routing over a request trace.
#[derive(Parser, Debug)]
#[command(name = "kvroute-sim", version, about)]
struct Args {
    /// Path to a request trace.
    #[arg(long)]
    trace: Option<std::path::PathBuf>,

    /// RNG seed for deterministic runs.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!(
        "kvroute-sim: block_size={} seed={} trace={:?}",
        kvroute_sim::BLOCK_SIZE,
        args.seed,
        args.trace
    );
    Ok(())
}
