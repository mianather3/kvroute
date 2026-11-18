use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use kvroute_core::ScoringWeights;
use kvroute_sim::engine::{self, SimConfig};
use kvroute_sim::policy::PolicyKind;
use kvroute_sim::worker::WorkerConfig;
use kvroute_sim::workload::{self, AgentWorkload};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PolicyArg {
    All,
    RoundRobin,
    ConsistentHash,
    LeastLoaded,
    ApproxTtl,
    #[value(name = "kvroute")]
    KvRoute,
}

impl PolicyArg {
    fn kinds(self) -> Vec<PolicyKind> {
        match self {
            PolicyArg::All => PolicyKind::ALL.to_vec(),
            PolicyArg::RoundRobin => vec![PolicyKind::RoundRobin],
            PolicyArg::ConsistentHash => vec![PolicyKind::ConsistentHash],
            PolicyArg::LeastLoaded => vec![PolicyKind::LeastLoaded],
            PolicyArg::ApproxTtl => vec![PolicyKind::ApproxTtl],
            PolicyArg::KvRoute => vec![PolicyKind::KvRoute],
        }
    }
}

/// Simulate KV-cache-aware routing over a Mooncake trace or a synthetic
/// multi-turn agent workload.
#[derive(Parser, Debug)]
#[command(name = "kvroute-sim", version, about)]
struct Args {
    /// Mooncake JSONL trace. Defaults to the synthetic multi-turn agent workload.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Arrival-rate multiplier applied to trace timestamps.
    #[arg(long, default_value_t = 1.0)]
    rate_scale: f64,
    #[arg(long, value_enum, default_value_t = PolicyArg::All)]
    policy: PolicyArg,
    #[arg(long, default_value_t = 8)]
    workers: u16,
    /// KV cache capacity per worker, in 512-token blocks.
    #[arg(long, default_value_t = 512)]
    kv_blocks: usize,
    /// Concurrent batch slots per worker.
    #[arg(long, default_value_t = 8)]
    max_batch: usize,
    /// Synthetic arrival rate (requests/s).
    #[arg(long, default_value_t = 14.0)]
    rps: f64,
    /// Requests to simulate (synthetic: generated; trace: truncated to). Default 2000 / whole trace.
    #[arg(long)]
    requests: Option<usize>,
    /// Concurrent synthetic agent sessions.
    #[arg(long, default_value_t = 64)]
    sessions: usize,
    /// Distinct shared system prompts in the synthetic workload.
    #[arg(long, default_value_t = 8)]
    system_prompts: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Worker -> router cache-event delay.
    #[arg(long, default_value_t = 20)]
    event_delay_ms: u64,
    /// Entry lifetime for the approx-ttl policy.
    #[arg(long, default_value_t = 30_000)]
    ttl_ms: u64,
    /// Router reward per matched prefix block.
    #[arg(long, default_value_t = 1.0)]
    alpha: f64,
    /// Router penalty per outstanding request.
    #[arg(long, default_value_t = 2.0)]
    beta: f64,
    /// Router penalty on committed KV utilization.
    #[arg(long, default_value_t = 2.0)]
    gamma: f64,
    /// Write per-policy summary rows to this CSV file.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Write the synthetic workload as Mooncake JSONL (a reusable fixture).
    #[arg(long)]
    dump_trace: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let a = Args::parse();

    let records = match &a.trace {
        Some(p) => {
            let mut recs = workload::load_mooncake(p)?;
            if let Some(n) = a.requests {
                recs.truncate(n);
            }
            recs
        }
        None => AgentWorkload {
            sessions: a.sessions,
            system_prompts: a.system_prompts,
            rps: a.rps,
            requests: a.requests.unwrap_or(2000),
            seed: a.seed,
        }
        .generate()?,
    };
    if let Some(p) = &a.dump_trace {
        let f = File::create(p).with_context(|| format!("create {}", p.display()))?;
        workload::write_mooncake(BufWriter::new(f), &records)?;
    }
    let reqs = workload::to_requests(&records, a.rate_scale);

    let cfg = SimConfig {
        workers: a.workers,
        worker: WorkerConfig {
            kv_capacity_blocks: a.kv_blocks,
            max_batch_size: a.max_batch,
            ..WorkerConfig::default()
        },
        event_delay_ms: a.event_delay_ms,
        ttl_ms: a.ttl_ms,
        weights: ScoringWeights {
            alpha: a.alpha,
            beta: a.beta,
            gamma: a.gamma,
        },
    };
    let rows: Vec<_> = a
        .policy
        .kinds()
        .into_iter()
        .map(|k| engine::run(&cfg, k, &reqs))
        .collect();

    print!("{}", engine::to_table(&rows));
    if let Some(p) = &a.csv {
        std::fs::write(p, engine::to_csv(&rows))
            .with_context(|| format!("write {}", p.display()))?;
    }
    Ok(())
}
