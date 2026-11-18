use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use kvroute_core::{AdmissionConfig, ScoringWeights, MAX_WORKERS};
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
    #[value(name = "kvroute-no-admission")]
    KvRouteNoAdmission,
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
            PolicyArg::KvRouteNoAdmission => vec![PolicyKind::KvRouteNoAdmission],
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
    /// Probability each worker cache event is dropped before reaching the router.
    #[arg(long, default_value_t = 0.0)]
    event_drop_prob: f64,
    /// Wipe the router's prefix index at this simulated time.
    #[arg(long)]
    router_restart_ms: Option<u64>,
    /// Workers republish resident blocks this long after a router restart.
    #[arg(long, default_value_t = 1_000)]
    resync_delay_ms: u64,
    /// Never resync after a restart; the index relearns from new events only.
    #[arg(long)]
    no_resync: bool,
    /// KV-budget high-water mark for kvroute admission.
    #[arg(long, default_value_t = 0.90)]
    max_kv_util: f64,
    /// Max outstanding requests per worker for kvroute admission (0 = unlimited).
    #[arg(long, default_value_t = 32)]
    max_queue_depth: usize,
    /// Disable admission control for kvroute (kvroute-no-admission is unaffected).
    #[arg(long)]
    no_admission: bool,
    /// Retry interval for requests deferred by admission control.
    #[arg(long, default_value_t = 5)]
    defer_backoff_ms: u64,
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
        seed: a.seed,
        event_drop_prob: a.event_drop_prob,
        router_restart_at_ms: a.router_restart_ms,
        resync_delay_ms: (!a.no_resync).then_some(a.resync_delay_ms),
        admission: (!a.no_admission).then_some(AdmissionConfig {
            max_kv_util: a.max_kv_util,
            max_queue_depth: (a.max_queue_depth > 0).then_some(a.max_queue_depth),
        }),
        defer_backoff_ms: a.defer_backoff_ms,
    };
    anyhow::ensure!(
        (0.0..=1.0).contains(&cfg.event_drop_prob),
        "--event-drop-prob must be in [0, 1]"
    );
    anyhow::ensure!(
        cfg.defer_backoff_ms > 0,
        "--defer-backoff-ms must be positive"
    );
    anyhow::ensure!(
        (1..=MAX_WORKERS).contains(&cfg.workers),
        "--workers must be in 1..={MAX_WORKERS}"
    );
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
