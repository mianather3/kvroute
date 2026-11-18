use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fmt::Write as _;

use kvroute_core::{CacheEvent, ScoringWeights, WorkerId, MAX_WORKERS};

use crate::policy::{Policy, PolicyKind};
use crate::worker::{Worker, WorkerConfig};
use crate::workload::SimRequest;

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub workers: u16,
    pub worker: WorkerConfig,
    /// Worker -> router propagation delay for cache events (staleness).
    pub event_delay_ms: u64,
    /// Expiry horizon for `ApproxTtl`.
    pub ttl_ms: u64,
    pub weights: ScoringWeights,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            workers: 8,
            worker: WorkerConfig::default(),
            event_delay_ms: 20,
            ttl_ms: 30_000,
            weights: ScoringWeights {
                alpha: 1.0,
                beta: 2.0,
                gamma: 2.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub policy: &'static str,
    pub requests: usize,
    pub prefix_hit_rate: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p95_ms: f64,
    pub ttft_p99_ms: f64,
    pub e2e_p50_ms: f64,
    pub e2e_p99_ms: f64,
    pub throughput_rps: f64,
    pub evictions: u64,
}

enum Kind {
    Arrival(usize),
    Complete { worker: WorkerId, req: usize },
    Cache(CacheEvent),
}

/// Ordered by `(t_us, seq)`; `seq` is the insertion counter, which makes
/// simultaneous events resolve identically on every run.
struct Event {
    t_us: u64,
    seq: u64,
    kind: Kind,
}

impl PartialEq for Event {
    fn eq(&self, o: &Self) -> bool {
        (self.t_us, self.seq) == (o.t_us, o.seq)
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Event {
    fn cmp(&self, o: &Self) -> Ordering {
        (self.t_us, self.seq).cmp(&(o.t_us, o.seq))
    }
}

struct Sim<'a> {
    cfg: &'a SimConfig,
    reqs: &'a [SimRequest],
    policy: Policy,
    workers: Vec<Worker>,
    heap: BinaryHeap<Reverse<Event>>,
    seq: u64,
    ttft_us: Vec<u64>,
    e2e_us: Vec<u64>,
    matched: u64,
    total: u64,
    last_done_us: u64,
}

impl Sim<'_> {
    fn push(&mut self, t_us: u64, kind: Kind) {
        self.seq += 1;
        self.heap.push(Reverse(Event {
            t_us,
            seq: self.seq,
            kind,
        }));
    }

    fn start(&mut self, w: WorkerId, idx: usize, now_us: u64) {
        let req = &self.reqs[idx];
        let s = self.workers[w.0 as usize].start(req, now_us / 1000);
        let ttft = now_us - req.arrival_ms * 1000 + s.prefill_us;
        self.ttft_us.push(ttft);
        self.e2e_us.push(ttft + s.decode_us);
        self.matched += s.matched_blocks as u64;
        self.total += req.block_hashes.len() as u64;
        self.push(
            now_us + s.prefill_us + s.decode_us,
            Kind::Complete {
                worker: w,
                req: idx,
            },
        );
        let delivered = now_us + self.cfg.event_delay_ms * 1000;
        for ev in s.events {
            self.push(delivered, Kind::Cache(ev));
        }
    }

    fn run(mut self) -> (Vec<u64>, Vec<u64>, u64, u64, u64, u64) {
        if !self.reqs.is_empty() {
            self.push(self.reqs[0].arrival_ms * 1000, Kind::Arrival(0));
        }
        while let Some(Reverse(Event { t_us, kind, .. })) = self.heap.pop() {
            let now_ms = t_us / 1000;
            match kind {
                Kind::Arrival(i) => {
                    if let Some(next) = self.reqs.get(i + 1) {
                        self.push(next.arrival_ms * 1000, Kind::Arrival(i + 1));
                    }
                    let req = &self.reqs[i];
                    let metrics: Vec<_> = self.workers.iter().map(Worker::metrics).collect();
                    let w = self.policy.route(req, &metrics, now_ms);
                    self.policy.on_dispatch(w, req, now_ms);
                    if self.workers[w.0 as usize].admit(i, req) {
                        self.start(w, i, t_us);
                    }
                }
                Kind::Complete { worker, req } => {
                    self.last_done_us = t_us;
                    if let Some(next) = self.workers[worker.0 as usize].finish(&self.reqs[req]) {
                        self.start(worker, next, t_us);
                    }
                }
                Kind::Cache(ev) => self.policy.on_cache_event(&ev),
            }
        }
        let evictions = self.workers.iter().map(|w| w.evictions).sum();
        (
            self.ttft_us,
            self.e2e_us,
            self.matched,
            self.total,
            evictions,
            self.last_done_us,
        )
    }
}

/// Nearest-rank percentile in milliseconds.
fn pct_ms(sorted_us: &[u64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * sorted_us.len() as f64).ceil() as usize;
    sorted_us[rank.clamp(1, sorted_us.len()) - 1] as f64 / 1000.0
}

/// Replays `reqs` (sorted by arrival) through `kind` and summarizes the outcome.
pub fn run(cfg: &SimConfig, kind: PolicyKind, reqs: &[SimRequest]) -> Summary {
    assert!(
        cfg.workers > 0 && cfg.workers <= MAX_WORKERS,
        "workers must be in 1..={MAX_WORKERS}"
    );
    debug_assert!(reqs.windows(2).all(|w| w[0].arrival_ms <= w[1].arrival_ms));
    let sim = Sim {
        cfg,
        reqs,
        policy: Policy::new(kind, cfg.workers, cfg.weights, cfg.ttl_ms),
        workers: (0..cfg.workers)
            .map(|w| Worker::new(WorkerId(w), cfg.worker))
            .collect(),
        heap: BinaryHeap::new(),
        seq: 0,
        ttft_us: Vec::with_capacity(reqs.len()),
        e2e_us: Vec::with_capacity(reqs.len()),
        matched: 0,
        total: 0,
        last_done_us: 0,
    };
    let (mut ttft, mut e2e, matched, total, evictions, last_done_us) = sim.run();
    ttft.sort_unstable();
    e2e.sort_unstable();

    let span_us = last_done_us.saturating_sub(reqs.first().map_or(0, |r| r.arrival_ms * 1000));
    Summary {
        policy: kind.name(),
        requests: reqs.len(),
        prefix_hit_rate: if total == 0 {
            0.0
        } else {
            matched as f64 / total as f64
        },
        ttft_p50_ms: pct_ms(&ttft, 50.0),
        ttft_p95_ms: pct_ms(&ttft, 95.0),
        ttft_p99_ms: pct_ms(&ttft, 99.0),
        e2e_p50_ms: pct_ms(&e2e, 50.0),
        e2e_p99_ms: pct_ms(&e2e, 99.0),
        throughput_rps: if span_us == 0 {
            0.0
        } else {
            reqs.len() as f64 * 1e6 / span_us as f64
        },
        evictions,
    }
}

const COLUMNS: [&str; 10] = [
    "policy",
    "requests",
    "prefix_hit_rate",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "e2e_p50_ms",
    "e2e_p99_ms",
    "throughput_rps",
    "evictions",
];

fn cells(s: &Summary) -> [String; 10] {
    [
        s.policy.to_string(),
        s.requests.to_string(),
        format!("{:.4}", s.prefix_hit_rate),
        format!("{:.3}", s.ttft_p50_ms),
        format!("{:.3}", s.ttft_p95_ms),
        format!("{:.3}", s.ttft_p99_ms),
        format!("{:.3}", s.e2e_p50_ms),
        format!("{:.3}", s.e2e_p99_ms),
        format!("{:.3}", s.throughput_rps),
        s.evictions.to_string(),
    ]
}

pub fn to_csv(rows: &[Summary]) -> String {
    let mut out = COLUMNS.join(",");
    out.push('\n');
    for r in rows {
        out.push_str(&cells(r).join(","));
        out.push('\n');
    }
    out
}

pub fn to_table(rows: &[Summary]) -> String {
    let body: Vec<[String; 10]> = rows.iter().map(cells).collect();
    let width: Vec<usize> = (0..COLUMNS.len())
        .map(|c| {
            body.iter()
                .map(|r| r[c].len())
                .chain([COLUMNS[c].len()])
                .max()
                .unwrap_or(0)
        })
        .collect();
    let mut out = String::new();
    let line = |out: &mut String, cols: &mut dyn Iterator<Item = &str>| {
        for (c, v) in cols.enumerate() {
            let sep = if c == 0 { "" } else { "  " };
            if c == 0 {
                let _ = write!(out, "{sep}{v:<w$}", w = width[c]);
            } else {
                let _ = write!(out, "{sep}{v:>w$}", w = width[c]);
            }
        }
        out.push('\n');
    };
    line(&mut out, &mut COLUMNS.iter().copied());
    for r in &body {
        line(&mut out, &mut r.iter().map(String::as_str));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_nearest_rank() {
        let v: Vec<u64> = (1..=100).map(|x| x * 1000).collect();
        assert_eq!(pct_ms(&v, 50.0), 50.0);
        assert_eq!(pct_ms(&v, 99.0), 99.0);
        assert_eq!(pct_ms(&[7000], 99.0), 7.0);
        assert_eq!(pct_ms(&[], 50.0), 0.0);
    }

    #[test]
    fn queueing_shows_up_in_ttft() {
        let cfg = SimConfig {
            workers: 1,
            worker: WorkerConfig {
                max_batch_size: 1,
                base_overhead_ms: 0.0,
                prefill_tok_per_ms: 1.0,
                decode_tok_per_ms: 1.0,
                ..WorkerConfig::default()
            },
            ..SimConfig::default()
        };
        let r = |id| SimRequest {
            id,
            arrival_ms: 0,
            input_tokens: 10,
            output_tokens: 10,
            block_hashes: vec![],
        };
        let s = run(&cfg, PolicyKind::RoundRobin, &[r(0), r(1)]);
        // Second request waits 20ms for the first, then prefills 10ms.
        assert_eq!((s.ttft_p50_ms, s.ttft_p99_ms), (10.0, 30.0));
        assert_eq!(s.e2e_p99_ms, 40.0);
        assert_eq!(s.throughput_rps, 2.0 / 0.040);
    }

    #[test]
    fn empty_workload() {
        let s = run(&SimConfig::default(), PolicyKind::KvRoute, &[]);
        assert_eq!(s.requests, 0);
        assert_eq!(s.prefix_hit_rate, 0.0);
    }
}
