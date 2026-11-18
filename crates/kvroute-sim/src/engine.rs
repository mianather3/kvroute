use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::fmt::Write as _;

use kvroute_core::{AdmissionConfig, CacheEvent, ScoringWeights, WorkerId, MAX_WORKERS};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

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
    /// Seeds the fault-injection RNG.
    pub seed: u64,
    /// Probability each worker cache event is lost in flight, leaving ghost
    /// (dropped removals) or invisible (dropped stores) blocks in the index.
    pub event_drop_prob: f64,
    /// Wipe the router's prefix index at this time.
    pub router_restart_at_ms: Option<u64>,
    /// After a restart, workers republish their resident blocks after this
    /// delay. With `None` the index relearns from new stores only: resident
    /// blocks stay invisible until evicted and recomputed, so recovery is slow.
    pub resync_delay_ms: Option<u64>,
    /// KV-budget admission for `PolicyKind::KvRoute`.
    pub admission: Option<AdmissionConfig>,
    /// Retry interval for the deferred-request queue.
    pub defer_backoff_ms: u64,
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
            seed: 0,
            event_drop_prob: 0.0,
            router_restart_at_ms: None,
            resync_delay_ms: Some(1_000),
            admission: Some(AdmissionConfig::default()),
            defer_backoff_ms: 5,
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
    /// Times a request was held back by admission control (initial deferral
    /// plus each failed retry).
    pub deferrals: u64,
}

/// Per-request outcome. Latencies are measured from the original arrival, so
/// time spent deferred by admission control counts against TTFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestRecord {
    pub id: u64,
    pub arrival_ms: u64,
    pub worker: WorkerId,
    pub start_us: u64,
    pub matched_blocks: usize,
    pub total_blocks: usize,
    pub ttft_us: u64,
    pub e2e_us: u64,
}

enum Kind {
    Arrival(usize),
    Retry,
    Complete { worker: WorkerId, req: usize },
    Cache(CacheEvent),
    RouterRestart,
    Resync,
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
    rng: ChaCha8Rng,
    records: Vec<RequestRecord>,
    /// Deferred requests in arrival order. While non-empty, new arrivals queue
    /// behind it so admission control cannot starve older requests.
    pending: VecDeque<usize>,
    retry_scheduled: bool,
    deferrals: u64,
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

    /// Sends a worker cache event toward the router, subject to drop and delay.
    fn emit(&mut self, ev: CacheEvent, now_us: u64) {
        let p = self.cfg.event_drop_prob;
        if p > 0.0 && self.rng.gen_bool(p) {
            return;
        }
        self.push(now_us + self.cfg.event_delay_ms * 1000, Kind::Cache(ev));
    }

    /// Routes and enqueues `idx` on a worker; false if admission deferred it.
    fn try_dispatch(&mut self, idx: usize, now_us: u64) -> bool {
        let req = &self.reqs[idx];
        let metrics: Vec<_> = self.workers.iter().map(Worker::metrics).collect();
        match self.policy.route(req, &metrics, now_us / 1000) {
            Ok(w) => {
                self.policy.on_dispatch(w, req, now_us / 1000);
                self.workers[w.0 as usize].admit(idx, req);
                self.drain(w, now_us);
                true
            }
            Err(_) => {
                self.deferrals += 1;
                false
            }
        }
    }

    fn defer(&mut self, idx: usize, now_us: u64) {
        self.pending.push_back(idx);
        if !self.retry_scheduled {
            self.retry_scheduled = true;
            self.push(now_us + self.cfg.defer_backoff_ms * 1000, Kind::Retry);
        }
    }

    /// Retries deferred requests oldest-first, admitting each one that fits
    /// now. A request that fits nowhere does not block younger ones, but every
    /// older request gets first claim on freed capacity at each tick.
    fn retry(&mut self, now_us: u64) {
        self.retry_scheduled = false;
        let queued = std::mem::take(&mut self.pending);
        for idx in queued {
            if !self.try_dispatch(idx, now_us) {
                self.pending.push_back(idx);
            }
        }
        if !self.pending.is_empty() {
            self.retry_scheduled = true;
            self.push(now_us + self.cfg.defer_backoff_ms * 1000, Kind::Retry);
        }
    }

    /// Starts every queued request on `w` that now fits.
    fn drain(&mut self, w: WorkerId, now_us: u64) {
        while let Some(idx) = self.workers[w.0 as usize].pop_startable(self.reqs) {
            let req = &self.reqs[idx];
            let s = self.workers[w.0 as usize].start(req, now_us / 1000);
            let ttft_us = now_us - req.arrival_ms * 1000 + s.prefill_us;
            self.records.push(RequestRecord {
                id: req.id,
                arrival_ms: req.arrival_ms,
                worker: w,
                start_us: now_us,
                matched_blocks: s.matched_blocks,
                total_blocks: req.block_hashes.len(),
                ttft_us,
                e2e_us: ttft_us + s.decode_us,
            });
            self.push(
                now_us + s.prefill_us + s.decode_us,
                Kind::Complete {
                    worker: w,
                    req: idx,
                },
            );
            for ev in s.events {
                self.emit(ev, now_us);
            }
        }
    }

    fn run(mut self) -> Self {
        if let Some(first) = self.reqs.first() {
            self.push(first.arrival_ms * 1000, Kind::Arrival(0));
        }
        if let Some(t) = self.cfg.router_restart_at_ms {
            self.push(t * 1000, Kind::RouterRestart);
        }
        while let Some(Reverse(Event { t_us, kind, .. })) = self.heap.pop() {
            match kind {
                Kind::Arrival(i) => {
                    if let Some(next) = self.reqs.get(i + 1) {
                        self.push(next.arrival_ms * 1000, Kind::Arrival(i + 1));
                    }
                    if !self.pending.is_empty() {
                        self.deferrals += 1;
                        self.defer(i, t_us);
                    } else if !self.try_dispatch(i, t_us) {
                        self.defer(i, t_us);
                    }
                }
                Kind::Retry => self.retry(t_us),
                Kind::Complete { worker, req } => {
                    self.last_done_us = t_us;
                    if let Some(ev) = self.workers[worker.0 as usize].finish(&self.reqs[req]) {
                        self.emit(ev, t_us);
                    }
                    self.drain(worker, t_us);
                }
                Kind::Cache(ev) => self.policy.on_cache_event(&ev),
                Kind::RouterRestart => {
                    self.policy.reset_index();
                    if let Some(d) = self.cfg.resync_delay_ms {
                        self.push(t_us + d * 1000, Kind::Resync);
                    }
                }
                Kind::Resync => {
                    for w in &self.workers {
                        let hashes = w.cache().snapshot();
                        if !hashes.is_empty() {
                            self.policy.on_cache_event(&CacheEvent::BlockStored {
                                worker: w.id,
                                hashes,
                                now_ms: t_us / 1000,
                            });
                        }
                    }
                }
            }
        }
        self
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
    run_detailed(cfg, kind, reqs).0
}

/// Like [`run`], also returning per-request records sorted by request id.
pub fn run_detailed(
    cfg: &SimConfig,
    kind: PolicyKind,
    reqs: &[SimRequest],
) -> (Summary, Vec<RequestRecord>) {
    assert!(
        cfg.workers > 0 && cfg.workers <= MAX_WORKERS,
        "workers must be in 1..={MAX_WORKERS}"
    );
    assert!(
        (0.0..=1.0).contains(&cfg.event_drop_prob),
        "event_drop_prob must be in [0, 1]"
    );
    assert!(
        cfg.defer_backoff_ms > 0,
        "defer_backoff_ms must be positive"
    );
    debug_assert!(reqs.windows(2).all(|w| w[0].arrival_ms <= w[1].arrival_ms));

    let sim = Sim {
        cfg,
        reqs,
        policy: Policy::new(kind, cfg.workers, cfg.weights, cfg.ttl_ms, cfg.admission),
        workers: (0..cfg.workers)
            .map(|w| Worker::new(WorkerId(w), cfg.worker))
            .collect(),
        heap: BinaryHeap::new(),
        seq: 0,
        rng: ChaCha8Rng::seed_from_u64(cfg.seed),
        records: Vec::with_capacity(reqs.len()),
        pending: VecDeque::new(),
        retry_scheduled: false,
        deferrals: 0,
        last_done_us: 0,
    }
    .run();

    let mut records = sim.records;
    records.sort_unstable_by_key(|r| r.id);
    let mut ttft: Vec<u64> = records.iter().map(|r| r.ttft_us).collect();
    let mut e2e: Vec<u64> = records.iter().map(|r| r.e2e_us).collect();
    ttft.sort_unstable();
    e2e.sort_unstable();
    let matched: usize = records.iter().map(|r| r.matched_blocks).sum();
    let total: usize = records.iter().map(|r| r.total_blocks).sum();

    let span_us = sim
        .last_done_us
        .saturating_sub(reqs.first().map_or(0, |r| r.arrival_ms * 1000));
    let summary = Summary {
        policy: kind.name(),
        requests: records.len(),
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
            records.len() as f64 * 1e6 / span_us as f64
        },
        evictions: sim.workers.iter().map(|w| w.evictions).sum(),
        deferrals: sim.deferrals,
    };
    (summary, records)
}

const COLUMNS: [&str; 11] = [
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
    "deferrals",
];

fn cells(s: &Summary) -> [String; 11] {
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
        s.deferrals.to_string(),
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
    let body: Vec<[String; 11]> = rows.iter().map(cells).collect();
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
    use kvroute_core::{BlockHash, BLOCK_SIZE};

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
    fn deferral_counts_against_ttft() {
        let cfg = SimConfig {
            workers: 1,
            worker: WorkerConfig {
                kv_capacity_blocks: 10,
                base_overhead_ms: 0.0,
                prefill_tok_per_ms: 512.0,
                ..WorkerConfig::default()
            },
            ..SimConfig::default()
        };
        let r = |id: u64| SimRequest {
            id,
            arrival_ms: 0,
            input_tokens: 8 * BLOCK_SIZE,
            output_tokens: 0,
            block_hashes: (0..8).map(|b| BlockHash(id * 100 + b)).collect(),
        };
        let reqs = [r(0), r(1)];
        // B needs 8 + 8 > 0.9 * 10 blocks: deferred at 0 and 5ms, admitted at
        // 10ms once A (8ms prefill, no decode) has finished.
        let (s, recs) = run_detailed(&cfg, PolicyKind::KvRoute, &reqs);
        assert_eq!(s.deferrals, 2);
        assert_eq!(recs[1].start_us, 10_000);
        assert_eq!(recs[1].ttft_us, 18_000);

        let s = run(&cfg, PolicyKind::KvRouteNoAdmission, &reqs);
        assert_eq!(s.deferrals, 0);
    }

    #[test]
    fn empty_workload() {
        let s = run(&SimConfig::default(), PolicyKind::KvRoute, &[]);
        assert_eq!(s.requests, 0);
        assert_eq!(s.prefix_hit_rate, 0.0);
    }
}
