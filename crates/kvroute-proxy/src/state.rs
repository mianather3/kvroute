//! Shared proxy state. Request tasks only read: an `ArcSwap` snapshot of the
//! router and per-worker atomics. The prefix index has exactly one writer,
//! [`run_event_actor`], fed through a bounded channel.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::ensure;
use arc_swap::ArcSwap;
use kvroute_core::{
    AdmissionConfig, CacheEvent, GlobalPrefixIndex, Router, ScoringWeights, WorkerId,
    WorkerMetrics, MAX_WORKERS,
};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;

/// Max channel messages applied per snapshot publish.
const MAX_BATCH: usize = 1024;

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Backend base URLs; index `i` is `WorkerId(i)`.
    pub workers: Vec<String>,
    pub max_concurrent: usize,
    pub admission: Option<AdmissionConfig>,
    pub weights: ScoringWeights,
    /// Approximate-TTL index mode; `None` trusts eviction events.
    pub ttl_ms: Option<u64>,
    /// Bounded index-channel capacity.
    pub event_queue: usize,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            workers: Vec::new(),
            max_concurrent: 256,
            admission: Some(AdmissionConfig::default()),
            weights: ScoringWeights {
                alpha: 1.0,
                beta: 2.0,
                gamma: 2.0,
            },
            ttl_ms: None,
            event_queue: 4096,
        }
    }
}

/// Messages to the single index writer.
#[derive(Debug)]
pub enum IndexMsg {
    Events(Vec<CacheEvent>),
    ResetIndex,
}

/// Live per-worker load. Written by request guards and the metrics endpoint,
/// read lock-free on every routing decision.
#[derive(Debug)]
pub struct WorkerState {
    pub id: WorkerId,
    pub url: String,
    pub queue_depth: AtomicUsize,
    pub in_flight_tokens: AtomicUsize,
    pub kv_used_blocks: AtomicUsize,
    /// 0 until the worker reports its capacity.
    pub kv_capacity_blocks: AtomicUsize,
    pub healthy: AtomicBool,
}

impl WorkerState {
    fn new(id: u16, url: String) -> Self {
        Self {
            id: WorkerId(id),
            url: url.trim_end_matches('/').to_string(),
            queue_depth: AtomicUsize::new(0),
            in_flight_tokens: AtomicUsize::new(0),
            kv_used_blocks: AtomicUsize::new(0),
            kv_capacity_blocks: AtomicUsize::new(0),
            healthy: AtomicBool::new(true),
        }
    }

    /// Point-in-time view. Fields are loaded independently; routing tolerates
    /// that skew the same way it tolerates stale cache events.
    pub fn metrics(&self) -> WorkerMetrics {
        WorkerMetrics {
            worker: self.id,
            queue_depth: self.queue_depth.load(Relaxed),
            in_flight_tokens: self.in_flight_tokens.load(Relaxed),
            kv_used_blocks: self.kv_used_blocks.load(Relaxed),
            kv_capacity_blocks: self.kv_capacity_blocks.load(Relaxed),
            healthy: self.healthy.load(Relaxed),
        }
    }
}

#[derive(Debug, Default)]
pub struct Counters {
    pub requests: AtomicU64,
    pub matched_blocks: AtomicU64,
    pub total_blocks: AtomicU64,
    pub deferrals: AtomicU64,
    pub saturated: AtomicU64,
    pub upstream_errors: AtomicU64,
    pub events_dropped: AtomicU64,
}

#[derive(Debug)]
pub struct AppState {
    pub router: Arc<ArcSwap<Router>>,
    pub workers: Vec<WorkerState>,
    pub tx: mpsc::Sender<IndexMsg>,
    pub admission: Option<AdmissionConfig>,
    pub permits: Arc<Semaphore>,
    pub counters: Counters,
    pub client: reqwest::Client,
    epoch: Instant,
}

impl AppState {
    /// Builds state plus the receiving end for [`spawn_event_actor`].
    pub fn new(cfg: &ProxyConfig) -> anyhow::Result<(Arc<Self>, mpsc::Receiver<IndexMsg>)> {
        ensure!(!cfg.workers.is_empty(), "at least one worker is required");
        ensure!(
            cfg.workers.len() <= MAX_WORKERS as usize,
            "at most {MAX_WORKERS} workers"
        );
        ensure!(cfg.max_concurrent > 0, "max_concurrent must be positive");
        ensure!(cfg.event_queue > 0, "event_queue must be positive");

        let (tx, rx) = mpsc::channel(cfg.event_queue);
        let router = Router::new(GlobalPrefixIndex::new(cfg.ttl_ms), cfg.weights);
        let state = Self {
            router: Arc::new(ArcSwap::from_pointee(router)),
            workers: cfg
                .workers
                .iter()
                .enumerate()
                .map(|(i, url)| WorkerState::new(i as u16, url.clone()))
                .collect(),
            tx,
            admission: cfg.admission,
            permits: Arc::new(Semaphore::new(cfg.max_concurrent)),
            counters: Counters::default(),
            client: reqwest::Client::new(),
            epoch: Instant::now(),
        };
        Ok((Arc::new(state), rx))
    }

    /// Proxy-local monotonic clock, the time base for TTL and routing.
    pub fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    pub fn metrics(&self) -> Vec<WorkerMetrics> {
        self.workers.iter().map(WorkerState::metrics).collect()
    }

    pub fn worker(&self, id: WorkerId) -> Option<&WorkerState> {
        self.workers.get(id.0 as usize)
    }
}

pub fn spawn_event_actor(state: &AppState, rx: mpsc::Receiver<IndexMsg>) -> JoinHandle<()> {
    tokio::spawn(run_event_actor(state.router.clone(), rx, state.epoch))
}

/// The only writer of the prefix index. Drains up to [`MAX_BATCH`] messages
/// per wake-up, applies them to its private `Router`, then publishes one
/// snapshot, so bursts cost one clone rather than one per event.
///
/// `BlockStored.now_ms` is restamped with the proxy clock: publisher clocks
/// are not comparable to ours and TTL expiry must use a single time base.
pub async fn run_event_actor(
    snapshot: Arc<ArcSwap<Router>>,
    mut rx: mpsc::Receiver<IndexMsg>,
    epoch: Instant,
) {
    let mut router = Router::clone(&snapshot.load());
    let ttl = router.index.ttl_ms();
    let mut expiry = ttl.map(|t| {
        let mut i = tokio::time::interval(Duration::from_millis((t / 2).max(10)));
        i.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        i
    });

    loop {
        let now = || epoch.elapsed().as_millis() as u64;
        tokio::select! {
            msg = rx.recv() => {
                let Some(msg) = msg else { return };
                apply(&mut router, msg, now());
                for _ in 1..MAX_BATCH {
                    match rx.try_recv() {
                        Ok(msg) => apply(&mut router, msg, now()),
                        Err(_) => break,
                    }
                }
            }
            _ = tick(&mut expiry) => {
                router.index.expire(now());
            }
        }
        snapshot.store(Arc::new(router.clone()));
    }
}

async fn tick(i: &mut Option<tokio::time::Interval>) {
    match i {
        Some(i) => {
            i.tick().await;
        }
        None => std::future::pending().await,
    }
}

fn apply(router: &mut Router, msg: IndexMsg, now_ms: u64) {
    match msg {
        IndexMsg::Events(events) => {
            for mut ev in events {
                if let CacheEvent::BlockStored { now_ms: t, .. } = &mut ev {
                    *t = now_ms;
                }
                router.apply_event(&ev);
            }
        }
        IndexMsg::ResetIndex => {
            router.index = GlobalPrefixIndex::new(router.index.ttl_ms());
        }
    }
}

#[cfg(test)]
mod tests {
    use kvroute_core::BlockHash;

    use super::*;

    fn cfg(n: usize) -> ProxyConfig {
        ProxyConfig {
            workers: (0..n).map(|i| format!("http://w{i}/")).collect(),
            ..ProxyConfig::default()
        }
    }

    fn stored(w: u16, ids: &[u64]) -> CacheEvent {
        CacheEvent::BlockStored {
            worker: WorkerId(w),
            hashes: ids.iter().copied().map(BlockHash).collect(),
            now_ms: u64::MAX,
        }
    }

    async fn settle(state: &AppState, want: usize) {
        for _ in 0..200 {
            if state.router.load().index.len() == want {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("index never reached {want} blocks");
    }

    #[test]
    fn rejects_bad_config() {
        assert!(AppState::new(&cfg(0)).is_err());
        assert!(AppState::new(&cfg(129)).is_err());
        let (s, _rx) = AppState::new(&cfg(2)).unwrap();
        assert_eq!(s.workers[0].url, "http://w0");
    }

    #[tokio::test]
    async fn actor_publishes_snapshots_and_resets() {
        let (state, rx) = AppState::new(&cfg(2)).unwrap();
        let before = state.router.load_full();
        spawn_event_actor(&state, rx);

        state
            .tx
            .send(IndexMsg::Events(vec![stored(1, &[1, 2, 3])]))
            .await
            .unwrap();
        settle(&state, 3).await;
        assert_eq!(before.index.len(), 0, "old snapshots are immutable");
        let m = state
            .router
            .load()
            .index
            .match_prefix(&[BlockHash(1), BlockHash(2)], 2, 0);
        assert_eq!(m, vec![0, 2]);

        state.tx.send(IndexMsg::ResetIndex).await.unwrap();
        settle(&state, 0).await;
    }

    #[tokio::test]
    async fn actor_restamps_and_expires_ttl() {
        let (state, rx) = AppState::new(&ProxyConfig {
            ttl_ms: Some(50),
            ..cfg(1)
        })
        .unwrap();
        spawn_event_actor(&state, rx);
        // now_ms = u64::MAX from the publisher would never expire if trusted.
        state
            .tx
            .send(IndexMsg::Events(vec![stored(0, &[7])]))
            .await
            .unwrap();
        settle(&state, 1).await;
        settle(&state, 0).await;
    }
}
