use serde::{Deserialize, Serialize};

use crate::hash::BlockHash;
use crate::scorer::{rank, RouteDecision, Router, WorkerMetrics};

/// Per-worker KV budget. Keeping committed KV below the high-water mark leaves
/// evictable headroom so cached prefixes survive bursts instead of thrashing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdmissionConfig {
    /// Max `(kv_used + needed) / kv_capacity` after admitting a request.
    pub max_kv_util: f64,
    /// Max outstanding requests per worker; `None` disables the cap.
    pub max_queue_depth: Option<usize>,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            max_kv_util: 0.90,
            max_queue_depth: Some(32),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeferReason {
    KvBudgetExceeded,
    QueueFull,
    NoHealthyWorkers,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AdmissionDecision {
    Admit(RouteDecision),
    Defer { reason: DeferReason },
}

impl AdmissionConfig {
    /// Why `m` cannot take a request needing `needed` new KV blocks, if it can't.
    ///
    /// An idle worker (`kv_used_blocks == 0`) always passes the KV check so a
    /// request larger than the budget still makes progress.
    pub fn check(&self, m: &WorkerMetrics, needed: usize) -> Option<DeferReason> {
        if self.max_queue_depth.is_some_and(|q| m.queue_depth >= q) {
            return Some(DeferReason::QueueFull);
        }
        let projected = (m.kv_used_blocks + needed) as f64 / m.kv_capacity_blocks.max(1) as f64;
        if m.kv_used_blocks > 0 && projected > self.max_kv_util {
            return Some(DeferReason::KvBudgetExceeded);
        }
        None
    }
}

impl Router {
    /// Like [`Router::route`], but only among workers that pass `admission`.
    ///
    /// Cache hits reduce a worker's KV cost, so a well-matched worker may fit
    /// when a cold one does not. If every healthy worker is rejected, the
    /// reason is `KvBudgetExceeded` if any worker failed on KV, else `QueueFull`.
    pub fn route_with_admission(
        &self,
        hashes: &[BlockHash],
        workers: &[WorkerMetrics],
        now_ms: u64,
        admission: Option<&AdmissionConfig>,
    ) -> AdmissionDecision {
        let defer = |reason| AdmissionDecision::Defer { reason };
        let Some(n) = workers
            .iter()
            .filter(|m| m.healthy)
            .map(|m| m.worker.0 + 1)
            .max()
        else {
            return defer(DeferReason::NoHealthyWorkers);
        };
        let matched = self.index.match_prefix(hashes, n, now_ms);

        let mut kv_rejected = false;
        let best = workers
            .iter()
            .filter(|m| m.healthy)
            .filter_map(|m| {
                let mb = matched[m.worker.0 as usize];
                let needed = hashes.len().saturating_sub(mb);
                match admission.and_then(|a| a.check(m, needed)) {
                    Some(DeferReason::KvBudgetExceeded) => {
                        kv_rejected = true;
                        None
                    }
                    Some(_) => None,
                    None => Some((m, mb, self.score(m, mb))),
                }
            })
            .max_by(|&a, &b| rank(a, b));

        match best {
            Some((m, mb, score)) => AdmissionDecision::Admit(RouteDecision {
                worker: m.worker,
                matched_blocks: mb,
                total_blocks: hashes.len(),
                score,
            }),
            None if kv_rejected => defer(DeferReason::KvBudgetExceeded),
            None => defer(DeferReason::QueueFull),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheEvent, GlobalPrefixIndex, ScoringWeights, WorkerId};

    const ADM: AdmissionConfig = AdmissionConfig {
        max_kv_util: 0.9,
        max_queue_depth: Some(4),
    };

    fn m(w: u16, used: usize, queue: usize) -> WorkerMetrics {
        WorkerMetrics {
            worker: WorkerId(w),
            queue_depth: queue,
            in_flight_tokens: 0,
            kv_used_blocks: used,
            kv_capacity_blocks: 100,
            healthy: true,
        }
    }

    fn hs(n: u64) -> Vec<BlockHash> {
        (0..n).map(BlockHash).collect()
    }

    /// Worker 0 holds the first 10 blocks of `hs(_)`.
    fn router() -> Router {
        let mut r = Router::new(
            GlobalPrefixIndex::new(None),
            ScoringWeights {
                alpha: 1.0,
                beta: 1.0,
                gamma: 0.0,
            },
        );
        r.apply_event(&CacheEvent::BlockStored {
            worker: WorkerId(0),
            hashes: hs(10),
            now_ms: 0,
        });
        r
    }

    fn admitted(d: AdmissionDecision) -> RouteDecision {
        match d {
            AdmissionDecision::Admit(d) => d,
            d => panic!("expected admit, got {d:?}"),
        }
    }

    #[test]
    fn without_admission_matches_route() {
        let r = router();
        let ws = [m(0, 95, 0), m(1, 0, 0)];
        let d = admitted(r.route_with_admission(&hs(20), &ws, 0, None));
        assert_eq!(Some(d), r.route(&hs(20), &ws, 0));
        assert_eq!(d.worker, WorkerId(0));
    }

    #[test]
    fn over_budget_match_falls_back_to_headroom() {
        let r = router();
        // Worker 0: 85 used + 10 uncached = 95% > 90%. Worker 1: 50 + 20 = 70%.
        let ws = [m(0, 85, 0), m(1, 50, 0)];
        let d = admitted(r.route_with_admission(&hs(20), &ws, 0, Some(&ADM)));
        assert_eq!((d.worker, d.matched_blocks), (WorkerId(1), 0));

        // Same worker 0 fits when the request is fully cached: 85 + 0 = 85%.
        let d = admitted(r.route_with_admission(&hs(10), &ws, 0, Some(&ADM)));
        assert_eq!((d.worker, d.matched_blocks), (WorkerId(0), 10));
    }

    #[test]
    fn defers_when_all_over_budget() {
        let r = router();
        let ws = [m(0, 85, 0), m(1, 80, 0)];
        assert_eq!(
            r.route_with_admission(&hs(20), &ws, 0, Some(&ADM)),
            AdmissionDecision::Defer {
                reason: DeferReason::KvBudgetExceeded
            }
        );
    }

    #[test]
    fn queue_cap_and_no_healthy() {
        let r = router();
        let ws = [m(0, 0, 4), m(1, 0, 9)];
        assert_eq!(
            r.route_with_admission(&hs(1), &ws, 0, Some(&ADM)),
            AdmissionDecision::Defer {
                reason: DeferReason::QueueFull
            }
        );
        let mut sick = m(0, 0, 0);
        sick.healthy = false;
        assert_eq!(
            r.route_with_admission(&hs(1), &[sick], 0, Some(&ADM)),
            AdmissionDecision::Defer {
                reason: DeferReason::NoHealthyWorkers
            }
        );
    }

    #[test]
    fn idle_worker_admits_oversized_request() {
        let r = router();
        let d = admitted(r.route_with_admission(&hs(500), &[m(1, 0, 0)], 0, Some(&ADM)));
        assert_eq!(d.worker, WorkerId(1));
        assert_eq!(
            r.route_with_admission(&hs(500), &[m(1, 1, 0)], 0, Some(&ADM)),
            AdmissionDecision::Defer {
                reason: DeferReason::KvBudgetExceeded
            }
        );
    }
}
