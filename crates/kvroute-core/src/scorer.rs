use std::cmp::{Ordering, Reverse};

use serde::{Deserialize, Serialize};

use crate::admission::AdmissionDecision;
use crate::hash::{BlockHash, WorkerId};
use crate::index::{CacheEvent, GlobalPrefixIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerMetrics {
    pub worker: WorkerId,
    pub queue_depth: usize,
    pub in_flight_tokens: usize,
    pub kv_used_blocks: usize,
    pub kv_capacity_blocks: usize,
    pub healthy: bool,
}

impl WorkerMetrics {
    /// KV utilization in `[0, 1]`; a worker with zero capacity counts as full.
    pub fn kv_util(&self) -> f64 {
        if self.kv_capacity_blocks == 0 {
            return 1.0;
        }
        (self.kv_used_blocks as f64 / self.kv_capacity_blocks as f64).min(1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Reward per contiguous matched prefix block.
    pub alpha: f64,
    /// Penalty per queued request.
    pub beta: f64,
    /// Penalty scaled by KV utilization.
    pub gamma: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub worker: WorkerId,
    pub matched_blocks: usize,
    pub total_blocks: usize,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct Router {
    pub index: GlobalPrefixIndex,
    pub weights: ScoringWeights,
}

impl Router {
    pub fn new(index: GlobalPrefixIndex, weights: ScoringWeights) -> Self {
        Self { index, weights }
    }

    pub fn apply_event(&mut self, event: &CacheEvent) {
        self.index.apply_event(event);
    }

    /// Picks the best healthy worker, or `None` if none are healthy.
    ///
    /// Ties break on score, then matched blocks, then shorter queue, then
    /// lower id, so identical inputs always yield identical decisions.
    pub fn route(
        &self,
        hashes: &[BlockHash],
        workers: &[WorkerMetrics],
        now_ms: u64,
    ) -> Option<RouteDecision> {
        match self.route_with_admission(hashes, workers, now_ms, None) {
            AdmissionDecision::Admit(d) => Some(d),
            AdmissionDecision::Defer { .. } => None,
        }
    }

    pub(crate) fn score(&self, m: &WorkerMetrics, matched: usize) -> f64 {
        let ScoringWeights { alpha, beta, gamma } = self.weights;
        alpha * matched as f64 - beta * m.queue_depth as f64 - gamma * m.kv_util()
    }
}

/// Total order over candidates: score, then matched blocks, then shorter
/// queue, then lower id.
pub(crate) fn rank(a: (&WorkerMetrics, usize, f64), b: (&WorkerMetrics, usize, f64)) -> Ordering {
    a.2.total_cmp(&b.2)
        .then(a.1.cmp(&b.1))
        .then(Reverse(a.0.queue_depth).cmp(&Reverse(b.0.queue_depth)))
        .then(Reverse(a.0.worker).cmp(&Reverse(b.0.worker)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: ScoringWeights = ScoringWeights {
        alpha: 1.0,
        beta: 0.5,
        gamma: 2.0,
    };

    fn m(w: u16, queue: usize, used: usize) -> WorkerMetrics {
        WorkerMetrics {
            worker: WorkerId(w),
            queue_depth: queue,
            in_flight_tokens: 0,
            kv_used_blocks: used,
            kv_capacity_blocks: 100,
            healthy: true,
        }
    }

    fn router(stores: &[(u16, &[u64])]) -> Router {
        let mut r = Router::new(GlobalPrefixIndex::new(None), W);
        for &(w, ids) in stores {
            r.apply_event(&CacheEvent::BlockStored {
                worker: WorkerId(w),
                hashes: ids.iter().copied().map(BlockHash).collect(),
                now_ms: 0,
            });
        }
        r
    }

    fn q(ids: &[u64]) -> Vec<BlockHash> {
        ids.iter().copied().map(BlockHash).collect()
    }

    #[test]
    fn prefers_prefix_match() {
        let r = router(&[(1, &[1, 2, 3])]);
        let d = r
            .route(&q(&[1, 2, 3, 4]), &[m(0, 0, 0), m(1, 1, 0)], 0)
            .unwrap();
        assert_eq!(d.worker, WorkerId(1));
        assert_eq!((d.matched_blocks, d.total_blocks), (3, 4));
        assert_eq!(d.score, 3.0 - 0.5);
    }

    #[test]
    fn load_outweighs_match() {
        let r = router(&[(1, &[1])]);
        let d = r.route(&q(&[1]), &[m(0, 0, 0), m(1, 10, 0)], 0).unwrap();
        assert_eq!(d.worker, WorkerId(0));
    }

    #[test]
    fn skips_unhealthy_and_handles_empty() {
        let r = router(&[(0, &[1, 2])]);
        let mut sick = m(0, 0, 0);
        sick.healthy = false;
        assert_eq!(
            r.route(&q(&[1, 2]), &[sick, m(1, 0, 0)], 0).unwrap().worker,
            WorkerId(1)
        );
        assert!(r.route(&q(&[1]), &[sick], 0).is_none());
        assert!(r.route(&q(&[1]), &[], 0).is_none());
    }

    #[test]
    fn tie_breaking() {
        // alpha == beta: worker 0 (1 match, 1 queued) and worker 1 (0, 0) tie on score.
        let mut r = router(&[(0, &[1])]);
        r.weights = ScoringWeights {
            alpha: 1.0,
            beta: 1.0,
            gamma: 0.0,
        };
        let d = r.route(&q(&[1]), &[m(1, 0, 0), m(0, 1, 0)], 0).unwrap();
        assert_eq!(
            d.worker,
            WorkerId(0),
            "higher matched_blocks wins score tie"
        );

        // Equal score and match: shorter queue wins (gamma offsets the queue gap).
        let mut r = router(&[]);
        r.weights = ScoringWeights {
            alpha: 0.0,
            beta: 1.0,
            gamma: 2.0,
        };
        // w0: -1 - 2*0.0 = -1; w1: 0 - 2*0.5 = -1 (exact in f64).
        let ws = [m(0, 1, 0), m(1, 0, 50)];
        let d = r.route(&q(&[]), &ws, 0).unwrap();
        assert_eq!(d.score, -1.0);
        assert_eq!(d.worker, WorkerId(1), "lower queue_depth wins");

        // Fully identical: lowest id wins regardless of input order.
        let r = router(&[]);
        let ws = [m(3, 0, 0), m(2, 0, 0), m(5, 0, 0)];
        assert_eq!(r.route(&[], &ws, 0).unwrap().worker, WorkerId(2));
        let rev: Vec<_> = ws.iter().rev().copied().collect();
        assert_eq!(r.route(&[], &rev, 0).unwrap().worker, WorkerId(2));
    }

    #[test]
    fn zero_capacity_counts_as_full() {
        let mut w = m(0, 0, 0);
        w.kv_capacity_blocks = 0;
        assert_eq!(w.kv_util(), 1.0);
        assert_eq!(m(0, 0, 250).kv_util(), 1.0);
    }
}
