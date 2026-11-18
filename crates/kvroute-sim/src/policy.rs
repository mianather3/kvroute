use clap::ValueEnum;
use kvroute_core::{
    CacheEvent, GlobalPrefixIndex, Router, ScoringWeights, WorkerId, WorkerMetrics,
};

use crate::workload::SimRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum PolicyKind {
    RoundRobin,
    ConsistentHash,
    LeastLoaded,
    ApproxTtl,
    #[value(name = "kvroute")]
    KvRoute,
}

impl PolicyKind {
    pub const ALL: [PolicyKind; 5] = [
        PolicyKind::RoundRobin,
        PolicyKind::ConsistentHash,
        PolicyKind::LeastLoaded,
        PolicyKind::ApproxTtl,
        PolicyKind::KvRoute,
    ];

    pub fn name(self) -> &'static str {
        match self {
            PolicyKind::RoundRobin => "round-robin",
            PolicyKind::ConsistentHash => "consistent-hash",
            PolicyKind::LeastLoaded => "least-loaded",
            PolicyKind::ApproxTtl => "approx-ttl",
            PolicyKind::KvRoute => "kvroute",
        }
    }
}

const VNODES: u64 = 64;

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Debug, Clone)]
pub enum Policy {
    RoundRobin {
        next: u16,
        n: u16,
    },
    /// Ring keyed on the first block hash, i.e. system-prompt affinity.
    ConsistentHash {
        ring: Vec<(u64, WorkerId)>,
    },
    LeastLoaded,
    /// Router fed only by its own dispatches; entries age out after the TTL
    /// instead of tracking real evictions.
    ApproxTtl {
        router: Router,
        last_expire_ms: u64,
    },
    /// Router fed by the workers' true (possibly delayed) cache events.
    KvRoute {
        router: Router,
    },
}

impl Policy {
    pub fn new(kind: PolicyKind, workers: u16, weights: ScoringWeights, ttl_ms: u64) -> Self {
        match kind {
            PolicyKind::RoundRobin => Policy::RoundRobin {
                next: 0,
                n: workers,
            },
            PolicyKind::ConsistentHash => {
                let mut ring: Vec<_> = (0..workers)
                    .flat_map(|w| {
                        (0..VNODES).map(move |v| (mix(((w as u64) << 32) | v), WorkerId(w)))
                    })
                    .collect();
                ring.sort_unstable();
                Policy::ConsistentHash { ring }
            }
            PolicyKind::LeastLoaded => Policy::LeastLoaded,
            PolicyKind::ApproxTtl => Policy::ApproxTtl {
                router: Router::new(GlobalPrefixIndex::new(Some(ttl_ms)), weights),
                last_expire_ms: 0,
            },
            PolicyKind::KvRoute => Policy::KvRoute {
                router: Router::new(GlobalPrefixIndex::new(None), weights),
            },
        }
    }

    pub fn route(&mut self, req: &SimRequest, ws: &[WorkerMetrics], now_ms: u64) -> WorkerId {
        match self {
            Policy::RoundRobin { next, n } => {
                let w = WorkerId(*next);
                *next = (*next + 1) % *n;
                w
            }
            Policy::ConsistentHash { ring } => {
                let key = req.block_hashes.first().map_or(mix(req.id), |h| mix(h.0));
                let i = ring.partition_point(|&(p, _)| p < key);
                ring[i % ring.len()].1
            }
            Policy::LeastLoaded => {
                ws.iter()
                    .filter(|m| m.healthy)
                    .min_by_key(|m| (m.queue_depth, m.in_flight_tokens, m.worker))
                    .expect("no healthy workers")
                    .worker
            }
            Policy::ApproxTtl { router, .. } | Policy::KvRoute { router } => {
                router
                    .route(&req.block_hashes, ws, now_ms)
                    .expect("no healthy workers")
                    .worker
            }
        }
    }

    /// Called after the request is assigned to `w`.
    pub fn on_dispatch(&mut self, w: WorkerId, req: &SimRequest, now_ms: u64) {
        if let Policy::ApproxTtl {
            router,
            last_expire_ms,
        } = self
        {
            router.apply_event(&CacheEvent::BlockStored {
                worker: w,
                hashes: req.block_hashes.clone(),
                now_ms,
            });
            let ttl = router.index.ttl_ms().unwrap_or(0);
            if now_ms.saturating_sub(*last_expire_ms) >= ttl {
                router.index.expire(now_ms);
                *last_expire_ms = now_ms;
            }
        }
    }

    pub fn on_cache_event(&mut self, ev: &CacheEvent) {
        if let Policy::KvRoute { router } = self {
            router.apply_event(ev);
        }
    }
}

#[cfg(test)]
mod tests {
    use kvroute_core::BlockHash;

    use super::*;

    const WEIGHTS: ScoringWeights = ScoringWeights {
        alpha: 1.0,
        beta: 1.0,
        gamma: 0.0,
    };

    fn req(first: u64) -> SimRequest {
        SimRequest {
            id: 0,
            arrival_ms: 0,
            input_tokens: 1024,
            output_tokens: 1,
            block_hashes: vec![BlockHash(first), BlockHash(first + 1)],
        }
    }

    fn idle(n: u16) -> Vec<WorkerMetrics> {
        (0..n)
            .map(|w| WorkerMetrics {
                worker: WorkerId(w),
                queue_depth: 0,
                in_flight_tokens: 0,
                kv_used_blocks: 0,
                kv_capacity_blocks: 100,
                healthy: true,
            })
            .collect()
    }

    #[test]
    fn round_robin_cycles() {
        let mut p = Policy::new(PolicyKind::RoundRobin, 3, WEIGHTS, 0);
        let got: Vec<u16> = (0..4).map(|_| p.route(&req(1), &idle(3), 0).0).collect();
        assert_eq!(got, vec![0, 1, 2, 0]);
    }

    #[test]
    fn consistent_hash_is_sticky_and_spreads() {
        let mut p = Policy::new(PolicyKind::ConsistentHash, 8, WEIGHTS, 0);
        let ws = idle(8);
        assert_eq!(p.route(&req(42), &ws, 0), p.route(&req(42), &ws, 0));
        let mut seen: Vec<u16> = (0..200)
            .map(|k| p.route(&req(k * 1000), &ws, 0).0)
            .collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn least_loaded_picks_min_queue() {
        let mut p = Policy::new(PolicyKind::LeastLoaded, 3, WEIGHTS, 0);
        let mut ws = idle(3);
        ws[0].queue_depth = 2;
        ws[1].queue_depth = 1;
        ws[2].queue_depth = 1;
        ws[2].in_flight_tokens = 10;
        assert_eq!(p.route(&req(1), &ws, 0), WorkerId(1));
    }

    #[test]
    fn approx_ttl_learns_from_dispatch_and_forgets() {
        let mut p = Policy::new(PolicyKind::ApproxTtl, 4, WEIGHTS, 100);
        let ws = idle(4);
        p.on_dispatch(WorkerId(3), &req(7), 0);
        assert_eq!(p.route(&req(7), &ws, 50), WorkerId(3));
        assert_eq!(p.route(&req(7), &ws, 500), WorkerId(0));
    }

    #[test]
    fn kvroute_follows_cache_events() {
        let mut p = Policy::new(PolicyKind::KvRoute, 4, WEIGHTS, 0);
        let ws = idle(4);
        p.on_dispatch(WorkerId(2), &req(7), 0);
        assert_eq!(
            p.route(&req(7), &ws, 0),
            WorkerId(0),
            "dispatch alone teaches nothing"
        );
        p.on_cache_event(&CacheEvent::BlockStored {
            worker: WorkerId(2),
            hashes: req(7).block_hashes,
            now_ms: 0,
        });
        assert_eq!(p.route(&req(7), &ws, 0), WorkerId(2));
    }
}
