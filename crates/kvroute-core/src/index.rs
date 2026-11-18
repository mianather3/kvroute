use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::hash::{BlockHash, WorkerId};

pub const MAX_WORKERS: u16 = 128;

/// Set of up to 128 workers packed into one `u128`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct WorkerBitset(u128);

impl WorkerBitset {
    pub const EMPTY: Self = Self(0);

    /// Workers `0..n`.
    pub fn first_n(n: u16) -> Self {
        assert!(n <= MAX_WORKERS, "at most {MAX_WORKERS} workers");
        match n {
            MAX_WORKERS => Self(u128::MAX),
            n => Self((1u128 << n) - 1),
        }
    }

    #[inline]
    fn bit(w: WorkerId) -> u128 {
        assert!(w.0 < MAX_WORKERS, "worker id {} out of range", w.0);
        1u128 << w.0
    }

    #[inline]
    pub fn insert(&mut self, w: WorkerId) {
        self.0 |= Self::bit(w);
    }

    #[inline]
    pub fn remove(&mut self, w: WorkerId) {
        self.0 &= !Self::bit(w);
    }

    #[inline]
    pub fn contains(self, w: WorkerId) -> bool {
        w.0 < MAX_WORKERS && self.0 & (1u128 << w.0) != 0
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Workers in ascending id order.
    pub fn iter_workers(self) -> impl Iterator<Item = WorkerId> {
        let mut bits = self.0;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let w = bits.trailing_zeros() as u16;
            bits &= bits - 1;
            Some(WorkerId(w))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheEvent {
    BlockStored {
        worker: WorkerId,
        hashes: Vec<BlockHash>,
        now_ms: u64,
    },
    BlockRemoved {
        worker: WorkerId,
        hashes: Vec<BlockHash>,
    },
    AllBlocksCleared {
        worker: WorkerId,
    },
}

/// Global view of which workers hold which prefix blocks.
///
/// With `ttl_ms` set, the index runs in approximate mode: a block is assumed
/// evicted once it has not been (re)stored for longer than the TTL, so routing
/// degrades gracefully when workers never publish `BlockRemoved`.
#[derive(Debug, Clone, Default)]
pub struct GlobalPrefixIndex {
    blocks: HashMap<BlockHash, WorkerBitset>,
    last_seen: HashMap<(WorkerId, BlockHash), u64>,
    ttl_ms: Option<u64>,
}

impl GlobalPrefixIndex {
    pub fn new(ttl_ms: Option<u64>) -> Self {
        Self {
            ttl_ms,
            ..Self::default()
        }
    }

    pub fn ttl_ms(&self) -> Option<u64> {
        self.ttl_ms
    }

    /// Number of distinct block hashes held by at least one worker.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn apply_event(&mut self, event: &CacheEvent) {
        match event {
            CacheEvent::BlockStored {
                worker,
                hashes,
                now_ms,
            } => {
                for &h in hashes {
                    self.blocks.entry(h).or_default().insert(*worker);
                    if self.ttl_ms.is_some() {
                        // max: a late-delivered older event must not age the entry.
                        let t = self.last_seen.entry((*worker, h)).or_insert(*now_ms);
                        *t = (*t).max(*now_ms);
                    }
                }
            }
            CacheEvent::BlockRemoved { worker, hashes } => {
                for &h in hashes {
                    self.remove(*worker, h);
                }
            }
            CacheEvent::AllBlocksCleared { worker } => {
                self.blocks.retain(|_, set| {
                    set.remove(*worker);
                    !set.is_empty()
                });
                self.last_seen.retain(|&(w, _), _| w != *worker);
            }
        }
    }

    fn remove(&mut self, worker: WorkerId, h: BlockHash) {
        if let Some(set) = self.blocks.get_mut(&h) {
            set.remove(worker);
            if set.is_empty() {
                self.blocks.remove(&h);
            }
        }
        self.last_seen.remove(&(worker, h));
    }

    #[inline]
    fn is_live(&self, w: WorkerId, h: BlockHash, ttl: u64, now_ms: u64) -> bool {
        self.last_seen
            .get(&(w, h))
            .is_some_and(|&t| now_ms.saturating_sub(t) <= ttl)
    }

    /// Drops every entry older than the TTL. No-op without a TTL.
    pub fn expire(&mut self, now_ms: u64) {
        let Some(ttl) = self.ttl_ms else { return };
        let blocks = &mut self.blocks;
        self.last_seen.retain(|&(w, h), &mut t| {
            if now_ms.saturating_sub(t) <= ttl {
                return true;
            }
            if let Some(set) = blocks.get_mut(&h) {
                set.remove(w);
                if set.is_empty() {
                    blocks.remove(&h);
                }
            }
            false
        });
    }

    /// Contiguous matched blocks from block 0 for each worker in `0..num_workers`.
    ///
    /// A worker's streak ends at its first missing block: KV prefix reuse is only
    /// valid for an unbroken prefix, so later hits after a gap are worthless.
    pub fn match_prefix(&self, hashes: &[BlockHash], num_workers: u16, now_ms: u64) -> Vec<usize> {
        let mut counts = vec![0usize; num_workers as usize];
        let mut alive = WorkerBitset::first_n(num_workers);
        for &h in hashes {
            let Some(&holders) = self.blocks.get(&h) else {
                break;
            };
            alive = alive.intersect(holders);
            if let Some(ttl) = self.ttl_ms {
                for w in alive.iter_workers() {
                    if !self.is_live(w, h, ttl, now_ms) {
                        alive.remove(w);
                    }
                }
            }
            if alive.is_empty() {
                break;
            }
            for w in alive.iter_workers() {
                counts[w.0 as usize] += 1;
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use proptest::prelude::*;

    use super::*;

    fn hs(ids: &[u64]) -> Vec<BlockHash> {
        ids.iter().copied().map(BlockHash).collect()
    }

    fn stored(w: u16, ids: &[u64], now_ms: u64) -> CacheEvent {
        CacheEvent::BlockStored {
            worker: WorkerId(w),
            hashes: hs(ids),
            now_ms,
        }
    }

    #[test]
    fn bitset_ops() {
        let mut s = WorkerBitset::EMPTY;
        assert!(s.is_empty());
        for w in [0, 5, 127] {
            s.insert(WorkerId(w));
        }
        assert!(s.contains(WorkerId(127)) && !s.contains(WorkerId(1)));
        assert!(!s.contains(WorkerId(500)));
        s.remove(WorkerId(5));
        let ws: Vec<_> = s.iter_workers().collect();
        assert_eq!(ws, vec![WorkerId(0), WorkerId(127)]);
        assert_eq!(WorkerBitset::first_n(128).iter_workers().count(), 128);
        assert!(WorkerBitset::first_n(0).is_empty());
    }

    #[test]
    fn gap_stops_streak() {
        let mut idx = GlobalPrefixIndex::new(None);
        idx.apply_event(&stored(0, &[1, 3], 0));
        idx.apply_event(&stored(1, &[1, 2, 3], 0));
        assert_eq!(idx.match_prefix(&hs(&[1, 2, 3]), 3, 0), vec![1, 3, 0]);
    }

    #[test]
    fn removal_and_clear() {
        let mut idx = GlobalPrefixIndex::new(None);
        idx.apply_event(&stored(0, &[1, 2, 3], 0));
        idx.apply_event(&stored(1, &[1, 2], 0));
        idx.apply_event(&CacheEvent::BlockRemoved {
            worker: WorkerId(0),
            hashes: hs(&[2]),
        });
        assert_eq!(idx.match_prefix(&hs(&[1, 2, 3]), 2, 0), vec![1, 2]);

        idx.apply_event(&CacheEvent::AllBlocksCleared {
            worker: WorkerId(1),
        });
        assert_eq!(idx.match_prefix(&hs(&[1, 2, 3]), 2, 0), vec![1, 0]);
        // Block 2 had no holders left and block 3 still has worker 0.
        assert_eq!(idx.len(), 2);

        idx.apply_event(&CacheEvent::AllBlocksCleared {
            worker: WorkerId(0),
        });
        assert!(idx.is_empty());
    }

    #[test]
    fn ttl_expiry() {
        let mut idx = GlobalPrefixIndex::new(Some(100));
        idx.apply_event(&stored(0, &[1, 2], 0));
        idx.apply_event(&stored(0, &[1], 80)); // refresh block 1 only
        let q = hs(&[1, 2]);
        assert_eq!(idx.match_prefix(&q, 1, 100), vec![2]);
        assert_eq!(idx.match_prefix(&q, 1, 101), vec![1]);
        assert_eq!(idx.match_prefix(&q, 1, 181), vec![0]);

        idx.expire(101);
        assert_eq!(idx.len(), 1);
        idx.expire(181);
        assert!(idx.is_empty());
    }

    #[test]
    fn ttl_per_worker() {
        let mut idx = GlobalPrefixIndex::new(Some(10));
        idx.apply_event(&stored(0, &[1], 0));
        idx.apply_event(&stored(1, &[1], 50));
        assert_eq!(idx.match_prefix(&hs(&[1]), 2, 55), vec![0, 1]);
    }

    #[derive(Debug, Clone)]
    enum Op {
        Store(u16, Vec<u64>),
        Remove(u16, Vec<u64>),
        Clear(u16),
    }

    const W: u16 = 4;

    fn op() -> impl Strategy<Value = Op> {
        let ids = prop::collection::vec(0u64..8, 0..6);
        prop_oneof![
            4 => (0..W, ids.clone()).prop_map(|(w, v)| Op::Store(w, v)),
            2 => (0..W, ids).prop_map(|(w, v)| Op::Remove(w, v)),
            1 => (0..W).prop_map(Op::Clear),
        ]
    }

    proptest! {
        #[test]
        fn match_equals_reference_model(
            ops in prop::collection::vec(op(), 0..40),
            query in prop::collection::vec(0u64..8, 0..8),
        ) {
            let mut idx = GlobalPrefixIndex::new(None);
            let mut model: HashSet<(u16, u64)> = HashSet::new();
            for op in &ops {
                let ev = match op {
                    Op::Store(w, v) => {
                        model.extend(v.iter().map(|&h| (*w, h)));
                        stored(*w, v, 0)
                    }
                    Op::Remove(w, v) => {
                        for h in v { model.remove(&(*w, *h)); }
                        CacheEvent::BlockRemoved { worker: WorkerId(*w), hashes: hs(v) }
                    }
                    Op::Clear(w) => {
                        model.retain(|&(mw, _)| mw != *w);
                        CacheEvent::AllBlocksCleared { worker: WorkerId(*w) }
                    }
                };
                idx.apply_event(&ev);
            }

            let got = idx.match_prefix(&hs(&query), W, 0);
            for w in 0..W {
                let want = query.iter().take_while(|&&h| model.contains(&(w, h))).count();
                prop_assert_eq!(got[w as usize], want);
                prop_assert!(got[w as usize] <= query.len());
            }
            let distinct: HashSet<u64> = model.iter().map(|&(_, h)| h).collect();
            prop_assert_eq!(idx.len(), distinct.len());
        }

        #[test]
        fn ttl_match_never_exceeds_exact(
            ops in prop::collection::vec((0..W, prop::collection::vec(0u64..8, 0..6), 0u64..200), 0..30),
            query in prop::collection::vec(0u64..8, 0..8),
            now in 0u64..400,
        ) {
            let mut exact = GlobalPrefixIndex::new(None);
            let mut ttl = GlobalPrefixIndex::new(Some(50));
            for (w, v, t) in &ops {
                exact.apply_event(&stored(*w, v, *t));
                ttl.apply_event(&stored(*w, v, *t));
            }
            let q = hs(&query);
            let a = exact.match_prefix(&q, W, now);
            let b = ttl.match_prefix(&q, W, now);
            for w in 0..W as usize {
                prop_assert!(b[w] <= a[w]);
            }
            // Pruning must not change what a TTL-aware lookup sees.
            ttl.expire(now);
            prop_assert_eq!(ttl.match_prefix(&q, W, now), b);
        }
    }
}
