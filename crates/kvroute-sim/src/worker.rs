use std::collections::{BTreeMap, HashMap, VecDeque};

use kvroute_core::{BlockHash, CacheEvent, WorkerId, WorkerMetrics, BLOCK_SIZE};

use crate::workload::SimRequest;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkerConfig {
    pub kv_capacity_blocks: usize,
    /// Concurrent requests per worker; the rest wait in a FIFO queue.
    pub max_batch_size: usize,
    pub base_overhead_ms: f64,
    pub prefill_tok_per_ms: f64,
    /// Per-request decode speed.
    pub decode_tok_per_ms: f64,
}

impl Default for WorkerConfig {
    /// Roughly a single-GPU 8B-class deployment.
    fn default() -> Self {
        Self {
            kv_capacity_blocks: 512,
            max_batch_size: 8,
            base_overhead_ms: 5.0,
            prefill_tok_per_ms: 10.0,
            decode_tok_per_ms: 0.05,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    refs: u32,
    tick: u64,
}

/// Prefix block cache with vLLM-style pinning: blocks referenced by running
/// requests are pinned; only unpinned blocks sit in the LRU and can be evicted.
/// The `BTreeMap` keyed by release tick keeps eviction order deterministic.
#[derive(Debug, Clone, Default)]
pub struct BlockCache {
    tick: u64,
    entries: HashMap<BlockHash, Entry>,
    lru: BTreeMap<u64, BlockHash>,
}

impl BlockCache {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn pinned(&self) -> usize {
        self.entries.len() - self.lru.len()
    }

    pub fn contains(&self, h: BlockHash) -> bool {
        self.entries.contains_key(&h)
    }

    fn is_pinned(&self, h: BlockHash) -> bool {
        self.entries.get(&h).is_some_and(|e| e.refs > 0)
    }

    /// Resident hashes in sorted order (deterministic snapshot).
    pub fn snapshot(&self) -> Vec<BlockHash> {
        let mut v: Vec<_> = self.entries.keys().copied().collect();
        v.sort_unstable();
        v
    }

    /// Pins `h`, inserting it if absent. Returns whether it was resident.
    pub fn acquire(&mut self, h: BlockHash) -> bool {
        match self.entries.get_mut(&h) {
            Some(e) => {
                if e.refs == 0 {
                    self.lru.remove(&e.tick);
                }
                e.refs += 1;
                true
            }
            None => {
                self.entries.insert(h, Entry { refs: 1, tick: 0 });
                false
            }
        }
    }

    /// Unpins `h`; at zero refs it becomes the most-recently-used evictable block.
    pub fn release(&mut self, h: BlockHash) {
        let e = self
            .entries
            .get_mut(&h)
            .expect("release of non-resident block");
        e.refs -= 1;
        if e.refs == 0 {
            self.tick += 1;
            e.tick = self.tick;
            self.lru.insert(self.tick, h);
        }
    }

    pub fn evict_lru(&mut self) -> Option<BlockHash> {
        let (_, h) = self.lru.pop_first()?;
        self.entries.remove(&h);
        Some(h)
    }
}

/// Result of starting a request on a worker.
#[derive(Debug, Clone, PartialEq)]
pub struct Started {
    /// Contiguous prefix blocks served from cache.
    pub matched_blocks: usize,
    pub prefill_us: u64,
    pub decode_us: u64,
    /// True cache changes, for routers that consume worker events.
    pub events: Vec<CacheEvent>,
}

/// One simulated vLLM worker: a FIFO queue feeding up to `max_batch_size`
/// slots, gated by KV blocks. Running requests pin their prompt blocks plus
/// private blocks (partial tail + decode); everything else is evictable cache.
#[derive(Debug, Clone)]
pub struct Worker {
    pub id: WorkerId,
    cfg: WorkerConfig,
    cache: BlockCache,
    running: usize,
    private_blocks: usize,
    waiting: VecDeque<usize>,
    waiting_blocks: usize,
    in_flight_tokens: usize,
    pub hit_blocks: u64,
    pub miss_blocks: u64,
    pub evictions: u64,
}

/// Blocks a request occupies for its whole lifetime.
fn kv_blocks(r: &SimRequest) -> usize {
    (r.input_tokens + r.output_tokens).div_ceil(BLOCK_SIZE)
}

/// Blocks only this request can use: the partial prompt tail and decode output.
fn private_blocks(r: &SimRequest) -> usize {
    kv_blocks(r).saturating_sub(r.block_hashes.len())
}

fn ms_to_us(ms: f64) -> u64 {
    (ms * 1000.0).round() as u64
}

impl Worker {
    pub fn new(id: WorkerId, cfg: WorkerConfig) -> Self {
        assert!(cfg.max_batch_size > 0, "max_batch_size must be positive");
        Self {
            id,
            cfg,
            cache: BlockCache::default(),
            running: 0,
            private_blocks: 0,
            waiting: VecDeque::new(),
            waiting_blocks: 0,
            in_flight_tokens: 0,
            hit_blocks: 0,
            miss_blocks: 0,
            evictions: 0,
        }
    }

    fn pinned_blocks(&self) -> usize {
        self.cache.pinned() + self.private_blocks
    }

    /// `queue_depth` is running + waiting. `kv_used_blocks` is committed KV:
    /// blocks pinned by running requests plus the full footprint of queued
    /// ones. Unpinned cache is reclaimable, so it is not counted.
    pub fn metrics(&self) -> WorkerMetrics {
        WorkerMetrics {
            worker: self.id,
            queue_depth: self.running + self.waiting.len(),
            in_flight_tokens: self.in_flight_tokens,
            kv_used_blocks: self.pinned_blocks() + self.waiting_blocks,
            kv_capacity_blocks: self.cfg.kv_capacity_blocks,
            healthy: true,
        }
    }

    pub fn cache(&self) -> &BlockCache {
        &self.cache
    }

    /// Queues request `idx`; call [`Worker::pop_startable`] to start what fits.
    pub fn admit(&mut self, idx: usize, req: &SimRequest) {
        self.waiting.push_back(idx);
        self.waiting_blocks += kv_blocks(req);
        self.in_flight_tokens += req.input_tokens + req.output_tokens;
    }

    /// Whether `req` can enter a slot now without exceeding KV capacity.
    /// An idle worker always accepts, so oversized requests cannot deadlock.
    fn fits(&self, req: &SimRequest) -> bool {
        if self.running >= self.cfg.max_batch_size {
            return false;
        }
        if self.running == 0 {
            return true;
        }
        let newly_pinned = req
            .block_hashes
            .iter()
            .filter(|&&h| !self.cache.is_pinned(h))
            .count();
        self.pinned_blocks() + newly_pinned + private_blocks(req) <= self.cfg.kv_capacity_blocks
    }

    /// Dequeues the head request if it fits (strict FIFO: no overtaking).
    pub fn pop_startable(&mut self, reqs: &[SimRequest]) -> Option<usize> {
        let &idx = self.waiting.front()?;
        if !self.fits(&reqs[idx]) {
            return None;
        }
        self.waiting.pop_front();
        self.waiting_blocks -= kv_blocks(&reqs[idx]);
        Some(idx)
    }

    /// Evicts unpinned blocks until usage fits capacity (or nothing is evictable).
    fn reclaim(&mut self) -> Vec<BlockHash> {
        let mut evicted = Vec::new();
        while self.cache.len() + self.private_blocks > self.cfg.kv_capacity_blocks {
            match self.cache.evict_lru() {
                Some(h) => evicted.push(h),
                None => break,
            }
        }
        self.evictions += evicted.len() as u64;
        evicted
    }

    /// Pins the request's blocks, evicts to make room, and computes service time.
    pub fn start(&mut self, req: &SimRequest, now_ms: u64) -> Started {
        let hashes = &req.block_hashes;
        let matched = hashes
            .iter()
            .take_while(|&&h| self.cache.contains(h))
            .count();
        self.hit_blocks += matched as u64;
        self.miss_blocks += (hashes.len() - matched) as u64;

        // Blocks past a gap are recomputed (prefill) but reuse resident storage.
        let inserted: Vec<_> = hashes
            .iter()
            .copied()
            .filter(|&h| !self.cache.acquire(h))
            .collect();
        self.running += 1;
        self.private_blocks += private_blocks(req);
        // Pinned blocks are never evicted, so `evicted` and `inserted` are disjoint.
        let evicted = self.reclaim();

        let mut events = Vec::with_capacity(2);
        if !evicted.is_empty() {
            events.push(CacheEvent::BlockRemoved {
                worker: self.id,
                hashes: evicted,
            });
        }
        if !inserted.is_empty() {
            events.push(CacheEvent::BlockStored {
                worker: self.id,
                hashes: inserted,
                now_ms,
            });
        }

        let prefill_tokens = req.input_tokens - matched * BLOCK_SIZE;
        let prefill_ms =
            self.cfg.base_overhead_ms + prefill_tokens as f64 / self.cfg.prefill_tok_per_ms;
        let decode_ms = req.output_tokens as f64 / self.cfg.decode_tok_per_ms;
        Started {
            matched_blocks: matched,
            prefill_us: ms_to_us(prefill_ms),
            decode_us: ms_to_us(decode_ms),
            events,
        }
    }

    /// Unpins a finished request's blocks (tail first, so prefix blocks stay
    /// warmest). Returns a removal event if an oversized request had pushed
    /// the cache past capacity.
    pub fn finish(&mut self, req: &SimRequest) -> Option<CacheEvent> {
        for &h in req.block_hashes.iter().rev() {
            self.cache.release(h);
        }
        self.running -= 1;
        self.private_blocks -= private_blocks(req);
        self.in_flight_tokens -= req.input_tokens + req.output_tokens;
        let evicted = self.reclaim();
        if evicted.is_empty() {
            return None;
        }
        Some(CacheEvent::BlockRemoved {
            worker: self.id,
            hashes: evicted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(ids: &[u64], trailing: usize) -> SimRequest {
        SimRequest {
            id: 0,
            arrival_ms: 0,
            input_tokens: ids.len() * BLOCK_SIZE + trailing,
            output_tokens: 0,
            block_hashes: ids.iter().copied().map(BlockHash).collect(),
        }
    }

    fn cfg(kv: usize, batch: usize) -> WorkerConfig {
        WorkerConfig {
            kv_capacity_blocks: kv,
            max_batch_size: batch,
            ..WorkerConfig::default()
        }
    }

    /// Admits and immediately starts `r` (the worker must have room).
    fn run(w: &mut Worker, r: &SimRequest) -> Started {
        w.admit(0, r);
        assert_eq!(w.pop_startable(std::slice::from_ref(r)), Some(0));
        w.start(r, 0)
    }

    fn split(events: &[CacheEvent]) -> (Vec<u64>, Vec<u64>) {
        let (mut removed, mut stored) = (vec![], vec![]);
        for e in events {
            match e {
                CacheEvent::BlockRemoved { hashes, .. } => {
                    removed.extend(hashes.iter().map(|h| h.0))
                }
                CacheEvent::BlockStored { hashes, .. } => stored.extend(hashes.iter().map(|h| h.0)),
                CacheEvent::AllBlocksCleared { .. } => unreachable!(),
            }
        }
        (removed, stored)
    }

    #[test]
    fn pinned_blocks_are_not_evictable() {
        let mut c = BlockCache::default();
        assert!(!c.acquire(BlockHash(1)));
        assert!(!c.acquire(BlockHash(2)));
        assert_eq!(c.evict_lru(), None);
        c.release(BlockHash(2));
        c.release(BlockHash(1));
        assert!(c.acquire(BlockHash(1)));
        assert_eq!(c.pinned(), 1);
        assert_eq!(c.evict_lru(), Some(BlockHash(2)));
        assert_eq!(c.evict_lru(), None);
        assert_eq!(c.snapshot(), vec![BlockHash(1)]);
    }

    #[test]
    fn cached_prefix_skips_prefill() {
        let cfg = WorkerConfig {
            kv_capacity_blocks: 16,
            base_overhead_ms: 1.0,
            prefill_tok_per_ms: 512.0,
            decode_tok_per_ms: 1.0,
            ..WorkerConfig::default()
        };
        let mut w = Worker::new(WorkerId(0), cfg);
        let mut r = req(&[1, 2, 3], 100);
        r.output_tokens = 10;
        let cold = run(&mut w, &r);
        assert_eq!(cold.matched_blocks, 0);
        assert_eq!(
            cold.prefill_us,
            ms_to_us(1.0 + (3.0 * 512.0 + 100.0) / 512.0)
        );
        assert_eq!(cold.decode_us, 10_000);
        w.finish(&r);

        let warm = w.start(&req(&[1, 2, 9], 100), 0);
        assert_eq!(warm.matched_blocks, 2);
        assert_eq!(warm.prefill_us, ms_to_us(1.0 + 612.0 / 512.0));
        assert_eq!(split(&warm.events), (vec![], vec![9]));
    }

    #[test]
    fn gap_breaks_match() {
        let mut w = Worker::new(WorkerId(0), cfg(16, 8));
        let a = req(&[1, 2, 3], 0);
        run(&mut w, &a);
        w.finish(&a);
        let s = w.start(&req(&[1, 5, 3], 0), 0);
        assert_eq!(s.matched_blocks, 1);
        assert_eq!(
            split(&s.events),
            (vec![], vec![5]),
            "block 3 reuses resident storage"
        );
    }

    #[test]
    fn eviction_takes_lru_tail_first() {
        let mut w = Worker::new(WorkerId(1), cfg(4, 8));
        let a = req(&[1, 2, 3], 0);
        run(&mut w, &a);
        w.finish(&a); // released 3, 2, 1: block 3 is now least recent
        let s = w.start(&req(&[7, 8], 0), 5);
        assert_eq!(split(&s.events), (vec![3], vec![7, 8]));
        assert_eq!(w.evictions, 1);
        assert_eq!(w.cache().len(), 4);
    }

    #[test]
    fn kv_gates_start_and_idle_worker_accepts_oversized() {
        let mut w = Worker::new(WorkerId(0), cfg(4, 8));
        let reqs = vec![
            req(&[1, 2, 3], 0),
            req(&[4, 5], 0),
            req(&[1, 2, 3, 6, 7, 8], 0),
        ];
        w.admit(0, &reqs[0]);
        w.admit(1, &reqs[1]);
        assert_eq!(w.metrics().kv_used_blocks, 5);
        assert_eq!(w.pop_startable(&reqs), Some(0));
        w.start(&reqs[0], 0);
        assert_eq!(w.pop_startable(&reqs), None, "3 pinned + 2 new > 4");
        assert_eq!(w.metrics().queue_depth, 2);

        w.finish(&reqs[0]);
        assert_eq!(w.pop_startable(&reqs), Some(1));
        w.start(&reqs[1], 0);
        w.finish(&reqs[1]);

        // 6 blocks into capacity 4 on an idle worker: runs, then trims on finish.
        w.admit(2, &reqs[2]);
        assert_eq!(w.pop_startable(&reqs), Some(2));
        let s = w.start(&reqs[2], 0);
        assert_eq!(w.cache().len(), 6);
        assert_eq!(s.matched_blocks, 2, "block 3 was evicted earlier");
        let ev = w.finish(&reqs[2]).expect("trim after oversized request");
        assert_eq!(split(&[ev]).0.len(), 2);
        assert_eq!(w.cache().len(), 4);
        assert_eq!(w.metrics().kv_used_blocks, 0);
    }

    #[test]
    fn fifo_batch_slots() {
        let mut w = Worker::new(WorkerId(0), cfg(512, 1));
        let reqs = vec![req(&[1], 0); 3];
        for (i, r) in reqs.iter().enumerate() {
            w.admit(i, r);
        }
        assert_eq!(w.pop_startable(&reqs), Some(0));
        w.start(&reqs[0], 0);
        assert_eq!(w.pop_startable(&reqs), None);
        assert_eq!(w.metrics().queue_depth, 3);
        w.finish(&reqs[0]);
        assert_eq!(w.pop_startable(&reqs), Some(1));
    }
}
