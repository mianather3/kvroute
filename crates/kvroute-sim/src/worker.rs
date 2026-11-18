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

/// Exact LRU over block hashes. The `BTreeMap` keyed by access tick gives
/// O(log n) touch/evict with fully deterministic ordering.
#[derive(Debug, Clone)]
pub struct LruCache {
    cap: usize,
    tick: u64,
    stamp: HashMap<BlockHash, u64>,
    order: BTreeMap<u64, BlockHash>,
}

impl LruCache {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            tick: 0,
            stamp: HashMap::new(),
            order: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.stamp.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stamp.is_empty()
    }

    pub fn contains(&self, h: BlockHash) -> bool {
        self.stamp.contains_key(&h)
    }

    /// Marks `h` most-recently-used, inserting it if absent.
    /// Returns `(was_resident, evicted)`.
    pub fn access(&mut self, h: BlockHash) -> (bool, Option<BlockHash>) {
        self.tick += 1;
        if let Some(old) = self.stamp.insert(h, self.tick) {
            self.order.remove(&old);
            self.order.insert(self.tick, h);
            return (true, None);
        }
        self.order.insert(self.tick, h);
        if self.stamp.len() <= self.cap {
            return (false, None);
        }
        let (_, victim) = self.order.pop_first().expect("non-empty");
        self.stamp.remove(&victim);
        (false, Some(victim))
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

#[derive(Debug, Clone)]
pub struct Worker {
    pub id: WorkerId,
    cfg: WorkerConfig,
    cache: LruCache,
    running: usize,
    waiting: VecDeque<usize>,
    outstanding: usize,
    in_flight_tokens: usize,
    committed_blocks: usize,
    pub hit_blocks: u64,
    pub miss_blocks: u64,
    pub evictions: u64,
}

fn kv_blocks(r: &SimRequest) -> usize {
    (r.input_tokens + r.output_tokens).div_ceil(BLOCK_SIZE)
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
            cache: LruCache::new(cfg.kv_capacity_blocks),
            running: 0,
            waiting: VecDeque::new(),
            outstanding: 0,
            in_flight_tokens: 0,
            committed_blocks: 0,
            hit_blocks: 0,
            miss_blocks: 0,
            evictions: 0,
        }
    }

    /// `queue_depth` counts running plus waiting requests; KV usage is the
    /// footprint those requests will need, not resident (reusable) cache.
    pub fn metrics(&self) -> WorkerMetrics {
        WorkerMetrics {
            worker: self.id,
            queue_depth: self.outstanding,
            in_flight_tokens: self.in_flight_tokens,
            kv_used_blocks: self.committed_blocks,
            kv_capacity_blocks: self.cfg.kv_capacity_blocks,
            healthy: true,
        }
    }

    pub fn cache(&self) -> &LruCache {
        &self.cache
    }

    /// Accepts request `idx`; returns true if a batch slot is free and it
    /// should start now, otherwise it is queued.
    pub fn admit(&mut self, idx: usize, req: &SimRequest) -> bool {
        self.outstanding += 1;
        self.in_flight_tokens += req.input_tokens + req.output_tokens;
        self.committed_blocks += kv_blocks(req);
        if self.running < self.cfg.max_batch_size {
            self.running += 1;
            true
        } else {
            self.waiting.push_back(idx);
            false
        }
    }

    /// Runs the cache lookup and computes service time for a request entering a slot.
    pub fn start(&mut self, req: &SimRequest, now_ms: u64) -> Started {
        let hashes = &req.block_hashes;
        let matched = hashes
            .iter()
            .take_while(|&&h| self.cache.contains(h))
            .count();

        // Touch every block in order: hits refresh recency, misses get computed
        // and inserted. Blocks past a gap are recomputed even if resident.
        let mut inserted = Vec::new();
        let mut evicted = Vec::new();
        for &h in hashes {
            let (hit, victim) = self.cache.access(h);
            if !hit {
                inserted.push(h);
            }
            evicted.extend(victim);
        }
        self.evictions += evicted.len() as u64;
        self.hit_blocks += matched as u64;
        self.miss_blocks += (hashes.len() - matched) as u64;

        // Report net changes only, so a block inserted then evicted within this
        // request never appears as both stored and removed.
        evicted.retain(|h| !inserted.contains(h));
        inserted.retain(|&h| self.cache.contains(h));
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

    /// Releases a finished request; returns the next queued request to start, if any.
    pub fn finish(&mut self, req: &SimRequest) -> Option<usize> {
        self.outstanding -= 1;
        self.in_flight_tokens -= req.input_tokens + req.output_tokens;
        self.committed_blocks -= kv_blocks(req);
        let next = self.waiting.pop_front();
        if next.is_none() {
            self.running -= 1;
        }
        next
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
            output_tokens: 10,
            block_hashes: ids.iter().copied().map(BlockHash).collect(),
        }
    }

    #[test]
    fn lru_evicts_least_recent() {
        let mut c = LruCache::new(2);
        c.access(BlockHash(1));
        c.access(BlockHash(2));
        assert_eq!(c.access(BlockHash(1)), (true, None));
        assert_eq!(c.access(BlockHash(3)), (false, Some(BlockHash(2))));
        assert!(c.contains(BlockHash(1)) && c.contains(BlockHash(3)));
        assert_eq!(c.len(), 2);
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
        let cold = w.start(&req(&[1, 2, 3], 100), 0);
        assert_eq!(cold.matched_blocks, 0);
        assert_eq!(
            cold.prefill_us,
            ms_to_us(1.0 + (3.0 * 512.0 + 100.0) / 512.0)
        );
        assert_eq!(cold.decode_us, 10_000);

        let warm = w.start(&req(&[1, 2, 9], 100), 0);
        assert_eq!(warm.matched_blocks, 2);
        assert_eq!(warm.prefill_us, ms_to_us(1.0 + 612.0 / 512.0));
        assert_eq!(
            warm.events,
            vec![CacheEvent::BlockStored {
                worker: WorkerId(0),
                hashes: vec![BlockHash(9)],
                now_ms: 0
            }]
        );
    }

    #[test]
    fn gap_breaks_match_and_evictions_emit_net_events() {
        let cfg = WorkerConfig {
            kv_capacity_blocks: 3,
            ..WorkerConfig::default()
        };
        let mut w = Worker::new(WorkerId(1), cfg);
        w.start(&req(&[1, 2, 3], 0), 0);
        assert_eq!(w.start(&req(&[1, 5, 3], 0), 0).matched_blocks, 1);

        // 4 fresh blocks into capacity 3: block 7 is stored then evicted by 10,
        // so it must appear in neither event.
        let s = w.start(&req(&[7, 8, 9, 10], 0), 5);
        let (mut removed, mut stored): (Vec<BlockHash>, Vec<BlockHash>) = (vec![], vec![]);
        for e in &s.events {
            match e {
                CacheEvent::BlockRemoved { hashes, .. } => removed.extend(hashes),
                CacheEvent::BlockStored { hashes, .. } => stored.extend(hashes),
                _ => unreachable!(),
            }
        }
        assert!(removed.iter().all(|h| !stored.contains(h)));
        assert_eq!(stored, vec![BlockHash(8), BlockHash(9), BlockHash(10)]);
        assert_eq!(removed, vec![BlockHash(1), BlockHash(5), BlockHash(3)]);
        assert_eq!(w.evictions, 5); // block 2 earlier, then 1, 5, 3, 7
    }

    #[test]
    fn fifo_batch_slots() {
        let cfg = WorkerConfig {
            max_batch_size: 1,
            ..WorkerConfig::default()
        };
        let mut w = Worker::new(WorkerId(0), cfg);
        let r = req(&[1], 0);
        assert!(w.admit(0, &r));
        assert!(!w.admit(1, &r));
        assert!(!w.admit(2, &r));
        assert_eq!(w.metrics().queue_depth, 3);
        assert_eq!(w.finish(&r), Some(1));
        assert_eq!(w.finish(&r), Some(2));
        assert_eq!(w.finish(&r), None);
        assert_eq!(w.metrics().queue_depth, 0);
    }
}
