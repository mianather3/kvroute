# kvroute design

kvroute routes LLM inference requests to the worker most likely to hold the
request's prompt prefix in its KV cache, without letting cache affinity
overload a worker. One routing implementation (`kvroute-core`) is shared by an
offline discrete-event simulator (`kvroute-sim`) and an online HTTP/SSE proxy
(`kvroute-proxy`), so what the simulator measures is what the proxy runs.

```
                ┌──────────────────────────────────────────────┐
                │ kvroute-core  (sync, deterministic, no I/O)  │
                │  hash  ·  GlobalPrefixIndex  ·  Router       │
                │  ScoringWeights  ·  AdmissionConfig          │
                └───────────────┬──────────────────┬───────────┘
                     same code  │                  │  same code
          ┌─────────────────────┴──┐        ┌──────┴──────────────────────┐
          │ kvroute-sim            │        │ kvroute-proxy               │
          │ DES event heap         │        │ axum + tokio + reqwest      │
          │ pinned-LRU workers     │        │ ArcSwap<Router> snapshots   │
          │ fault injection        │        │ single-writer event actor   │
          └────────────────────────┘        └─────────────────────────────┘
```

## 1. Prefix identity: chained block hashing

Prompts are cut into 512-token blocks (`BLOCK_SIZE`). A trailing partial block
is ignored, matching vLLM's full-block prefix caching. Each block hash folds
every token of the block into the previous block's hash:

```
h_0 = fold(SEED, block_0)        fold(h, w) = splitmix64(h ^ w), per token
h_i = fold(h_{i-1}, block_i)
```

SplitMix64's finalizer is a bijection, so for a fixed `h` each step is
injective in `w`. `h_i` therefore commits to the whole prefix
`tokens[0 .. (i+1)·512)`. Two requests share `h_i` exactly when they share
that prefix, up to 64-bit collisions; a property test checks this on a
3-symbol alphabet where real collisions between prefixes are frequent. A
change in block `k` changes `h_k` and every later hash, but no earlier one.

**Mooncake compatibility.** Mooncake FAST'25 traces give per-block content ids
(`hash_ids`) rather than tokens. `chain_external_hashes` chains those ids
through the same `fold`, so the same id at two positions gets two different
`BlockHash`es, exactly as token hashing would. The loader keeps only the
`input_length / 512` full blocks.

**Matching.** `GlobalPrefixIndex` maps `BlockHash → u128` worker bitset (up to
128 workers). `match_prefix` walks the request's hashes, intersecting a
"still matching" bitset at each block and stopping when it is empty. A worker
holding blocks 0 and 2 but not 1 matches 1 block: KV reuse needs an unbroken
prefix. A property test checks this against a set-based reference model under
random store/remove/clear sequences.

**Scoring.** `score = α·matched − β·queue_depth − γ·kv_util`. Ties break on
matched blocks, then shorter queue, then lower worker id, compared with
`f64::total_cmp`, so identical inputs always give identical decisions.

## 2. KV-budget admission

`route_with_admission` drops a candidate worker when
`(kv_used + needed) / kv_capacity > max_kv_util`, where
`needed = blocks − matched`, or when its queue is at `max_queue_depth`. It then
scores the remaining workers. Because a cache hit lowers `needed`, a warm
worker can still fit when a cold one cannot. An idle worker
(`kv_used == 0`) always passes the KV check, so a request larger than the
budget still runs. If no worker passes, the result is `Defer { reason }`.

In the simulator, deferred requests wait in one queue that is retried
oldest-first every `defer_backoff_ms`. Each retry admits every request that
now fits (first-fit), and new arrivals queue behind it. TTFT is measured from
the original arrival, so deferral cannot hide latency. Two alternatives were
tried and rejected, measured at 40 rps with 128-block workers and 1500
requests:

- Per-request retry timers let new arrivals overtake old ones: p99 TTFT
  rose from 51 s to 72 s.
- A strict head-of-line queue removed reordering entirely and dropped the
  hit rate to 0.42.

## 3. Simulator model (`kvroute-sim`)

- **Events.** A `BinaryHeap<Reverse<Event>>` ordered by `(t_us, seq)`. Time is
  integer microseconds and `seq` is the insertion counter, so simultaneous
  events resolve the same way on every run. Fault injection uses a seeded
  `ChaCha8Rng`. The same seed gives byte-identical CSVs, checked by an
  integration test and by `make bench`.
- **Workers.** A FIFO queue feeds `max_batch_size` slots. As in vLLM, a running
  request **pins** its prompt blocks plus its private blocks (partial tail and
  decode output). Unpinned blocks sit in an LRU (a `BTreeMap` keyed by release
  tick) and are the only ones that can be evicted. A queued request starts only
  when its newly pinned blocks fit. Blocks are released tail-first on
  completion, so prefix heads stay warmest.
  Without pinning, a KV budget would cost nothing and admission control would
  have nothing to protect.
- **Timing.** `prefill_ms = 5 + (uncached_blocks·512 + tail) / 10 tok/ms`,
  `decode_ms = output_tokens / 0.05 tok/ms`, `TTFT = queue_wait + prefill`.
  This approximates one GPU serving an 8B-class model. Batch slots do not
  share compute, which flatters throughput under heavy load.
- **Router inputs.** `queue_depth` is running plus waiting requests.
  `kv_used_blocks` is committed KV: pinned blocks plus the full footprint of
  queued requests. Reusable cache is not counted (counting only pinned blocks
  was tested and was worse at every load). Cache events reach the router after
  `event_delay_ms` and can be dropped with probability `event_drop_prob`.
  `router_restart_at_ms` wipes the index; workers optionally republish their
  resident blocks after `resync_delay_ms`.
- **Workload.** `N` concurrent agent sessions with Poisson arrivals. Each
  session uses one of 8 shared system prompts (4–8 blocks) and adds 2–6 history
  blocks per turn over 3–8 turns. The generator emits Mooncake records, which
  go through the same loader as real traces; `--dump-trace` writes them out.

## 4. Proxy architecture (`kvroute-proxy`)

```
 client ──POST /v1/{chat/}completions──▶ handler (any tokio task)
                                          │ 1. hash prompt (token_ids | prompt | messages)
                                          │ 2. Semaphore::try_acquire  ──full──▶ 503 Retry-After
                                          │ 3. ArcSwap::load()  ← lock-free snapshot
                                          │    + per-worker atomics
                                          │ 4. route_with_admission ──Defer──▶ 503 Retry-After
                                          │ 5. InFlightGuard (+queue_depth, +tokens, holds permit)
                                          ▼
                                   reqwest → worker ── bytes_stream ──▶ client
                                                 (guard is moved into the stream)

 worker ──POST /internal/kv-events──▶ try_send ──bounded mpsc(4096)──▶ event actor
                                       └ full: drop + count            (only index writer)
 admin ──POST /internal/reset-index──▶ send(ResetIndex) ──────────────▶   drain ≤1024 msgs
                                                                          apply, restamp clock
                                                                          ArcSwap::store(new Arc)
```

- **The request path takes no locks.** Handlers read an immutable
  `Arc<Router>` snapshot and per-worker `AtomicUsize`/`AtomicBool` values. The
  crate's `src/` contains no `Mutex` or `RwLock`.
- **One writer.** `run_event_actor` owns the only mutable `Router`. It drains
  up to 1024 messages per wake-up and publishes one snapshot, so a burst of
  events costs one clone, not one per event. It replaces each
  `BlockStored.now_ms` with the proxy's monotonic clock, because publisher
  clocks don't match it and TTL expiry needs one time base.
- **Backpressure.** Event ingestion uses `try_send`. When the queue is full
  the batch is dropped and `kvroute_events_dropped_total` is incremented. A
  stale index is better than a blocked worker. `ResetIndex` uses `send().await`
  so a reset is never lost.
- **`InFlightGuard`.** Increments the worker's `queue_depth` and
  `in_flight_tokens` and holds a global semaphore permit. It is moved into the
  response body stream, so normal completion, upstream errors and client
  disconnects (hyper drops the body) all release it the same way. An
  end-to-end test aborts a live SSE stream and checks the counters return to 0.
- **Prompts without token ids** are packed 4 UTF-8 bytes per pseudo-token
  (about 2 KiB of text per block). Chat messages are serialized as
  `<|role|>content…` in order, so each turn extends the previous turn's byte
  string and shares its block hashes.

## 5. Trade-offs found by benchmarking

All numbers come from `make bench` (seed 42, 2000 requests, 8 workers); see
`README.md` for the plots.

**Queue weight β = 2.0.** Sweeping β over {0.5, 1, 2, 4, 8, 16} at 14 rps gave
these p99 TTFTs: 2579, 1313, **759**, 890, 1131 and 1474 ms. A small β
over-concentrates sessions on warm workers; a large β throws affinity away
(hit rate fell from 0.80 to 0.59). The best β grows with load: at 16 rps,
β = 4 beats β = 2 on p99 (915 vs 1108 ms) at a hit-rate cost (0.74 vs 0.77).
A fixed β is a compromise; a β that adapts to queue pressure is an open
improvement.

**Dispatch-to-event lag is the main weakness of exact indexing under
overload.** The exact index learns a block only when the request *starts* on
a worker and the worker reports it. Above cluster capacity (about 20 rps
here), requests wait seconds in worker queues first (`kvroute` p50 TTFT is
4.1 s at 24 rps and 11.5 s at 28 rps). A session's next turn arrives before
its previous turn has even started, finds no match, and is sent elsewhere.
Hit rate falls from 0.77 to 0.53 by 32 rps, while `approx-ttl`, which records
blocks at dispatch, holds 0.77. A one-off experiment that also recorded blocks
at dispatch for `kvroute-no-admission` reproduced `approx-ttl` exactly (0.772
hit rate, 7262 ms p99 at 24 rps). The fix is a short-lived "pending" overlay
of dispatched-but-unconfirmed blocks, which real events then confirm or
retract. It is not implemented yet.

**Admission control has a crossover.** With 128-block workers, admission is
worse below about 17–19 rps: it diverts sessions from their over-budget warm
worker to cold workers (at 16 rps, hit rate 0.50 vs 0.65 and p99 3.5 s vs
2.2 s). Above that it wins (at 28 rps, 21% fewer evictions and p99 36 s vs
51 s). Part of the overload win is the dispatch-lag effect above: deferring
requests in the router keeps them out of worker queues. With dispatch-aware
indexing, the 28 rps gap reversed (hit rate 0.575 with admission vs 0.601
without). A better admission rule would defer, not divert, when the
over-budget worker is the one holding the prefix.

**Restart recovery.** The prefix index is soft state. After a wipe, a worker
resync (republishing resident blocks) restores the hit rate in one 2.5 s
window. Without resync, blocks that stay resident are never re-announced; the
index relearns them only as they are evicted and recomputed, which took about
15 s here.
