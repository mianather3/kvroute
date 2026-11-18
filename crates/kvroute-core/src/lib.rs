//! Pure, synchronous, deterministic routing logic shared by `kvroute-sim` and `kvroute-proxy`.
//!
//! No async runtime, I/O, wall-clock time, or global RNG may enter this crate:
//! callers pass `now_ms` explicitly so simulated and live routing are bit-identical.

mod hash;
mod index;
mod scorer;

pub use hash::{chain_external_hashes, hash_prefix_blocks, BlockHash, WorkerId, BLOCK_SIZE};
pub use index::{CacheEvent, GlobalPrefixIndex, WorkerBitset, MAX_WORKERS};
pub use scorer::{RouteDecision, Router, ScoringWeights, WorkerMetrics};
