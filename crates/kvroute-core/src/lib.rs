//! Pure, synchronous, deterministic routing logic shared by `kvroute-sim` and `kvroute-proxy`.
//!
//! No async runtime dependencies may be added to this crate.

/// Tokens per KV block; prefix hashes are chained at this granularity.
pub const BLOCK_SIZE: usize = 512;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        assert_eq!(BLOCK_SIZE, 512);
    }
}
