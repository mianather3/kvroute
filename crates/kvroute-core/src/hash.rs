use serde::{Deserialize, Serialize};

/// Tokens per KV block; prefix hashes are chained at this granularity.
pub const BLOCK_SIZE: usize = 512;

/// Hash of a full KV block chained with every block before it, so equal
/// hashes imply equal prefixes (up to 64-bit collisions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockHash(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerId(pub u16);

/// Chain root. Arbitrary but fixed: changing it invalidates every stored hash.
const SEED: u64 = 0x6b76_726f_7574_6531; // "kvroute1"

/// SplitMix64 finalizer. Bijective, so `fold(h, w)` is injective in `w` for a fixed `h`.
#[inline]
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn fold(h: u64, w: u64) -> u64 {
    mix(h ^ w)
}

/// Chained hashes of every *full* block in `tokens`; a trailing partial block
/// is ignored, matching vLLM's full-block prefix caching.
pub fn hash_prefix_blocks(tokens: &[u32], block_size: usize) -> Vec<BlockHash> {
    assert!(block_size > 0, "block_size must be non-zero");
    let mut h = SEED;
    tokens
        .chunks_exact(block_size)
        .map(|block| {
            h = block.iter().fold(h, |acc, &t| fold(acc, t as u64));
            BlockHash(h)
        })
        .collect()
}

/// Chains per-block ids (e.g. Mooncake `hash_ids`, which identify a block's
/// content but not its position) into prefix-identifying hashes using the
/// same construction as [`hash_prefix_blocks`].
pub fn chain_external_hashes(raw_block_ids: &[u64]) -> Vec<BlockHash> {
    let mut h = SEED;
    raw_block_ids
        .iter()
        .map(|&id| {
            h = fold(h, id);
            BlockHash(h)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const BS: usize = 4;

    #[test]
    fn partial_block_ignored() {
        assert!(hash_prefix_blocks(&[1, 2, 3], BS).is_empty());
        assert_eq!(hash_prefix_blocks(&[1, 2, 3, 4, 5, 6], BS).len(), 1);
        assert_eq!(
            hash_prefix_blocks(&vec![7; BLOCK_SIZE * 2 + 1], BLOCK_SIZE).len(),
            2
        );
    }

    #[test]
    fn divergence_at_block_k() {
        let base: Vec<u32> = (0..(BS * 5) as u32).collect();
        let a = hash_prefix_blocks(&base, BS);
        for k in 0..5 {
            let mut alt = base.clone();
            alt[k * BS + 1] ^= 0xFFFF;
            let b = hash_prefix_blocks(&alt, BS);
            assert_eq!(a[..k], b[..k], "blocks before {k} must match");
            for i in k..5 {
                assert_ne!(a[i], b[i], "block {i} must differ after divergence at {k}");
            }
        }
    }

    #[test]
    fn same_block_content_differs_by_position() {
        let h = hash_prefix_blocks(&[9; BS * 2], BS);
        assert_ne!(h[0], h[1]);
        let e = chain_external_hashes(&[42, 42]);
        assert_ne!(e[0], e[1]);
    }

    #[test]
    fn external_divergence() {
        let a = chain_external_hashes(&[1, 2, 3, 4]);
        let b = chain_external_hashes(&[1, 2, 9, 4]);
        assert_eq!(a[..2], b[..2]);
        assert_ne!(a[2], b[2]);
        assert_ne!(a[3], b[3]);
    }

    proptest! {
        #[test]
        fn shared_prefix_hashes_match(
            prefix in prop::collection::vec(any::<u32>(), 0..(BS * 6)),
            ta in prop::collection::vec(any::<u32>(), 0..(BS * 4)),
            tb in prop::collection::vec(any::<u32>(), 0..(BS * 4)),
        ) {
            let a: Vec<u32> = prefix.iter().chain(&ta).copied().collect();
            let b: Vec<u32> = prefix.iter().chain(&tb).copied().collect();
            let (ha, hb) = (hash_prefix_blocks(&a, BS), hash_prefix_blocks(&b, BS));
            let shared = prefix.len() / BS;
            prop_assert_eq!(&ha[..shared], &hb[..shared]);
            prop_assert_eq!(ha.len(), a.len() / BS);
        }

        #[test]
        fn hashes_equal_iff_prefixes_equal(
            a in prop::collection::vec(0u32..3, 0..(BS * 6)),
            b in prop::collection::vec(0u32..3, 0..(BS * 6)),
        ) {
            let (ha, hb) = (hash_prefix_blocks(&a, BS), hash_prefix_blocks(&b, BS));
            for i in 0..ha.len().min(hb.len()) {
                let end = (i + 1) * BS;
                prop_assert_eq!(ha[i] == hb[i], a[..end] == b[..end]);
            }
        }

        #[test]
        fn external_shared_prefix_matches(
            prefix in prop::collection::vec(any::<u64>(), 0..8),
            ta in prop::collection::vec(any::<u64>(), 0..4),
            tb in prop::collection::vec(any::<u64>(), 0..4),
        ) {
            let a: Vec<u64> = prefix.iter().chain(&ta).copied().collect();
            let b: Vec<u64> = prefix.iter().chain(&tb).copied().collect();
            let (ha, hb) = (chain_external_hashes(&a), chain_external_hashes(&b));
            prop_assert_eq!(&ha[..prefix.len()], &hb[..prefix.len()]);
        }
    }
}
