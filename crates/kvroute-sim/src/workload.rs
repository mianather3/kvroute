use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{ensure, Context};
use kvroute_core::{chain_external_hashes, BlockHash, BLOCK_SIZE};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimRequest {
    pub id: u64,
    pub arrival_ms: u64,
    pub input_tokens: usize,
    pub output_tokens: usize,
    /// Chained hashes of the full prompt blocks only.
    pub block_hashes: Vec<BlockHash>,
}

/// One line of a Mooncake FAST'25 trace. `hash_ids` are per-block content ids
/// (512-token blocks) and include the trailing partial block, if any.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MooncakeRecord {
    pub timestamp: u64,
    pub input_length: usize,
    pub output_length: usize,
    pub hash_ids: Vec<u64>,
}

pub fn parse_mooncake(r: impl BufRead) -> anyhow::Result<Vec<MooncakeRecord>> {
    let mut out = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec = serde_json::from_str(&line)
            .with_context(|| format!("mooncake trace line {}", i + 1))?;
        out.push(rec);
    }
    Ok(out)
}

pub fn load_mooncake(path: &Path) -> anyhow::Result<Vec<MooncakeRecord>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    parse_mooncake(BufReader::new(f))
}

pub fn write_mooncake(mut w: impl Write, recs: &[MooncakeRecord]) -> anyhow::Result<()> {
    for r in recs {
        serde_json::to_writer(&mut w, r)?;
        w.write_all(b"\n")?;
    }
    Ok(())
}

/// Converts trace records into arrival-ordered requests. Timestamps are made
/// relative to the first arrival; `rate_scale > 1` compresses inter-arrival gaps.
pub fn to_requests(recs: &[MooncakeRecord], rate_scale: f64) -> Vec<SimRequest> {
    assert!(rate_scale > 0.0, "rate_scale must be positive");
    let mut order: Vec<&MooncakeRecord> = recs.iter().collect();
    order.sort_by_key(|r| r.timestamp); // stable: ties keep file order
    let t0 = order.first().map_or(0, |r| r.timestamp);
    order
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            let full = (r.input_length / BLOCK_SIZE).min(r.hash_ids.len());
            SimRequest {
                id: i as u64,
                arrival_ms: ((r.timestamp - t0) as f64 / rate_scale).round() as u64,
                input_tokens: r.input_length,
                output_tokens: r.output_length,
                block_hashes: chain_external_hashes(&r.hash_ids[..full]),
            }
        })
        .collect()
}

/// Multi-turn agent sessions: each session shares one of a few system prompts
/// and grows its own history every turn, the access pattern prefix-aware
/// routing is built for.
#[derive(Debug, Clone)]
pub struct AgentWorkload {
    /// Concurrently active sessions; a finished session is replaced immediately.
    pub sessions: usize,
    pub system_prompts: usize,
    /// Mean Poisson arrival rate across all sessions.
    pub rps: f64,
    pub requests: usize,
    pub seed: u64,
}

struct Session {
    blocks: Vec<u64>,
    turns_left: u32,
}

impl AgentWorkload {
    pub fn generate(&self) -> anyhow::Result<Vec<MooncakeRecord>> {
        ensure!(self.sessions > 0, "sessions must be positive");
        ensure!(self.system_prompts > 0, "system_prompts must be positive");
        ensure!(self.rps > 0.0, "rps must be positive");

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut next_id = 0u64;
        let mut fresh = |n: usize| -> Vec<u64> {
            let ids = (next_id..next_id + n as u64).collect();
            next_id += n as u64;
            ids
        };

        let prompts: Vec<Vec<u64>> = (0..self.system_prompts)
            .map(|_| fresh(rng.gen_range(4..=8)))
            .collect();
        let new_session = |rng: &mut ChaCha8Rng| Session {
            blocks: prompts[rng.gen_range(0..prompts.len())].clone(),
            turns_left: rng.gen_range(3..=8),
        };
        let mut sessions: Vec<Session> =
            (0..self.sessions).map(|_| new_session(&mut rng)).collect();

        let mut t_ms = 0.0f64;
        let mut out = Vec::with_capacity(self.requests);
        for _ in 0..self.requests {
            let u: f64 = rng.gen();
            t_ms += -(1.0 - u).ln() / self.rps * 1000.0;

            let s = &mut sessions[rng.gen_range(0..self.sessions)];
            s.blocks.extend(fresh(rng.gen_range(2..=6)));
            let trailing = rng.gen_range(0..BLOCK_SIZE);
            let mut hash_ids = s.blocks.clone();
            if trailing > 0 {
                // Partial block content is turn-specific; next turn rewrites it.
                hash_ids.extend(fresh(1));
            }
            out.push(MooncakeRecord {
                timestamp: t_ms as u64,
                input_length: s.blocks.len() * BLOCK_SIZE + trailing,
                output_length: rng.gen_range(16..=256),
                hash_ids,
            });

            s.turns_left -= 1;
            if s.turns_left == 0 {
                *s = new_session(&mut rng);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wl(seed: u64) -> AgentWorkload {
        AgentWorkload {
            sessions: 8,
            system_prompts: 2,
            rps: 10.0,
            requests: 200,
            seed,
        }
    }

    #[test]
    fn parse_skips_blank_and_reports_line() {
        let src = "{\"timestamp\":5,\"input_length\":10,\"output_length\":2,\"hash_ids\":[1]}\n\n";
        assert_eq!(parse_mooncake(src.as_bytes()).unwrap().len(), 1);
        let err = parse_mooncake("\n{bad}\n".as_bytes()).unwrap_err();
        assert!(format!("{err:#}").contains("line 2"));
    }

    #[test]
    fn requests_normalized_scaled_and_full_blocks_only() {
        let recs = vec![
            MooncakeRecord {
                timestamp: 2000,
                input_length: 1100,
                output_length: 3,
                hash_ids: vec![7, 8, 9],
            },
            MooncakeRecord {
                timestamp: 1000,
                input_length: 512,
                output_length: 4,
                hash_ids: vec![7],
            },
        ];
        let reqs = to_requests(&recs, 2.0);
        assert_eq!(reqs[0].arrival_ms, 0);
        assert_eq!(reqs[1].arrival_ms, 500);
        assert_eq!(reqs[0].block_hashes.len(), 1);
        assert_eq!(reqs[1].block_hashes.len(), 2); // 1100 tokens = 2 full blocks
        assert_eq!(reqs[0].block_hashes[0], reqs[1].block_hashes[0]);
    }

    #[test]
    fn generator_is_deterministic_and_round_trips() {
        let a = wl(7).generate().unwrap();
        assert_eq!(a, wl(7).generate().unwrap());
        assert_ne!(a, wl(8).generate().unwrap());

        let mut buf = Vec::new();
        write_mooncake(&mut buf, &a).unwrap();
        assert_eq!(parse_mooncake(buf.as_slice()).unwrap(), a);
    }

    #[test]
    fn generator_sessions_grow_and_share_prompts() {
        let reqs = to_requests(&wl(1).generate().unwrap(), 1.0);
        assert!(reqs.windows(2).all(|w| w[0].arrival_ms <= w[1].arrival_ms));
        for r in &reqs {
            assert!(r.block_hashes.len() >= 4 + 2);
            assert_eq!(r.block_hashes.len(), r.input_tokens / BLOCK_SIZE);
        }
        // Two system prompts: every request's first block is one of two hashes.
        let mut roots: Vec<_> = reqs.iter().map(|r| r.block_hashes[0]).collect();
        roots.sort();
        roots.dedup();
        assert_eq!(roots.len(), 2);
    }
}
