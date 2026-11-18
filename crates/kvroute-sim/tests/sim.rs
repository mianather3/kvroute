use std::path::Path;

use kvroute_sim::engine::{run, to_csv, SimConfig, Summary};
use kvroute_sim::policy::PolicyKind;
use kvroute_sim::workload::{load_mooncake, to_requests, AgentWorkload, SimRequest};

fn agents(seed: u64) -> Vec<SimRequest> {
    let recs = AgentWorkload {
        sessions: 64,
        system_prompts: 8,
        rps: 14.0,
        requests: 1500,
        seed,
    }
    .generate()
    .unwrap();
    to_requests(&recs, 1.0)
}

fn run_all(reqs: &[SimRequest]) -> Vec<Summary> {
    let cfg = SimConfig::default();
    PolicyKind::ALL
        .iter()
        .map(|&k| run(&cfg, k, reqs))
        .collect()
}

#[test]
fn same_seed_is_bit_identical() {
    let a = to_csv(&run_all(&agents(42)));
    let b = to_csv(&run_all(&agents(42)));
    assert_eq!(a, b);
    assert_ne!(a, to_csv(&run_all(&agents(43))));
}

#[test]
fn kvroute_beats_load_balancers_on_agent_workload() {
    for seed in [1, 2, 3] {
        let rows = run_all(&agents(seed));
        let get = |k: PolicyKind| rows.iter().find(|r| r.policy == k.name()).unwrap();
        let kv = get(PolicyKind::KvRoute);
        for base in [PolicyKind::RoundRobin, PolicyKind::LeastLoaded] {
            let b = get(base);
            assert!(
                kv.prefix_hit_rate > b.prefix_hit_rate + 0.2,
                "seed {seed}: hit rate kvroute {:.3} vs {} {:.3}",
                kv.prefix_hit_rate,
                b.policy,
                b.prefix_hit_rate
            );
            assert!(
                kv.ttft_p99_ms < 0.8 * b.ttft_p99_ms,
                "seed {seed}: p99 TTFT kvroute {:.1} vs {} {:.1}",
                kv.ttft_p99_ms,
                b.policy,
                b.ttft_p99_ms
            );
            assert!(kv.evictions < b.evictions);
        }
    }
}

#[test]
fn mooncake_fixture_parses_and_runs() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mooncake_sample.jsonl");
    let recs = load_mooncake(&path).unwrap();
    assert_eq!(recs.len(), 4);
    assert_eq!(recs[3].hash_ids, vec![46]);

    let reqs = to_requests(&recs, 1.0);
    assert_eq!(
        reqs.iter().map(|r| r.arrival_ms).collect::<Vec<_>>(),
        vec![0, 0, 511, 2518]
    );
    // Full blocks only: 6955 / 512 = 13, 6472 / 512 = 12, 1088 / 512 = 2.
    assert_eq!(
        reqs.iter()
            .map(|r| r.block_hashes.len())
            .collect::<Vec<_>>(),
        vec![13, 12, 2, 1]
    );
    // Shared hash_ids prefix -> shared chained hashes, until the ids diverge.
    assert_eq!(reqs[0].block_hashes[..12], reqs[1].block_hashes[..]);
    assert_eq!(reqs[0].block_hashes[0], reqs[3].block_hashes[0]);
    assert_ne!(reqs[0].block_hashes[0], reqs[2].block_hashes[0]);

    let s = run(&SimConfig::default(), PolicyKind::KvRoute, &reqs);
    assert_eq!(s.requests, 4);
    assert!(s.prefix_hit_rate > 0.0);
}
