//! Admission control and fault-injection behaviour of the full simulator.

use kvroute_sim::engine::{run, run_detailed, RequestRecord, SimConfig};
use kvroute_sim::policy::PolicyKind;
use kvroute_sim::worker::WorkerConfig;
use kvroute_sim::workload::{to_requests, AgentWorkload, SimRequest};

fn agents(rps: f64, requests: usize, seed: u64) -> Vec<SimRequest> {
    let recs = AgentWorkload {
        sessions: 64,
        system_prompts: 8,
        rps,
        requests,
        seed,
    }
    .generate()
    .unwrap();
    to_requests(&recs, 1.0)
}

/// Hit rate over requests that started in `[from_ms, to_ms)`.
fn window_hit_rate(recs: &[RequestRecord], from_ms: u64, to_ms: u64) -> f64 {
    let (m, t) = recs
        .iter()
        .filter(|r| (from_ms * 1000..to_ms * 1000).contains(&r.start_us))
        .fold((0, 0), |(m, t), r| {
            (m + r.matched_blocks, t + r.total_blocks)
        });
    assert!(t > 0, "empty window [{from_ms}, {to_ms})");
    m as f64 / t as f64
}

#[test]
fn admission_tames_eviction_storm() {
    // 128-block workers at 2x overload: committed KV far exceeds capacity.
    let cfg = SimConfig {
        worker: WorkerConfig {
            kv_capacity_blocks: 128,
            ..WorkerConfig::default()
        },
        ..SimConfig::default()
    };
    for seed in [1, 2, 3] {
        let reqs = agents(28.0, 800, seed);
        let adm = run(&cfg, PolicyKind::KvRoute, &reqs);
        let open = run(&cfg, PolicyKind::KvRouteNoAdmission, &reqs);
        assert_eq!(adm.requests, reqs.len());
        assert!(adm.deferrals > 0 && open.deferrals == 0);
        assert!(
            adm.evictions < open.evictions,
            "seed {seed}: {adm:?} vs {open:?}"
        );
        assert!(
            adm.prefix_hit_rate > open.prefix_hit_rate + 0.02,
            "seed {seed}: {adm:?} vs {open:?}"
        );
        assert!(
            adm.ttft_p99_ms < open.ttft_p99_ms,
            "seed {seed}: {adm:?} vs {open:?}"
        );
    }
}

#[test]
fn staleness_and_drops_degrade_gracefully() {
    let reqs = agents(14.0, 1000, 1);
    let hit = |event_delay_ms, event_drop_prob| {
        let cfg = SimConfig {
            event_delay_ms,
            event_drop_prob,
            ..SimConfig::default()
        };
        let s = run(&cfg, PolicyKind::KvRoute, &reqs);
        assert_eq!(s.requests, reqs.len(), "every request must complete");
        s.prefix_hit_rate
    };
    let blind = run(&SimConfig::default(), PolicyKind::RoundRobin, &reqs).prefix_hit_rate;

    let by_delay: Vec<f64> = [20, 2_000, 10_000].map(|d| hit(d, 0.0)).to_vec();
    let by_drop: Vec<f64> = [0.0, 0.3, 0.7, 1.0].map(|p| hit(20, p)).to_vec();
    for w in by_delay.windows(2).chain(by_drop.windows(2)) {
        assert!(
            w[1] <= w[0] + 0.01,
            "hit rate should not improve with worse info: {by_delay:?} {by_drop:?}"
        );
    }
    assert!(by_delay[2] < by_delay[0] - 0.05, "{by_delay:?}");
    assert!(by_drop[2] < by_drop[0] - 0.05, "{by_drop:?}");
    // With every event lost the router is blind: no better than round-robin,
    // and not catastrophically worse.
    assert!(
        (by_drop[3] - blind).abs() < 0.05,
        "{by_drop:?} vs round-robin {blind}"
    );
}

#[test]
fn router_restart_dips_then_recovers() {
    let reqs = agents(14.0, 1500, 2);
    let t = 60_000;
    let resync = 5_000;
    let cfg = SimConfig {
        router_restart_at_ms: Some(t),
        resync_delay_ms: Some(resync),
        ..SimConfig::default()
    };
    let (s, recs) = run_detailed(&cfg, PolicyKind::KvRoute, &reqs);
    assert_eq!(s.requests, reqs.len());

    let before = window_hit_rate(&recs, t - 20_000, t);
    let cold = window_hit_rate(&recs, t, t + resync);
    let after = window_hit_rate(&recs, t + resync, t + resync + 20_000);
    println!("restart: before={before:.3} cold={cold:.3} after={after:.3}");
    assert!(
        cold < before - 0.15,
        "index wipe should hurt: {before} -> {cold}"
    );
    assert!(
        after > before - 0.05,
        "resync should restore hit rate: {before} -> {after}"
    );

    // Without a resync the index relearns only from new stores; resident
    // blocks stay invisible until evicted and recomputed, so recovery is
    // slower but still converges.
    let cfg = SimConfig {
        resync_delay_ms: None,
        ..cfg
    };
    let (_, recs) = run_detailed(&cfg, PolicyKind::KvRoute, &reqs);
    let early = window_hit_rate(&recs, t + resync, t + 2 * resync);
    let late = window_hit_rate(&recs, t + 20_000, t + 40_000);
    let early_resynced = {
        let (_, r) = run_detailed(
            &SimConfig {
                resync_delay_ms: Some(resync),
                ..cfg.clone()
            },
            PolicyKind::KvRoute,
            &reqs,
        );
        window_hit_rate(&r, t + resync, t + 2 * resync)
    };
    println!("no resync: early={early:.3} (resynced {early_resynced:.3}) late={late:.3}");
    assert!(early < early_resynced - 0.05, "{early} vs {early_resynced}");
    assert!(late > before - 0.05, "organic recovery: {before} -> {late}");
}
