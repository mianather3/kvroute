#!/usr/bin/env python3
"""Render kvroute benchmark figures from benchmarks/results/*.csv.

Pure stdlib csv + matplotlib. Every annotated number (capacity line,
admission crossover, restart dip) is computed from the CSVs, never hard-coded.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
PLOTS = ROOT / "benchmarks" / "plots"

# One fixed style per policy across every figure.
STYLE = {
    "round-robin": dict(color="#8c8c8c", marker="o", label="round-robin"),
    "least-loaded": dict(color="#b5895a", marker="s", label="least-loaded"),
    "consistent-hash": dict(color="#8e6bbf", marker="D", label="consistent-hash"),
    "approx-ttl": dict(color="#e08a2c", marker="^", label="approx-ttl"),
    "kvroute": dict(color="#1f5fbf", marker="o", label="kvroute", lw=2.4, zorder=5),
    "kvroute-no-admission": dict(
        color="#6fa8e8", marker="v", ls="--", label="kvroute (no admission)", lw=2.0
    ),
}

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
    }
)

SECONDS = FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}s" if v >= 1000 else f"{v:g}ms")


def read(name):
    path = RESULTS / name
    if not path.exists():
        sys.exit(f"missing {path}; run scripts/run_benchmarks.sh first")
    with path.open() as f:
        return list(csv.DictReader(f))


def series(rows, key, x, y, **match):
    """{key value: ([x...], [y...])} for rows matching `match`, sorted by x."""
    out = defaultdict(list)
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            out[r[key]].append((float(r[x]), float(r[y])))
    return {k: tuple(map(list, zip(*sorted(v)))) for k, v in out.items()}


def plot_policy(ax, xs, ys, policy, **kw):
    style = dict(STYLE[policy])
    style.update(kw)
    ax.plot(xs, ys, **style)


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {PLOTS / name}")


def crossover(xs, a, b):
    """Interpolated x where a - b changes sign from >= 0 to < 0, or None."""
    d = [p - q for p, q in zip(a, b)]
    for i in range(1, len(d)):
        if d[i - 1] >= 0 > d[i]:
            return xs[i - 1] + (xs[i] - xs[i - 1]) * d[i - 1] / (d[i - 1] - d[i])
    return None


def load_figure():
    rows = read("load_sweep.csv")
    hit = series(rows, "policy", "rps", "prefix_hit_rate")
    p99 = series(rows, "policy", "rps", "ttft_p99_ms")
    capacity = max(float(r["throughput_rps"]) for r in rows)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for p in ["round-robin", "least-loaded", "consistent-hash", "approx-ttl", "kvroute"]:
        plot_policy(a, *hit[p], p)
        plot_policy(b, *p99[p], p)
    for ax in (a, b):
        ax.axvline(capacity, color="#c0392b", ls=":", lw=1.2)
        ax.set_xlabel("offered load (requests/s)")
    a.text(capacity + 0.4, 0.93, f"measured capacity\n≈ {capacity:.1f} rps", color="#c0392b", fontsize=8, va="top")
    a.set_ylim(0, 1)
    a.set_ylabel("prefix hit rate")
    a.set_title("(a) Prefix hit rate vs. load")
    b.set_yscale("log")
    b.yaxis.set_major_formatter(SECONDS)
    b.set_ylabel("p99 TTFT")
    b.set_title("(b) p99 time-to-first-token vs. load")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Multi-turn agent workload · 8 workers × 512 KV blocks · 2000 requests · seed 42", fontsize=9, y=1.0)
    save(fig, "ttft_and_hitrate_vs_load.png")


def staleness_figure():
    rows = read("staleness_sweep.csv")
    kv = [r for r in rows if r["policy"] == "kvroute"]
    rr = next(r for r in rows if r["policy"] == "round-robin")
    hit = series(kv, "event_drop_prob", "event_delay_ms", "prefix_hit_rate")
    p99 = series(kv, "event_drop_prob", "event_delay_ms", "ttft_p99_ms")

    drops = sorted(hit, key=float)
    delays = hit[drops[0]][0]
    pos = {d: i for i, d in enumerate(delays)}
    cmap = plt.get_cmap("Blues")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, d in enumerate(drops):
        color = cmap(0.4 + 0.6 * (1 - i / max(1, len(drops) - 1)))
        label = f"drop {float(d):.0%}"
        a.plot([pos[x] for x in hit[d][0]], hit[d][1], marker="o", color=color, label=label)
        b.plot([pos[x] for x in p99[d][0]], p99[d][1], marker="o", color=color, label=label)
    a.axhline(float(rr["prefix_hit_rate"]), color=STYLE["round-robin"]["color"], ls="--", lw=1.2)
    a.text(0, float(rr["prefix_hit_rate"]) + 0.012, "round-robin (event-blind)", color="#6b6b6b", fontsize=8)
    b.axhline(float(rr["ttft_p99_ms"]), color=STYLE["round-robin"]["color"], ls="--", lw=1.2)
    b.text(0, float(rr["ttft_p99_ms"]) * 0.96, "round-robin (event-blind)", color="#6b6b6b", fontsize=8, va="top")
    for ax in (a, b):
        ax.set_xticks(range(len(delays)), [f"{d:g}" for d in delays])
        ax.set_xlabel("KV-event propagation delay (ms)")
    a.set_ylim(0.3, 0.85)
    a.set_ylabel("prefix hit rate")
    a.set_title("(a) Hit rate under stale / dropped KV events")
    b.set_ylim(0, None)
    b.yaxis.set_major_formatter(SECONDS)
    b.set_ylabel("p99 TTFT")
    b.set_title("(b) p99 TTFT under stale / dropped KV events")
    a.legend(loc="lower left", ncol=3, fontsize=8)
    fig.suptitle("kvroute at 16 rps · 8 workers × 512 KV blocks · seed 42", fontsize=9, y=1.0)
    save(fig, "staleness_degradation.png")


def admission_figure():
    rows = read("admission_ablation.csv")
    ev = series(rows, "policy", "rps", "evictions")
    p99 = series(rows, "policy", "rps", "ttft_p99_ms")
    xs = ev["kvroute"][0]
    x_ev = crossover(xs, ev["kvroute"][1], ev["kvroute-no-admission"][1])
    x_p99 = crossover(xs, p99["kvroute"][1], p99["kvroute-no-admission"][1])

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, data, x in ((a, ev, x_ev), (b, p99, x_p99)):
        plot_policy(ax, *data["kvroute"], "kvroute")
        plot_policy(ax, *data["kvroute-no-admission"], "kvroute-no-admission")
        plot_policy(ax, *data["approx-ttl"], "approx-ttl", alpha=0.45, lw=1.2)
        if x is not None:
            ax.axvspan(min(xs), x, color="#c0392b", alpha=0.07, lw=0)
            ax.axvspan(x, max(xs), color="#27ae60", alpha=0.07, lw=0)
            ax.axvline(x, color="#444", ls=":", lw=1)
            ax.annotate(f"crossover ≈ {x:.1f} rps", (x, 0.97), xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points", fontsize=8, va="top")
        ax.set_xlabel("offered load (requests/s)")
    a.set_ylabel("LRU block evictions")
    a.set_title("(a) Evictions: kvroute with vs. without admission")
    b.set_ylabel("p99 TTFT")
    b.yaxis.set_major_formatter(SECONDS)
    b.set_title("(b) p99 TTFT: kvroute with vs. without admission")
    regions = [
        Patch(color="#c0392b", alpha=0.15, label="admission hurts: diverts to cold workers"),
        Patch(color="#27ae60", alpha=0.15, label="admission helps: caps committed KV"),
    ]
    handles, _ = b.get_legend_handles_labels()
    fig.legend(handles + regions, [h.get_label() for h in handles + regions], loc="lower center",
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Tight KV budget · 8 workers × 128 KV blocks · max_kv_util 0.90 · 2000 requests · seed 42", fontsize=9, y=1.0)
    save(fig, "admission_ablation.png")


def restart_figure():
    rows = [r for r in read("restart_recovery.csv") if int(r["requests"]) >= 10]
    hit = series(rows, "variant", "window_start_ms", "hit_rate")
    style = {
        "no-restart": dict(color="#b0b0b0", ls="--", lw=1.2, label="no restart (reference)"),
        "resync-1000ms": dict(color=STYLE["kvroute"]["color"], lw=2.2, marker="o", label="kvroute + worker resync (1 s)"),
        "no-resync": dict(color="#c0392b", lw=1.8, marker="s", label="kvroute, no resync (organic relearn)"),
        "approx-ttl": dict(color=STYLE["approx-ttl"]["color"], lw=1.6, marker="^", label="approx-ttl"),
    }
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for v, s in style.items():
        xs, ys = hit[v]
        ax.plot([x / 1000 + 1.25 for x in xs], ys, markersize=3.5, **s)  # plot at window centers
    ax.axvline(60, color="#444", ls=":", lw=1.2)
    ax.annotate("router restart\n(index wiped)", (60, 0.2), xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax.axvspan(60, 61, color=STYLE["kvroute"]["color"], alpha=0.12, lw=0)
    ax.set_xlim(30, 105)
    ax.set_ylim(0, 1)
    ax.set_xlabel("simulated time (s), 2.5 s windows")
    ax.set_ylabel("prefix hit rate")
    ax.set_title("Hit-rate dip and recovery across a mid-run router restart")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("16 rps · 8 workers × 512 KV blocks · restart at t = 60 s · seed 42", fontsize=9, y=1.0)
    save(fig, "restart_recovery.png")


if __name__ == "__main__":
    load_figure()
    staleness_figure()
    admission_figure()
    restart_figure()
