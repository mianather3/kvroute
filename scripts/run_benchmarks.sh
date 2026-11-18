#!/usr/bin/env bash
# Reproducible kvroute-sim parameter sweeps. Writes raw CSVs to benchmarks/results/.
# Every run is seeded, so re-running yields byte-identical CSVs.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=benchmarks/results
SIM=target/release/kvroute-sim
SEED=42
REQS=2000
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cargo build --release -q -p kvroute-sim
mkdir -p "$OUT"

# run <out.csv> <param-header> <param-values> <sim args...>
# Prepends swept-parameter columns to the simulator's summary CSV rows.
run() {
    local out=$1 hdr=$2 vals=$3
    shift 3
    "$SIM" --seed "$SEED" --requests "$REQS" --csv "$TMP/run.csv" "$@" >/dev/null
    if [[ ! -s $out ]]; then
        echo "$hdr,$(head -1 "$TMP/run.csv")" >"$out"
    fi
    tail -n +2 "$TMP/run.csv" | sed "s/^/$vals,/" >>"$out"
}

echo "[1/4] load sweep"
f=$OUT/load_sweep.csv; rm -f "$f"
for rps in 4 8 12 16 20 24 28 32; do
    for p in round-robin consistent-hash least-loaded approx-ttl kvroute; do
        run "$f" rps "$rps" --rps "$rps" --policy "$p"
    done
done

echo "[2/4] staleness sweep"
f=$OUT/staleness_sweep.csv; rm -f "$f"
for drop in 0.0 0.05 0.10 0.25 0.50; do
    for delay in 0 10 25 50 100 250 500 1000; do
        run "$f" event_delay_ms,event_drop_prob "$delay,$drop" \
            --rps 16 --policy kvroute --event-delay-ms "$delay" --event-drop-prob "$drop"
    done
done
# Event-blind baseline for reference.
run "$f" event_delay_ms,event_drop_prob "0,0.0" --rps 16 --policy round-robin

echo "[3/4] admission ablation"
f=$OUT/admission_ablation.csv; rm -f "$f"
for rps in 12 16 20 24 28 32 36 40; do
    for p in kvroute kvroute-no-admission approx-ttl; do
        run "$f" rps "$rps" --kv-blocks 128 --rps "$rps" --policy "$p"
    done
done

echo "[4/4] restart recovery"
f=$OUT/restart_recovery.csv; rm -f "$f"
# windowed <variant> <sim args...>
windowed() {
    local variant=$1
    shift
    "$SIM" --seed "$SEED" --requests "$REQS" --rps 16 --window-ms 2500 \
        --windowed-csv "$TMP/w.csv" "$@" >/dev/null
    if [[ ! -s $f ]]; then
        echo "variant,$(head -1 "$TMP/w.csv")" >"$f"
    fi
    tail -n +2 "$TMP/w.csv" | sed "s/^/$variant,/" >>"$f"
}
windowed no-restart --policy kvroute
windowed resync-1000ms --policy kvroute --router-restart-ms 60000 --resync-delay-ms 1000
windowed no-resync --policy kvroute --router-restart-ms 60000 --no-resync
windowed approx-ttl --policy approx-ttl --router-restart-ms 60000

echo "wrote: $(ls "$OUT"/*.csv | tr '\n' ' ')"
