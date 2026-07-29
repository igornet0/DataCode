#!/usr/bin/env bash
# Parallel A* benchmark — one independent datacode process per run.
#
# A* 1000×5000 is memory-heavy (~1–3 GiB RSS/process). Too many workers causes
# CPU/RAM contention (per-run time can jump from ~1.4 s to ~5+ s). Default workers
# are capped at 4; override with DC_ASTAR_WORKERS or the second argument.
#
# Usage (from repo root):
#   ./scripts/bench_astar_parallel.sh [RUNS] [WORKERS]
#
# Examples:
#   ./scripts/bench_astar_parallel.sh 10          # auto workers (≤4)
#   ./scripts/bench_astar_parallel.sh 10 2        # conservative (laptop / 16 GiB RAM)
#   DC_ASTAR_WORKERS=6 ./scripts/bench_astar_parallel.sh 12
#
# Build tip (less allocator contention under load):
#   cargo build --release --features allocator_jemalloc --bin datacode

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${ROOT}/examples/ru/04-продвинутые/алгоритмы/графы/advanced/a_start_adv_2_single.dc"
RUNS="${1:-10}"

CORES="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
# Default: at most 4 parallel A* jobs (RAM-bound), never more than cores or runs.
DEFAULT_WORKERS="$CORES"
if (( DEFAULT_WORKERS > 4 )); then DEFAULT_WORKERS=4; fi
if (( DEFAULT_WORKERS > RUNS )); then DEFAULT_WORKERS=RUNS; fi
WORKERS="${2:-${DC_ASTAR_WORKERS:-$DEFAULT_WORKERS}}"
if (( WORKERS > RUNS )); then WORKERS=RUNS; fi
if (( WORKERS < 1 )); then WORKERS=1; fi

if command -v datacode >/dev/null 2>&1; then
  DC="datacode"
elif [[ -x "${ROOT}/target/release/datacode" ]]; then
  DC="${ROOT}/target/release/datacode"
else
  echo "Building datacode..."
  (cd "$ROOT" && CARGO_TARGET_DIR=target cargo build --release --bin datacode)
  DC="${ROOT}/target/release/datacode"
fi

echo "Parallel A* benchmark: runs=${RUNS} workers=${WORKERS} (cores=${CORES})"
echo "Script: ${SCRIPT}"
echo "Binary: ${DC}"
echo "Hint: if mean >> min, lower WORKERS (RAM/CPU contention)."
echo

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

run_one() {
  # glibc: fewer per-thread arenas → lower RSS when many processes run at once
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-1}"
  "$DC" --no-gui "$SCRIPT" 2>/dev/null | grep -E '^(ELAPSED|PATH_LEN):'
}

export -f run_one
export DC SCRIPT

echo "=== sequential baseline (1 run, same machine state) ==="
BASELINE="$(run_one | grep '^ELAPSED:' | cut -d: -f2)"
echo "ELAPSED:${BASELINE}"
echo

TIMING_FILE="${TMP}/wall_start.txt"
python3 -c 'import time; open("'"${TIMING_FILE}"'", "w").write(str(time.perf_counter()))'

pids=()
for _ in $(seq 1 "$RUNS"); do
  while ((${#pids[@]} >= WORKERS)); do
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  done
  run_one >> "${TMP}/results.txt" &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

WALL="$(python3 -c 'import time; s=float(open("'"${TIMING_FILE}"'").read()); print(f"{time.perf_counter()-s:.4f}")')"

echo "=== per-run (completion order) ==="
cat "${TMP}/results.txt"
echo

python3 - "${TMP}/results.txt" "$BASELINE" "$WALL" "$RUNS" "$WORKERS" <<'PY'
import sys
from statistics import median

path, baseline_s, wall_s, runs_s, workers_s = sys.argv[1:6]
runs = int(runs_s)
workers = int(workers_s)
baseline = float(baseline_s)
wall = float(wall_s)

times = []
with open(path) as f:
    for line in f:
        if line.startswith("ELAPSED:"):
            times.append(float(line.split(":", 1)[1]))
if not times:
    sys.exit("no ELAPSED lines")

times_sorted = sorted(times)
mean = sum(times) / len(times)
med = median(times)
tmin = min(times)
tmax = max(times)
seq_sum = sum(times)
speedup = seq_sum / wall if wall > 0 else 0.0
contention = mean / tmin if tmin > 0 else 1.0

print(f"=== summary ({len(times)} runs, {workers} workers) ===")
print(f"sequential baseline:     {baseline:.4f} s")
print(f"parallel wall clock:     {wall:.4f} s")
print(f"sum of per-run times:    {seq_sum:.4f} s")
print(f"parallel speedup:        {speedup:.2f}x  (sum/wall; ideal≈min(workers,runs))")
print(f"mean per-run:            {mean:.4f} s")
print(f"median per-run:          {med:.4f} s")
print(f"min / max per-run:       {tmin:.4f} / {tmax:.4f} s")
print(f"contention ratio mean/min: {contention:.2f}x")

if contention > 1.35:
    print()
    print("⚠ Contention detected: per-run times spread widely.")
    print(f"  Try: ./scripts/bench_astar_parallel.sh {runs} {max(1, workers // 2)}")
    print("  Or build with: cargo build --release --features allocator_jemalloc --bin datacode")

ideal_wall = seq_sum / min(workers, runs)
if wall > ideal_wall * 1.25:
    print(f"  Wall clock {wall:.2f}s vs ideal ~{ideal_wall:.2f}s — reduce WORKERS or add RAM.")
PY
