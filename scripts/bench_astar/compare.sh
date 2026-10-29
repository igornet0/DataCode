#!/usr/bin/env bash
# A* DataCode vs Python на одинаковых входных данных (issue #22, этап 5).
#
# astar.dc и astar.py — один и тот же алгоритм; препятствия генерируются общим
# LCG, поэтому обе стороны ищут путь на одной и той же сетке (blocked и
# path_len в выводе должны совпасть — иначе скрипт падает).
#
# Usage (from repo root):
#   ./scripts/bench_astar/compare.sh [RUNS]
#
# Env:
#   PYTHON=python3.13   интерпретатор Python (по умолчанию python3)
#   DC=path/to/datacode бинарь DataCode (по умолчанию target/release/datacode)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="${ROOT}/scripts/bench_astar"
RUNS="${1:-10}"
PYTHON="${PYTHON:-python3}"

if [[ -z "${DC:-}" ]]; then
  if [[ -x "${ROOT}/target/release/datacode" ]]; then
    DC="${ROOT}/target/release/datacode"
  elif command -v datacode >/dev/null 2>&1; then
    DC="datacode"
  else
    echo "Building datacode..."
    (cd "$ROOT" && cargo build --release --bin datacode)
    DC="${ROOT}/target/release/datacode"
  fi
fi

# name rows cols blocked goal_r goal_c
CASES=(
  "small 100 500 1000 50 234"
  "large 1000 5000 10000 559 1234"
)

echo "DataCode: $DC"
echo "Python:   $("$PYTHON" --version 2>&1)"
echo "Runs:     $RUNS"
echo

OUT=""
for case in "${CASES[@]}"; do
  read -r name rows cols blocked gr gc <<<"$case"
  export ASTAR_ROWS="$rows" ASTAR_COLS="$cols" ASTAR_BLOCKED="$blocked" \
         ASTAR_GOAL_R="$gr" ASTAR_GOAL_C="$gc" ASTAR_RUNS="$RUNS"
  echo "== ${name}: ${rows}x${cols}, ${blocked} blocks, goal (${gr},${gc})"
  dc_line="$("$DC" --no-gui "${HERE}/astar.dc" | grep '^RESULT')"
  py_line="$("$PYTHON" "${HERE}/astar.py" | grep '^RESULT')"
  echo "  $dc_line"
  echo "  $py_line"
  OUT+="${name} ${rows}x${cols} ${dc_line}"$'\n'"${name} ${rows}x${cols} ${py_line}"$'\n'
done

echo
"$PYTHON" - "$OUT" <<'PY'
import sys

rows = {}
for line in sys.argv[1].strip().splitlines():
    name, size, _, *fields = line.split()
    kv = dict(f.split("=", 1) for f in fields)
    rows.setdefault((name, size), {})[kv["lang"]] = kv

print(f"{'case':<16}{'DC median':>11}{'Py median':>11}{'DC/Py':>8}")
failed = False
for (name, size), langs in rows.items():
    dc, py = langs["dc"], langs["py"]
    if (dc["blocked"], dc["path_len"]) != (py["blocked"], py["path_len"]):
        print(f"{name}: inputs/results differ: dc={dc} py={py}")
        failed = True
        continue
    d, p = float(dc["median"]), float(py["median"])
    print(f"{name + ' ' + size:<16}{d:>10.4f}s{p:>10.4f}s{d / p:>8.2f}")
sys.exit(1 if failed else 0)
PY
