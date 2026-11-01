#!/usr/bin/env bash
# VM-микробенчмарки s01–s10 (этапы 24/25): чередующиеся прогоны двух бинарей
# DataCode (baseline vs candidate) + опционально Python, в одинаковых условиях.
#
# Usage (from repo root):
#   ./scripts/bench_vm/compare.sh [RUNS] [SCENARIO_GLOB]
#
# Env:
#   BASE=path   бинарь baseline (обязателен)
#   CAND=path   бинарь candidate (по умолчанию target/release/datacode)
#   PY=1        также гонять python3 (по умолчанию нет)
#
# Вывод (TSV): run scenario who real user sys max_rss_bytes instructions|RESULT-строка
# Метрики берутся из /usr/bin/time -l (macOS).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="${ROOT}/scripts/bench_vm"
RUNS="${1:-3}"
GLOB="${2:-s*}"
CAND="${CAND:-${ROOT}/target/release/datacode}"
: "${BASE:?set BASE=path/to/baseline/datacode}"

measure() { # who cmd file
  local who="$1" cmd="$2" f="$3" out
  out="$( { /usr/bin/time -l "$cmd" "$f" 2>&1 1>/tmp/.bench_vm_out.$$; } )"
  local real user sys rss ins
  real=$(echo "$out" | awk '/ real /{print $1}')
  user=$(echo "$out" | awk '/ real /{print $3}')
  sys=$(echo "$out" | awk '/ real /{print $5}')
  rss=$(echo "$out" | awk '/maximum resident set size/{print $1}')
  ins=$(echo "$out" | awk '/instructions retired/{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s|%s\n' "$r" "$(basename "${f%.*}")" "$who" \
    "$real" "$user" "$sys" "$rss" "$ins" "$(grep RESULT /tmp/.bench_vm_out.$$ | head -1)"
}

for r in $(seq 1 "$RUNS"); do
  for f in "$HERE"/$GLOB.dc; do
    [[ "$(basename "$f")" == s[0-9]* ]] || continue
    if (( r % 2 )); then measure base "$BASE" "$f"; measure cand "$CAND" "$f";
    else measure cand "$CAND" "$f"; measure base "$BASE" "$f"; fi
    if [[ "${PY:-0}" == 1 ]]; then measure py python3 "${f%.dc}.py"; fi
  done
done
rm -f /tmp/.bench_vm_out.$$
