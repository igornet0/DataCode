#!/usr/bin/env bash
# Табличный бенчмарк DataCode vs Python+pandas на одинаковых данных.
#
# gen.py генерирует CSV для всех размеров ДО замеров; bench.dc и bench.py
# выполняют одни и те же сценарии (load, filter, nulls, newcol, groupby,
# sort_top, join, pipeline) и печатают проверочные поля. Скрипт сверяет их
# (целые — точно, float — с относительным допуском 1e-9, схему и число null —
# строкой) и падает при расхождении.
#
# Фазы: load (чтение CSV), compute (filter/nulls/newcol/groupby/sort_top/join
# на уже загруженных данных), pipeline (полный конвейер от чтения до агрегата).
# Для каждой фазы печатается медиана, разброс и отношение DC/pandas.
#
# Память: после замеров времени каждая сторона запускается ещё раз отдельно
# (BENCH_RUNS=1) под /usr/bin/time, берётся пиковый RSS процесса. Замеры
# времени этим прогоном не затрагиваются. MEM=0 отключает.
#
# Usage (from repo root):
#   ./scripts/bench_tabular/compare.sh [RUNS] [SIZES...]
#   ./scripts/bench_tabular/compare.sh 5 100000 1000000 3000000
#
# Env:
#   PYTHON=python3.13   интерпретатор с pandas/numpy (см. requirements.txt)
#   DC=path/to/datacode бинарь DataCode (по умолчанию target/release/datacode)
#   DATA_DIR=...        куда класть CSV (по умолчанию scripts/bench_tabular/data)
#   OUT_DIR=...         куда писать results.json/results.csv (по умолчанию scripts/bench_tabular/results)
#   MEM=0               не измерять пиковую память

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="${ROOT}/scripts/bench_tabular"
RUNS="${1:-5}"
shift || true
SIZES=("$@")
[[ ${#SIZES[@]} -eq 0 ]] && SIZES=(100000 1000000 3000000)
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-${HERE}/data}"
OUT_DIR="${OUT_DIR:-${HERE}/results}"
MEM="${MEM:-1}"

# Пиковый RSS (байты) отдельного прогона; stdout команды отбрасывается.
peak_rss() {
  local log
  log="$(mktemp)"
  if [[ "$(uname)" == "Darwin" ]]; then
    /usr/bin/time -l "$@" >/dev/null 2>"$log"
    awk '/maximum resident set size/ {print $1}' "$log"
  else
    /usr/bin/time -v "$@" >/dev/null 2>"$log"
    awk -F: '/Maximum resident set size/ {gsub(/ /, "", $2); print $2 * 1024}' "$log"
  fi
  rm -f "$log"
}

if [[ -z "${DC:-}" ]]; then
  if [[ -x "${ROOT}/target/release/datacode" ]]; then
    DC="${ROOT}/target/release/datacode"
  else
    echo "Building datacode..."
    (cd "$ROOT" && cargo build --release --bin datacode)
    DC="${ROOT}/target/release/datacode"
  fi
fi

echo "DataCode: $DC"
echo "Python:   $("$PYTHON" -c 'import sys, pandas, numpy; print(sys.version.split()[0], "pandas", pandas.__version__, "numpy", numpy.__version__)')"
echo "Runs:     $RUNS"
echo "Sizes:    ${SIZES[*]}"
echo

# Данные — до любых замеров; существующие файлы переиспользуются.
for rows in "${SIZES[@]}"; do
  d="${DATA_DIR}/${rows}"
  if [[ ! -f "${d}/fact.csv" ]]; then
    echo "generating ${rows} rows -> ${d}"
    "$PYTHON" "${HERE}/gen.py" "$rows" "$d"
  fi
done

OUT=""
for rows in "${SIZES[@]}"; do
  export BENCH_DIR="${DATA_DIR}/${rows}" BENCH_RUNS="$RUNS"
  echo "== ${rows} rows"
  dc_out="$("$DC" --no-gui "${HERE}/bench.dc" | grep '^RESULT')"
  py_out="$("$PYTHON" "${HERE}/bench.py" | grep '^RESULT')"
  while IFS= read -r l; do OUT+="${rows} ${l}"$'\n'; done <<<"$dc_out"$'\n'"$py_out"
  if [[ "$MEM" != "0" ]]; then
    dc_rss="$(BENCH_RUNS=1 peak_rss "$DC" --no-gui "${HERE}/bench.dc")"
    py_rss="$(BENCH_RUNS=1 peak_rss "$PYTHON" "${HERE}/bench.py")"
    OUT+="${rows} MEM dc=${dc_rss} py=${py_rss}"$'\n'
  fi
done

mkdir -p "$OUT_DIR"
echo
"$PYTHON" - "$OUT" "$OUT_DIR" "$RUNS" <<'PY'
import csv, json, math, platform, sys

lines, out_dir, runs = sys.argv[1], sys.argv[2], int(sys.argv[3])
TIMING = {"lang", "scenario", "runs", "median", "min", "max"}

PHASE = {"load": "load", "pipeline": "pipeline"}
rows, mem = {}, {}
for line in lines.strip().splitlines():
    size, tag, *fields = line.split()
    if tag == "MEM":
        kv = dict(f.split("=", 1) for f in fields)
        mem[int(size)] = {k: int(v) for k, v in kv.items() if v}
        continue
    kv = dict(f.split("=", 1) for f in fields)
    rows.setdefault((int(size), kv["scenario"]), {})[kv["lang"]] = kv


def same(a, b):
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6)


records, failed = [], False
def spread(kv):
    return (float(kv["max"]) - float(kv["min"])) / float(kv["median"]) * 100


print(f"{'rows':>9} {'phase':<9}{'scenario':<10}{'DC median':>11}{'pandas':>11}{'DC/pd':>8}  DC [min..max] ±% / pandas [min..max] ±%")
for (size, scen), langs in sorted(rows.items()):
    dc, py = langs["dc"], langs["py"]
    checks = sorted(k for k in dc if k not in TIMING)
    bad = [k for k in checks if not same(dc.get(k), py.get(k))]
    if bad or set(checks) != {k for k in py if k not in TIMING}:
        failed = True
        print(f"{size:>9} {'':<9}{scen:<10} MISMATCH {[(k, dc.get(k), py.get(k)) for k in bad]}")
    d, p = float(dc["median"]), float(py["median"])
    phase = PHASE.get(scen, "compute")
    print(
        f"{size:>9} {phase:<9}{scen:<10}{d:>10.4f}s{p:>10.4f}s{d / p:>8.2f}  "
        f"[{float(dc['min']):.4f}..{float(dc['max']):.4f}] {spread(dc):.0f}% / "
        f"[{float(py['min']):.4f}..{float(py['max']):.4f}] {spread(py):.0f}%"
    )
    records.append({
        "rows": size, "phase": phase, "scenario": scen, "runs": runs,
        "dc_median": d, "dc_min": float(dc["min"]), "dc_max": float(dc["max"]),
        "pd_median": p, "pd_min": float(py["min"]), "pd_max": float(py["max"]),
        "ratio": d / p, "validated": not bad,
        "checks": {k: dc[k] for k in checks},
    })

# Сводка по фазам: сумма медиан сценариев фазы.
print(f"\n{'rows':>9} {'phase':<9}{'DC':>11}{'pandas':>11}{'DC/pd':>8}{'DC RSS':>10}{'pd RSS':>10}")
phases = {}
for r in records:
    k = (r["rows"], r["phase"])
    a = phases.setdefault(k, [0.0, 0.0])
    a[0] += r["dc_median"]
    a[1] += r["pd_median"]
summary = []
for (size, phase), (d, p) in sorted(phases.items(), key=lambda x: (x[0][0], ["load", "compute", "pipeline"].index(x[0][1]))):
    m = mem.get(size, {})
    mb = lambda b: f"{b / 2**20:.0f}MB" if b else "-"
    rss = phase == "pipeline"  # память — одна строка на размер, печатаем у последней фазы
    print(f"{size:>9} {phase:<9}{d:>10.4f}s{p:>10.4f}s{d / p:>8.2f}"
          f"{mb(m.get('dc')) if rss else '':>10}{mb(m.get('py')) if rss else '':>10}")
    summary.append({"rows": size, "phase": phase, "dc_median_sum": d, "pd_median_sum": p, "ratio": d / p})

meta = {"platform": platform.platform(), "machine": platform.machine(), "runs": runs}
with open(f"{out_dir}/results.json", "w") as f:
    json.dump({"meta": meta, "results": records, "phases": summary,
               "peak_rss_bytes": {str(k): v for k, v in mem.items()}}, f, indent=2, ensure_ascii=False)
with open(f"{out_dir}/results.csv", "w", newline="") as f:
    cols = ["rows", "phase", "scenario", "runs", "dc_median", "dc_min", "dc_max",
            "pd_median", "pd_min", "pd_max", "ratio", "validated"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(records)
print(f"\nwrote {out_dir}/results.json, {out_dir}/results.csv")
sys.exit(1 if failed else 0)
PY
