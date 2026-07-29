#!/usr/bin/env bash
# Run all .dc examples in this folder via datacode.
#
# Usage:
#   ./run_all.sh              — all examples, final failure report
#   ./run_all.sh --stop       — stop on first failure (exit or assert)
#   ./run_all.sh --quiet      — no script output, status and summary only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STOP_ON_FIRST=false
QUIET=false
for arg in "$@"; do
  case "$arg" in
    --stop|--stop-on-first) STOP_ON_FIRST=true ;;
    --quiet|-q) QUIET=true ;;
    -h|--help)
      echo "Usage: $0 [--stop] [--quiet]"
      echo "  --stop   stop on first failure (exit code or «Test: FAIL»)"
      echo "  --quiet  do not print .dc output, only line status and summary"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (see $0 --help)" >&2
      exit 2
      ;;
  esac
done

if [[ -x "${ROOT}/../../../target/release/datacode" ]]; then
  DC=("${ROOT}/../../../target/release/datacode")
elif [[ -x "${ROOT}/../../../target/debug/datacode" ]]; then
  DC=("${ROOT}/../../../target/debug/datacode")
elif command -v datacode &>/dev/null; then
  DC=(datacode)
else
  echo "datacode not found in PATH or target/{debug,release}/" >&2
  exit 127
fi

mapfile -t SCRIPTS < <(
  find "$ROOT" -type f -name '*.dc' ! -name '_*' | LC_ALL=C sort
)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
  echo "No .dc files in $ROOT" >&2
  exit 1
fi

echo "Running ${#SCRIPTS[@]} examples (${DC[*]})"
echo "Directory: $ROOT"
echo "---"

ok=0
exit_failures=()
assert_failures=()

run_one() {
  local script=$1
  local rel=$2
  local idx=$3
  local total=$4
  local tmp
  tmp=$(mktemp)

  printf '[%d/%d] %s ... ' "$idx" "$total" "$rel"

  set +e
  if $QUIET; then
    "${DC[@]}" "$script" >"$tmp" 2>&1
  else
    "${DC[@]}" "$script" 2>&1 | tee "$tmp"
  fi
  local code=${PIPESTATUS[0]}
  set -e

  local out
  out=$(<"$tmp")
  rm -f "$tmp"

  local assert_fail=false
  if echo "$out" | grep -q 'Test: FAIL'; then
    assert_fail=true
  fi

  if [[ $code -ne 0 ]]; then
    echo "FAIL (exit $code)"
    exit_failures+=("$rel (exit $code)")
    if $QUIET; then
      echo "$out" | sed 's/^/  /'
    fi
    echo "$out" | grep -E 'Test:|FAIL|Error|error\[' | sed 's/^/  /' || true
    return 1
  fi

  if $assert_fail; then
    echo "FAIL (assert)"
    assert_failures+=("$rel")
    echo "$out" | grep -E 'Test:|FAIL' | sed 's/^/  /' || true
    return 2
  fi

  echo OK
  return 0
}

i=0
for script in "${SCRIPTS[@]}"; do
  i=$((i + 1))
  rel="${script#"$ROOT"/}"

  if run_one "$script" "$rel" "$i" "${#SCRIPTS[@]}"; then
    ok=$((ok + 1))
  else
    rc=$?
    if $STOP_ON_FIRST; then
      echo "---"
      if [[ $rc -eq 1 ]]; then
        echo "Stopped: first failure (exit) — $rel" >&2
      else
        echo "Stopped: first failure (assert) — $rel" >&2
      fi
      exit "$rc"
    fi
  fi
done

total=${#SCRIPTS[@]}
failed=$((total - ok))

echo "---"
if [[ $failed -eq 0 ]]; then
  echo "All $total examples passed."
  exit 0
fi

echo "Total: $total examples, $ok OK, $failed failures"

if [[ ${#exit_failures[@]} -gt 0 ]]; then
  echo ""
  echo "Failures (exit code):"
  for f in "${exit_failures[@]}"; do
    echo "  - $f"
  done
fi

if [[ ${#assert_failures[@]} -gt 0 ]]; then
  echo ""
  echo "Failures (Test: FAIL):"
  for f in "${assert_failures[@]}"; do
    echo "  - $f"
  done
fi

exit 1
