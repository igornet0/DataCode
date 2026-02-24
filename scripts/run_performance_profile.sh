#!/usr/bin/env bash
# Run performance .dc scripts with profile feature and collect store_allocations, store_get_count, RSS, page faults.
# Usage: from repo root: scripts/run_performance_profile.sh [path_to_script.dc ...]
#        If no args: runs all tests/performance_tests/*.dc
# Build: cargo build --release --features profile (binary: target/release/datacode)

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BIN="${REPO_ROOT}/target/release/datacode"
DC_DIR="${REPO_ROOT}/tests/performance_tests"

# Build with profile if needed
if [[ ! -x "$BIN" ]] || ! "$BIN" --version &>/dev/null; then
    echo "Building datacode with profile feature..."
    cargo build --release --features profile -p data-code 2>&1
fi

# Ensure we have the profile binary (feature may be compiled in at build time)
cargo build --release --features profile -p data-code 2>&1

run_one() {
    local script="$1"
    local name
    name=$(basename "$script" .dc)
    local tmpout
    tmpout=$(mktemp)
    local tmperr
    tmperr=$(mktemp)
    if [[ "$(uname)" == "Darwin" && -x /usr/bin/time ]]; then
        # macOS: time -l prints maxrss (bytes), page reclaims/faults on stderr
        /usr/bin/time -l "$BIN" "$script" 2>"$tmperr" >"$tmpout" || true
    elif [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v "$BIN" "$script" 2>"$tmperr" >"$tmpout" || true
    else
        "$BIN" "$script" 2>"$tmperr" >"$tmpout" || true
    fi
    echo "========== $name =========="
    grep '\[profile\]' "$tmperr" 2>/dev/null | sed 's/\[profile\] /  /' || true
    echo "  --- time/resource (stderr) ---"
    grep -E 'maximum resident|Maximum resident|real|user|sys|page reclaims|page faults' "$tmperr" 2>/dev/null || true
    rm -f "$tmpout" "$tmperr"
}

# Collect scripts
if [[ $# -ge 1 ]]; then
    SCRIPTS=("$@")
else
    SCRIPTS=("$DC_DIR"/*.dc)
fi

echo "=== DataCode performance profile (store_allocations, store_get_count, RSS) ==="
for script in "${SCRIPTS[@]}"; do
    [[ -f "$script" ]] || continue
    run_one "$script"
    echo ""
done
echo "Done. For full stderr profile output (top by alloc/get), run: $BIN <script.dc> 2>&1 | grep '\[profile\]'"
