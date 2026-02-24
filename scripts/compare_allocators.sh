#!/usr/bin/env bash
# Compare jemalloc vs system allocator under multi-threaded VM load.
# Run from repo root: scripts/compare_allocators.sh
# Output: table with T1, T8, T16, T16/T1 per mode; peak memory from time -l (macOS) or time -v (Linux).

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEST_NAME="test_allocator_comparison"

# Peak memory: macOS uses /usr/bin/time -l (max RSS in bytes); Linux uses /usr/bin/time -v (max RSS in kbytes).
# On macOS we write time's stderr to a separate file and append it (avoids any redirect quirks).
run_with_allocator() {
    local feature="$1"
    local outfile="$2"   # optional: where to write (Darwin only)
    local timefile="$3"  # optional: where time -l stderr goes (Darwin only)
    local cmd
    if [[ -n "$feature" ]]; then
        cargo clean
        cmd=(cargo test --release --features "$feature" -- "$TEST_NAME" --ignored --nocapture)
    else
        cargo clean
        cmd=(cargo test --release -- "$TEST_NAME" --ignored --nocapture)
    fi
    if [[ "$(uname)" == "Darwin" && -n "$outfile" && -n "$timefile" ]]; then
        /usr/bin/time -l "${cmd[@]}" 2>"$timefile" >"$outfile"
    elif [[ "$(uname)" == "Darwin" ]]; then
        /usr/bin/time -l "${cmd[@]}" 2>&1
    elif [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v "${cmd[@]}" 2>&1
    else
        "${cmd[@]}" 2>&1
    fi
}

# Format peak memory: input in kB, output e.g. "256kB" or "128MiB"
format_peak_kb() {
    local kb="$1"
    if [[ -z "$kb" ]]; then
        echo "N/A"
    elif [[ "$kb" -ge 1024 ]]; then
        echo "$(( kb / 1024 ))MiB"
    else
        echo "${kb}kB"
    fi
}

# Extract peak RSS from time output.
# If timefile is set (Darwin: raw time -l stderr), search there first; else search outfile (last 40 lines).
# Linux (time -v): "Maximum resident set size (kbytes): 12345"
# macOS (time -l): "   4317085696  maximum resident set size" (value in bytes)
extract_peak_mem() {
    local outfile="$1"
    local timefile="${2:-}"
    local searchfile="$outfile"
    if [[ -n "$timefile" && -r "$timefile" ]]; then
        searchfile="$timefile"
    elif [[ ! -r "$outfile" ]]; then
        echo "N/A"
        return
    fi
    local kb
    # Linux: value already in kbytes
    if kb=$(grep -E 'Maximum resident set size \(kbytes\):' "$searchfile" 2>/dev/null | sed -E 's/.*: *([0-9]+)/\1/' | head -1); then
        [[ -n "$kb" ]] && format_peak_kb "$kb" && return
    fi
    # macOS: line with "resident set size" and a number (bytes). Use awk for portability.
    local bytes
    bytes=$(awk '/resident.*set.*size/ { for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) { print $i; exit } }' "$searchfile" 2>/dev/null | head -1)
    if [[ -n "$bytes" && "$bytes" -gt 0 ]]; then
        kb=$(( bytes / 1024 ))
        format_peak_kb "$kb"
    else
        echo "N/A"
    fi
}

parse_alloc_bench() {
    local outfile="$1"
    grep '^ALLOC_BENCH ' "$outfile" 2>/dev/null | while read -r line; do
        local mode="" t1="" t8="" t16="" r8="" r16=""
        for pair in $line; do
            case "$pair" in
                Mode=*) mode="${pair#Mode=}" ;;
                T1=*)   t1="${pair#T1=}" ;;
                T8=*)   t8="${pair#T8=}" ;;
                T16=*)  t16="${pair#T16=}" ;;
                T8/T1=*) r8="${pair#T8/T1=}" ;;
                T16/T1=*) r16="${pair#T16/T1=}" ;;
            esac
        done
        printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$t1" "$t8" "$t16" "$r16"
    done
}

# Write each run to a temp file; on Darwin keep a separate time -l stderr file per run for reliable peak parsing
TMP_JEMALLOC=$(mktemp)
TMP_SYSTEM=$(mktemp)
TMP_TIME_JEMALLOC=$(mktemp)
TMP_TIME_SYSTEM=$(mktemp)
trap 'rm -f "$TMP_JEMALLOC" "$TMP_SYSTEM" "$TMP_TIME_JEMALLOC" "$TMP_TIME_SYSTEM"' EXIT

run_one() {
    local feature="$1"
    local outfile="$2"
    local timefile="$3"
    if [[ "$(uname)" == "Darwin" && -n "$timefile" ]]; then
        run_with_allocator "$feature" "$outfile" "$timefile"
        cat "$timefile" >> "$outfile"
    else
        run_with_allocator "$feature" > "$outfile"
    fi
}

echo "Building and running with allocator_jemalloc..."
run_one "allocator_jemalloc" "$TMP_JEMALLOC" "$TMP_TIME_JEMALLOC"
echo "Building and running with system allocator..."
run_one "" "$TMP_SYSTEM" "$TMP_TIME_SYSTEM"

if [[ -n "${DEBUG_ALLOC:-}" && "$(uname)" == "Darwin" ]]; then
    echo "--- DEBUG: time file size and content (jemalloc run) ---" >&2
    wc -c "$TMP_TIME_JEMALLOC" >&2
    echo "--- First 3 and last 15 lines ---" >&2
    head -3 "$TMP_TIME_JEMALLOC" >&2
    echo "..." >&2
    tail -15 "$TMP_TIME_JEMALLOC" >&2
    echo "--- End DEBUG ---" >&2
fi

# Collect parseable results: allocator, mode, t1, t8, t16, t16/t1, peak_mem (one peak per run)
# On Darwin pass the dedicated time file so we parse peak from raw time -l stderr
collect() {
    local alloc="$1"
    local outfile="$2"
    local timefile="${3:-}"
    local peak
    peak=$(extract_peak_mem "$outfile" "$timefile")
    parse_alloc_bench "$outfile" | while IFS=$'\t' read -r mode t1 t8 t16 r16; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$alloc" "$mode" "$t1" "$t8" "$t16" "$r16" "$peak"
    done
}

RESULTS=$(mktemp)
{
    if [[ "$(uname)" == "Darwin" ]]; then
        collect "jemalloc" "$TMP_JEMALLOC" "$TMP_TIME_JEMALLOC"
        collect "system"   "$TMP_SYSTEM"   "$TMP_TIME_SYSTEM"
    else
        collect "jemalloc" "$TMP_JEMALLOC"
        collect "system"   "$TMP_SYSTEM"
    fi
} > "$RESULTS"

echo ""
echo "| Allocator | Mode | T1 | T8 | T16 | T16/T1 | Peak Memory | Notes |"
echo "|-----------|------|-----|-----|-----|--------|-------------|-------|"
awk -F'\t' -v OFS=' | ' '
  { alloc=$1; mode=$2; t1=$3; t8=$4; t16=$5; r=$6; peak=$7
    printf "| %-8s | %-4s | %s | %s | %s | %-6s | %-11s |       |\n", alloc, mode, t1, t8, t16, r, peak
  }
' "$RESULTS"
rm -f "$RESULTS"

echo ""
echo "Interpretation: T16/T1 near 1 => best scaling; ratio > 3 => contention. Peak memory from /usr/bin/time -l (macOS) or time -v (Linux)."
