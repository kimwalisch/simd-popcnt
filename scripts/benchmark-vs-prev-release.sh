#!/bin/bash

# Benchmark the latest code vs. the previous release.
#
# Usage:
#   ./benchmark-vs-prev-release.sh
#
# Description:
#   Builds the `benchmark` example (release mode) from the current code and from
#   the previous release (latest git tag of the form vX.Y.Z) and compares their
#   throughput (GB/s) at several array sizes. The new code must be faster than,
#   or within 2% of, the previous release. Each size is retried until it passes
#   or fails 3 times in a row, to tolerate the noise of shared CI runners.
#
#   Uses only POSIX tools (awk) for the floating-point math, so it runs on
#   Linux, macOS and Windows (Git Bash) without extra dependencies.

# Exit if any error occurs.
set -e

# Run from the repository root (this script lives in scripts/).
cd "$(dirname "$0")/.."

# Array sizes (in bytes) to benchmark.
SIZES="23 101 1001 10001 16384"

# The new code must reach at least this fraction of the old code's throughput
# (0.98 => at most 2% slower).
THRESHOLD=0.98

# Number of runs per binary to take the best (maximum) throughput of. Benchmark
# noise is one-sided — interference only slows a run down — so the fastest run is
# the most accurate estimate of the true throughput.
BEST_OF=5

# awk helpers ("1"/"0" for the comparisons, formatted percentage).
gt()      { awk -v a="$1" -v b="$2" 'BEGIN { print (a  > b)     ? 1 : 0 }'; }
ge()      { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN { print (a >= b*t) ? 1 : 0 }'; }
percent() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f", 100 * a / b }'; }

# Resolve the example binary in the given directory (benchmark[.exe]).
resolve_bin() {
    if [ -f "$1/benchmark.exe" ]; then echo "$1/benchmark.exe"; else echo "$1/benchmark"; fi
}

# Run the `benchmark` example once at the given size (default iteration count)
# and print the measured throughput in GB/s.
one_run() {
    "$1" "$2" 2>/dev/null | grep 'GB/s' | awk '{ print $1 }'
}

# Benchmark one array size, retrying up to 3 times before declaring failure.
benchmark_size() {
    local size=$1

    for attempt in 1 2 3
    do
        echo ""
        echo "=== Benchmark array size: $size bytes (attempt $attempt/3) ==="
        echo ""

        # Best-of-N for each binary, alternating which runs first on each round
        # so neither benchmark is systematically favoured by warm-up / frequency
        # effects.
        local best_old=0 best_new=0 old new round
        for round in $(seq 1 "$BEST_OF")
        do
            if [ $((round % 2)) -eq 1 ]
            then
                old=$(one_run "$PREV_BIN" "$size"); new=$(one_run "$CURR_BIN" "$size")
            else
                new=$(one_run "$CURR_BIN" "$size"); old=$(one_run "$PREV_BIN" "$size")
            fi

            if [ -z "$old" ] || [ -z "$new" ]
            then
                echo "Error: failed to capture GB/s output."
                exit 1
            fi

            if [ "$(gt "$old" "$best_old")" -eq 1 ]; then best_old=$old; fi
            if [ "$(gt "$new" "$best_new")" -eq 1 ]; then best_new=$new; fi
        done

        echo "Old code (${PREV_TAG}): $best_old GB/s (100.0%)"
        echo "New code:            $best_new GB/s ($(percent "$best_new" "$best_old")%)"

        if [ "$(ge "$best_new" "$best_old" "$THRESHOLD")" -eq 1 ]
        then
            echo "Array size $size bytes: performance test passed!"
            SUMMARY="${SUMMARY}$(printf '%10s %12s %12s %9s' \
                "$size" "$best_old" "$best_new" "$(percent "$best_new" "$best_old")%")
"
            return
        fi

        echo "Array size $size bytes: attempt $attempt failed."
    done

    echo ""
    echo "Error: new code is more than 2% slower than $PREV_TAG for array size $size bytes!"
    exit 1
}

# Previous release: the latest tag of the form vX.Y.Z.
PREV_TAG=$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -1)
if [ -z "$PREV_TAG" ]
then
    echo "No previous release tag (vX.Y.Z) found; skipping benchmark."
    exit 0
fi
echo "Previous release: $PREV_TAG"

# Build the current code (release mode).
cargo build --release --example benchmark
CURR_BIN=$(resolve_bin "$(pwd)/target/release/examples")

# Build the previous release in a detached worktree so the current working tree
# (including this running script) is left untouched.
WORKTREE=$(mktemp -d)
trap 'git worktree remove --force "$WORKTREE" 2>/dev/null; rm -rf "$WORKTREE"' EXIT
git worktree add --detach "$WORKTREE" "$PREV_TAG"
( cd "$WORKTREE" && cargo build --release --example benchmark )
PREV_BIN=$(resolve_bin "$WORKTREE/target/release/examples")

echo ""
echo "=== Old code (previous release) ==="
"$PREV_BIN" 16384 1 2>/dev/null | grep Algorithm || true
echo "=== New code ==="
"$CURR_BIN" 16384 1 2>/dev/null | grep Algorithm || true

# Accumulates one summary row per array size (filled in by benchmark_size).
SUMMARY=""

for size in $SIZES
do
    benchmark_size "$size"
done

echo ""
echo "=== Summary: throughput in GB/s (best of $BEST_OF runs) ==="
printf '%10s %12s %12s %9s\n' "size(B)" "old ($PREV_TAG)" "new" "new/old"
printf '%s' "$SUMMARY"
echo ""
echo "All benchmarks passed: the new code is faster than, or within 2% of, $PREV_TAG."
