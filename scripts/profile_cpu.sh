#!/bin/bash
# Скрипт для профилирования CPU версии с cargo flamegraph

set -e

echo "Building release version for CPU profiling..."
cargo build --release --bin datacode

echo "Running CPU profiling with flamegraph..."
echo "This will take several minutes..."

# Check if flamegraph is installed
if ! command -v cargo-flamegraph &> /dev/null; then
    echo "cargo-flamegraph is not installed. Installing..."
    cargo install flamegraph
fi

# Run with flamegraph
# Note: On macOS, you may need to run with sudo for dtrace
cargo flamegraph --bin datacode --release -- \
    examples/en/11-mnist-mlp/mnist_mlp.dc

echo "Flamegraph saved to flamegraph.svg"
echo "Open it in a browser to analyze performance hotspots"

