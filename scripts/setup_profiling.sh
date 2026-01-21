#!/bin/bash
# Скрипт для настройки инфраструктуры профилирования

set -e

echo "Setting up profiling infrastructure..."

# Check if scripts directory exists
mkdir -p scripts

# Install cargo-flamegraph if not installed
if ! command -v cargo-flamegraph &> /dev/null; then
    echo "Installing cargo-flamegraph..."
    cargo install flamegraph
else
    echo "cargo-flamegraph is already installed"
fi

echo ""
echo "Profiling infrastructure setup complete!"
echo ""
echo "Available profiling tools:"
echo "  1. CPU profiling: ./scripts/profile_cpu.sh"
echo "  2. GPU profiling: Use Instruments (macOS) - see PROFILING_GUIDE.md"
echo "  3. Performance testing: ./test_performance.sh"
echo ""
echo "Note: GPU profiling with Instruments requires manual setup:"
echo "  - Open Instruments.app"
echo "  - Select 'Metal System Trace' template"
echo "  - Target: ./target/release/datacode"
echo "  - See PROFILING_GUIDE.md for detailed instructions"

