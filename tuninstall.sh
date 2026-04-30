#!/bin/bash

# DataCode Uninstallation Script
# This script removes DataCode global command

set -e

echo "🧠 DataCode Uninstallation Script"
echo "================================="
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Rust/Cargo is not installed"
    exit 1
fi

echo "✅ Rust/Cargo found"

# Check if DataCode is installed
if ! command -v datacode &> /dev/null; then
    echo "⚠️  DataCode command not found - it may not be installed globally"
    echo "🔍 Checking cargo installation list..."
    
    if cargo install --list | grep -q "data-code"; then
        echo "✅ Found DataCode in cargo install list"
    else
        echo "❌ DataCode not found in cargo installations"
        echo "💡 It may have been installed differently or already removed"
        exit 1
    fi
else
    echo "✅ DataCode command found"
fi

# Uninstall using cargo
echo ""
echo "🗑️  Uninstalling DataCode..."
cargo uninstall data-code

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to uninstall DataCode"
    exit 1
fi

echo "✅ DataCode uninstalled successfully!"

# Verify removal
echo ""
echo "🧪 Verifying removal..."
if command -v datacode &> /dev/null; then
    echo "⚠️  Warning: DataCode command still found"
    echo "💡 You may need to restart your terminal"
else
    echo "✅ DataCode command successfully removed"
fi

echo ""
echo "🎉 Uninstallation completed!"
echo ""
echo "💡 Note: Your DataCode project files are still intact"
echo "🔄 To reinstall, run: ./install.sh"
