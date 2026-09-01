#!/usr/bin/env bash
# Initialize git submodules for DataCode.
#
# Branch layout:
#   dev  — datacode_sdk (recursive) + datacode_registry_index; nested datacode_abi inside SDK
#   main — datacode_abi only + datacode_registry_index (lighter checkout for release builds)
#
# Usage: ./scripts/init-submodules.sh

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if git config -f .gitmodules --get submodule.datacode_abi.url >/dev/null 2>&1; then
    echo "init-submodules: main layout — datacode_abi (direct)"
    git submodule update --init datacode_abi
else
    echo "init-submodules: dev layout — datacode_sdk (recursive, includes datacode_abi)"
    git submodule update --init --recursive datacode_sdk
fi

if git config -f .gitmodules --get submodule.datacode_registry_index.url >/dev/null 2>&1; then
    git submodule update --init datacode_registry_index
fi
