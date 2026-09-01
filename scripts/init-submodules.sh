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
    abi_path="$(git config -f .gitmodules --get submodule.datacode_abi.path)"
    echo "init-submodules: main layout — datacode_abi at ${abi_path}"
    git submodule sync -- "${abi_path}"
    git submodule update --init -- "${abi_path}"
else
    echo "init-submodules: dev layout — datacode_sdk (recursive, includes datacode_abi)"
    git submodule update --init --recursive datacode_sdk
fi

if git config -f .gitmodules --get submodule.datacode_registry_index.url >/dev/null 2>&1; then
    registry_path="$(git config -f .gitmodules --get submodule.datacode_registry_index.path)"
    git submodule sync -- "${registry_path}"
    git submodule update --init -- "${registry_path}"
fi
