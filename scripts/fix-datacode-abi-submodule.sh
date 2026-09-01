#!/usr/bin/env bash
# Repair submodule layout after moving datacode_abi to crates/datacode_abi.
#
# Symptom:
#   fatal: No url found for submodule path 'datacode_abi' in .gitmodules
#
# Cause: git index still tracks old path `datacode_abi`, while .gitmodules points to `crates/datacode_abi`.
#
# Usage (from repo root, on main):
#   ./scripts/fix-datacode-abi-submodule.sh

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

abi_path="$(git config -f .gitmodules --get submodule.datacode_abi.path 2>/dev/null || true)"
abi_url="$(git config -f .gitmodules --get submodule.datacode_abi.url 2>/dev/null || true)"

if [[ -z "${abi_path}" || -z "${abi_url}" ]]; then
    echo "error: submodule.datacode_abi is missing in .gitmodules" >&2
    exit 1
fi

echo "target ABI path: ${abi_path}"

# Drop stale gitlink/config at legacy root path.
if git ls-files --stage -- datacode_abi | grep -q '^160000'; then
    echo "removing stale gitlink: datacode_abi"
    git submodule deinit -f datacode_abi 2>/dev/null || true
    git rm -f datacode_abi
fi
if [[ -d .git/modules/datacode_abi ]]; then
    rm -rf .git/modules/datacode_abi
fi
git config --remove-section submodule.datacode_abi 2>/dev/null || true

mkdir -p "$(dirname "${abi_path}")"

if ! git ls-files --stage -- "${abi_path}" | grep -q '^160000'; then
    echo "registering submodule at ${abi_path}"
    git submodule add -b main "${abi_url}" "${abi_path}"
else
    echo "submodule already registered at ${abi_path}, syncing"
    git submodule sync -- "${abi_path}"
    git submodule update --init -- "${abi_path}"
fi

echo "done"
