#!/usr/bin/env bash
set -euo pipefail

# cargo-xwin (and some build scripts) may pass MSVC-style include flags as:
#   /imsvc <dir>
# which clang treats as paths. Translate them to:
#   -imsvc <dir>
#
# Also supports concatenated form: /imsvc<dir>

out=()
i=0
argc=$#

while [ "$i" -lt "$argc" ]; do
  i=$((i + 1))
  a="${!i}"

  if [ "$a" = "/imsvc" ]; then
    if [ "$i" -ge "$argc" ]; then
      # Trailing /imsvc, keep as-is.
      out+=("$a")
      continue
    fi
    i=$((i + 1))
    dir="${!i}"
    out+=("-imsvc" "$dir")
    continue
  fi

  if [[ "$a" == /imsvc* ]]; then
    dir="${a#/imsvc}"
    if [ -n "$dir" ]; then
      out+=("-imsvc" "$dir")
      continue
    fi
  fi

  out+=("$a")
done

exec clang "${out[@]}"
