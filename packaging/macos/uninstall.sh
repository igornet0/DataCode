#!/usr/bin/env bash
# Удаление бинарников DataCode из /usr/local/bin
set -euo pipefail

DEST="/usr/local/bin"
BINS=(datacode dpm datacode-server)

need_sudo() {
  if [[ -w "$DEST" ]] 2>/dev/null; then
    return 1
  fi
  return 0
}

run() {
  if need_sudo; then
    sudo "$@"
  else
    "$@"
  fi
}

for name in "${BINS[@]}"; do
  p="${DEST}/${name}"
  if [[ -e "$p" ]] || [[ -L "$p" ]]; then
    run rm -f "$p"
    echo "Удалено: $p"
  else
    echo "Нет: $p"
  fi
done

echo "Готово."
