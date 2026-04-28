#!/usr/bin/env bash
# Удаление символических ссылок в /usr/local/bin (обратно install-from-dmg.sh)
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
  if [[ -L "$p" ]]; then
    run rm -f "$p"
    echo "Удалён симлинк: $p"
  elif [[ -e "$p" ]]; then
    echo "Пропуск (не симлинк): $p" >&2
  fi
done

echo "Готово."
