#!/usr/bin/env bash
# Установка Datacode из распакованного бандла (Linux, .tar.gz) в /usr/local/bin
# Требуется: бинарники в ./bin/ рядом с этим скриптом.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${BUNDLE_DIR}/bin"
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

# Разрешение пути к бинарнику: сначала bin/, затем корень бандла
path_for() {
  local n="$1"
  if [[ -f "${BIN_DIR}/${n}" ]]; then
    echo "${BIN_DIR}/${n}"
  elif [[ -f "${BUNDLE_DIR}/${n}" ]]; then
    echo "${BUNDLE_DIR}/${n}"
  else
    echo ""
  fi
}

if [[ ! -d "$BIN_DIR" ]] && [[ ! -f "${BUNDLE_DIR}/datacode" ]]; then
  echo "Ожидались бинарники в ${BIN_DIR} или в ${BUNDLE_DIR}" >&2
  exit 1
fi

if [[ ! -d "$DEST" ]]; then
  run mkdir -p "$DEST"
fi

for name in "${BINS[@]}"; do
  src="$(path_for "$name")"
  if [[ -z "$src" ]] || [[ ! -f "$src" ]]; then
    echo "Не найден: ${name} (ожидался ${BIN_DIR}/${name} или ${BUNDLE_DIR}/${name})" >&2
    exit 1
  fi
  if command -v install >/dev/null 2>&1; then
    run install -m 0755 "$src" "${DEST}/${name}"
  else
    run cp -f "$src" "${DEST}/${name}"
    run chmod 0755 "${DEST}/${name}"
  fi
  echo "Установлено: ${DEST}/${name}"
done

echo ""
echo "Перезапустите терминал (или: hash -r) и проверьте:"
echo "  datacode --version"
echo "  dpm --version"
echo "  datacode-server --help"
