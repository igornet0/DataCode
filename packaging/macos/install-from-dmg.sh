#!/usr/bin/env bash
# Создание символических ссылок в /usr/local/bin на бинарники из DataCode.app
# (после копирования DataCode.app в /Applications)
set -euo pipefail

APP_DIR="${DATA_CODE_APP:-/Applications/DataCode.app}"
MACOS_DIR="${APP_DIR}/Contents/MacOS"
DEST="/usr/local/bin"
BINS=(datacode dpm datacode-server)

if [[ ! -d "$MACOS_DIR" ]]; then
  echo "Не найдено приложение: ${APP_DIR}" >&2
  echo "Перетащите DataCode.app в Программы или задайте DATA_CODE_APP=/путь/к/DataCode.app" >&2
  exit 1
fi

if [[ ! -d "$DEST" ]]; then
  if [[ -w "/usr/local" ]] 2>/dev/null; then
    mkdir -p "$DEST"
  else
    sudo mkdir -p "$DEST"
  fi
fi

linked=0
for name in "${BINS[@]}"; do
  if [[ ! -f "${MACOS_DIR}/${name}" ]]; then
    echo "Пропуск (нет в .app): ${name}" >&2
    continue
  fi
  if [[ -w "$DEST" ]]; then
    ln -sfn "${MACOS_DIR}/${name}" "${DEST}/${name}"
  else
    sudo ln -sfn "${MACOS_DIR}/${name}" "${DEST}/${name}"
  fi
  echo "Симлинк: ${DEST}/${name} -> ${MACOS_DIR}/${name}"
  linked=1
done
if [[ "$linked" -eq 0 ]]; then
  echo "Не найдено ни одного бинарника в ${MACOS_DIR}" >&2
  exit 1
fi

echo ""
echo "Проверьте: datacode --version"
