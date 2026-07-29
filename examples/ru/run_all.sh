#!/usr/bin/env bash
# Прогон всех .dc-примеров в examples/ru через datacode.
#
# Умный проход:
#   • порядок курса: 01-основы … 18-debug (LC_ALL=C sort внутри папок);
#   • каждый скрипт запускается из своей директории: datacode name.dc;
#   • 10-графики — с --no-gui (без блокирующих окон);
#   • пропускаются только вспомогательные/неавтономные файлы (см. should_skip).
#
# Использование:
#   ./run_all.sh           — все примеры, остановка на первом сбое
#   ./run_all.sh --quiet   — без вывода скриптов, только статус
#   ./run_all.sh --list    — показать список без запуска
#
# Запуск из корня репозитория или из examples/ru:
#   examples/ru/run_all.sh
#   cd examples/ru && ./run_all.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"

QUIET=false
LIST_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --quiet|-q) QUIET=true ;;
    --list) LIST_ONLY=true ;;
    -h|--help)
      sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Неизвестный аргумент: $arg (см. $0 --help)" >&2
      exit 2
      ;;
  esac
done

if [[ -x "$REPO_ROOT/target/release/datacode" ]]; then
  DC=("$REPO_ROOT/target/release/datacode")
elif [[ -x "$REPO_ROOT/target/debug/datacode" ]]; then
  DC=("$REPO_ROOT/target/debug/datacode")
elif command -v datacode &>/dev/null; then
  DC=(datacode)
else
  echo "datacode не найден в PATH и в target/{debug,release}/" >&2
  exit 127
fi

# rel — путь относительно examples/ru
should_skip() {
  local rel=$1
  case "$rel" in
    # Внутренние bootstrap-модули, не точки входа
    */__lib__.dc) return 0 ;;
    # Компоненты пакета 14-модули (точка входа — main.dc)
    14-модули/core/*) return 0 ;;
    # Локальные тестовые сниппеты, не часть курса
    08-создание\ модели\ данных/test.dc|\
    08-создание\ модели\ данных/test_and_or.dc) return 0 ;;
    # Требует живую SMB-шару и smb_connect
    07-websocket/dc/test_smb_load_data.dc) return 0 ;;
  esac
  return 1
}

# Дополнительные флаги datacode для отдельных разделов
extra_args() {
  local rel=$1
  case "$rel" in
    10-графики/*) printf '%s\n' --no-gui ;;
  esac
}

# Каталог запуска и аргумент datacode (name.dc или путь от REPO_ROOT)
run_context() {
  local script=$1
  local rel=$2
  case "$rel" in
    # Пути относительно корня репозитория (tests/test_data/…)
    03-типы\ данных/archive.dc|15-datasource/02-file.dc)
      printf '%s\n' "$REPO_ROOT" "examples/ru/$rel"
      ;;
    *)
      printf '%s\n' "$(dirname "$script")" "$(basename "$script")"
      ;;
  esac
}

mapfile -t ALL_SCRIPTS < <(
  find "$ROOT" -type f -name '*.dc' ! -name '_*' | LC_ALL=C sort
)

SCRIPTS=()
SKIPPED=()
for script in "${ALL_SCRIPTS[@]}"; do
  rel="${script#"$ROOT"/}"
  if should_skip "$rel"; then
    SKIPPED+=("$rel")
  else
    SCRIPTS+=("$script")
  fi
done

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
  echo "Нет .dc файлов для запуска в $ROOT" >&2
  exit 1
fi

if $LIST_ONLY; then
  echo "Будет запущено ${#SCRIPTS[@]} примеров (${DC[*]}):"
  for script in "${SCRIPTS[@]}"; do
    rel="${script#"$ROOT"/}"
    mapfile -t _ctx < <(run_context "$script" "$rel")
    args=()
    while IFS= read -r a; do [[ -n $a ]] && args+=("$a"); done < <(extra_args "$rel")
    printf '  cd %q && datacode' "${_ctx[0]}"
    ((${#args[@]})) && printf ' %q' "${args[@]}"
    printf ' %q\n' "${_ctx[1]}"
  done
  if ((${#SKIPPED[@]})); then
    echo ""
    echo "Пропущено ${#SKIPPED[@]}:"
    for s in "${SKIPPED[@]}"; do
      echo "  - $s"
    done
  fi
  exit 0
fi

echo "Запуск ${#SCRIPTS[@]} примеров (${DC[*]})"
echo "Каталог: $ROOT"
if ((${#SKIPPED[@]})); then
  echo "Пропущено ${#SKIPPED[@]} вспомогательных файлов (см. --list)"
fi
echo "---"

run_one() {
  local script=$1
  local rel=$2
  local idx=$3
  local total=$4
  local run_dir dc_arg tmp code out assert_fail
  local -a args=()

  mapfile -t _ctx < <(run_context "$script" "$rel")
  run_dir="${_ctx[0]}"
  dc_arg="${_ctx[1]}"
  while IFS= read -r a; do [[ -n $a ]] && args+=("$a"); done < <(extra_args "$rel")

  tmp=$(mktemp)
  printf '[%d/%d] %s ... ' "$idx" "$total" "$rel"

  set +e
  if $QUIET; then
    (cd "$run_dir" && "${DC[@]}" "${args[@]}" "$dc_arg") >"$tmp" 2>&1
    code=$?
  else
    (cd "$run_dir" && "${DC[@]}" "${args[@]}" "$dc_arg") 2>&1 | tee "$tmp"
    code=${PIPESTATUS[0]}
  fi
  set -e

  out=$(<"$tmp")
  rm -f "$tmp"

  assert_fail=false
  if echo "$out" | grep -q 'Тест: FAIL'; then
    assert_fail=true
  fi

  if [[ $code -ne 0 ]]; then
    echo "FAIL (exit $code)"
    if $QUIET; then
      echo "$out" | sed 's/^/  /'
    fi
    echo "$out" | grep -E 'Тест:|FAIL|Ошибка|error\[|Caused by:' | sed 's/^/  /' || true
    return 1
  fi

  if $assert_fail; then
    echo "FAIL (assert «Тест: FAIL»)"
    echo "$out" | grep -E 'Тест:|FAIL' | sed 's/^/  /' || true
    return 2
  fi

  echo OK
  return 0
}

i=0
for script in "${SCRIPTS[@]}"; do
  i=$((i + 1))
  rel="${script#"$ROOT"/}"

  set +e
  run_one "$script" "$rel" "$i" "${#SCRIPTS[@]}"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "---"
    if [[ $rc -eq 1 ]]; then
      echo "Остановка: сбой (exit) — $rel" >&2
    else
      echo "Остановка: сбой (assert) — $rel" >&2
    fi
    exit "$rc"
  fi
done

echo "---"
echo "Все ${#SCRIPTS[@]} примеров прошли успешно."
exit 0
