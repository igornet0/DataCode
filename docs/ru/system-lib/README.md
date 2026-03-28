# Модуль `system` — окружение, рантайм, ОС и sandbox

Модуль **`system`** — встроенный модуль VM (реализован в ядре интерпретатора, не загружается как `.dylib`/`.so` через ABI). Он даёт иерархический API вида `system.env.get_os()`, `system.fs.read(path)` и служит единой точкой для доступа к ОС, путям, версии DataCode, железу и сети. Опасные операции (`fs`, `process.exec`, `set_env`) могут ограничиваться политикой `PermissionPolicy` на стороне встраивания (CLI/хост), см. публичный API `data_code::PermissionPolicy` и методы `Vm::set_permission_policy` / `can_system_permission`.

**Английская версия:** [system module (English)](../../en/system-lib/README.md)

## Импорт

```datacode
import system
```

## Обзор пространств имён

| Пространство | Назначение |
|--------------|------------|
| `system.env` | ОС, архитектура, версия ОС, хост, пользователь, каталоги, переменные окружения |
| `system.runtime` | Версии DataCode/VM, пути, загруженные модули, URL реестра пакетов |
| `system.hardware` | CPU, память, GPU (заглушка v1) |
| `system.time` | Текущее время (строка ISO 8601), задержка, uptime |
| `system.permissions` | Проверка и запрос прав (строковые ключи) |
| `system.log` | Логирование в stderr |
| `system.net` | Локальный IP, список сетевых интерфейсов |
| `system.process` | Запуск команды через shell (`exec`) |
| `system.fs` | Чтение/запись файлов |

## `system.env`

| Метод | Описание |
|-------|----------|
| `get_os()` | `"linux"` \| `"macos"` \| `"windows"` (или сырое имя ОС, если иное) |
| `get_arch()` | `"x86_64"`, `"arm64"` (для `aarch64`) и т.д. |
| `get_version()` | Строка версии ОС (через sysinfo) |
| `get_hostname()` | Имя хоста |
| `get_username()` | Имя пользователя (`USER` / `USERNAME`) |
| `get_home_dir()` | Домашний каталог |
| `get_temp_dir()` | Каталог временных файлов |
| `get(key)` | Значение переменной окружения или `null` |
| `set_env(key, value)` | Установить переменную окружения (**требует право `env.write` в режиме Restricted**) |

Пример:

```datacode
import system
print(system.env.get_os())
print(system.env.get("PATH"))
```

## `system.runtime`

| Метод | Описание |
|-------|----------|
| `get_datacode_version()` | Версия пакета DataCode (`CARGO_PKG_VERSION`) |
| `get_vm_version()` | Версия VM (сейчас совпадает с версией пакета) |
| `get_module_path()` | Базовый путь скрипта / проекта или `null` |
| `get_venv_path()` | `VIRTUAL_ENV` или `DATACODE_VENV`, иначе `null` |
| `get_dpm_env_base()` | Каталог из **`DPM_ENV_BASE`** (то, что задаёт `dpm --env-path=...` для процесса) или `null`, если переменная не задана |
| `get_dpm_env_root()` | Вычисленный **корень окружения DPM** для текущего проекта (`dpm.toml`): где лежит `packages/`, кэш `…/name-hash/`, либо `<project>/.dpm/` — или `null`, если проект без DPM / ошибка |
| `get_loaded_modules()` | Массив строк — имена загруженных модулей |
| `get_registry_url()` | URL индекса реестра пакетов (`DATACODE_REGISTRY_URL` или значение по умолчанию) |

## `system.hardware`

| Метод | Описание |
|-------|----------|
| `cpu_count()` | Число логических CPU (не меньше 1) |
| `memory_total()` | Объём RAM, байты |
| `memory_free()` | Доступная память, байты |
| `gpu_count()` | Пока `0` (заглушка) |
| `gpu_info()` | Пустой массив (заглушка; контракт может расширяться) |

## `system.time`

| Метод | Описание |
|-------|----------|
| `now()` | Текущий момент в UTC, строка **RFC 3339** (в языке нет отдельного типа Date в значении «объект даты» — используется строка) |
| `sleep(ms)` | Пауза в миллисекундах |
| `uptime()` | Время работы системы с загрузки, секунды |

## `system.permissions`

| Метод | Описание |
|-------|----------|
| `has_permission(key)` | `true`/`false` в соответствии с политикой VM |
| `request_permission(key)` | Заглушка: пока всегда `true` (интерактивный запрос — в будущем) |

Известные ключи, блокируемые в режиме **Restricted**: `"fs.read"`, `"fs.write"`, `"process.exec"`, `"env.write"`.

## `system.log`

Сообщения идут в **stderr** с префиксом `[system]`.

| Метод | Описание |
|-------|----------|
| `log_info(...)` | Уровень INFO |
| `log_warn(...)` | Уровень WARN |
| `log_error(...)` | Уровень ERROR |
| `debug(...)` | DEBUG; вывод только при включённом отладочном режиме (`DATACODE_DEBUG`) |

## `system.net`

| Метод | Описание |
|-------|----------|
| `get_ip()` | Локальный IPv4, если доступен, иначе `null` |
| `get_interfaces()` | Массив объектов с полями `name`, `ip`, `is_loopback` |

## `system.process`

| Метод | Описание |
|-------|----------|
| `exec(command)` | Одна строка команды: на Unix — `sh -c`, на Windows — `cmd /C`. Возвращает объединённый stdout/stderr. (**`process.exec`** проверяется в Restricted) |

## `system.fs`

| Метод | Описание |
|-------|----------|
| `read(path)` | Текст файла UTF-8 или строка с текстом ошибки |
| `write(path, content)` | Запись строки в файл |

При отказе по политике возвращается строка вида `permission denied: fs.read`.

### DPM и `--env-path`

Команда `dpm --env-path=/путь/к/базе …` выставляет переменную окружения **`DPM_ENV_BASE`** (см. `src/dpm_main.rs`). Её видит тот же процесс; при отдельном запуске **`datacode script.dc`** переменная **не подставляется автоматически**, если вы не экспортировали её в shell или не положили `env_base` в `dpm.toml`.

- Узнать базу из кода: `system.runtime.get_dpm_env_base()` или `system.env.get("DPM_ENV_BASE")`.
- Узнать **фактический** каталог окружения пакетов для проекта (после `dpm init`): `system.runtime.get_dpm_env_root()` — не зависит от того, передан ли `--env-path` в текущем процессе, если есть корректный `dpm.toml` и известен корень проекта.

## Реализация и безопасность

- Нативы реализованы в Rust в репозитории (`src/lib/system/natives.rs`), не через SDK ABI.
- Политика: `data_code::PermissionPolicy`, методы VM `set_permission_policy` / `permission_policy` / `can_system_permission`.
- Подробнее о встроенных модулях и импорте: [Модули и импорты](../modules_and_imports.md).
