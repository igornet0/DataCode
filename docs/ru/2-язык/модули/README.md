# Встроенные модули

Документация по модулям, поставляемым с DataCode.

| Модуль | Описание |
|--------|----------|
| [plot](./plot/README.md) | Графики, изображения, окна |
| [uuid](./uuid/README.md) | Генерация UUID |
| [settings_env](./settings_env/README.md) | Переменные окружения из `.env` |
| [system](./system/README.md) | ОС, рантайм, sandbox |
| [database](./database/README.md) | `database_engine`, SQL, SQLite |
| [heapq](./heapq/README.md) | Min-куча (очередь с приоритетом) |
| [pathfind](./pathfind/README.md) | Native A* на сетке (production SLA) |
| [grid](./grid/README.md) | Плоские буферы + `grid.astar` для DC A* |
| [crypto](./crypto/README.md) | Argon2id, bcrypt, `secure_compare` |
| [debug](./debug/README.md) | `debug.operators()` — таблица операторов VM |

### Устанавливаемые пакеты (DPM)

| Модуль | Описание |
|--------|----------|
| [ml](./ml-модуль.md) | Машинное обучение — `dpm add ml`, [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib) |

Импорт и структура пакетов: [0-синтаксис/модули-и-импорты](../../0-синтаксис/11-модули-и-импорты.md).

Реализация загрузки модулей: [200-разработчикам/module_import_system.md](../../200-разработчикам/module_import_system.md).
