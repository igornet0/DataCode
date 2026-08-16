# WebSocket сервер DataCode

WebSocket сервер для удалённого выполнения DataCode через **DCP-пакеты** (формат Datacode Package).

**📚 Примеры:**
- [`examples/ru/07-websocket/`](../../../../examples/ru/07-websocket/)
- Формат DCP: [DCP-python](https://github.com/igornet0/DCP-python/blob/main/README.md)

## Запуск сервера

```bash
datacode --websocket
datacode --websocket --host 0.0.0.0 --port 8899 --build_model
datacode examples/ru/07-websocket/dc/ws_app.dc --websocket --host 0.0.0.0 --port 8899
```

Для каждого клиента — sandbox (`getcwd()` пустой, пути относительно ASSET в DCP). Файлы читаются из **in-memory VFS**, каталог `temp_sessions` **не создаётся**.

Локальная запись через `save()` **запрещена** в DCP WebSocket-сессиях.

### Скрипт приложения (`ws_app.dc`)

Опциональный скрипт настройки. Выполняется один раз при старте.

```dc
from websocket import configure, disable_builtin

configure({"execute_policy": "restricted"})

@ws_route("ping")
fn ping(req) {
    return {"success": true, "message": "pong"}
}
```

| Функция | Описание |
|---------|----------|
| `websocket.configure({...})` | `execute_policy`: `"allow_all"` или `"restricted"` |
| `websocket.disable_builtin("type")` | Отключить встроенный JSON-обработчик |
| `websocket.enable_builtin("type")` | Включить обратно |

## Протокол

### Выполнение через DCP

| Направление | Формат |
|-------------|--------|
| Запрос | **Binary WebSocket frame** — сырые байты `.dcp` (magic `DCPK`) |
| Ответ | **JSON text** — результат выполнения |

**Сборка пакета:**

```bash
dcp create -o job.dcp --code script.dc --assets-dir ./data
```

**Отправка:** один binary frame с содержимым файла.

**Ответ:**

```json
{
  "success": true,
  "output": "asset: hello\n2 + 2 = 4\n",
  "error": null
}
```

С `--build_model` при успехе может быть поле `sqlite_db` (base64 SQLite).

### Режим проверки FK (`__config__`)

В DCP можно добавить секцию **`__config__`** (`SectionType.CONFIG = 5`) с JSON `{"fk_check":"warn"}`. Флаг сервера не меняется: `datacode --websocket --build_model` без этой секции — **`strict`**.

```python
from datacode_dcp import DCPEncoder

DCPEncoder().code(script).fk_check("warn").write(buf)
```

| режим | FK в DDL | сироты | экспорт |
|-------|----------|--------|---------|
| `strict` (по умолчанию) | все `relate` | нет | fail |
| `warn` | только выполняемые | да, без сломанного FK | ok + `warning` |
| `skip` | все `relate` | да | ok, БД с нарушениями |

Невалидное значение `fk_check` — ошибка decode пакета. При `warn` / `skip` в JSON-ответе может быть поле `warning` (`success: true`, `sqlite_db` заполнен).

### SQL после сборки модели (секция `__sql__`)

В DCP можно добавить секцию **`__sql__`** (`SectionType.SQL = 7`) с UTF-8 SQL-скриптом. Сервер выполняет его **после** экспорта глобальных таблиц в SQLite, в одной транзакции, **только с флагом `--build_model`**.

```python
from datacode_dcp import DCPEncoder

DCPEncoder().code("""
global t = table([[1, "a"]], ["id", "name"])
""").sql("CREATE VIEW v_names AS SELECT name FROM t;").write(buf)
```

| Исход | `success` | `error` | `sqlite_db` |
|-------|-----------|---------|-------------|
| Код + экспорт + SQL OK | `true` | `null` | БД **с** SQL |
| Код OK, SQL упал | `false` | `"SQL error: ..."` | БД **без** SQL |
| SQL без `--build_model` | `false` | `"SQL section requires --build_model"` | `null` |

Пример: [`examples/ru/07-websocket/python/test_dcp_sql.py`](../../../../examples/ru/07-websocket/python/test_dcp_sql.py).

### Мягкий SQL к таблицам (`sql_table` / `table_insert`)

Вставки после экспорта `--build_model`. Ошибки (нет таблицы, несовпадение колонок) — **пропуск + warning в консоль**, пакет не падает.

```python
DCPEncoder().code("""
global t = table([[1, "a"]], ["id", "name"])
""").table_insert("t", {"id": 2, "name": "b"}).write(buf)
```

Секция `__sql_table__` (type=8). Применяется **до** жёсткого `__sql__`.

Пакет должен содержать секцию `CODE` (`__code__`). ASSET-секции монтируются в in-memory VFS. Секции `ARROW_TABLE` доступны через встроенный модуль **`ws`** (см. ниже).

### Модуль `ws` (API текущего DCP)

Пока на WebSocket выполняется DCP-пакет, код может импортировать `ws` для **безопасного** доступа к данным пакета — без путей сервера, host/port и учётных данных.

```dc
from ws import source_table, tables, assets, content_assets, package_info

print(tables())
print(content_assets())  # SHA-256 id content-addressed assets
global orders = source_table("orders", ["id", "date", "value"])
global renamed = source_table("orders", {"id": null, "value": "amount"})
```

| Функция | Описание |
|---------|----------|
| `tables()` | Имена секций ARROW_TABLE |
| `has_table(name)` | Есть ли таблица |
| `source_table(name, columns?)` | Загрузить таблицу; проверяет ссылки `asset://` |
| `assets()` | Логические пути path-based ASSET (VFS) |
| `has_asset(name)` | Есть ли path asset |
| `content_assets()` | SHA-256 id content-addressed assets (`assets/{id}`) |
| `has_content_asset(id)` | Есть ли content asset |
| `content_asset(id)` | `Image` для image kind/mime, иначе `ByteBuffer` |
| `metadata()` | Пользовательские metadata (строки) |
| `metadata_get(key)` | Одно значение или `null` |
| `package_info()` | Сводка (включая `content_asset_count`) |

**Фильтр колонок `source_table`** (строже, чем `read(..., header=...)`):

- без аргумента / `null` — все колонки
- массив строк — только перечисленные, порядок как в массиве; **ошибка**, если колонки нет
- объект — только ключи; строка переименовывает, `null` оставляет имя; **ошибка**, если ключ отсутствует в таблице

**Content-addressed assets:** в ячейках таблиц — UTF-8 `asset://{sha256}`; байты в секциях `ASSET` с именем `assets/{sha256}` (дедуп по хешу). Path-based VFS assets для `read(path(...))` остаются отдельно. При `--build_model` content assets пишутся в SQLite-таблицу `__assets`; бизнес-колонки сохраняют текст `asset://`.

Вне активной DCP-сессии: `ws: no active DCP session`.

Пример: [`examples/ru/07-websocket/dc/ws_source_table.dc`](../../../../examples/ru/07-websocket/dc/ws_source_table.dc).

### SMB (JSON text)

```json
{
  "type": "smb_connect",
  "ip": "192.168.1.100",
  "login": "username",
  "password": "password",
  "domain": "WORKGROUP",
  "share_name": "share_name"
}
```

После подключения отправляйте DCP с кодом, использующим `lib://share_name/path`.

## Пример клиента (Python)

```python
import asyncio, io, json, websockets
from datacode_dcp import DCPEncoder

async def run():
    buf = io.BytesIO()
    DCPEncoder().code("print('Hello from DCP')").write(buf)
    async with websockets.connect("ws://127.0.0.1:8899") as ws:
        await ws.send(buf.getvalue())
        print(json.loads(await ws.recv())["output"])

asyncio.run(run())
```

Полный пример: [`examples/ru/07-websocket/python/test_dcp_run.py`](../../../../examples/ru/07-websocket/python/test_dcp_run.py)

## Безопасность

⚠️ Аутентификация не реализована. Не используйте на публичных сетях без дополнительной защиты.

---

**См. также:**
- [Примеры WebSocket](../../../../examples/ru/07-websocket/)
- [DCP-python README](../../../../DCP-python/README.md)
