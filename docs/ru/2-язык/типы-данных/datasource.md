# Тип DataSource

← [Справочник типов](../2-язык/справочник-типов.md)

Объект `datasource` — универсальный коннектор для HTTP API, файлов и SQLite. Создаётся функцией `datasource()` (см. [встроенные функции](../функции/README.md)).

## Объект конфигурации

Используйте литерал объекта с двоеточиями:

```datacode
datasource({ type: "http", url: "https://api.example.com" })
datasource({ type: "file", path: "./data" })
datasource({ type: "sqlite", url: "sqlite:///./app.db" })
```

Поля: `type`, `url`, `path`, `name`, `enabled`, `timeout`, `headers`, `token`, `api_key`, `username`, `password`.

## Свойства

| Свойство | Тип | Описание |
|----------|-----|----------|
| `type` | `string` | Тип коннектора |
| `name` | `string` | Имя из конфигурации |
| `url` | `string` | URL или строка подключения |
| `enabled` | `bool` | Включён ли источник |

## Методы

| Метод | Возвращает | Описание |
|-------|------------|----------|
| `request(spec)` | `response` | Сырой запрос (HTTP / файл / SQL) |
| `get_table(spec)` | `table` | Получить и распарсить таблицу |
| `send_table(spec)` | `null` | Отправить таблицу (файл / HTTP POST / SQL INSERT) |
| `connect()` | `null` | Открыть соединение |
| `disconnect()` | `null` | Закрыть соединение |
| `ping()` | `bool` | Проверка доступности |
| `test()` | `object` | Диагностика `{ ok, message, ... }` |
| `clone()` | `datasource` | Поверхностная копия handle |

## Тип Response

Возвращается из `request()`. `typeof` → `"response"`.

**Свойства:** `status`, `success`, `url`, `headers`, `content_type`, `content_length`, `elapsed`, `body`, `text`, `bytes`

**Методы:** `json()`, `table()`, `csv()`, `save(path)`, `save_text(path)`, `save_json(path)`

## Примеры

### HTTP

```datacode
api = datasource({ type: "http", url: "https://httpbin.org", timeout: 30 })
resp = api.request({ method: "GET", path: "/get", query: { foo: "bar" } })
print(resp.status)
data = resp.json()
```

### Файлы

```datacode
files = datasource({ type: "file", path: "./data" })
t = files.get_table({ path: "sales.csv" })
```

### SQLite

```datacode
db = datasource({ type: "sqlite", url: "sqlite:///./app.db" })
db.connect()
rows = db.get_table({ sql: "SELECT * FROM users" })
db.send_table({ table: rows, mode: "append", table_name: "users" })
db.disconnect()
```

## typeof / isinstance

```datacode
typeof(api)                    # "datasource"
isinstance(api, "datasource") # true
typeof(resp)                   # "response"
```
