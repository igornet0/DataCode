# Тип DataSource

← [Справочник типов](../2-язык/справочник-типов.md)

Объект `datasource` — универсальный коннектор для HTTP API, файлов, SQL-БД и MongoDB. Создаётся функцией `datasource()` (см. [встроенные функции](../функции/README.md)).

## Объект конфигурации

Используйте литерал объекта с двоеточиями:

```datacode
datasource({ type: "http", url: "https://api.example.com" })
datasource({ type: "file", path: "./data" })
datasource({ type: "sqlite", url: "sqlite:///./app.db" })
datasource({ type: "postgresql", url: "postgres://user:pass@localhost:5432/app" })
datasource({ type: "mysql", url: "mysql://user:pass@localhost:3306/app" })
datasource({ type: "mssql", url: "mssql://user:pass@localhost:1433/app" })
datasource({
    type: "mongodb",
    url: "mongodb://localhost:27017",
    database: "app",
    collection: "users"
})
```

Поля: `type`, `url` / `uri`, `path`, `host`, `port`, `database`, `collection`, `schema`, `name`, `enabled`, `timeout`, `headers`, `token`, `api_key`, `username` / `user`, `password`, вложенный `connection: { ... }`.

Типы SQL: `sqlite`, `postgresql` / `postgres`, `mysql` / `mariadb`, `mssql` / `sqlserver`.  
Документная БД: `mongodb` / `mongo`.

## Свойства

| Свойство | Тип | Описание |
|----------|-----|----------|
| `type` | `string` | Тип коннектора |
| `name` | `string` | Имя из конфигурации |
| `url` | `string` | URL или строка подключения |
| `enabled` | `bool` | Включён ли источник |
| `capabilities` | `object` | Флаги возможностей (`supports_filter`, `supports_sql`, …) |

## Методы

| Метод | Возвращает | Описание |
|-------|------------|----------|
| `request(spec)` | `response` | Сырой запрос (HTTP / файл / SQL / Mongo op) |
| `get_table(spec)` | `table` | Получить и распарсить таблицу |
| `send_table(spec)` | `null` | Отправить таблицу (файл / HTTP POST / SQL INSERT / Mongo insert) |
| `connect()` | `null` | Открыть соединение |
| `disconnect()` | `null` | Закрыть соединение |
| `ping()` | `bool` | Проверка доступности |
| `test()` | `object` | Диагностика `{ ok, message, ... }` (пароль в URL не выводится) |
| `clone()` | `datasource` | Поверхностная копия handle |

## SQL: get_table / send_table

```datacode
db = datasource({ type: "postgresql", url: "postgres://…" })
db.connect()
rows = db.get_table({ sql: "SELECT * FROM users", limit: 100 })
db.send_table({ table: rows, mode: "append", table_name: "users" })
db.disconnect()
```

## MongoDB → Table

Документы BSON проходят через Normalizer: вложенные объекты становятся колонками с точкой (`profile.age`). Массивы по умолчанию остаются значением ячейки; `array_mode: "explode"` размножает строки.

```datacode
db = datasource({
    type: "mongodb",
    url: "mongodb://localhost:27017",
    database: "app",
    collection: "users"
})
db.connect()
t = db.get_table({
    filter: { age: { gt: 18 } },
    sort: { age: 1 },
    limit: 100,
    array_mode: "keep"   # или "explode"
})
# native aggregation
t2 = db.get_table({
    aggregation: [
        { "$match": { age: { "$gte": 18 } } },
        { "$group": { "_id": "$city", "count": { "$sum": 1 } } }
    ]
})
db.request({ op: "count", filter: {} })
db.disconnect()
```

Операторы фильтра: `eq`/`=`, `ne`/`!=`, `gt`/`>`, `gte`/`>=`, `lt`/`<`, `lte`/`<=`, `contains`, `starts_with`, `ends_with`, `in`, `not_in`, `is_null`, `is_not_null` (также с префиксом `$`).

BSON: ObjectId → `string` (hex), Decimal128 → `string`, Date → `date`, Binary → bytes.

## Тип Response

Возвращается из `request()`. `typeof` → `"response"`.

**Свойства:** `status`, `success`, `url`, `headers`, `content_type`, `content_length`, `elapsed`, `body`, `text`, `bytes`

**Методы:** `json()`, `table()`, `csv()`, `save(path)`, `save_text(path)`, `save_json(path)`

Для JSON → table с flatten: `get_table({ path: "…", flatten: true, array_mode: "keep" })`.

## typeof / isinstance

```datacode
typeof(api)                    # "datasource"
isinstance(api, "datasource") # true
typeof(resp)                   # "response"
```
