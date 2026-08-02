# DataSource type

← [Type reference](../type-reference.md)

The `datasource` object is a universal connector for HTTP APIs, files, SQL databases, and MongoDB. Created by the `datasource()` function (see [built-in functions](../functions/README.md)).

## Config object

Use an object literal with colons:

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

Fields: `type`, `url` / `uri`, `path`, `host`, `port`, `database`, `collection`, `schema`, `name`, `enabled`, `timeout`, `headers`, `token`, `api_key`, `username` / `user`, `password`, nested `connection: { ... }`.

SQL types: `sqlite`, `postgresql` / `postgres`, `mysql` / `mariadb`, `mssql` / `sqlserver`.  
Document DB: `mongodb` / `mongo`.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `type` | `string` | Connector type |
| `name` | `string` | Name from config |
| `url` | `string` | URL or connection string |
| `enabled` | `bool` | Whether the source is enabled |
| `capabilities` | `object` | Capability flags (`supports_filter`, `supports_sql`, …) |

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `request(spec)` | `response` | Raw request (HTTP / file / SQL / Mongo op) |
| `get_table(spec)` | `table` | Fetch and parse a table |
| `send_table(spec)` | `null` | Send a table (file / HTTP POST / SQL INSERT / Mongo insert) |
| `connect()` | `null` | Open connection |
| `disconnect()` | `null` | Close connection |
| `ping()` | `bool` | Availability check |
| `test()` | `object` | Diagnostics `{ ok, message, ... }` (password redacted in URL) |
| `clone()` | `datasource` | Shallow copy of handle |

## SQL: get_table / send_table

```datacode
db = datasource({ type: "postgresql", url: "postgres://…" })
db.connect()
rows = db.get_table({ sql: "SELECT * FROM users", limit: 100 })
db.send_table({ table: rows, mode: "append", table_name: "users" })
db.disconnect()
```

## MongoDB → Table

BSON documents go through a Normalizer: nested objects become dotted columns (`profile.age`). Arrays stay as cell values by default; `array_mode: "explode"` expands rows.

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
    array_mode: "keep"   # or "explode"
})
t2 = db.get_table({
    aggregation: [
        { "$match": { age: { "$gte": 18 } } },
        { "$group": { "_id": "$city", "count": { "$sum": 1 } } }
    ]
})
db.request({ op: "count", filter: {} })
db.disconnect()
```

Filter operators: `eq`/`=`, `ne`/`!=`, `gt`/`>`, `gte`/`>=`, `lt`/`<`, `lte`/`<=`, `contains`, `starts_with`, `ends_with`, `in`, `not_in`, `is_null`, `is_not_null` (also with `$` prefix).

BSON mapping: ObjectId → `string` (hex), Decimal128 → `string`, Date → `date`, Binary → bytes.

## Response type

Returned from `request()`. `typeof` → `"response"`.

**Properties:** `status`, `success`, `url`, `headers`, `content_type`, `content_length`, `elapsed`, `body`, `text`, `bytes`

**Methods:** `json()`, `table()`, `csv()`, `save(path)`, `save_text(path)`, `save_json(path)`

For JSON → table with flatten: `get_table({ path: "…", flatten: true, array_mode: "keep" })`.

## typeof / isinstance

```datacode
typeof(api)                    # "datasource"
isinstance(api, "datasource")  # true
typeof(resp)                   # "response"
```
