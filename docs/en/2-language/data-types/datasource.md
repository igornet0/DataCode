# DataSource type

← [Type reference](../type-reference.md)

The `datasource` object is a universal connector for HTTP APIs, files, and SQLite. Created by the `datasource()` function (see [built-in functions](../functions/README.md)).

## Config object

Use an object literal with colons:

```datacode
datasource({ type: "http", url: "https://api.example.com" })
datasource({ type: "file", path: "./data" })
datasource({ type: "sqlite", url: "sqlite:///./app.db" })
```

Fields: `type`, `url`, `path`, `name`, `enabled`, `timeout`, `headers`, `token`, `api_key`, `username`, `password`.

## Properties

| Property | Type | Description |
|----------|-----|----------|
| `type` | `string` | Connector type |
| `name` | `string` | Name from config |
| `url` | `string` | URL or connection string |
| `enabled` | `bool` | Whether the source is enabled |

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `request(spec)` | `response` | Raw request (HTTP / file / SQL) |
| `get_table(spec)` | `table` | Fetch and parse a table |
| `send_table(spec)` | `null` | Send a table (file / HTTP POST / SQL INSERT) |
| `connect()` | `null` | Open connection |
| `disconnect()` | `null` | Close connection |
| `ping()` | `bool` | Availability check |
| `test()` | `object` | Diagnostics `{ ok, message, ... }` |
| `clone()` | `datasource` | Shallow copy of handle |

## Response type

Returned from `request()`. `typeof` → `"response"`.

**Properties:** `status`, `success`, `url`, `headers`, `content_type`, `content_length`, `elapsed`, `body`, `text`, `bytes`

**Methods:** `json()`, `table()`, `csv()`, `save(path)`, `save_text(path)`, `save_json(path)`

## Examples

### HTTP

```datacode
api = datasource({ type: "http", url: "https://httpbin.org", timeout: 30 })
resp = api.request({ method: "GET", path: "/get", query: { foo: "bar" } })
print(resp.status)
data = resp.json()
```

### Files

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
isinstance(api, "datasource")  # true
typeof(resp)                   # "response"
```
