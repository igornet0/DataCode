# DataCode WebSocket Server

WebSocket server for remote execution of DataCode via **DCP packages** (Datacode Package format).

**📚 Usage examples:**
- WebSocket clients: [`examples/en/07-websocket/`](../../../examples/en/07-websocket/)
- DCP format: [DCP-python](https://github.com/igornet0/DCP-python/blob/main/README.md)

## Starting the server

```bash
# Default address (127.0.0.1:8080)
datacode --websocket

# Custom host and port
datacode --websocket --host 0.0.0.0 --port 8899 --build_model

# With optional ws_app.dc
datacode examples/en/07-websocket/dc/ws_app.dc --websocket --host 0.0.0.0 --port 8899
```

Each client connection runs in a sandbox (`getcwd()` empty, paths relative to DCP assets). Asset files are served from an **in-memory VFS** populated from DCP ASSET sections — no `temp_sessions` directory on disk.

Local `save()` / file writes are **blocked** in DCP WebSocket sessions.

### Application script (`ws_app.dc`)

Optional setup script. Runs once at startup before accepting connections.

**Example** [`examples/en/07-websocket/dc/ws_app.dc`](../../../examples/en/07-websocket/dc/ws_app.dc):

```dc
from websocket import configure, disable_builtin

configure({"execute_policy": "restricted"})

@ws_route("ping")
fn ping(req) {
    return {"success": true, "message": "pong"}
}
```

| Function | Description |
|----------|-------------|
| `websocket.configure({...})` | `execute_policy`: `"allow_all"` (default) or `"restricted"` |
| `websocket.disable_builtin("type")` | Disable built-in JSON handler (e.g. `smb_connect`) |
| `websocket.enable_builtin("type")` | Re-enable |

**`@ws_route("type")`** — custom JSON handler. Receives the request object, returns a response object.

## Protocol

### DCP execution (primary)

| Direction | Format |
|-----------|--------|
| Request | **Binary WebSocket frame** — raw `.dcp` bytes (magic `DCPK`) |
| Response | **JSON text** — execution result |

**Build a package** (Python CLI or library):

```bash
dcp create -o job.dcp --code script.dc --assets-dir ./data
```

**Send** the file bytes as one binary WebSocket message.

**Response:**

```json
{
  "success": true,
  "output": "asset: hello\n2 + 2 = 4\n",
  "error": null,
  "sqlite_db": null
}
```

With `--build_model`, successful runs may include `sqlite_db` (base64-encoded SQLite file).

### FK check mode (`__config__`)

The DCP package may include a **`__config__`** section (`SectionType.CONFIG = 5`) with JSON `{"fk_check":"warn"}`. The server CLI flag is unchanged: `datacode --websocket --build_model` without this section is **`strict`**.

```python
from datacode_dcp import DCPEncoder

DCPEncoder().code(script).fk_check("warn").write(buf)
```

| mode | FK in DDL | orphans | export |
|------|-----------|---------|--------|
| `strict` (default) | all `relate` | none | fail |
| `warn` | only FKs the data satisfies | yes, without the broken FK | ok + `warning` |
| `skip` | all `relate` | yes | ok, DB may violate FKs |

Invalid `fk_check` values fail package decode. On `warn` / `skip` the JSON response may include `warning` (export still `success: true` with `sqlite_db`).

### SQL post-model transactions (`__sql__` section)

DCP packages may include a **`__sql__`** section (`SectionType.SQL = 7`) with a UTF-8 SQL script. The server applies it **after** exporting global tables to SQLite, inside a single transaction, **only when `--build_model` is enabled**.

Python encoder:

```python
from datacode_dcp import DCPEncoder

DCPEncoder().code("""
global t = table([[1, "a"]], ["id", "name"])
""").sql("""
CREATE VIEW v_names AS SELECT name FROM t;
""").write(buf)
```

| Outcome | `success` | `error` | `sqlite_db` |
|---------|-----------|---------|-------------|
| Code + export + SQL OK | `true` | `null` | DB **with** SQL applied |
| Code OK, SQL failed | `false` | `"SQL error: ..."` | DB **without** SQL (pre-transaction snapshot) |
| SQL section without `--build_model` | `false` | `"SQL section requires --build_model"` | `null` |

## Soft table SQL (`sql_table` / `table_insert`)

Best-effort inserts after `--build_model` export. Failures (missing table, column mismatch) are **skipped with a console warning** — the package still succeeds.

```python
DCPEncoder().code("""
global t = table([[1, "a"]], ["id", "name"])
""").table_insert("t", {"id": 2, "name": "b"}).sql_table(
    'INSERT INTO "t" ("id", "name") VALUES (3, \'c\');'
).write(buf)

# Or from pandas:
# .table_insert("t", df)
```

Section: `__sql_table__` (`SectionType.SQL_TABLE = 8`). Applied **before** hard `__sql__`.

| API | Behavior |
|-----|----------|
| `.sql(...)` | Hard: one transaction; fail → `success=false` + DB without that SQL |
| `.sql_table(...)` | Soft: per-statement; warn + skip |
| `.table_insert(name, data)` | Soft: generates `INSERT` from dict / list[dict] / DataFrame |

See [`examples/en/07-websocket/python/test_dcp_sql.py`](../../../examples/en/07-websocket/python/test_dcp_sql.py).

The DCP package must contain a `CODE` section (`__code__`). Optional `ASSET` sections (relative paths) are mounted in an in-memory VFS. Optional `ARROW_TABLE` sections are available through the built-in **`ws`** module (see below).

### `ws` module (DCP session API)

While a DCP package is executing on a WebSocket connection, code can import the built-in `ws` module for **safe** access to the current package — no server paths, host/port, or credentials are exposed.

```dc
from ws import source_table, tables, assets, content_assets, package_info

print(tables())          # ["orders", ...] — ARROW_TABLE section names
print(assets())          # logical asset paths, e.g. ["data/sample.csv"]
print(content_assets())  # SHA-256 ids of content-addressed assets
print(package_info())    # { table_count, asset_count, content_asset_count, ... }

global orders = source_table("orders")
global subset = source_table("orders", ["id", "date", "value"])
global renamed = source_table("orders", {"id": null, "value": "amount"})
```

| Function | Description |
|----------|-------------|
| `tables()` | Array of ARROW_TABLE section names |
| `has_table(name)` | Whether a table section exists |
| `source_table(name, columns?)` | Load table by section name into `Value::Table`; validates `asset://` refs |
| `assets()` | Logical path-based ASSET paths (VFS; no absolute server paths) |
| `has_asset(name)` | Whether a path asset exists |
| `content_assets()` | SHA-256 ids of content-addressed assets (`assets/{id}` sections) |
| `has_content_asset(id)` | Whether a content asset id exists |
| `content_asset(id)` | Bytes as `Image` when kind/mime is image, otherwise `ByteBuffer` |
| `metadata()` | Custom DCP metadata as object (string values) |
| `metadata_get(key)` | Single metadata value or `null` |
| `package_info()` | Safe summary counts (includes `content_asset_count`) |

**Column filter for `source_table(name, columns)`** (stricter than `read(..., header=...)`):

- `columns` omitted or `null` — all columns
- Array of strings — select columns in array order; **error** if a name is missing
- Object — select only listed keys; string value renames, `null` keeps the name; **error** if a key is missing

**Content-addressed assets:** table cells may store UTF-8 refs `asset://{sha256}`. Bytes live in ASSET sections named `assets/{sha256}` (deduped by hash). Path-based VFS assets (for `read(path(...))`) remain separate. With `--build_model`, content assets are written to SQLite table `__assets` (`id`, `kind`, `mime_type`, `filename`, `size`, `data`); business columns keep the `asset://` text.

Outside an active DCP WebSocket session, `ws.*` calls fail with `ws: no active DCP session`.

Embed Arrow tables when building a package (Python):

```python
import pyarrow as pa
from datacode_dcp import DCPEncoder

table = pa.table({"id": [1, 2], "value": [10.0, 20.0]})
DCPEncoder().code(code).table("orders", table).write(buf)

# Or with content assets (bytes/paths → asset:// refs):
# enc.table_with_assets("employees", rows, asset_columns={"avatar": "image"})
```

See [`examples/en/07-websocket/dc/ws_source_table.dc`](../../../examples/en/07-websocket/dc/ws_source_table.dc).

### SMB control (JSON text)

SMB operations still use JSON text messages with a `type` field:

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

After `smb_connect`, run DCP code that uses `lib://share_name/path`.

## Python client example

```python
import asyncio
import io
import json
import websockets
from datacode_dcp import DCPEncoder

async def run():
    buf = io.BytesIO()
    DCPEncoder().code("print('Hello from DCP')").write(buf)

    async with websockets.connect("ws://127.0.0.1:8899") as ws:
        await ws.send(buf.getvalue())          # binary frame
        result = json.loads(await ws.recv())   # JSON response
        print(result["output"])

asyncio.run(run())
```

See [`examples/en/07-websocket/python/test_dcp_run.py`](../../../examples/en/07-websocket/python/test_dcp_run.py) for a full example with assets.

## Features

1. **DCP packages** — code, metadata, Arrow tables, and files in one binary payload
2. **In-memory VFS** — DCP assets without disk temp directories
3. **`ws` module** — `source_table()` and safe package introspection during execution
4. **Output capture** — `print()` output in the `output` field
5. **SMB** — per-client SMB connections via JSON control messages
6. **Custom routes** — `@ws_route` in `ws_app.dc`

## Security

⚠️ No authentication or rate limiting. Do not expose on public networks without additional protection.

---

**See also:**
- [WebSocket examples](../../../examples/en/07-websocket/)
- [DCP-python README](../../../../DCP-python/README.md)
