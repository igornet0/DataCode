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

The DCP package must contain a `CODE` section (`__code__`). Optional `ASSET` sections (relative paths) are written into the client session before execution.

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

1. **DCP packages** — code, metadata, and files in one binary payload
2. **Session isolation** — per-client temp directory for assets
3. **Output capture** — `print()` output in the `output` field
4. **SMB** — per-client SMB connections via JSON control messages
5. **Custom routes** — `@ws_route` in `ws_app.dc`

## Security

⚠️ No authentication or rate limiting. Do not expose on public networks without additional protection.

---

**See also:**
- [WebSocket examples](../../../examples/en/07-websocket/)
- [DCP-python README](../../../../DCP-python/README.md)
