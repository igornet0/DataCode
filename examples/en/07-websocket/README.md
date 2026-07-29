# WebSocket Examples for DataCode

Examples for the DataCode WebSocket server with **DCP packages** (code + files in one binary payload).

See [WebSocket server docs](../../../docs/en/2-language/services/websocket-server.md) and [DCP-python README](../../../DCP-python/README.md).

## Quick start

**Server:**
```bash
datacode --websocket --host 0.0.0.0 --port 8899 --build_model
```

Assets from DCP are read from **in-memory VFS** (no files written to `temp_sessions`).

**Client:**
```bash
pip install websockets
pip install git+https://github.com/igornet0/DCP-python.git
python3 python/test_dcp_run.py
```

**Build DCP manually:**
```bash
dcp create -o job.dcp --code dc/upload_data.dc --assets-dir data
# Send job.dcp as a binary WebSocket frame; response is JSON text
```

## Key files

| File | Description |
|------|-------------|
| `python/test_dcp_run.py` | Build DCP, send binary frame, print JSON result |
| `python/test_smb_connection.py` | SMB JSON control messages |
| `dc/ws_app.dc` | Optional `@ws_route` + `websocket.configure()` |
| `requests/websocket_test_requests.md` | Protocol cheat sheet |

JSON `execute` / `upload_file` are **removed** — use DCP instead.

Custom JSON routes (e.g. `{"type":"ping"}`) still work when `ws_app.dc` is loaded.

## SMB

SMB share setup still uses JSON text (`smb_connect`, etc.). Run DataCode from a DCP package that references `lib://share/path`.

## Other clients

Older scripts (`test_websocket.py`, `test_file_upload.py`, HTML client) target the legacy protocol and need updating for DCP. Use `test_dcp_run.py` as the reference client.
