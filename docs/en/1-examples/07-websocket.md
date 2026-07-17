# Examples: 07 — WebSocket

Remote execution of DataCode code over WebSocket.

**Examples folder:** [`examples/en/07-websocket/`](../../../examples/en/07-websocket/)

## Structure

| Folder | Purpose |
|--------|---------|
| `dc/` | DataCode scripts (`ws_app.dc`, data processing examples) |
| `python/` | Python clients |
| `node/` | Node.js client |
| `bash/` | Testing via websocat |
| `html/` | Browser client |

## Quick start

```bash
# 1. Server
datacode --websocket --host 127.0.0.1 --port 8899

# 2. Client
cd examples/en/07-websocket/python
pip install -r requirements.txt
python3 test_websocket.py
```

## Scenarios

| Task | Server command | Client |
|------|----------------|--------|
| Execute code | `--websocket --port 8899` | `python/test_websocket.py` |
| Custom routes | `dc/ws_app.dc --websocket --port 8899` | `{"type":"ping"}` |
| File upload | `… --use-ve` | `python/test_file_upload.py` + `dc/upload_data.dc` |
| SMB share | `… --port 8899` | `python/test_smb_connection.py` |

Protocol documentation: [websocket-server](../2-language/services/websocket-server.md)
