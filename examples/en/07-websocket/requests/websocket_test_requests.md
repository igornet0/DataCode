# WebSocket test requests (DCP)

Server must be running:

```bash
datacode --websocket --host 127.0.0.1 --port 8899
```

## DCP execution (binary frame)

Use the Python helper:

```bash
python3 examples/en/07-websocket/python/test_dcp_run.py
```

Or build a package manually:

```bash
dcp create -o /tmp/job.dcp --code examples/en/07-websocket/dc/upload_data.dc --assets-dir examples/en/07-websocket/data
```

Send `/tmp/job.dcp` bytes as a **binary** WebSocket frame. Response is JSON text.

## Custom route (JSON text)

```json
{"type": "ping"}
```

Requires `ws_app.dc` bootstrap (see `examples/en/07-websocket/dc/ws_app.dc`).

## SMB connect (JSON text)

```json
{
  "type": "smb_connect",
  "ip": "192.168.1.100",
  "login": "user",
  "password": "pass",
  "domain": "WORKGROUP",
  "share_name": "data"
}
```
