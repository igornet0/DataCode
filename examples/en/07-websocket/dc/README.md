# DataCode scripts for WebSocket

| File | Purpose |
|------|---------|
| `ws_app.dc` | Server application: `@ws_route`, `execute` policy |
| `upload_data.dc` | Process uploaded files (`--use-ve`) |
| `test_smb_load_data.dc` | Read files from SMB share (`lib://`) |

Run the application:

```bash
datacode examples/en/07-websocket/dc/ws_app.dc --websocket --host 127.0.0.1 --port 8899
```
