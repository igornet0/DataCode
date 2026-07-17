# Python WebSocket clients

```bash
pip install -r requirements.txt
python3 test_websocket.py
```

| Script | Purpose |
|--------|---------|
| `test_websocket.py` | Basic `execute` tests |
| `test_file_upload.py` | File upload (server with `--use-ve`) |
| `test_smb_connection.py` | SMB + run `../dc/test_smb_load_data.dc` |
| `test_websocket_sqlite_export.py` | SQLite export (`--build_model`) |

Server: `datacode --websocket --host 127.0.0.1 --port 8899`
