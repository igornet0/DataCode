# Cheat sheet: manual WebSocket testing

Full guide — in [README.md](../README.md).

## Start server

```bash
datacode --websocket --host 127.0.0.1 --port 8899
```

## wscat

```bash
npm install -g wscat
wscat -c ws://127.0.0.1:8899
```

```json
{"type": "execute", "code": "print('Hello!')"}
{"type": "ping"}
```

## websocat

```bash
cargo install websocat
echo '{"type":"execute","code":"print(\"ok\")"}' | websocat ws://127.0.0.1:8899
```

## Ready-made requests

See [`websocket_requests.json`](./websocket_requests.json).

## Automated tests

```bash
cd ../python
pip install -r requirements.txt
python3 test_websocket.py
python3 test_file_upload.py      # server with --use-ve
python3 test_smb_connection.py   # default ../dc/test_smb_load_data.dc
```
