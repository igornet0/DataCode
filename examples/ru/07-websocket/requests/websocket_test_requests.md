# Шпаргалка: ручное тестирование WebSocket

Полное руководство — в [README.md](../README.md).

## Запуск сервера

```bash
datacode --websocket --host 127.0.0.1 --port 8899
```

## wscat

```bash
npm install -g wscat
wscat -c ws://127.0.0.1:8899
```

```json
{"type": "execute", "code": "print('Привет!')"}
{"type": "ping"}
```

## websocat

```bash
cargo install websocat
echo '{"type":"execute","code":"print(\"ok\")"}' | websocat ws://127.0.0.1:8899
```

## Готовые запросы

См. [`websocket_requests.json`](./websocket_requests.json).

## Автотесты

```bash
cd ../python
pip install -r requirements.txt
python3 test_websocket.py
python3 test_file_upload.py      # сервер с --use-ve
python3 test_smb_connection.py   # по умолчанию ../dc/test_smb_load_data.dc
```
