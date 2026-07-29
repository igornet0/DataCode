# Тестовые запросы WebSocket (DCP)

Сервер должен быть запущен:

```bash
datacode --websocket --host 127.0.0.1 --port 8899
```

## Выполнение DCP (binary frame)

```bash
python3 examples/ru/07-websocket/python/test_dcp_run.py
```

Или вручную:

```bash
dcp create -o /tmp/job.dcp --code examples/ru/07-websocket/dc/upload_data.dc --assets-dir examples/ru/07-websocket/data
```

Отправьте байты `/tmp/job.dcp` как **binary** WebSocket frame. Ответ — JSON text.

## Пользовательский маршрут (JSON text)

```json
{"type": "ping"}
```

Нужен `ws_app.dc` (см. `examples/ru/07-websocket/dc/ws_app.dc`).

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
