# Примеры: 07 — WebSocket

Удалённое выполнение кода DataCode по WebSocket.

**Папка с примерами:** [`examples/ru/07-websocket/`](../../../examples/ru/07-websocket/)

## Структура

| Папка | Назначение |
|-------|------------|
| `dc/` | Скрипты DataCode (`ws_app.dc`, примеры обработки данных) |
| `python/` | Клиенты на Python |
| `node/` | Клиент на Node.js |
| `bash/` | Тест через websocat |
| `html/` | Клиент в браузере |

## Быстрый старт

```bash
# 1. Сервер
datacode --websocket --host 127.0.0.1 --port 8899

# 2. Клиент
cd examples/ru/07-websocket/python
pip install -r requirements.txt
python3 test_websocket.py
```

## Сценарии

| Задача | Команда сервера | Клиент |
|--------|-----------------|--------|
| Выполнить код | `--websocket --port 8899` | `python/test_websocket.py` |
| Свои маршруты | `dc/ws_app.dc --websocket --port 8899` | `{"type":"ping"}` |
| Загрузка файлов | `… --use-ve` | `python/test_file_upload.py` + `dc/upload_data.dc` |
| SMB-шара | `… --port 8899` | `python/test_smb_connection.py` |

Документация протокола: [websocket-сервер](../2-язык/сервисы/websocket-сервер.md)
