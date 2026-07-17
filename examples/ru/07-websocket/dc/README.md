# Скрипты DataCode для WebSocket

| Файл | Назначение |
|------|------------|
| `ws_app.dc` | Приложение сервера: `@ws_route`, политика `execute` |
| `upload_data.dc` | Обработка загруженных файлов (`--use-ve`) |
| `test_smb_load_data.dc` | Чтение файлов с SMB-шары (`lib://`) |

Запуск приложения:

```bash
datacode examples/ru/07-websocket/dc/ws_app.dc --websocket --host 127.0.0.1 --port 8899
```
