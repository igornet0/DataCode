# Python-клиенты WebSocket

```bash
pip install -r requirements.txt
python3 test_websocket.py
```

| Скрипт | Назначение |
|--------|------------|
| `test_websocket.py` | Базовые тесты `execute` |
| `test_file_upload.py` | Загрузка файлов (сервер с `--use-ve`) |
| `test_smb_connection.py` | SMB + выполнение `../dc/test_smb_load_data.dc` |
| `test_websocket_sqlite_export.py` | Экспорт SQLite (`--build_model`) |

Сервер: `datacode --websocket --host 127.0.0.1 --port 8899`
