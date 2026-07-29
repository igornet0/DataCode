# WebSocket в DataCode

Удалённое выполнение DataCode через **DCP-пакеты** по WebSocket: клиент отправляет binary frame с `.dcp`, сервер декодирует, извлекает файлы, выполняет код и возвращает JSON.

**Документация:** [`docs/ru/2-язык/сервисы/websocket-сервер.md`](../../../docs/ru/2-язык/сервисы/websocket-сервер.md) · [DCP-python](../../../DCP-python/README.md)

---

## Быстрый старт

```bash
# Сервер
datacode --websocket --host 127.0.0.1 --port 8899 --build_model

# Клиент (assets читаются из in-memory VFS, temp_sessions не создаётся)
pip install websockets
pip install git+https://github.com/igornet0/DCP-python.git
python3 python/test_dcp_run.py
```

Сборка пакета вручную:
```bash
dcp create -o job.dcp --code dc/upload_data.dc --assets-dir data
```

Протоколы JSON `execute` / `upload_file` **удалены** — используйте DCP.

---

## Структура папки

```
07-websocket/
├── dc/          # скрипты DataCode (серверное приложение и примеры)
├── python/      # клиенты на Python
├── node/        # клиент на Node.js
├── bash/        # тест через websocat
├── html/        # интерактивный клиент в браузере
├── requests/    # готовые JSON-запросы и шпаргалка
└── data/        # тестовые CSV, XLSX, TXT для загрузки
```

| Папка | Содержимое |
|-------|------------|
| [`dc/`](./dc/) | `ws_app.dc`, `upload_data.dc`, `test_smb_load_data.dc` |
| [`python/`](./python/) | `test_dcp_run.py`, `test_smb_connection.py`, … |
| [`node/`](./node/) | `test_websocket.js` |
| [`bash/`](./bash/) | `test_websocket.sh` |
| [`html/`](./html/) | `websocket_client_example.html` |
| [`requests/`](./requests/) | `websocket_requests.json`, шпаргалка для wscat |
| [`data/`](./data/) | Примеры файлов для `upload_file` |

---

## Быстрый старт

### 1. Запустите сервер

```bash
# из корня репозитория
datacode --websocket --host 127.0.0.1 --port 8899
```

Сервер слушает `ws://127.0.0.1:8899`. Каждое подключение — отдельная сессия интерпретатора.

### 2. Отправьте запрос (Python)

```bash
cd examples/ru/07-websocket/python
pip install -r requirements.txt
python3 test_websocket.py
```

**wscat** (ручная проверка):

```bash
npm install -g wscat
wscat -c ws://127.0.0.1:8899
```

```json
{"type": "execute", "code": "print('Привет из WebSocket!')"}
```

Ожидаемый ответ:

```json
{"success": true, "output": "Привет из WebSocket!\n", "error": null}
```

### 3. Скрипт приложения (опционально)

```bash
datacode examples/ru/07-websocket/dc/ws_app.dc --websocket --host 127.0.0.1 --port 8899 --use-ve
```

Проверка маршрута: `{"type": "ping"}` → `{"success": true, "message": "pong"}`.

---

## Протокол (кратко)

Все сообщения — JSON. Поле `type` задаёт операцию.

### `execute` — выполнить код

```json
{
  "type": "execute",
  "code": "global x = 10\nprint('x =', x)"
}
```

| Поле ответа | Описание |
|-------------|----------|
| `success` | `true` / `false` |
| `output` | Вывод всех `print()` |
| `error` | Текст ошибки или `null` |

### `upload_file` — загрузить файл (нужен `--use-ve`)

```json
{
  "type": "upload_file",
  "filename": "report.csv",
  "content": "id,name\n1,Alice"
}
```

Бинарные файлы: `"content": "base64:<данные в base64>"`.

### `smb_connect` — подключить SMB-шару

```json
{
  "type": "smb_connect",
  "ip": "192.168.1.100",
  "login": "user",
  "password": "secret",
  "domain": "WORKGROUP",
  "share_name": "data"
}
```

После подключения в скриптах используйте пути `lib://share_name/path/to/file`.

---

## Режимы запуска

| Команда | Когда использовать |
|---------|-------------------|
| `datacode --websocket --port 8899` | Минимальный сервер |
| `datacode dc/ws_app.dc --websocket --port 8899` | Свои маршруты + политика `execute` |
| `… --use-ve` | Изолированная сессия, загрузка файлов |
| `… --build_model` | Экспорт SQLite (см. `python/test_websocket_sqlite_export.py`) |

---

## Сценарии

### A. Выполнение кода

```bash
# терминал 1
datacode --websocket --port 8899

# терминал 2
cd examples/ru/07-websocket/python
python3 test_websocket.py
```

### B. Загрузка и обработка файлов

```bash
# терминал 1
datacode --websocket --port 8899 --use-ve

# терминал 2
cd examples/ru/07-websocket/python
python3 test_file_upload.py
```

Скрипт `dc/upload_data.dc` перебирает файлы в `data_dir/` после загрузки.

### C. Пользовательские маршруты

```bash
datacode examples/ru/07-websocket/dc/ws_app.dc --websocket --port 8899
```

```json
{"type": "version"}
{"type": "health"}
```

### D. Работа с SMB-шарой

1. Учётные данные — в `python/test_smb_connection.py`.
2. `SHARE_NAME` — в `dc/test_smb_load_data.dc`.
3. Запуск:

```bash
datacode --websocket --port 8899
cd examples/ru/07-websocket/python
python3 test_smb_connection.py
```

### E. Node.js

```bash
cd examples/ru/07-websocket/node
npm install
node test_websocket.js
```

### F. Браузер

Откройте [`html/websocket_client_example.html`](./html/websocket_client_example.html) (адрес: `ws://127.0.0.1:8899`).

### G. Bash (websocat)

```bash
cd examples/ru/07-websocket/bash
cargo install websocat   # если ещё не установлен
bash test_websocket.sh
```

---

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| `DATACODE_WS_ADDRESS` | Адрес по умолчанию, например `0.0.0.0:8899` |
| `DATACODE_WS_URL` | URL для Python-клиентов SMB, по умолчанию `ws://127.0.0.1:8899` |

---

## Безопасность

- Нет встроенной аутентификации — не выставляйте сервер в интернет без прокси и TLS (WSS).
- Пароли SMB передаются в JSON открытым текстом.
- Для продакшена: `dc/ws_app.dc` с `execute_policy: "restricted"`.

---

## См. также

- [Документация WebSocket-сервера](../../../docs/ru/2-язык/сервисы/websocket-сервер.md)
- [Примеры в docs/ru/1-примеры](../../../docs/ru/1-примеры/07-websocket.md)
- [Шпаргалка запросов](./requests/websocket_test_requests.md)
