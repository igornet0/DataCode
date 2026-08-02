# DataSource: загрузка из БД

Примеры чтения данных через `datasource` из разных СУБД.

| Файл | БД | Требования |
|------|-----|------------|
| `01-postgresql.dc` | PostgreSQL | сервер на localhost:5432 |
| `02-mysql.dc` | MySQL | сервер на localhost:3306 |
| `03-sqlite.dc` | SQLite | нет (in-memory) |
| `04-mongodb.dc` | MongoDB | сервер на localhost:27017 |
| `05-mssql.dc` | Microsoft SQL Server | сервер на localhost:1433 |

Общий поток: `datasource` → `connect` → `get_table` / `send_table` → `disconnect`.

Для MongoDB документы нормализуются в таблицу (flatten вложенных объектов, `array_mode`).

```bash
datacode examples/ru/15-datasource/04-database/03-sqlite.dc
```

Параметры можно задавать на верхнем уровне или во вложенном `connection: { ... }` (те же ключи). Алиасы: `uri` = `url`, `user` = `username`.

---

## Общие опциональные поля (все типы)

| Параметр | Обязательный | По умолчанию | Описание |
|----------|--------------|--------------|----------|
| `name` | нет | — | Отображаемое имя datasource |
| `description` | нет | — | Текстовое описание |
| `enabled` | нет | `true` | Если `false`, источник отключён |
| `timeout` | нет | зависит от драйвера | Общий таймаут в секундах |
| `connect_timeout` | нет | — | Таймаут подключения (секунды) |
| `read_timeout` | нет | — | Таймаут чтения (секунды) |
| `retry_count` | нет | `0` | Число повторных попыток |
| `options` | нет | `{}` | Доп. опции драйвера |
| `connection` | нет | — | Вложенный объект; ключи сливаются с верхним уровнем |

---

## PostgreSQL (`type: "postgresql"` или `"postgres"`)

**Обязательно:** полный `url` / `uri` **или** набор полей для сборки URL (`host` + обычно `database`; учётные данные по необходимости).

| Параметр | Обязательный | По умолчанию | Описание |
|----------|--------------|--------------|----------|
| `type` | **да** | — | `"postgresql"` или `"postgres"` |
| `url` / `uri` | **да\*** | — | например `postgres://user:pass@host:5432/dbname` |
| `host` | **да\*** | `localhost` | Используется, если нет `url` |
| `port` | нет | `5432` | Используется, если нет `url` |
| `database` | рекомендуется\* | `""` | Имя БД в пути URL при сборке из частей |
| `username` / `user` | нет | — | Вставляется в URL при сборке |
| `password` | нет | — | Вставляется в URL при сборке |
| `schema` | нет | — | Парсится; в сессию автоматически не применяется |
| `timeout` | нет | — | Подсказка statement timeout (секунды) |

\* Нужен **`url`/`uri`**, либо **`host`** (и обычно `database` / учётные данные) для сборки URL.

```datacode
datasource({
    type: "postgresql",
    url: "postgres://postgres:postgres@localhost:5432/datacode",
    timeout: 10
})

# или без полного URL:
datasource({
    type: "postgresql",
    host: "localhost",
    port: 5432,
    database: "datacode",
    username: "postgres",
    password: "postgres"
})
```

---

## MySQL (`type: "mysql"` или `"mariadb"`)

**Обязательно:** полный `url` / `uri` **или** поля для сборки URL.

| Параметр | Обязательный | По умолчанию | Описание |
|----------|--------------|--------------|----------|
| `type` | **да** | — | `"mysql"` или `"mariadb"` |
| `url` / `uri` | **да\*** | — | например `mysql://user:pass@host:3306/dbname` |
| `host` | **да\*** | `localhost` | Используется, если нет `url` |
| `port` | нет | `3306` | Используется, если нет `url` |
| `database` | рекомендуется\* | `""` | Имя схемы/базы |
| `username` / `user` | нет | — | Пользователь |
| `password` | нет | — | Пароль |
| `timeout` | нет | — | Таймаут чтения/записи (секунды) |

```datacode
datasource({
    type: "mysql",
    url: "mysql://root:root@localhost:3306/datacode",
    timeout: 10
})
```

---

## SQLite (`type: "sqlite"` или `"sql"`)

**Обязательно:** одно из `url`, `database` или `path`, указывающее на SQLite.

| Параметр | Обязательный | По умолчанию | Описание |
|----------|--------------|--------------|----------|
| `type` | **да** | — | `"sqlite"` или `"sql"` |
| `url` | **да\*** | — | `sqlite:///:memory:` или `sqlite:///./app.db` |
| `database` | **да\*** | — | Путь к файлу или URL `sqlite:…` (альтернатива `url`) |
| `path` | **да\*** | — | Путь к файлу → `sqlite:///<path>` |
| `timeout` | нет | — | Сохраняется в конфиге engine |

\* Достаточно одного из **`url`**, **`database`**, **`path`**.

```datacode
datasource({ type: "sqlite", url: "sqlite:///:memory:" })
datasource({ type: "sqlite", url: "sqlite:///./app.db" })
datasource({ type: "sqlite", path: "./app.db" })
datasource({ type: "sqlite", database: "./app.db" })
```

---

## MongoDB (`type: "mongodb"` или `"mongo"`)

**Обязательно:** цель подключения (`url`/`uri` или части host) **и** `database`.  
`collection` нужна для `get_table` / `send_table` / count, если не передана в spec запроса.

| Параметр | Обязательный | По умолчанию | Описание |
|----------|--------------|--------------|----------|
| `type` | **да** | — | `"mongodb"` или `"mongo"` |
| `url` / `uri` | **да\*** | — | например `mongodb://localhost:27017` или `mongodb+srv://…` |
| `host` | **да\*** | `localhost` | Используется, если нет `url` |
| `port` | нет | `27017` | Используется, если нет `url` |
| `database` | **да** | — | Имя базы (обязательно при connect/query) |
| `collection` | рекомендуется\*\* | — | Коллекция по умолчанию; можно переопределить в вызове |
| `username` / `user` | нет | — | Добавляется в URI, если ещё нет |
| `password` | нет | — | Добавляется в URI вместе с username |
| `timeout` | нет | `30` | Таймаут выбора сервера / подключения (секунды) |
| `connect_timeout` | нет | как `timeout` | Отдельный таймаут подключения |

\* Нужен **`url`/`uri`**, либо **`host`** (+ опционально port/auth/database).  
\*\* Обязателен в конфиге **или** в `get_table` / `send_table` / `request` через `collection`.

```datacode
datasource({
    type: "mongodb",
    url: "mongodb://localhost:27017",
    database: "datacode",
    collection: "users",
    timeout: 10
})

# вложенная форма:
datasource({
    type: "mongodb",
    connection: {
        uri: "mongodb://localhost:27017",
        database: "datacode",
        collection: "users"
    }
})
```

---

## Microsoft SQL Server (`type: "mssql"` или `"sqlserver"`)

**Обязательно:** полный `url` / `uri` **или** поля для сборки.  
Формат URL: `mssql://user:pass@host:1433/database` (также `sqlserver://…`).

| Параметр | Обязательный | По умолчанию | Описание |
|----------|--------------|--------------|----------|
| `type` | **да** | — | `"mssql"` или `"sqlserver"` |
| `url` / `uri` | **да\*** | — | например `mssql://sa:Password@localhost:1433/master` |
| `host` | **да\*** | `localhost` | Используется, если нет `url` |
| `port` | нет | `1433` | Используется, если нет `url` |
| `database` | рекомендуется\* | `""` | Имя базы |
| `username` / `user` | рекомендуется\* | — | Пользователь SQL Server (обычно нужен в URL) |
| `password` | рекомендуется\* | — | Пароль SQL Server |
| `timeout` | нет | — | Таймаут подключения (секунды) |

\* Нужен **`url`/`uri`**, либо **`host`** + учётные данные + `database`.

```datacode
datasource({
    type: "mssql",
    url: "mssql://sa:Your_password123@localhost:1433/master",
    timeout: 10
})
```
