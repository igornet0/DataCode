# DataSource: load from databases

Examples of reading data via `datasource` from different databases.

| File | DB | Requirements |
|------|-----|--------------|
| `01-postgresql.dc` | PostgreSQL | server on localhost:5432 |
| `02-mysql.dc` | MySQL | server on localhost:3306 |
| `03-sqlite.dc` | SQLite | none (in-memory) |
| `04-mongodb.dc` | MongoDB | server on localhost:27017 |
| `05-mssql.dc` | Microsoft SQL Server | server on localhost:1433 |

Flow: `datasource` → `connect` → `get_table` / `send_table` → `disconnect`.

For MongoDB, documents are normalized into a table (nested object flatten, `array_mode`).

```bash
datacode examples/en/15-datasource/04-database/03-sqlite.dc
```

You can pass fields at the top level or nest them under `connection: { ... }` (same keys). Alias: `uri` = `url`, `user` = `username`.

---

## Common optional fields (all types)

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `name` | no | — | Display name for the datasource |
| `description` | no | — | Free-text description |
| `enabled` | no | `true` | If `false`, source is disabled |
| `timeout` | no | driver-dependent | Overall timeout in seconds |
| `connect_timeout` | no | — | Connect timeout (seconds) |
| `read_timeout` | no | — | Read timeout (seconds) |
| `retry_count` | no | `0` | Retry attempts |
| `options` | no | `{}` | Extra driver options object |
| `connection` | no | — | Nested object; keys merge into top-level config |

---

## PostgreSQL (`type: "postgresql"` or `"postgres"`)

**Required:** either a full `url` / `uri`, **or** enough pieces to build one (`host` + usually `database`; auth as needed).

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `type` | **yes** | — | `"postgresql"` or `"postgres"` |
| `url` / `uri` | **yes\*** | — | e.g. `postgres://user:pass@host:5432/dbname` |
| `host` | **yes\*** | `localhost` | Used when `url` is omitted |
| `port` | no | `5432` | Used when `url` is omitted |
| `database` | recommended\* | `""` | DB name in URL path when building from parts |
| `username` / `user` | no | — | Embedded in URL when building from parts |
| `password` | no | — | Embedded in URL when building from parts |
| `schema` | no | — | Parsed; not applied automatically to session |
| `timeout` | no | — | Statement timeout hint (seconds) |

\* Provide **`url`/`uri`**, or **`host`** (and typically `database` / credentials) so a URL can be built.

```datacode
datasource({
    type: "postgresql",
    url: "postgres://postgres:postgres@localhost:5432/datacode",
    timeout: 10
})

# or without a full URL:
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

## MySQL (`type: "mysql"` or `"mariadb"`)

**Required:** either a full `url` / `uri`, **or** pieces to build one.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `type` | **yes** | — | `"mysql"` or `"mariadb"` |
| `url` / `uri` | **yes\*** | — | e.g. `mysql://user:pass@host:3306/dbname` |
| `host` | **yes\*** | `localhost` | Used when `url` is omitted |
| `port` | no | `3306` | Used when `url` is omitted |
| `database` | recommended\* | `""` | Schema/database name |
| `username` / `user` | no | — | Auth user |
| `password` | no | — | Auth password |
| `timeout` | no | — | Read/write timeout (seconds) |

```datacode
datasource({
    type: "mysql",
    url: "mysql://root:root@localhost:3306/datacode",
    timeout: 10
})
```

---

## SQLite (`type: "sqlite"` or `"sql"`)

**Required:** one of `url`, `database`, or `path` that resolves to a SQLite location.

Files produced by `--build_model` / `save_sqlite` include `_datacode_schema` and `_datacode_version`. On `get_table` / `query`, Datacode restores `date` / `duration` from that metadata (falling back to declared column types like `DATE` / `DATETIME`).

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `type` | **yes** | — | `"sqlite"` or `"sql"` |
| `url` | **yes\*** | — | `sqlite:///:memory:` or `sqlite:///./app.db` |
| `database` | **yes\*** | — | File path or `sqlite:…` URL (alternative to `url`) |
| `path` | **yes\*** | — | File path → becomes `sqlite:///<path>` |
| `timeout` | no | — | Stored on engine config |

\* Exactly one of **`url`**, **`database`**, or **`path`** is enough.

```datacode
datasource({ type: "sqlite", url: "sqlite:///:memory:" })
datasource({ type: "sqlite", url: "sqlite:///./app.db" })
datasource({ type: "sqlite", path: "./app.db" })
datasource({ type: "sqlite", database: "./app.db" })
```

---

## MongoDB (`type: "mongodb"` or `"mongo"`)

**Required:** connection target (`url`/`uri` or host parts) **and** `database`.  
`collection` is required for `get_table` / `send_table` / count unless passed in the request spec.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `type` | **yes** | — | `"mongodb"` or `"mongo"` |
| `url` / `uri` | **yes\*** | — | e.g. `mongodb://localhost:27017` or `mongodb+srv://…` |
| `host` | **yes\*** | `localhost` | Used when `url` is omitted |
| `port` | no | `27017` | Used when `url` is omitted |
| `database` | **yes** | — | Database name (required at connect/query time) |
| `collection` | recommended\*\* | — | Default collection; can override per call |
| `username` / `user` | no | — | Injected into URI if not already present |
| `password` | no | — | Injected into URI with username |
| `timeout` | no | `30` | Server selection / connect timeout (seconds) |
| `connect_timeout` | no | same as `timeout` | Connect timeout override |

\* Provide **`url`/`uri`**, or **`host`** (+ optional port/auth/database) to build a URL.  
\*\* Required on config **or** in `get_table` / `send_table` / `request` via `collection`.

```datacode
datasource({
    type: "mongodb",
    url: "mongodb://localhost:27017",
    database: "datacode",
    collection: "users",
    timeout: 10
})

# nested form:
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

## Microsoft SQL Server (`type: "mssql"` or `"sqlserver"`)

**Required:** either a full `url` / `uri`, **or** pieces to build one.  
URL shape when built or passed: `mssql://user:pass@host:1433/database` (also accepts `sqlserver://…`).

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `type` | **yes** | — | `"mssql"` or `"sqlserver"` |
| `url` / `uri` | **yes\*** | — | e.g. `mssql://sa:Password@localhost:1433/master` |
| `host` | **yes\*** | `localhost` | Used when `url` is omitted |
| `port` | no | `1433` | Used when `url` is omitted |
| `database` | recommended\* | `""` | Database name |
| `username` / `user` | recommended\* | — | SQL Server auth user (required in URL for typical setups) |
| `password` | recommended\* | — | SQL Server auth password |
| `timeout` | no | — | Connect-related timeout (seconds) |

\* Provide **`url`/`uri`**, or **`host`** + credentials + `database`.

```datacode
datasource({
    type: "mssql",
    url: "mssql://sa:Your_password123@localhost:1433/master",
    timeout: 10
})
```
