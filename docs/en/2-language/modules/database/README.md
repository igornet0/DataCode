# Database engine module

The built-in `database_engine` module provides engine and cluster types for connecting to databases (SQLite in MVP) and running SQL.

## Import

```datacode
from database_engine import engine, DatabaseCluster, MetaData, Column, select
```

## Engine

- **engine(url, echo?, echo_pool?, pool_size?, max_overflow?, timeout?, connect_args?)**  
  Creates a database engine. URL scheme determines backend (`sqlite://` for SQLite).

- **Methods on engine** (e.g. `conn.execute(...)`):
  - **connect()** – returns connection (for SQLite, same as engine)
  - **execute(sql, params?)** – execute SQL, return row count
  - **query(sql, params?)** – execute SELECT, return Table (SQLite: typed cells via `_datacode_schema` / declared types)
  - **run(callable_or_instance)** – create_all (DDL), model instance (INSERT), select(Model) (SELECT)

For SQLite, ORM `create_all` and `--build_model` export write `_datacode_schema` / `_datacode_version` alongside user tables. String cells that look like ISO dates/datetimes or numbers are auto-detected; after `__sql__`, metadata is resynced from PRAGMA.

## Introspection

Catalog methods are the same on every SQL backend. SQLite, PostgreSQL, MySQL and MSSQL each implement them against their own system catalogs.

- **schemas()** – catalog schemas (SQLite: `main` / `temp` / attached DBs; PostgreSQL: user schemas; MySQL: databases; MSSQL: schemas)
- **tables(schema?)** – user tables as objects `{ name, schema, type }`. Omit `schema` to use the backend default (`main`, `public`, current database, `dbo`)
- **views(schema?)** – views in the same shape
- **columns(table, schema?)** – `{ name, type, nullable, default, datacode_type? }`
- **indexes(table, schema?)** – `{ name, columns, unique, primary }`
- **primary_key(table, schema?)** – primary-key index object, or `null`
- **foreign_keys(table, schema?)** – `{ name, columns, referenced_table, referenced_schema, referenced_columns }`
- **inspect()** – full tree: `inspect.schemas[].tables[]` / `views[]` with nested columns, indexes, primary_key, foreign_keys
- **table(name, schema?)** – `SELECT *` into a Datacode `Table`

System tables (`sqlite_*`, `_datacode_*`) are omitted from `tables()`.

```datacode
from database_engine import engine

conn = engine("sqlite:///:memory:").connect()
conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", [])

for t in conn.tables() {
    print(t.name)
}

for c in conn.columns("users") {
    print(c.name, c["type"], c.nullable)
}

users = conn.table("users")
info = conn.inspect()
```

## DatabaseCluster

A cluster holds named database connections so you can add several engines and use them by name.

- **DatabaseCluster()**  
  Creates an empty cluster.

- **cluster.add(name, engine)**  
  Adds a connection under the given name. Replaces existing connection with the same name.

- **cluster.add(engine)**  
  Adds a connection using the engine URL as the name.

- **cluster.get(name)**  
  Returns the engine for that name, or `null` if not found.

- **cluster.names()**  
  Returns an array of connection names.

### Example

```datacode
from database_engine import engine, DatabaseCluster

cluster = DatabaseCluster()
cluster.add("main", engine("sqlite:///main.db"))
cluster.add("replica", engine("sqlite:///replica.db"))

conn = cluster.get("main")
conn.execute("CREATE TABLE IF NOT EXISTS t (id INT)")
cluster.get("main").query("SELECT * FROM t")

names = cluster.names()   # ["main", "replica"]
```

## MetaData, Column, select

Used for ORM-style model definitions and `metadata.create_all(engine)`; see data model creation examples.
