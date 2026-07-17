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
  - **query(sql, params?)** – execute SELECT, return Table
  - **run(callable_or_instance)** – create_all (DDL), model instance (INSERT), select(Model) (SELECT)

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
