# DatabaseCluster

This section shows how to use the `DatabaseCluster` type from the `database` module to manage multiple named database connections and run SQL on them by name.

## Files

- **[`01-cluster-example.dc`](01-cluster-example.dc)** – Create a cluster, add two SQLite engines (in-memory), get connections by name, run DDL and queries.

## Usage

```datacode
from database import engine, DatabaseCluster

cluster = DatabaseCluster()
cluster.add("main", engine("sqlite:///main.db"))
cluster.add("replica", engine("sqlite:///replica.db"))

conn = cluster.get("main")
conn.execute("CREATE TABLE IF NOT EXISTS t (id INT)")
cluster.get("main").query("SELECT * FROM t")

names = cluster.names()   # ["main", "replica"]
```

See [docs/en/database/README.md](../../../docs/en/database/README.md) for full module documentation.
