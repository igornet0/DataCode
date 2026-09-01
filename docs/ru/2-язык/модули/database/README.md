# Модуль database_engine

Встроенный модуль `database_engine` предоставляет типы движка и кластера для подключения к базам данных (в MVP — SQLite) и выполнения SQL.

## Импорт

```datacode
from database_engine import engine, DatabaseCluster, MetaData, Column, select

```

## Движок (engine)

- **engine(url, echo?, echo_pool?, pool_size?, max_overflow?, timeout?, connect_args?)**  
  Создаёт движок БД. Схема URL определяет бэкенд (`sqlite://` для SQLite).

- **Методы движка** (например `conn.execute(...)`):
  - **connect()** — возвращает соединение (для SQLite то же, что и движок)
  - **execute(sql, params?)** — выполнить SQL, вернуть число затронутых строк
  - **query(sql, params?)** — выполнить SELECT, вернуть Table (SQLite: типизация ячеек через `_datacode_schema` / declared types)
  - **run(callable_or_instance)** — create_all (DDL), экземпляр модели (INSERT), select(Model) (SELECT)

Для SQLite ORM `create_all` и экспорт `--build_model` пишут `_datacode_schema` / `_datacode_version` рядом с пользовательскими таблицами. Строковые ячейки с ISO date/datetime и числами определяются автоматически; после `__sql__` metadata синхронизируется с PRAGMA.

## Интроспекция

Методы каталога одинаковы для всех SQL-бэкендов. SQLite, PostgreSQL, MySQL и MSSQL реализуют их через свои системные каталоги.

- **schemas()** — схемы каталога (SQLite: `main` / `temp` / attached; PostgreSQL: пользовательские схемы; MySQL: базы; MSSQL: схемы)
- **tables(schema?)** — пользовательские таблицы как объекты `{ name, schema, type }`. Без `schema` берётся схема по умолчанию (`main`, `public`, текущая БД, `dbo`)
- **views(schema?)** — представления в том же формате
- **columns(table, schema?)** — `{ name, type, nullable, default, datacode_type? }`
- **indexes(table, schema?)** — `{ name, columns, unique, primary }`
- **primary_key(table, schema?)** — объект первичного ключа или `null`
- **foreign_keys(table, schema?)** — `{ name, columns, referenced_table, referenced_schema, referenced_columns }`
- **inspect()** — полное дерево: `inspect.schemas[].tables[]` / `views[]` с вложенными columns, indexes, primary_key, foreign_keys
- **table(name, schema?)** — `SELECT *` в Datacode-таблицу `Table`

Системные таблицы (`sqlite_*`, `_datacode_*`) в `tables()` не попадают.

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

Кластер хранит именованные подключения к БД: можно добавить несколько движков и обращаться к ним по имени.

- **DatabaseCluster()**  
  Создаёт пустой кластер.

- **cluster.add(name, engine)**  
  Добавляет подключение с указанным именем. Заменяет существующее подключение с тем же именем.

- **cluster.add(engine)**  
  Добавляет подключение, используя URL движка в качестве имени.

- **cluster.get(name)**  
  Возвращает движок для этого имени или `null`, если не найдено.

- **cluster.names()**  
  Возвращает массив имён подключений.

### Пример

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

Используются для объявления моделей в стиле ORM и `metadata.create_all(engine)`; см. примеры создания моделей данных.
