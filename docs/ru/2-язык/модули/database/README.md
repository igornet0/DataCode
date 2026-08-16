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
