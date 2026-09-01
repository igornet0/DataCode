from core.database.adapters.base import BaseAdapter
from core.database.adapters.postgres import PostgresAdapter
from core.database.adapters.sqlite import SqliteAdapter
from core.database.adapters.mysql import MysqlAdapter
from core.database.adapters.mariadb import MariadbAdapter
from core.database.adapters.mongodb import MongodbAdapter
from core.database.adapters.mssql import MssqlAdapter
from core.database.adapters.oracle import OracleAdapter
from core.database.adapters.couchdb import CouchdbAdapter
from core.database.adapters.redis import RedisAdapter
from core.database.adapters.memcached import MemcachedAdapter
from core.database.adapters.neo4j import Neo4jAdapter
from core.database.adapters.arangodb import ArangodbAdapter
from core.database.adapters.clickhouse import ClickhouseAdapter
from core.database.adapters.cassandra import CassandraAdapter
from core.database.adapters.influxdb import InfluxdbAdapter

fn get_adapter(db_type, driver) {
    if db_type == "postgresql" {
        return PostgresAdapter()
    }
    if db_type == "sqlite" {
        return SqliteAdapter()
    }
    if db_type == "mysql" {
        return MysqlAdapter()
    }
    if db_type == "mariadb" {
        return MariadbAdapter()
    }
    if db_type == "mongodb" {
        return MongodbAdapter()
    }
    if db_type == "mssql" {
        return MssqlAdapter()
    }
    if db_type == "oracle" {
        return OracleAdapter()
    }
    if db_type == "couchdb" {
        return CouchdbAdapter()
    }
    if db_type == "redis" {
        return RedisAdapter()
    }
    if db_type == "memcached" {
        return MemcachedAdapter()
    }
    if db_type == "neo4j" {
        return Neo4jAdapter()
    }
    if db_type == "arangodb" {
        return ArangodbAdapter()
    }
    if db_type == "clickhouse" {
        return ClickhouseAdapter()
    }
    if db_type == "cassandra" {
        return CassandraAdapter()
    }
    if db_type == "influxdb" {
        return InfluxdbAdapter()
    }
    raise NotImplementedError()
}
