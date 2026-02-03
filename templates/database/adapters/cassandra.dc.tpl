from core.database.adapters.base import BaseAdapter

cls CassandraAdapter extends BaseAdapter {
    fn connect(config) {
        # cassandra connection logic (keyspace)
        raise NotImplementedError()
    }
}
