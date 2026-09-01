from core.database.adapters.base import BaseAdapter

cls ClickhouseAdapter extends BaseAdapter {
    fn connect(config) {
        # clickhouse connection logic
        raise NotImplementedError()
    }
}
