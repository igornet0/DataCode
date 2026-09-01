from core.database.adapters.base import BaseAdapter

cls SqliteAdapter extends BaseAdapter {
    fn connect(config) {
        # sqlite connection logic
        raise NotImplementedError()
    }
}
