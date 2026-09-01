from core.database.adapters.base import BaseAdapter

cls PostgresAdapter extends BaseAdapter {
    fn connect(config) {
        # postgres connection logic
        raise NotImplementedError()
    }
}
