from core.database.adapters.base import BaseAdapter

cls MysqlAdapter extends BaseAdapter {
    fn connect(config) {
        # mysql connection logic
        raise NotImplementedError()
    }
}
