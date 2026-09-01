from core.database.adapters.base import BaseAdapter

cls MongodbAdapter extends BaseAdapter {
    fn connect(config) {
        # mongodb connection logic
        raise NotImplementedError()
    }
}
