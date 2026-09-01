from core.database.adapters.base import BaseAdapter

cls CouchdbAdapter extends BaseAdapter {
    fn connect(config) {
        # couchdb http connection logic
        raise NotImplementedError()
    }
}
