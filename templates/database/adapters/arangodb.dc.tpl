from core.database.adapters.base import BaseAdapter

cls ArangodbAdapter extends BaseAdapter {
    fn connect(config) {
        # arangodb http connection logic (_db/name)
        raise NotImplementedError()
    }
}
