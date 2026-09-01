from core.database.adapters.base import BaseAdapter

cls MemcachedAdapter extends BaseAdapter {
    fn connect(config) {
        # memcached connection logic (often no auth)
        raise NotImplementedError()
    }
}
