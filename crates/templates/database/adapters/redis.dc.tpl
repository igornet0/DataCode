from core.database.adapters.base import BaseAdapter

cls RedisAdapter extends BaseAdapter {
    fn connect(config) {
        # redis connection logic (often no auth)
        raise NotImplementedError()
    }
}
