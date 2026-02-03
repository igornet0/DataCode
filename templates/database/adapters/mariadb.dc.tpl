from core.database.adapters.base import BaseAdapter

cls MariadbAdapter extends BaseAdapter {
    fn connect(config) {
        # mariadb connection logic (same as MySQL)
        raise NotImplementedError()
    }
}
