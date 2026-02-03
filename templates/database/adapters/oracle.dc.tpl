from core.database.adapters.base import BaseAdapter

cls OracleAdapter extends BaseAdapter {
    fn connect(config) {
        # oracle+cx_oracle connection logic (service name e.g. orclpdb1)
        raise NotImplementedError()
    }
}
