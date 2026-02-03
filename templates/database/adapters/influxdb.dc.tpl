from core.database.adapters.base import BaseAdapter

cls InfluxdbAdapter extends BaseAdapter {
    fn connect(config) {
        # influxdb connection logic
        raise NotImplementedError()
    }
}
