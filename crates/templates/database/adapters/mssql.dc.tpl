from core.database.adapters.base import BaseAdapter

cls MssqlAdapter extends BaseAdapter {
    fn connect(config) {
        # mssql+pyodbc connection logic (e.g. driver=ODBC+Driver+17+for+SQL+Server)
        raise NotImplementedError()
    }
}
