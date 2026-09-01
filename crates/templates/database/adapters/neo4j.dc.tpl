from core.database.adapters.base import BaseAdapter

cls Neo4jAdapter extends BaseAdapter {
    fn connect(config) {
        # neo4j connection logic
        raise NotImplementedError()
    }
}
