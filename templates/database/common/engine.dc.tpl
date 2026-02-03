from core.database.config import database
from core.database.adapters import get_adapter

fn create_engine() {
    adapter = get_adapter(database.type, database.driver)
    return adapter.connect(database)
}
