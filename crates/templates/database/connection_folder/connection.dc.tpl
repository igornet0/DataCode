from core.database.{{ connection_module }}.engine import create_engine

fn get_connection() {
    return create_engine()
}
