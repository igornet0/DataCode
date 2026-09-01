from core.database.engine import create_engine

# Connection helper: call create_engine() to get a connected engine/connection.
# Use it in your app entry point or request handlers.

fn get_connection() {
    return create_engine()
}
