# core/database package: config, engine, connection, models, adapters.
# Run scripts with base_dir = project root (where core/ lives) so that
# "from core.database.config import database" resolves.

from core.database.config import database
from core.database.engine import create_engine
from core.database.connection import get_connection
from core.database import models
