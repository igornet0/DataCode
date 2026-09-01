# Connection: {{ connection_name }}
# Use: from core.database.{{ connection_module }} import get_connection

from core.database.{{ connection_module }}.config import database
from core.database.{{ connection_module }}.engine import create_engine
from core.database.{{ connection_module }}.connection import get_connection
