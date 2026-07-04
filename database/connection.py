import os
import sqlite3
from config.settings import settings
from core.logger import get_logger

logger = get_logger("database.connection")

# We choose four separate database files for modularity and easy operations
DB_FILES = {
    "market": "data/market.db",
    "signals": "data/signals.db",
    "trades": "data/trades.db",
    "portfolio": "data/portfolio.db"
}

def get_connection(db_name: str) -> sqlite3.Connection:
    """
    Get a database connection to the specified SQLite database.
    Supported db_names: 'market', 'signals', 'trades', 'portfolio'.
    """
    if db_name not in DB_FILES:
        raise ValueError(f"Unknown database: {db_name}. Supported are: {list(DB_FILES.keys())}")
        
    db_path = DB_FILES[db_name]
    
    # Ensure parent directories exist
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    try:
        conn = sqlite3.connect(db_path)
        # Enable foreign key support and WAL mode for efficient concurrent access
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        logger.debug(f"Connected to database: {db_name} at {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database {db_name} at {db_path}: {e}")
        raise

def initialize_databases():
    """Create the physical database files and parent directories."""
    for name in DB_FILES:
        conn = get_connection(name)
        conn.close()
    logger.info("Successfully initialized empty SQLite database files.")
