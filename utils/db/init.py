from typing import Tuple, Optional
import json
import psycopg2
from psycopg2.extras import RealDictCursor


def init(config_path: Optional[str] = "/home/d_mao/.config/amazon_rds.json") -> Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """
    Initializes a connection to the PostgreSQL database and returns the connection and cursor.
    """
    with open(config_path, mode='r') as f:
        db_config = json.load(f)
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    return conn, cursor
