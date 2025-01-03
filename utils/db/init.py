from typing import Tuple, Optional
import json
import psycopg2
from psycopg2.extras import RealDictCursor


def init(config_path: Optional[str] = "/home/d_mao/.config/amazon_rds.json") -> Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """
    Initializes a connection to a PostgreSQL database and returns the connection and cursor objects.

    Args:
        config_path (Optional[str]):
            Path to the JSON file containing the database configuration.
            Defaults to "/home/d_mao/.config/amazon_rds.json".

    Returns:
        Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
            A tuple containing:
            - `connection`: The active PostgreSQL connection object.
            - `cursor`: A cursor object configured to return query results as dictionaries.
    """
    # Load database configuration from the JSON file
    with open(config_path, mode='r') as f:
        db_config = json.load(f)

    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True  # Enable autocommit mode for the connection

    # Create a cursor object that returns query results as dictionaries
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    return conn, cursor
