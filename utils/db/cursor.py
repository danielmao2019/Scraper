from typing import Tuple, List, Dict, Any, Optional
import json
import psycopg2
from psycopg2.extras import RealDictCursor


# def init(config_path: Optional[str] = "/home/d_mao/.config/amazon_rds.json"):
#     with open(config_path, mode='r') as f:
#         db_config = json.load(f)
#     conn = mysql.connector.connect(**db_config)
#     cursor = conn.cursor()
#     return conn, cursor


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


# def execute(cursor: mysql.connector.cursor_cext.CMySQLCursor, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
#     cursor.execute(query, params, multi=False)
#     results: List = cursor.fetchall()
#     assert type(results) == list, f"{type(results)=}"
#     assert all(map(lambda x: type(x) == tuple, results)), f"{results=}"
#     if cursor.description is not None:
#         assert type(cursor.description) == list, f"{type(cursor.description)=}"
#         assert all(map(lambda x: type(x) == tuple, cursor.description)), f"{cursor.description=}"
#         cols = list(map(lambda x: x[0], cursor.description))
#         results = list(map(lambda x: dict(zip(cols, x)), results))
#     else:
#         assert len(results) == 0
#     return results


def _loads(x: dict) -> dict:
    result = {}
    for key in x:
        try:
            result[key] = json.loads(x[key])
        except:
            result[key] = x[key]
    return result


def execute(cursor: psycopg2.extensions.cursor, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    """
    Executes a query and returns the results as a list of dictionaries.
    If no results are returned, an empty list is returned.
    If an error occurs, it handles the exception and returns an empty list.
    """
    cursor.execute(query, params)
    try:
        results = cursor.fetchall()
    except psycopg2.Error as e:
        assert str(e) == "no results to fetch", f"{str(e)=}"
        results = []
    results = list(map(dict, results))
    results = list(map(_loads, results))
    return results
