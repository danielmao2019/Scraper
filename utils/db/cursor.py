from typing import Tuple, List, Dict, Any, Optional
import json
import mysql.connector


def init(config_path: Optional[str] = "/home/d_mao/.config/amazon_rds.json"):
    with open(config_path, mode='r') as f:
        db_config = json.load(f)
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    return conn, cursor


def execute(cursor: mysql.connector.cursor_cext.CMySQLCursor, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    cursor.execute(query, params, multi=False)
    results: List = cursor.fetchall()
    assert type(results) == list, f"{type(results)=}"
    assert all(map(lambda x: type(x) == tuple, results)), f"{results=}"
    if cursor.description is not None:
        assert type(cursor.description) == list, f"{type(cursor.description)=}"
        assert all(map(lambda x: type(x) == tuple, cursor.description)), f"{cursor.description=}"
        cols = list(map(lambda x: x[0], cursor.description))
        results = list(map(lambda x: dict(zip(cols, x)), results))
    else:
        assert len(results) == 0
    return results
