from typing import Tuple, List, Dict, Any, Optional
import json
import psycopg2


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
