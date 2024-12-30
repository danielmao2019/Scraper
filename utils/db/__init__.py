"""
UTILS.DB API
"""
from utils.db.cursor import init, execute
from utils.db.insert import insert_record, insert_records


__all__ = (
    'init', 'execute',
    'insert_record', 'insert_records',
)
