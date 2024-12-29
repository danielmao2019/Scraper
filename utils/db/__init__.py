"""
UTILS.DB API
"""
from utils.db.cursor import init
from utils.db.insert import insert_record, insert_records


__all__ = (
    'init',
    'insert_record', 'insert_records',
)
