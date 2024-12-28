"""
UTILS.DB API
"""
from utils.db.cursor import init, execute
from utils.db.insert import insert


__all__ = (
    'init', 'execute',
    'insert',
)
