"""
UTILS.SOUP API
"""
from utils.soup.get_soup import get_soup
from utils.soup.extract import (
    extract_pub_name,
    extract_authors,
    extract_abstract,
)


__all__ = (
    'get_soup',
    'extract_pub_name',
    'extract_authors',
    'extract_abstract',
)
