"""
SCRAPE.UTILS.SOUP API
"""
from scrape.utils.soup.get_soup import get_soup
from scrape.utils.soup.extract import (
    extract_pub_name,
    extract_pub_date,
    extract_authors,
    extract_abstract,
)


__all__ = (
    'get_soup',
    'extract_pub_name',
    'extract_pub_date',
    'extract_authors',
    'extract_abstract',
)
