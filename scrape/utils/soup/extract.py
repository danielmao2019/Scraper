from typing import List
import re
from scrape.utils.parse import parse_pub_name


def _extract_meta_content(soup, key: str, val: str, result: List[str]) -> None:
    pub_name = soup.findAll(name='meta', attrs={key: val})
    pub_name = [x['content'] for x in pub_name]
    result += pub_name

# ==================================================
# pub name
# ==================================================

def extract_pub_name(soup) -> str:
    extracted: List[str] = []
    for key in ['name', 'property']:
        for val in [
            'citation_conference_title', 'citation_journal_title',
        ]:
            _extract_meta_content(soup, key, val, extracted)
    parsed = [parse_pub_name(x) for x in extracted]
    assert len(parsed) >= 1, f"No pub name extracted and parsed from soup."
    assert all([x == parsed[0] for x in parsed]), f"{parsed=}"
    return parsed[0]

# ==================================================
# pub year
# ==================================================

def extract_pub_year(soup) -> str:
    extracted: List[str] = []
    for key in ['name', 'property']:
        for val in [
            'citation_publication_date',
            'citation_date', 'citation_online_date',
            'dc.Date', 'DC.Date.created',
        ]:
            _extract_meta_content(soup, key, val, extracted)
    assert len(extracted) >= 1, f"No pub name extracted from soup."
    return extracted[0]

# ==================================================
# authors
# ==================================================

def _parse_authors_citation_authors(citation_authors) -> str:
    assert len(citation_authors) == 1
    authors = citation_authors[0]['content']
    authors = ", ".join(filter(lambda x: x.strip(), authors.split(';')))
    return authors


def _parse_authors_citation_author_list(citation_author_list) -> str:
    authors = ", ".join([ca['content'] for ca in citation_author_list])
    return authors


def extract_authors(soup) -> str:
    citation_authors = soup.findAll('meta', {'name': 'citation_authors'})
    if len(citation_authors) > 0:
        return _parse_authors_citation_authors(citation_authors)
    citation_author_list = soup.findAll('meta', {'name': 'citation_author'})
    if len(citation_author_list) > 0:
        return _parse_authors_citation_author_list(citation_author_list)
    raise RuntimeError("Cannot parse authors.")

# ==================================================
# abstract
# ==================================================

def _extract_text_with_spaces(element) -> str:
    text: List[str] = []
    for item in element.descendants:
        if item.name is None:  # Only add non-tag elements
            text.append(item)
    return ''.join(text)


def extract_abstract(soup) -> str:
    # Match 'Abstract' with flexibility for spaces, newlines, and case
    h2_tag = soup.find('h2', text=re.compile(r'\bAbstract\b', re.IGNORECASE))
    assert h2_tag, "Abstract heading not found"

    # Find the next <div> after the matched <h2>
    abstract_div = h2_tag.find_next('div')
    assert abstract_div, "Abstract content not found"

    # Extract text from the abstract's <div>
    abstract = _extract_text_with_spaces(abstract_div)
    return abstract
