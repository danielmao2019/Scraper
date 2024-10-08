from typing import Dict
from .utils import get_soup


def scrape_pubmed(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = ""
    # get year
    year = soup.findAll('meta', {'name': 'citation_publication_date'})
    assert len(year) == 1
    year = f"`{year[0]['content'].split('/')[0]}`"
    # get authors
    authors = soup.findAll('meta', {'name': 'citation_author'})
    authors = ", ".join([t['content'] for t in authors])
    # get abstract
    abstract = soup.findAll('meta', {'name': 'citation_abstract'})
    assert len(abstract) == 1
    abstract = abstract[0]['content']
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "PubMed",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
