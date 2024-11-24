from typing import Dict
import utils


def scrape_pubmed(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = ""
    # get year
    year = soup.findAll('meta', {'name': 'citation_date'})
    assert len(year) == 1, f"{year=}"
    year = year[0]['content']
    # get authors
    authors = utils.parse_authors_from_meta(soup)
    # get abstract
    abstract = utils.parse_abstract_after_h2(soup)
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
