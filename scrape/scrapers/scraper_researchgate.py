from typing import Dict, Any
from scrape import utils


def scrape_researchgate(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll('meta', attrs={'property': "citation_title"})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll('meta', attrs={'property': "citation_pdf_url"})
    if len(pdf_url) == 1:
        pdf_url = pdf_url[0]['content']
    else:
        assert len(pdf_url) == 0
        pdf_url = ""
    # get pub name
    pub_name = utils.soup.extract_pub_name(soup)
    # get pub year
    pub_year = utils.soup.extract_pub_date(soup)
    # get authors
    authors = utils.soup.extract_authors(soup)
    # get abstract
    abstract = soup.findAll('div', attrs={'itemprop': "description"})
    if len(abstract) == 1:
        abstract = abstract[0].text
    elif len(abstract) == 0:
        abstract = ""
    else:
        assert 0
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_date': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
