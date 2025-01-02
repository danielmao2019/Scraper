from typing import Dict, Any
from scrape import utils


def scrape_jmlr(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll('meta', {'name': 'citation_pdf_url'})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['content']
    # get pub year
    pub_year = utils.soup.extract_pub_date(soup)
    # get authors
    authors = utils.soup.extract_authors(soup)
    # get abstract
    abstract = soup.find('h3', text="Abstract").findNextSiblings('p')
    abstract = ' '.join([p.text.strip() for p in abstract])
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': "JMLR",
        'pub_date': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
