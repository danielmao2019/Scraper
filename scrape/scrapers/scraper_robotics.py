from typing import Dict, Any
from scrape import utils


def scrape_robotics(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get pdf url
    pdf_url = '.'.join(url.split('.')[:-1]) + ".pdf"
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pub name
    pub_name = utils.soup.extract_pub_name(soup)
    # get pub year
    pub_year = utils.soup.extract_pub_date(soup)
    # get authors
    authors = utils.soup.extract_authors(soup)
    # get abstract
    abstract = soup.findAll(name='p', attrs={'style': 'text-align: justify;'})
    assert len(abstract) == 1
    abstract = abstract[0].text
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_date': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
