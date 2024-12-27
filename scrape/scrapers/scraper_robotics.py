from typing import Dict
from scrape import utils


def scrape_robotics(url: str) -> Dict[str, str]:
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
    pub_year = utils.soup.extract_pub_year(soup)
    # get authors
    authors = soup.findAll(name='meta', attrs={'name': 'citation_author'})
    authors = ', '.join([a['content'] for a in authors])
    # get abstract
    abstract = soup.findAll(name='p', attrs={'style': 'text-align: justify;'})
    assert len(abstract) == 1
    abstract = abstract[0].text
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
