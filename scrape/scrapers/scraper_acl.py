from typing import Dict, Any
import re
from scrape import utils


def scrape_acl(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll(name='meta', attrs={'name': 'citation_pdf_url'})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['content']
    # get pub year
    pub_date = re.findall(pattern=r'year = "(\d+)"', string=str(soup))
    assert len(pub_date) == 1
    pub_date = pub_date[0]
    # get authors
    authors = soup.findAll(name='meta', attrs={'name': 'citation_author'})
    authors = list(map(lambda x: x['content'], authors))
    # get abstract
    abstract = ""
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': "ACL",
        'pub_date': pub_date,
        'authors': authors,
        'abstract': abstract,
    }
