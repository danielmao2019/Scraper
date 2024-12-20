from typing import Dict
import re
import utils


def scrape_acl(url: str) -> Dict[str, str]:
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
    # get year
    year = re.findall(pattern=r'year = "(\d+)"', string=str(soup))
    assert len(year) == 1
    year = year[0]
    # get authors
    authors = soup.findAll(name='meta', attrs={'name': 'citation_author'})
    authors = ', '.join([a['content'] for a in authors])
    # get abstract
    abstract = ""
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "ACL",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
