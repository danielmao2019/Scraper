from typing import Dict, Any
import re
from datetime import datetime
from scrape import utils


def scrape_pmlr (url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.find('title').text
    # get pdf url
    pdf_url = re.findall(pattern="content=\"([^\"]+)\" name=\"citation_pdf_url\"", string=soup.__str__())
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]
    # get pub year
    pub_year = utils.soup.extract_pub_date(soup)
    # get authors
    authors = utils.soup.extract_authors(soup)
    # get abstract
    abstract = soup.find('div', class_="abstract").text
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': "PMLR",
        'pub_date': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
