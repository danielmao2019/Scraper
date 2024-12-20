from typing import Dict
import re
from datetime import datetime
import utils


def scrape_pmlr (url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.find('title').text
    # get pdf url
    pdf_url = re.findall(pattern="content=\"([^\"]+)\" name=\"citation_pdf_url\"", string=soup.__str__())
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]
    # get pub year
    pub_year = utils.soup.extract_pub_year(soup)
    # get authors
    authors = re.findall(pattern="content=\"([^\"]+)\" name=\"citation_author\"", string=soup.__str__())
    authors = ", ".join(authors)
    # get abstract
    abstract = soup.find('div', class_="abstract").text
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "PMLR",
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
