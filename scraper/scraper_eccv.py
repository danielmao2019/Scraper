from typing import Dict
import re
from . import utils


def scrape_eccv(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # generate links
    rel_pdf_url = soup.find("a", string="pdf")['href']
    pdf_url = utils.get_pdf_url(url, rel_pdf_url)
    # get title
    title = soup.find("div", id="papertitle").text.strip()
    # get year
    pattern = "eccv_(\d\d\d\d)"
    year = re.findall(pattern=pattern, string=url)
    assert len(year) == 1, f"url={url}, pattern={pattern}, year={year}"
    assert 2000 <= int(year[0]) <= 2030
    year = f"`{year[0]}`"
    # get authors
    authors = soup.find("div", id="authors").text.strip()
    authors = re.sub(pattern='\n', repl="", string=authors)
    # get abstract
    abstract = soup.find("div", id="abstract").text.strip()
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "ECCV",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
