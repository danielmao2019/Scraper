from typing import Dict, Any
import re
from scrape import utils


def scrape_eccv(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # generate links
    rel_pdf_url = soup.find("a", string="pdf")['href']
    pdf_url = utils.get_pdf_url(url, rel_pdf_url)
    # get title
    title = soup.find("div", id="papertitle").text.strip()
    # get pub year
    pattern = "eccv_(\d\d\d\d)"
    pub_date = re.findall(pattern=pattern, string=url)
    assert len(pub_date) == 1, f"url={url}, pattern={pattern}, year={pub_date}"
    assert 2000 <= int(pub_date[0]) <= 2030
    pub_date = pub_date[0]
    # get authors
    authors = soup.find("div", id="authors").text.strip()
    authors = re.sub(pattern='\n', repl="", string=authors)
    authors = authors.split(", ")
    # get abstract
    abstract = soup.find("div", id="abstract").text.strip()
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': "ECCV",
        'pub_date': pub_date,
        'authors': authors,
        'abstract': abstract,
    }
