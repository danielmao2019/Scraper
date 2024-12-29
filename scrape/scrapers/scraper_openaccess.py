from typing import Dict
from scrape import utils


def scrape_openaccess(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # extract
    title = soup.find("div", id="papertitle").text.strip()
    pdf_url = utils.get_pdf_url(url, soup.find("a", string="pdf")['href'])
    pub_name = utils.parse_pub_name(url.split('/')[-1])
    pub_year = utils.soup.extract_pub_year(soup)
    authors = soup.find("div", id="authors").text.strip().split(';')[0]
    abstract = soup.find("div", id="abstract").text.strip()
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
