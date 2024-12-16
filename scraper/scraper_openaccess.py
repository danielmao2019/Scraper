from typing import Dict
import utils


def scrape_openaccess(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # extract
    title = soup.find("div", id="papertitle").text.strip()
    pdf_url = utils.get_pdf_url(url, soup.find("a", string="pdf")['href'])
    conf_name, conf_year = utils.parse_publisher(url.split('/')[-1])
    authors = soup.find("div", id="authors").text.strip().split(';')[0]
    abstract = soup.find("div", id="abstract").text.strip()
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': conf_name,
        'pub_year': conf_year,
        'authors': authors,
        'abstract': abstract,
    }
