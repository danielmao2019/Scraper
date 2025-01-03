from typing import Dict, Any
from scrape import utils


def scrape_openaccess(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # extract
    title = soup.find("div", id="papertitle").text.strip()
    pdf_url = utils.get_pdf_url(url, soup.find("a", string="pdf")['href'])
    # get pub name
    parse_string = url
    parse_string = parse_string[len("https://openaccess.thecvf.com/content"):]
    parse_string = '/'.join(parse_string.split('/')[:-1])
    pub_name = utils.parse_pub_name(parse_string)
    # get pub date
    pub_year = utils.soup.extract_pub_date(soup)
    authors = soup.find("div", id="authors").text.strip().split(';')[0]
    authors = authors.split(", ")
    abstract = soup.find("div", id="abstract").text.strip()
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_date': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
