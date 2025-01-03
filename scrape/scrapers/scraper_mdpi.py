from typing import Dict, Any
from scrape import utils


def scrape_mdpi(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.find("h1", attrs={'class': "title hypothesis_container"}).text.strip()
    # get pdf url
    pdf_url = soup.findAll('meta', attrs={'name': "citation_pdf_url"})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['content']
    # get pub name
    pub_name = utils.soup.extract_pub_name(soup)
    # get pub year
    pub_year = utils.soup.extract_pub_date(soup)
    # get authors
    authors = utils.soup.extract_authors(soup)
    # get abstract
    abstract = soup.findAll("meta", attrs={'name': "description"})
    assert len(abstract) == 1
    abstract = abstract[0]['content']
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
