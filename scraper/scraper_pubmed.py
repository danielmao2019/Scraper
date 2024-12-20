from typing import Dict
import utils


def scrape_pubmed(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = ""
    # get pub year
    pub_year = utils.soup.extract_pub_year(soup)
    # get authors
    authors = utils.soup.extract_authors(soup)
    # get abstract
    abstract = utils.soup.extract_abstract(soup)
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "PubMed",
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
