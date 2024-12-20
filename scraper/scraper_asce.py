from typing import Dict
import utils


def process_author(author: str) -> str:
    author = author.split(", ")
    author = author[1] + ' ' + author[0]
    return author


def scrape_asce(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll('meta', attrs={'name': "citation_title"})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = ""
    # get pub name
    pub_name = utils.soup.extract_pub_name(soup)
    # get pub year
    pub_year = utils.soup.extract_pub_year(soup)
    # get authors
    authors = soup.findAll('meta', attrs={'name': "citation_author"})
    authors = ", ".join([process_author(author['content']) for author in authors])
    # get abstract
    abstract = utils.soup.extract_abstract(soup)
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
