from typing import Dict
import utils


def process_author(author: str) -> str:
    author = author.split(", ")
    author = author[1] + ' ' + author[0]
    return author


def scrape_asce(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get title
    title = soup.findAll('meta', attrs={'name': "citation_title"})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = ""
    # get pub name
    pub_name = soup.findAll('meta', attrs={'name': "citation_journal_title"})
    assert len(pub_name) == 1
    pub_name = pub_name[0]['content']
    # get pub year
    pub_year = soup.findAll('meta', attrs={'name': "citation_publication_date"})
    assert len(pub_year) == 1
    pub_year = pub_year[0]['content']
    # get authors
    authors = soup.findAll('meta', attrs={'name': "citation_author"})
    authors = ", ".join([process_author(author['content']) for author in authors])
    # get abstract
    abstract = utils.parse_abstract_after_h2(soup)
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
