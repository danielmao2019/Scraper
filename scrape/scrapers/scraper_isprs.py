from typing import Dict, Any
from scrape import utils
from bs4 import BeautifulSoup


def _process_abstract(abstract: str) -> str:
    decoded_html = abstract.replace('&lt;', '<').replace('&gt;', '>')
    soup = BeautifulSoup(decoded_html, 'html.parser')

    # Extract the text content from the <p> tag, skipping <strong> tag's content
    p_tag = soup.find('p')
    if p_tag:
        abstract_text = p_tag.text
        # Remove the "Abstract." prefix if needed
        if 'Abstract.' in abstract_text:
            abstract_text = abstract_text.replace('Abstract.', '').strip()
            return abstract_text
    raise RuntimeError("Error processing abstract.")


def scrape_isprs(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll('meta', attrs={'name': "citation_title"})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll('meta', attrs={'name': "citation_pdf_url"})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['content']
    # get pub name
    pub_name = "ISPRS"
    # get pub year
    pub_year = utils.soup.extract_pub_date(soup)
    # get authors
    authors = utils.soup.extract_authors(soup)
    # get abstract
    abstract = soup.findAll('meta', attrs={'name': "citation_abstract"})
    assert len(abstract) == 1
    abstract = abstract[0]['content']
    abstract = _process_abstract(abstract)
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
