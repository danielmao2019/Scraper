from typing import Dict
import utils


def scrape_mdpi(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get title
    title = soup.find("h1", attrs={'class': "title hypothesis_container"}).text.strip()
    # get pdf url
    pdf_url = soup.findAll('meta', attrs={'name': "citation_pdf_url"})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['content']
    # get pub name
    pub_name = soup.findAll('meta', attrs={'name': "citation_journal_title"})
    assert len(pub_name) == 1
    pub_name = pub_name[0]['content']
    # get pub year
    pub_year = soup.findAll('meta', attrs={'name': "citation_publication_date"})
    assert len(pub_year) == 1
    pub_year = pub_year[0]['content']
    # get authors
    authors = soup.findAll("meta", attrs={'name': "dc.creator"})
    authors = ', '.join([author['content'] for author in authors])
    # get abstract
    abstract = soup.findAll("meta", attrs={'name': "description"})
    assert len(abstract) == 1
    abstract = abstract[0]['content']
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
