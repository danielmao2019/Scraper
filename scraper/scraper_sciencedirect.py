from typing import Dict
import utils


def scrape_sciencedirect(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = url + "/pdfft"
    # get pub year
    pub_name = soup.findAll(name='meta', attrs={'name': 'citation_journal_title'})
    assert len(pub_name) == 1
    pub_name = pub_name[0]['content']
    # get pub year
    pub_year = soup.findAll(name='meta', attrs={'name': 'citation_publication_date'})
    assert len(pub_year) == 1
    pub_year = pub_year[0]['content']
    assert len(pub_year.split('/')) == 3, f"{pub_year=}"
    # get authors
    authors = ', '.join([
        first_name.get_text(strip=True) + ' ' + last_name.get_text(strip=True)
        for first_name, last_name in zip(
            soup.findAll(name='span', attrs={'class': 'given-name'}),
            soup.findAll(name='span', attrs={'class': 'text surname'})
        )
    ])
    # get abstract
    abstract = utils.parse_abstract_after_h2(soup)
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
