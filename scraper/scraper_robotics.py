from typing import Dict
from . import utils


def pub_name_mapping(name: str) -> str:
    if name.startswith("Robotics: Science and Systems"):
        return "RSS"
    else:
        return name


def scrape_robotics(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get pdf url
    pdf_url = '.'.join(url.split('.')[:-1]) + ".pdf"
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pub name
    pub_name = soup.findAll(name='meta', attrs={'name': 'citation_conference_title'})
    assert len(pub_name) == 1
    pub_name = pub_name[0]['content']
    pub_name = pub_name_mapping(pub_name)
    # get pub year
    pub_year = soup.findAll(name='meta', attrs={'name': 'citation_publication_date'})
    assert len(pub_year) == 1
    pub_year = f"`{pub_year[0]['content'].split('/')[0]}`"
    # get authors
    authors = soup.findAll(name='meta', attrs={'name': 'citation_author'})
    authors = ', '.join([a['content'] for a in authors])
    # get abstract
    abstract = soup.findAll(name='p', attrs={'style': 'text-align: justify;'})
    assert len(abstract) == 1
    abstract = abstract[0].text
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
