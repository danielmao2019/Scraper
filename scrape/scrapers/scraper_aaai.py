from typing import Dict, Any
from scrape import utils


def scrape_aaai(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'DC.Title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll(name='a', attrs={'class': 'obj_galley_link pdf'})
    assert len(pdf_url) == 1, f"pdf_url={pdf_url}"
    pdf_url = pdf_url[0]['href']
    # get pub year
    pub_date = utils.soup.extract_pub_date(soup)
    # get authors
    authors = soup.findAll(name='meta', attrs={'name': 'DC.Creator.PersonalName'})
    authors = list(map(lambda x: x['content'], authors))
    # get abstract
    abstract = soup.findAll(name='meta', attrs={'name': 'DC.Description'})
    assert len(abstract) == 1
    abstract = abstract[0]['content']
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': "AAAI",
        'pub_date': pub_date,
        'authors': authors,
        'abstract': abstract,
    }
