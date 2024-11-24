from typing import Dict
import utils


def scrape_aaai(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'DC.Title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll(name='a', attrs={'class': 'obj_galley_link pdf'})
    assert len(pdf_url) == 1, f"pdf_url={pdf_url}"
    pdf_url = pdf_url[0]['href']
    # get year
    year = soup.findAll(name='meta', attrs={'name': 'DC.Date.created'})
    assert len(year) == 1, f"year={year}"
    year = f"`{year[0]['content'].split('-')[0]}`"
    # get authors
    authors = soup.findAll(name='meta', attrs={'name': 'DC.Creator.PersonalName'})
    authors = ", ".join([t['content'] for t in authors])
    # get abstract
    abstract = soup.findAll(name='meta', attrs={'name': 'DC.Description'})
    assert len(abstract) == 1
    abstract = abstract[0]['content']
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "AAAI",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
