from typing import Dict
from scrape import utils


def scrape_scholarsportal(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll(name='title')
    assert len(title) == 1
    title = title[0].text.strip().split(" | ")[0]
    assert any([title == x.text.strip() for x in soup.findAll(name='span', attrs={'class': "article-title"})])
    # get pdf url
    pdf_url = soup.findAll(name='div', attrs={'class': "download-btn"})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0].find('a')
    assert "PDF Download" in pdf_url.text.strip()
    pdf_url = pdf_url['href']
    # get pub name
    pub_name = soup.findAll(name='div', attrs={'class': "journal-title"})
    assert len(pub_name) == 1
    pub_name = pub_name[0].findAll(name='journal-title')
    assert len(pub_name) == 1
    pub_name = pub_name[0].text.strip()
    # get pub year
    pub_year = soup.findAll(name='div', attrs={'class': "journal-info"})
    pub_year = pub_year[0]
    assert len(pub_year.findAll(name='h2', attrs={'class': "sr-only"})) == 1
    pub_year = pub_year.find('span').find('span').find('span').text.strip()
    assert ',' in pub_year, f"{pub_year=}"
    pub_year = pub_year[:-1]
    # get authors
    authors = soup.findAll(name='div', attrs={'id': "authors"})
    assert len(authors) == 1
    authors = authors[0]
    authors = authors.findAll(name='div', attrs={'class': "authors"})
    assert len(authors) == 1
    authors = authors[0]
    authors = authors.findAll(name='li')
    authors = list(map(lambda x: x.find('a'), authors))
    authors = list(map(lambda x: x.text.strip(), authors))
    assert all(authors)
    authors = ", ".join(authors)
    # get abstract
    abstract = soup.findAll(name='div', attrs={'class': "journal-abstract"})
    assert len(abstract) == 1
    abstract = abstract[0].find('div').find('div').findAll('p')
    assert len(abstract) == 1
    abstract = abstract[0].text.strip()
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
