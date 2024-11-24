from typing import Dict
import utils


def scrape_jmlr(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll('meta', {'name': 'citation_pdf_url'})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['content']
    # get year
    year = soup.findAll('meta', {'name': 'citation_publication_date'})
    assert len(year) == 1
    year = f"`{year[0]['content']}`"
    # get authors
    authors = soup.findAll('meta', {'name': 'citation_author'})
    authors = ", ".join([t['content'] for t in authors])
    # get abstract
    abstract = soup.find('h3', text="Abstract").findNextSiblings('p')
    abstract = ' '.join([p.text.strip() for p in abstract])
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "JMLR",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
