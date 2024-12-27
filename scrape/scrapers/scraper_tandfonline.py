from typing import Dict
import re
from scrape import utils


def scrape_tandfonline(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll(name='meta', attrs={'property': "og:title"})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll(name='a', attrs={'class': "show-pdf"})
    pdf_url = [x for x in pdf_url if x.text.strip() == "Download PDF"]
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['href']
    # get pub name
    pub_name = utils.soup.extract_pub_name(soup)
    # get pub year
    pub_year = utils.soup.extract_pub_year(soup)
    # get authors
    authors = soup.findAll(name='meta', attrs={'name': "dc.Creator"})
    authors = ", ".join([
        ' '.join(x['content'].split()) for x in authors
    ])
    # get abstract
    h2_tag = soup.find('h2', text=re.compile(r'\bAbstract\b', re.IGNORECASE))
    assert h2_tag, "Abstract heading not found"
    abstract_p = h2_tag.findNext('p')
    assert abstract_p
    abstract = abstract_p.text.strip()
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
