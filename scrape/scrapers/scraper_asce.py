from typing import Dict
from scrape import utils


def process_author(author: str) -> str:
    author = author.split(", ")
    author = author[1] + ' ' + author[0]
    return author


def scrape_asce(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll('meta', attrs={'name': "citation_title"})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll(name='a', attrs={'title': "View PDF"})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['href']
    # get pub name
    try:
        pub_name = utils.soup.extract_pub_name(soup)
    except:
        pub_name_cands = []
        pub_name_cands += soup.findAll(name='span', attrs={'property': "name"})
        pub_name_cands += soup.findAll(name='a', attrs={'class': "article__tocHeading"})
        pub_name = []
        for cand in pub_name_cands:
            try:
                pub_name.append(utils.parse_pub_name(cand.text.strip()))
            except:
                pass
        assert len(pub_name) > 0
        assert all([x == pub_name[0] for x in pub_name]), f"{pub_name=}"
        pub_name = pub_name[0]
    # get pub year
    pub_year = utils.soup.extract_pub_year(soup)
    # get authors
    authors = soup.findAll('meta', attrs={'name': "citation_author"})
    authors = ", ".join([process_author(author['content']) for author in authors])
    # get abstract
    abstract = utils.soup.extract_abstract(soup)
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
