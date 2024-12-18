from typing import Dict
import utils


def scrape_researchgate(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get title
    title = soup.findAll('meta', attrs={'property': "citation_title"})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll('meta', attrs={'property': "citation_pdf_url"})
    if len(pdf_url) == 1:
        pdf_url = pdf_url[0]['content']
    else:
        assert len(pdf_url) == 0
        pdf_url = ""
    # get pub name
    try:
        pub_name = soup.findAll('meta', attrs={'property': "citation_journal_title"})
        assert len(pub_name) == 1
        pub_name = pub_name[0]['content']
    except:
        try:
            pub_name = soup.findAll('meta', attrs={'property': "citation_conference_title"})
            assert len(pub_name) == 1
            pub_name = pub_name[0]['content']
        except:
            pub_name = ""
    # get pub year
    pub_year = soup.findAll('meta', attrs={'property': "citation_publication_date"})
    assert len(pub_year) == 1
    pub_year = pub_year[0]['content']
    # get authors
    authors = soup.findAll('meta', attrs={'property': "citation_author"})
    authors = ", ".join([author['content'] for author in authors])
    # get abstract
    abstract = soup.findAll('div', attrs={'itemprop': "description"})
    if len(abstract) == 1:
        abstract = abstract[0].text
    elif len(abstract) == 0:
        abstract = ""
    else:
        assert 0
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
