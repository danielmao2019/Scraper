from typing import Dict
import json
import utils


def scrape_openreview(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # construct json
    json_str = soup.findAll('script', type="application/json")
    assert len(json_str) == 1
    json_str = json_str[0].text.strip()
    json_dict = json.loads(json_str)
    json_dict = json_dict['props']['pageProps']['forumNote']
    # get title
    title = soup.findAll(name='meta', attrs={'name': "citation_title"})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll(name='a', attrs={'title': "Download PDF"})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['href']
    if pdf_url.startswith("/pdf?id="):
        pdf_url = "https://openreview.net" + pdf_url
    # get pub name and year
    try:
        pub_name = soup.findAll(name='meta', attrs={'name': "citation_journal_title"})
        assert len(pub_name) == 1
        pub_name = pub_name[0]['content']
        pub_year = soup.findAll(name='meta', attrs={'name': "citation_publication_date"})
        assert len(pub_year) == 1
        pub_year = pub_year[0]['content']
    except:
        pub_name, pub_year = utils.parse_publisher(json_dict['content']['venue'])
    # get authors
    authors = ", ".join([
        a['content'] for a in soup.findAll(name='meta', attrs={'name': "citation_author"})
    ])
    # get abstract
    abstract = soup.findAll(name='meta', attrs={'name': "citation_abstract"})
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
