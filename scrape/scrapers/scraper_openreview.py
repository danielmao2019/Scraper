from typing import Dict, Any
import json
from scrape import utils


def scrape_openreview(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
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
        pub_name = utils.soup.extract_pub_name(soup)
    except:
        json_str = soup.findAll('script', type="application/json")
        assert len(json_str) == 1
        json_str = json_str[0].text.strip()
        json_dict = json.loads(json_str)
        json_dict = json_dict['props']['pageProps']['forumNote']
        pub_name = utils.parse_pub_name(json_dict['content']['venue']['value'])
    pub_year = utils.soup.extract_pub_date(soup)
    # get authors
    authors = utils.soup.extract_authors(soup)
    # get abstract
    abstract = soup.findAll(name='meta', attrs={'name': "citation_abstract"})
    assert len(abstract) == 1
    abstract = abstract[0]['content']
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
