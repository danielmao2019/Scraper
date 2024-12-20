from typing import Dict
import requests
import re
import json
import utils


def scrape_springer(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # construct json
    json_str = soup.findAll('script', type="application/ld+json")
    assert len(json_str) == 1
    json_str = json_str[0].text.strip()
    json_dict = json.loads(json_str)
    if 'mainEntity' in json_dict:
        json_dict = json_dict['mainEntity']
    assert set(['author', 'isPartOf', 'description']).issubset(set(json_dict.keys()))
    # get title
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    # get pdf url
    pdf_url = soup.findAll(name='meta', attrs={'name': 'citation_pdf_url'})
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]['content']
    r = requests.get(pdf_url)
    assert r.status_code == 200, f"r.status_code={r.status_code}, pdf_url={pdf_url}"
    # get authors
    authors = ", ".join(a['name'] for a in json_dict['author'])
    # get publisher
    pub_name = json_dict['isPartOf']['name']
    pub_name = re.findall(pattern="([A-Z]+) \d+", string=pub_name)
    if len(pub_name) == 1:
        pub_name = pub_name[0]
    elif len(pub_name) == 0:
        pub_name = json_dict['isPartOf']['name']
    else:
        assert 0
    # get pub year
    pub_year = utils.soup.extract_pub_year(soup)
    # get abstract
    abstract = json_dict['description']
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
