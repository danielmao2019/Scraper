import requests
import re
import json

from . import utils


def scrape_springer(url: str) -> dict:
    assert type(url) == str, f"{type(url)=}"
    soup = utils.get_soup(url)
    # construct json
    json_str = soup.findAll('script', type="application/ld+json")
    assert len(json_str) == 1
    json_str = json_str[0].text.strip()
    json_dict = json.loads(json_str)
    # extract from json
    title = json_dict['headline']
    pdf_url = re.findall(pattern="content=\"(https://link.springer.com/content/pdf/.*\.pdf)\"", string=str(soup))
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]
    r = requests.get(pdf_url)
    assert r.status_code == 200, f"{r.status_code=}, {pdf_url=}"
    authors = ", ".join(a['name'] for a in json_dict['author'])
    pub_name = json_dict['isPartOf']['name']
    pub_name = re.findall(pattern="Computer Vision . (\w+) \d+", string=pub_name)
    assert len(pub_name) == 1, f"{pub_name=}"
    pub_name = pub_name[0]
    pub_year = f"`{json_dict['datePublished']}`"
    abstract = json_dict['description'].strip()
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
