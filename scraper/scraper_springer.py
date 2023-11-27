from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import re
import json


def scrape_springer(url: str) -> dict:
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
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
    year = f"`{json_dict['datePublished']}`"
    abstract = json_dict['description'].strip()
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "Springer",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
