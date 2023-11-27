from urllib.request import urlopen
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import re
import json


def scrape_ieee(url: str) -> dict:
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # construct json
    json_str = re.findall(pattern="xplGlobal.document.metadata=(.*);\n", string=str(soup))
    assert len(json_str) == 1
    json_str = json_str[0]
    json_dict = json.loads(json_str)
    # extract from json
    title = json_dict['title']
    pdf_url = urljoin(url, json_dict['pdfUrl'])
    year = f"`{json_dict['publicationYear']}`"
    authors = ", ".join([a['name'] for a in json_dict['authors']])
    abstract = json_dict['abstract']
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "IEEE",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
