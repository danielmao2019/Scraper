from typing import Dict, Any
from urllib.parse import urljoin
import re
import json
from scrape import utils


def scrape_ieee(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # construct json
    json_str = re.findall(pattern="xplGlobal.document.metadata=(.*);\n", string=str(soup))
    assert len(json_str) == 1
    json_str = json_str[0]
    json_dict = json.loads(json_str)
    # extract from json
    title = json_dict['title']
    pdf_url = urljoin(url, json_dict['pdfUrl'])
    year = json_dict['publicationDate']
    authors = list(map(lambda x: x['name'], json_dict['authors']))
    abstract = json_dict['abstract']
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': "IEEE",
        'pub_date': year,
        'authors': authors,
        'abstract': abstract,
    }
