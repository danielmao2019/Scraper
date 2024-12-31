from typing import List, Dict
import os
import time
import re
import hashlib
import fitz  # PyMuPDF
from scrape import utils


def _extract_text_wget(url: str) -> str:
    """
    Returns:
        text (str): text in the file from url.
    """
    # download pdf
    time.sleep(4)  # avoid CAPTCHA
    cmd = f'wget "{url}" --output-document tmp.pdf --quiet'
    os.system(cmd)
    # extract text
    text = ""
    pdf_doc = fitz.open("tmp.pdf")
    for page in pdf_doc:
        text += page.get_text() + "\n"
    pdf_doc.close()
    os.system("rm tmp.pdf")
    return text


def _extract_text_elsevier(url: str) -> str:
    # get Elsevier client
    from elsapy.elsclient import ElsClient
    config = {
        "apikey": "1be9aac6ca066332daef54b7ff83996e",
        "insttoken": ""
    }
    client = ElsClient(config['apikey'])
    client.inst_token = config['insttoken']
    # get pii
    from elsapy.elsdoc import FullDoc
    assert url.endswith("/pdfft")
    id = url.split('/')[-2]
    pii_doc = FullDoc(sd_pii=id)
    assert pii_doc.read(client)
    text = pii_doc.__dict__['_data']['originalText']
    return text


def _extract_text_scholarsportal(url: str) -> str:
    soup = utils.soup.get_soup(url)
    # get title
    title = soup.findAll(name='title')
    assert len(title) == 1
    title = title[0].text.strip().split(" | ")[0]
    assert any([title == x.text.strip() for x in soup.findAll(name='span', attrs={'class': "article-title"})])
    # get keywords
    keywords = soup.findAll(name='ul', attrs={'class': "keywords-list"})
    assert len(keywords) == 1
    keywords = keywords[0]
    keywords = keywords.findAll(name='a')
    keywords = list(map(lambda x: x.text.strip(), keywords))
    keywords = '\n'.join(keywords)
    # get abstract
    abstract = soup.findAll(name='div', attrs={'class': "journal-abstract"})
    assert len(abstract) == 1
    abstract = abstract[0].find('div').find('div').findAll('p')
    assert len(abstract) == 1
    abstract = abstract[0].text.strip()
    # get body
    body = soup.findAll(name='div', attrs={'id': "article-body"})
    assert len(body) == 1
    body = list(body[0].children)
    body = list(filter(lambda x: x.name is not None, body))
    assert body[0].find('h3').text.strip() == "Table of Contents"
    body = body[1:]  # ignore table of contents
    body = '\n'.join(list(map(
        lambda x: '\n'.join(list(filter(
            lambda y: y.name is None, x.descendants
        ))), body,
    )))
    # full text
    return '\n'.join([title, keywords, abstract, body])


def extract_text(url: str) -> str:
    root_dir = os.path.join("search", "downloads")
    os.makedirs(root_dir, exist_ok=True)
    code: str = hashlib.sha256(url.encode('utf-8')).hexdigest()
    filepath = os.path.join(root_dir, code+".txt")
    try:
        assert os.path.isfile(filepath), f"File does not exist: {filepath}"
        assert os.path.getsize(filepath) > 0, f"File empty: {filepath}"
        with open(filepath, mode='r') as f:
            return f.read()
    except Exception as e:
        print(f"{e=}")
        if url.startswith("https://www.sciencedirect.com"):
            text = _extract_text_elsevier(url)
        elif url.startswith("https://journals.scholarsportal.info"):
            text = _extract_text_scholarsportal(url)
        else:
            text = _extract_text_wget(url)
        assert type(text) == str, f"{type(text)=}"
        with open(filepath, mode='w') as f:
            f.write(text)
        return text
