from typing import List, Dict
import os
import time
import re
import hashlib
import fitz  # PyMuPDF
from scrape import utils


def _url2text_wget(url: str) -> str:
    """
    Returns:
        text (str): text in the file from url.
    """
    filename = hashlib.sha256(url.encode('utf-8')).hexdigest() + ".pdf"
    os.makedirs(os.path.join("search", "downloads"), exist_ok=True)
    filepath = os.path.join("search", "downloads", filename)
    if os.path.isfile(filepath) and os.path.getsize(filepath):
        try:
            text = ""
            pdf_document = fitz.open(filepath)
            for page in pdf_document:
                text += page.get_text() + "\n"
            pdf_document.close()
            return text
        except:
            pass
    else:
        time.sleep(4)  # avoid CAPTCHA
        cmd = f'wget "{url}" --output-document "{filepath}" --quiet'
        os.system(cmd)
        try:
            text = ""
            pdf_document = fitz.open(filepath)
            for page in pdf_document:
                text += page.get_text() + "\n"
            pdf_document.close()
            return text
        except Exception as e:
            raise RuntimeError(f"{e} {url=}")


def _url2text_elsevier(url: str) -> str:
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


def _url2text_scholarsportal(url: str) -> str:
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


def _url2text(url: str) -> str:
    if url.startswith("https://www.sciencedirect.com"):
        result = _url2text_elsevier(url)
    elif url.startswith("https://journals.scholarsportal.info"):
        result = _url2text_scholarsportal(url)
    else:
        result = _url2text_wget(url)
    assert type(result) == str, f"{type(result)=}"
    return result


def _keyword2regex(keyword: str) -> str:
    result = "(?:\s+|\s*-\s*)".join([
        "-?\n?".join(list(w)) for w in keyword.split(' ')
    ])
    return result


def search_in_file(url: str, keywords: List[str]) -> Dict[str, int]:
    """
    Args:
        url (str): the url of the pdf file to search within.
        keywords (List[str]): the keywords to search within the file.
    Returns:
        counts (List[int]): the count of instances of each keyword in the file.
    """
    # input checks
    assert type(keywords) == list, f"{type(keywords)=}"
    assert all([type(k) == str and k != ""] for k in keywords)
    # extract lines
    text = _url2text(url)
    # search for keyword
    counts = {
        kw: len(re.findall(pattern=_keyword2regex(kw), string=text, flags=re.IGNORECASE))
        for kw in keywords
    }
    return counts
