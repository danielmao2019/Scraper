from typing import Tuple, List, Dict
import glob
import os
import time
import re
import tqdm
import hashlib
import fitz  # PyMuPDF
import sys
sys.path.append("..")
from scraper.scrape import scrape


def _url2text_wget(url: str) -> str:
    """
    Returns:
        text (str): text in the file from url.
    """
    filename = hashlib.sha256(url.encode('utf-8')).hexdigest() + ".pdf"
    os.makedirs("downloads", exist_ok=True)
    filepath = os.path.join("downloads", filename)
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


def _url2text(url: str) -> str:
    if "sciencedirect" in url:
        result = _url2text_elsevier(url)
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


def main(files: List[str], output_dir: str, keywords: List[str]) -> None:
    # get list of pdf urls
    content = ""
    for filepath in sum([sorted(glob.glob(f)) for f in files], start=[]):
        with open(filepath, mode='r', encoding='utf-8') as f:
            content += f.read()
    html_urls: List[str] = re.findall(pattern=r"\[abs-[^\]]*\]\((http.+)\)", string=content)
    pdf_urls: List[str] = re.findall(pattern=r"\[pdf-[^\]]*\]\((http.+)\)", string=content)
    assert len(html_urls) == len(pdf_urls)
    urls: List[Tuple[str, str]] = sorted(list(zip(html_urls, pdf_urls)), key=lambda x: x[0])
    print(f"Found {len(urls)} urls.")
    # search relevant documents
    failures: List[str] = []
    results: Dict[str, List[Tuple[int, str]]] = {kw: [] for kw in keywords}
    for url in tqdm.tqdm(urls):
        try:
            counts: Dict[str, int] = search_in_file(url=url[1], keywords=keywords)
            info: str = scrape(url[0])
            for kw in keywords:
                if counts[kw] > 0:
                    results[kw].append((counts[kw], info))
        except Exception as e:
            print(e)
            print(f"{url=}")
            failures.append(url[0]+'\n'+url[1])
    # save to disk
    for kw in keywords:
        output_name = re.sub(pattern=' ', repl='_', string=kw) + ".md"
        os.makedirs(name=os.path.join("results", output_dir), exist_ok=True)
        with open(os.path.join("results", output_dir, output_name), mode='w') as f:
            f.write("".join(list(map(
                lambda x: x[1] + '\n' + f"    Count: {x[0]}",
                sorted(results[kw], key=lambda x: x[0], reverse=True),
            ))))
    # logging
    print(f"Failure cases:")
    print('\n'.join(failures))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+', default=[])
    parser.add_argument('-d', '--output-dir', type=str)
    parser.add_argument('-k', '--keywords', nargs='+', default=[])
    args = parser.parse_args()
    main(files=args.files, output_dir=args.output_dir, keywords=args.keywords)
