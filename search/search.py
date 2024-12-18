from typing import Tuple, List, Dict
import glob
import os
import time
import re
import tqdm
import fitz  # PyMuPDF


def _url2text_wget(url: str) -> str:
    """
    Returns:
        text (str): text in the file from url.
    """
    # avoid CAPTCHA
    time.sleep(4)

    # Download the PDF
    os.system('wget {} --output-document tmp.pdf --quiet'.format(url))

    # Extract text
    text = ""
    pdf_document = fitz.open("tmp.pdf")
    for page in pdf_document:
        text += page.get_text() + "\n"
    pdf_document.close()

    # Cleanup
    os.remove("tmp.pdf")
    return text


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
        return _url2text_elsevier(url)
    else:
        return _url2text_wget(url)


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
    all_pdf_urls = re.findall(pattern=r"\[pdf-.+\]\((http.+)\)", string=content)
    all_pdf_urls = set(all_pdf_urls)
    print(f"Found {len(all_pdf_urls)} pdf urls.")
    # search relevant documents
    failures: List[str] = []
    results: Dict[str, List[Tuple[int, str]]] = {
        k: [] for k in keywords
    }
    for url in tqdm.tqdm(all_pdf_urls):
        try:
            counts: Dict[str, int] = search_in_file(url=url, keywords=keywords)
            for k in keywords:
                if counts[k] > 0:
                    results[k].append((counts[k], url))
        except Exception as e:
            print(e)
            failures.append(url)
    # save to disk
    for k in keywords:
        output_name = re.sub(pattern=' ', repl='_', string=k) + ".md"
        os.makedirs(name=os.path.join("results", output_dir), exist_ok=True)
        with open(os.path.join("results", output_dir, output_name), mode='w') as f:
            f.write("".join(list(map(
                lambda x: str(x[0]) + ' ' + x[1] + '\n',
                sorted(results[k], key=lambda x: x[0], reverse=True),
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
