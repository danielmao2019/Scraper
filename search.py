from typing import Tuple, List, Dict
import os
import time
import re
import tqdm
from PyPDF2 import PdfReader


def url2text(url: str) -> str:
    """
    Returns:
        text (str): text in the file from url.
    """
    # avoid CAPTCHA
    time.sleep(4)
    # download pdf to temporary file
    os.system(' '.join([
        'wget', url, '--output-document', "tmp.pdf", '--quiet',
    ]))
    # extract lines
    reader = PdfReader("tmp.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    # cleanup
    os.system(' '.join([
        'rm', "tmp.pdf",
    ]))
    return text


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
    text = url2text(url)
    # search for keyword
    counts = {
        k: len(re.findall(pattern=k, string=text))
        for k in keywords
    }
    return counts


def main(filepath: str, keywords: List[str]) -> None:
    # get list of pdf urls
    with open(filepath, mode='r', encoding='utf-8') as f:
        content = f.read()
    all_pdf_urls = re.findall(pattern="\((http.+\.pdf)\)", string=content)
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
        except:
            failures.append(url)
    for k in keywords:
        with open(f"search_results_{os.path.basename(filepath).split('.')[0]}_{k}.md", mode='w') as f:
            f.write("".join(list(map(
                lambda x: str(x[0]) + ' ' + x[1] + '\n',
                sorted(results[k], key=lambda x: x[0], reverse=True),
            ))))
    print(f"Failure cases:")
    print('\n'.join(failures))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str)
    parser.add_argument('-k', '--keywords', nargs='+', default=[])
    args = parser.parse_args()
    main(filepath=args.filepath, keywords=args.keywords)
