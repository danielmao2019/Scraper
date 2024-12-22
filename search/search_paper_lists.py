from typing import Tuple, List, Dict
import os
import re
import tqdm
from search import search_in_file


def html2pdf(url: str) -> str:
    result = url
    if url.startswith("https://openaccess.thecvf.com"):
        result = re.sub(pattern=r"/html/", repl="/papers/", string=result)
        result = re.sub(pattern=r"\.html", repl=".pdf", string=result)
        result = re.sub(pattern=r"content_iccv", repl="content_ICCV", string=result)
    elif url.startswith("https://papers.nips.cc"):
        result = re.sub(pattern=r"/hash/", repl="/file/", string=result)
        result = re.sub(pattern=r"-Abstract", repl="-Paper", string=result)
        result = re.sub(pattern=r"\.html", repl=".pdf", string=result)
    else:
        result = ""
    return result


def main(files: List[str], output_dir: str, keywords: List[str]) -> None:
    # get list of pdf urls
    all_pdf_urls: List[str] = []
    for filepath in files:
        with open(filepath, mode='r') as f:
            html_urls = re.findall(pattern=r"https.+\.html", string=f.read())
            pdf_urls = [html2pdf(url) for url in html_urls]
            pdf_urls = filter(lambda x: x != "", pdf_urls)
            all_pdf_urls += pdf_urls
    all_pdf_urls = sorted(set(all_pdf_urls))
    # search
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
