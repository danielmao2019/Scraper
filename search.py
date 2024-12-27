from typing import Tuple, List, Dict
import glob
import os
import re
import tqdm
import sys
sys.path.append("..")
from scrape import scrape
from search import search_in_file


def main(files: List[str], output_dir: str, keywords: List[str]) -> None:
    # get list of pdf urls
    content = ""
    for filepath in sum([sorted(glob.glob(f)) for f in files], start=[]):
        with open(filepath, mode='r', encoding='utf-8') as f:
            content += f.read()
    html_urls: List[str] = re.findall(pattern=r"\[abs-[^\]]*\]\(([^\)]*)\)", string=content)
    pdf_urls: List[str] = re.findall(pattern=r"\[pdf-[^\]]*\]\(([^\)]*)\)", string=content)
    assert len(html_urls) == len(pdf_urls)
    urls: List[Tuple[str, str]] = sorted(list(zip(html_urls, pdf_urls)), key=lambda x: x[0])
    urls = list(filter(lambda x: x[0] != "" and x[1] != "", urls))
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
        os.makedirs(os.path.join("search", "results", output_dir), exist_ok=True)
        with open(os.path.join("search", "results", output_dir, output_name), mode='w') as f:
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
