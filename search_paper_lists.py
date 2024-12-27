from typing import Tuple, List, Dict
import glob
import os
import re
import tqdm
from search import search_in_file
from scrape import scrape


PAPER_LISTS_ROOT: str = "/home/d_mao/repos/Machine-Learning-Knowledge-Base/paper-collections/deep-learning/_paper_lists_"


def html2pdf(url: str) -> str:
    result = url
    if url.startswith("https://openaccess.thecvf.com"):
        result = re.sub(pattern=r"/html/", repl="/papers/", string=result)
        result = re.sub(pattern=r"\.html", repl=".pdf", string=result)
        result = re.sub(pattern=r"content_iccv", repl="content_ICCV", string=result)
        result = re.sub(pattern=r"content_ICCV_2013", repl="content_iccv_2013", string=result)
        result = re.sub(pattern=r"content_ICCV_2015", repl="content_iccv_2015", string=result)
    elif url.startswith("https://papers.nips.cc"):
        result = re.sub(pattern=r"/hash/", repl="/file/", string=result)
        result = re.sub(pattern=r"-Abstract", repl="-Paper", string=result)
        result = re.sub(pattern=r"\.html", repl=".pdf", string=result)
    elif url.startswith("https://journals.scholarsportal.info"):
        result = url
    else:
        result = ""
    return result


def main(venues: List[str], output_dir: str, keywords: List[str]) -> None:
    # initialization
    if len(venues) == 0:
        venues = ['cvpr', 'iccv', 'neurips', 'automation_in_construction']
    os.makedirs(name=os.path.join("search", "results", output_dir), exist_ok=True)
    failures: List[Tuple[str, str]] = []
    # main loop
    for venue in venues:
        files: List[str] = sorted(glob.glob(os.path.join(PAPER_LISTS_ROOT, venue, "*.txt")), reverse=True)
        result: Dict[str, List[Tuple[int, str]]] = {kw: [] for kw in keywords}
        for filepath in files:
            # get html and pdf urls
            with open(filepath, mode='r') as f:
                html_urls = re.findall(pattern=r"https.+\.(?:html|xml)", string=f.read())
            pdf_urls = [html2pdf(url) for url in html_urls]
            urls: List[Tuple[str, str]] = list(zip(html_urls, pdf_urls))
            urls = filter(lambda x: x[1] != "", urls)
            urls = sorted(set(urls), key=lambda x: x[0])
            # search and scrape
            for url in tqdm.tqdm(urls, desc=os.path.basename(filepath).split('.')[0]):
                html_url, pdf_url = url
                try:
                    counts: Dict[str, int] = search_in_file(url=pdf_url, keywords=keywords)
                    info: str = scrape(html_url)
                    for kw in keywords:
                        if counts[kw] > 0:
                            result[kw].append((counts[kw], info))
                except Exception as e:
                    print(e)
                    failures.append(url)
        # save to disk
        for kw in keywords:
            output_name = re.sub(pattern=' ', repl='_', string=kw) + f"_{venue}" + ".md"
            kw_result = "".join(list(map(
                lambda x: f"count={x[0]}\n" + x[1],
                sorted(result[kw], key=lambda x: x[0], reverse=True),
            )))
            with open(os.path.join("search", "results", output_dir, output_name), mode='w') as f:
                f.write(kw_result)
    # logging
    print(f"Failure cases:")
    print('\n'.join(list(map(lambda x: x[0]+'\n'+x[1], failures))))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--venues', nargs='+', default=[])
    parser.add_argument('-d', '--output-dir', type=str)
    parser.add_argument('-k', '--keywords', nargs='+', default=[])
    args = parser.parse_args()
    main(venues=args.venues, output_dir=args.output_dir, keywords=args.keywords)
