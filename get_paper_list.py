import os
from urllib.request import urlopen
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import argparse


class GetPaperList:

    papers_count_neurips = {
        2012: 370,
        2013: 360,
        2014: 411,
        2015: 403,
        2016: 569,
        2017: 679,
        2018: 1009,
        2019: 1428,
        2020: 1898,
        2021: 2334,
        2022: 2834,
    }

    paper_lists_root = "/home/d6mao/repos/Machine-Learning-Knowledge-Base/papers-cv/_paper_lists_"

    @staticmethod
    def get_papers_neurips(year: int = None):
        url = f"https://papers.nips.cc/paper_files/paper/{year}"
        page = urlopen(url)
        html = page.read().decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        papers = soup.findAll(name='a', title="paper title")
        assert len(papers) == GetPaperList.papers_count_neurips[year], f"{len(papers)=}"
        filepath = os.path.join(GetPaperList.paper_lists_root, f"neurips/neurips{year}.txt")
        with open(filepath, mode='w') as f:
            for a in papers:
                f.write(a.text.strip() + '\n')
                f.write(urljoin(url, a['href']) + '\n')
                f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conference', type=str)
    parser.add_argument('-y', '--year', type=int)
    args = parser.parse_args()
    method = getattr(GetPaperList, f"get_papers_{args.conference}")
    method(args.year)
