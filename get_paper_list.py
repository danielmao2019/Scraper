from typing import List
import os
from urllib.request import urlopen
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import re
import argparse


class GetPaperList:

    papers_count_cvpr = {
        2013: 471,
        2014: 540,
        2016: 643,
        2017: 783,
        2018: 367 + 370 + 242,
        2019: 431 + 432 + 431,
        2020: 483 + 480 + 503,
        2021: 1660,
        2022: 2074,
        2023: 2353,
    }

    papers_count_iccv = {
        2017: 621,
        2019: 294 + 153 + 318 + 310,
        2021: 1612,
        2023: 2156,
    }

    papers_count_eccv = {
        2018: 776,
    }

    papers_count_wacv = {
        2020: 378,
        2021: 406,
    }

    papers_count_accv = {
        2020: 254,
        2022: 277,
    }

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

    # assume this file is located under a directory ("Scraper") that is in parallel to "Machine-Learning-Knowledge-Base"
    paper_lists_root = "../Machine-Learning-Knowledge-Base/paper-collections/papers-cv/_paper_lists_"

    def get_urls_cvf(
        self,
        conference: str = None,
        year: int = None,
    ) -> List[str]:
        base_url = f"https://openaccess.thecvf.com/{conference.upper()}{year}"
        page = urlopen(base_url)
        html = page.read().decode("utf-8")
        if ".pdf" in html:
            return [base_url]
        soup = BeautifulSoup(html, "html.parser")
        content = soup.find(name='div', id="content")
        urls = content.findAll('a')
        if "day=all" in urls[-1]['href']:
            urls = urls[-1:]
        urls = [re.sub(pattern=".py", repl="", string=url['href']) for url in urls]
        return [urljoin(base_url, url) for url in urls]

    def get_url_neurips(self, year: int = None):
        return f"https://papers.nips.cc/paper_files/paper/{year}"

    def get_papers_cvf(
        self,
        conference: str = None,
        year: int = None,
    ) -> None:
        urls = self.get_urls_cvf(conference, year)
        papers = []
        for url in urls:
            page = urlopen(url)
            html = page.read().decode("utf-8")
            soup = BeautifulSoup(html, "html.parser")
            papers += soup.findAll(name='dt', class_="ptitle")
        assert len(papers) == getattr(self, f"papers_count_{conference}")[year], f"{len(papers)=}"
        filepath = os.path.join(self.paper_lists_root, f"{conference}/{conference}{year}.txt")
        with open(filepath, mode='w') as f:
            for dt in papers:
                a = dt.find('a')
                f.write(a.text.strip() + '\n')
                f.write(urljoin("https://openaccess.thecvf.com", a['href']) + '\n')
                f.write('\n')

    def get_papers_cvpr(self, year: int = None):
        self.get_papers_cvf("cvpr", year)

    def get_papers_iccv(self, year: int = None):
        self.get_papers_cvf("iccv", year)

    def get_papers_eccv(self, year: int = None):
        self.get_papers_cvf("eccv", year)

    def get_papers_accv(self, year: int = None):
        self.get_papers_cvf("accv", year)

    def get_papers_wacv(self, year: int = None):
        self.get_papers_cvf("wacv", year)

    def get_papers_neurips(self, year: int = None):
        url = self.get_url_neurips(year)
        page = urlopen(url)
        html = page.read().decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        papers = soup.findAll(name='a', title="paper title")
        assert len(papers) == self.papers_count_neurips[year], f"{len(papers)=}"
        filepath = os.path.join(self.paper_lists_root, f"neurips/neurips{year}.txt")
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
    getter = GetPaperList()
    method = getattr(getter, f"get_papers_{args.conference}")
    method(args.year)
