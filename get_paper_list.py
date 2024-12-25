from typing import List, Optional
import os
from urllib.parse import urljoin
import bs4
import re
import argparse
from utils.soup import get_soup


class GetPaperList:

    papers_count_cvpr = {
        2013: 471,
        2014: 540,
        2015: 602,
        2016: 643,
        2017: 783,
        2018: 367 + 370 + 242,
        2019: 431 + 432 + 431,
        2020: 483 + 480 + 503,
        2021: 1660,
        2022: 2074,
        2023: 2353,
        2024: 2716,
    }

    papers_count_iccv = {
        2013: 454,
        2015: 526,
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
        2022: 406,
        2023: 639,
        2024: 846,
    }

    papers_count_accv = {
        2020: 254,
        2022: 277,
        2024: 269,
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
        2023: 3540,
    }

    # assume this file is located under a directory ("Scraper") that is in parallel to "Machine-Learning-Knowledge-Base"
    paper_lists_root = "../Machine-Learning-Knowledge-Base/paper-collections/deep-learning/_paper_lists_"

    # ==================================================
    # CVF
    # ==================================================

    @staticmethod
    def _get_urls_cvf_post_processing(result: List[str], base_url: str) -> List[str]:
        result = list(map(lambda x: x['href'], result))
        result = list(map(lambda x: urljoin(base_url, x), result))
        result = list(map(lambda x: re.sub(pattern=".py", repl="", string=x), result))
        return result

    @staticmethod
    def _get_urls_cvf_conference(conference: str, year: int) -> List[str]:
        base_url = f"https://openaccess.thecvf.com/{conference.upper()}{year}"
        soup = get_soup(base_url)
        if len(soup.findAll(name='dt')):
            return [base_url]
        result = soup.find(name='div', id="content").findAll('a')
        if "day=all" in result[-1]['href']:
            result = result[-1:]
        result = GetPaperList._get_urls_cvf_post_processing(result, base_url)
        return result

    @staticmethod
    def _get_urls_cvf_workshop(workshop: str, year: int) -> List[str]:
        base_url = f"https://openaccess.thecvf.com/{workshop[:-1].upper()}{year}_workshops/menu"
        soup = get_soup(base_url)
        result = soup.findAll(name='dd')
        result = list(map(lambda x: x.findAll(name='a'), result))
        assert all([len(x) == 1 for x in result])
        result = list(map(lambda x: x[0], result))
        result = GetPaperList._get_urls_cvf_post_processing(result, base_url)
        return result

    @staticmethod
    def _get_papers_cvf(urls: List[str], filepath: str, expected_count: Optional[int] = None) -> None:
        # get papers
        papers = []
        for url in urls:
            soup = get_soup(url)
            papers += soup.findAll(name='dt', attrs={'class': "ptitle"})
        papers = list(map(lambda x: x.findAll(name='a'), papers))
        assert all([len(x) == 1 for x in papers])
        papers = list(map(lambda x: x[0], papers))
        assert all([type(x) == bs4.element.Tag for x in papers])
        # sanity check
        if expected_count is not None:
            assert len(papers) == expected_count
        # save to disk
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode='w') as f:
            for a_tag in papers:
                f.write(a_tag.text.strip() + '\n')
                f.write(urljoin("https://openaccess.thecvf.com", a_tag['href']) + '\n')
                f.write('\n')

    def get_papers_cvf_conference(self, conference: str, year: int) -> None:
        self._get_papers_cvf(
            urls=self._get_urls_cvf_conference(conference, year),
            filepath=os.path.join(self.paper_lists_root, f"{conference}/{conference}{year}.txt"),
            expected_count=getattr(self, f"papers_count_{conference}")[year],
        )

    def get_papers_cvf_workshop(self, workshop: str, year: int):
        self._get_papers_cvf(
            urls=self._get_urls_cvf_workshop(workshop, year),
            filepath=os.path.join(self.paper_lists_root, f"{workshop}/{workshop}{year}.txt"),
            expected_count=None,
        )

    def get_papers_cvpr(self, year: int):
        self.get_papers_cvf_conference("cvpr", year)

    def get_papers_cvprw(self, year: int):
        self.get_papers_cvf_workshop("cvprw", year)

    def get_papers_iccv(self, year: int):
        self.get_papers_cvf_conference("iccv", year)

    def get_papers_iccvw(self, year: int):
        self.get_papers_cvf_workshop("iccvw", year)

    def get_papers_eccv(self, year: int):
        self.get_papers_cvf_conference("eccv", year)

    def get_papers_accv(self, year: int):
        self.get_papers_cvf_conference("accv", year)

    def get_papers_wacv(self, year: int):
        self.get_papers_cvf_conference("wacv", year)

    # ==================================================
    # NeurIPS
    # ==================================================

    def get_url_neurips(self, year: int):
        return f"https://papers.nips.cc/paper_files/paper/{year}"

    def get_papers_neurips(self, year: int):
        url = self.get_url_neurips(year)
        soup = get_soup(url)
        papers = soup.findAll(name='a', title="paper title")
        assert len(papers) == self.papers_count_neurips[year], f"len(papers)={len(papers)}"
        filepath = os.path.join(self.paper_lists_root, f"neurips/neurips{year}.txt")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode='w') as f:
            for a in papers:
                f.write(a.text.strip() + '\n')
                f.write(urljoin(url, a['href']) + '\n')
                f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--venue', type=str)
    parser.add_argument('-y', '--year', type=int)
    args = parser.parse_args()
    getter = GetPaperList()
    method = getattr(getter, f"get_papers_{args.venue}")
    method(args.year)
