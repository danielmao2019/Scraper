from scraper.scraper_arxiv import scrape_arxiv
from scraper.scraper_aaai import scrape_aaai
from scraper.scraper_acl import scrape_acl
from scraper.scraper_acm import scrape_acm
from scraper.scraper_eccv import scrape_eccv
from scraper.scraper_ieee import scrape_ieee
from scraper.scraper_jmlr import scrape_jmlr
from scraper.scraper_mdpi import scrape_mdpi
from scraper.scraper_neurips import scrape_neurips
from scraper.scraper_openaccess import scrape_openaccess
from scraper.scraper_openreview import scrape_openreview
from scraper.scraper_pmlr import scrape_pmlr
from scraper.scraper_pubmed import scrape_pubmed
from scraper.scraper_robotics import scrape_robotics
from scraper.scraper_sciencedirect import scrape_sciencedirect
from scraper.scraper_springer import scrape_springer

import utils


def scrape(url: str) -> str:
    info_dict = None
    if url.startswith("https://arxiv.org"):
        info_dict = scrape_arxiv(url)
    if url.startswith("https://ojs.aaai.org"):
        info_dict = scrape_aaai(url)
    if url.startswith("https://aclanthology.org"):
        info_dict = scrape_acl(url)
    if url.startswith("https://dl.acm.org/doi"):
        info_dict = scrape_acm(url)
    if url.startswith("https://www.ecva.net"):
        info_dict = scrape_eccv(url)
    if url.startswith("https://ieeexplore.ieee.org"):
        info_dict = scrape_ieee(url)
    if url.startswith("https://www.jmlr.org") or url.startswith("https://jmlr.org/"):
        info_dict = scrape_jmlr(url)
    if url.startswith("https://www.mdpi.com"):
        info_dict = scrape_mdpi(url)
    if url.startswith("https://papers.nips.cc") or url.startswith("https://proceedings.neurips.cc"):
        info_dict = scrape_neurips(url)
    if url.startswith("https://openaccess.thecvf.com"):
        info_dict = scrape_openaccess(url)
    if url.startswith("https://openreview.net"):
        info_dict = scrape_openreview(url)
    if url.startswith("https://proceedings.mlr.press"):
        info_dict = scrape_pmlr(url)
    if url.startswith("https://pubmed.ncbi.nlm.nih.gov"):
        info_dict = scrape_pubmed(url)
    if url.startswith("https://www.roboticsproceedings.org"):
        info_dict = scrape_robotics(url)
    if url.startswith("https://www.sciencedirect.com"):
        info_dict = scrape_sciencedirect(url)
    if url.startswith("https://link.springer.com"):
        info_dict = scrape_springer(url)
    if info_dict is not None:
        return utils.compile_markdown(**info_dict)
    else:
        raise ValueError(f"No scraper implemented for {url}.")
