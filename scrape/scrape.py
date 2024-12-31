from typing import Dict, Union, Optional
from scrape import scrapers
from scrape import utils


def scrape(url: str, compile: Optional[bool] = True) -> Union[Dict[str, str], str]:
    info_dict = None
    if url.startswith("https://arxiv.org"):
        info_dict = scrapers.scrape_arxiv(url)
    if url.startswith("https://ascelibrary.org"):
        info_dict = scrapers.scrape_asce(url)
    if url.startswith("https://ojs.aaai.org"):
        info_dict = scrapers.scrape_aaai(url)
    if url.startswith("https://aclanthology.org"):
        info_dict = scrapers.scrape_acl(url)
    if url.startswith("https://dl.acm.org/doi"):
        info_dict = scrapers.scrape_acm(url)
    if url.startswith("https://www.ecva.net"):
        info_dict = scrapers.scrape_eccv(url)
    if url.startswith("https://ieeexplore.ieee.org"):
        info_dict = scrapers.scrape_ieee(url)
    if url.startswith("https://isprs"):
        info_dict = scrapers.scrape_isprs(url)
    if url.startswith("https://www.jmlr.org") or url.startswith("https://jmlr.org"):
        info_dict = scrapers.scrape_jmlr(url)
    if url.startswith("https://www.mdpi.com"):
        info_dict = scrapers.scrape_mdpi(url)
    if url.startswith("https://papers.nips.cc") or url.startswith("https://proceedings.neurips.cc"):
        info_dict = scrapers.scrape_neurips(url)
    if url.startswith("https://openaccess.thecvf.com"):
        info_dict = scrapers.scrape_openaccess(url)
    if url.startswith("https://openreview.net"):
        info_dict = scrapers.scrape_openreview(url)
    if url.startswith("https://proceedings.mlr.press"):
        info_dict = scrapers.scrape_pmlr(url)
    if url.startswith("https://pubmed.ncbi.nlm.nih.gov"):
        info_dict = scrapers.scrape_pubmed(url)
    if url.startswith("https://www.researchgate.net"):
        info_dict = scrapers.scrape_researchgate(url)
    if url.startswith("https://www.roboticsproceedings.org"):
        info_dict = scrapers.scrape_robotics(url)
    if url.startswith("https://www.sciencedirect.com"):
        info_dict = scrapers.scrape_sciencedirect(url)
    if url.startswith("https://journals.scholarsportal.info"):
        info_dict = scrapers.scrape_scholarsportal(url)
    if url.startswith("https://link.springer.com"):
        info_dict = scrapers.scrape_springer(url)
    if url.startswith("https://www.tandfonline.com"):
        info_dict = scrapers.scrape_tandfonline(url)
    if info_dict is not None:
        if compile:
            return utils.compile_markdown(**info_dict)
        else:
            return info_dict
    else:
        raise ValueError(f"No scraper implemented for {url}.")
