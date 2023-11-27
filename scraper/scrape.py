from scraper.scraper_arxiv import scrape_arxiv
from scraper.scraper_eccv import scrape_eccv
from scraper.scraper_ieee import scrape_ieee
from scraper.scraper_mdpi import scrape_mdpi
from scraper.scraper_neurips import scrape_neurips
from scraper.scraper_openaccess import scrape_openaccess
from scraper.scraper_openreview import scrape_openreview
from scraper.scraper_springer import scrape_springer
from . import utils


def scrape(url):
    info_dict = None
    if url.startswith("https://arxiv.org"):
        info_dict = scrape_arxiv(url)
    if url.startswith("https://openaccess.thecvf.com"):
        info_dict = scrape_openaccess(url)
    if url.startswith("https://openreview.net"):
        info_dict = scrape_openreview(url)
    if url.startswith("https://ieeexplore.ieee.org"):
        info_dict = scrape_ieee(url)
    if url.startswith("https://link.springer.com"):
        info_dict = scrape_springer(url)
    if url.startswith("https://papers.nips.cc"):
        info_dict = scrape_neurips(url)
    if url.startswith("https://www.ecva.net"):
        info_dict = scrape_eccv(url)
    if url.startswith("https://www.mdpi.com"):
        info_dict = scrape_mdpi(url)
    if info_dict is not None:
        return utils.compile_markdown(**info_dict)
    else:
        raise ValueError(f"No scraper implemented for {url}.")
