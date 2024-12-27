"""
SCRAPE.SCRAPERS API
"""
from scrape.scrapers.scraper_aaai import scrape_aaai
from scrape.scrapers.scraper_acl import scrape_acl
from scrape.scrapers.scraper_acm import scrape_acm
from scrape.scrapers.scraper_arxiv import scrape_arxiv
from scrape.scrapers.scraper_asce import scrape_asce
from scrape.scrapers.scraper_eccv import scrape_eccv
from scrape.scrapers.scraper_ieee import scrape_ieee
from scrape.scrapers.scraper_isprs import scrape_isprs
from scrape.scrapers.scraper_jmlr import scrape_jmlr
from scrape.scrapers.scraper_mdpi import scrape_mdpi
from scrape.scrapers.scraper_neurips import scrape_neurips
from scrape.scrapers.scraper_openaccess import scrape_openaccess
from scrape.scrapers.scraper_openreview import scrape_openreview
from scrape.scrapers.scraper_pmlr import scrape_pmlr
from scrape.scrapers.scraper_pubmed import scrape_pubmed
from scrape.scrapers.scraper_researchgate import scrape_researchgate
from scrape.scrapers.scraper_robotics import scrape_robotics
from scrape.scrapers.scraper_scholarsportal import scrape_scholarsportal
from scrape.scrapers.scraper_sciencedirect import scrape_sciencedirect
from scrape.scrapers.scraper_springer import scrape_springer
from scrape.scrapers.scraper_tandfonline import scrape_tandfonline


__all__ = (
    'scrape_aaai',
    'scrape_acl',
    'scrape_acm',
    'scrape_arxiv',
    'scrape_asce',
    'scrape_eccv',
    'scrape_ieee',
    'scrape_isprs',
    'scrape_jmlr',
    'scrape_mdpi',
    'scrape_neurips',
    'scrape_openaccess',
    'scrape_openreview',
    'scrape_pmlr',
    'scrape_pubmed',
    'scrape_researchgate',
    'scrape_robotics',
    'scrape_scholarsportal',
    'scrape_sciencedirect',
    'scrape_springer',
    'scrape_tandfonline',
)
