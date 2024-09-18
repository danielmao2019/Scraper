from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
from datetime import datetime
from . import utils


def scrape_pmlr (url: str) -> dict:
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # get title
    title = soup.find('title').text
    # get pdf url
    pdf_url = re.findall(pattern="content=\"([^\"]+)\" name=\"citation_pdf_url\"", string=soup.__str__())
    assert len(pdf_url) == 1
    pdf_url = pdf_url[0]
    # get year
    year = re.findall(pattern="content=\"([^\"]+)\" name=\"citation_publication_date\"", string=soup.__str__())
    assert len(year) == 1
    year = datetime.strptime(year[0], "%Y/%m/%d")
    year = year.strftime("%d %b %Y")
    date, month, year = year.split(' ')
    year = f"{date}, {month}, `{year}`"
    # get authors
    authors = re.findall(pattern="content=\"([^\"]+)\" name=\"citation_author\"", string=soup.__str__())
    authors = ", ".join(authors)
    # get abstract
    abstract = soup.find('div', class_="abstract").text
    abstract = utils.post_process_abstract(abstract)
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "PMLR",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
