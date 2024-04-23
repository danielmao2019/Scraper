from typing import Dict
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from . import utils


def scrape_acm(url: str) -> Dict[str, str]:
    response = requests.get(url)
    assert response.status_code == 200, f"{response.status_code=}"
    soup = BeautifulSoup(response.content, "html.parser")
    # get title
    title = soup.find('h1', class_="citation__title").text
    # get year
    year = soup.find('ul', class_="rlist article-chapter-history-list")
    year = [y.text for y in year.findAll('li') if y.text.startswith("Published: ")]
    assert len(year) == 1
    year = year[0][len("Published: "):]
    assert len(year.split(' ')) == 3, f"{year=}"
    year = datetime.strptime(year, "%d %B %Y")
    year = year.strftime("%d %b %Y")
    year = f"`{year}`"
    # get authors
    authors = soup.findAll('div', class_="author-data")
    authors = ", ".join([a.text for a in authors])
    # get abstract
    abstract = soup.find('div', class_="abstractSection abstractInFull").text
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': "",
        'pub_name': "ACM",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
