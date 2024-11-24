from typing import Dict
from datetime import datetime
import utils


def scrape_acm(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get title
    title = soup.find('h1', class_="citation__title").text
    # get year
    year = soup.find('ul', class_="rlist article-chapter-history-list")
    year = [y.text for y in year.findAll('li') if y.text.startswith("Published: ")]
    assert len(year) == 1
    year = year[0][len("Published: "):]
    assert len(year.split(' ')) == 3, f"year={year}"
    year = datetime.strptime(year, "%d %B %Y").strftime("%d %b %Y")
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
