from typing import Dict, Any
import re
from scrape import utils


def scrape_neurips(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get title
    title = str(soup.findAll(name="h4")[0])[4:-5]
    # get pub year
    year = re.findall(pattern="/(\d\d\d\d)/", string=url)
    assert len(year) == 1
    year = year[0]
    # get authors
    authors = str(soup.findAll(name="p")[1])[6:-8]
    authors = authors.split(", ")
    # get abstract
    p_idx = 2 + (soup.findAll(name="p")[2].text.strip() == "")
    abstract = soup.findAll(name="p")[p_idx].text.strip()
    # get pdf url
    pdf_url = url
    pdf_url = re.sub(pattern="hash", repl="file", string=pdf_url)
    pdf_url = re.sub(pattern="Abstract", repl="Paper", string=pdf_url)
    pdf_url = re.sub(pattern=".html", repl=".pdf", string=pdf_url)
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': "NeurIPS",
        'pub_date': year,
        'authors': authors,
        'abstract': abstract,
    }
