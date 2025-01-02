from typing import Dict, Any
import re
from scrape import utils


def scrape_arxiv(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    with open("tmp.html", mode='w') as f:
        f.write(str(soup))
    # generate links
    idx = url.split('/')[-1]
    pdf_url = f"https://arxiv.org/pdf/{idx}.pdf"
    # get title
    title = soup.find("h1", class_="title mathjax").text.strip()
    title = re.sub(pattern=r"Title: *", repl="", string=title)
    # get year
    pub_date = soup.find("div", class_="dateline").text.strip()
    pub_date = re.findall(pattern=r"Submitted on \d+ \w{3} \d{4}", string=pub_date)[0]
    pub_date = re.sub(pattern=r"Submitted on ", repl="", string=pub_date)
    # get authors
    authors = utils.soup.extract_authors(soup)
    # get abstract
    abstract = soup.find("blockquote", class_="abstract mathjax").text.strip()
    abstract = re.sub(pattern=r"Abstract: *", repl="", string=abstract)
    abstract = re.sub(pattern=r"\n+", repl=" ", string=abstract)
    # return
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': None,
        'pub_date': pub_date,
        'authors': authors,
        'abstract': abstract,
    }
