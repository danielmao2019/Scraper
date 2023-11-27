from urllib.request import urlopen
from bs4 import BeautifulSoup
import re


def scrape_arxiv(url: str) -> dict:
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # generate links
    idx = url.split('/')[-1]
    pdf_url = f"https://arxiv.org/pdf/{idx}.pdf"
    # get title
    title = soup.find("h1", class_="title mathjax").text.strip()
    title = re.sub(pattern=r"Title: *", repl="", string=title)
    # get year
    year = soup.find("div", class_="dateline").text.strip()
    year = re.findall(pattern=r"Submitted on \d+ \w{3} \d{4}", string=year)[0]
    year = re.sub(pattern=r"Submitted on ", repl="", string=year)
    year = year.split(' ')
    year[0] = '0' * (2-len(year[0])) + year[0]
    year[2] = '`' + year[2] + '`'
    year = ' '.join(year)
    # get authors
    authors = soup.find("div", class_="authors").text.strip()
    authors = re.sub(pattern=r"Authors: *", repl="", string=authors)
    # get abstract
    abstract = soup.find("blockquote", class_="abstract mathjax").text.strip()
    abstract = re.sub(pattern=r"Abstract: *", repl="", string=abstract)
    abstract = re.sub(pattern=r"\n+", repl=" ", string=abstract)
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "arXiv",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
