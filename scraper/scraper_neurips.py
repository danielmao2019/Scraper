from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
from . import utils


def scrape_neurips(url):
    try:
        page = urlopen(url)
    except:
        raise RuntimeError(f"[ERROR] Exceptions raised when opening {url}.")
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # get title
    title = str(soup.findAll(name="h4")[0])[4:-5]
    # get year
    year = re.findall(pattern="/(\d\d\d\d)/", string=url)
    assert len(year) == 1
    year = f"`{year[0]}`"
    # get authors
    authors = str(soup.findAll(name="p")[1])[6:-8]
    # get abstract
    p_idx = 2 + (soup.findAll(name="p")[2].text.strip() == "")
    abstract = utils.post_process_abstract(soup.findAll(name="p")[p_idx].text.strip())
    # get pdf url
    pdf_url = url
    pdf_url = re.sub(pattern="hash", repl="file", string=pdf_url)
    pdf_url = re.sub(pattern="Abstract", repl="Paper", string=pdf_url)
    pdf_url = re.sub(pattern=".html", repl=".pdf", string=pdf_url)
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "NeurIPS",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
