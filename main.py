import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import urljoin
import requests

from code_names_mapping import mapping
from papers_tmp import papers_tmp


import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


INDENT = ' ' * 4


def get_all_urls(src):
    with open(src, mode='r', encoding='utf8') as f:
        text = f.read()
        url_list = re.findall(pattern=r"https://arxiv\.org/abs/\d+\.\d+", string=text)
    return url_list


def scrape_arxiv(url):
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # generate links
    idx = url.split('/')[-1]
    url_abs = url
    url_pdf = f"https://arxiv.org/pdf/{idx}.pdf"
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
    # define string
    string = ""
    string += f"* {mapping.get(title, title)}" + '\n'
    string += INDENT + f"[[abs-arXiv]({url_abs})]" + '\n'
    string += INDENT + f"[[pdf-arXiv]({url_pdf})]" + '\n'
    string += INDENT + "* Title: " + title + '\n'
    string += INDENT + "* Year: " + year + '\n'
    string += INDENT + "* Authors: " + authors + '\n'
    string += INDENT + "* Abstract: " + abstract + '\n'
    return string


def _get_pdf_url_(
    abs_url: str = None,
    rel_pdf_url: str = None,
) -> str:
    pdf_url = urljoin(base=abs_url, url=rel_pdf_url)
    # check if new url exists
    r = requests.get(pdf_url)
    assert r.status_code == 200, f"{r.status_code=}, {abs_url=}, {pdf_url=}"
    return pdf_url


def _parse_conference_(conference):
    r"""
    Args:
        conference (str)
    Returns:
        conf_name (str): name of conference.
        conf_year (int): year of conference.
    """
    # get name
    conf_name = re.findall(pattern="(cvpr|iccv|wacv|iclr)", string=conference.lower())
    assert len(conf_name) == 1
    conf_name = conf_name[0].upper()
    # get year
    conf_year = re.findall(pattern=r"\d\d\d\d", string=conference)
    assert len(conf_year) == 1, f"{conference=}, {conf_year=}"
    conf_year = int(conf_year[0])
    return conf_name, conf_year


def _remove_quotes_(string):
    if string.startswith('\"'):
        string = string[1:]
    if string.endswith('\"'):
        string = string[:-1]
    return string


def _compile_markdown_(
    title: str = None,
    abs_url: str = None,
    pdf_url: str = None,
    conf_name: str = None,
    conf_year: int = None,
    authors: str = None,
    abstract: str = None,
) -> str:
    string = ""
    string += f"* {mapping.get(title, title)}\n"
    string += f"{INDENT}[[abs-{conf_name}]({abs_url})]\n"
    string += f"{INDENT}[[pdf-{conf_name}]({pdf_url})]\n"
    string += f"{INDENT}* Title: {title}\n"
    string += f"{INDENT}* Year: `{conf_year}`\n"
    string += f"{INDENT}* Authors: {authors}\n"
    string += f"{INDENT}* Abstract: {abstract}\n"
    return string


def scrape_openaccess(url):
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # generate links
    rel_pdf_url = soup.find("a", string="pdf")['href']
    pdf_url = _get_pdf_url_(url, rel_pdf_url)
    # get title
    title = soup.find("div", id="papertitle").text.strip()
    # get conference
    conference = url.split('/')[-3]
    conf_name, conf_year = _parse_conference_(conference)
    # get authors
    authors = soup.find("div", id="authors").text.strip().split(';')[0]
    # get abstract
    abstract = soup.find("div", id="abstract").text.strip()
    abstract = re.sub(pattern='\n', repl=" ", string=abstract)
    # compile markdown
    markdown = _compile_markdown_(title, url, pdf_url, conf_name, conf_year, authors, abstract)
    return markdown


def scrape_openreview(url):
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # generate links
    rel_pdf_url = soup.find('a', title="Download PDF")['href']
    pdf_url = _get_pdf_url_(url, rel_pdf_url)
    # get title
    title = soup.find('h2', class_="note_content_title citation_title").text.strip()
    # get conference
    conference = soup.findAll('div', class_="meta_row")[1].findAll('span', class_="item")[1].text.strip()
    conf_name, conf_year = _parse_conference_(conference)
    # get authors
    authors = soup.find('h3', class_="signatures author").findAll('a')
    authors = ", ".join([a.text.strip() for a in authors])
    # get abstract
    abstract = soup.findAll('span', class_="note-content-value")[1].text.strip()
    # compile markdown
    markdown = _compile_markdown_(title, url, pdf_url, conf_name, conf_year, authors, abstract)
    return markdown


def scrape_eccv(url):
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # generate links
    rel_pdf_url = soup.find("a", string="pdf")['href']
    pdf_url = _get_pdf_url_(url, rel_pdf_url)
    # get title
    title = soup.find("div", id="papertitle").text.strip()
    # get year
    pattern = "eccv_(\d\d\d\d)"
    year = re.findall(pattern=pattern, string=url)
    assert len(year) == 1, f"{url=}, {pattern=}, {year=}"
    year = year[0]
    assert 2000 <= int(year) <= 2030
    year = f"`{year}`"
    # get authors
    authors = soup.find("div", id="authors").text.strip()
    authors = re.sub(pattern='\n', repl="", string=authors)
    # get abstract
    abstract = soup.find("div", id="abstract").text.strip()
    abstract = re.sub(pattern='\n', repl=" ", string=abstract)
    abstract = _remove_quotes_(abstract)
    # define string
    string = ""
    string += f"* {mapping.get(title, title)}\n"
    string += INDENT + f"[[abs-ECCV]({url})]" + '\n'
    string += INDENT + f"[[pdf-ECCV]({pdf_url})]" + '\n'
    string += INDENT + "* Title: " + title + '\n'
    string += INDENT + "* Year: " + year + '\n'
    string += INDENT + "* Authors: " + authors + '\n'
    string += INDENT + "* Abstract: " + abstract + '\n'
    return string


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
    year = year[0]
    # get authors
    authors = str(soup.findAll(name="p")[1])[6:-8]
    # get abstract
    p_idx = 2 + (soup.findAll(name="p")[2].text.strip() == "")
    abstract = soup.findAll(name="p")[p_idx].text.strip()
    abstract = _remove_quotes_(abstract)
    # get pdf url
    pdf_url = url
    pdf_url = re.sub(pattern="hash", repl="file", string=pdf_url)
    pdf_url = re.sub(pattern="Abstract", repl="Paper", string=pdf_url)
    pdf_url = re.sub(pattern=".html", repl=".pdf", string=pdf_url)
    # define string
    string = ""
    string += f"* {mapping.get(title, title)}\n"
    string += f"{INDENT}[[abs-NeurIPS]({url})]\n"
    string += f"{INDENT}[[pdf-NeurIPS]({pdf_url})]\n"
    string += f"{INDENT}* Title: {title}\n"
    string += f"{INDENT}* Year: `{year}`\n"
    string += f"{INDENT}* Authors: {authors}\n"
    string += f"{INDENT}* Abstract: {abstract}\n"
    return string


def scrape_mdpi(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # get title
        title = soup.find("h1", class_="title hypothesis_container").text.strip()
        # get year
        year = soup.find("div", class_="pubhistory").text.strip()
        year = year.split('/')[0].split(':')[1].strip().split(' ')
        year[0] = '0' * (2-len(year[0])) + year[0]
        year[1] = year[1][:3]
        year[2] = '`' + year[2] + '`'
        year = ' '.join(year)
        # get authors
        authors = soup.find_all("a", class_="sciprofiles-link__link")
        authors = ', '.join([author.text for author in authors])
        # get abstract
        abstract = soup.find("div", class_="html-p").text.strip()
        # define string
        string = ""
        string += f"* [[{mapping.get(title, title)}]({url})]" + '\n'
        string += INDENT + "* Title: " + title + '\n'
        string += INDENT + "* Year: " + year + '\n'
        string += INDENT + "* Authors: " + authors + '\n'
        string += INDENT + "* Abstract: " + abstract + '\n'
        return string
    else:
        logging.error(f"Failed to get response from {url}.")
        return ""


def scrape_single(url):
    if url.startswith("https://arxiv.org"):
        return scrape_arxiv(url)
    if url.startswith("https://openaccess.thecvf.com"):
        return scrape_openaccess(url)
    if url.startswith("https://openreview.net"):
        return scrape_openreview(url)
    if url.startswith("https://www.ecva.net"):
        return scrape_eccv(url)
    if url.startswith("https://papers.nips.cc"):
        return scrape_neurips(url)
    if url.startswith("https://www.mdpi.com"):
        return scrape_mdpi(url)
    else:
        logging.error(f"No scraper implemented for {url}.")
        return ""


def main(url_dict, src=None, dst=None):
    """
    Arguments:
        url_dict (dict): dictionary of lists of urls to be scrapped.
        src (str): source file to extract urls from.
        dst (str): will be replaced with temp files if `url_dict` is not None.
    """
    if url_dict is None:
        assert src is not None
        logging.info(f"Argument `url_dict` not provided. Extracting urls from {src}.")
        url_dict = get_all_urls(src)
    else:
        assert type(url_dict) == dict, f"{type(url_dict)=}"
        logging.info(f"Argument `url_dict` provided. Suppressing arguments `src` and `dst`.")
        for group in url_dict:
            url_list = url_dict[group]
            logging.info(f"Processing group '{group}'." + (
                " Nothing given." if len(url_list) == 0 else f" {len(url_list)} urls found."))
            # get unique urls and preserve original order
            url_list_unique = []
            for url in url_list:
                if url not in url_list_unique:
                    url_list_unique.append(url)
            if len(url_list_unique) != len(url_list):
                logging.info(f"Provided list contains duplicates. Reduced to {len(url_list_unique)} papers.")
            # iterate through unique url list and scrape each
            dst = f"scraped_papers_{group}.md"
            with open(dst, mode='w', encoding='utf8') as f:
                for idx, url in enumerate(url_list_unique):
                    logging.info(f"[{idx+1}/{len(url_list_unique)}] Scraping {url}")
                    f.write(scrape_single(url))
    logging.info(f"Process terminated.")


if __name__ == "__main__":
    main(url_dict=papers_tmp)
