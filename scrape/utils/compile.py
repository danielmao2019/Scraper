from typing import List, Dict, Union, Optional
import re
from .code_names_mapping import mapping


INDENT = ' ' * 4


def _post_process_date(date: Union[str, int]) -> str:
    if isinstance(date, int):
        date = str(date)
    assert isinstance(date, str)
    assert type(date) == str, f"{type(date)=}"
    _year = re.findall(pattern=r"\d{4}", string=date)
    assert len(_year) == 1, f"{date=}, {_year=}"
    assert int(_year[0]), f"{_year=}"
    date = re.sub(pattern=r"(\d{4})", repl=r"`\1`", string=date)
    return date


def _post_process_abstract(abstract):
    abstract = abstract.strip()
    if abstract.startswith('\"'):
        abstract = abstract[1:]
    if abstract.endswith('\"'):
        abstract = abstract[:-1]
    abstract = re.sub(pattern='\n', repl=" ", string=abstract)
    abstract = re.sub(pattern=" {2,}", repl=" ", string=abstract)
    return abstract


web_name_mapping = {
    "https://arxiv.org": "arXiv",
    "https://ascelibrary.org": "ASCE",
    "https://ojs.aaai.org": "AAAI",
    "https://aclanthology.org": "ACL",
    "https://dl.acm.org/doi": "ACM",
    "https://www.ecva.net": "ECCV",
    "https://ieeexplore.ieee.org": "IEEE",
    "https://isprs": "ISPRS",
    "https://www.jmlr.org": "JMLR",
    "https://jmlr.org": "JMLR",
    "https://www.mdpi.com": "MDPI",
    "https://papers.nips.cc": "NIPS",
    "https://proceedings.neurips.cc": "NIPS",
    "https://openaccess.thecvf.com": "CVF",
    "https://openreview.net": "OpenReview",
    "https://proceedings.mlr.press": "PMLR",
    "https://pubmed.ncbi.nlm.nih.gov": "PubMed",
    "https://www.researchgate.net": "ResearchGate",
    "https://www.roboticsproceedings.org": "Robotics",
    "https://www.sciencedirect.com": "ScienceDirect",
    "https://journals.scholarsportal.info": "ScholarsPortal",
    "https://link.springer.com": "Springer",
    "https://www.tandfonline.com": "T&F",
}


def _get_web_name(url: str) -> str:
    for key in web_name_mapping:
        if url.startswith(key):
            return web_name_mapping[key]
    raise ValueError(f"No matching website found for URL: {url}")


def compile_markdown(
    title: str,
    urls: Optional[Dict[str, Dict[str, List[str]]]] = None,
    html_url: Optional[str] = None,
    pdf_url: Optional[str] = None,
    pub_name: str = None,
    pub_date: Union[str, int] = None,
    authors: List[str] = None,
    abstract: str = None,
    **kwargs,
) -> str:
    r"""
    Args:
        kwargs: unused keyword arguments.
    """
    # input checks
    assert isinstance(authors, list), f"{type(authors)=}"
    assert all(isinstance(x, str) for x in authors), f"{[type(x) for x in authors]=}"
    assert (html_url is None) == (pdf_url is None)
    assert (urls is not None) ^ (html_url is not None and pdf_url is not None)
    if html_url is not None and pdf_url is not None:
        assert type(html_url) == type(pdf_url) == str
        urls = {
            'html': [html_url],
            'pdf': [pdf_url],
        }
    assert type(urls) == dict and set(urls.keys()) == set(['html', 'pdf'])
    assert type(urls['html']) == list and all(type(x) == str for x in urls['html'])
    assert type(urls['pdf']) == list and all(type(x) == str for x in urls['pdf'])
    # compile
    string = ""
    string += f"* {mapping.get(title, title)}\n"
    for html_url, pdf_url in zip(urls['html'], urls['pdf']):
        string += f"{INDENT}[[abs-{_get_web_name(html_url)}]({html_url})]\n"
        string += f"{INDENT}[[pdf-{_get_web_name(html_url)}]({pdf_url})]\n"
    string += f"{INDENT}* Title: {title}\n"
    string += f"{INDENT}* Publisher: {pub_name}\n"
    string += f"{INDENT}* Publication Date: {_post_process_date(pub_date)}\n"
    string += f"{INDENT}* Authors: {', '.join(authors)}\n"
    string += f"{INDENT}* Abstract: {_post_process_abstract(abstract)}\n"
    return string
