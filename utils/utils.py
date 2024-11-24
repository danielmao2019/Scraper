from typing import List, Tuple, Union
from urllib.parse import urljoin
import requests
import re

from code_names_mapping import mapping


RECOGNIZED_CONFERENCES = ['cvpr', 'iccv', 'eccv', 'accv', 'wacv', 'iclr']
RECOGNIZED_JOURNALS = ['tmlr']
INDENT = ' ' * 4


def get_value(item):
    if type(item) == dict:
        return item['value']
    else:
        assert type(item) in [str, list]
        return item


def get_pdf_url(
    abs_url: str = None,
    rel_pdf_url: str = None,
) -> str:
    pdf_url = urljoin(base=abs_url, url=rel_pdf_url)
    # check if new url exists
    r = requests.get(pdf_url)
    assert r.status_code == 200, f"r.status_code={r.status_code}, abs_url={abs_url}, rel_pdf_url={rel_pdf_url}"
    return pdf_url


def _parse_conference_(string: str) -> Tuple[str, str]:
    r"""
    Args:
        string (str): a string that contains the name and year of the conference.
    Returns:
        name (str): name of conference.
        year (str): year of conference.
    """
    # get name
    name = re.findall(pattern=f"({'|'.join(RECOGNIZED_CONFERENCES)})w?", string=string.lower())
    assert len(name) == 1, f"string={string}, name={name}"
    name = name[0].upper()
    # get year
    year = re.findall(pattern=r"\d\d\d\d", string=string)
    assert len(year) == 1, f"string={string}, year={year}"
    year = year[0]
    return name, year


def _parse_journal_(string: str) -> Tuple[str, str]:
    r"""
    Args:
        string (str): a string that contains the name and year of the journal.
    Returns:
        name (str): name of journal.
        year (str): an empty string.
    """
    # get name
    name = re.findall(pattern=f"({'|'.join(RECOGNIZED_JOURNALS)})", string=string.lower())
    assert len(name) == 1, f"string={string}, name={name}"
    name = name[0].upper()
    return name, ""


def parse_writers(value: Union[str, List[str]]) -> Tuple[str]:
    if type(value) == list:
        assert len(value) == 1, f"len(value)={len(value)}"
        value = value[0]
    try:
        return _parse_conference_(value)
    except:
        try:
            return _parse_journal_(value)
        except:
            raise ValueError(f"[ERROR] Cannot parse writers.")


def post_process_abstract(abstract):
    abstract = abstract.strip()
    if abstract.startswith('\"'):
        abstract = abstract[1:]
    if abstract.endswith('\"'):
        abstract = abstract[:-1]
    abstract = re.sub(pattern='\n', repl=" ", string=abstract)
    abstract = re.sub(pattern=" {2,}", repl=" ", string=abstract)
    return abstract


def compile_markdown(
    title: str = None,
    abs_url: str = None,
    pdf_url: str = None,
    pub_name: str = None,
    pub_year: str = None,
    authors: str = None,
    abstract: str = None,
) -> str:
    string = ""
    string += f"* {mapping.get(title, title)}\n"
    string += f"{INDENT}[[abs-{pub_name}]({abs_url})]\n"
    string += f"{INDENT}[[pdf-{pub_name}]({pdf_url})]\n"
    string += f"{INDENT}* Title: {title}\n"
    string += f"{INDENT}* Year: {pub_year}\n"
    string += f"{INDENT}* Authors: {authors}\n"
    string += f"{INDENT}* Abstract: {post_process_abstract(abstract)}\n"
    return string
