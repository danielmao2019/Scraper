import re
from .code_names_mapping import mapping


INDENT = ' ' * 4


def _post_process_date(year: str) -> str:
    assert type(year) == str, f"{type(year)=}"
    _year = re.findall(pattern=r"\d{4}", string=year)
    assert len(_year) == 1, f"{year=}, {_year=}"
    assert int(_year[0]), f"{_year=}"
    year = re.sub(pattern=r"(\d{4})", repl=r"`\1`", string=year)
    return year


def _post_process_abstract(abstract):
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
    html_url: str = None,
    pdf_url: str = None,
    pub_name: str = None,
    pub_date: str = None,
    authors: str = None,
    abstract: str = None,
    **kwargs,
) -> str:
    string = ""
    string += f"* {mapping.get(title, title)}\n"
    string += f"{INDENT}[[abs-{pub_name}]({html_url})]\n"
    string += f"{INDENT}[[pdf-{pub_name}]({pdf_url})]\n"
    string += f"{INDENT}* Title: {title}\n"
    string += f"{INDENT}* Year: {_post_process_date(pub_date)}\n"
    string += f"{INDENT}* Authors: {authors}\n"
    string += f"{INDENT}* Abstract: {_post_process_abstract(abstract)}\n"
    return string
