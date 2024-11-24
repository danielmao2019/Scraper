from typing import List, Tuple, Union
import re


RECOGNIZED_CONFERENCES = ['cvpr', 'iccv', 'eccv', 'accv', 'wacv', 'iclr']
RECOGNIZED_JOURNALS = ['tmlr']


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
