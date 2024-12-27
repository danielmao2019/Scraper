import pytest
import sys
sys.path.append("../..")
from scrape import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://aclanthology.org/P07-1033",
        "gt/ACL_Frustratingly_Easy_Domain_Adaptation.md",
    ),
])
def test_acl(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
