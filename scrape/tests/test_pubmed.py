import pytest
import sys
sys.path.append("../..")
from scrape import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://pubmed.ncbi.nlm.nih.gov/7584893",
        "gt/PubMed_An_information_maximization_approach_to_blind_separation_and_blind_deconvolution.md",
    ),
])
def test_pubmed(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
