import pytest
import sys
sys.path.append("../..")
from scrape import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://www.sciencedirect.com/science/article/pii/S1574954121001011",
        "gt/sciencedirect_Analysis_on_change_detection_techniques_for_remote_sensing_applications_A_review.md",
    ),
])
def test_sciencedirect(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
