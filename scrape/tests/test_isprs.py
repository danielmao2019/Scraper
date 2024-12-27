import pytest
import sys
sys.path.append("../..")
from scrape import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://isprs-archives.copernicus.org/articles/XLII-2/565/2018/",
        "gt/isprs_CHANGE_DETECTION_IN_REMOTE_SENSING_IMAGES_USING_CONDITIONAL_ADVERSARIAL_NETWORKS.md",
    ),
])
def test_isprs(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
