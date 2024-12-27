import pytest
import sys
sys.path.append("..")
from scraper import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://journals.scholarsportal.info/details/09265805/v124icomplete/nfp_abcdwacopc.xml",
        "gt/scholarsportal_Automated_building_change_detection_with_amodal_completion_of_point_clouds.md",
    ),
])
def test_scholarsportal(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
