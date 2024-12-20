import pytest
import sys
sys.path.append("..")
from scraper import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://www.mdpi.com/2072-4292/6/9/8310",
        "gt/mdpi_Building_Change_Detection_from_Historical_Aerial_Photographs_Using_Dense_Image_Matching_and_Object_Based_Image_Analysis.md",
    ),
])
def test_mdpi(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
