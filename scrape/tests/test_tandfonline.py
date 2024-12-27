import pytest
import sys
sys.path.append("../..")
from scrape import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://www.tandfonline.com/doi/full/10.1080/2150704X.2018.1562581",
        "gt/tandfonline_Pointwise_SAR_image_change_detection_using_stereo_graph_cuts_with_spatio_temporal_information.md",
    ),
])
def test_tandfonline(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
