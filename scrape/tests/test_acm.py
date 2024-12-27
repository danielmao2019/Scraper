import pytest
import sys
sys.path.append("../..")
from scrape import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://dl.acm.org/doi/10.1145/3610548.3618154",
        "gt/ACM_Break_A_Scene_Extracting_Multiple_Concepts_from_a_Single_Image.md",
    ),
    (
        "https://dl.acm.org/doi/abs/10.1145/3550454.3555475",
        "gt/ACM_Geo_Metric_A_Perceptual_Dataset_of_Distortions_on_Faces.md",
    ),
])
def test_acm(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
