import pytest
import sys
sys.path.append("../..")
from scrape import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Michelle_Guo_Focus_on_the_ECCV_2018_paper.php",
    ),
    (
        "https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3590_ECCV_2020_paper.php",
    ),
    (
        "https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4220_ECCV_2022_paper.php"
        "gt/ASCE_Automated_Schedule_and_Progress_Updating_of_IFC_Based_4D_BIMs.md",
    ),
])
def test_eccv(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
