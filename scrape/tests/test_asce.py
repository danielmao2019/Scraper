import pytest
import sys
sys.path.append("../..")
from scrape import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://ascelibrary.org/doi/10.1061/%28ASCE%29CP.1943-5487.0000660",
        "gt/ASCE_Automated_Schedule_and_Progress_Updating_of_IFC_Based_4D_BIMs.md",
    ),
    (
        "https://ascelibrary.org/doi/10.1061/9780784413029.095",
        "gt/ASCE_Development_of_a_System_for_Automated_Schedule_Update_Using_a_4D_Building_Information_Model_and_3D_Point_Cloud.md",
    ),
    (
        "https://ascelibrary.org/doi/10.1061/9780784480830.019",
        "gt/ASCE_Proactive_Construction_Project_Controls_via_Predictive_Visual_Data_Analytics.md",
    ),
])
def test_asce(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
