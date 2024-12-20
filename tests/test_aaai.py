import pytest
import sys
sys.path.append("..")
from scraper import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://ojs.aaai.org/index.php/AAAI/article/view/9597",
        "gt/AAAI_Active_Learning_by_Learning.md",
    ),
    (
        "https://ojs.aaai.org/index.php/AIES/article/view/31754",
        "gt/AAAI_Stable_Diffusion_Exposed_Gender_Bias_from_Prompt_to_Image.md",
    ),
    (
        "https://ojs.aaai.org/index.php/HCOMP/article/view/5284",
        "gt/AAAI_The_Effects_of_Meaningful_and_Meaningless_Explanations_on_Trust_and_Perceived_System_Accuracy_in_Intelligent_Systems.md",
    ),
])
def test_aaai(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
