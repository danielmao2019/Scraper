import pytest
import sys
sys.path.append("../..")
from scrape import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://www.researchgate.net/publication/367989910_Automatic_classification_of_asphalt_pavement_cracks_using_a_novel_integrated_generative_adversarial_networks_and_improved_VGG_model",
        "gt/researchgate_Automatic_classification_of_asphalt_pavement_cracks_using_a_novel_integrated_generative_adversarial_networks_and_improved_VGG_model.md",
    ),
])
def test_researchgate(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
