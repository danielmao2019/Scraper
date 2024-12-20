import pytest
import sys
sys.path.append("..")
from scraper import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://openreview.net/forum?id=7UyBKTFrtd",
        "gt/openreview_Interpreting_CLIP_with_Sparse_Linear_Concept_Embeddings_SpLiCE.md",
    ),
    (
        "https://openreview.net/forum?id=KunlRYApGc",
        "gt/openreview_Latent_Space_Disentanglement_in_Diffusion_Transformers_Enables_Zero_shot_Fine_grained_Semantic_Editing.md",
    ),
])
def test_openreview(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
