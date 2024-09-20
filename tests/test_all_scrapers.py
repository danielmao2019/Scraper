import pytest
import sys
sys.path.append("..")
from scraper import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://link.springer.com/chapter/10.1007/978-3-319-13560-1_76",
        "gt_Springer_Domain_Adaptive_Neural_Networks_for_Object_Recognition.md",
    ),
    (
        "https://link.springer.com/article/10.1007/s10994-007-5040-8",
        "gt_Springer_Convex_multi_task_feature_learning.md",
    ),
    (
        "https://pubmed.ncbi.nlm.nih.gov/7584893/",
        "gt_PubMed_An_information_maximization_approach_to_blind_separation_and_blind_deconvolution.md",
    ),
    (
        "https://aclanthology.org/P07-1033/",
        "gt_ACL_Frustratingly_Easy_Domain_Adaptation.md",
    ),
    (
        "https://www.roboticsproceedings.org/rss11/p01.html",
        "gt_RSS_ElasticFusion_Dense_SLAM_Without_A_Pose_Graph.md",
    ),
])
def test_springer(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
