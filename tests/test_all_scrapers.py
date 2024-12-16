import pytest
import sys
sys.path.append("..")
from scraper import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://aclanthology.org/P07-1033",
        "gt_ACL_Frustratingly_Easy_Domain_Adaptation.md",
    ),
    (
        "https://dl.acm.org/doi/abs/10.1145/3550454.3555475",
        "gt_ACM_Geo_Metric_A_Perceptual_Dataset_of_Distortions_on_Faces.md",
    ),
    (
        "https://isprs-archives.copernicus.org/articles/XLII-2/565/2018/",
        "gt_isprs_CHANGE_DETECTION_IN_REMOTE_SENSING_IMAGES_USING_CONDITIONAL_ADVERSARIAL_NETWORKS.md",
    ),
    (
        "https://www.mdpi.com/2072-4292/6/9/8310",
        "gt_mdpi_Building_Change_Detection_from_Historical_Aerial_Photographs_Using_Dense_Image_Matching_and_Object_Based_Image_Analysis.md",
    ),
    (
        "https://openreview.net/forum?id=7UyBKTFrtd",
        "gt_openreview_Interpreting_CLIP_with_Sparse_Linear_Concept_Embeddings_SpLiCE.md",
    ),
    (
        "https://openreview.net/forum?id=KunlRYApGc",
        "gt_openreview_Latent_Space_Disentanglement_in_Diffusion_Transformers_Enables_Zero_shot_Fine_grained_Semantic_Editing.md",
    ),
    (
        "https://pubmed.ncbi.nlm.nih.gov/7584893",
        "gt_PubMed_An_information_maximization_approach_to_blind_separation_and_blind_deconvolution.md",
    ),
    (
        "https://www.researchgate.net/publication/367989910_Automatic_classification_of_asphalt_pavement_cracks_using_a_novel_integrated_generative_adversarial_networks_and_improved_VGG_model",
        "gt_researchgate_Automatic_classification_of_asphalt_pavement_cracks_using_a_novel_integrated_generative_adversarial_networks_and_improved_VGG_model.md",
    ),
    (
        "https://www.roboticsproceedings.org/rss11/p01.html",
        "gt_RSS_ElasticFusion_Dense_SLAM_Without_A_Pose_Graph.md",
    ),
    (
        "https://www.sciencedirect.com/science/article/pii/S1574954121001011",
        "gt_science_direct_Analysis_on_change_detection_techniques_for_remote_sensing_applications__A_review.md",
    ),
    (
        "https://link.springer.com/chapter/10.1007/978-3-319-13560-1_76",
        "gt_Springer_Domain_Adaptive_Neural_Networks_for_Object_Recognition.md",
    ),
    (
        "https://link.springer.com/article/10.1007/s10994-007-5040-8",
        "gt_Springer_Convex_multi_task_feature_learning.md",
    ),
])
def test_springer(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
