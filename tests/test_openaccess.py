import pytest
import sys
sys.path.append("..")
from scraper import scrape
import os


@pytest.mark.parametrize("url, expected", [
    (
        "https://openaccess.thecvf.com/content/CVPR2024/html/Zeng_Unmixing_Diffusion_for_Self-Supervised_Hyperspectral_Image_Denoising_CVPR_2024_paper.html",
        "gt/openaccess_CVPR_Unmixing_Diffusion_for_Self_Supervised_Hyperspectral_Image_Denoising.md",
    ),
    (
        "https://openaccess.thecvf.com/content/ICCV2023/html/Han_Towards_Attack-tolerant_Federated_Learning_via_Critical_Parameter_Analysis_ICCV_2023_paper.html",
        "gt/openaccess_ICCV_Towards_Attack_tolerant_Federated_Learning_via_Critical_Parameter_Analysis.md",
    ),
    (
        "https://openaccess.thecvf.com/content/ACCV2020/html/Zheng_Localin_Reshuffle_Net_Toward_Naturally_and_Efficiently_Facial_Image_Blending_ACCV_2020_paper.html",
        "gt/openaccess_ACCV_Localin_Reshuffle_Net_Toward_Naturally_and_Efficiently_Facial_Image_Blending.md",
    ),
    (
        "https://openaccess.thecvf.com/content_WACV_2020/html/Sang_Inferring_Super-Resolution_Depth_from_a_Moving_Light-Source_Enhanced_RGB-D_Sensor_WACV_2020_paper.html",
        "gt/openaccess_WACV_Inferring_Super_Resolution_Depth_from_a_Moving_Light_Source_Enhanced_RGB_D_Sensor_A_Variational_Approach.md",
    ),
])
def test_openaccess(url: str, expected: str) -> None:
    assert type(url) == str, f"type(url)={type(url)}"
    assert type(expected) == str, f'expected={expected}'
    assert os.path.isfile(expected), f"expected={expected}"
    with open(expected, mode='r') as f:
        produced = scrape(url)
        assert produced == f.read(), f"produced={produced}"
