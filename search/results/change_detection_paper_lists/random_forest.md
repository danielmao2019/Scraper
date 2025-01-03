count=2
* PSMNet-FusionX3: LiDAR-Guided Deep Learning Stereo Dense Matching on Aerial Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wu_PSMNet-FusionX3_LiDAR-Guided_Deep_Learning_Stereo_Dense_Matching_on_Aerial_Images_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Wu_PSMNet-FusionX3_LiDAR-Guided_Deep_Learning_Stereo_Dense_Matching_on_Aerial_Images_CVPRW_2023_paper.pdf)]
    * Title: PSMNet-FusionX3: LiDAR-Guided Deep Learning Stereo Dense Matching on Aerial Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Teng Wu, Bruno Vallet, Marc Pierrot-Deseilligny
    * Abstract: Dense image matching (DIM) and LiDAR are two complementary techniques for recovering the 3D geometry of real scenes. While DIM provides dense surfaces, they are often noisy and contaminated with outliers. Conversely, LiDAR is more accurate and robust, but less dense and more expensive compared to DIM. In this work, we investigate learning-based methods to refine surfaces produced by photogrammetry with sparse LiDAR point clouds. Unlike the current state-of-the-art approaches in the computer vision community, our focus is on aerial acquisitions typical in photogrammetry. We propose a densification pipeline that adopts a PSMNet backbone with triangulated irregular network interpolation based expansion, feature enhancement in cost volume, and conditional cost volume normalization, i.e. PSMNet-FusionX3. Our method works better on low density and is less sensitive to distribution, demonstrating its effectiveness across a range of LiDAR point cloud densities and distributions, including analyses of dataset shifts. Furthermore, we have made both our aerial (image and disparity) dataset and code available for public use. Further information can be found at https://github.com/whuwuteng/PSMNet-FusionX3.

