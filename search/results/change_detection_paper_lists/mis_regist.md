count=4
* From Single Image Query to Detailed 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Schonberger_From_Single_Image_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Schonberger_From_Single_Image_2015_CVPR_paper.pdf)]
    * Title: From Single Image Query to Detailed 3D Reconstruction
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Johannes L. Schonberger, Filip Radenovic, Ondrej Chum, Jan-Michael Frahm
    * Abstract: Structure-from-Motion for unordered image collections has significantly advanced in scale over the last decade. This impressive progress can be in part attributed to the introduction of efficient retrieval methods for those systems. While this boosts scalability, it also limits the amount of detail that the large-scale reconstruction systems are able to produce. In this paper, we propose a joint reconstruction and retrieval system that maintains the scalability of large-scale Structure-from-Motion systems while also recovering the often lost ability of reconstructing fine details of the scene. We demonstrate our proposed method on a large-scale dataset of 7.4 million images downloaded from the Internet.

count=4
* Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.pdf)]
    * Title: Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Qingan Yan, Long Yang, Ling Zhang, Chunxia Xiao
    * Abstract: A perennial problem in structure from motion (SfM) is visual ambiguity posed by repetitive structures. Recent disambiguating algorithms infer ambiguities mainly via explicit background context, thus face limitations in highly ambiguous scenes which are visually indistinguishable. Instead of analyzing local visual information, we propose a novel algorithm for SfM disambiguation that explores the global topology as encoded in photo collections. An important adaptation of this work is to approximate the available imagery using a manifold of viewpoints. We note that, while ambiguous images appear deceptively similar in appearance, they are actually located far apart on geodesics. We establish the manifold by adaptively identifying cameras with adjacent viewpoint, and detect ambiguities via a new measure, geodesic consistency. We demonstrate the accuracy and efficiency of the proposed approach on a range of complex ambiguity datasets, even including the challenging scenes without background conflicts.

count=3
* Differentiable Registration of Images and LiDAR Point Clouds with VoxelPoint-to-Pixel Matching
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/a0a53fefef4c2ad72d5ab79703ba70cb-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/a0a53fefef4c2ad72d5ab79703ba70cb-Paper-Conference.pdf)]
    * Title: Differentiable Registration of Images and LiDAR Point Clouds with VoxelPoint-to-Pixel Matching
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Junsheng Zhou, Baorui Ma, Wenyuan Zhang, Yi Fang, Yu-Shen Liu, Zhizhong Han
    * Abstract: Cross-modality registration between 2D images captured by cameras and 3D point clouds from LiDARs is a crucial task in computer vision and robotic. Previous methods estimate 2D-3D correspondences by matching point and pixel patterns learned by neural networks, and use Perspective-n-Points (PnP) to estimate rigid transformation during post-processing. However, these methods struggle to map points and pixels to a shared latent space robustly since points and pixels have very different characteristics with patterns learned in different manners (MLP and CNN), and they also fail to construct supervision directly on the transformation since the PnP is non-differentiable, which leads to unstable registration results. To address these problems, we propose to learn a structured cross-modality latent space to represent pixel features and 3D features via a differentiable probabilistic PnP solver. Specifically, we design a triplet network to learn VoxelPoint-to-Pixel matching, where we represent 3D elements using both voxels and points to learn the cross-modality latent space with pixels. We design both the voxel and pixel branch based on CNNs to operate convolutions on voxels/pixels represented in grids, and integrate an additional point branch to regain the information lost during voxelization. We train our framework end-to-end by imposing supervisions directly on the predicted pose distribution with a probabilistic PnP solver. To explore distinctive patterns of cross-modality features, we design a novel loss with adaptive-weighted optimization for cross-modality feature description. The experimental results on KITTI and nuScenes datasets show significant improvements over the state-of-the-art methods.

count=2
* Structure-From-Motion Revisited
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)]
    * Title: Structure-From-Motion Revisited
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Johannes L. Schonberger, Jan-Michael Frahm
    * Abstract: Incremental Structure-from-Motion is a prevalent strategy for 3D reconstruction from unordered image collections. While incremental reconstruction systems have tremendously advanced in all regards, robustness, accuracy, completeness, and scalability remain the key problems towards building a truly general-purpose pipeline. We propose a new SfM technique that improves upon the state of the art to make a further step towards this ultimate goal. The full reconstruction pipeline is released to the public as an open-source implementation.

count=1
* Geospatial Correspondences for Multimodal Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.pdf)]
    * Title: Geospatial Correspondences for Multimodal Registration
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Diego Marcos, Raffay Hamid, Devis Tuia
    * Abstract: The growing availability of very high resolution (<1 m/pixel) satellite and aerial images has opened up unprecedented opportunities to monitor and analyze the evolution of land-cover and land-use across the world. To do so, images of the same geographical areas acquired at different times and, potentially, with different sensors must be efficiently parsed to update maps and detect land-cover changes. However, a naive transfer of ground truth labels from one location in the source image to the corresponding location in the target image is not generally feasible, as these images are often only loosely registered (with up to +- 50m of non-uniform errors). Furthermore, land-cover changes in an area over time must be taken into account for an accurate ground truth transfer. To tackle these challenges, we propose a mid-level sensor-invariant representation that encodes image regions in terms of the spatial distribution of their spectral neighbors. We incorporate this representation in a Markov Random Field to simultaneously account for nonlinear mis-registrations and enforce locality priors to find matches between multi-sensor images. We show how our approach can be used to assist in several multimodal land-cover update and change detection problems.

count=1
* Do It Yourself Hyperspectral Imaging With Everyday Digital Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Oh_Do_It_Yourself_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Oh_Do_It_Yourself_CVPR_2016_paper.pdf)]
    * Title: Do It Yourself Hyperspectral Imaging With Everyday Digital Cameras
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Seoung Wug Oh, Michael S. Brown, Marc Pollefeys, Seon Joo Kim
    * Abstract: Capturing hyperspectral images requires expensive and specialized hardware that is not readily accessible to most users. Digital cameras, on the other hand, are significantly cheaper in comparison and can be easily purchased and used. In this paper, we present a framework for reconstructing hyperspectral images by using multiple consumer-level digital cameras. Our approach works by exploiting the different spectral sensitivities of different camera sensors. In particular, due to the differences in spectral sensitivities of the cameras, different cameras yield different RGB measurements for the same spectral signal. We introduce an algorithm that is able to combine and convert these different RGB measurements into a single hyperspectral image for both indoor and outdoor scenes. This camera-based approach allows hyperspectral imaging at a fraction of the cost of most existing hyperspectral hardware. We validate the accuracy of our reconstruction against ground truth hyperspectral images (using both synthetic and real cases) and show its usage on relighting applications.

count=1
* LiFF: Light Field Features in Scale and Depth
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Dansereau_LiFF_Light_Field_Features_in_Scale_and_Depth_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dansereau_LiFF_Light_Field_Features_in_Scale_and_Depth_CVPR_2019_paper.pdf)]
    * Title: LiFF: Light Field Features in Scale and Depth
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Donald G. Dansereau,  Bernd Girod,  Gordon Wetzstein
    * Abstract: Feature detectors and descriptors are key low-level vision tools that many higher-level tasks build on. Unfortunately these fail in the presence of challenging light transport effects including partial occlusion, low contrast, and reflective or refractive surfaces. Building on spatio-angular imaging modalities offered by emerging light field cameras, we introduce a new and computationally efficient 4D light field feature detector and descriptor: LiFF. LiFF is scale invariant and utilizes the full 4D light field to detect features that are robust to changes in perspective. This is particularly useful for structure from motion (SfM) and other tasks that match features across viewpoints of a scene. We demonstrate significantly improved 3D reconstructions via SfM when using LiFF instead of the leading 2D or 4D features, and show that LiFF runs an order of magnitude faster than the leading 4D approach. Finally, LiFF inherently estimates depth for each feature, opening a path for future research in light field-based SfM.

count=1
* SrvfRegNet: Elastic Function Registration Using Deep Neural Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/DiffCVML/html/Chen_SrvfRegNet_Elastic_Function_Registration_Using_Deep_Neural_Networks_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/DiffCVML/papers/Chen_SrvfRegNet_Elastic_Function_Registration_Using_Deep_Neural_Networks_CVPRW_2021_paper.pdf)]
    * Title: SrvfRegNet: Elastic Function Registration Using Deep Neural Networks
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Chao Chen, Anuj Srivastava
    * Abstract: Registering functions (curves) using time warpings (reparameterizations) is central to many computer vision and shape analysis solutions. While traditional registration methods minimize penalized-L2 norm, the elastic Riemannian metric and square-root velocity functions (SRVFs) have resulted in significant improvements in terms of theory and practical performance. This solution uses the dynamic programming algorithm to minimize the L2 norm between SRVFs of given functions. However, the computational cost of this elastic dynamic programming framework - O(nT2k) - where T is the number of time samples along a curve, n is the number of curves, and k < T is a parameter - limits its use in applications involving big data. This paper introduces a deep-learning approach, named SRVF Registration Net or SrvfRegNet to overcome these limitations. SrvfRegNet architecture trains by optimizing the elastic metric-based objective function on the training data and then applies this trained network to the test data to perform super-fast registration. In case the training and the test data are from different classes, it generalizes to the test data using transfer learning, i.e., retraining of only the last few layers. It achieves close to the state-of-the-art alignment performance but at much reduced computational cost. We demonstrate the efficiency and efficacy of this framework using several standard curve datasets

count=1
* Dual Task Learning by Leveraging Both Dense Correspondence and Mis-Correspondence for Robust Change Detection With Imperfect Matches
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.pdf)]
    * Title: Dual Task Learning by Leveraging Both Dense Correspondence and Mis-Correspondence for Robust Change Detection With Imperfect Matches
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jin-Man Park, Ue-Hwan Kim, Seon-Hoon Lee, Jong-Hwan Kim
    * Abstract: Accurate change detection enables a wide range of tasks in visual surveillance, anomaly detection and mobile robotics. However, contemporary change detection approaches assume an ideal matching between the current and stored scenes, whereas only coarse matching is possible in real-world scenarios. Thus, contemporary approaches fail to show the reported performance in real-world settings. To overcome this limitation, we propose SimSaC. SimSaC concurrently conducts scene flow estimation and change detection and is able to detect changes with imperfect matches. To train SimSaC without additional manual labeling, we propose a training scheme with random geometric transformations and the cut-paste method. Moreover, we design an evaluation protocol which reflects performance in real-world settings. In designing the protocol, we collect a test benchmark dataset, which we claim as another contribution. Our comprehensive experiments verify that SimSaC displays robust performance even given imperfect matches and the performance margin compared to contemporary approaches is huge.

count=1
* L1BSR: Exploiting Detector Overlap for Self-Supervised Single-Image Super-Resolution of Sentinel-2 L1B Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Nguyen_L1BSR_Exploiting_Detector_Overlap_for_Self-Supervised_Single-Image_Super-Resolution_of_Sentinel-2_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Nguyen_L1BSR_Exploiting_Detector_Overlap_for_Self-Supervised_Single-Image_Super-Resolution_of_Sentinel-2_CVPRW_2023_paper.pdf)]
    * Title: L1BSR: Exploiting Detector Overlap for Self-Supervised Single-Image Super-Resolution of Sentinel-2 L1B Imagery
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Ngoc Long Nguyen, Jérémy Anger, Axel Davy, Pablo Arias, Gabriele Facciolo
    * Abstract: High-resolution satellite imagery is a key element for many Earth monitoring applications. Satellites such as Sentinel-2 feature characteristics that are favorable for super-resolution algorithms such as aliasing and band-misalignment. Unfortunately the lack of reliable high-resolution (HR) ground truth limits the application of deep learning methods to this task. In this work we propose L1BSR, a deep learning-based method for single-image super-resolution and band alignment of Sentinel-2 L1B 10m bands. The method is trained with self-supervision directly on real L1B data by leveraging overlapping areas in L1B images produced by adjacent CMOS detectors, thus not requiring HR ground truth. Our self-supervised loss is designed to enforce the super-resolved output image to have all the bands correctly aligned. This is achieved via a novel cross-spectral registration network (CSR) which computes an optical flow between images of different spectral bands. The CSR network is also trained with self-supervision using an Anchor-Consistency loss, which we also introduce in this work. We demonstrate the performance of the proposed approach on synthetic and real L1B data, where we show that it obtains comparable results to supervised methods.

count=1
* Comprehensive Quality Assessment of Optical Satellite Imagery Using Weakly Supervised Video Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Pasquarella_Comprehensive_Quality_Assessment_of_Optical_Satellite_Imagery_Using_Weakly_Supervised_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Pasquarella_Comprehensive_Quality_Assessment_of_Optical_Satellite_Imagery_Using_Weakly_Supervised_CVPRW_2023_paper.pdf)]
    * Title: Comprehensive Quality Assessment of Optical Satellite Imagery Using Weakly Supervised Video Learning
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Valerie J. Pasquarella, Christopher F. Brown, Wanda Czerwinski, William J. Rucklidge
    * Abstract: Identifying high-quality (i.e., relatively clear) measurements of surface conditions is a near-universal first step in working with optical satellite imagery. Many cloud masking algorithms have been developed to characterize the likelihood that reflectance measurements for individual pixels were influenced by clouds, cloud shadows, and other atmospheric effects. However, due to the continuous density of the atmospheric volume, we argue that quantification of occlusion and corruption effects is better treated as a regression problem rather than a discretized classification problem as done in prior work. We propose a space-time context network trained using a bootstrapping procedure that leverages millions of automatically-mined video sequences informed by a weakly supervised measure of atmospheric similarity. We find that our approach outperforms existing machine learning and physical basis cloud and cloud shadow detection algorithms, producing state-of-the-art results for Sentinel-2 imagery on two different out-of-distribution reference datasets. The resulting product offers a flexible quality assessment (QA) solution that enables both standard cloud and cloud shadow masking via thresholding and more complex image grading for compositing or downstream models. By way of generality, minimal supervision, and scale of our training data, our approach has the potential to significantly improve the utility and usability of optical remote sensing imagery.

count=1
* S2A: Wasserstein GAN With Spatio-Spectral Laplacian Attention for Multi-Spectral Band Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Rout_S2A_Wasserstein_GAN_With_Spatio-Spectral_Laplacian_Attention_for_Multi-Spectral_Band_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Rout_S2A_Wasserstein_GAN_With_Spatio-Spectral_Laplacian_Attention_for_Multi-Spectral_Band_CVPRW_2020_paper.pdf)]
    * Title: S2A: Wasserstein GAN With Spatio-Spectral Laplacian Attention for Multi-Spectral Band Synthesis
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Litu Rout, Indranil Misra, S Manthira Moorthi, Debajyoti Dhar
    * Abstract: Intersection of adversarial learning and satellite image processing is an emerging field in remote sensing. In this study, we intend to address synthesis of high resolution multi-spectral satellite imagery using adversarial learning. Guided by the discovery of attention mechanism, we regulate the process of band synthesis through spatio-spectral Laplacian attention. Further, we use Wasserstein GAN with gradient penalty norm to improve training and stability of adversarial learning. In this regard, we introduce a new cost function for the discriminator based on spatial attention and domain adaptation loss. We critically analyze the qualitative and quantitative results compared with state-of-the-art methods using widely adopted evaluation metrics. Our experiments on datasets of three different sensors, namely LISS-3, LISS-4, and WorldView-2 show that attention learning performs favorably against state-of-the-art methods. Using the proposed method we provide an additional data product in consistent with existing high resolution bands. Furthermore, we synthesize over 4000 high resolution scenes covering various terrains to analyze scientific fidelity. At the end, we demonstrate plausible large scale real world applications of the synthesized band.

count=1
* Selective Encoding for Recognizing Unreliably Localized Faces
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Li_Selective_Encoding_for_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Li_Selective_Encoding_for_ICCV_2015_paper.pdf)]
    * Title: Selective Encoding for Recognizing Unreliably Localized Faces
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Ang Li, Vlad Morariu, Larry S. Davis
    * Abstract: Most existing face verification systems rely on precise face detection and registration. However, these two components are fallible under unconstrained scenarios (e.g., mobile face authentication) due to partial occlusions, pose variations, lighting conditions and limited view-angle coverage of mobile cameras. We address the unconstrained face verification problem by encoding face images directly without any explicit models of detection or registration. We propose a selective encoding framework which injects relevance information (e.g., foreground/background probabilities) into each cluster of a descriptor codebook. An additional selector component also discards distractive image patches and improves spatial robustness. We evaluate our framework using Gaussian mixture models and Fisher vectors on challenging face verification datasets. We apply selective encoding to Fisher vector features, which in our experiments degrade quickly with inaccurate face localization; our framework improves robustness with no extra test time computation. We also apply our approach to mobile based active face authentication task, demonstrating its utility in real scenarios.

count=1
* Fast and Accurate: Video Enhancement Using Sparse Depth
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Feng_Fast_and_Accurate_Video_Enhancement_Using_Sparse_Depth_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Feng_Fast_and_Accurate_Video_Enhancement_Using_Sparse_Depth_WACV_2023_paper.pdf)]
    * Title: Fast and Accurate: Video Enhancement Using Sparse Depth
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Yu Feng, Patrick Hansen, Paul N. Whatmough, Guoyu Lu, Yuhao Zhu
    * Abstract: This paper presents a general framework to build fast and accurate algorithms for video enhancement tasks such as super-resolution, deblurring, and denoising. Essential to our framework is the realization that the accuracy, rather than the density, of pixel flows is what is required for high-quality video enhancement. Most of prior works take the opposite approach: they estimate dense (per-pixel)--but generally less robust--flows, mostly using computationally costly algorithms. Instead, we propose a lightweight flow estimation algorithm; it fuses the sparse point cloud data and (even sparser and less reliable) IMU data available in modern autonomous agents to estimate the flow information. Building on top of the flow estimation, we demonstrate a general framework that integrates the flows in a plug-and-play fashion with different task-specific layers. Algorithms built in our framework achieve 1.78x -- 187.41x speedup while providing a 0.42dB - 6.70 dB quality improvement over competing methods.

