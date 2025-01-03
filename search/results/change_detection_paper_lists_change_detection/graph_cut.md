count=23
* Ensemble Video Object Cut in Highly Dynamic Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Ren_Ensemble_Video_Object_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ren_Ensemble_Video_Object_2013_CVPR_paper.pdf)]
    * Title: Ensemble Video Object Cut in Highly Dynamic Scenes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Xiaobo Ren, Tony X. Han, Zhihai He
    * Abstract: We consider video object cut as an ensemble of framelevel background-foreground object classifiers which fuses information across frames and refine their segmentation results in a collaborative and iterative manner. Our approach addresses the challenging issues of modeling of background with dynamic textures and segmentation of foreground objects from cluttered scenes. We construct patch-level bagof-words background models to effectively capture the background motion and texture dynamics. We propose a foreground salience graph (FSG) to characterize the similarity of an image patch to the bag-of-words background models in the temporal domain and to neighboring image patches in the spatial domain. We incorporate this similarity information into a graph-cut energy minimization framework for foreground object segmentation. The background-foreground classification results at neighboring frames are fused together to construct a foreground probability map to update the graph weights. The resulting object shapes at neighboring frames are also used as constraints to guide the energy minimization process during graph cut. Our extensive experimental results and performance comparisons over a diverse set of challenging videos with dynamic scenes, including the new Change Detection Challenge Dataset, demonstrate that the proposed ensemble video object cut method outperforms various state-ofthe-art algorithms.

count=23
* Graph-Cut RANSAC
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.pdf)]
    * Title: Graph-Cut RANSAC
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Daniel Barath, Jiří Matas
    * Abstract: A novel method for robust estimation, called Graph-Cut RANSAC, GC-RANSAC in short, is introduced. To separate inliers and outliers, it runs the graph-cut algorithm in the local optimization (LO) step which is applied when a so-far-the-best model is found. The proposed LO step is conceptually simple, easy to implement, globally optimal and efficient. GC-RANSAC is shown experimentally, both on synthesized tests and real image pairs, to be more geometrically accurate than state-of-the-art methods on a range of problems, e.g. line fitting, homography, affine transformation, fundamental and essential matrix estimation. It runs in real-time for many problems at a speed approximately equal to that of the less accurate alternatives (in milliseconds on standard CPU).

count=8
* Dense Semantic Labeling of Very-High-Resolution Aerial Imagery and LiDAR With Fully-Convolutional Neural Networks and Higher-Order CRFs
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Liu_Dense_Semantic_Labeling_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Liu_Dense_Semantic_Labeling_CVPR_2017_paper.pdf)]
    * Title: Dense Semantic Labeling of Very-High-Resolution Aerial Imagery and LiDAR With Fully-Convolutional Neural Networks and Higher-Order CRFs
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yansong Liu, Sankaranarayanan Piramanayagam, Sildomar T. Monteiro, Eli Saber
    * Abstract: Efficient and effective multisensor fusion techniques are demanded in order to fully exploit two complementary data modalities, e.g aerial optical imagery, and the LiDAR data. Recent efforts have been mostly devoted to exploring how to properly combine both sensor data using pre-trained deep convolutional neural networks (DCNNs) at the feature level. In this paper, we propose a decision-level fusion approach with a simpler architecture for the task of dense semantic labeling. Our proposed method first obtains two initial probabilistic labeling results from a fully-convolutional neural network and a simple classifier, e.g. logistic regression exploiting spectral channels and LiDAR data, respectively. These two outcomes are then combined within a higher-order conditional random field (CRF). The CRF inference will estimate the final dense semantic labeling results. The proposed method generates the state-of-the-art semantic labeling results.

count=6
* Mutual Foreground Segmentation With Multispectral Stereo Pairs
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w6/html/St-Charles_Mutual_Foreground_Segmentation_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w6/St-Charles_Mutual_Foreground_Segmentation_ICCV_2017_paper.pdf)]
    * Title: Mutual Foreground Segmentation With Multispectral Stereo Pairs
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: Foreground-background segmentation of video sequences is a low-level process commonly used in machine vision, and highly valued in video content analysis and smart surveillance applications. Its efficacy relies on the contrast between objects observed by the sensor. In this work, we study how the combination of sensors operating in the long-wavelength infrared (LWIR) and visible spectra can improve the performance of foreground-background segmentation methods. As opposed to a classic visible spectrum stereo pair, this multispectral pair is more adequate for object segmentation since it reduces the odds of observing low-contrast regions simultaneously in both images. We show that by alternately minimizing stereo disparity and binary segmentation energies with dynamic priors, we can drastically improve the results of a traditional video segmentation approach applied to each sensor individually. Our implementation is freely available online for anyone wishing to recreate our results.

count=5
* Semantic 3D Reconstruction With Continuous Regularization and Ray Potentials Using a Visibility Consistency Constraint
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Savinov_Semantic_3D_Reconstruction_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Savinov_Semantic_3D_Reconstruction_CVPR_2016_paper.pdf)]
    * Title: Semantic 3D Reconstruction With Continuous Regularization and Ray Potentials Using a Visibility Consistency Constraint
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Nikolay Savinov, Christian Hane, Lubor Ladicky, Marc Pollefeys
    * Abstract: We propose an approach for dense semantic 3D reconstruction which uses a data term that is defined as potentials over viewing rays, combined with continuous surface area penalization. Our formulation is a convex relaxation which we augment with a crucial non-convex constraint that ensures exact handling of visibility. To tackle the non-convex minimization problem, we propose a majorize-minimize type strategy which converges to a critical point. We demonstrate the benefits of using the non-convex constraint experimentally. For the geometry-only case, we set a new state of the art on two datasets of the commonly used Middlebury multi-view stereo benchmark. Moreover, our general-purpose formulation directly reconstructs thin objects, which are usually treated with specialized algorithms. A qualitative evaluation on the dense semantic 3D reconstruction task shows that we improve significantly over previous methods.

count=4
* Should We Encode Rain Streaks in Video as Deterministic or Stochastic?
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Wei_Should_We_Encode_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wei_Should_We_Encode_ICCV_2017_paper.pdf)]
    * Title: Should We Encode Rain Streaks in Video as Deterministic or Stochastic?
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Wei Wei, Lixuan Yi, Qi Xie, Qian Zhao, Deyu Meng, Zongben Xu
    * Abstract: Videos taken in the wild sometimes contain unexpected rain streaks, which brings difficulty in subsequent video processing tasks. Rain streak removal in a video (RSRV) is thus an important issue and has been attracting much attention in computer vision. Different from previous RSRV methods formulating rain streaks as a deterministic message, this work first encodes the rains in a stochastic manner, i.e., a patch-based mixture of Gaussians. Such modification makes the proposed model capable of finely adapting a wider range of rain variations instead of certain types of rain configurations as traditional. By integrating with the spatiotemporal smoothness configuration of moving objects and low-rank structure of background scene, we propose a concise model for RSRV, containing one likelihood term imposed on the rain streak layer and two prior terms on the moving object and background scene layers of the video. Experiments implemented on videos with synthetic and real rains verify the superiority of the proposed method, as com- pared with the state-of-the-art methods, both visually and quantitatively in various performance metrics.

count=3
* Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Video_Rain_Streak_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Video_Rain_Streak_CVPR_2018_paper.pdf)]
    * Title: Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Minghan Li, Qi Xie, Qian Zhao, Wei Wei, Shuhang Gu, Jing Tao, Deyu Meng
    * Abstract: Videos captured by outdoor surveillance equipments sometimes contain unexpected rain streaks, which brings difficulty in subsequent video processing tasks. Rain streak removal from a video is thus an important topic in recent computer vision research. In this paper, we raise two intrinsic characteristics specifically possessed by rain streaks. Firstly, the rain streaks in a video contain repetitive local patterns sparsely scattered over different positions of the video. Secondly, the rain streaks are with multiscale configurations due to their occurrence on positions with different distances to the cameras. Based on such understanding, we specifically formulate both characteristics into a multiscale convolutional sparse coding (MS-CSC) model for the video rain streak removal task. Specifically, we use multiple convolutional filters convolved on the sparse feature maps to deliver the former characteristic, and further use multiscale filters to represent different scales of rain streaks. Such a new encoding manner makes the proposed method capable of properly extracting rain streaks from videos, thus getting fine video deraining effects. Experiments implemented on synthetic and real videos verify the superiority of the proposed method, as compared with the state-of-the-art ones along this research line, both visually and quantitatively.

count=3
* From Video Matching to Video Grounding
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W20/html/Evangelidis_From_Video_Matching_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W20/papers/Evangelidis_From_Video_Matching_2013_ICCV_paper.pdf)]
    * Title: From Video Matching to Video Grounding
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Georgios Evangelidis, Ferran Diego, Radu Horaud
    * Abstract: This paper addresses the background estimation problem for videos captured by moving cameras, referred to as video grounding. It essentially aims at reconstructing a video, as if it would be without foreground objects, e.g. cars or people. What differentiates video grounding from known background estimation methods is that the camera follows unconstrained motion so that background undergoes ongoing changes. We build on video matching aspects since more videos contribute to the reconstruction. Without loss of generality, we investigate a challenging case where videos are recorded by in-vehicle cameras that follow the same road. Other than video synchronization and spatiotemporal alignment, we focus on the background reconstruction by exploiting interand intra-sequence similarities. In this context, we propose a Markov random field formulation that integrates the temporal coherence of videos while it exploits the decisions of a support vector machine classifier about the backgroundness of regions in video frames. Experiments with real sequences recorded by moving vehicles verify the potential of the video grounding algorithm against state-ofart baselines.

count=3
* Learning a Multi-View Stereo Machine
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/9c838d2e45b2ad1094d42f4ef36764f6-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/9c838d2e45b2ad1094d42f4ef36764f6-Paper.pdf)]
    * Title: Learning a Multi-View Stereo Machine
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Abhishek Kar, Christian Häne, Jitendra Malik
    * Abstract: We present a learnt system for multi-view stereopsis. In contrast to recent learning based methods for 3D reconstruction, we leverage the underlying 3D geometry of the problem through feature projection and unprojection along viewing rays. By formulating these operations in a differentiable manner, we are able to learn the system end-to-end for the task of metric 3D reconstruction. End-to-end learning allows us to jointly reason about shape priors while conforming to geometric constraints, enabling reconstruction from much fewer images (even a single image) than required by classical approaches as well as completion of unseen surfaces. We thoroughly evaluate our approach on the ShapeNet dataset and demonstrate the benefits over classical approaches and recent learning based methods.

count=2
* Detecting Changes in 3D Structure of a Scene from Multi-view Images Captured by a Vehicle-Mounted Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Sakurada_Detecting_Changes_in_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Sakurada_Detecting_Changes_in_2013_CVPR_paper.pdf)]
    * Title: Detecting Changes in 3D Structure of a Scene from Multi-view Images Captured by a Vehicle-Mounted Camera
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Ken Sakurada, Takayuki Okatani, Koichiro Deguchi
    * Abstract: This paper proposes a method for detecting temporal changes of the three-dimensional structure of an outdoor scene from its multi-view images captured at two separate times. For the images, we consider those captured by a camera mounted on a vehicle running in a city street. The method estimates scene structures probabilistically, not deterministically, and based on their estimates, it evaluates the probability of structural changes in the scene, where the inputs are the similarity of the local image patches among the multi-view images. The aim of the probabilistic treatment is to maximize the accuracy of change detection, behind which there is our conjecture that although it is difficult to estimate the scene structures deterministically, it should be easier to detect their changes. The proposed method is compared with the methods that use multi-view stereo (MVS) to reconstruct the scene structures of the two time points and then differentiate them to detect changes. The experimental results show that the proposed method outperforms such MVS-based methods.

count=2
* High-Speed Image Reconstruction Through Short-Term Plasticity for Spiking Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_High-Speed_Image_Reconstruction_Through_Short-Term_Plasticity_for_Spiking_Cameras_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_High-Speed_Image_Reconstruction_Through_Short-Term_Plasticity_for_Spiking_Cameras_CVPR_2021_paper.pdf)]
    * Title: High-Speed Image Reconstruction Through Short-Term Plasticity for Spiking Cameras
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yajing Zheng, Lingxiao Zheng, Zhaofei Yu, Boxin Shi, Yonghong Tian, Tiejun Huang
    * Abstract: Fovea, located in the centre of the retina, is specialized for high-acuity vision. Mimicking the sampling mechanism of the fovea, a retina-inspired camera, named spiking camera, is developed to record the external information with a sampling rate of 40,000 Hz, and outputs asynchronous binary spike streams. Although the temporal resolution of visual information is improved, how to reconstruct the scenes is still a challenging problem. In this paper, we present a novel high-speed image reconstruction model through the short-term plasticity (STP) mechanism of the brain. We derive the relationship between postsynaptic potential regulated by STP and the firing frequency of each pixel. By setting up the STP model at each pixel of the spiking camera, we can infer the scene radiance with the temporal regularity of the spike stream. Moreover, we show that STP can be used to distinguish the static and motion areas and further enhance the reconstruction results. The experimental results show that our methods achieve state-of-the-art performance in both image quality and computing time.

count=2
* Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.pdf)]
    * Title: Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Maria Vakalopoulou, Konstantinos Karantzalos, Nikos Komodakis, Nikos Paragios
    * Abstract: In order to exploit the currently continuous streams of massive, multi-temporal, high-resolution remote sensing datasets there is an emerging need to address efficiently the image registration and change detection challenges. To this end, in this paper we propose a modular, scalable, metric free single shot change detection/registration method. The approach exploits a decomposed interconnected graphical model formulation where registration similarity constraints are relaxed in the presence of change detection. The deformation space is discretized, while efficient linear programming and duality principles are used to optimize a joint solution space where local consistency is imposed on the deformation and the detection space. Promising results on large scale experiments demonstrate the extreme potentials of our method.

count=2
* Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.pdf)]
    * Title: Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jhony H. Giraldo, Sajid Javed, Naoufel Werghi, Thierry Bouwmans
    * Abstract: Moving Object Detection (MOD) is a fundamental step for many computer vision applications. MOD becomes very challenging when a video sequence captured from a static or moving camera suffers from the challenges: camouflage, shadow, dynamic backgrounds, and lighting variations, to name a few. Deep learning methods have been successfully applied to address MOD with competitive performance. However, in order to handle the overfitting problem, deep learning methods require a large amount of labeled data which is a laborious task as exhaustive annotations are always not available. Moreover, some MOD deep learning methods show performance degradation in the presence of unseen video sequences because the testing and training splits of the same sequences are involved during the network learning process. In this work, we pose the problem of MOD as a node classification problem using Graph Convolutional Neural Networks (GCNNs). Our algorithm, dubbed as GraphMOD-Net, encompasses instance segmentation, background initialization, feature extraction, and graph construction. GraphMOD-Net is tested on unseen videos and outperforms state-of-the-art methods in unsupervised, semi-supervised, and supervised learning in several challenges of the Change Detection 2014 (CDNet2014) and UCSD background subtraction datasets.

count=1
* Patches, Planes and Probabilities: A Non-Local Prior for Volumetric 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Ulusoy_Patches_Planes_and_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Ulusoy_Patches_Planes_and_CVPR_2016_paper.pdf)]
    * Title: Patches, Planes and Probabilities: A Non-Local Prior for Volumetric 3D Reconstruction
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Ali Osman Ulusoy, Michael J. Black, Andreas Geiger
    * Abstract: In this paper, we propose a non-local structured prior for volumetric multi-view 3D reconstruction. Towards this goal, we present a novel Markov random field model based on ray potentials in which assumptions about large 3D surface patches such as planarity or Manhattan world constraints can be efficiently encoded as probabilistic priors. We further derive an inference algorithm that reasons jointly about voxels, pixels and image segments, and estimates marginal distributions of appearance, occupancy, depth, normals and planarity. Key to tractable inference is a novel hybrid representation that spans both voxel and pixel space and that integrates non-local information from 2D image segmentations in a principled way. We compare our non-local prior to commonly employed local smoothness assumptions and a variety of state-of-the-art volumetric reconstruction baselines on challenging outdoor scenes with textureless and reflective surfaces. Our experiments indicate that regularizing over larger distances has the potential to resolve ambiguities where local regularizers fail.

count=1
* Object Tracking by Reconstruction With View-Specific Discriminative Correlation Filters
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Kart_Object_Tracking_by_Reconstruction_With_View-Specific_Discriminative_Correlation_Filters_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kart_Object_Tracking_by_Reconstruction_With_View-Specific_Discriminative_Correlation_Filters_CVPR_2019_paper.pdf)]
    * Title: Object Tracking by Reconstruction With View-Specific Discriminative Correlation Filters
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Ugur Kart,  Alan Lukezic,  Matej Kristan,  Joni-Kristian Kamarainen,  Jiri Matas
    * Abstract: Standard RGB-D trackers treat the target as a 2D structure, which makes modelling appearance changes related even to out-of-plane rotation challenging. This limitation is addressed by the proposed long-term RGB-D tracker called OTR - Object Tracking by Reconstruction. OTR performs online 3D target reconstruction to facilitate robust learning of a set of view-specific discriminative correlation filters (DCFs). The 3D reconstruction supports two performance- enhancing features: (i) generation of an accurate spatial support for constrained DCF learning from its 2D projection and (ii) point-cloud based estimation of 3D pose change for selection and storage of view-specific DCFs which robustly localize the target after out-of-view rotation or heavy occlusion. Extensive evaluation on the Princeton RGB-D tracking and STC Benchmarks shows OTR outperforms the state-of-the-art by a large margin.

count=1
* Space-Time-Aware Multi-Resolution Video Enhancement
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Haris_Space-Time-Aware_Multi-Resolution_Video_Enhancement_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Haris_Space-Time-Aware_Multi-Resolution_Video_Enhancement_CVPR_2020_paper.pdf)]
    * Title: Space-Time-Aware Multi-Resolution Video Enhancement
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Muhammad Haris,  Greg Shakhnarovich,  Norimichi Ukita
    * Abstract: We consider the problem of space-time super-resolution (ST-SR): increasing spatial resolution of video frames and simultaneously interpolating frames to increase the frame rate. Modern approaches handle these axes one at a time. In contrast, our proposed model called STARnet super-resolves jointly in space and time. This allows us to leverage mutually informative relationships between time and space: higher resolution can provide more detailed information about motion, and higher frame-rate can provide better pixel alignment. The components of our model that generate latent low- and high-resolution representations during ST-SR can be used to finetune a specialized mechanism for just spatial or just temporal super-resolution. Experimental results demonstrate that STARnet improves the performances of space-time, spatial, and temporal video super-resolution by substantial margins on publicly available datasets.

count=1
* Density Invariant Contrast Maximization for Neuromorphic Earth Observations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/html/Arja_Density_Invariant_Contrast_Maximization_for_Neuromorphic_Earth_Observations_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Arja_Density_Invariant_Contrast_Maximization_for_Neuromorphic_Earth_Observations_CVPRW_2023_paper.pdf)]
    * Title: Density Invariant Contrast Maximization for Neuromorphic Earth Observations
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Sami Arja, Alexandre Marcireau, Richard L. Balthazor, Matthew G. McHarg, Saeed Afshar, Gregory Cohen
    * Abstract: Contrast maximization (CMax) techniques are widely used in event-based vision systems to estimate the motion parameters of the camera and generate high-contrast images. However, these techniques are noise-intolerance and suffer from the multiple extrema problem which arises when the scene contains more noisy events than structure, causing the contrast to be higher at multiple locations. This makes the task of estimating the camera motion extremely challenging, which is a problem for neuromorphic earth observation, because, without a proper estimation of the motion parameters, it is not possible to generate a map with high contrast, causing important details to be lost. Similar methods that use CMax addressed this problem by changing or augmenting the objective function to enable it to converge to the correct motion parameters. Our proposed solution overcomes the multiple extrema and noise-intolerance problems by correcting the warped event before calculating the contrast and offers the following advantages: it does not depend on the event data, it does not require a prior about the camera motion and keeps the rest of the CMax pipeline unchanged. This is to ensure that the contrast is only high around the correct motion parameters. Our approach enables the creation of better motion-compensated maps through an analytical compensation technique using a novel dataset from the International Space Station (ISS). Code is available at https://github.com/neuromorphicsystems/event_warping

count=1
* Effective Semantic Pixel Labelling With Convolutional Networks and Conditional Random Fields
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Paisitkriangkrai_Effective_Semantic_Pixel_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Paisitkriangkrai_Effective_Semantic_Pixel_2015_CVPR_paper.pdf)]
    * Title: Effective Semantic Pixel Labelling With Convolutional Networks and Conditional Random Fields
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Sakrapee Paisitkriangkrai, Jamie Sherrah, Pranam Janney, Anton Van-Den Hengel
    * Abstract: Large amounts of available training data and increasing computing power have led to the recent success of deep convolutional neural networks (CNN) on a large number of applications. In this paper, we propose an effective semantic pixel labelling using CNN features, hand-crafted features and Conditional Random Fields (CRFs). Both CNN and hand-crafted features are applied to dense image patches to produce per-pixel class probabilities. The CRF infers a labelling that smooths regions while respecting the edges present in the imagery. The method is applied to the ISPRS 2D semantic labelling challenge dataset with competitive classification accuracy.

count=1
* A Generic Deformation Model for Dense Non-rigid Surface Registration: A Higher-Order MRF-Based Approach
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Zeng_A_Generic_Deformation_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Zeng_A_Generic_Deformation_2013_ICCV_paper.pdf)]
    * Title: A Generic Deformation Model for Dense Non-rigid Surface Registration: A Higher-Order MRF-Based Approach
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Yun Zeng, Chaohui Wang, Xianfeng Gu, Dimitris Samaras, Nikos Paragios
    * Abstract: We propose a novel approach for dense non-rigid 3D surface registration, which brings together Riemannian geometry and graphical models. To this end, we first introduce a generic deformation model, called Canonical Distortion Coefficients (CDCs), by characterizing the deformation of every point on a surface using the distortions along its two principle directions. This model subsumes the deformation groups commonly used in surface registration such as isometry and conformality, and is able to handle more complex deformations. We also derive its discrete counterpart which can be computed very efficiently in a closed form. Based on these, we introduce a higher-order Markov Random Field (MRF) model which seamlessly integrates our deformation model and a geometry/texture similarity metric. Then we jointly establish the optimal correspondences for all the points via maximum a posteriori (MAP) inference. Moreover, we develop a parallel optimization algorithm to efficiently perform the inference for the proposed higher-order MRF model. The resulting registration algorithm outperforms state-of-the-art methods in both dense non-rigid 3D surface registration and tracking.

count=1
* Rescan: Inductive Instance Segmentation for Indoor RGBD Scans
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Halber_Rescan_Inductive_Instance_Segmentation_for_Indoor_RGBD_Scans_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Halber_Rescan_Inductive_Instance_Segmentation_for_Indoor_RGBD_Scans_ICCV_2019_paper.pdf)]
    * Title: Rescan: Inductive Instance Segmentation for Indoor RGBD Scans
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Maciej Halber,  Yifei Shi,  Kai Xu,  Thomas Funkhouser
    * Abstract: In depth-sensing applications ranging from home robotics to AR/VR, it will be common to acquire 3D scans of interior spaces repeatedly at sparse time intervals (e.g., as part of regular daily use). We propose an algorithm that analyzes these "rescans" to infer a temporal model of a scene with semantic instance information. Our algorithm operates inductively by using the temporal model resulting from past observations to infer an instance segmentation of a new scan, which is then used to update the temporal model. The model contains object instance associations across time and thus can be used to track individual objects, even though there are only sparse observations. During experiments with a new benchmark for the new task, our algorithm outperforms alternate approaches based on state-of-the-art networks for semantic instance segmentation.

count=1
* A Convex Relaxation Approach to Space Time Multi-view 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W09/html/Oswald_A_Convex_Relaxation_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W09/papers/Oswald_A_Convex_Relaxation_2013_ICCV_paper.pdf)]
    * Title: A Convex Relaxation Approach to Space Time Multi-view 3D Reconstruction
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Martin R. Oswald, Daniel Cremers
    * Abstract: We propose a convex relaxation approach to space-time 3D reconstruction from multiple videos. Generalizing the works [16], [8] to the 4D setting, we cast the problem of reconstruction over time as a binary labeling problem in a 4D space. We propose a variational formulation which combines a photoconsistency based data term with a spatiotemporal total variation regularization. In particular, we propose a novel data term that is both faster to compute and better suited for wide-baseline camera setups when photoconsistency measures are unreliable or missing. The proposed functional can be globally minimized using convex relaxation techniques. Numerous experiments on a variety of publically available data sets demonstrate that we can compute detailed and temporally consistent reconstructions. In particular, the temporal regularization allows to reduce jittering of voxels over time.

count=1
* DiSC: Differential Spectral Clustering of Features
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/a84953147312ea2e8b020e53a267321b-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/a84953147312ea2e8b020e53a267321b-Paper-Conference.pdf)]
    * Title: DiSC: Differential Spectral Clustering of Features
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Ram Dyuthi Sristi, Gal Mishne, Ariel Jaffe
    * Abstract: Selecting subsets of features that differentiate between two conditions is a key task in a broad range of scientific domains. In many applications, the features of interest form clusters with similar effects on the data at hand. To recover such clusters we develop DiSC, a data-driven approach for detecting groups of features that differentiate between conditions. For each condition, we construct a graph whose nodes correspond to the features and whose weights are functions of the similarity between them for that condition. We then apply a spectral approach to compute subsets of nodes whose connectivity pattern differs significantly between the condition-specific feature graphs. On the theoretical front, we analyze our approach with a toy example based on the stochastic block model. We evaluate DiSC on a variety of datasets, including MNIST, hyperspectral imaging, simulated scRNA-seq and task fMRI, and demonstrate that DiSC uncovers features that better differentiate between conditions compared to competing methods.

