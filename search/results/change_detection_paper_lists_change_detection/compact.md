count=10
* A Wide-Field-Of-View Monocentric Light Field Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Dansereau_A_Wide-Field-Of-View_Monocentric_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dansereau_A_Wide-Field-Of-View_Monocentric_CVPR_2017_paper.pdf)]
    * Title: A Wide-Field-Of-View Monocentric Light Field Camera
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Donald G. Dansereau, Glenn Schuster, Joseph Ford, Gordon Wetzstein
    * Abstract: Light field (LF) capture and processing are important in an expanding range of computer vision applications, offering rich textural and depth information and simplification of conventionally complex tasks. Although LF cameras are commercially available, no existing device offers wide field-of-view (FOV) imaging. This is due in part to the limitations of fisheye lenses, for which a fundamentally constrained entrance pupil diameter severely limits depth sensitivity. In this work we describe a novel, compact optical design that couples a monocentric lens with multiple sensors using microlens arrays, allowing LF capture with an unprecedented FOV. Leveraging capabilities of the LF representation, we propose a novel method for efficiently coupling the spherical lens and planar sensors, replacing expensive and bulky fiber bundles. We construct a single-sensor LF camera prototype, rotating the sensor relative to a fixed main lens to emulate a wide-FOV multi-sensor scenario. Finally, we describe a processing toolchain, including a convenient spherical LF parameterization, and demonstrate depth estimation and post-capture refocus for indoor and outdoor panoramas with 15 x 15 x 1600 x 200 pixels (72 MPix) and a 138-degree FOV.

count=4
* Joint Learning From Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Audebert_Joint_Learning_From_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Audebert_Joint_Learning_From_CVPR_2017_paper.pdf)]
    * Title: Joint Learning From Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Nicolas Audebert, Bertrand Le Saux, Sebastien Lefevre
    * Abstract: We investigate the use of OSM data for semantic labeling of EO images. Deep neural networks have been used in the past for remote sensing data classification from various sensors, including multispectral, hyperspectral, Radar and Lidar data. However, OSM is an abundant data source that has already been used as ground truth data, but rarely exploited as an input information layer. We study different use cases and deep network architectures to leverage this OSM data for semantic labeling of aerial and satellite images. Especially, we look into fusion based architectures and coarse-to-fine segmentation to include the OSM layer into multispectral-based deep fully convolutional networks. We illustrate how these methods can be used successfully on two public datasets: the ISPRS Potsdam and the DFC2017. We show that OSM data can efficiently be integrated into the vision-based deep learning models and that it significantly improves both the accuracy performance and the convergence.

count=4
* Learning Compositional Representation for 4D Captures With Neural ODE
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.pdf)]
    * Title: Learning Compositional Representation for 4D Captures With Neural ODE
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Boyan Jiang, Yinda Zhang, Xingkui Wei, Xiangyang Xue, Yanwei Fu
    * Abstract: Learning based representation has become the key to the success of many computer vision systems. While many 3D representations have been proposed, it is still an unaddressed problem how to represent a dynamically changing 3D object. In this paper, we introduce a compositional representation for 4D captures, i.e. a deforming 3D object over a temporal span, that disentangles shape, initial state, and motion respectively. Each component is represented by a latent code via a trained encoder. To model the motion, a neural Ordinary Differential Equation (ODE) is trained to update the initial state conditioned on the learned motion code, and a decoder takes the shape code and the updated state code to reconstruct the 3D model at each time stamp. To this end, we propose an Identity Exchange Training (IET) strategy to encourage the network to learn effectively decoupling each component. Extensive experiments demonstrate that the proposed method outperforms existing state-of-the-art deep learning based methods on 4D reconstruction, and significantly improves on various tasks, including motion transfer and completion.

count=4
* Scene Categorization With Spectral Features
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Khan_Scene_Categorization_With_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Khan_Scene_Categorization_With_ICCV_2017_paper.pdf)]
    * Title: Scene Categorization With Spectral Features
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Salman H. Khan, Munawar Hayat, Fatih Porikli
    * Abstract: Spectral signatures of natural scenes were earlier found to be distinctive for different scene types with varying spatial envelope properties such as openness, naturalness, ruggedness, and symmetry. Recently, such handcrafted features have been outclassed by deep learning based representations. This paper proposes a novel spectral description of convolution features, implemented efficiently as a unitary transformation within deep network architectures. To the best of our knowledge, this is the first attempt to use deep learning based spectral features explicitly for image classification task. We show that the spectral transformation decorrelates convolutional activations, which reduces co-adaptation between feature detections, thus acts as an effective regularizer. Our approach achieves significant improvements on three large-scale scene-centric datasets (MIT-67, SUN-397, and Places-205). Furthermore, we evaluated the proposed approach on the attribute detection task where its superior performance manifests its relevance to semantically meaningful characteristics of natural scenes.

count=3
* Fast Image Gradients Using Binary Feature Convolutions
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/St-Charles_Fast_Image_Gradients_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/St-Charles_Fast_Image_Gradients_CVPR_2016_paper.pdf)]
    * Title: Fast Image Gradients Using Binary Feature Convolutions
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: The recent increase in popularity of binary feature descriptors has opened the door to new lightweight computer vision applications. Most research efforts thus far have been dedicated to the introduction of new large-scale binary features, which are primarily used for keypoint description and matching. In this paper, we show that the side products of small-scale binary feature computations can efficiently filter images and estimate image gradients. The improved efficiency of low-level operations can be especially useful in time-constrained applications. Through our experiments, we show that efficient binary feature convolutions can be used to mimic various image processing operations, and even outperform Sobel gradient estimation in the edge detection problem, both in terms of speed and F-Measure.

count=3
* HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Sironi_HATS_Histograms_of_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf)]
    * Title: HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Amos Sironi, Manuele Brambilla, Nicolas Bourdis, Xavier Lagorce, Ryad Benosman
    * Abstract: Event-based cameras have recently drawn the attention of the Computer Vision community thanks to their advantages in terms of high temporal resolution, low power consumption and high dynamic range, compared to traditional frame-based cameras. These properties make event-based cameras an ideal choice for autonomous vehicles, robot navigation or UAV vision, among others. However, the accuracy of event-based object classification algorithms, which is of crucial importance for any reliable system working in real-world conditions, is still far behind their frame-based counterparts. Two main reasons for this performance gap are: 1. The lack of effective low-level representations and architectures for event-based object classification and 2. The absence of large real-world event-based datasets. In this paper we address both problems. First, we introduce a novel event-based feature representation together with a new machine learning architecture. Compared to previous approaches, we use local memory units to efficiently leverage past temporal information and build a robust event-based representation. Second, we release the first large real-world event-based dataset for object classification. We compare our method to the state-of-the-art with extensive experiments, showing better classification performance and real-time computation.

count=3
* Semantic-Aware Domain Generalized Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.pdf)]
    * Title: Semantic-Aware Domain Generalized Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Duo Peng, Yinjie Lei, Munawar Hayat, Yulan Guo, Wen Li
    * Abstract: Deep models trained on source domain lack generalization when evaluated on unseen target domains with different data distributions. The problem becomes even more pronounced when we have no access to target domain samples for adaptation. In this paper, we address domain generalized semantic segmentation, where a segmentation model is trained to be domain-invariant without using any target domain data. Existing approaches to tackle this problem standardize data into a unified distribution. We argue that while such a standardization promotes global normalization, the resulting features are not discriminative enough to get clear segmentation boundaries. To enhance separation between categories while simultaneously promoting domain invariance, we propose a framework including two novel modules: Semantic-Aware Normalization (SAN) and Semantic-Aware Whitening (SAW). Specifically, SAN focuses on category-level center alignment between features from different image styles, while SAW enforces distributed alignment for the already center-aligned features. With the help of SAN and SAW, we encourage both intraclass compactness and inter-class separability. We validate our approach through extensive experiments on widely-used datasets (i.e. GTAV, SYNTHIA, Cityscapes, Mapillary and BDDS). Our approach shows significant improvements over existing state-of-the-art on various backbone networks. Code is available at https://github.com/leolyj/SAN-SAW

count=3
* Learning to Detect Carried Objects with Minimal Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/html/Dondera_Learning_to_Detect_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/papers/Dondera_Learning_to_Detect_2013_CVPR_paper.pdf)]
    * Title: Learning to Detect Carried Objects with Minimal Supervision
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Radu Dondera, Vlad Morariu, Larry Davis
    * Abstract: We propose a learning-based method for detecting carried objects that generates candidate image regions from protrusion, color contrast and occlusion boundary cues, and uses a classifier to filter out the regions unlikely to be carried objects. The method achieves higher accuracy than state of the art, which can only detect protrusions from the human shape, and the discriminative model it builds for the silhouette context-based region features generalizes well. To reduce annotation effort, we investigate training the model in a Multiple Instance Learning framework where the only available supervision is "walk" and "carry" labels associated with intervals of human tracks, i.e., the spatial extent of carried objects is not annotated. We present an extension to the miSVM algorithm that uses knowledge of the fraction of positive instances in positive bags and that scales to training sets of hundreds of thousands of instances.

count=3
* Dynamic Probabilistic Volumetric Models
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Ulusoy_Dynamic_Probabilistic_Volumetric_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Ulusoy_Dynamic_Probabilistic_Volumetric_2013_ICCV_paper.pdf)]
    * Title: Dynamic Probabilistic Volumetric Models
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Ali Osman Ulusoy, Octavian Biris, Joseph L. Mundy
    * Abstract: This paper presents a probabilistic volumetric framework for image based modeling of general dynamic 3-d scenes. The framework is targeted towards high quality modeling of complex scenes evolving over thousands of frames. Extensive storage and computational resources are required in processing large scale space-time (4-d) data. Existing methods typically store separate 3-d models at each time step and do not address such limitations. A novel 4-d representation is proposed that adaptively subdivides in space and time to explain the appearance of 3-d dynamic surfaces. This representation is shown to achieve compression of 4-d data and provide efficient spatio-temporal processing. The advances of the proposed framework is demonstrated on standard datasets using free-viewpoint video and 3-d tracking applications.

count=3
* Detection and Localization of Changes in Conditional Distributions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/eb189151ced0ff808abafd16a51fec92-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/eb189151ced0ff808abafd16a51fec92-Paper-Conference.pdf)]
    * Title: Detection and Localization of Changes in Conditional Distributions
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Lizhen Nie, Dan Nicolae
    * Abstract: We study the change point problem that considers alterations in the conditional distribution of an inferential target on a set of covariates. This paired data scenario is in contrast to the standard setting where a sequentially observed variable is analyzed for potential changes in the marginal distribution. We propose new methodology for solving this problem, by starting from a simpler task that analyzes changes in conditional expectation, and generalizing the tools developed for that task to conditional distributions. Large sample properties of the proposed statistics are derived. In empirical studies, we illustrate the performance of the proposed method against baselines adapted from existing tools. Two real data applications are presented to demonstrate its potential.

count=3
* GEO-Bench: Toward Foundation Models for Earth Monitoring
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/a0644215d9cff6646fa334dfa5d29c5a-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: GEO-Bench: Toward Foundation Models for Earth Monitoring
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Björn Lütjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, Mehmet Gunturkun, Gabriel Huang, David Vazquez, Dava Newman, Yoshua Bengio, Stefano Ermon, Xiaoxiang Zhu
    * Abstract: Recent progress in self-supervision has shown that pre-training large neural networks on vast amounts of unsupervised data can lead to substantial increases in generalization to downstream tasks. Such models, recently coined foundation models, have been transformational to the field of natural language processing.Variants have also been proposed for image data, but their applicability to remote sensing tasks is limited.To stimulate the development of foundation models for Earth monitoring, we propose a benchmark comprised of six classification and six segmentation tasks, which were carefully curated and adapted to be both relevant to the field and well-suited for model evaluation. We accompany this benchmark with a robust methodology for evaluating models and reporting aggregated results to enable a reliable assessment of progress. Finally, we report results for 20 baselines to gain information about the performance of existing models.We believe that this benchmark will be a driver of progress across a variety of Earth monitoring tasks.

count=2
* Semantic 3D Reconstruction With Continuous Regularization and Ray Potentials Using a Visibility Consistency Constraint
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Savinov_Semantic_3D_Reconstruction_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Savinov_Semantic_3D_Reconstruction_CVPR_2016_paper.pdf)]
    * Title: Semantic 3D Reconstruction With Continuous Regularization and Ray Potentials Using a Visibility Consistency Constraint
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Nikolay Savinov, Christian Hane, Lubor Ladicky, Marc Pollefeys
    * Abstract: We propose an approach for dense semantic 3D reconstruction which uses a data term that is defined as potentials over viewing rays, combined with continuous surface area penalization. Our formulation is a convex relaxation which we augment with a crucial non-convex constraint that ensures exact handling of visibility. To tackle the non-convex minimization problem, we propose a majorize-minimize type strategy which converges to a critical point. We demonstrate the benefits of using the non-convex constraint experimentally. For the geometry-only case, we set a new state of the art on two datasets of the commonly used Middlebury multi-view stereo benchmark. Moreover, our general-purpose formulation directly reconstructs thin objects, which are usually treated with specialized algorithms. A qualitative evaluation on the dense semantic 3D reconstruction task shows that we improve significantly over previous methods.

count=2
* Speed Invariant Time Surface for Learning to Detect Corner Points With Event-Based Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Manderscheid_Speed_Invariant_Time_Surface_for_Learning_to_Detect_Corner_Points_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Manderscheid_Speed_Invariant_Time_Surface_for_Learning_to_Detect_Corner_Points_CVPR_2019_paper.pdf)]
    * Title: Speed Invariant Time Surface for Learning to Detect Corner Points With Event-Based Cameras
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jacques Manderscheid,  Amos Sironi,  Nicolas Bourdis,  Davide Migliore,  Vincent Lepetit
    * Abstract: We propose a learning approach to corner detection for event-based cameras that is stable even under fast and abrupt motions. Event-based cameras offer high temporal resolution, power efficiency, and high dynamic range. However, the properties of event-based data are very different compared to standard intensity images, and simple extensions of corner detection methods designed for these images do not perform well on event-based data. We first introduce an efficient way to compute a time surface that is invariant to the speed of the objects. We then show that we can train a Random Forest to recognize events generated by a moving corner from our time surface. Random Forests are also extremely efficient, and therefore a good choice to deal with the high capture frequency of event-based cameras ---our implementation processes up to 1.6Mev/s on a single CPU. Thanks to our time surface formulation and this learning approach, our method is significantly more robust to abrupt changes of direction of the corners compared to previous ones. Our method also naturally assigns a confidence score for the corners, which can be useful for postprocessing. Moreover, we introduce a high-resolution dataset suitable for quantitative evaluation and comparison of corner detection methods for event-based cameras. We call our approach SILC, for Speed Invariant Learned Corners, and compare it to the state-of-the-art with extensive experiments, showing better performance.

count=2
* Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Warburg_Mapillary_Street-Level_Sequences_A_Dataset_for_Lifelong_Place_Recognition_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Warburg_Mapillary_Street-Level_Sequences_A_Dataset_for_Lifelong_Place_Recognition_CVPR_2020_paper.pdf)]
    * Title: Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Frederik Warburg,  Soren Hauberg,  Manuel Lopez-Antequera,  Pau Gargallo,  Yubin Kuang,  Javier Civera
    * Abstract: Lifelong place recognition is an essential and challenging task in computer vision with vast applications in robust localization and efficient large-scale 3D reconstruction. Progress is currently hindered by a lack of large, diverse, publicly available datasets. We contribute with Mapillary Street-Level Sequences (SLS), a large dataset for urban and suburban place recognition from image sequences. It contains more than 1.6 million images curated from the Mapillary collaborative mapping platform. The dataset is orders of magnitude larger than current data sources, and is designed to reflect the diversities of true lifelong learning. It features images from 30 major cities across six continents, hundreds of distinct cameras, and substantially different viewpoints and capture times, spanning all seasons over a nine year period. All images are geo-located with GPS and compass, and feature high-level attributes such as road type. We propose a set of benchmark tasks designed to push state-of-the-art performance and provide baseline studies. We show that current state-of-the-art methods still have a long way to go, and that the lack of diversity in existing datasets have prevented generalization to new environments. The dataset and benchmarks are available for academic research.

count=2
* Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.pdf)]
    * Title: Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Peri Akiva, Matthew Purri, Matthew Leotta
    * Abstract: Self-supervised learning aims to learn image feature representations without the usage of manually annotated labels. It is often used as a precursor step to obtain useful initial network weights which contribute to faster convergence and superior performance of downstream tasks. While self-supervision allows one to reduce the domain gap between supervised and unsupervised learning without the usage of labels, the self-supervised objective still requires a strong inductive bias to downstream tasks for effective transfer learning. In this work, we present our material and texture based self-supervision method named MATTER (MATerial and TExture Representation Learning), which is inspired by classical material and texture methods. Material and texture can effectively describe any surface, including its tactile properties, color, and specularity. By extension, effective representation of material and texture can describe other semantic classes strongly associated with said material and texture. MATTER leverages multi-temporal, spatially aligned remote sensing imagery over unchanged regions to learn invariance to illumination and viewing angle as a mechanism to achieve consistency of material and texture representation. We show that our self-supervision pre-training method allows for up to 24.22% and 6.33% performance increase in unsupervised and fine-tuned setups, and up to 76% faster convergence on change detection, land cover classification, and semantic segmentation tasks.

count=2
* Towards Progressive Multi-Frequency Representation for Image Warping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Towards_Progressive_Multi-Frequency_Representation_for_Image_Warping_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Towards_Progressive_Multi-Frequency_Representation_for_Image_Warping_CVPR_2024_paper.pdf)]
    * Title: Towards Progressive Multi-Frequency Representation for Image Warping
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jun Xiao, Zihang Lyu, Cong Zhang, Yakun Ju, Changjian Shui, Kin-Man Lam
    * Abstract: Image warping a classic task in computer vision aims to use geometric transformations to change the appearance of images. Recent methods learn the resampling kernels for warping through neural networks to estimate missing values in irregular grids which however fail to capture local variations in deformed content and produce images with distortion and less high-frequency details. To address this issue this paper proposes an effective method namely MFR to learn Multi-Frequency Representations from input images for image warping. Specifically we propose a progressive filtering network to learn image representations from different frequency subbands and generate deformable images in a coarse-to-fine manner. Furthermore we employ learnable Gabor wavelet filters to improve the model's capability to learn local spatial-frequency representations. Comprehensive experiments including homography transformation equirectangular to perspective projection and asymmetric image super-resolution demonstrate that the proposed MFR significantly outperforms state-of-the-art image warping methods. Our method also showcases superior generalization to out-of-distribution domains where the generated images are equipped with rich details and less distortion thereby high visual quality. The source code is available at https://github.com/junxiao01/MFR.

count=2
* Living Scenes: Multi-object Relocalization and Reconstruction in Changing 3D Environments
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Living_Scenes_Multi-object_Relocalization_and_Reconstruction_in_Changing_3D_Environments_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Living_Scenes_Multi-object_Relocalization_and_Reconstruction_in_Changing_3D_Environments_CVPR_2024_paper.pdf)]
    * Title: Living Scenes: Multi-object Relocalization and Reconstruction in Changing 3D Environments
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Liyuan Zhu, Shengyu Huang, Konrad Schindler, Iro Armeni
    * Abstract: Research into dynamic 3D scene understanding has primarily focused on short-term change tracking from dense observations while little attention has been paid to long-term changes with sparse observations. We address this gap with MoRE a novel approach for multi-object relocalization and reconstruction in evolving environments. We view these environments as Living Scenes and consider the problem of transforming scans taken at different points in time into a 3D reconstruction of the object instances whose accuracy and completeness increase over time. At the core of our method lies an SE(3) equivariant representation in a single encoder-decoder network trained on synthetic data. This representation enables us to seamlessly tackle instance matching registration and reconstruction. We also introduce a joint optimization algorithm that facilitates the accumulation of point clouds originating from the same instance across multiple scans taken at different points in time. We validate our method on synthetic and real-world data and demonstrate state-of-the-art performance in both end-to-end performance and individual subtasks.

count=2
* Robust and Optimal Sum-of-Squares-Based Point-to-Plane Registration of Image Sets and Structured Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Paudel_Robust_and_Optimal_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Paudel_Robust_and_Optimal_ICCV_2015_paper.pdf)]
    * Title: Robust and Optimal Sum-of-Squares-Based Point-to-Plane Registration of Image Sets and Structured Scenes
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Danda Pani Paudel, Adlane Habed, Cedric Demonceaux, Pascal Vasseur
    * Abstract: This paper deals with the problem of registering a known structured 3D scene and its metric Structure-from-Motion (SfM) counterpart. The proposed work relies on a prior plane segmentation of the 3D scene and aligns the data obtained from both modalities by solving the point-to-plane assignment problem. An inliers-maximization approach within a Branch-and-Bound (BnB) search scheme is adopted. For the first time in this paper, a Sum-of-Squares optimization theory framework is employed for identifying point-to-plane mismatches (i.e. outliers) with certainty. This allows us to iteratively build potential inliers sets and converge to the solution satisfied by the largest number of point-to-plane assignments. Furthermore, our approach is boosted by new plane visibility conditions which are also introduced in this paper. Using this framework, we solve the registration problem in two cases: (i) a set of putative point-to-plane correspondences (with possibly overwhelmingly many outliers) is given as input and (ii) no initial correspondences are given. In both cases, our approach yields outstanding results in terms of robustness and optimality.

count=2
* Towards Geospatial Foundation Models via Continual Pretraining
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.pdf)]
    * Title: Towards Geospatial Foundation Models via Continual Pretraining
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Matías Mendieta, Boran Han, Xingjian Shi, Yi Zhu, Chen Chen
    * Abstract: Geospatial technologies are becoming increasingly essential in our world for a wide range of applications, including agriculture, urban planning, and disaster response. To help improve the applicability and performance of deep learning models on these geospatial tasks, various works have begun investigating foundation models for this domain. Researchers have explored two prominent approaches for introducing such models in geospatial applications, but both have drawbacks in terms of limited performance benefit or prohibitive training cost. Therefore, in this work, we propose a novel paradigm for building highly effective geospatial foundation models with minimal resource cost and carbon impact. We first construct a compact yet diverse dataset from multiple sources to promote feature diversity, which we term GeoPile. Then, we investigate the potential of continual pretraining from large-scale ImageNet-22k models and propose a multi-objective continual pretraining paradigm, which leverages the strong representations of ImageNet while simultaneously providing the freedom to learn valuable in-domain features. Our approach outperforms previous state-of-the-art geospatial pretraining methods in an extensive evaluation on seven downstream datasets covering various tasks such as change detection, classification, multi-label classification, semantic segmentation, and super-resolution. Code is available at https://github.com/mmendiet/GFM.

count=2
* Multi-Frame Recurrent Adversarial Network for Moving Object Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Patil_Multi-Frame_Recurrent_Adversarial_Network_for_Moving_Object_Segmentation_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Patil_Multi-Frame_Recurrent_Adversarial_Network_for_Moving_Object_Segmentation_WACV_2021_paper.pdf)]
    * Title: Multi-Frame Recurrent Adversarial Network for Moving Object Segmentation
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Prashant W. Patil, Akshay Dudhane, Subrahmanyam Murala
    * Abstract: Moving object segmentation (MOS) in different practical scenarios like weather degraded, dynamic background, etc. videos is a challenging and high demanding task for various computer vision applications. Existing supervised approaches achieve remarkable performance with complicated training or extensive fine-tuning or inappropriate training-testing data distribution. Also, the generalized effect of existing works with completely unseen data is difficult to identify. In this work, the recurrent feature sharing based generative adversarial network is proposed with unseen video analysis. The proposed network comprises of dilated convolution to extract the spatial features at multiple scales. Along with the temporally sampled multiple frames, previous frame output is considered as input to the network. As the motion is very minute between the two consecutive frames, the previous frame decoder features are shared with encoder features recurrently for current frame foreground segmentation. This recurrent feature sharing of different layers helps the encoder network to learn the hierarchical interactions between the motion and appearance based features. Also, the learning of the proposed network is concentrated in different ways, like disjoint and global training-testing for MOS. An extensive experimental analysis of the proposed network is carried out on two benchmark video datasets with seen and unseen MOS video. Qualitative and quantitative experimental study shows that the proposed network outperforms the existing methods.

count=2
* GAF-Net: Improving the Performance of Remote Sensing Image Fusion Using Novel Global Self and Cross Attention Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Jha_GAF-Net_Improving_the_Performance_of_Remote_Sensing_Image_Fusion_Using_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Jha_GAF-Net_Improving_the_Performance_of_Remote_Sensing_Image_Fusion_Using_WACV_2023_paper.pdf)]
    * Title: GAF-Net: Improving the Performance of Remote Sensing Image Fusion Using Novel Global Self and Cross Attention Learning
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Ankit Jha, Shirsha Bose, Biplab Banerjee
    * Abstract: The notion of self and cross-attention learning has been found to substantially boost the performance of remote sensing (RS) image fusion. However, while the self-attention models fail to incorporate the global context due to the limited size of the receptive fields, cross-attention learning may generate ambiguous features as the feature extractors for all the modalities are jointly trained. This results in the generation of redundant multi-modal features, thus limiting the fusion performance. To address these issues, we propose a novel fusion architecture called Global Attention based Fusion Network (GAF-Net), equipped with novel self and cross-attention learning techniques. We introduce the within-modality feature refinement module through global spectral-spatial attention learning using the query-key-value processing where both the global spatial and channel contexts are used to generate two channel attention masks. Since it is non-trivial to generate the cross-attention from within the fusion network, we propose to leverage two auxiliary tasks of modality-specific classification to produce highly discriminative cross-attention masks. Finally, to ensure non-redundancy, we propose to penalize the high correlation between attended modality-specific features. Our extensive experiments on five benchmark datasets, including optical, multispectral (MS), hyperspectral (HSI), light detection and ranging (LiDAR), synthetic aperture radar (SAR), and audio modalities establish the superiority of GAF-Net concerning the literature.

count=2
* Neural Tangent Kernel Maximum Mean Discrepancy
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/348a38cd25abeab0e440f37510e9b1fa-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/348a38cd25abeab0e440f37510e9b1fa-Paper.pdf)]
    * Title: Neural Tangent Kernel Maximum Mean Discrepancy
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Xiuyuan Cheng, Yao Xie
    * Abstract: We present a novel neural network Maximum Mean Discrepancy (MMD) statistic by identifying a new connection between neural tangent kernel (NTK) and MMD. This connection enables us to develop a computationally efficient and memory-efficient approach to compute the MMD statistic and perform NTK based two-sample tests towards addressing the long-standing challenge of memory and computational complexity of the MMD statistic, which is essential for online implementation to assimilating new samples. Theoretically, such a connection allows us to understand the NTK test statistic properties, such as the Type-I error and testing power for performing the two-sample test, by adapting existing theories for kernel MMD. Numerical experiments on synthetic and real-world datasets validate the theory and demonstrate the effectiveness of the proposed NTK-MMD statistic.

count=2
* Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/6b8c6f846c3575e1d1ad496abea28826-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/6b8c6f846c3575e1d1ad496abea28826-Paper-Conference.pdf)]
    * Title: Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Congyue Deng, Jiahui Lei, William B Shen, Kostas Daniilidis, Leonidas J. Guibas
    * Abstract: Equivariance has gained strong interest as a desirable network property that inherently ensures robust generalization. However, when dealing with complex systems such as articulated objects or multi-object scenes, effectively capturing inter-part transformations poses a challenge, as it becomes entangled with the overall structure and local transformations. The interdependence of part assignment and per-part group action necessitates a novel equivariance formulation that allows for their co-evolution. In this paper, we present Banana, a Banach fixed-point network for equivariant segmentation with inter-part equivariance by construction. Our key insight is to iteratively solve a fixed-point problem, where point-part assignment labels and per-part SE(3)-equivariance co-evolve simultaneously. We provide theoretical derivations of both per-step equivariance and global convergence, which induces an equivariant final convergent state. Our formulation naturally provides a strict definition of inter-part equivariance that generalizes to unseen inter-part configurations. Through experiments conducted on both articulated objects and multi-object scans, we demonstrate the efficacy of our approach in achieving strong generalization under inter-part transformations, even when confronted with substantial changes in pointcloud geometry and topology.

count=1
* Recognizing Car Fluents From Video
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Li_Recognizing_Car_Fluents_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Li_Recognizing_Car_Fluents_CVPR_2016_paper.pdf)]
    * Title: Recognizing Car Fluents From Video
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Bo Li, Tianfu Wu, Caiming Xiong, Song-Chun Zhu
    * Abstract: Physical fluents, a term originally used by Newton [40], refers to time-varying object states in dynamic scenes. In this paper, we are interested in inferring the fluents of vehicles from video. For example, a door (hood, trunk) is open or closed through various actions, light is blinking to turn. Recognizing these fluents has broad applications, yet have received scant attention in the computer vision literature. Car fluent recognition entails a unified framework for car detection, car part localization and part status recognition, which is made difficult by large structural and appearance variations, low resolutions and occlusions. This paper learns a spatial-temporal And-Or hierarchical model to represent car fluents. The learning of this model is formulated under the latent structural SVM framework. Since there are no publicly related dataset, we collect and annotate a car fluent dataset consisting of car videos with diverse fluents. In experiments, the proposed method outperforms several highly related baseline methods in terms of car fluent recognition and car part localization.

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
* Background Subtraction Using Local SVD Binary Pattern
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/Guo_Background_Subtraction_Using_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/Guo_Background_Subtraction_Using_CVPR_2016_paper.pdf)]
    * Title: Background Subtraction Using Local SVD Binary Pattern
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Lili Guo, Dan Xu, Zhenping Qiang
    * Abstract: Background subtraction is a basic problem for change detection in videos and also the first step of high-level computer vision applications. Most background subtraction methods rely on color and texture feature. However, due to illuminations changes in different scenes and affections of noise pixels, those methods often resulted in high false positives in a complex environment. To solve this problem, we propose an adaptive background subtraction model which uses a novel Local SVD Binary Pattern (named LSBP) feature instead of simply depending on color intensity. This feature can describe the potential structure of the local regions in a given image, thus, it can enhance the robustness to illumination variation, noise, and shadows. We use a sample consensus model which is well suited for our LSBP feature. Experimental results on CDnet 2012 dataset demonstrate that our background subtraction method using LSBP feature is more effective than many state-of-the-art methods.

count=1
* Semantic Depth Map Fusion for Moving Vehicle Detection in Aerial Video
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w25/html/Poostchi_Semantic_Depth_Map_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w25/papers/Poostchi_Semantic_Depth_Map_CVPR_2016_paper.pdf)]
    * Title: Semantic Depth Map Fusion for Moving Vehicle Detection in Aerial Video
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Mahdieh Poostchi, Hadi Aliakbarpour, Raphael Viguier, Filiz Bunyak, Kannappan Palaniappan, Guna Seetharaman
    * Abstract: Automatic moving object detection and segmentation is one of the fundamental low-level tasks for many of the urban traffic surveillance applications. We develop an automatic moving vehicle detection system for aerial video based on semantic fusion of trace of the flux tensor and tall structures altitude mask. Trace of the flux tensor provides spatio-temporal information of moving edges including undesirable motion of tall structures caused by parallax effects. The parallax induced motions are filtered out by incorporating buildings altitude masks obtained from available dense 3D point clouds. Using a level-set based geodesic active contours framework, the coarse thresholded building depth masks evolved into the actual building boundaries. Experiments are carried out on a cropped 2kx2k region of interest for 200 frames from Albuquerque urban aerial imagery. An average precision of 83% and recall of 76% have been reported using an object-level detection performance evaluation method.

count=1
* Detecting Anomalous Objects on Mobile Platforms
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.pdf)]
    * Title: Detecting Anomalous Objects on Mobile Platforms
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Wallace Lawson, Laura Hiatt, Keith Sullivan
    * Abstract: We present an approach where a robot patrols a fixed path through an environment, autonomously locating suspicious or anomalous objects. To learn, the robot patrols this environment building a dictionary describing what is present. The dictionary is built by clustering features from a deep neural network. The objects present vary depending on the scene, which means that an object that is anomalous in one scene may be completely normal in another. To reason about this, the robot uses a computational cognitive model to learn the dictionary elements that are typically found in each scene. Once the dictionary and model has been built, the robot can patrol the environment matching objects against the dictionary, and querying the model to find the most likely objects present and to determine which objects (if any) are anomalous. We demonstrate our approach by patrolling two indoor and one outdoor environments.

count=1
* Learning Shape Trends: Parameter Estimation in Diffusions on Shape Manifolds
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w7/html/Staneva_Learning_Shape_Trends_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w7/papers/Staneva_Learning_Shape_Trends_CVPR_2017_paper.pdf)]
    * Title: Learning Shape Trends: Parameter Estimation in Diffusions on Shape Manifolds
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Valentina Staneva, Laurent Younes
    * Abstract: Learning the dynamics of shape is at the heart of many computer vision problems: object tracking, change detection, longitudinal shape analysis, trajectory classification, etc. In this work we address the problem of statistical inference of diffusion processes of shapes. We formulate a general It\^o diffusion on the manifold of deformable landmarks and propose several drift models for the evolution of shapes. We derive explicit formulas for the maximum likelihood estimators of the unknown parameters in these models, and demonstrate their convergence properties on simulated sequences when true parameters are known. We further discuss how these models can be extended to a more general non-parametric approach to shape estimation.

count=1
* TPNet: Trajectory Proposal Network for Motion Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_TPNet_Trajectory_Proposal_Network_for_Motion_Prediction_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_TPNet_Trajectory_Proposal_Network_for_Motion_Prediction_CVPR_2020_paper.pdf)]
    * Title: TPNet: Trajectory Proposal Network for Motion Prediction
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Liangji Fang,  Qinhong Jiang,  Jianping Shi,  Bolei Zhou
    * Abstract: Making accurate motion prediction of the surrounding traffic agents such as pedestrians, vehicles, and cyclists is crucial for autonomous driving. Recent data-driven motion prediction methods have attempted to learn to directly regress the exact future position or its distribution from massive amount of trajectory data. However, it remains difficult for these methods to provide multimodal predictions as well as integrate physical constraints such as traffic rules and movable areas. In this work we propose a novel two-stage motion prediction framework, Trajectory Proposal Network (TPNet). TPNet first generates a candidate set of future trajectories as hypothesis proposals, then makes the final predictions by classifying and refining the proposals which meets the physical constraints. By steering the proposal generation process, safe and multimodal predictions are realized. Thus this framework effectively mitigates the complexity of motion prediction problem while ensuring the multimodal output. Experiments on four large-scale trajectory prediction datasets, i.e. the ETH, UCY, Apollo and Argoverse datasets, show that TPNet achieves the state-of-the-art results both quantitatively and qualitatively.

count=1
* An End-to-End Edge Aggregation Network for Moving Object Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Patil_An_End-to-End_Edge_Aggregation_Network_for_Moving_Object_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Patil_An_End-to-End_Edge_Aggregation_Network_for_Moving_Object_Segmentation_CVPR_2020_paper.pdf)]
    * Title: An End-to-End Edge Aggregation Network for Moving Object Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Prashant W. Patil,  Kuldeep M. Biradar,  Akshay Dudhane,  Subrahmanyam Murala
    * Abstract: Moving object segmentation in videos (MOS) is a highly demanding task for security-based applications like automated outdoor video surveillance. Most of the existing techniques proposed for MOS are highly depend on fine-tuning a model on the first frame(s) of test sequence or complicated training procedure, which leads to limited practical serviceability of the algorithm. In this paper, the inherent correlation learning-based edge extraction mechanism (EEM) and dense residual block (DRB) are proposed for the discriminative foreground representation. The multi-scale EEM module provides the efficient foreground edge related information (with the help of encoder) to the decoder through skip connection at subsequent scale. Further, the response of the optical flow encoder stream and the last EEM module are embedded in the bridge network. The bridge network comprises of multi-scale residual blocks with dense connections to learn the effective and efficient foreground relevant features. Finally, to generate accurate and consistent foreground object maps, a decoder block is proposed with skip connections from respective multi-scale EEM module feature maps and the subsequent down-sampled response of previous frame output. Specifically, the proposed network does not require any pre-trained models or fine-tuning of the parameters with the initial frame(s) of the test video. The performance of the proposed network is evaluated with different configurations like disjoint, cross-data, and global training-testing techniques. The ablation study is conducted to analyse each model of the proposed network. To demonstrate the effectiveness of the proposed framework, a comprehensive analysis on four benchmark video datasets is conducted. Experimental results show that the proposed approach outperforms the state-of-the-art methods for MOS.

count=1
* Shadow Neural Radiance Fields for Multi-View Satellite Photogrammetry
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Derksen_Shadow_Neural_Radiance_Fields_for_Multi-View_Satellite_Photogrammetry_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Derksen_Shadow_Neural_Radiance_Fields_for_Multi-View_Satellite_Photogrammetry_CVPRW_2021_paper.pdf)]
    * Title: Shadow Neural Radiance Fields for Multi-View Satellite Photogrammetry
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Dawa Derksen, Dario Izzo
    * Abstract: We present a new generic method for shadow-aware multi-view satellite photogrammetry of Earth Observation scenes. Our proposed method, the Shadow Neural Radiance Field (S-NeRF) follows recent advances in implicit volumetric representation learning. For each scene, we train S-NeRF using very high spatial resolution optical images taken from known viewing angles. The learning requires no labels or shape priors: it is self-supervised by an image reconstruction loss. To accommodate for changing light source conditions both from a directional light source (the Sun) and a diffuse light source (the sky), we extend the NeRF approach in two ways. First, direct illumination from the Sun is modeled via a local light source visibility field. Second, indirect illumination from a diffuse light source is learned as a non-local color field as a function of the position of the Sun. Quantitatively, the combination of these factors reduces the altitude and color errors in shaded areas, compared to NeRF. The S-NeRF methodology not only performs novel view synthesis and full 3D shape estimation, it also enables shadow detection, albedo synthesis, and transient object filtering, without any explicit shape supervision.

count=1
* HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Bandara_HyperTransformer_A_Textural_and_Spectral_Feature_Fusion_Transformer_for_Pansharpening_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bandara_HyperTransformer_A_Textural_and_Spectral_Feature_Fusion_Transformer_for_Pansharpening_CVPR_2022_paper.pdf)]
    * Title: HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Wele Gedara Chaminda Bandara, Vishal M. Patel
    * Abstract: Pansharpening aims to fuse a registered high-resolution panchromatic image (PAN) with a low-resolution hyperspectral image (LR-HSI) to generate an enhanced HSI with high spectral and spatial resolution. Existing pansharpening approaches neglect using an attention mechanism to transfer HR texture features from PAN to LR-HSI features, resulting in spatial and spectral distortions. In this paper, we present a novel attention mechanism for pansharpening called HyperTransformer, in which features of LR-HSI and PAN are formulated as queries and keys in a transformer, respectively. HyperTransformer consists of three main modules, namely two separate feature extractors for PAN and HSI, a multi-head feature soft attention module, and a spatial-spectral feature fusion module. Such a network improves both spatial and spectral quality measures of the pansharpened HSI by learning cross-feature space dependencies and long-range details of PAN and LR-HSI. Furthermore, HyperTransformer can be utilized across multiple spatial scales at the backbone for obtaining improved performance. Extensive experiments conducted on three widely used datasets demonstrate that HyperTransformer achieves significant improvement over the state-of-the-art methods on both spatial and spectral quality measures. Implementation code and pre-trained weights can be accessed at https://github.com/wgcban/HyperTransformer.

count=1
* Continuous Scene Representations for Embodied AI
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Gadre_Continuous_Scene_Representations_for_Embodied_AI_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Gadre_Continuous_Scene_Representations_for_Embodied_AI_CVPR_2022_paper.pdf)]
    * Title: Continuous Scene Representations for Embodied AI
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Samir Yitzhak Gadre, Kiana Ehsani, Shuran Song, Roozbeh Mottaghi
    * Abstract: We propose Continuous Scene Representations (CSR), a scene representation constructed by an embodied agent navigating within a space, where objects and their relationships are modeled by continuous valued embeddings. Our method captures feature relationships between objects, composes them into a graph structure on-the-fly, and situates an embodied agent within the representation. Our key insight is to embed pair-wise relationships between objects in a latent space. This allows for a richer representation compared to discrete relations (e.g., [support], [next-to]) commonly used for building scene representations. CSR can track objects as the agent moves in a scene, update the representation accordingly, and detect changes in room configurations. Using CSR, we outperform state-of-the-art approaches for the challenging downstream task of visual room rearrangement, without any task specific training. Moreover, we show the learned embeddings capture salient spatial details of the scene and show applicability to real world data. A summery video and code is available at https://prior.allenai.org/projects/csr.

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
* CAMION: Cascade Multi-Input Multi-Output Network for Skeleton Extraction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/DLGC/html/Fang_CAMION_Cascade_Multi-Input_Multi-Output_Network_for_Skeleton_Extraction_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/DLGC/papers/Fang_CAMION_Cascade_Multi-Input_Multi-Output_Network_for_Skeleton_Extraction_CVPRW_2022_paper.pdf)]
    * Title: CAMION: Cascade Multi-Input Multi-Output Network for Skeleton Extraction
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Sheng Fang, Kaiyu Li, Zhe Li
    * Abstract: Skeletonization is an important process of extracting the medial axis of the object shape while maintaining the original geometric and topological properties. Some recent studies have demonstrated that deep learning-based segmentation models can extract the main skeleton from objects more robustly. However, we find that the skeleton extracted by a vanilla segmentation process is always discontinuous and not accurate enough. In this paper, we propose a general cascade deep learning pipeline that achieves competitive performance only using a simple U-shape network. The semantic information contained in the shapes is limited, so we introduce a ConvNet with multi-source input and multi-task output, CAMION for short, on top of the basic shape-to-skeleton network. With the multi-source inputs, CAMION can converge faster than using only binary shapes; and with the introduction of multi-task learning, relevant and suitable auxiliary tasks (e.g., feature point detection and contour extraction) bring considerable gains for the extraction of skeleton. Our code used in Pixel SkelNetOn - CVPR 2022 challenge will be released at https://github.com/likyoo/CAMION-CVPRW2022.

count=1
* Cross-Dataset Learning for Generalizable Land Use Scene Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Gominski_Cross-Dataset_Learning_for_Generalizable_Land_Use_Scene_Classification_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Gominski_Cross-Dataset_Learning_for_Generalizable_Land_Use_Scene_Classification_CVPRW_2022_paper.pdf)]
    * Title: Cross-Dataset Learning for Generalizable Land Use Scene Classification
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Dimitri Gominski, Valérie Gouet-Brunet, Liming Chen
    * Abstract: Few-shot and cross-domain land use scene classification methods propose solutions to classify unseen classes or unseen visual distributions, but are hardly applicable to real-world situations due to restrictive assumptions. Few-shot methods involve episodic training on restrictive training subsets with small feature extractors, while cross-domain methods are only applied to common classes. The underlying challenge remains open: can we accurately classify new scenes on new datasets? In this paper, we propose a new framework for few-shot, cross-domain classification. Our retrieval-inspired approach exploits the interrelations in both the training and testing data to output class labels using compact descriptors. Results show that our method can accurately produce land-use predictions on unseen datasets and unseen classes, going beyond the traditional few-shot or cross-domain formulation, and allowing cross-dataset training.

count=1
* EFEM: Equivariant Neural Field Expectation Maximization for 3D Object Segmentation Without Scene Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lei_EFEM_Equivariant_Neural_Field_Expectation_Maximization_for_3D_Object_Segmentation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Lei_EFEM_Equivariant_Neural_Field_Expectation_Maximization_for_3D_Object_Segmentation_CVPR_2023_paper.pdf)]
    * Title: EFEM: Equivariant Neural Field Expectation Maximization for 3D Object Segmentation Without Scene Supervision
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jiahui Lei, Congyue Deng, Karl Schmeckpeper, Leonidas Guibas, Kostas Daniilidis
    * Abstract: We introduce Equivariant Neural Field Expectation Maximization (EFEM), a simple, effective, and robust geometric algorithm that can segment objects in 3D scenes without annotations or training on scenes. We achieve such unsupervised segmentation by exploiting single object shape priors. We make two novel steps in that direction. First, we introduce equivariant shape representations to this problem to eliminate the complexity induced by the variation in object configuration. Second, we propose a novel EM algorithm that can iteratively refine segmentation masks using the equivariant shape prior. We collect a novel real dataset Chairs and Mugs that contains various object configurations and novel scenes in order to verify the effectiveness and robustness of our method. Experimental results demonstrate that our method achieves consistent and robust performance across different scenes where the (weakly) supervised methods may fail. Code and data available at https://www.cis.upenn.edu/ leijh/projects/efem

count=1
* Coupled Laplacian Eigenmaps for Locally-Aware 3D Rigid Point Cloud Matching
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Bastico_Coupled_Laplacian_Eigenmaps_for_Locally-Aware_3D_Rigid_Point_Cloud_Matching_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Bastico_Coupled_Laplacian_Eigenmaps_for_Locally-Aware_3D_Rigid_Point_Cloud_Matching_CVPR_2024_paper.pdf)]
    * Title: Coupled Laplacian Eigenmaps for Locally-Aware 3D Rigid Point Cloud Matching
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Matteo Bastico, Etienne Decencière, Laurent Corté, Yannick Tillier, David Ryckelynck
    * Abstract: Point cloud matching a crucial technique in computer vision medical and robotics fields is primarily concerned with finding correspondences between pairs of point clouds or voxels. In some practical scenarios emphasizing local differences is crucial for accurately identifying a correct match thereby enhancing the overall robustness and reliability of the matching process. Commonly used shape descriptors have several limitations and often fail to provide meaningful local insights about the paired geometries. In this work we propose a new technique based on graph Laplacian eigenmaps to match point clouds by taking into account fine local structures. To deal with the order and sign ambiguity of Laplacian eigenmaps we introduce a new operator called Coupled Laplacian that allows to easily generate aligned eigenspaces for multiple registered geometries. We show that the similarity between those aligned high-dimensional spaces provides a locally meaningful score to match shapes. We firstly evaluate the performance of the proposed technique in a point-wise manner focusing on the task of object anomaly localization on the MVTec 3D-AD dataset. Additionally we define a new medical task called automatic Bone Side Estimation (BSE) which we address through a global similarity score derived from coupled eigenspaces. In order to test it we propose a benchmark collecting bone surface structures from various public datasets. Our matching technique based on Coupled Laplacian outperforms other methods by reaching an impressive accuracy on both tasks.

count=1
* PLGSLAM: Progressive Neural Scene Represenation with Local to Global Bundle Adjustment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Deng_PLGSLAM_Progressive_Neural_Scene_Represenation_with_Local_to_Global_Bundle_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_PLGSLAM_Progressive_Neural_Scene_Represenation_with_Local_to_Global_Bundle_CVPR_2024_paper.pdf)]
    * Title: PLGSLAM: Progressive Neural Scene Represenation with Local to Global Bundle Adjustment
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tianchen Deng, Guole Shen, Tong Qin, Jianyu Wang, Wentao Zhao, Jingchuan Wang, Danwei Wang, Weidong Chen
    * Abstract: Neural implicit scene representations have recently shown encouraging results in dense visual SLAM. However existing methods produce low-quality scene reconstruction and low-accuracy localization performance when scaling up to large indoor scenes and long sequences. These limitations are mainly due to their single global radiance field with finite capacity which does not adapt to large scenarios. Their end-to-end pose networks are also not robust enough with the growth of cumulative errors in large scenes. To this end we introduce PLGSLAM a neural visual SLAM system capable of high-fidelity surface reconstruction and robust camera tracking in real-time. To handle large-scale indoor scenes PLGSLAM proposes a progressive scene representation method which dynamically allocates new local scene representation trained with frames within a local sliding window. This allows us to scale up to larger indoor scenes and improves robustness (even under pose drifts). In local scene representation PLGSLAM utilizes tri-planes for local high-frequency features with multi-layer perceptron (MLP) networks for the low-frequency feature achieving smoothness and scene completion in unobserved areas. Moreover we propose local-to-global bundle adjustment method with a global keyframe database to address the increased pose drifts on long sequences. Experimental results demonstrate that PLGSLAM achieves state-of-the-art scene reconstruction results and tracking performance across various datasets and scenarios (both in small and large-scale indoor environments).

count=1
* S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.pdf)]
    * Title: S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xuyang Li, Danfeng Hong, Jocelyn Chanussot
    * Abstract: In the expansive domain of computer vision a myriad of pre-trained models are at our disposal. However most of these models are designed for natural RGB images and prove inadequate for spectral remote sensing (RS) images. Spectral RS images have two main traits: (1) multiple bands capturing diverse feature information (2) spatial alignment and consistent spectral sequencing within the spatial-spectral dimension. In this paper we introduce Spatial-SpectralMAE (S2MAE) a specialized pre-trained architecture for spectral RS imagery. S2MAE employs a 3D transformer for masked autoencoder modeling integrating learnable spectral-spatial embeddings with a 90% masking ratio. The model efficiently captures local spectral consistency and spatial invariance using compact cube tokens demonstrating versatility to diverse input characteristics. This adaptability facilitates progressive pretraining on extensive spectral datasets. The effectiveness of S2MAE is validated through continuous pretraining on two sizable datasets totaling over a million training images. The pre-trained model is subsequently applied to three distinct downstream tasks with in-depth ablation studies conducted to emphasize its efficacy.

count=1
* Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.pdf)]
    * Title: Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Nicolae-C?t?lin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah
    * Abstract: We propose an efficient abnormal event detection model based on a lightweight masked auto-encoder (AE) applied at the video frame level. The novelty of the proposed model is threefold. First we introduce an approach to weight tokens based on motion gradients thus shifting the focus from the static background scene to the foreground objects. Second we integrate a teacher decoder and a student decoder into our architecture leveraging the discrepancy between the outputs given by the two decoders to improve anomaly detection. Third we generate synthetic abnormal events to augment the training videos and task the masked AE model to jointly reconstruct the original frames (without anomalies) and the corresponding pixel-level anomaly maps. Our design leads to an efficient and effective model as demonstrated by the extensive experiments carried out on four benchmarks: Avenue ShanghaiTech UBnormal and UCSD Ped2. The empirical results show that our model achieves an excellent trade-off between speed and accuracy obtaining competitive AUC scores while processing 1655 FPS. Hence our model is between 8 and 70 times faster than competing methods. We also conduct an ablation study to justify our design. Our code is freely available at: https://github.com/ristea/aed-mae.

count=1
* SatSynth: Augmenting Image-Mask Pairs through Diffusion Models for Aerial Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Toker_SatSynth_Augmenting_Image-Mask_Pairs_through_Diffusion_Models_for_Aerial_Semantic_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Toker_SatSynth_Augmenting_Image-Mask_Pairs_through_Diffusion_Models_for_Aerial_Semantic_CVPR_2024_paper.pdf)]
    * Title: SatSynth: Augmenting Image-Mask Pairs through Diffusion Models for Aerial Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Aysim Toker, Marvin Eisenberger, Daniel Cremers, Laura Leal-Taixé
    * Abstract: In recent years semantic segmentation has become a pivotal tool in processing and interpreting satellite imagery. Yet a prevalent limitation of supervised learning techniques remains the need for extensive manual annotations by experts. In this work we explore the potential of generative image diffusion to address the scarcity of annotated data in earth observation tasks. The main idea is to learn the joint data manifold of images and labels leveraging recent advancements in denoising diffusion probabilistic models. To the best of our knowledge we are the first to generate both images and corresponding masks for satellite segmentation. We find that the obtained pairs not only display high quality in fine-scale features but also ensure a wide sampling diversity. Both aspects are crucial for earth observation data where semantic classes can vary severely in scale and occurrence frequency. We employ the novel data instances for downstream segmentation as a form of data augmentation. In our experiments we provide comparisons to prior works based on discriminative diffusion models or GANs. We demonstrate that integrating generated samples yields significant quantitative improvements for satellite semantic segmentation -- both compared to baselines and when training only on the original data.

count=1
* Empowering Resampling Operation for Ultra-High-Definition Image Enhancement with Model-Aware Guidance
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Empowering_Resampling_Operation_for_Ultra-High-Definition_Image_Enhancement_with_Model-Aware_Guidance_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Empowering_Resampling_Operation_for_Ultra-High-Definition_Image_Enhancement_with_Model-Aware_Guidance_CVPR_2024_paper.pdf)]
    * Title: Empowering Resampling Operation for Ultra-High-Definition Image Enhancement with Model-Aware Guidance
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Wei Yu, Jie Huang, Bing Li, Kaiwen Zheng, Qi Zhu, Man Zhou, Feng Zhao
    * Abstract: Image enhancement algorithms have made remarkable advancements in recent years but directly applying them to Ultra-high-definition (UHD) images presents intractable computational overheads. Therefore previous straightforward solutions employ resampling techniques to reduce the resolution by adopting a "Downsampling-Enhancement-Upsampling" processing paradigm. However this paradigm disentangles the resampling operators and inner enhancement algorithms which results in the loss of information that is favored by the model further leading to sub-optimal outcomes. In this paper we propose a novel method of Learning Model-Aware Resampling (LMAR) which learns to customize resampling by extracting model-aware information from the UHD input image under the guidance of model knowledge. Specifically our method consists of two core designs namely compensatory kernel estimation and steganographic resampling. At the first stage we dynamically predict compensatory kernels tailored to the specific input and resampling scales. At the second stage the image-wise compensatory information is derived with the compensatory kernels and embedded into the rescaled input images. This promotes the representation of the newly derived downscaled inputs to be more consistent with the full-resolution UHD inputs as perceived by the model. Our LMAR enables model-aware and model-favored resampling while maintaining compatibility with existing resampling operators. Extensive experiments on multiple UHD image enhancement datasets and different backbones have shown consistent performance gains after correlating resizer and enhancer e.g. up to 1.2dB PSNR gain for x1.8 resampling scale on UHD-LOL4K. The code is available at \href https://github.com/YPatrickW/LMAR https://github.com/YPatrickW/LMAR .

count=1
* CDnet 2014: An Expanded Change Detection Benchmark Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Wang_CDnet_2014_An_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Wang_CDnet_2014_An_2014_CVPR_paper.pdf)]
    * Title: CDnet 2014: An Expanded Change Detection Benchmark Dataset
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Yi Wang, Pierre-Marc Jodoin, Fatih Porikli, Janusz Konrad, Yannick Benezeth, Prakash Ishwar
    * Abstract: Change detection is one of the most important low-level tasks in video analytics. In 2012, we introduced the changedetection.net (CDnet) benchmark, a video dataset devoted to the evaluation of change and motion detection approaches. Here, we present the latest release of the CDnet dataset, which includes 22 additional videos (~70,000 pixel-wise annotated frames) spanning 5 new categories that incorporate challenges encountered in many surveillance settings. We describe these categories in detail and provide an overview of the results of more than a dozen methods submitted to the IEEE Change Detection Workshop 2014. We highlight strengths and weaknesses of these methods and identify remaining issues in change detection.

count=1
* Static and Moving Object Detection Using Flux Tensor with Split Gaussian Models
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Wang_Static_and_Moving_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Wang_Static_and_Moving_2014_CVPR_paper.pdf)]
    * Title: Static and Moving Object Detection Using Flux Tensor with Split Gaussian Models
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Rui Wang, Filiz Bunyak, Guna Seetharaman, Kannappan Palaniappan
    * Abstract: In this paper, we present a moving object detection system named Flux Tensor with Split Gaussian models (FTSG) that exploits the benefits of fusing a motion computation method based on spatio-temporal tensor formulation, a novel foreground and background modeling scheme, and a multi-cue appearance comparison. This hybrid system can handle challenges such as shadows, illumination changes, dynamic background, stopped and removed objects. Extensive testing performed on the CVPR 2014 Change Detection benchmark dataset shows that FTSG outperforms state-ofthe-art methods.

count=1
* A Spatiotemporal Oriented Energy Network for Dynamic Texture Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Hadji_A_Spatiotemporal_Oriented_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hadji_A_Spatiotemporal_Oriented_ICCV_2017_paper.pdf)]
    * Title: A Spatiotemporal Oriented Energy Network for Dynamic Texture Recognition
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Isma Hadji, Richard P. Wildes
    * Abstract: This paper presents a novel hierarchical spatiotemporal orientation representation for spacetime image analysis. It is designed to combine the benefits of the multilayer architecture of ConvNets and a more controlled approach to spacetime analysis. A distinguishing aspect of the approach is that unlike most contemporary convolutional networks no learning is involved; rather, all design decisions are specified analytically with theoretical motivations. This approach makes it possible to understand what information is being extracted at each stage and layer of processing as well as to minimize heuristic choices in design. Another key aspect of the network is its recurrent nature, whereby the output of each layer of processing feeds back to the input. To keep the network size manageable across layers, a novel cross-channel feature pooling is proposed. The multilayer architecture that results systematically reveals hierarchical image structure in terms of multiscale, multiorientation properties of visual spacetime. To illustrate its utility, the network has been applied to the task of dynamic texture recognition. Empirical evaluation on multiple standard datasets shows that it sets a new state-of-the-art.

count=1
* The Visual Object Tracking VOT2017 Challenge Results
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w28/html/Kristan_The_Visual_Object_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Kristan_The_Visual_Object_ICCV_2017_paper.pdf)]
    * Title: The Visual Object Tracking VOT2017 Challenge Results
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pflugfelder, Luka Cehovin Zajc, Tomas Vojir, Gustav Hager, Alan Lukezic, Abdelrahman Eldesokey, Gustavo Fernandez
    * Abstract: The Visual Object Tracking challenge VOT2017 is the fifth annual tracker benchmarking activity organized by the VOT initiative. Results of 51 trackers are presented; many are state-of-the-art published at major computer vision conferences or journals in recent years. The evaluation included the standard VOT and other popular methodologies and a new "real-time" experiment simulating a situation where a tracker processes images as if provided by a continuously running sensor. Performance of the tested trackers typically by far exceeds standard baselines. The source code for most of the trackers is publicly available from the VOT page. The VOT2017 goes beyond its predecessors by (i) improving the VOT public dataset and introducing a separate VOT2017 sequestered dataset, (ii) introducing a real-time tracking experiment and (iii) releasing a redesigned toolkit that supports complex experiments. The dataset, the evaluation kit and the results are publicly available at the challenge w ....

count=1
* Improving Speaker Turn Embedding by Crossmodal Transfer Learning From Face Embedding
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w8/html/Le_Improving_Speaker_Turn_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w8/Le_Improving_Speaker_Turn_ICCV_2017_paper.pdf)]
    * Title: Improving Speaker Turn Embedding by Crossmodal Transfer Learning From Face Embedding
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Nam Le, Jean-Marc Odobez
    * Abstract: Learning speaker turn embeddings has shown considerable improvement in situations where conventional speaker modeling approaches fail. However, this improvement is relatively limited when compared to the gain observed in face embedding learning, which has proven very successful for face verification and clustering tasks. Assuming that face and voices from the same identities share some latent properties (like age, gender, ethnicity), we propose two transfer learning approaches to leverage the knowledge from the face domain learned from thousands of images and identities for tasks in the speaker domain. These approaches, namely target embedding transfer and clustering structure transfer, utilize the structure of the source face embedding space at different granularities to regularize the target speaker turn embedding space as optimizing terms. Our methods are evaluated on two public broadcast corpora and yield promising advances over competitive baselines in verification and audio clustering tasks, especially when dealing with short speaker utterances. The analysis of the results also gives insight into characteristics of the embedding spaces and shows their potential applications.

count=1
* Sparse-to-Dense Feature Matching: Intra and Inter Domain Cross-Modal Learning in Domain Adaptation for 3D Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Sparse-to-Dense_Feature_Matching_Intra_and_Inter_Domain_Cross-Modal_Learning_in_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Peng_Sparse-to-Dense_Feature_Matching_Intra_and_Inter_Domain_Cross-Modal_Learning_in_ICCV_2021_paper.pdf)]
    * Title: Sparse-to-Dense Feature Matching: Intra and Inter Domain Cross-Modal Learning in Domain Adaptation for 3D Semantic Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Duo Peng, Yinjie Lei, Wen Li, Pingping Zhang, Yulan Guo
    * Abstract: Domain adaptation is critical for success when confronting with the lack of annotations in a new domain. As the huge time consumption of labeling process on 3D point cloud, domain adaptation for 3D semantic segmentation is of great expectation. With the rise of multi-modal datasets, large amount of 2D images are accessible besides 3D point clouds. In light of this, we propose to further leverage 2D data for 3D domain adaptation by intra and inter domain cross modal learning. As for intra-domain cross modal learning, most existing works sample the dense 2D pixel-wise features into the same size with sparse 3D point-wise features, resulting in the abandon of numerous useful 2D features. To address this problem, we propose Dynamic sparse-to-dense Cross Modal Learning (DsCML) to increase the sufficiency of multi-modality information interaction for domain adaptation. For inter-domain cross modal learning, we further advance Cross Modal Adversarial Learning (CMAL) on 2D and 3D data which contains different semantic content aiming to promote high-level modal complementarity. We evaluate our model under various multi-modality domain adaptation settings including day-to-night, country-to-country and dataset-to-dataset, brings large improvements over both uni-modal and multi-modal domain adaptation methods on all settings.

count=1
* Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.pdf)]
    * Title: Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jhony H. Giraldo, Sajid Javed, Naoufel Werghi, Thierry Bouwmans
    * Abstract: Moving Object Detection (MOD) is a fundamental step for many computer vision applications. MOD becomes very challenging when a video sequence captured from a static or moving camera suffers from the challenges: camouflage, shadow, dynamic backgrounds, and lighting variations, to name a few. Deep learning methods have been successfully applied to address MOD with competitive performance. However, in order to handle the overfitting problem, deep learning methods require a large amount of labeled data which is a laborious task as exhaustive annotations are always not available. Moreover, some MOD deep learning methods show performance degradation in the presence of unseen video sequences because the testing and training splits of the same sequences are involved during the network learning process. In this work, we pose the problem of MOD as a node classification problem using Graph Convolutional Neural Networks (GCNNs). Our algorithm, dubbed as GraphMOD-Net, encompasses instance segmentation, background initialization, feature extraction, and graph construction. GraphMOD-Net is tested on unseen videos and outperforms state-of-the-art methods in unsupervised, semi-supervised, and supervised learning in several challenges of the Change Detection 2014 (CDNet2014) and UCSD background subtraction datasets.

count=1
* Dynamic Subtitles: A Multimodal Video Accessibility Enhancement Dedicated to Deaf and Hearing Impaired Users
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/ACVR/Tapu_Dynamic_Subtitles_A_Multimodal_Video_Accessibility_Enhancement_Dedicated_to_Deaf_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/ACVR/Tapu_Dynamic_Subtitles_A_Multimodal_Video_Accessibility_Enhancement_Dedicated_to_Deaf_ICCVW_2019_paper.pdf)]
    * Title: Dynamic Subtitles: A Multimodal Video Accessibility Enhancement Dedicated to Deaf and Hearing Impaired Users
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Ruxandra Tapu, Bogdan Mocanu, Titus Zaharia
    * Abstract: In this paper, we introduce a novel dynamic subtitle positioning system designed to increase the accessibility of the deaf and hearing impaired people to video documents. Our framework places the subtitle in the near vicinity of the active speaker in order to allow the viewer to follow the visual content while regarding the textual information. The proposed system is based on a multimodal fusion of text, audio and visual information in order to detect and recognize the identity of the active speaker. The experimental evaluation, performed on a large dataset of more than 30 videos, validates the methodology with average accuracy and recognition rates superior to 92%. The subjective evaluation demonstrates the effectiveness of our approach outperforming both conventional (static) subtitling and other state of the art techniques in terms of enhancement of the overall viewing experience and eyestrain reduction.

count=1
* Panoramic Video Separation with Online Grassmannian Robust Subspace Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/RSL-CV/Gilman_Panoramic_Video_Separation_with_Online_Grassmannian_Robust_Subspace_Estimation_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/RSL-CV/Gilman_Panoramic_Video_Separation_with_Online_Grassmannian_Robust_Subspace_Estimation_ICCVW_2019_paper.pdf)]
    * Title: Panoramic Video Separation with Online Grassmannian Robust Subspace Estimation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Kyle Gilman, Laura Balzano
    * Abstract: In this work, we propose a new total variation (TV)-regularized robust principal component analysis (RPCA) algorithm for panoramic video data with incremental gradient descent on the Grassmannian. The resulting algorithm has performance competitive with state-of-the-art panoramic RPCA algorithms and can be computed frame-by-frame to separate foreground/background in video with a freely moving camera and heavy sparse noise. We show that our algorithm scales favorably in computation time and memory. Finally we compare foreground detection accuracy and computation time of our method versus several existing methods.

count=1
* Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/html/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/papers/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.pdf)]
    * Title: Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Csaba Beleznai, Peter Gemeiner, Christian Zinner
    * Abstract: Reliable and timely detection of abandoned items in public places still represents an unsolved problem for automated visual surveillance. Typical surveilled scenarios are associated with high visual ambiguity such as shadows, occlusions, illumination changes and substantial clutter consisting of a mixture of dynamic and stationary objects. Motivated by these challenges we propose a reliable left item detection approach based on the combination of intensity and depth data from a passive stereo setup. The employed in-house developed stereo system consists of low-cost sensors and it is capable to perform detection in environments of up to 10m x 10m in size. The proposed algorithm is tested on a set of indoor sequences and compared to manually annotated ground truth data. Obtained results show that many failure modes of intensity-based approaches are absent and even small-sized objects such as a handbag can be reliably detected when left behind in a scene. The presented results display a very promising approach, which can robustly detect left luggage in dynamic environments at a close to real-time computational speed.

count=1
* Behind the Scenes: What Moving Targets Reveal about Static Scene Geometry
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/html/Taylor_Behind_the_Scenes_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/papers/Taylor_Behind_the_Scenes_2013_ICCV_paper.pdf)]
    * Title: Behind the Scenes: What Moving Targets Reveal about Static Scene Geometry
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Geoffrey Taylor, Fei Mai
    * Abstract: Reasoning about 3D scene structure is an important component of visual scene understanding. Often, reasoning proceeds from low-level cues without resorting to full 3D reconstruction. However, existing geometric cues may require multiple viewpoints, supervised training, constraints on scene structure or information from auxiliary sensors. To address these limitations, this paper demonstrates how geometric context for a single static camera can be recovered from the location and shape of moving foreground targets. In particular, we propose methods to compute the likelihood of a static occlusion boundary and floor region at each pixel. Importantly, these cues do not require supervised training, or prior knowledge of camera geometry or scene structure. Finally, we show how the proposed geometric cues can be used to infer an ordinal depth map and demonstrate its use in compositing with correct occlusion handling.

count=1
* A Novel Inspection System For Variable Data Printing Using Deep Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Haik_A_Novel_Inspection_System_For_Variable_Data_Printing_Using_Deep_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Haik_A_Novel_Inspection_System_For_Variable_Data_Printing_Using_Deep_WACV_2020_paper.pdf)]
    * Title: A Novel Inspection System For Variable Data Printing Using Deep Learning
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Oren Haik,  Oded Perry,  Eli Chen,  Peter Klammer
    * Abstract: We present a novel approach for inspecting variable data prints (VDP) with an ultra-low false alarm rate (0.005%) and potential applicability to other real-world problems. The system is based on a comparison between two images: a reference image and an image captured by low-cost scanners. The comparison task is challenging as low-cost imaging systems create artifacts that may erroneously be classified as true (genuine) defects. To address this challenge we introduce two new fusion methods, for change detection applications, which are both fast and efficient. The first is an early fusion method that combines the two input images into a single pseudo-color image. The second, called Change-Detection Single Shot Detector (CD-SSD) leverages the SSD by fusing features in the middle of the network. We demonstrate the effectiveness of the proposed deep learning-based approach with a large dataset from real-world printing scenarios. Finally, we evaluate our models on a different domain of aerial imagery change detection (AICD). Our best method clearly outperforms the state-of-the-art baseline on this dataset.

count=1
* Automatic Open-World Reliability Assessment
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Jafarzadeh_Automatic_Open-World_Reliability_Assessment_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Jafarzadeh_Automatic_Open-World_Reliability_Assessment_WACV_2021_paper.pdf)]
    * Title: Automatic Open-World Reliability Assessment
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Mohsen Jafarzadeh, Touqeer Ahmad, Akshay Raj Dhamija, Chunchun Li, Steve Cruz, Terrance E. Boult
    * Abstract: Image classification in the open-world must handle out-of-distribution (OOD) images. Systems should ideally reject OOD images, or they will map atop of known classes and reduce reliability. Using open-set classifiers that can reject OOD inputs can help. However, optimal accuracy of open-set classifiers depend on the frequency of OOD data. Thus, for either standard or open-set classifiers, it is important to be able to determine when the world changes and increasing OOD inputs will result in reduced system reliability. However, during operations, we cannot directly assess accuracy as there are no labels. Thus, the reliability assessment of these classifiers must be done by human operators, made more complex because networks are not 100% accurate, so some failures are to be expected. To automate this process, herein, we formalize the open-world recognition reliability problem and propose multiple automatic reliability assessment policies to address this new problem using only the distribution of reported scores/probability data. The distributional algorithms can be applied to both classic classifiers with SoftMax as well as the open-world Extreme Value Machine (EVM) to provide automated reliability assessment. We show that all of the new algorithms significantly outperform detection using the mean of SoftMax.

count=1
* Detection and Localization of Facial Expression Manipulations
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Mazaheri_Detection_and_Localization_of_Facial_Expression_Manipulations_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Mazaheri_Detection_and_Localization_of_Facial_Expression_Manipulations_WACV_2022_paper.pdf)]
    * Title: Detection and Localization of Facial Expression Manipulations
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Ghazal Mazaheri, Amit K. Roy-Chowdhury
    * Abstract: Concerns regarding the wide-spread use of forged images and videos in social media necessitate precise detection of such fraud. Facial manipulations can be created by Identity swap (DeepFake) or Expression swap. Contrary to the identity swap, which can easily be detected with novel deepfake detection methods, expression swap detection has not yet been addressed extensively. The importance of facial expressions in inter-person communication is known. Consequently, it is important to develop methods that can detect and localize manipulations in facial expressions. To this end, we present a novel framework to exploit the underlying feature representations of facial expressions learned from expression recognition models to identify the manipulated features. Using discriminative feature maps extracted from a facial expression recognition framework, our manipulation detector is able to localize the manipulated regions of input images and videos. On the Face2Face dataset, (abundant expression manipulation), and NeuralTextures dataset (facial expressions manipulation corresponding to the mouth regions), our method achieves higher accuracy for both classification and localization of manipulations compared to state-of-the-art methods. Furthermore, we demonstrate that our method performs at-par with the state-of-the-art methods in cases where the expression is not manipulated, but rather the identity is changed, leading to a generalized approach for facial manipulation detection.

count=1
* Semantic Segmentation in Aerial Imagery Using Multi-Level Contrastive Learning With Local Consistency
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Tang_Semantic_Segmentation_in_Aerial_Imagery_Using_Multi-Level_Contrastive_Learning_With_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Tang_Semantic_Segmentation_in_Aerial_Imagery_Using_Multi-Level_Contrastive_Learning_With_WACV_2023_paper.pdf)]
    * Title: Semantic Segmentation in Aerial Imagery Using Multi-Level Contrastive Learning With Local Consistency
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Maofeng Tang, Konstantinos Georgiou, Hairong Qi, Cody Champion, Marc Bosch
    * Abstract: Semantic segmentation in large-scale aerial images is an extremely challenging task. On one hand, the limited ground truth, as compared to the vast area the images cover, greatly hinders the development of supervised representation learning. On the other hand, the large footprint from remote sensing raises new challenges for semantic segmentation. In addition, the complex and ever changing image acquisition conditions further complicate the problem where domain shifting commonly occurs. In this paper, we exploit self-supervised contrastive learning (CL) methodologies for semantic segmentation in aerial imagery. In addition to performing CL at the feature level as most practices do, we add another level of contrastive learning, at the semantic level, taking advantage of the segmentation output from the downstream task. Further, we embed local mutual information in the semantic-level CL to enforce local consistency. This has largely enhanced the representation power at each pixel and improved the generalization capacity of the trained model. We refer to the proposed approach as multi-level contrastive learning with local consistency (mCL-LC). The experimental results on different benchmarks indicate that the proposed mCL-LC exhibits superior performance as compared to other state-of-the-art contrastive learning frameworks for the semantic segmentation task. mCL-LC also carries better generalization capacity especially when domain shifting exists.

count=1
* Multi-View 3D Object Reconstruction and Uncertainty Modelling With Neural Shape Prior
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Liao_Multi-View_3D_Object_Reconstruction_and_Uncertainty_Modelling_With_Neural_Shape_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Liao_Multi-View_3D_Object_Reconstruction_and_Uncertainty_Modelling_With_Neural_Shape_WACV_2024_paper.pdf)]
    * Title: Multi-View 3D Object Reconstruction and Uncertainty Modelling With Neural Shape Prior
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Ziwei Liao, Steven L. Waslander
    * Abstract: 3D object reconstruction is important for semantic scene understanding. It is challenging to reconstruct detailed 3D shapes from monocular images directly due to a lack of depth information, occlusion and noise. Most current methods generate deterministic object models without any awareness of the uncertainty of the reconstruction. We tackle this problem by leveraging a neural object representation which learns an object shape distribution from large dataset of 3d object models and maps it into a latent space. We propose a method to model uncertainty as part of the representation and define an uncertainty-aware encoder which generates latent codes with uncertainty directly from individual input images. Further, we propose a method to propagate the uncertainty in the latent code to SDF values and generate a 3d object mesh with local uncertainty for each mesh component. Finally, we propose an incremental fusion method under a Bayesian framework to fuse the latent codes from multi-view observations. We evaluate the system in both synthetic and real datasets to demonstrate the effectiveness of uncertainty-based fusion to improve 3D object reconstruction accuracy.

count=1
* Implicit Neural Representation for Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.pdf)]
    * Title: Implicit Neural Representation for Change Detection
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Peter Naylor, Diego Di Carlo, Arianna Traviglia, Makoto Yamada, Marco Fiorucci
    * Abstract: Identifying changes in a pair of 3D aerial LiDAR point clouds, obtained during two distinct time periods over the same geographic region presents a significant challenge due to the disparities in spatial coverage and the presence of noise in the acquisition system. The most commonly used approaches to detecting changes in point clouds are based on supervised methods which necessitate extensive labelled data often unavailable in real-world applications. To address these issues, we propose an unsupervised approach that comprises two components: Implicit Neural Representation (INR) for continuous shape reconstruction and a Gaussian Mixture Model for categorising changes. INR offers a grid-agnostic representation for encoding bi-temporal point clouds, with unmatched spatial support that can be regularised to enhance high-frequency details and reduce noise. The reconstructions at each timestamp are compared at arbitrary spatial scales, leading to a significant increase in detection capabilities. We apply our method to a benchmark dataset comprising simulated LiDAR point clouds for urban sprawling. This dataset encompasses diverse challenging scenarios, varying in resolutions, input modalities and noise levels. This enables a comprehensive multi-scenario evaluation, comparing our method with the current state-of-the-art approach. We outperform the previous methods by a margin of 10% in the intersection over union metric. In addition, we put our techniques to practical use by applying them in a real-world scenario to identify instances of illicit excavation of archaeological sites and validate our results by comparing them with findings from field experts.

count=1
* Tracking Time-varying Graphical Structure
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2013/hash/233509073ed3432027d48b1a83f5fbd2-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2013/file/233509073ed3432027d48b1a83f5fbd2-Paper.pdf)]
    * Title: Tracking Time-varying Graphical Structure
    * Publisher: NeurIPS
    * Publication Date: `2013`
    * Authors: Erich Kummerfeld, David Danks
    * Abstract: Structure learning algorithms for graphical models have focused almost exclusively on stable environments in which the underlying generative process does not change; that is, they assume that the generating model is globally stationary. In real-world environments, however, such changes often occur without warning or signal. Real-world data often come from generating models that are only locally stationary. In this paper, we present LoSST, a novel, heuristic structure learning algorithm that tracks changes in graphical model structure or parameters in a dynamic, real-time manner. We show by simulation that the algorithm performs comparably to batch-mode learning when the generating graphical structure is globally stationary, and significantly better when it is only locally stationary.

count=1
* Structure Learning with Side Information: Sample Complexity
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/e025b6279c1b88d3ec0eca6fcb6e6280-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/e025b6279c1b88d3ec0eca6fcb6e6280-Paper.pdf)]
    * Title: Structure Learning with Side Information: Sample Complexity
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Saurabh Sihag, Ali Tajer
    * Abstract: Graphical models encode the stochastic dependencies among random variables (RVs). The vertices represent the RVs, and the edges signify the conditional dependencies among the RVs. Structure learning is the process of inferring the edges by observing realizations of the RVs, and it has applications in a wide range of technological, social, and biological networks. Learning the structure of graphs when the vertices are treated in isolation from inferential information known about them is well-investigated. In a wide range of domains, however, often there exist additional inferred knowledge about the structure, which can serve as valuable side information. For instance, the gene networks that represent different subtypes of the same cancer share similar edges across all subtypes and also have exclusive edges corresponding to each subtype, rendering partially similar graphical models for gene expression in different cancer subtypes. Hence, an inferential decision regarding a gene network can serve as side information for inferring other related gene networks. When such side information is leveraged judiciously, it can translate to significant improvement in structure learning. Leveraging such side information can be abstracted as inferring structures of distinct graphical models that are {\sl partially} similar. This paper focuses on Ising graphical models, and considers the problem of simultaneously learning the structures of two {\sl partially} similar graphs, where any inference about the structure of one graph offers side information for the other graph. The bounded edge subclass of Ising models is considered, and necessary conditions (information-theoretic ), as well as sufficient conditions (algorithmic) for the sample complexity for achieving a bounded probability of error, are established. Furthermore, specific regimes are identified in which the necessary and sufficient conditions coincide, rendering the optimal sample complexity.

count=1
* SSMF: Shifting Seasonal Matrix Factorization
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/1fb2a1c37b18aa4611c3949d6148d0f8-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/1fb2a1c37b18aa4611c3949d6148d0f8-Paper.pdf)]
    * Title: SSMF: Shifting Seasonal Matrix Factorization
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Koki Kawabata, Siddharth Bhatia, Rui Liu, Mohit Wadhwa, Bryan Hooi
    * Abstract: Given taxi-ride counts information between departure and destination locations, how can we forecast their future demands? In general, given a data stream of events with seasonal patterns that innovate over time, how can we effectively and efficiently forecast future events? In this paper, we propose Shifting Seasonal Matrix Factorization approach, namely SSMF, that can adaptively learn multiple seasonal patterns (called regimes), as well as switching between them. Our proposed method has the following properties: (a) it accurately forecasts future events by detecting regime shifts in seasonal patterns as the data stream evolves; (b) it works in an online setting, i.e., processes each observation in constant time and memory; (c) it effectively realizes regime shifts without human intervention by using a lossless data compression scheme. We demonstrate that our algorithm outperforms state-of-the-art baseline methods by accurately forecasting upcoming events on three real-world data streams.

count=1
* Detecting and Adapting to Irregular Distribution Shifts in Bayesian Online Learning
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/362387494f6be6613daea643a7706a42-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/362387494f6be6613daea643a7706a42-Paper.pdf)]
    * Title: Detecting and Adapting to Irregular Distribution Shifts in Bayesian Online Learning
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Aodong Li, Alex Boyd, Padhraic Smyth, Stephan Mandt
    * Abstract: We consider the problem of online learning in the presence of distribution shifts that occur at an unknown rate and of unknown intensity. We derive a new Bayesian online inference approach to simultaneously infer these distribution shifts and adapt the model to the detected changes by integrating ideas from change point detection, switching dynamical systems, and Bayesian online learning. Using a binary ‘change variable,’ we construct an informative prior such that--if a change is detected--the model partially erases the information of past model updates by tempering to facilitate adaptation to the new data distribution. Furthermore, the approach uses beam search to track multiple change-point hypotheses and selects the most probable one in hindsight. Our proposed method is model-agnostic, applicable in both supervised and unsupervised learning settings, suitable for an environment of concept drifts or covariate drifts, and yields improvements over state-of-the-art Bayesian online learning approaches.

count=1
* D^2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/d2cc447db9e56c13b993c11b45956281-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/d2cc447db9e56c13b993c11b45956281-Paper-Conference.pdf)]
    * Title: D^2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Tianhao Wu, Fangcheng Zhong, Andrea Tagliasacchi, Forrester Cole, Cengiz Oztireli
    * Abstract: Given a monocular video, segmenting and decoupling dynamic objects while recovering the static environment is a widely studied problem in machine intelligence. Existing solutions usually approach this problem in the image domain, limiting their performance and understanding of the environment. We introduce Decoupled Dynamic Neural Radiance Field (D^2NeRF), a self-supervised approach that takes a monocular video and learns a 3D scene representation which decouples moving objects, including their shadows, from the static background. Our method represents the moving objects and the static background by two separate neural radiance fields with only one allowing for temporal changes. A naive implementation of this approach leads to the dynamic component taking over the static one as the representation of the former is inherently more general and prone to overfitting. To this end, we propose a novel loss to promote correct separation of phenomena. We further propose a shadow field network to detect and decouple dynamically moving shadows. We introduce a new dataset containing various dynamic objects and shadows and demonstrate that our method can achieve better performance than state-of-the-art approaches in decoupling dynamic and static 3D objects, occlusion and shadow removal, and image segmentation for moving objects. Project page: https://d2nerf.github.io/

count=1
* Robust Lipschitz Bandits to Adversarial Corruptions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/238f3b98bbe998b4f2234443907fe663-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/238f3b98bbe998b4f2234443907fe663-Paper-Conference.pdf)]
    * Title: Robust Lipschitz Bandits to Adversarial Corruptions
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Yue Kang, Cho-Jui Hsieh, Thomas Chun Man Lee
    * Abstract: Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.

count=1
* Change point detection and inference in multivariate non-parametric models under mixing conditions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/42a0de6b8a1809ceba8fdad1661be06c-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/42a0de6b8a1809ceba8fdad1661be06c-Paper-Conference.pdf)]
    * Title: Change point detection and inference in multivariate non-parametric models under mixing conditions
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, OSCAR HERNAN MADRID PADILLA, Yi Yu
    * Abstract: This paper addresses the problem of localizing and inferring multiple change points, in non-parametric multivariate time series settings. Specifically, we consider a multivariate time series with potentially short-range dependence, whose underlying distributions have Hölder smooth densities and can change over time in a piecewise-constant manner. The change points, which correspond to the times when the distribution changes, are unknown. We present the limiting distributions of the change point estimators under the scenarios where the minimal jump size vanishes or remains constant. Such results have not been revealed in the literature in non-parametric change point settings. As byproducts, we develop a sharp estimator that can accurately localize the change points in multivariate non-parametric time series, and a consistent block-type long-run variance estimator. Numerical studies are provided to complement our theoretical findings.

