count=52
* Learning to Detect Carried Objects with Minimal Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/html/Dondera_Learning_to_Detect_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/papers/Dondera_Learning_to_Detect_2013_CVPR_paper.pdf)]
    * Title: Learning to Detect Carried Objects with Minimal Supervision
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Radu Dondera, Vlad Morariu, Larry Davis
    * Abstract: We propose a learning-based method for detecting carried objects that generates candidate image regions from protrusion, color contrast and occlusion boundary cues, and uses a classifier to filter out the regions unlikely to be carried objects. The method achieves higher accuracy than state of the art, which can only detect protrusions from the human shape, and the discriminative model it builds for the silhouette context-based region features generalizes well. To reduce annotation effort, we investigate training the model in a Multiple Instance Learning framework where the only available supervision is "walk" and "carry" labels associated with intervals of human tracks, i.e., the spatial extent of carried objects is not annotated. We present an extension to the miSVM algorithm that uses knowledge of the fraction of positive instances in positive bags and that scales to training sets of hundreds of thousands of instances.

count=45
* Learning High-Density Regions for a Generalized Kolmogorov-Smirnov Test in High-Dimensional Data
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/6855456e2fe46a9d49d3d3af4f57443d-Paper.pdf)]
    * Title: Learning High-Density Regions for a Generalized Kolmogorov-Smirnov Test in High-Dimensional Data
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Assaf Glazer, Michael Lindenbaum, Shaul Markovitch
    * Abstract: We propose an efficient, generalized, nonparametric, statistical Kolmogorov-Smirnov test for detecting distributional change in high-dimensional data. To implement the test, we introduce a novel, hierarchical, minimum-volume sets estimator to represent the distributions to be tested. Our work is motivated by the need to detect changes in data streams, and the test is especially efficient in this context. We provide the theoretical foundations of our test and show its superiority over existing methods.

count=32
* Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Makihara_Joint_Intensity_and_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Makihara_Joint_Intensity_and_CVPR_2017_paper.pdf)]
    * Title: Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yasushi Makihara, Atsuyuki Suzuki, Daigo Muramatsu, Xiang Li, Yasushi Yagi
    * Abstract: This paper describes a joint intensity metric learning method to improve the robustness of gait recognition with silhouette-based descriptors such as gait energy images. Because existing methods often use the difference of image intensities between a matching pair (e.g., the absolute difference of gait energies for the l_1-norm) to measure a dissimilarity, large intrasubject differences derived from covariate conditions (e.g., large gait energies caused by carried objects vs. small gait energies caused by the background), may wash out subtle intersubject differences (e.g., the difference of middle-level gait energies derived from motion differences). We therefore introduce a metric on joint intensity to mitigate the large intrasubject differences as well as leverage the subtle intersubject differences. More specifically, we formulate the joint intensity and spatial metric learning in a unified framework and alternately optimize it by linear or ranking support vector machines. Experiments using the OU-ISIR treadmill data set B with the largest clothing variation and large population data set with bag, b version containing carrying status in the wild demonstrate the effectiveness of the proposed method.

count=20
* Finding Time Together: Detection and Classification of Focused Interaction in Egocentric Video
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w34/html/individuals_s.banodundee.ac.uk_s.j.z.mckennadundee.ac.uk_j.n.zhangdundee.ac.uk_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w34/individuals_s.banodundee.ac.uk_s.j.z.mckennadundee.ac.uk_j.n.zhangdundee.ac.uk_ICCV_2017_paper.pdf)]
    * Title: Finding Time Together: Detection and Classification of Focused Interaction in Egocentric Video
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Sophia Bano, Stephen J. McKenna, Jianguo Zhang
    * Abstract: Focused interaction occurs when co-present individuals, having mutual focus of attention, interact by establishing face-to-face engagement and direct conversation. Face-to-face engagement is often not maintained throughout the entirety of a focused interaction. In this paper, we present an online method for automatic classification of unconstrained egocentric (first-person perspective) videos into segments having no focused interaction, focused interaction when the camera wearer is stationary and focused interaction when the camera wearer is moving. We extract features from both audio and video data streams and perform temporal segmentation by using support vector machines with linear and non-linear kernels. We provide empirical evidence that fusion of visual face track scores, camera motion profile and audio voice activity scores is an effective combination for focused interaction classification.

count=16
* Fine-Grained Change Detection of Misaligned Scenes With Varied Illuminations
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Feng_Fine-Grained_Change_Detection_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Feng_Fine-Grained_Change_Detection_ICCV_2015_paper.pdf)]
    * Title: Fine-Grained Change Detection of Misaligned Scenes With Varied Illuminations
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Wei Feng, Fei-Peng Tian, Qian Zhang, Nan Zhang, Liang Wan, Jizhou Sun
    * Abstract: Detecting fine-grained subtle changes among a scene is critically important in practice. Previous change detection methods, focusing on detecting large-scale significant changes, cannot do this well. This paper proposes a feasible end-to-end approach to this challenging problem. We start from active camera relocation that quickly relocates camera to nearly the same pose and position of the last time observation. To guarantee detection sensitivity and accuracy of minute changes, in an observation, we capture a group of images under multiple illuminations, which need only to be roughly aligned to the last time lighting conditions. Given two times observations, we formulate fine-grained change detection as a joint optimization problem of three related factors, i.e., normal-aware lighting difference, camera geometry correction flow, and real scene change mask. We solve the three factors in a coarse-to-fine manner and achieve reliable change decision by rank minimization. We build three real-world datasets to benchmark fine-grained change detection of misaligned scenes under varied multiple lighting conditions. Extensive experiments show the superior performance of our approach over state-of-the-art change detection methods and its ability to distinguish real scene changes from false ones caused by lighting variations.

count=14
* Exploring Weak Stabilization for Motion Feature Extraction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Park_Exploring_Weak_Stabilization_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Park_Exploring_Weak_Stabilization_2013_CVPR_paper.pdf)]
    * Title: Exploring Weak Stabilization for Motion Feature Extraction
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Dennis Park, C. L. Zitnick, Deva Ramanan, Piotr Dollar
    * Abstract: We describe novel but simple motion features for the problem of detecting objects in video sequences. Previous approaches either compute optical flow or temporal differences on video frame pairs with various assumptions about stabilization. We describe a combined approach that uses coarse-scale flow and fine-scale temporal difference features. Our approach performs weak motion stabilization by factoring out camera motion and coarse object motion while preserving nonrigid motions that serve as useful cues for recognition. We show results for pedestrian detection and human pose estimation in video sequences, achieving state-of-the-art results in both. In particular, given a fixed detection rate our method achieves a five-fold reduction in false positives over prior art on the Caltech Pedestrian benchmark. Finally, we perform extensive diagnostic experiments to reveal what aspects of our system are crucial for good performance. Proper stabilization, long time-scale features, and proper normalization are all critical.

count=12
* Temporal Vegetation Modelling Using Long Short-Term Memory Networks for Crop Identification From Medium-Resolution Multi-Spectral Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Russwurm_Temporal_Vegetation_Modelling_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Russwurm_Temporal_Vegetation_Modelling_CVPR_2017_paper.pdf)]
    * Title: Temporal Vegetation Modelling Using Long Short-Term Memory Networks for Crop Identification From Medium-Resolution Multi-Spectral Satellite Images
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Marc Russwurm, Marco Korner
    * Abstract: Land-cover classification is one of the key problems in earth observation and extensively investigated over the recent decades. Usually, approaches concentrate on single-time and multi- or hyperspectral reflectance space- or airborne sensor measurements observed. However, land-cover classes, e.g., crops, change their reflective characteristics over time complicating classification at one observation time. Contrary, these features change in a systematic and predictive manner, which can be utilized in a multi-temporal approach. We use long short-term memory (LSTM) networks to extract temporal characteristics from a sequence of Sentinel-2 observations. We compare the performance of LSTM and other network architectures and a SVM baseline to show the effectiveness of dynamic temporal feature extraction. A large test area combined with rich ground truth labels was used for training and evaluation. Our LSTM variant achieves state-of-the art performance opening potential for further research.

count=11
* Scene Categorization With Spectral Features
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Khan_Scene_Categorization_With_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Khan_Scene_Categorization_With_ICCV_2017_paper.pdf)]
    * Title: Scene Categorization With Spectral Features
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Salman H. Khan, Munawar Hayat, Fatih Porikli
    * Abstract: Spectral signatures of natural scenes were earlier found to be distinctive for different scene types with varying spatial envelope properties such as openness, naturalness, ruggedness, and symmetry. Recently, such handcrafted features have been outclassed by deep learning based representations. This paper proposes a novel spectral description of convolution features, implemented efficiently as a unitary transformation within deep network architectures. To the best of our knowledge, this is the first attempt to use deep learning based spectral features explicitly for image classification task. We show that the spectral transformation decorrelates convolutional activations, which reduces co-adaptation between feature detections, thus acts as an effective regularizer. Our approach achieves significant improvements on three large-scale scene-centric datasets (MIT-67, SUN-397, and Places-205). Furthermore, we evaluated the proposed approach on the attribute detection task where its superior performance manifests its relevance to semantically meaningful characteristics of natural scenes.

count=10
* Modeling Actions through State Changes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Fathi_Modeling_Actions_through_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Fathi_Modeling_Actions_through_2013_CVPR_paper.pdf)]
    * Title: Modeling Actions through State Changes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Alireza Fathi, James M. Rehg
    * Abstract: In this paper we present a model of action based on the change in the state of the environment. Many actions involve similar dynamics and hand-object relationships, but differ in their purpose and meaning. The key to differentiating these actions is the ability to identify how they change the state of objects and materials in the environment. We propose a weakly supervised method for learning the object and material states that are necessary for recognizing daily actions. Once these state detectors are learned, we can apply them to input videos and pool their outputs to detect actions. We further demonstrate that our method can be used to segment discrete actions from a continuous video of an activity. Our results outperform state-of-the-art action recognition and activity segmentation results.

count=10
* A Spatiotemporal Oriented Energy Network for Dynamic Texture Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Hadji_A_Spatiotemporal_Oriented_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hadji_A_Spatiotemporal_Oriented_ICCV_2017_paper.pdf)]
    * Title: A Spatiotemporal Oriented Energy Network for Dynamic Texture Recognition
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Isma Hadji, Richard P. Wildes
    * Abstract: This paper presents a novel hierarchical spatiotemporal orientation representation for spacetime image analysis. It is designed to combine the benefits of the multilayer architecture of ConvNets and a more controlled approach to spacetime analysis. A distinguishing aspect of the approach is that unlike most contemporary convolutional networks no learning is involved; rather, all design decisions are specified analytically with theoretical motivations. This approach makes it possible to understand what information is being extracted at each stage and layer of processing as well as to minimize heuristic choices in design. Another key aspect of the network is its recurrent nature, whereby the output of each layer of processing feeds back to the input. To keep the network size manageable across layers, a novel cross-channel feature pooling is proposed. The multilayer architecture that results systematically reveals hierarchical image structure in terms of multiscale, multiorientation properties of visual spacetime. To illustrate its utility, the network has been applied to the task of dynamic texture recognition. Empirical evaluation on multiple standard datasets shows that it sets a new state-of-the-art.

count=9
* Human vs. Computer in Scene and Object Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Borji_Human_vs._Computer_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Borji_Human_vs._Computer_2014_CVPR_paper.pdf)]
    * Title: Human vs. Computer in Scene and Object Recognition
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Ali Borji, Laurent Itti
    * Abstract: Several decades of research in computer and primate vision have resulted in many models (some specialized for one problem, others more general) and invaluable experimental data. Here, to help focus research efforts onto the hardest unsolved problems, and bridge computer and human vision, we define a battery of 5 tests that measure the gap between human and machine performances in several dimensions (generalization across scene categories, generalization from images to edge maps and line drawings, invariance to rotation and scaling, local/global information with jumbled images, and object recognition performance). We measure model accuracy and the correlation between model and human error patterns. Experimenting over 7 datasets, where human data is available, and gauging 14 well-established models, we find that none fully resembles humans in all aspects, and we learn from each test which models and features are more promising in approaching humans in the tested dimension. Across all tests, we find that models based on local edge histograms consistently resemble humans more, while several scene statistics or "gist" models do perform well with both scenes and objects. While computer vision has long been inspired by human vision, we believe systematic efforts, such as this, will help better identify shortcomings of models and find new paths forward.

count=6
* Density-Difference Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/f2fc990265c712c49d51a18a32b39f0c-Paper.pdf)]
    * Title: Density-Difference Estimation
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Masashi Sugiyama, Takafumi Kanamori, Taiji Suzuki, Marthinus Plessis, Song Liu, Ichiro Takeuchi
    * Abstract: We address the problem of estimating the difference between two probability densities. A naive approach is a two-step procedure of first estimating two densities separately and then computing their difference. However, such a two-step procedure does not necessarily work well because the first step is performed without regard to the second step and thus a small estimation error incurred in the first stage can cause a big error in the second stage. In this paper, we propose a single-shot procedure for directly estimating the density difference without separately estimating two densities. We derive a non-parametric finite-sample error bound for the proposed single-shot density-difference estimator and show that it achieves the optimal convergence rate. We then show how the proposed density-difference estimator can be utilized in L2-distance approximation. Finally, we experimentally demonstrate the usefulness of the proposed method in robust distribution comparison such as class-prior estimation and change-point detection.

count=6
* Trimmed Density Ratio Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/ea204361fe7f024b130143eb3e189a18-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/ea204361fe7f024b130143eb3e189a18-Paper.pdf)]
    * Title: Trimmed Density Ratio Estimation
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Song Liu, Akiko Takeda, Taiji Suzuki, Kenji Fukumizu
    * Abstract: Density ratio estimation is a vital tool in both machine learning and statistical community. However, due to the unbounded nature of density ratio, the estimation proceudre can be vulnerable to corrupted data points, which often pushes the estimated ratio toward infinity. In this paper, we present a robust estimator which automatically identifies and trims outliers. The proposed estimator has a convex formulation, and the global optimum can be obtained via subgradient descent. We analyze the parameter estimation error of this estimator under high-dimensional settings. Experiments are conducted to verify the effectiveness of the estimator.

count=5
* HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Sironi_HATS_Histograms_of_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf)]
    * Title: HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Amos Sironi, Manuele Brambilla, Nicolas Bourdis, Xavier Lagorce, Ryad Benosman
    * Abstract: Event-based cameras have recently drawn the attention of the Computer Vision community thanks to their advantages in terms of high temporal resolution, low power consumption and high dynamic range, compared to traditional frame-based cameras. These properties make event-based cameras an ideal choice for autonomous vehicles, robot navigation or UAV vision, among others. However, the accuracy of event-based object classification algorithms, which is of crucial importance for any reliable system working in real-world conditions, is still far behind their frame-based counterparts. Two main reasons for this performance gap are: 1. The lack of effective low-level representations and architectures for event-based object classification and 2. The absence of large real-world event-based datasets. In this paper we address both problems. First, we introduce a novel event-based feature representation together with a new machine learning architecture. Compared to previous approaches, we use local memory units to efficiently leverage past temporal information and build a robust event-based representation. Second, we release the first large real-world event-based dataset for object classification. We compare our method to the state-of-the-art with extensive experiments, showing better classification performance and real-time computation.

count=4
* Large-Scale Damage Detection Using Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.pdf)]
    * Title: Large-Scale Damage Detection Using Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Lionel Gueguen, Raffay Hamid
    * Abstract: Satellite imagery is a valuable source of information for assessing damages in distressed areas undergoing a calamity, such as an earthquake or an armed conflict. However, the sheer amount of data required to be inspected for this assessment makes it impractical to do it manually. To address this problem, we present a semi-supervised learning framework for large-scale damage detection in satellite imagery. We present a comparative evaluation of our framework using over 88 million images collected from 4,665 square kilometers from 12 different locations around the world. To enable accurate and efficient damage detection, we introduce a novel use of hierarchical shape features in the bags-of-visual words setting. We analyze how practical factors such as sun, sensor-resolution, and satellite-angle differences impact the effectiveness of our proposed representation, and compare it to five alternative features in multiple learning settings. Finally, we demonstrate through a user-study that our semi-supervised framework results in a ten-fold reduction in human annotation time at a minimal loss in detection accuracy compared to an exhaustive manual inspection.

count=4
* The Visual Object Tracking VOT2017 Challenge Results
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w28/html/Kristan_The_Visual_Object_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Kristan_The_Visual_Object_ICCV_2017_paper.pdf)]
    * Title: The Visual Object Tracking VOT2017 Challenge Results
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pflugfelder, Luka Cehovin Zajc, Tomas Vojir, Gustav Hager, Alan Lukezic, Abdelrahman Eldesokey, Gustavo Fernandez
    * Abstract: The Visual Object Tracking challenge VOT2017 is the fifth annual tracker benchmarking activity organized by the VOT initiative. Results of 51 trackers are presented; many are state-of-the-art published at major computer vision conferences or journals in recent years. The evaluation included the standard VOT and other popular methodologies and a new "real-time" experiment simulating a situation where a tracker processes images as if provided by a continuously running sensor. Performance of the tested trackers typically by far exceeds standard baselines. The source code for most of the trackers is publicly available from the VOT page. The VOT2017 goes beyond its predecessors by (i) improving the VOT public dataset and introducing a separate VOT2017 sequestered dataset, (ii) introducing a real-time tracking experiment and (iii) releasing a redesigned toolkit that supports complex experiments. The dataset, the evaluation kit and the results are publicly available at the challenge w ....

count=3
* Geospatial Correspondences for Multimodal Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.pdf)]
    * Title: Geospatial Correspondences for Multimodal Registration
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Diego Marcos, Raffay Hamid, Devis Tuia
    * Abstract: The growing availability of very high resolution (<1 m/pixel) satellite and aerial images has opened up unprecedented opportunities to monitor and analyze the evolution of land-cover and land-use across the world. To do so, images of the same geographical areas acquired at different times and, potentially, with different sensors must be efficiently parsed to update maps and detect land-cover changes. However, a naive transfer of ground truth labels from one location in the source image to the corresponding location in the target image is not generally feasible, as these images are often only loosely registered (with up to +- 50m of non-uniform errors). Furthermore, land-cover changes in an area over time must be taken into account for an accurate ground truth transfer. To tackle these challenges, we propose a mid-level sensor-invariant representation that encodes image regions in terms of the spatial distribution of their spectral neighbors. We incorporate this representation in a Markov Random Field to simultaneously account for nonlinear mis-registrations and enforce locality priors to find matches between multi-sensor images. We show how our approach can be used to assist in several multimodal land-cover update and change detection problems.

count=3
* The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.pdf)]
    * Title: The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, Antonio M. Lopez
    * Abstract: Vision-based semantic segmentation in urban scenarios is a key functionality for autonomous driving. Recent revolutionary results of deep convolutional neural networks (DCNNs) foreshadow the advent of reliable classifiers to perform such visual tasks. However, DCNNs require learning of many parameters from raw images; thus, having a sufficient amount of diverse images with class annotations is needed. These annotations are obtained via cumbersome, human labour which is particularly challenging for semantic segmentation since pixel-level annotations are required. In this paper, we propose to use a virtual world to automatically generate realistic synthetic images with pixel-level annotations. Then, we address the question of how useful such data can be for semantic segmentation -- in particular, when using a DCNN paradigm. In order to answer this question we have generated a synthetic collection of diverse urban images, named SYNTHIA, with automatically generated class annotations. We use SYNTHIA in combination with publicly available real-world urban images with manually provided annotations. Then, we conduct experiments with DCNNs that show how the inclusion of SYNTHIA in the training stage significantly improves performance on the semantic segmentation task.

count=3
* Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Teutsch_Robust_Detection_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Teutsch_Robust_Detection_of_CVPR_2016_paper.pdf)]
    * Title: Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Michael Teutsch, Michael Grinberg
    * Abstract: Multiple object tracking in Wide Area Motion Imagery (WAMI) data is usually based on initial detections coming from background subtraction or frame differencing. However, these methods are prone to produce split and merged detections. Appearance based vehicle detection can be an alternative but is not well-suited for WAMI data since classifier models are of weak discriminative power for vehicles in top view at low resolution. We introduce a moving vehicle detection algorithm that combines 2-frame differencing with a vehicle appearance model to improve object detection. Our main contributions are (1) integration of robust vehicle detection with split/merge handling and (2) estimation of assignment likelihoods between object hypotheses in consecutive frames using an appearance based similarity measure. Without using any prior knowledge, we achieve state-of-the-art detection rates and produce tracklets that considerably simplify the data association problem for multiple object tracking.

count=3
* Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.pdf)]
    * Title: Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Vivien Sainte Fare Garnot,  Loic Landrieu,  Sebastien Giordano,  Nesrine Chehata
    * Abstract: Satellite image time series, bolstered by their growing availability, are at the forefront of an extensive effort towards automated Earth monitoring by international institutions. In particular, large-scale control of agricultural parcels is an issue of major political and economic importance. In this regard, hybrid convolutional-recurrent neural architectures have shown promising results for the automated classification of satellite image time series. We propose an alternative approach in which the convolutional layers are advantageously replaced with encoders operating on unordered sets of pixels to exploit the typically coarse resolution of publicly available satellite images. We also propose to extract temporal features using a bespoke neural architecture based on self-attention instead of recurrent networks. We demonstrate experimentally that our method not only outperforms previous state-of-the-art approaches in terms of precision, but also significantly decreases processing time and memory requirements. Lastly, we release a large open-access annotated dataset as a benchmark for future work on satellite image time series.

count=3
* A Multi-sensor Fusion Framework in 3-D
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/html/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/papers/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.pdf)]
    * Title: A Multi-sensor Fusion Framework in 3-D
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Vishal Jain, Andrew C. Miller, Joseph L. Mundy
    * Abstract: The majority of existing image fusion techniques operate in the 2-d image domain which perform well for imagery of planar regions but fails in presence of any 3-d relief and provides inaccurate alignment of imagery from different sensors. A framework for multi-sensor image fusion in 3-d is proposed in this paper. The imagery from different sensors, specifically EO and IR, are fused in a common 3-d reference coordinate frame. A dense probabilistic and volumetric 3-d model is reconstructed from each of the sensors. The imagery is registered by aligning the 3-d models as the underlying 3-d structure in the images is the true invariant information. The image intensities are back-projected onto a 3-d model and every discretized location (voxel) of the 3-d model stores an array of intensities from different modalities. This 3-d model is forward-projected to produce a fused image of EO and IR from any viewpoint.

count=3
* SIXO: Smoothing Inference with Twisted Objectives
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/fddc79681b2df2734c01444f9bc2a17e-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/fddc79681b2df2734c01444f9bc2a17e-Paper-Conference.pdf)]
    * Title: SIXO: Smoothing Inference with Twisted Objectives
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Dieterich Lawson, Allan Raventós, Andrew Warrington, Scott Linderman
    * Abstract: Sequential Monte Carlo (SMC) is an inference algorithm for state space models that approximates the posterior by sampling from a sequence of target distributions. The target distributions are often chosen to be the filtering distributions, but these ignore information from future observations, leading to practical and theoretical limitations in inference and model learning. We introduce SIXO, a method that instead learns target distributions that approximate the smoothing distributions, incorporating information from all observations. The key idea is to use density ratio estimation to fit functions that warp the filtering distributions into the smoothing distributions. We then use SMC with these learned targets to define a variational objective for model and proposal learning. SIXO yields provably tighter log marginal lower bounds and offers more accurate posterior inferences and parameter estimates in a variety of domains.

count=2
* Recognizing Car Fluents From Video
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Li_Recognizing_Car_Fluents_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Li_Recognizing_Car_Fluents_CVPR_2016_paper.pdf)]
    * Title: Recognizing Car Fluents From Video
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Bo Li, Tianfu Wu, Caiming Xiong, Song-Chun Zhu
    * Abstract: Physical fluents, a term originally used by Newton [40], refers to time-varying object states in dynamic scenes. In this paper, we are interested in inferring the fluents of vehicles from video. For example, a door (hood, trunk) is open or closed through various actions, light is blinking to turn. Recognizing these fluents has broad applications, yet have received scant attention in the computer vision literature. Car fluent recognition entails a unified framework for car detection, car part localization and part status recognition, which is made difficult by large structural and appearance variations, low resolutions and occlusions. This paper learns a spatial-temporal And-Or hierarchical model to represent car fluents. The learning of this model is formulated under the latent structural SVM framework. Since there are no publicly related dataset, we collect and annotate a car fluent dataset consisting of car videos with diverse fluents. In experiments, the proposed method outperforms several highly related baseline methods in terms of car fluent recognition and car part localization.

count=2
* Unsupervised Human Action Detection by Action Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/html/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/papers/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.pdf)]
    * Title: Unsupervised Human Action Detection by Action Matching
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Basura Fernando, Sareh Shirazi, Stephen Gould
    * Abstract: We propose a new task of unsupervised action detection by action matching. Given two long videos, the objective is to temporally detect all pairs of matching video segments. A pair of video segments are matched if they share the same human action. The task is category independent---it does not matter what action is being performed---and no supervision is used to discover such video segments. Unsupervised action detection by action matching allows us to align videos in a meaningful manner. As such, it can be used to discover new action categories or as an action proposal technique within, say, an action detection pipeline. We solve this new task using an effective and efficient method. We use an unsupervised temporal encoding method and exploit the temporal consistency in human actions to obtain candidate action segments. We evaluate our method on this challenging task using three activity recognition benchmarks.

count=2
* Learning to Detect Fine-Grained Change Under Variant Imaging Conditions
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w42/html/Huang_Learning_to_Detect_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w42/Huang_Learning_to_Detect_ICCV_2017_paper.pdf)]
    * Title: Learning to Detect Fine-Grained Change Under Variant Imaging Conditions
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Rui Huang, Wei Feng, Zezheng Wang, Mingyuan Fan, Liang Wan, Jizhou Sun
    * Abstract: Fine-grained change detection under variant imaging conditions is an important and challenging task for high-value scene monitoring in culture heritage. In this paper, we show that after a simple coarse alignment of lighting and camera differences, fine-grained change detection can be reliably solved by a deep network model, which is specifically composed of three functional parts, i.e., camera pose correction network (PCN), fine-grained change detection network (FCDN), and detection confidence boosting. Since our model is properly pre-trained and fine-tuned on both general and specialized data, it exhibits very good generalization capability to produce high-quality minute change detection on real-world scenes under varied imaging conditions. Extensive experiments validate the superior effectiveness and reliability over state-of-the-art methods. We have achieved 67.41% relative F1-measure improvement over the best competitor on real-world benchmark dataset.

count=2
* From Video Matching to Video Grounding
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W20/html/Evangelidis_From_Video_Matching_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W20/papers/Evangelidis_From_Video_Matching_2013_ICCV_paper.pdf)]
    * Title: From Video Matching to Video Grounding
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Georgios Evangelidis, Ferran Diego, Radu Horaud
    * Abstract: This paper addresses the background estimation problem for videos captured by moving cameras, referred to as video grounding. It essentially aims at reconstructing a video, as if it would be without foreground objects, e.g. cars or people. What differentiates video grounding from known background estimation methods is that the camera follows unconstrained motion so that background undergoes ongoing changes. We build on video matching aspects since more videos contribute to the reconstruction. Without loss of generality, we investigate a challenging case where videos are recorded by in-vehicle cameras that follow the same road. Other than video synchronization and spatiotemporal alignment, we focus on the background reconstruction by exploiting interand intra-sequence similarities. In this context, we propose a Markov random field formulation that integrates the temporal coherence of videos while it exploits the decisions of a support vector machine classifier about the backgroundness of regions in video frames. Experiments with real sequences recorded by moving vehicles verify the potential of the video grounding algorithm against state-ofart baselines.

count=2
* Trading robust representations for sample complexity through self-supervised visual experience
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2018/hash/c344336196d5ec19bd54fd14befdde87-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2018/file/c344336196d5ec19bd54fd14befdde87-Paper.pdf)]
    * Title: Trading robust representations for sample complexity through self-supervised visual experience
    * Publisher: NeurIPS
    * Publication Date: `2018`
    * Authors: Andrea Tacchetti, Stephen Voinea, Georgios Evangelopoulos
    * Abstract: Learning in small sample regimes is among the most remarkable features of the human perceptual system. This ability is related to robustness to transformations, which is acquired through visual experience in the form of weak- or self-supervision during development. We explore the idea of allowing artificial systems to learn representations of visual stimuli through weak supervision prior to downstream supervised tasks. We introduce a novel loss function for representation learning using unlabeled image sets and video sequences, and experimentally demonstrate that these representations support one-shot learning and reduce the sample complexity of multiple recognition tasks. We establish the existence of a trade-off between the sizes of weakly supervised, automatically obtained from video sequences, and fully supervised data sets. Our results suggest that equivalence sets other than class labels, which are abundant in unlabeled visual experience, can be used for self-supervised learning of semantically relevant image embeddings.

count=1
* Articulated Motion Discovery Using Pairs of Trajectories
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Pero_Articulated_Motion_Discovery_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Pero_Articulated_Motion_Discovery_2015_CVPR_paper.pdf)]
    * Title: Articulated Motion Discovery Using Pairs of Trajectories
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Luca Del Pero, Susanna Ricco, Rahul Sukthankar, Vittorio Ferrari
    * Abstract: We propose an unsupervised approach for discovering characteristic motion patterns in videos of highly articulated objects performing natural, unscripted behaviors, such as tigers in the wild. We discover consistent patterns in a bottom-up manner by analyzing the relative displacements of large numbers of ordered trajectory pairs through time, such that each trajectory is attached to a different moving part on the object. The pairs of trajectories descriptor relies entirely on motion and is more discriminative than state-of-the-art features that employ single trajectories. Our method generates temporal video intervals, each automatically trimmed to one instance of the discovered behavior, and clusters them by type (e.g., running, turning head, drinking water). We present experiments on two datasets: dogs from YouTube-Objects and a new dataset of National Geographic tiger videos. Results confirm that our proposed descriptor outperforms existing appearance- and trajectory-based descriptors (e.g., HOG and DTFs) on both datasets and enables us to segment unconstrained animal video into intervals containing single behaviors.

count=1
* Dense Semantic Labeling of Very-High-Resolution Aerial Imagery and LiDAR With Fully-Convolutional Neural Networks and Higher-Order CRFs
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Liu_Dense_Semantic_Labeling_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Liu_Dense_Semantic_Labeling_CVPR_2017_paper.pdf)]
    * Title: Dense Semantic Labeling of Very-High-Resolution Aerial Imagery and LiDAR With Fully-Convolutional Neural Networks and Higher-Order CRFs
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yansong Liu, Sankaranarayanan Piramanayagam, Sildomar T. Monteiro, Eli Saber
    * Abstract: Efficient and effective multisensor fusion techniques are demanded in order to fully exploit two complementary data modalities, e.g aerial optical imagery, and the LiDAR data. Recent efforts have been mostly devoted to exploring how to properly combine both sensor data using pre-trained deep convolutional neural networks (DCNNs) at the feature level. In this paper, we propose a decision-level fusion approach with a simpler architecture for the task of dense semantic labeling. Our proposed method first obtains two initial probabilistic labeling results from a fully-convolutional neural network and a simple classifier, e.g. logistic regression exploiting spectral channels and LiDAR data, respectively. These two outcomes are then combined within a higher-order conditional random field (CRF). The CRF inference will estimate the final dense semantic labeling results. The proposed method generates the state-of-the-art semantic labeling results.

count=1
* Going From Image to Video Saliency: Augmenting Image Salience With Dynamic Attentional Push
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Gorji_Going_From_Image_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gorji_Going_From_Image_CVPR_2018_paper.pdf)]
    * Title: Going From Image to Video Saliency: Augmenting Image Salience With Dynamic Attentional Push
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Siavash Gorji, James J. Clark
    * Abstract: We present a novel method to incorporate the recent advent in static saliency models to predict the saliency in videos. Our model augments the static saliency models with the Attentional Push effect of the photographer and the scene actors in a shared attention setting. We demonstrate that not only it is imperative to use static Attentional Push cues, noticeable performance improvement is achievable by learning the time-varying nature of Attentional Push. We propose a multi-stream Convolutional Long Short-Term Memory network (ConvLSTM) structure which augments state-of-the-art in static saliency models with dynamic Attentional Push. Our network contains four pathways, a saliency pathway and three Attentional Push pathways. The multi-pathway structure is followed by an augmenting convnet that learns to combine the complementary and time-varying outputs of the ConvLSTMs by minimizing the relative entropy between the augmented saliency and viewers fixation patterns on videos. We evaluate our model by comparing the performance of several augmented static saliency models with state-of-the-art in spatiotemporal saliency on three largest dynamic eye tracking datasets, HOLLYWOOD2, UCF-Sport and DIEM. Experimental results illustrates that solid performance gain is achievable using the proposed methodology.

count=1
* A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Lin_A_Prior-Less_Method_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lin_A_Prior-Less_Method_CVPR_2018_paper.pdf)]
    * Title: A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Chung-Ching Lin, Ying Hung
    * Abstract: This paper presents a prior-less method for tracking and clustering an unknown number of human faces and maintaining their individual identities in unconstrained videos. The key challenge is to accurately track faces with partial occlusion and drastic appearance changes in multiple shots resulting from significant variations of makeup, facial expression, head pose and illumination. To address this challenge, we propose a new multi-face tracking and re-identification algorithm, which provides high accuracy in face association in the entire video with automatic cluster number generation, and is robust to outliers. We develop a co-occurrence model of multiple body parts to seamlessly create face tracklets, and recursively link tracklets to construct a graph for extracting clusters. A Gaussian Process model is introduced to compensate the deep feature insufficiency, and is further used to refine the linking results. The advantages of the proposed algorithm are demonstrated using a variety of challenging music videos and newly introduced body-worn camera videos. The proposed method obtains significant improvements over the state of the art [51], while relying less on handling video-specific prior information to achieve high performance.

count=1
* Speed Invariant Time Surface for Learning to Detect Corner Points With Event-Based Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Manderscheid_Speed_Invariant_Time_Surface_for_Learning_to_Detect_Corner_Points_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Manderscheid_Speed_Invariant_Time_Surface_for_Learning_to_Detect_Corner_Points_CVPR_2019_paper.pdf)]
    * Title: Speed Invariant Time Surface for Learning to Detect Corner Points With Event-Based Cameras
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jacques Manderscheid,  Amos Sironi,  Nicolas Bourdis,  Davide Migliore,  Vincent Lepetit
    * Abstract: We propose a learning approach to corner detection for event-based cameras that is stable even under fast and abrupt motions. Event-based cameras offer high temporal resolution, power efficiency, and high dynamic range. However, the properties of event-based data are very different compared to standard intensity images, and simple extensions of corner detection methods designed for these images do not perform well on event-based data. We first introduce an efficient way to compute a time surface that is invariant to the speed of the objects. We then show that we can train a Random Forest to recognize events generated by a moving corner from our time surface. Random Forests are also extremely efficient, and therefore a good choice to deal with the high capture frequency of event-based cameras ---our implementation processes up to 1.6Mev/s on a single CPU. Thanks to our time surface formulation and this learning approach, our method is significantly more robust to abrupt changes of direction of the corners compared to previous ones. Our method also naturally assigns a confidence score for the corners, which can be useful for postprocessing. Moreover, we introduce a high-resolution dataset suitable for quantitative evaluation and comparison of corner detection methods for event-based cameras. We call our approach SILC, for Speed Invariant Learned Corners, and compare it to the state-of-the-art with extensive experiments, showing better performance.

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
* Seeing without Looking: Contextual Rescoring of Object Detections for AP Maximization
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Pato_Seeing_without_Looking_Contextual_Rescoring_of_Object_Detections_for_AP_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pato_Seeing_without_Looking_Contextual_Rescoring_of_Object_Detections_for_AP_CVPR_2020_paper.pdf)]
    * Title: Seeing without Looking: Contextual Rescoring of Object Detections for AP Maximization
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Lourenco V. Pato,  Renato Negrinho,  Pedro M. Q. Aguiar
    * Abstract: The majority of current object detectors lack context: class predictions are made independently from other detections. We propose to incorporate context in object detection by post-processing the output of an arbitrary detector to rescore the confidences of its detections. Rescoring is done by conditioning on contextual information from the entire set of detections: their confidences, predicted classes, and positions. We show that AP can be improved by simply reassigning the detection confidence values such that true positives that survive longer (i.e., those with the correct class and large IoU) are scored higher than false positives or detections with small IoU. In this setting, we use a bidirectional RNN with attention for contextual rescoring and introduce a training target that uses the IoU with ground truth to maximize AP for the given set of detections. The fact that our approach does not require access to visual features makes it computationally inexpensive and agnostic to the detection architecture. In spite of this simplicity, our model consistently improves AP over strong pre-trained baselines (Cascade R-CNN and Faster R-CNN with several backbones), particularly by reducing the confidence of duplicate detections (a learned form of non-maximum suppression) and removing out-of-context objects by conditioning on the confidences, classes, positions, and sizes of the co-occurrent detections. Code is available at https://github.com/LourencoVazPato/seeing-without-looking/

count=1
* Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.pdf)]
    * Title: Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah
    * Abstract: Anomaly detection in video is a challenging computer vision problem. Due to the lack of anomalous events at training time, anomaly detection requires the design of learning methods without full supervision. In this paper, we approach anomalous event detection in video through self-supervised and multi-task learning at the object level. We first utilize a pre-trained detector to detect objects. Then, we train a 3D convolutional neural network to produce discriminative anomaly-specific information by jointly learning multiple proxy tasks: three self-supervised and one based on knowledge distillation. The self-supervised tasks are: (i) discrimination of forward/backward moving objects (arrow of time), (ii) discrimination of objects in consecutive/intermittent frames (motion irregularity) and (iii) reconstruction of object-specific appearance information. The knowledge distillation task takes into account both classification and detection information, generating large prediction discrepancies between teacher and student models when anomalies occur. To the best of our knowledge, we are the first to approach anomalous event detection in video as a multi-task learning problem, integrating multiple self-supervised and knowledge distillation proxy tasks in a single architecture. Our lightweight architecture outperforms the state-of-the-art methods on three benchmarks: Avenue, ShanghaiTech and UCSD Ped2. Additionally, we perform an ablation study demonstrating the importance of integrating self-supervised learning and normality-specific distillation in a multi-task learning setting.

count=1
* Detecting and Suppressing Marine Snow for Underwater Visual SLAM
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/IMW/html/Hodne_Detecting_and_Suppressing_Marine_Snow_for_Underwater_Visual_SLAM_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/IMW/papers/Hodne_Detecting_and_Suppressing_Marine_Snow_for_Underwater_Visual_SLAM_CVPRW_2022_paper.pdf)]
    * Title: Detecting and Suppressing Marine Snow for Underwater Visual SLAM
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Lars Martin Hodne, Eirik Leikvoll, Mauhing Yip, Andreas Langeland Teigen, Annette Stahl, Rudolf Mester
    * Abstract: Conventional SLAM methods which work very well in typical above-water situations, are based on detecting keypoints that are tracked between images, from which egomotion and the 3D structure of the scene are estimated. However, in underwater environments with marine snow -- small particles of organic matter which are carried by ocean currents throughout the water column -- keypoint detectors are prone to detect the marine snow particles. As the vast majority of SLAM front ends are sensitive against outliers, and the marine snow acts as severe "motion noise", failure of the regular egomotion and 3D structure estimation is expected. For this reason, we investigate the structure and appearance of marine snow and developed two schemes which classify keypoints into "marine snow" or "clean" based on either the image patches obtained from usual keypoint detectors or the descriptors computed from these patches. This way the subsequent SLAM pipeline is protected against 'false' keypoints. We quantitatively evaluate the performance of our marine snow classifier on both real underwater video scenes as well as on simulated underwater footage that contains marine snow. These simulated image sequences have been created by extracting real marine snow elements from real underwater footage, and subsequently overlaying these on "clean" underwater videos. Qualitative evaluation is also done on a nightime road sequence with snowfall to demonstrate applicability in other areas of autonomy. We furthermore evaluate the performance and the effect of marine snow detection & suppression by integrating the snow suppression module in a full SLAM pipeline based on the pySLAM system.

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
* Unmasking the Abnormal Events in Video
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Ionescu_Unmasking_the_Abnormal_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Ionescu_Unmasking_the_Abnormal_ICCV_2017_paper.pdf)]
    * Title: Unmasking the Abnormal Events in Video
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Radu Tudor Ionescu, Sorina Smeureanu, Bogdan Alexe, Marius Popescu
    * Abstract: We propose a novel framework for abnormal event detection in video that requires no training sequences. Our framework is based on unmasking, a technique previously used for authorship verification in text documents, which we adapt to our task. We iteratively train a binary classifier to distinguish between two consecutive video sequences while removing at each step the most discriminant features. Higher training accuracy rates of the intermediately obtained classifiers represent abnormal events. To the best of our knowledge, this is the first work to apply unmasking for a computer vision task. We compare our method with several state-of-the-art supervised and unsupervised methods on four benchmark data sets. The empirical results indicate that our abnormal event detection framework can achieve state-of-the-art results, while running in real-time at 20 frames per second.

count=1
* Should We Encode Rain Streaks in Video as Deterministic or Stochastic?
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Wei_Should_We_Encode_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wei_Should_We_Encode_ICCV_2017_paper.pdf)]
    * Title: Should We Encode Rain Streaks in Video as Deterministic or Stochastic?
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Wei Wei, Lixuan Yi, Qi Xie, Qian Zhao, Deyu Meng, Zongben Xu
    * Abstract: Videos taken in the wild sometimes contain unexpected rain streaks, which brings difficulty in subsequent video processing tasks. Rain streak removal in a video (RSRV) is thus an important issue and has been attracting much attention in computer vision. Different from previous RSRV methods formulating rain streaks as a deterministic message, this work first encodes the rains in a stochastic manner, i.e., a patch-based mixture of Gaussians. Such modification makes the proposed model capable of finely adapting a wider range of rain variations instead of certain types of rain configurations as traditional. By integrating with the spatiotemporal smoothness configuration of moving objects and low-rank structure of background scene, we propose a concise model for RSRV, containing one likelihood term imposed on the rain streak layer and two prior terms on the moving object and background scene layers of the video. Experiments implemented on videos with synthetic and real rains verify the superiority of the proposed method, as com- pared with the state-of-the-art methods, both visually and quantitatively in various performance metrics.

count=1
* Curvature Generation in Curved Spaces for Few-Shot Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Gao_Curvature_Generation_in_Curved_Spaces_for_Few-Shot_Learning_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Curvature_Generation_in_Curved_Spaces_for_Few-Shot_Learning_ICCV_2021_paper.pdf)]
    * Title: Curvature Generation in Curved Spaces for Few-Shot Learning
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zhi Gao, Yuwei Wu, Yunde Jia, Mehrtash Harandi
    * Abstract: Few-shot learning describes the challenging problem of recognizing samples from unseen classes given very few labeled examples. In many cases, few-shot learning is cast as learning an embedding space that assigns test samples to their corresponding class prototypes. Previous methods assume that data of all few-shot learning tasks comply with a fixed geometrical structure, mostly a Euclidean structure. Questioning this assumption that is clearly difficult to hold in real-world scenarios and incurs distortions to data, we propose to learn a task-aware curved embedding space by making use of the hyperbolic geometry. As a result, task-specific embedding spaces where suitable curvatures are generated to match the characteristics of data are constructed, leading to more generic embedding spaces. We then leverage on intra-class and inter-class context information in the embedding space to generate class prototypes for discriminative classification. We conduct a comprehensive set of experiments on inductive and transductive few-shot learning, demonstrating the benefits of our proposed method over existing embedding methods.

count=1
* The Seventh Visual Object Tracking VOT2019 Challenge Results
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/VOT/Kristan_The_Seventh_Visual_Object_Tracking_VOT2019_Challenge_Results_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Kristan_The_Seventh_Visual_Object_Tracking_VOT2019_Challenge_Results_ICCVW_2019_paper.pdf)]
    * Title: The Seventh Visual Object Tracking VOT2019 Challenge Results
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Matej Kristan, Jiri Matas, Ales Leonardis, Michael Felsberg, Roman Pflugfelder, Joni-Kristian Kamarainen, Luka Cehovin Zajc, Ondrej Drbohlav, Alan Lukezic, Amanda Berg, Abdelrahman Eldesokey, Jani Kapyla, Gustavo Fernandez, Abel Gonzalez-Garcia, Alireza Memarmoghadam, Andong Lu, Anfeng He, Anton Varfolomieiev, Antoni Chan, Ardhendu Shekhar Tripathi, Arnold Smeulders, Bala Suraj Pedasingu, Bao Xin Chen, Baopeng Zhang, Baoyuan Wu, Bi Li, Bin He, Bin Yan, Bing Bai, Bing Li, Bo Li, Byeong Hak Kim, Byeong Hak Ki
    * Abstract: The Visual Object Tracking challenge VOT2019 is the seventh annual tracker benchmarking activity organized by the VOT initiative. Results of 81 trackers are presented; many are state-of-the-art trackers published at major computer vision conferences or in journals in the recent years. The evaluation included the standard VOT and other popular methodologies for short-term tracking analysis as well as the standard VOT methodology for long-term tracking analysis. The VOT2019 challenge was composed of five challenges focusing on different tracking domains: (i) VOTST2019 challenge focused on short-term tracking in RGB, (ii) VOT-RT2019 challenge focused on "real-time" shortterm tracking in RGB, (iii) VOT-LT2019 focused on longterm tracking namely coping with target disappearance and reappearance. Two new challenges have been introduced: (iv) VOT-RGBT2019 challenge focused on short-term tracking in RGB and thermal imagery and (v) VOT-RGBD2019 challenge focused on long-term tracking in RGB and depth imagery. The VOT-ST2019, VOT-RT2019 and VOT-LT2019 datasets were refreshed while new datasets were introduced for VOT-RGBT2019 and VOT-RGBD2019. The VOT toolkit has been updated to support both standard shortterm, long-term tracking and tracking with multi-channel imagery. Performance of the tested trackers typically by far exceeds standard baselines. The source code for most of the trackers is publicly available from the VOT page. The dataset, the evaluation kit and the results are publicly available at the challenge website.

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
* M-Statistic for Kernel Change-Point Detection
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2015/hash/eb1e78328c46506b46a4ac4a1e378b91-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2015/file/eb1e78328c46506b46a4ac4a1e378b91-Paper.pdf)]
    * Title: M-Statistic for Kernel Change-Point Detection
    * Publisher: NeurIPS
    * Publication Date: `2015`
    * Authors: Shuang Li, Yao Xie, Hanjun Dai, Le Song
    * Abstract: Detecting the emergence of an abrupt change-point is a classic problem in statistics and machine learning. Kernel-based nonparametric statistics have been proposed for this task which make fewer assumptions on the distributions than traditional parametric approach. However, none of the existing kernel statistics has provided a computationally efficient way to characterize the extremal behavior of the statistic. Such characterization is crucial for setting the detection threshold, to control the significance level in the offline case as well as the average run length in the online case. In this paper we propose two related computationally efficient M-statistics for kernel-based change-point detection when the amount of background data is large. A novel theoretical result of the paper is the characterization of the tail probability of these statistics using a new technique based on change-of-measure. Such characterization provides us accurate detection thresholds for both offline and online cases in computationally efficient manner, without the need to resort to the more expensive simulations such as bootstrapping. We show that our methods perform well in both synthetic and real world data.

count=1
* Dynamic Mode Decomposition with Reproducing Kernels for Koopman Spectral Analysis
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2016/hash/1728efbda81692282ba642aafd57be3a-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2016/file/1728efbda81692282ba642aafd57be3a-Paper.pdf)]
    * Title: Dynamic Mode Decomposition with Reproducing Kernels for Koopman Spectral Analysis
    * Publisher: NeurIPS
    * Publication Date: `2016`
    * Authors: Yoshinobu Kawahara
    * Abstract: A spectral analysis of the Koopman operator, which is an infinite dimensional linear operator on an observable, gives a (modal) description of the global behavior of a nonlinear dynamical system without any explicit prior knowledge of its governing equations. In this paper, we consider a spectral analysis of the Koopman operator in a reproducing kernel Hilbert space (RKHS). We propose a modal decomposition algorithm to perform the analysis using finite-length data sequences generated from a nonlinear system. The algorithm is in essence reduced to the calculation of a set of orthogonal bases for the Krylov matrix in RKHS and the eigendecomposition of the projection of the Koopman operator onto the subspace spanned by the bases. The algorithm returns a decomposition of the dynamics into a finite number of modes, and thus it can be thought of as a feature extraction procedure for a nonlinear dynamical system. Therefore, we further consider applications in machine learning using extracted features with the presented analysis. We illustrate the method on the applications using synthetic and real-world data.

count=1
* Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/466accbac9a66b805ba50e42ad715740-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/466accbac9a66b805ba50e42ad715740-Paper.pdf)]
    * Title: Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Vincent LE GUEN, Nicolas THOME
    * Abstract: This paper addresses the problem of time series forecasting for non-stationary signals and multiple future steps prediction. To handle this challenging task, we introduce DILATE (DIstortion Loss including shApe and TimE), a new objective function for training deep neural networks. DILATE aims at accurately predicting sudden changes, and explicitly incorporates two terms supporting precise shape and temporal change detection. We introduce a differentiable loss function suitable for training deep neural nets, and provide a custom back-prop implementation for speeding up optimization. We also introduce a variant of DILATE, which provides a smooth generalization of temporally-constrained Dynamic TimeWarping (DTW). Experiments carried out on various non-stationary datasets reveal the very good behaviour of DILATE compared to models trained with the standard Mean Squared Error (MSE) loss function, and also to DTW and variants. DILATE is also agnostic to the choice of the model, and we highlight its benefit for training fully connected networks as well as specialized recurrent architectures, showing its capacity to improve over state-of-the-art trajectory forecasting approaches.

count=1
* Change point detection and inference in multivariate non-parametric models under mixing conditions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/42a0de6b8a1809ceba8fdad1661be06c-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/42a0de6b8a1809ceba8fdad1661be06c-Paper-Conference.pdf)]
    * Title: Change point detection and inference in multivariate non-parametric models under mixing conditions
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, OSCAR HERNAN MADRID PADILLA, Yi Yu
    * Abstract: This paper addresses the problem of localizing and inferring multiple change points, in non-parametric multivariate time series settings. Specifically, we consider a multivariate time series with potentially short-range dependence, whose underlying distributions have Hölder smooth densities and can change over time in a piecewise-constant manner. The change points, which correspond to the times when the distribution changes, are unknown. We present the limiting distributions of the change point estimators under the scenarios where the minimal jump size vanishes or remains constant. Such results have not been revealed in the literature in non-parametric change point settings. As byproducts, we develop a sharp estimator that can accurately localize the change points in multivariate non-parametric time series, and a consistent block-type long-run variance estimator. Numerical studies are provided to complement our theoretical findings.

