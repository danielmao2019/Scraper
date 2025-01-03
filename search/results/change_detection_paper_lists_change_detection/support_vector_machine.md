count=5
* Trimmed Density Ratio Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/ea204361fe7f024b130143eb3e189a18-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/ea204361fe7f024b130143eb3e189a18-Paper.pdf)]
    * Title: Trimmed Density Ratio Estimation
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Song Liu, Akiko Takeda, Taiji Suzuki, Kenji Fukumizu
    * Abstract: Density ratio estimation is a vital tool in both machine learning and statistical community. However, due to the unbounded nature of density ratio, the estimation proceudre can be vulnerable to corrupted data points, which often pushes the estimated ratio toward infinity. In this paper, we present a robust estimator which automatically identifies and trims outliers. The proposed estimator has a convex formulation, and the global optimum can be obtained via subgradient descent. We analyze the parameter estimation error of this estimator under high-dimensional settings. Experiments are conducted to verify the effectiveness of the estimator.

count=4
* Temporal Vegetation Modelling Using Long Short-Term Memory Networks for Crop Identification From Medium-Resolution Multi-Spectral Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Russwurm_Temporal_Vegetation_Modelling_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Russwurm_Temporal_Vegetation_Modelling_CVPR_2017_paper.pdf)]
    * Title: Temporal Vegetation Modelling Using Long Short-Term Memory Networks for Crop Identification From Medium-Resolution Multi-Spectral Satellite Images
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Marc Russwurm, Marco Korner
    * Abstract: Land-cover classification is one of the key problems in earth observation and extensively investigated over the recent decades. Usually, approaches concentrate on single-time and multi- or hyperspectral reflectance space- or airborne sensor measurements observed. However, land-cover classes, e.g., crops, change their reflective characteristics over time complicating classification at one observation time. Contrary, these features change in a systematic and predictive manner, which can be utilized in a multi-temporal approach. We use long short-term memory (LSTM) networks to extract temporal characteristics from a sequence of Sentinel-2 observations. We compare the performance of LSTM and other network architectures and a SVM baseline to show the effectiveness of dynamic temporal feature extraction. A large test area combined with rich ground truth labels was used for training and evaluation. Our LSTM variant achieves state-of-the art performance opening potential for further research.

count=4
* Finding Time Together: Detection and Classification of Focused Interaction in Egocentric Video
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w34/html/individuals_s.banodundee.ac.uk_s.j.z.mckennadundee.ac.uk_j.n.zhangdundee.ac.uk_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w34/individuals_s.banodundee.ac.uk_s.j.z.mckennadundee.ac.uk_j.n.zhangdundee.ac.uk_ICCV_2017_paper.pdf)]
    * Title: Finding Time Together: Detection and Classification of Focused Interaction in Egocentric Video
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Sophia Bano, Stephen J. McKenna, Jianguo Zhang
    * Abstract: Focused interaction occurs when co-present individuals, having mutual focus of attention, interact by establishing face-to-face engagement and direct conversation. Face-to-face engagement is often not maintained throughout the entirety of a focused interaction. In this paper, we present an online method for automatic classification of unconstrained egocentric (first-person perspective) videos into segments having no focused interaction, focused interaction when the camera wearer is stationary and focused interaction when the camera wearer is moving. We extract features from both audio and video data streams and perform temporal segmentation by using support vector machines with linear and non-linear kernels. We provide empirical evidence that fusion of visual face track scores, camera motion profile and audio voice activity scores is an effective combination for focused interaction classification.

count=3
* Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Makihara_Joint_Intensity_and_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Makihara_Joint_Intensity_and_CVPR_2017_paper.pdf)]
    * Title: Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yasushi Makihara, Atsuyuki Suzuki, Daigo Muramatsu, Xiang Li, Yasushi Yagi
    * Abstract: This paper describes a joint intensity metric learning method to improve the robustness of gait recognition with silhouette-based descriptors such as gait energy images. Because existing methods often use the difference of image intensities between a matching pair (e.g., the absolute difference of gait energies for the l_1-norm) to measure a dissimilarity, large intrasubject differences derived from covariate conditions (e.g., large gait energies caused by carried objects vs. small gait energies caused by the background), may wash out subtle intersubject differences (e.g., the difference of middle-level gait energies derived from motion differences). We therefore introduce a metric on joint intensity to mitigate the large intrasubject differences as well as leverage the subtle intersubject differences. More specifically, we formulate the joint intensity and spatial metric learning in a unified framework and alternately optimize it by linear or ranking support vector machines. Experiments using the OU-ISIR treadmill data set B with the largest clothing variation and large population data set with bag, b version containing carrying status in the wild demonstrate the effectiveness of the proposed method.

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
* Phenology Alignment Network: A Novel Framework for Cross-Regional Time Series Crop Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AgriVision/html/Wang_Phenology_Alignment_Network_A_Novel_Framework_for_Cross-Regional_Time_Series_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AgriVision/papers/Wang_Phenology_Alignment_Network_A_Novel_Framework_for_Cross-Regional_Time_Series_CVPRW_2021_paper.pdf)]
    * Title: Phenology Alignment Network: A Novel Framework for Cross-Regional Time Series Crop Classification
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Ziqiao Wang, Hongyan Zhang, Wei He, Liangpei Zhang
    * Abstract: Timely and accurate crop type classification plays an essential role in the study of agricultural application. However, large area or cross-regional crop classification confronts huge challenges owing to dramatic phenology discrepancy among training and test regions. In this work, we propose a novel framework to address these challenges based on deep recurrent network and unsupervised domain adaptation (DA). Specifically, we firstly propose a Temporal Spatial Network (TSNet) for pixelwise crop classification, which contains stacked RNN and self-attention module to adaptively extract multi-level features from crop samples under various planting conditions. To deal with the cross-regional challenge, an unsupervised DA-based framework named Phenology Alignment Network (PAN) is proposed. PAN consists of two branches of two identical TSNet pre-trained on source domain; one branch takes source samples while the other takes target samples as input. Through aligning the hierarchical deep features extracted from two branches, the discrepancy between two regions is decreased and the pre-trained model is adapted to the target domain without using target label information. As another contribution, a time series dataset based on Sentinel-2 was annotated containing winter crop samples collected on three study sites of China. Cross-regional experiments demonstrate that TSNet shows comparable accuracy to state-of-the-art methods, and PAN further improves the overall accuracy by 5.62%, and macro average F1 score by 0.094 unsupervisedly.

count=3
* Density-Difference Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/f2fc990265c712c49d51a18a32b39f0c-Paper.pdf)]
    * Title: Density-Difference Estimation
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Masashi Sugiyama, Takafumi Kanamori, Taiji Suzuki, Marthinus Plessis, Song Liu, Ichiro Takeuchi
    * Abstract: We address the problem of estimating the difference between two probability densities. A naive approach is a two-step procedure of first estimating two densities separately and then computing their difference. However, such a two-step procedure does not necessarily work well because the first step is performed without regard to the second step and thus a small estimation error incurred in the first stage can cause a big error in the second stage. In this paper, we propose a single-shot procedure for directly estimating the density difference without separately estimating two densities. We derive a non-parametric finite-sample error bound for the proposed single-shot density-difference estimator and show that it achieves the optimal convergence rate. We then show how the proposed density-difference estimator can be utilized in L2-distance approximation. Finally, we experimentally demonstrate the usefulness of the proposed method in robust distribution comparison such as class-prior estimation and change-point detection.

count=2
* Large-Scale Damage Detection Using Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.pdf)]
    * Title: Large-Scale Damage Detection Using Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Lionel Gueguen, Raffay Hamid
    * Abstract: Satellite imagery is a valuable source of information for assessing damages in distressed areas undergoing a calamity, such as an earthquake or an armed conflict. However, the sheer amount of data required to be inspected for this assessment makes it impractical to do it manually. To address this problem, we present a semi-supervised learning framework for large-scale damage detection in satellite imagery. We present a comparative evaluation of our framework using over 88 million images collected from 4,665 square kilometers from 12 different locations around the world. To enable accurate and efficient damage detection, we introduce a novel use of hierarchical shape features in the bags-of-visual words setting. We analyze how practical factors such as sun, sensor-resolution, and satellite-angle differences impact the effectiveness of our proposed representation, and compare it to five alternative features in multiple learning settings. Finally, we demonstrate through a user-study that our semi-supervised framework results in a ten-fold reduction in human annotation time at a minimal loss in detection accuracy compared to an exhaustive manual inspection.

count=2
* Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Teutsch_Robust_Detection_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Teutsch_Robust_Detection_of_CVPR_2016_paper.pdf)]
    * Title: Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Michael Teutsch, Michael Grinberg
    * Abstract: Multiple object tracking in Wide Area Motion Imagery (WAMI) data is usually based on initial detections coming from background subtraction or frame differencing. However, these methods are prone to produce split and merged detections. Appearance based vehicle detection can be an alternative but is not well-suited for WAMI data since classifier models are of weak discriminative power for vehicles in top view at low resolution. We introduce a moving vehicle detection algorithm that combines 2-frame differencing with a vehicle appearance model to improve object detection. Our main contributions are (1) integration of robust vehicle detection with split/merge handling and (2) estimation of assignment likelihoods between object hypotheses in consecutive frames using an appearance based similarity measure. Without using any prior knowledge, we achieve state-of-the-art detection rates and produce tracklets that considerably simplify the data association problem for multiple object tracking.

count=2
* Learning without Exact Guidance: Updating Large-scale High-resolution Land Cover Maps from Low-resolution Historical Labels
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Learning_without_Exact_Guidance_Updating_Large-scale_High-resolution_Land_Cover_Maps_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Learning_without_Exact_Guidance_Updating_Large-scale_High-resolution_Land_Cover_Maps_CVPR_2024_paper.pdf)]
    * Title: Learning without Exact Guidance: Updating Large-scale High-resolution Land Cover Maps from Low-resolution Historical Labels
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zhuohong Li, Wei He, Jiepan Li, Fangxiao Lu, Hongyan Zhang
    * Abstract: Large-scale high-resolution (HR) land-cover mapping is a vital task to survey the Earth's surface and resolve many challenges facing humanity. However it is still a non-trivial task hindered by complex ground details various landforms and the scarcity of accurate training labels over a wide-span geographic area. In this paper we propose an efficient weakly supervised framework (Paraformer) to guide large-scale HR land-cover mapping with easy-access historical land-cover data of low resolution (LR). Specifically existing land-cover mapping approaches reveal the dominance of CNNs in preserving local ground details but still suffer from insufficient global modeling in various landforms. Therefore we design a parallel CNN-Transformer feature extractor in Paraformer consisting of a downsampling-free CNN branch and a Transformer branch to jointly capture local and global contextual information. Besides facing the spatial mismatch of training data a pseudo-label-assisted training (PLAT) module is adopted to reasonably refine LR labels for weakly supervised semantic segmentation of HR images. Experiments on two large-scale datasets demonstrate the superiority of Paraformer over other state-of-the-art methods for automatically updating HR land-cover maps from LR historical labels.

count=2
* CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_CorrMatch_Label_Propagation_via_Correlation_Matching_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_CorrMatch_Label_Propagation_via_Correlation_Matching_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf)]
    * Title: CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Boyuan Sun, Yuqi Yang, Le Zhang, Ming-Ming Cheng, Qibin Hou
    * Abstract: This paper presents a simple but performant semi-supervised semantic segmentation approach called CorrMatch. Previous approaches mostly employ complicated training strategies to leverage unlabeled data but overlook the role of correlation maps in modeling the relationships between pairs of locations. We observe that the correlation maps not only enable clustering pixels of the same category easily but also contain good shape information which previous works have omitted. Motivated by these we aim to improve the use efficiency of unlabeled data by designing two novel label propagation strategies. First we propose to conduct pixel propagation by modeling the pairwise similarities of pixels to spread the high-confidence pixels and dig out more. Then we perform region propagation to enhance the pseudo labels with accurate class-agnostic masks extracted from the correlation maps. CorrMatch achieves great performance on popular segmentation benchmarks. Taking the DeepLabV3+ with ResNet-101 backbone as our segmentation model we receive a 76%+ mIoU score on the Pascal VOC 2012 dataset with only 92 annotated images. Code is available at https://github.com/BBBBchan/CorrMatch .

count=2
* Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf)]
    * Title: Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Derrick Bonafilia, Beth Tellman, Tyler Anderson, Erica Issenberg
    * Abstract: Accurate flood mapping at global scales can support disaster relief and recovery efforts. Improving flood relief with more accurate data is of great importance due to expected increases in the frequency and magnitude of flood events with climatic and demographic changes. To assist efforts to operationalize deep learning algorithms for flood mapping at global scales, we introduce Sen1Floods11, a surface water data set including classified permanent water, flood water, and raw Sentinel-1 imagery. This dataset consists of 4,831 512x512 chips covering 120,406 km\textsuperscript 2 and spans all 14 biomes, 357 ecoregions, and 6 continents of the world across 11 flood events. We used Sen1Floods11 to train, validate, and test fully convolutional neural networks (FCNN) to segment permanent and flood water. We compare results of classifying permanent, flood, and total surface water from training four FCNN models: i) 446 hand labeled chips of surface water from flood events; ii) 814 chips of publicly available permanent water data labels from Landsat (JRC surface water dataset); iii) 4385 chips of surface water classified from Sentinel-2 images from flood events and iv) 4385 chips of surface water classified from Sentinel-1 imagery from flood events. We compare these four models to a common remote sensing approach of thresholding radar backscatter to identify surface water. Future research to operationalize computer vision approaches to mapping flood and surface water could build new models from Sen1Floods11 and expand this dataset to include additional sensors and flood events. We provide Sen1Floods11, as well as our training and evaluation code at: https://github.com/cloudtostreet/Sen1Floods11

count=2
* A Multi-sensor Fusion Framework in 3-D
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/html/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/papers/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.pdf)]
    * Title: A Multi-sensor Fusion Framework in 3-D
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Vishal Jain, Andrew C. Miller, Joseph L. Mundy
    * Abstract: The majority of existing image fusion techniques operate in the 2-d image domain which perform well for imagery of planar regions but fails in presence of any 3-d relief and provides inaccurate alignment of imagery from different sensors. A framework for multi-sensor image fusion in 3-d is proposed in this paper. The imagery from different sensors, specifically EO and IR, are fused in a common 3-d reference coordinate frame. A dense probabilistic and volumetric 3-d model is reconstructed from each of the sensors. The imagery is registered by aligning the 3-d models as the underlying 3-d structure in the images is the true invariant information. The image intensities are back-projected onto a 3-d model and every discretized location (voxel) of the 3-d model stores an array of intensities from different modalities. This 3-d model is forward-projected to produce a fused image of EO and IR from any viewpoint.

count=2
* Learning to Detect Carried Objects with Minimal Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/html/Dondera_Learning_to_Detect_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/papers/Dondera_Learning_to_Detect_2013_CVPR_paper.pdf)]
    * Title: Learning to Detect Carried Objects with Minimal Supervision
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Radu Dondera, Vlad Morariu, Larry Davis
    * Abstract: We propose a learning-based method for detecting carried objects that generates candidate image regions from protrusion, color contrast and occlusion boundary cues, and uses a classifier to filter out the regions unlikely to be carried objects. The method achieves higher accuracy than state of the art, which can only detect protrusions from the human shape, and the discriminative model it builds for the silhouette context-based region features generalizes well. To reduce annotation effort, we investigate training the model in a Multiple Instance Learning framework where the only available supervision is "walk" and "carry" labels associated with intervals of human tracks, i.e., the spatial extent of carried objects is not annotated. We present an extension to the miSVM algorithm that uses knowledge of the fraction of positive instances in positive bags and that scales to training sets of hundreds of thousands of instances.

count=2
* The Visual Object Tracking VOT2017 Challenge Results
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w28/html/Kristan_The_Visual_Object_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Kristan_The_Visual_Object_ICCV_2017_paper.pdf)]
    * Title: The Visual Object Tracking VOT2017 Challenge Results
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pflugfelder, Luka Cehovin Zajc, Tomas Vojir, Gustav Hager, Alan Lukezic, Abdelrahman Eldesokey, Gustavo Fernandez
    * Abstract: The Visual Object Tracking challenge VOT2017 is the fifth annual tracker benchmarking activity organized by the VOT initiative. Results of 51 trackers are presented; many are state-of-the-art published at major computer vision conferences or journals in recent years. The evaluation included the standard VOT and other popular methodologies and a new "real-time" experiment simulating a situation where a tracker processes images as if provided by a continuously running sensor. Performance of the tested trackers typically by far exceeds standard baselines. The source code for most of the trackers is publicly available from the VOT page. The VOT2017 goes beyond its predecessors by (i) improving the VOT public dataset and introducing a separate VOT2017 sequestered dataset, (ii) introducing a real-time tracking experiment and (iii) releasing a redesigned toolkit that supports complex experiments. The dataset, the evaluation kit and the results are publicly available at the challenge w ....

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
* M-Statistic for Kernel Change-Point Detection
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2015/hash/eb1e78328c46506b46a4ac4a1e378b91-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2015/file/eb1e78328c46506b46a4ac4a1e378b91-Paper.pdf)]
    * Title: M-Statistic for Kernel Change-Point Detection
    * Publisher: NeurIPS
    * Publication Date: `2015`
    * Authors: Shuang Li, Yao Xie, Hanjun Dai, Le Song
    * Abstract: Detecting the emergence of an abrupt change-point is a classic problem in statistics and machine learning. Kernel-based nonparametric statistics have been proposed for this task which make fewer assumptions on the distributions than traditional parametric approach. However, none of the existing kernel statistics has provided a computationally efficient way to characterize the extremal behavior of the statistic. Such characterization is crucial for setting the detection threshold, to control the significance level in the offline case as well as the average run length in the online case. In this paper we propose two related computationally efficient M-statistics for kernel-based change-point detection when the amount of background data is large. A novel theoretical result of the paper is the characterization of the tail probability of these statistics using a new technique based on change-of-measure. Such characterization provides us accurate detection thresholds for both offline and online cases in computationally efficient manner, without the need to resort to the more expensive simulations such as bootstrapping. We show that our methods perform well in both synthetic and real world data.

count=1
* Protecting Against Screenshots: An Image Processing Approach
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Chia_Protecting_Against_Screenshots_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Chia_Protecting_Against_Screenshots_2015_CVPR_paper.pdf)]
    * Title: Protecting Against Screenshots: An Image Processing Approach
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Alex Yong-Sang Chia, Udana Bandara, Xiangyu Wang, Hiromi Hirano
    * Abstract: Motivated by reasons related to data security and privacy, we propose a method to limit meaningful visual contents of a display from being captured by screenshots. Traditional methods take a system architectural approach to protect against screenshots. We depart from this framework, and instead exploit image processing techniques to distort visual data of a display and present the distorted data to the viewer. Given that a screenshot captures distorted visual contents, it yields limited useful data. We exploit the human visual system to empower viewers to automatically and mentally recover the distorted contents into a meaningful form in real-time. Towards this end, we leverage on findings from psychological studies which show that blending of visual information from recent and current fixations enables human to form meaningful representation of a scene. We model this blending of information by an additive process, and exploit this to design a visual contents distortion algorithm that supports real-time contents recovery by the human visual system. Our experiments and user study demonstrate the feasibility of our method to allow viewers to readily interpret visual contents of a display, while limiting meaningful contents from being captured by screenshots.

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
* Rare Event Detection Using Disentangled Representation Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Hamaguchi_Rare_Event_Detection_Using_Disentangled_Representation_Learning_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hamaguchi_Rare_Event_Detection_Using_Disentangled_Representation_Learning_CVPR_2019_paper.pdf)]
    * Title: Rare Event Detection Using Disentangled Representation Learning
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Ryuhei Hamaguchi,  Ken Sakurada,  Ryosuke Nakamura
    * Abstract: This paper presents a novel method for rare event detection from an image pair with class-imbalanced datasets. A straightforward approach for event detection tasks is to train a detection network from a large-scale dataset in an end-to-end manner. However, in many applications such as building change detection on satellite images, few positive samples are available for the training. Moreover, an image pair of scenes contains many trivial events, such as in illumination changes or background motions. These many trivial events and the class imbalance problem lead to false alarms for rare event detection. In order to overcome these difficulties, we propose a novel method to learn disentangled representations from only low-cost negative samples. The proposed method disentangles the different aspects in a pair of observations: variant and invariant factors that represent trivial events and image contents, respectively. The effectiveness of the proposed approach is verified by the quantitative evaluations on four change detection datasets, and the qualitative analysis shows that the proposed method can acquire the representations that disentangle rare events from trivial ones.

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
* Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.pdf)]
    * Title: Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Peri Akiva, Matthew Purri, Matthew Leotta
    * Abstract: Self-supervised learning aims to learn image feature representations without the usage of manually annotated labels. It is often used as a precursor step to obtain useful initial network weights which contribute to faster convergence and superior performance of downstream tasks. While self-supervision allows one to reduce the domain gap between supervised and unsupervised learning without the usage of labels, the self-supervised objective still requires a strong inductive bias to downstream tasks for effective transfer learning. In this work, we present our material and texture based self-supervision method named MATTER (MATerial and TExture Representation Learning), which is inspired by classical material and texture methods. Material and texture can effectively describe any surface, including its tactile properties, color, and specularity. By extension, effective representation of material and texture can describe other semantic classes strongly associated with said material and texture. MATTER leverages multi-temporal, spatially aligned remote sensing imagery over unchanged regions to learn invariance to illumination and viewing angle as a mechanism to achieve consistency of material and texture representation. We show that our self-supervision pre-training method allows for up to 24.22% and 6.33% performance increase in unsupervised and fine-tuned setups, and up to 76% faster convergence on change detection, land cover classification, and semantic segmentation tasks.

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
* Progressive Unsupervised Deep Transfer Learning for Forest Mapping in Satellite Image
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/html/Ahmed_Progressive_Unsupervised_Deep_Transfer_Learning_for_Forest_Mapping_in_Satellite_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/papers/Ahmed_Progressive_Unsupervised_Deep_Transfer_Learning_for_Forest_Mapping_in_Satellite_ICCVW_2021_paper.pdf)]
    * Title: Progressive Unsupervised Deep Transfer Learning for Forest Mapping in Satellite Image
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Nouman Ahmed, Sudipan Saha, Muhammad Shahzad, Muhammad Moazam Fraz, Xiao Xiang Zhu
    * Abstract: Automated forest mapping is important to understand our forests that play a key role in ecological system. However, efforts towards forest mapping is impeded by difficulty to collect labeled forest images that show large intraclass variation. Recently unsupervised learning has shown promising capability when exploiting limited labeled data. Motivated by this, we propose a progressive unsupervised deep transfer learning method for forest mapping. The proposed method exploits a pre-trained model that is subsequently fine-tuned over the target forest domain. We propose two different fine-tuning mechanism, one works in a totally unsupervised setting by jointly learning the parameters of CNN and the k-means based cluster assignments of the resulting features and the other one works in a semi-supervised setting by exploiting the extracted knearest neighbor based pseudo labels. The proposed progressive scheme is evaluated on publicly available EuroSAT dataset using the relevant base model trained on BigEarthNet labels. The results show that the proposed method greatly improves the forest regions classification accuracy as compared to the unsupervised baseline, nearly approaching the supervised classification approach.

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
* Learning High-Density Regions for a Generalized Kolmogorov-Smirnov Test in High-Dimensional Data
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/6855456e2fe46a9d49d3d3af4f57443d-Paper.pdf)]
    * Title: Learning High-Density Regions for a Generalized Kolmogorov-Smirnov Test in High-Dimensional Data
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Assaf Glazer, Michael Lindenbaum, Shaul Markovitch
    * Abstract: We propose an efficient, generalized, nonparametric, statistical Kolmogorov-Smirnov test for detecting distributional change in high-dimensional data. To implement the test, we introduce a novel, hierarchical, minimum-volume sets estimator to represent the distributions to be tested. Our work is motivated by the need to detect changes in data streams, and the test is especially efficient in this context. We provide the theoretical foundations of our test and show its superiority over existing methods.

