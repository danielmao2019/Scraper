count=10
* Keep it Accurate and Diverse: Enhancing Action Recognition Performance by Ensemble Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W09/html/Bagheri_Keep_it_Accurate_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W09/papers/Bagheri_Keep_it_Accurate_2015_CVPR_paper.pdf)]
    * Title: Keep it Accurate and Diverse: Enhancing Action Recognition Performance by Ensemble Learning
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Mohammad Bagheri, Qigang Gao, Sergio Escalera, Albert Clapes, Kamal Nasrollahi, Michael B. Holte, Thomas B. Moeslund
    * Abstract: The performance of different action recognition techniques has recently been studied by several computer vision researchers. However, the potential improvement in classification through classifier fusion by ensemble-based methods has remained unattended. In this work, we evaluate the performance of an ensemble of action learning techniques, each performing the recognition task from a different perspective. The underlying idea is that instead of aiming a very sophisticated and powerful representation/learning technique, we can learn action categories using a set of relatively simple and diverse classifiers, each trained with different feature set. In addition, combining the outputs of several learners can reduce the risk of an unfortunate selection of a learner on an unseen action recognition scenario. This leads to having a more robust and general-applicable framework. In order to improve the recognition performance, a powerful combination strategy is utilized based on the Dempster-Shafer theory, which can effectively make use of diversity of base learners trained on different sources of information. The recognition results of the individual classifiers are compared with those obtained from fusing the classifiers' output, showing enhanced performance of the proposed methodology.

count=8
* Trustworthy Long-Tailed Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Trustworthy_Long-Tailed_Classification_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Trustworthy_Long-Tailed_Classification_CVPR_2022_paper.pdf)]
    * Title: Trustworthy Long-Tailed Classification
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Bolian Li, Zongbo Han, Haining Li, Huazhu Fu, Changqing Zhang
    * Abstract: Classification on long-tailed distributed data is a challenging problem, which suffers from serious class-imbalance and accordingly unpromising performance especially on tail classes. Recently, the ensembling based methods achieve the state-of-the-art performance and show great potential. However, there are two limitations for current methods. First, their predictions are not trustworthy for failure-sensitive applications. This is especially harmful for the tail classes where the wrong predictions is basically frequent. Second, they assign unified numbers of experts to all samples, which is redundant for easy samples with excessive computational cost. To address these issues, we propose a Trustworthy Long-tailed Classification (TLC) method to jointly conduct classification and uncertainty estimation to identify hard samples in a multi-expert framework. Our TLC obtains the evidence-based uncertainty (EvU) and evidence for each expert, and then combines these uncertainties and evidences under the Dempster-Shafer Evidence Theory (DST). Moreover, we propose a dynamic expert engagement to reduce the number of engaged experts for easy samples and achieve efficiency while maintaining promising performances. Finally, we conduct comprehensive experiments on the tasks of classification, tail detection, OOD detection and failure prediction. The experimental results show that the proposed TLC outperforms existing methods and is trustworthy with reliable uncertainty.

count=7
* Causal Property based Anti-Conflict Modeling with Hybrid Data Augmentation for Unbiased Scene Graph Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2022/html/Zhang_Causal_Property_based_Anti-Conflict_Modeling_with_Hybrid_Data_Augmentation_for_ACCV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2022/papers/Zhang_Causal_Property_based_Anti-Conflict_Modeling_with_Hybrid_Data_Augmentation_for_ACCV_2022_paper.pdf)]
    * Title: Causal Property based Anti-Conflict Modeling with Hybrid Data Augmentation for Unbiased Scene Graph Generation
    * Publisher: ACCV
    * Publication Date: `2022`
    * Authors: Ruonan Zhang, Gaoyun An
    * Abstract: Scene Graph Generation(SGG) aims to detect visual triplets of pairwise objects based on object detection. There are three key factors being explored to determine a scene graph: visual information, local and global context, and prior knowledge. However, conventional methods balancing losses among these factors lead to conflict, causing ambiguity, inaccuracy, and inconsistency. In this work, to apply evidence theory to scene graph generation, a novel plug-and-play Causal Property based Anti-conflict Modeling (CPAM) module is proposed, which models key factors by Dempster-Shafer evidence theory, and integrates quantitative information effectively. Compared with the existing methods, the proposed CPAM makes the training process interpretable, and also manages to cover more fine-grained relationships after inconsistencies reduction. Furthermore, we propose a Hybrid Data Augmentation (HDA) method, which facilitates data transfer as well as conventional debiasing methods to enhance the dataset. By combining CPAM with HDA, significant improvement has been achieved over the previous state-of-the-art methods. And extensive ablation studies have also been conducted to demonstrate the effectiveness of our method.

count=7
* Evidence Based Feature Selection and Collaborative Representation Towards Learning Based PSF Estimation for Motion Deblurring
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/RLQ/Dhanakshirur_Evidence_Based_Feature_Selection_and_Collaborative_Representation_Towards_Learning_Based_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/RLQ/Dhanakshirur_Evidence_Based_Feature_Selection_and_Collaborative_Representation_Towards_Learning_Based_ICCVW_2019_paper.pdf)]
    * Title: Evidence Based Feature Selection and Collaborative Representation Towards Learning Based PSF Estimation for Motion Deblurring
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Rohan Raju Dhanakshirur, Ramesh Ashok Tabib, Ujwala Patil, Uma Mudenagudi
    * Abstract: The motion blur in an image is due to the relative motion between the camera and the scene being captured. Due to the degraded quality of the motion-blurred images, it is challenging to use them in different applications such as text detection, scene understanding, content-based image retrieval, etc. Typically, a motion-blurred image is modeled as a convolution between the un-blurred image and a Point Spread Function (PSF). Motion de-blurring is sensitive to the estimated PSF. In this paper, we propose to address the problem of motion deblurring by estimating PSF using a learning-based approach. We model motion blur as a function of length and angle and propose to estimate these parameters using a learning-based framework. It is challenging to find distinct features to precisely learn the extent of motion blur through deep learning. To address this, we model an evidence-based technique to select the relevant features for learning from a set of features, based on the confidence generated by combining the evidences using Dempster Shafer Combination Rule (DSCR). We propose to use Clustering and Collaborative Representation (CCR) of feature spaces to learn length and angle. We model the deblurred image as an MRF (Markov Random Field) and use MAP (maximum a posteriori) estimate as the final solution. We demonstrate the results on real and synthetic datasets and compare the results with different state of art methods using various quality metrics and vision tools.

count=5
* One-Class Multiple-Look Fusion: A Theoretical Comparison of Different Approaches with Examples from Infrared Video
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/html/Koch_One-Class_Multiple-Look_Fusion_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/papers/Koch_One-Class_Multiple-Look_Fusion_2013_CVPR_paper.pdf)]
    * Title: One-Class Multiple-Look Fusion: A Theoretical Comparison of Different Approaches with Examples from Infrared Video
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Mark W. Koch
    * Abstract: Multiple-look fusion is quickly becoming more important in statistical pattern recognition. With increased computing power and memory one can make many measurements on an object of interest using, for example, video imagery or radar. By obtaining more views of an object, a system can make decisions with lower missed detection and false alarm errors. There are many approaches for combining information from multiple looks and we mathematically compare and contrast the sequential probability ratio test, Bayesian fusion, and Dempster-Shafer theory of evidence. Using a consistent probabilistic framework we demonstrate the differences and similarities between the approaches and show results for an application in infrared video classification.

count=5
* A Modified Sequential Monte Carlo Bayesian Occupancy Filter Using Linear Opinion Pool for Grid Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w9/html/Oh_A_Modified_Sequential_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w9/papers/Oh_A_Modified_Sequential_ICCV_2015_paper.pdf)]
    * Title: A Modified Sequential Monte Carlo Bayesian Occupancy Filter Using Linear Opinion Pool for Grid Mapping
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Sang-Il Oh, Hang-Bong Kang
    * Abstract: Occupancy grid state mapping is a key process in robotics and autonomous driving systems. It divides the environment into grid cells that contain information states. In this paper, we propose a modified SMC-BOF method to map and predict occupancy grids. The original SMC-BOF has been widely used in the occupancy grid mapping because it has lower computational costs than the BOF method. However, there are some issues related to conflicting information in dynamic situations. The original SMC-BOF cannot completely control an elongated vehicle that has conflicting information caused by the height difference between backward of vehicle and ground. To overcome this problem, we add confidence weights onto a part of the grid mapping process of the original SMC-BOF using the Linear Opinion Pool. We evaluate our method by LIDAR and stereo vision data in the KITTI dataset.

count=4
* Evidential Active Recognition: Intelligent and Prudent Open-World Embodied Perception
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Evidential_Active_Recognition_Intelligent_and_Prudent_Open-World_Embodied_Perception_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Evidential_Active_Recognition_Intelligent_and_Prudent_Open-World_Embodied_Perception_CVPR_2024_paper.pdf)]
    * Title: Evidential Active Recognition: Intelligent and Prudent Open-World Embodied Perception
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Lei Fan, Mingfu Liang, Yunxuan Li, Gang Hua, Ying Wu
    * Abstract: Active recognition enables robots to intelligently explore novel observations thereby acquiring more information while circumventing undesired viewing conditions. Recent approaches favor learning policies from simulated or collected data wherein appropriate actions are more frequently selected when the recognition is accurate. However most recognition modules are developed under the closed-world assumption which makes them ill-equipped to handle unexpected inputs such as the absence of the target object in the current observation. To address this issue we propose treating active recognition as a sequential evidence-gathering process providing by-step uncertainty quantification and reliable prediction under the evidence combination theory. Additionally the reward function developed in this paper effectively characterizes the merit of actions when operating in open-world environments. To evaluate the performance we collect a dataset from an indoor simulator encompassing various recognition challenges such as distance occlusion levels and visibility. Through a series of experiments on recognition and robustness analysis we demonstrate the necessity of introducing uncertainties to active recognition and the superior performance of the proposed method.

count=4
* Evidential Sparsification of Multimodal Latent Spaces in Conditional Variational Autoencoders
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/73f95ee473881dea4afd89c06165fa66-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/73f95ee473881dea4afd89c06165fa66-Paper.pdf)]
    * Title: Evidential Sparsification of Multimodal Latent Spaces in Conditional Variational Autoencoders
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Masha Itkina, Boris Ivanovic, Ransalu Senanayake, Mykel J. Kochenderfer, Marco Pavone
    * Abstract: Discrete latent spaces in variational autoencoders have been shown to effectively capture the data distribution for many real-world problems such as natural language understanding, human intent prediction, and visual scene representation. However, discrete latent spaces need to be sufficiently large to capture the complexities of real-world data, rendering downstream tasks computationally challenging. For instance, performing motion planning in a high-dimensional latent representation of the environment could be intractable. We consider the problem of sparsifying the discrete latent space of a trained conditional variational autoencoder, while preserving its learned multimodality. As a post hoc latent space reduction technique, we use evidential theory to identify the latent classes that receive direct evidence from a particular input condition and filter out those that do not. Experiments on diverse tasks, such as image generation and human behavior prediction, demonstrate the effectiveness of our proposed technique at reducing the discrete latent sample space size of a model while maintaining its learned multimodality.

count=3
* Switchable Representation Learning Framework With Self-Compatibility
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Switchable_Representation_Learning_Framework_With_Self-Compatibility_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Switchable_Representation_Learning_Framework_With_Self-Compatibility_CVPR_2023_paper.pdf)]
    * Title: Switchable Representation Learning Framework With Self-Compatibility
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Shengsen Wu, Yan Bai, Yihang Lou, Xiongkun Linghu, Jianzhong He, Ling-Yu Duan
    * Abstract: Real-world visual search systems involve deployments on multiple platforms with different computing and storage resources. Deploying a unified model that suits the minimal-constrain platforms leads to limited accuracy. It is expected to deploy models with different capacities adapting to the resource constraints, which requires features extracted by these models to be aligned in the metric space. The method to achieve feature alignments is called "compatible learning". Existing research mainly focuses on the one-to-one compatible paradigm, which is limited in learning compatibility among multiple models. We propose a Switchable representation learning Framework with Self-Compatibility (SFSC). SFSC generates a series of compatible sub-models with different capacities through one training process. The optimization of sub-models faces gradients conflict, and we mitigate this problem from the perspective of the magnitude and direction. We adjust the priorities of sub-models dynamically through uncertainty estimation to co-optimize sub-models properly. Besides, the gradients with conflicting directions are projected to avoid mutual interference. SFSC achieves state-of-the-art performance on the evaluated datasets.

count=3
* Flexible Visual Recognition by Evidential Modeling of Confusion and Ignorance
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Fan_Flexible_Visual_Recognition_by_Evidential_Modeling_of_Confusion_and_Ignorance_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Fan_Flexible_Visual_Recognition_by_Evidential_Modeling_of_Confusion_and_Ignorance_ICCV_2023_paper.pdf)]
    * Title: Flexible Visual Recognition by Evidential Modeling of Confusion and Ignorance
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Lei Fan, Bo Liu, Haoxiang Li, Ying Wu, Gang Hua
    * Abstract: In real-world scenarios, typical visual recognition systems could fail under two major causes, i.e., the misclassification between known classes and the excusable misbehavior on unknown-class images. To tackle these deficiencies, flexible visual recognition should dynamically predict multiple classes when they are unconfident between choices and reject making predictions when the input is entirely out of the training distribution. Two challenges emerge along with this novel task. First, prediction uncertainty should be separately quantified as confusion depicting inter-class uncertainties and ignorance identifying out-of-distribution samples. Second, both confusion and ignorance should be comparable between samples to enable effective decision-making. In this paper, we propose to model these two sources of uncertainty explicitly with the theory of Subjective Logic. Regarding recognition as an evidence-collecting process, confusion is then defined as conflicting evidence, while ignorance is the absence of evidence. By predicting Dirichlet concentration parameters for singletons, comprehensive subjective opinions, including confusion and ignorance, could be achieved via further evidence combinations. Through a series of experiments on synthetic data analysis, visual recognition, and open-set detection, we demonstrate the effectiveness of our methods in quantifying two sources of uncertainties and dealing with flexible recognition.

count=3
* Prototype-based Aleatoric Uncertainty Quantification for Cross-modal Retrieval
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/4d893f766ab60e5337659b9e71883af4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/4d893f766ab60e5337659b9e71883af4-Paper-Conference.pdf)]
    * Title: Prototype-based Aleatoric Uncertainty Quantification for Cross-modal Retrieval
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Hao Li, Jingkuan Song, Lianli Gao, Xiaosu Zhu, Hengtao Shen
    * Abstract: Cross-modal Retrieval methods build similarity relations between vision and language modalities by jointly learning a common representation space. However, the predictions are often unreliable due to the Aleatoric uncertainty, which is induced by low-quality data, e.g., corrupt images, fast-paced videos, and non-detailed texts. In this paper, we propose a novel Prototype-based Aleatoric Uncertainty Quantification (PAU) framework to provide trustworthy predictions by quantifying the uncertainty arisen from the inherent data ambiguity. Concretely, we first construct a set of various learnable prototypes for each modality to represent the entire semantics subspace. Then Dempster-Shafer Theory and Subjective Logic Theory are utilized to build an evidential theoretical framework by associating evidence with Dirichlet Distribution parameters. The PAU model induces accurate uncertainty and reliable predictions for cross-modal retrieval. Extensive experiments are performed on four major benchmark datasets of MSR-VTT, MSVD, DiDeMo, and MS-COCO, demonstrating the effectiveness of our method. The code is accessible at https://github.com/leolee99/PAU.

count=2
* Shape from Silhouette Probability Maps: Reconstruction of Thin Objects in the Presence of Silhouette Extraction and Calibration Error
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Tabb_Shape_from_Silhouette_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Tabb_Shape_from_Silhouette_2013_CVPR_paper.pdf)]
    * Title: Shape from Silhouette Probability Maps: Reconstruction of Thin Objects in the Presence of Silhouette Extraction and Calibration Error
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Amy Tabb
    * Abstract: This paper considers the problem of reconstructing the shape of thin, texture-less objects such as leafless trees when there is noise or deterministic error in the silhouette extraction step or there are small errors in camera calibration. Traditional intersection-based techniques such as the visual hull are not robust to error because they penalize false negative and false positive error unequally. We provide a voxel-based formalism that penalizes false negative and positive error equally, by casting the reconstruction problem as a pseudo-Boolean minimization problem, where voxels are the variables of a pseudo-Boolean function and are labeled occupied or empty. Since the pseudo-Boolean minimization problem is NP-Hard for nonsubmodular functions, we developed an algorithm for an approximate solution using local minimum search. Our algorithm treats input binary probability maps (in other words, silhouettes) or continuously-valued probability maps identically, and places no constraints on camera placement. The algorithm was tested on three different leafless trees and one metal object where the number of voxels is 54.4 million (voxel sides measure 3.6 mm). Results show that our approach reconstructs the complicated branching structure of thin, texture-less objects in the presence of error where intersection-based approaches currently fail. 1

count=2
* Locality-Sensitive Deconvolution Networks With Gated Fusion for RGB-D Indoor Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Cheng_Locality-Sensitive_Deconvolution_Networks_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cheng_Locality-Sensitive_Deconvolution_Networks_CVPR_2017_paper.pdf)]
    * Title: Locality-Sensitive Deconvolution Networks With Gated Fusion for RGB-D Indoor Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yanhua Cheng, Rui Cai, Zhiwei Li, Xin Zhao, Kaiqi Huang
    * Abstract: This paper focuses on indoor semantic segmentation using RGB-D data. Although the commonly used deconvolution networks (DeconvNet) have achieved impressive results on this task, we find there is still room for improvements in two aspects. One is about the boundary segmentation. DeconvNet aggregates large context to predict the label of each pixel, inherently limiting the segmentation precision of object boundaries. The other is about RGB-D fusion. Recent state-of-the-art methods generally fuse RGB and depth networks with equal-weight score fusion, regardless of the varying contributions of the two modalities on delineating different categories in different scenes. To address the two problems, we first propose a locality-sensitive DeconvNet (LS-DeconvNet) to refine the boundary segmentation over each modality. LS-DeconvNet incorporates locally visual and geometric cues from the raw RGB-D data into each DeconvNet, which is able to learn to upsample the coarse convolutional maps with large context whilst recovering sharp object boundaries. Towards RGB-D fusion, we introduce a gated fusion layer to effectively combine the two LS-DeconvNets. This layer can learn to adjust the contributions of RGB and depth over each pixel for high-performance object recognition. Experiments on the large-scale SUN RGB-D dataset and the popular NYU-Depth v2 dataset show that our approach achieves new state-of-the-art results for RGB-D indoor semantic segmentation.

count=2
* Tampering Detection and Localization Through Clustering of Camera-Based CNN Features
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w28/html/Tubaro_Tampering_Detection_and_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w28/papers/Tubaro_Tampering_Detection_and_CVPR_2017_paper.pdf)]
    * Title: Tampering Detection and Localization Through Clustering of Camera-Based CNN Features
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Luca Bondi
    * Abstract: Due to the rapid proliferation of image capturing devices and user-friendly editing software suites, image manipulation is at everyone's hand. For this reason, the forensic community has developed a series of techniques to determine image authenticity. In this paper, we propose an algorithm for image tampering detection and localization, leveraging characteristic footprints left on images by different camera models. The rationale behind our algorithm is that all pixels of pristine images should be detected as being shot with a single device. Conversely, if a picture is obtained through image composition, traces of multiple devices can be detected. The proposed algorithm exploits a convolutional neural network (CNN) to extract characteristic camera model features from image patches. These features are then analyzed by means of iterative clustering techniques in order to detect whether an image has been forged, and localize the alien region.

count=2
* Cascade Evidential Learning for Open-World Weakly-Supervised Temporal Action Localization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Cascade_Evidential_Learning_for_Open-World_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Cascade_Evidential_Learning_for_Open-World_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.pdf)]
    * Title: Cascade Evidential Learning for Open-World Weakly-Supervised Temporal Action Localization
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Mengyuan Chen, Junyu Gao, Changsheng Xu
    * Abstract: Targeting at recognizing and localizing action instances with only video-level labels during training, Weakly-supervised Temporal Action Localization (WTAL) has achieved significant progress in recent years. However, living in the dynamically changing open world where unknown actions constantly spring up, the closed-set assumption of existing WTAL methods is invalid. Compared with traditional open-set recognition tasks, Open-world WTAL (OWTAL) is challenging since not only are the annotations of unknown samples unavailable, but also the fine-grained annotations of known action instances can only be inferred ambiguously from the video category labels. To address this problem, we propose a Cascade Evidential Learning framework at an evidence level, which targets at OWTAL for the first time. Our method jointly leverages multi-scale temporal contexts and knowledge-guided prototype information to progressively collect cascade and enhanced evidence for known action, unknown action, and background separation. Extensive experiments conducted on THUMOS-14 and ActivityNet-v1.3 verify the effectiveness of our method. Besides the classification metrics adopted by previous open-set recognition methods, we also evaluate our method on localization metrics which are more reasonable for OWTAL.

count=2
* Collecting Cross-Modal Presence-Absence Evidence for Weakly-Supervised Audio-Visual Event Perception
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Gao_Collecting_Cross-Modal_Presence-Absence_Evidence_for_Weakly-Supervised_Audio-Visual_Event_Perception_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_Collecting_Cross-Modal_Presence-Absence_Evidence_for_Weakly-Supervised_Audio-Visual_Event_Perception_CVPR_2023_paper.pdf)]
    * Title: Collecting Cross-Modal Presence-Absence Evidence for Weakly-Supervised Audio-Visual Event Perception
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Junyu Gao, Mengyuan Chen, Changsheng Xu
    * Abstract: With only video-level event labels, this paper targets at the task of weakly-supervised audio-visual event perception (WS-AVEP), which aims to temporally localize and categorize events belonging to each modality. Despite the recent progress, most existing approaches either ignore the unsynchronized property of audio-visual tracks or discount the complementary modality for explicit enhancement. We argue that, for an event residing in one modality, the modality itself should provide ample presence evidence of this event, while the other complementary modality is encouraged to afford the absence evidence as a reference signal. To this end, we propose to collect Cross-Modal Presence-Absence Evidence (CMPAE) in a unified framework. Specifically, by leveraging uni-modal and cross-modal representations, a presence-absence evidence collector (PAEC) is designed under Subjective Logic theory. To learn the evidence in a reliable range, we propose a joint-modal mutual learning (JML) process, which calibrates the evidence of diverse audible, visible, and audi-visible events adaptively and dynamically. Extensive experiments show that our method surpasses state-of-the-arts (e.g., absolute gains of 3.6% and 6.1% in terms of event-level visual and audio metrics). Code is available in github.com/MengyuanChen21/CVPR2023-CMPAE.

count=2
* Reliable and Interpretable Personalized Federated Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Qin_Reliable_and_Interpretable_Personalized_Federated_Learning_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Qin_Reliable_and_Interpretable_Personalized_Federated_Learning_CVPR_2023_paper.pdf)]
    * Title: Reliable and Interpretable Personalized Federated Learning
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zixuan Qin, Liu Yang, Qilong Wang, Yahong Han, Qinghua Hu
    * Abstract: Federated learning can coordinate multiple users to participate in data training while ensuring data privacy. The collaboration of multiple agents allows for a natural connection between federated learning and collective intelligence. When there are large differences in data distribution among clients, it is crucial for federated learning to design a reliable client selection strategy and an interpretable client communication framework to better utilize group knowledge. Herein, a reliable personalized federated learning approach, termed RIPFL, is proposed and fully interpreted from the perspective of social learning. RIPFL reliably selects and divides the clients involved in training such that each client can use different amounts of social information and more effectively communicate with other clients. Simultaneously, the method effectively integrates personal information with the social information generated by the global model from the perspective of Bayesian decision rules and evidence theory, enabling individuals to grow better with the help of collective wisdom. An interpretable federated learning mind is well scalable, and the experimental results indicate that the proposed method has superior robustness and accuracy than other state-of-the-art federated learning algorithms.

count=2
* Exploring and Exploiting Uncertainty for Incomplete Multi-View Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_Exploring_and_Exploiting_Uncertainty_for_Incomplete_Multi-View_Classification_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_Exploring_and_Exploiting_Uncertainty_for_Incomplete_Multi-View_Classification_CVPR_2023_paper.pdf)]
    * Title: Exploring and Exploiting Uncertainty for Incomplete Multi-View Classification
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Mengyao Xie, Zongbo Han, Changqing Zhang, Yichen Bai, Qinghua Hu
    * Abstract: Classifying incomplete multi-view data is inevitable since arbitrary view missing widely exists in real-world applications. Although great progress has been achieved, existing incomplete multi-view methods are still difficult to obtain a trustworthy prediction due to the relatively high uncertainty nature of missing views. First, the missing view is of high uncertainty, and thus it is not reasonable to provide a single deterministic imputation. Second, the quality of the imputed data itself is of high uncertainty. To explore and exploit the uncertainty, we propose an Uncertainty-induced Incomplete Multi-View Data Classification (UIMC) model to classify the incomplete multi-view data under a stable and reliable framework. We construct a distribution and sample multiple times to characterize the uncertainty of missing views, and adaptively utilize them according to the sampling quality. Accordingly, the proposed method realizes more perceivable imputation and controllable fusion. Specifically, we model each missing data with a distribution conditioning on the available views and thus introducing uncertainty. Then an evidence-based fusion strategy is employed to guarantee the trustworthy integration of the imputed views. Extensive experiments are conducted on multiple benchmark data sets and our method establishes a state-of-the-art performance in terms of both performance and trustworthiness.

count=2
* Interpretable Model-Agnostic Plausibility Verification for 2D Object Detectors Using Domain-Invariant Concept Bottleneck Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/html/Keser_Interpretable_Model-Agnostic_Plausibility_Verification_for_2D_Object_Detectors_Using_Domain-Invariant_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/papers/Keser_Interpretable_Model-Agnostic_Plausibility_Verification_for_2D_Object_Detectors_Using_Domain-Invariant_CVPRW_2023_paper.pdf)]
    * Title: Interpretable Model-Agnostic Plausibility Verification for 2D Object Detectors Using Domain-Invariant Concept Bottleneck Models
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Mert Keser, Gesina Schwalbe, Azarm Nowzad, Alois Knoll
    * Abstract: Despite the unchallenged performance, deep neural network (DNN) based object detectors (OD) for computer vision have inherent, hard-to-verify limitations like brittleness, opacity, and unknown behavior on corner cases. Therefore, operation-time safety measures like monitors will be inevitable--even mandatory--for use in safetycritical applications like automated driving (AD). This paper presents an approach for plausibilization of OD detections using a small model-agnostic, robust, interpretable, and domain-invariant image classification model. The safety requirements of interpretability and robustness are achieved by using a small concept bottleneck model (CBM), a DNN intercepted by interpretable intermediate outputs. The domain-invariance is necessary for robustness against common domain shifts, and for cheap adaptation to diverse AD settings. While vanilla CBMs are here shown to fail in case of domain shifts like natural perturbations, we substantially improve the CBM via combination with trainable color-invariance filters developed for domain adaptation. Furthermore, the monitor that utilizes CBMs with trainable color-invarince filters is successfully applied in an AD OD setting for detection of hallucinated objects with zero-shot domain adaptation, and to false positive detection with few-shot adaptation, proving this to be a promising approach for error monitoring.

count=2
* Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Think_Twice_Before_Selection_Federated_Evidential_Active_Learning_for_Medical_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Think_Twice_Before_Selection_Federated_Evidential_Active_Learning_for_Medical_CVPR_2024_paper.pdf)]
    * Title: Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jiayi Chen, Benteng Ma, Hengfei Cui, Yong Xia
    * Abstract: Federated learning facilitates the collaborative learning of a global model across multiple distributed medical institutions without centralizing data. Nevertheless the expensive cost of annotation on local clients remains an obstacle to effectively utilizing local data. To mitigate this issue federated active learning methods suggest leveraging local and global model predictions to select a relatively small amount of informative local data for annotation. However existing methods mainly focus on all local data sampled from the same domain making them unreliable in realistic medical scenarios with domain shifts among different clients. In this paper we make the first attempt to assess the informativeness of local data derived from diverse domains and propose a novel methodology termed Federated Evidential Active Learning (FEAL) to calibrate the data evaluation under domain shift. Specifically we introduce a Dirichlet prior distribution in both local and global models to treat the prediction as a distribution over the probability simplex and capture both aleatoric and epistemic uncertainties by using the Dirichlet-based evidential model. Then we employ the epistemic uncertainty to calibrate the aleatoric uncertainty. Afterward we design a diversity relaxation strategy to reduce data redundancy and maintain data diversity. Extensive experiments and analysis on five real multi-center medical image datasets demonstrate the superiority of FEAL over the state-of-the-art active learning methods in federated scenarios with domain shifts. The code will be available at https://github.com/JiayiChen815/FEAL.

count=2
* Fusion Transformer with Object Mask Guidance for Image Forgery Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/WMF/html/Karageorgiou_Fusion_Transformer_with_Object_Mask_Guidance_for_Image_Forgery_Analysis_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Karageorgiou_Fusion_Transformer_with_Object_Mask_Guidance_for_Image_Forgery_Analysis_CVPRW_2024_paper.pdf)]
    * Title: Fusion Transformer with Object Mask Guidance for Image Forgery Analysis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Dimitrios Karageorgiou, Giorgos Kordopatis-Zilos, Symeon Papadopoulos
    * Abstract: In this work we introduce OMG-Fuser a fusion transformer-based network designed to extract information from various forensic signals to enable robust image forgery detection and localization. Our approach can operate with an arbitrary number of forensic signals and leverages object information for their analysis -- unlike previous methods that rely on fusion schemes with few signals and often disregard image semantics. To this end we design a forensic signal stream composed of a transformer guided by an object attention mechanism associating patches that depict the same objects. In that way we incorporate object-level information from the image. Each forensic signal is processed by a different stream that adapts to its peculiarities. A token fusion transformer efficiently aggregates the outputs of an arbitrary number of network streams and generates a fused representation for each image patch. % These representations are finally processed by a long-range dependencies transformer that captures the intrinsic relations between the image patches. We assess two fusion variants on top of the proposed approach: (i) score-level fusion that fuses the outputs of multiple image forensics algorithms and (ii) feature-level fusion that fuses low-level forensic traces directly. Both variants exceed state-of-the-art performance on seven datasets for image forgery detection and localization with a relative average improvement of 12.1% and 20.4% in terms of F1. Our model is robust against traditional and novel forgery attacks and can be expanded with new signals without training from scratch. Our code is publicly available at: https://github.com/mever-team/omgfuser

count=2
* Evidential Deep Learning for Open Set Action Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Bao_Evidential_Deep_Learning_for_Open_Set_Action_Recognition_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Bao_Evidential_Deep_Learning_for_Open_Set_Action_Recognition_ICCV_2021_paper.pdf)]
    * Title: Evidential Deep Learning for Open Set Action Recognition
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Wentao Bao, Qi Yu, Yu Kong
    * Abstract: In a real-world scenario, human actions are typically out of the distribution from training data, which requires a model to both recognize the known actions and reject the unknown. Different from image data, video actions are more challenging to be recognized in an open-set setting due to the uncertain temporal dynamics and static bias of human actions. In this paper, we propose a Deep Evidential Action Recognition (DEAR) method to recognize actions in an open testing set. Specifically, we formulate the action recognition problem from the evidential deep learning (EDL) perspective and propose a novel model calibration method to regularize the EDL training. Besides, to mitigate the static bias of video representation, we propose a plug-and-play module to debias the learned representation through contrastive learning. Experimental results show that our DEAR method achieves consistent performance gain on multiple mainstream action recognition models and benchmarks. Code and pre-trained models are available at https://www.rit.edu/actionlab/dear.

count=2
* Uncertainty Aware Semi-Supervised Learning on Graph Data
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/968c9b4f09cbb7d7925f38aea3484111-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/968c9b4f09cbb7d7925f38aea3484111-Paper.pdf)]
    * Title: Uncertainty Aware Semi-Supervised Learning on Graph Data
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Xujiang Zhao, Feng Chen, Shu Hu, Jin-Hee Cho
    * Abstract: Thanks to graph neural networks (GNNs), semi-supervised node classification has shown the state-of-the-art performance in graph data. However, GNNs have not considered different types of uncertainties associated with class probabilities to minimize risk of increasing misclassification under uncertainty in real life. In this work, we propose a multi-source uncertainty framework using a GNN that reflects various types of predictive uncertainties in both deep learning and belief/evidence theory domains for node classification predictions. By collecting evidence from the given labels of training nodes, the Graph-based Kernel Dirichlet distribution Estimation (GKDE) method is designed for accurately predicting node-level Dirichlet distributions and detecting out-of-distribution (OOD) nodes. We validated the outperformance of our proposed model compared to the state-of-the-art counterparts in terms of misclassification detection and OOD detection based on six real network datasets. We found that dissonance-based detection yielded the best results on misclassification detection while vacuity-based detection was the best for OOD detection. To clarify the reasons behind the results, we provided the theoretical proof that explains the relationships between different types of uncertainties considered in this work.

count=2
* Multifaceted Uncertainty Estimation for Label-Efficient Deep Learning
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/c80d9ba4852b67046bee487bcd9802c0-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/c80d9ba4852b67046bee487bcd9802c0-Paper.pdf)]
    * Title: Multifaceted Uncertainty Estimation for Label-Efficient Deep Learning
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Weishi Shi, Xujiang Zhao, Feng Chen, Qi Yu
    * Abstract: We present a novel multi-source uncertainty prediction approach that enables deep learning (DL) models to be actively trained with much less labeled data. By leveraging the second-order uncertainty representation provided by subjective logic (SL), we conduct evidence-based theoretical analysis and formally decompose the predicted entropy over multiple classes into two distinct sources of uncertainty: vacuity and dissonance, caused by lack of evidence and conflict of strong evidence, respectively. The evidence based entropy decomposition provides deeper insights on the nature of uncertainty, which can help effectively explore a large and high-dimensional unlabeled data space. We develop a novel loss function that augments DL based evidence prediction with uncertainty anchor sample identification. The accurately estimated multiple sources of uncertainty are systematically integrated and dynamically balanced using a data sampling function for label-efficient active deep learning (ADL). Experiments conducted over both synthetic and real data and comparison with competitive AL methods demonstrate the effectiveness of the proposed ADL model.

count=2
* Reasoning about Uncertainties in Discrete-Time Dynamical Systems using Polynomial Forms.
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/ca886eb9edb61a42256192745c72cd79-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/ca886eb9edb61a42256192745c72cd79-Paper.pdf)]
    * Title: Reasoning about Uncertainties in Discrete-Time Dynamical Systems using Polynomial Forms.
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Sriram Sankaranarayanan, Yi Chou, Eric Goubault, Sylvie Putot
    * Abstract: In this paper, we propose polynomial forms to represent distributions of state variables over time for discrete-time stochastic dynamical systems. This problem arises in a variety of applications in areas ranging from biology to robotics. Our approach allows us to rigorously represent the probability distribution of state variables over time, and provide guaranteed bounds on the expectations, moments and probabilities of tail events involving the state variables. First, we recall ideas from interval arithmetic, and use them to rigorously represent the state variables at time t as a function of the initial state variables and noise symbols that model the random exogenous inputs encountered before time t. Next, we show how concentration of measure inequalities can be employed to prove rigorous bounds on the tail probabilities of these state variables. We demonstrate interesting applications that demonstrate how our approach can be useful in some situations to establish mathematically guaranteed bounds that are of a different nature from those obtained through simulations with pseudo-random numbers.

count=2
* Computing a human-like reaction time metric from stable recurrent vision models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/2e351740d4ec4200df6160f34cd181c3-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/2e351740d4ec4200df6160f34cd181c3-Paper-Conference.pdf)]
    * Title: Computing a human-like reaction time metric from stable recurrent vision models
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Lore Goetschalckx, Lakshmi Narasimhan Govindarajan, Alekh Karkada Ashok, Aarit Ahuja, David Sheinberg, Thomas Serre
    * Abstract: The meteoric rise in the adoption of deep neural networks as computational models of vision has inspired efforts to ``align” these models with humans. One dimension of interest for alignment includes behavioral choices, but moving beyond characterizing choice patterns to capturing temporal aspects of visual decision-making has been challenging. Here, we sketch a general-purpose methodology to construct computational accounts of reaction times from a stimulus-computable, task-optimized model. Specifically, we introduce a novel metric leveraging insights from subjective logic theory summarizing evidence accumulation in recurrent vision models. We demonstrate that our metric aligns with patterns of human reaction times for stimulus manipulations across four disparate visual decision-making tasks spanning perceptual grouping, mental simulation, and scene categorization. This work paves the way for exploring the temporal alignment of model and human visual strategies in the context of various other cognitive tasks toward generating testable hypotheses for neuroscience. Links to the code and data can be found on the project page: https://serre-lab.github.io/rnnrtssite/.

count=2
* Beyond Unimodal: Generalising Neural Processes for Multimodal Uncertainty Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/839e23e5b1c52cfd1268f4023a3af0d6-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/839e23e5b1c52cfd1268f4023a3af0d6-Paper-Conference.pdf)]
    * Title: Beyond Unimodal: Generalising Neural Processes for Multimodal Uncertainty Estimation
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Myong Chol Jung, He Zhao, Joanna Dipnall, Lan Du
    * Abstract: Uncertainty estimation is an important research area to make deep neural networks (DNNs) more trustworthy. While extensive research on uncertainty estimation has been conducted with unimodal data, uncertainty estimation for multimodal data remains a challenge. Neural processes (NPs) have been demonstrated to be an effective uncertainty estimation method for unimodal data by providing the reliability of Gaussian processes with efficient and powerful DNNs. While NPs hold significant potential for multimodal uncertainty estimation, the adaptation of NPs for multimodal data has not been carefully studied. To bridge this gap, we propose Multimodal Neural Processes (MNPs) by generalising NPs for multimodal uncertainty estimation. Based on the framework of NPs, MNPs consist of several novel and principled mechanisms tailored to the characteristics of multimodal data. In extensive empirical evaluation, our method achieves state-of-the-art multimodal uncertainty estimation performance, showing its appealing robustness against noisy samples and reliability in out-of-distribution detection with faster computation time compared to the current state-of-the-art multimodal uncertainty estimation method.

count=1
* Quality-based Multimodal Classification using Tree-Structured Sparsity
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Bahrampour_Quality-based_Multimodal_Classification_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Bahrampour_Quality-based_Multimodal_Classification_2014_CVPR_paper.pdf)]
    * Title: Quality-based Multimodal Classification using Tree-Structured Sparsity
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Soheil Bahrampour, Asok Ray, Nasser M. Nasrabadi, Kenneth W. Jenkins
    * Abstract: Recent studies have demonstrated advantages of information fusion based on sparsity models for multimodal classification. Among several sparsity models, tree-structured sparsity provides a flexible framework for extraction of cross-correlated information from different sources and for enforcing group sparsity at multiple granularities. However, the existing algorithm only solves an approximated version of the cost functional and the resulting solution is not necessarily sparse at group levels. This paper reformulates the tree-structured sparse model for multimodal classification task. An accelerated proximal algorithm is proposed to solve the optimization problem, which is an efficient tool for feature-level fusion among either homogeneous or heterogeneous sources of information. In addition, a (fuzzy-set-theoretic) possibilistic scheme is proposed to weight the available modalities, based on their respective reliability, in a joint optimization problem for finding the sparsity codes. This approach provides a general framework for quality-based fusion that offers added robustness to several sparsity-based multimodal classification algorithms. To demonstrate their efficacy, the proposed methods are evaluated on three different applications - multiview face recognition, multimodal face recognition, and target classification.

count=1
* Embedded Computing Framework for Vision-Based Real-Time Surround Threat Analysis and Driver Assistance
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/html/Lu_Embedded_Computing_Framework_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/papers/Lu_Embedded_Computing_Framework_CVPR_2016_paper.pdf)]
    * Title: Embedded Computing Framework for Vision-Based Real-Time Surround Threat Analysis and Driver Assistance
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Frankie Lu, Sean Lee, Ravi Kumar Satzoda, Mohan Trivedi
    * Abstract: In this paper, we present a distributed embedded vision system that enables surround scene analysis and vehicle threat estimation. The proposed system analyzes the surroundings of the ego-vehicle using four cameras, each connected to a separate embedded processor. Each processor runs a set of optimized vision-based techniques to detect surrounding vehicles, so that the entire system operates at real-time speeds. This setup has been demonstrated on multiple vehicle testbeds with high levels of robustness under real-world driving conditions and is scalable to additional cameras. Finally, we present a detailed evaluation which shows over 95% accuracy and operation at nearly 15 frames per second.

count=1
* Stochastic Classifiers for Unsupervised Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Lu_Stochastic_Classifiers_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_Stochastic_Classifiers_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf)]
    * Title: Stochastic Classifiers for Unsupervised Domain Adaptation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zhihe Lu,  Yongxin Yang,  Xiatian Zhu,  Cong Liu,  Yi-Zhe Song,  Tao Xiang
    * Abstract: A common strategy adopted by existing state-of-the-art unsupervised domain adaptation (UDA) methods is to employ two classifiers to identify the misaligned local regions between source and target domain. Following the 'wisdom of the crowd' principle, one has to ask: why stop at two? Indeed, we find that using more classifiers leads to better performance, but also introduces more model parameters, therefore risking overfitting. In this paper, we introduce a novel method called STochastic clAssifieRs (STAR) for addressing this problem. Instead of representing one classifier as a weight vector, STAR models it as a Gaussian distribution with its variance representing the inter-classifier discrepancy. With STAR, we can now sample an arbitrary number of classifiers from the distribution, whilst keeping the model size the same as having two classifiers. Extensive experiments demonstrate that a variety of existing UDA methods can greatly benefit from STAR and achieve the state-of-the-art performance on both image classification and semantic segmentation tasks.

count=1
* OpenTAL: Towards Open Set Temporal Action Localization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Bao_OpenTAL_Towards_Open_Set_Temporal_Action_Localization_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bao_OpenTAL_Towards_Open_Set_Temporal_Action_Localization_CVPR_2022_paper.pdf)]
    * Title: OpenTAL: Towards Open Set Temporal Action Localization
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Wentao Bao, Qi Yu, Yu Kong
    * Abstract: Temporal Action Localization (TAL) has experienced remarkable success under the supervised learning paradigm. However, existing TAL methods are rooted in the closed set assumption, which cannot handle the inevitable unknown actions in open-world scenarios. In this paper, we, for the first time, step toward the Open Set TAL (OSTAL) problem and propose a general framework OpenTAL based on Evidential Deep Learning (EDL). Specifically, the OpenTAL consists of uncertainty-aware action classification, actionness prediction, and temporal location regression. With the proposed importance-balanced EDL method, classification uncertainty is learned by collecting categorical evidence majorly from important samples. To distinguish the unknown actions from background video frames, the actionness is learned by the positive-unlabeled learning. The classification uncertainty is further calibrated by leveraging the guidance from the temporal localization quality. The OpenTAL is general to enable existing TAL models for open set scenarios, and experimental results on THUMOS14 and ActivityNet1.3 benchmarks show the effectiveness of our method. The code and pre-trained models are released at https://www.rit.edu/actionlab/opental.

count=1
* Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR_2023_paper.pdf)]
    * Title: Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Kemal Oksuz, Tom Joy, Puneet K. Dokania
    * Abstract: The current approach for testing the robustness of object detectors suffers from serious deficiencies such as improper methods of performing out-of-distribution detection and using calibration metrics which do not consider both localisation and classification quality. In this work, we address these issues, and introduce the Self Aware Object Detection (SAOD) task, a unified testing framework which respects and adheres to the challenges that object detectors face in safety-critical environments such as autonomous driving. Specifically, the SAOD task requires an object detector to be: robust to domain shift; obtain reliable uncertainty estimates for the entire scene; and provide calibrated confidence scores for the detections. We extensively use our framework, which introduces novel metrics and large scale test datasets, to test numerous object detectors in two different use-cases, allowing us to highlight critical insights into their robustness performance. Finally, we introduce a simple baseline for the SAOD task, enabling researchers to benchmark future proposed methods and move towards robust object detectors which are fit for purpose. Code is available at: https://github.com/fiveai/saod

count=1
* Open Set Action Recognition via Multi-Label Evidential Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Open_Set_Action_Recognition_via_Multi-Label_Evidential_Learning_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Open_Set_Action_Recognition_via_Multi-Label_Evidential_Learning_CVPR_2023_paper.pdf)]
    * Title: Open Set Action Recognition via Multi-Label Evidential Learning
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Chen Zhao, Dawei Du, Anthony Hoogs, Christopher Funk
    * Abstract: Existing methods for open set action recognition focus on novelty detection that assumes video clips show a single action, which is unrealistic in the real world. We propose a new method for open set action recognition and novelty detection via MUlti-Label Evidential learning (MULE), that goes beyond previous novel action detection methods by addressing the more general problems of single or multiple actors in the same scene, with simultaneous action(s) by any actor. Our Beta Evidential Neural Network estimates multi-action uncertainty with Beta densities based on actor-context-object relation representations. An evidence debiasing constraint is added to the objective func- tion for optimization to reduce the static bias of video representations, which can incorrectly correlate predictions and static cues. We develop a primal-dual average scheme update-based learning algorithm to optimize the proposed problem and provide corresponding theoretical analysis. Besides, uncertainty and belief-based novelty estimation mechanisms are formulated to detect novel actions. Extensive experiments on two real-world video datasets show that our proposed approach achieves promising performance in single/multi-actor, single/multi-action settings. Our code and models are released at https://github.com/charliezhaoyinpeng/mule.

count=1
* Beyond AUROC & Co. for Evaluating Out-of-Distribution Detection Performance
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/html/Humblot-Renaux_Beyond_AUROC__Co._for_Evaluating_Out-of-Distribution_Detection_Performance_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/papers/Humblot-Renaux_Beyond_AUROC__Co._for_Evaluating_Out-of-Distribution_Detection_Performance_CVPRW_2023_paper.pdf)]
    * Title: Beyond AUROC & Co. for Evaluating Out-of-Distribution Detection Performance
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Galadrielle Humblot-Renaux, Sergio Escalera, Thomas B. Moeslund
    * Abstract: While there has been a growing research interest in developing out-of-distribution (OOD) detection methods, there has been comparably little discussion around how these methods should be evaluated. Given their relevance for safe(r) AI, it is important to examine whether the basis for comparing OOD detection methods is consistent with practical needs. In this work, we take a closer look at the go-to metrics for evaluating OOD detection, and question the approach of exclusively reducing OOD detection to a binary classification task with little consideration for the detection threshold. We illustrate the limitations of current metrics (AUROC & its friends) and propose a new metric - Area Under the Threshold Curve (AUTC), which explicitly penalizes poor separation between ID and OOD samples. Scripts and data are available at https://github.com/glhr/beyond-auroc

count=1
* Accurate Training Data for Occupancy Map Prediction in Automated Driving Using Evidence Theory
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kalble_Accurate_Training_Data_for_Occupancy_Map_Prediction_in_Automated_Driving_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Kalble_Accurate_Training_Data_for_Occupancy_Map_Prediction_in_Automated_Driving_CVPR_2024_paper.pdf)]
    * Title: Accurate Training Data for Occupancy Map Prediction in Automated Driving Using Evidence Theory
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jonas Kälble, Sascha Wirges, Maxim Tatarchenko, Eddy Ilg
    * Abstract: Automated driving fundamentally requires knowledge about the surrounding geometry of the scene. Modern approaches use only captured images to predict occupancy maps that represent the geometry. Training these approaches requires accurate data that may be acquired with the help of LiDAR scanners. We show that the techniques used for current benchmarks and training datasets to convert LiDAR scans into occupancy grid maps yield very low quality and subsequently present a novel approach using evidence theory that yields more accurate reconstructions. We demonstrate that these are superior by a large margin both qualitatively and quantitatively and that we additionally obtain meaningful uncertainty estimates. When converting the occupancy maps back to depth estimates and comparing them with the raw LiDAR measurements our method yields a MAE improvement of 30% to 52% on nuScenes and 53% on Waymo over other occupancy ground-truth data. Finally we use the improved occupancy maps to train a state-of-the-art occupancy prediction method and demonstrate that it improves the MAE by 25% on nuScenes.

count=1
* Various Approaches for Driver and Driving Behavior Monitoring: A Review
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W20/html/Kang_Various_Approaches_for_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W20/papers/Kang_Various_Approaches_for_2013_ICCV_paper.pdf)]
    * Title: Various Approaches for Driver and Driving Behavior Monitoring: A Review
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Hang-Bong Kang
    * Abstract: In recent years, driver drowsiness and distraction have been important factors in a large number of accidents because they reduce driver perception level and decision making capability, which negatively affect the ability to control the vehicle. One way to reduce these kinds of accidents would be through monitoring driver and driving behavior and alerting the driver when they are drowsy or in a distracted state. In addition, if it were possible to predict unsafe driving behavior in advance, this would also contribute to safe driving. In this paper, we will discuss various monitoring methods for driver and driving behavior as well as for predicting unsafe driving behaviors. In respect to measurement methods of driver drowsiness, we discussed visual and non-visual features of driver behavior, as well as driving performance behaviors related to vehicle-based features. Visual feature measurements such as eye related measurements, yawning detection, facial expression are discussed in detail. As for non-visual features, we explore various physiological signals and possible drowsiness detection methods that use these signals. As for vehicle-based features, we describe steering wheel movement and the standard deviation of lateral position. To detect driver distraction, we describe head pose and gaze direction methods. To predict unsafe driving behavior, we explain predicting methods based on facial expressions and car dynamics. Finally, we discuss several issues to be tackled for active driver safety systems. They are 1) hybrid measures for drowsiness detection, 2) driving context awareness for safe driving, 3) the necessity for public data sets of simulated and real driving conditions.

count=1
* Evidential Deep Learning to Quantify Classification Uncertainty
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf)]
    * Title: Evidential Deep Learning to Quantify Classification Uncertainty
    * Publisher: NeurIPS
    * Publication Date: `2018`
    * Authors: Murat Sensoy, Lance Kaplan, Melih Kandemir
    * Abstract: Deterministic neural nets have been shown to learn effective predictors on a wide range of machine learning problems. However, as the standard approach is to train the network to minimize a prediction loss, the resultant model remains ignorant to its prediction confidence. Orthogonally to Bayesian neural nets that indirectly infer prediction uncertainty through weight uncertainties, we propose explicit modeling of the same using the theory of subjective logic. By placing a Dirichlet distribution on the class probabilities, we treat predictions of a neural net as subjective opinions and learn the function that collects the evidence leading to these opinions by a deterministic neural net from data. The resultant predictor for a multi-class classification problem is another Dirichlet distribution whose parameters are set by the continuous output of a neural net. We provide a preliminary analysis on how the peculiarities of our new loss function drive improved uncertainty estimation. We observe that our method achieves unprecedented success on detection of out-of-distribution queries and endurance against adversarial perturbations.

count=1
* Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/543e83748234f7cbab21aa0ade66565f-Paper.pdf)]
    * Title: Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Jeremiah Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax Weiss, Balaji Lakshminarayanan
    * Abstract: Bayesian neural networks (BNN) and deep ensembles are principled approaches to estimate the predictive uncertainty of a deep learning model. However their practicality in real-time, industrial-scale applications are limited due to their heavy memory and inference cost. This motivates us to study principled approaches to high-quality uncertainty estimation that require only a single deep neural network (DNN). By formalizing the uncertainty quantification as a minimax learning problem, we first identify input distance awareness, i.e., the model’s ability to quantify the distance of a testing example from the training data in the input space, as a necessary condition for a DNN to achieve high-quality (i.e., minimax optimal) uncertainty estimation. We then propose Spectral-normalized Neural Gaussian Process (SNGP), a simple method that improves the distance-awareness ability of modern DNNs, by adding a weight normalization step during training and replacing the output layer. On a suite of vision and language understanding tasks and on modern architectures (Wide-ResNet and BERT), SNGP is competitive with deep ensembles in prediction, calibration and out-of-domain detection, and outperforms the other single-model approaches.

count=1
* Evidential Softmax for Sparse Multimodal Distributions in Deep Generative Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/60243f9b1ac2dba11ff8131c8f4431e0-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/60243f9b1ac2dba11ff8131c8f4431e0-Paper.pdf)]
    * Title: Evidential Softmax for Sparse Multimodal Distributions in Deep Generative Models
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Phil Chen, Mikhal Itkina, Ransalu Senanayake, Mykel J Kochenderfer
    * Abstract: Many applications of generative models rely on the marginalization of their high-dimensional output probability distributions. Normalization functions that yield sparse probability distributions can make exact marginalization more computationally tractable. However, sparse normalization functions usually require alternative loss functions for training since the log-likelihood is undefined for sparse probability distributions. Furthermore, many sparse normalization functions often collapse the multimodality of distributions. In this work, we present ev-softmax, a sparse normalization function that preserves the multimodality of probability distributions. We derive its properties, including its gradient in closed-form, and introduce a continuous family of approximations to ev-softmax that have full support and can be trained with probabilistic loss functions such as negative log-likelihood and Kullback-Leibler divergence. We evaluate our method on a variety of generative models, including variational autoencoders and auto-regressive architectures. Our method outperforms existing dense and sparse normalization techniques in distributional accuracy. We demonstrate that ev-softmax successfully reduces the dimensionality of probability distributions while maintaining multimodality.

count=1
* Honesty Is the Best Policy: Defining and Mitigating AI Deception
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/06fc7ae4a11a7eb5e20fe018db6c036f-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/06fc7ae4a11a7eb5e20fe018db6c036f-Paper-Conference.pdf)]
    * Title: Honesty Is the Best Policy: Defining and Mitigating AI Deception
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Francis Ward, Francesca Toni, Francesco Belardinelli, Tom Everitt
    * Abstract: Deceptive agents are a challenge for the safety, trustworthiness, and cooperation of AI systems. We focus on the problem that agents might deceive in order to achieve their goals (for instance, in our experiments with language models, the goal of being evaluated as truthful).There are a number of existing definitions of deception in the literature on game theory and symbolic AI, but there is no overarching theory of deception for learning agents in games. We introduce a formaldefinition of deception in structural causal games, grounded in the philosophyliterature, and applicable to real-world machine learning systems.Several examples and results illustrate that our formal definition aligns with the philosophical and commonsense meaning of deception.Our main technical result is to provide graphical criteria for deception. We show, experimentally, that these results can be used to mitigate deception in reinforcement learning agents and language models.

