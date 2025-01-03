count=5
* On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/57587d8d6a7ede0e5302fc22d0878c53-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/57587d8d6a7ede0e5302fc22d0878c53-Paper-Conference.pdf)]
    * Title: On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Qi CHEN, Changjian Shui, Ligong Han, Mario Marchand
    * Abstract: We focus on Continual Meta-Learning (CML), which targets accumulating and exploiting meta-knowledge on a sequence of non-i.i.d. tasks. The primary challenge is to strike a balance between stability and plasticity, where a model should be stable to avoid catastrophic forgetting in previous tasks and plastic to learn generalizable concepts from new tasks. To address this, we formulate the CML objective as controlling the average excess risk upper bound of the task sequence, which reflects the trade-off between forgetting and generalization. Based on the objective, we introduce a unified theoretical framework for CML in both static and shifting environments, providing guarantees for various task-specific learning algorithms. Moreover, we first present a rigorous analysis of a bi-level trade-off in shifting environments. To approach the optimal trade-off, we propose a novel algorithm that dynamically adjusts the meta-parameter and its learning rate w.r.t environment change. Empirical evaluations on synthetic and real datasets illustrate the effectiveness of the proposed theory and algorithm.

count=4
* A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Lin_A_Prior-Less_Method_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lin_A_Prior-Less_Method_CVPR_2018_paper.pdf)]
    * Title: A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Chung-Ching Lin, Ying Hung
    * Abstract: This paper presents a prior-less method for tracking and clustering an unknown number of human faces and maintaining their individual identities in unconstrained videos. The key challenge is to accurately track faces with partial occlusion and drastic appearance changes in multiple shots resulting from significant variations of makeup, facial expression, head pose and illumination. To address this challenge, we propose a new multi-face tracking and re-identification algorithm, which provides high accuracy in face association in the entire video with automatic cluster number generation, and is robust to outliers. We develop a co-occurrence model of multiple body parts to seamlessly create face tracklets, and recursively link tracklets to construct a graph for extracting clusters. A Gaussian Process model is introduced to compensate the deep feature insufficiency, and is further used to refine the linking results. The advantages of the proposed algorithm are demonstrated using a variety of challenging music videos and newly introduced body-worn camera videos. The proposed method obtains significant improvements over the state of the art [51], while relying less on handling video-specific prior information to achieve high performance.

count=4
* Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.pdf)]
    * Title: Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Olaf Wysocki, Yan Xia, Magdalena Wysocki, Eleonora Grilli, Ludwig Hoegner, Daniel Cremers, Uwe Stilla
    * Abstract: Reconstructing semantic 3D building models at the level of detail (LoD) 3 is a long-standing challenge. Unlike mesh-based models, they require watertight geometry and object-wise semantics at the facade level. The principal challenge of such demanding semantic 3D reconstruction is reliable facade-level semantic segmentation of 3D input data. We present a novel method, called Scan2LoD3, that accurately reconstructs semantic LoD3 building models by improving facade-level semantic 3D segmentation. To this end, we leverage laser physics and 3D building model priors to probabilistically identify model conflicts. These probabilistic physical conflicts propose locations of model openings: Their final semantics and shapes are inferred in a Bayesian network fusing multimodal probabilistic maps of conflicts, 3D point clouds, and 2D images. To fulfill demanding LoD3 requirements, we use the estimated shapes to cut openings in 3D building priors and fit semantic 3D objects from a library of facade objects. Extensive experiments on the TUM city campus datasets demonstrate the superior performance of the proposed Scan2LoD3 over the state-of-the-art methods in facade-level detection, semantic segmentation, and LoD3 building model reconstruction. We believe our method can foster the development of probability-driven semantic 3D reconstruction at LoD3 since not only the high-definition reconstruction but also reconstruction confidence becomes pivotal for various applications such as autonomous driving and urban simulations.

count=3
* Underwater Moving Object Detection Using an End-to-End Encoder-Decoder Architecture and GraphSage With Aggregator and Refactoring
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/WiCV/html/Kapoor_Underwater_Moving_Object_Detection_Using_an_End-to-End_Encoder-Decoder_Architecture_and_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/WiCV/papers/Kapoor_Underwater_Moving_Object_Detection_Using_an_End-to-End_Encoder-Decoder_Architecture_and_CVPRW_2023_paper.pdf)]
    * Title: Underwater Moving Object Detection Using an End-to-End Encoder-Decoder Architecture and GraphSage With Aggregator and Refactoring
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Meghna Kapoor, Suvam Patra, Badri Narayan Subudhi, Vinit Jakhetiya, Ankur Bansal
    * Abstract: Underwater environments are greatly affected by several factors, including low visibility, high turbidity, back-scattering, dynamic background, etc., and hence pose challenges in object detection. Several algorithms consider convolutional neural networks to extract deep features and then object detection using the same. However, the dependency on the kernel's size and the network's depth results in fading relationships of latent space features and also are unable to characterize the spatial-contextual bonding of the pixels. Hence, they are unable to procure satisfactory results in complex underwater scenarios. To re-establish this relationship, we propose a unique architecture for underwater object detection where U-Net architecture is considered with the ResNet-50 backbone. Further, the latent space features from the encoder are fed to the decoder through a GraphSage model. GraphSage-based model is explored to reweight the node relationship in non-euclidean space using different aggregator functions and hence characterize the spatio-contextual bonding among the pixels. Further, we explored the dependency on different aggregator functions: mean, max, and LSTM, to evaluate the model's performance. We evaluated the proposed model on two underwater benchmark databases: F4Knowledge and underwater change detection. The performance of the proposed model is evaluated against eleven state-of-the-art techniques in terms of both visual and quantitative evaluation measures.

count=1
* 3MASSIV: Multilingual, Multimodal and Multi-Aspect Dataset of Social Media Short Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Gupta_3MASSIV_Multilingual_Multimodal_and_Multi-Aspect_Dataset_of_Social_Media_Short_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Gupta_3MASSIV_Multilingual_Multimodal_and_Multi-Aspect_Dataset_of_Social_Media_Short_CVPR_2022_paper.pdf)]
    * Title: 3MASSIV: Multilingual, Multimodal and Multi-Aspect Dataset of Social Media Short Videos
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Vikram Gupta, Trisha Mittal, Puneet Mathur, Vaibhav Mishra, Mayank Maheshwari, Aniket Bera, Debdoot Mukherjee, Dinesh Manocha
    * Abstract: We present 3MASSIV, a multilingual, multimodal and multi-aspect, expertly-annotated dataset of diverse short videos extracted from a social media platform. 3MASSIV comprises of 50k short videos (20 seconds average duration) and 100K unlabeled videos in 11 different languages and captures popular short video trends like pranks, fails, romance, comedy expressed via unique audio-visual formats like self-shot videos, reaction videos, lip-synching, self-sung songs, etc. 3MASSIV presents an opportunity for multimodal and multilingual semantic understanding on these unique videos by annotating them for concepts, affective states, media types, and audio language. We present a thorough analysis of 3MASSIV and highlight the variety and unique aspects of our dataset compared to other contemporary popular datasets with strong baselines. We also show how the social media content in 3MASSIV is dynamic and temporal in nature which can be used for various semantic understanding tasks and cross-lingual analysis.

count=1
* UrbanSARFloods: Sentinel-1 SLC-Based Benchmark Dataset for Urban and Open-Area Flood Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Zhao_UrbanSARFloods_Sentinel-1_SLC-Based_Benchmark_Dataset_for_Urban_and_Open-Area_Flood_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Zhao_UrbanSARFloods_Sentinel-1_SLC-Based_Benchmark_Dataset_for_Urban_and_Open-Area_Flood_CVPRW_2024_paper.pdf)]
    * Title: UrbanSARFloods: Sentinel-1 SLC-Based Benchmark Dataset for Urban and Open-Area Flood Mapping
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jie Zhao, Zhitong Xiong, Xiao Xiang Zhu
    * Abstract: Due to its cloud-penetrating capability and independence from solar illumination satellite Synthetic Aperture Radar (SAR) is the preferred data source for large-scale flood mapping providing global coverage and including various land cover classes. However most studies on large-scale SAR-derived flood mapping using deep learning algorithms have primarily focused on flooded open areas utilizing available open-access datasets (e.g. Sen1Floods11) and with limited attention to urban floods. To address this gap we introduce UrbanSARFloods a floodwater dataset featuring pre-processed Sentinel-1 intensity data and interferometric coherence imagery acquired before and during flood events. It contains 8879 512 x 512 chips covering 807500 km2 across 20 land cover classes and 5 continents spanning 18 flood events. We used UrbanSARFloods to benchmark existing state-of-the-art convolutional neural networks (CNNs) for segmenting open and urban flood areas. Our findings indicate that prevalent approaches including the Weighted Cross-Entropy (WCE) loss and the application of transfer learning with pretrained models fall short in overcoming the obstacles posed by imbalanced data and the constraints of a small training dataset. Urban flood detection remains challenging. Future research should explore strategies for addressing imbalanced data challenges and investigate transfer learning's potential for SAR-based large-scale flood mapping. Besides expanding this dataset to include additional flood events holds promise for enhancing its utility and contributing to advancements in flood mapping techniques. The UrbanSARFloods dataset including training validation data and raw data can be found at https://github.com/jie666-6/UrbanSARFloods.

count=1
* EgoTV: Egocentric Task Verification from Natural Language Task Descriptions
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Hazra_EgoTV_Egocentric_Task_Verification_from_Natural_Language_Task_Descriptions_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Hazra_EgoTV_Egocentric_Task_Verification_from_Natural_Language_Task_Descriptions_ICCV_2023_paper.pdf)]
    * Title: EgoTV: Egocentric Task Verification from Natural Language Task Descriptions
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Rishi Hazra, Brian Chen, Akshara Rai, Nitin Kamra, Ruta Desai
    * Abstract: To enable progress towards egocentric agents capable of understanding everyday tasks specified in natural language, we propose a benchmark and a synthetic dataset called Egocentric Task Verification (EgoTV). The goal in EgoTV is to verify the execution of tasks from egocentric videos based on the natural language description of these tasks. EgoTV contains pairs of videos and their task descriptions for multi-step tasks -- these tasks contain multiple sub-task decompositions, state changes, object interactions, and sub-task ordering constraints. In addition, EgoTV also provides abstracted task descriptions that contain only partial details about ways to accomplish a task. Consequently, EgoTV requires causal, temporal, and compositional reasoning of video and language modalities, which is missing in existing datasets. We also find that existing vision-language models struggle at such all round reasoning needed for task verification in EgoTV. Inspired by the needs of EgoTV, we propose a novel Neuro-Symbolic Grounding (NSG) approach that leverages symbolic representations to capture the compositional and temporal structure of tasks. We demonstrate NSG's capability towards task tracking and verification on our EgoTV dataset and a real-world dataset derived from CrossTask (CTV). We open-source the EgoTV and CTV datasets and the NSG model for future research on egocentric assistive agents.

count=1
* Towards Geospatial Foundation Models via Continual Pretraining
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.pdf)]
    * Title: Towards Geospatial Foundation Models via Continual Pretraining
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Matías Mendieta, Boran Han, Xingjian Shi, Yi Zhu, Chen Chen
    * Abstract: Geospatial technologies are becoming increasingly essential in our world for a wide range of applications, including agriculture, urban planning, and disaster response. To help improve the applicability and performance of deep learning models on these geospatial tasks, various works have begun investigating foundation models for this domain. Researchers have explored two prominent approaches for introducing such models in geospatial applications, but both have drawbacks in terms of limited performance benefit or prohibitive training cost. Therefore, in this work, we propose a novel paradigm for building highly effective geospatial foundation models with minimal resource cost and carbon impact. We first construct a compact yet diverse dataset from multiple sources to promote feature diversity, which we term GeoPile. Then, we investigate the potential of continual pretraining from large-scale ImageNet-22k models and propose a multi-objective continual pretraining paradigm, which leverages the strong representations of ImageNet while simultaneously providing the freedom to learn valuable in-domain features. Our approach outperforms previous state-of-the-art geospatial pretraining methods in an extensive evaluation on seven downstream datasets covering various tasks such as change detection, classification, multi-label classification, semantic segmentation, and super-resolution. Code is available at https://github.com/mmendiet/GFM.

count=1
* ZRG: A Dataset for Multimodal 3D Residential Rooftop Understanding
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Corley_ZRG_A_Dataset_for_Multimodal_3D_Residential_Rooftop_Understanding_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Corley_ZRG_A_Dataset_for_Multimodal_3D_Residential_Rooftop_Understanding_WACV_2024_paper.pdf)]
    * Title: ZRG: A Dataset for Multimodal 3D Residential Rooftop Understanding
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Isaac Corley, Jonathan Lwowski, Peyman Najafirad
    * Abstract: A crucial part of any home is the roof over our heads to protect us from the elements. In this paper we present the Zeitview Rooftop Geometry (ZRG) dataset for residential rooftop understanding. ZRG is a large-scale residential rooftop inspection dataset of over 20k properties from across the U.S. and includes high resolution aerial orthomosaics, digital surface models (DSM), colored point clouds, and 3D roof wireframe annotations. We provide an in-depth analysis and perform several experimental baselines including roof outline extraction, monocular height estimation, and planar roof structure extraction, to illustrate a few of the numerous applications unlocked by this dataset.

count=1
* WATCH: Wide-Area Terrestrial Change Hypercube
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Greenwell_WATCH_Wide-Area_Terrestrial_Change_Hypercube_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Greenwell_WATCH_Wide-Area_Terrestrial_Change_Hypercube_WACV_2024_paper.pdf)]
    * Title: WATCH: Wide-Area Terrestrial Change Hypercube
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Connor Greenwell, Jon Crall, Matthew Purri, Kristin Dana, Nathan Jacobs, Armin Hadzic, Scott Workman, Matt Leotta
    * Abstract: Monitoring Earth activity using data collected from multiple satellite imaging platforms in a unified way is a significant challenge, especially with large variability in image resolution, spectral bands, and revisit rates. Further, the availability of sensor data varies across time as new platforms are launched. In this work, we introduce an adaptable framework and network architecture capable of predicting on subsets of the available platforms, bands, or temporal ranges it was trained on. Our system, called WATCH, is highly general and can be applied to a variety of geospatial tasks. In this work, we analyze the performance of WATCH using the recent IARPA SMART public dataset and metrics. We focus primarily on the problem of broad area search for heavy construction sites. Experiments validate the robustness of WATCH during inference to limited sensor availability, as well the the ability to alter inference-time spatial or temporal sampling. WATCH is open source and available for use on this or other remote sensing problems. Code and model weights are available at: https://gitlab.kitware.com/computer-vision/geowatch

count=1
* Robust Lipschitz Bandits to Adversarial Corruptions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/238f3b98bbe998b4f2234443907fe663-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/238f3b98bbe998b4f2234443907fe663-Paper-Conference.pdf)]
    * Title: Robust Lipschitz Bandits to Adversarial Corruptions
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Yue Kang, Cho-Jui Hsieh, Thomas Chun Man Lee
    * Abstract: Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.

