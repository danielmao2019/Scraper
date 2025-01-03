count=88
* Improving Speaker Turn Embedding by Crossmodal Transfer Learning From Face Embedding
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w8/html/Le_Improving_Speaker_Turn_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w8/Le_Improving_Speaker_Turn_ICCV_2017_paper.pdf)]
    * Title: Improving Speaker Turn Embedding by Crossmodal Transfer Learning From Face Embedding
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Nam Le, Jean-Marc Odobez
    * Abstract: Learning speaker turn embeddings has shown considerable improvement in situations where conventional speaker modeling approaches fail. However, this improvement is relatively limited when compared to the gain observed in face embedding learning, which has proven very successful for face verification and clustering tasks. Assuming that face and voices from the same identities share some latent properties (like age, gender, ethnicity), we propose two transfer learning approaches to leverage the knowledge from the face domain learned from thousands of images and identities for tasks in the speaker domain. These approaches, namely target embedding transfer and clustering structure transfer, utilize the structure of the source face embedding space at different granularities to regularize the target speaker turn embedding space as optimizing terms. Our methods are evaluated on two public broadcast corpora and yield promising advances over competitive baselines in verification and audio clustering tasks, especially when dealing with short speaker utterances. The analysis of the results also gives insight into characteristics of the embedding spaces and shows their potential applications.

count=65
* A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Lin_A_Prior-Less_Method_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lin_A_Prior-Less_Method_CVPR_2018_paper.pdf)]
    * Title: A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Chung-Ching Lin, Ying Hung
    * Abstract: This paper presents a prior-less method for tracking and clustering an unknown number of human faces and maintaining their individual identities in unconstrained videos. The key challenge is to accurately track faces with partial occlusion and drastic appearance changes in multiple shots resulting from significant variations of makeup, facial expression, head pose and illumination. To address this challenge, we propose a new multi-face tracking and re-identification algorithm, which provides high accuracy in face association in the entire video with automatic cluster number generation, and is robust to outliers. We develop a co-occurrence model of multiple body parts to seamlessly create face tracklets, and recursively link tracklets to construct a graph for extracting clusters. A Gaussian Process model is introduced to compensate the deep feature insufficiency, and is further used to refine the linking results. The advantages of the proposed algorithm are demonstrated using a variety of challenging music videos and newly introduced body-worn camera videos. The proposed method obtains significant improvements over the state of the art [51], while relying less on handling video-specific prior information to achieve high performance.

count=55
* Articulated Motion Discovery Using Pairs of Trajectories
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Pero_Articulated_Motion_Discovery_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Pero_Articulated_Motion_Discovery_2015_CVPR_paper.pdf)]
    * Title: Articulated Motion Discovery Using Pairs of Trajectories
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Luca Del Pero, Susanna Ricco, Rahul Sukthankar, Vittorio Ferrari
    * Abstract: We propose an unsupervised approach for discovering characteristic motion patterns in videos of highly articulated objects performing natural, unscripted behaviors, such as tigers in the wild. We discover consistent patterns in a bottom-up manner by analyzing the relative displacements of large numbers of ordered trajectory pairs through time, such that each trajectory is attached to a different moving part on the object. The pairs of trajectories descriptor relies entirely on motion and is more discriminative than state-of-the-art features that employ single trajectories. Our method generates temporal video intervals, each automatically trimmed to one instance of the discovered behavior, and clusters them by type (e.g., running, turning head, drinking water). We present experiments on two datasets: dogs from YouTube-Objects and a new dataset of National Geographic tiger videos. Results confirm that our proposed descriptor outperforms existing appearance- and trajectory-based descriptors (e.g., HOG and DTFs) on both datasets and enables us to segment unconstrained animal video into intervals containing single behaviors.

count=50
* Self-Supervised Object Detection from Egocentric Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.pdf)]
    * Title: Self-Supervised Object Detection from Egocentric Videos
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Peri Akiva, Jing Huang, Kevin J Liang, Rama Kovvuri, Xingyu Chen, Matt Feiszli, Kristin Dana, Tal Hassner
    * Abstract: Understanding the visual world from the perspective of humans (egocentric) has been a long-standing challenge in computer vision. Egocentric videos exhibit high scene complexity and irregular motion flows compared to typical video understanding tasks. With the egocentric domain in mind, we address the problem of self-supervised, class-agnostic object detection, which aims to locate all objects in a given view, regardless of category, without any annotations or pre-training weights. Our method, self-supervised object Detection from Egocentric VIdeos (DEVI), generalizes appearance-based methods to learn features that are category-specific and invariant to viewing angles and illumination conditions from highly ambiguous environments in an end-to-end manner. Our approach leverages typical human behavior and its egocentric perception to sample diverse views of the same objects for our multi-view and scale-regression loss functions. With our learned cluster residual module, we are able to effectively describe multi-category patches for better complex scene understanding. DEVI provides a boost in performance on recent egocentric datasets, with performance gains up to 4.11% AP50, 0.11% AR1, 1.32% AR10, and 5.03% AR100, while significantly reducing model complexity. We also demonstrate competitive performance on out-of-domain datasets without additional training or fine-tuning.

count=46
* Crowd Activity Change Point Detection in Videos via Graph Stream Mining
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w6/html/Yang_Crowd_Activity_Change_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Yang_Crowd_Activity_Change_CVPR_2018_paper.pdf)]
    * Title: Crowd Activity Change Point Detection in Videos via Graph Stream Mining
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Meng Yang, Lida Rashidi, Sutharshan Rajasegarar, Christopher Leckie, Aravinda S. Rao, Marimuthu Palaniswami
    * Abstract: In recent years, there has been a growing interest in detecting anomalous behavioral patterns in video. In this work, we address this task by proposing a novel activity change point detection method to identify crowd movement anomalies for video surveillance. In our proposed novel framework, a hyperspherical clustering algorithm is utilized for the automatic identification of interesting regions, then the density of pedestrian flows between every pair of interesting regions over consecutive time intervals is monitored and represented as a sequence of adjacency matrices where the direction and density of flows are captured through a directed graph. Finally, we use graph edit distance as well as a cumulative sum test to detect change points in the graph sequence. We conduct experiments on four real-world video datasets: Dublin, New Orleans, Abbey Road and MCG Datasets. We observe that our proposed approach achieves a high F-measure, i.e., in the range [0.7, 1], for these datasets. The evaluation reveals that our proposed method can successfully detect the change points in all datasets at both global and local levels. Our results also demonstrate the efficiency and effectiveness of our proposed algorithm for change point detection and segmentation tasks.

count=46
* Taming Local Effects in Graph-based Spatiotemporal Forecasting
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/ad58c61c71efd5436134a3ecc87da6ea-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/ad58c61c71efd5436134a3ecc87da6ea-Paper-Conference.pdf)]
    * Title: Taming Local Effects in Graph-based Spatiotemporal Forecasting
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Andrea Cini, Ivan Marisca, Daniele Zambon, Cesare Alippi
    * Abstract: Spatiotemporal graph neural networks have shown to be effective in time series forecasting applications, achieving better performance than standard univariate predictors in several settings. These architectures take advantage of a graph structure and relational inductive biases to learn a single (global) inductive model to predict any number of the input time series, each associated with a graph node. Despite the gain achieved in computational and data efficiency w.r.t. fitting a set of local models, relying on a single global model can be a limitation whenever some of the time series are generated by a different spatiotemporal stochastic process. The main objective of this paper is to understand the interplay between globality and locality in graph-based spatiotemporal forecasting, while contextually proposing a methodological framework to rationalize the practice of including trainable node embeddings in such architectures. We ascribe to trainable node embeddings the role of amortizing the learning of specialized components. Moreover, embeddings allow for 1) effectively combining the advantages of shared message-passing layers with node-specific parameters and 2) efficiently transferring the learned model to new node sets. Supported by strong empirical evidence, we provide insights and guidelines for specializing graph-based models to the dynamics of each time series and show how this aspect plays a crucial role in obtaining accurate predictions.

count=39
* Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.pdf)]
    * Title: Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Peri Akiva, Matthew Purri, Matthew Leotta
    * Abstract: Self-supervised learning aims to learn image feature representations without the usage of manually annotated labels. It is often used as a precursor step to obtain useful initial network weights which contribute to faster convergence and superior performance of downstream tasks. While self-supervision allows one to reduce the domain gap between supervised and unsupervised learning without the usage of labels, the self-supervised objective still requires a strong inductive bias to downstream tasks for effective transfer learning. In this work, we present our material and texture based self-supervision method named MATTER (MATerial and TExture Representation Learning), which is inspired by classical material and texture methods. Material and texture can effectively describe any surface, including its tactile properties, color, and specularity. By extension, effective representation of material and texture can describe other semantic classes strongly associated with said material and texture. MATTER leverages multi-temporal, spatially aligned remote sensing imagery over unchanged regions to learn invariance to illumination and viewing angle as a mechanism to achieve consistency of material and texture representation. We show that our self-supervision pre-training method allows for up to 24.22% and 6.33% performance increase in unsupervised and fine-tuned setups, and up to 76% faster convergence on change detection, land cover classification, and semantic segmentation tasks.

count=37
* DiSC: Differential Spectral Clustering of Features
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/a84953147312ea2e8b020e53a267321b-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/a84953147312ea2e8b020e53a267321b-Paper-Conference.pdf)]
    * Title: DiSC: Differential Spectral Clustering of Features
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Ram Dyuthi Sristi, Gal Mishne, Ariel Jaffe
    * Abstract: Selecting subsets of features that differentiate between two conditions is a key task in a broad range of scientific domains. In many applications, the features of interest form clusters with similar effects on the data at hand. To recover such clusters we develop DiSC, a data-driven approach for detecting groups of features that differentiate between conditions. For each condition, we construct a graph whose nodes correspond to the features and whose weights are functions of the similarity between them for that condition. We then apply a spectral approach to compute subsets of nodes whose connectivity pattern differs significantly between the condition-specific feature graphs. On the theoretical front, we analyze our approach with a toy example based on the stochastic block model. We evaluate DiSC on a variety of datasets, including MNIST, hyperspectral imaging, simulated scRNA-seq and task fMRI, and demonstrate that DiSC uncovers features that better differentiate between conditions compared to competing methods.

count=36
* Decoding the Enigma: Benchmarking Humans and AIs on the Many Facets of Working Memory
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/ea8758dbe6cc5e6e1764c009acb4c31e-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/ea8758dbe6cc5e6e1764c009acb4c31e-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Decoding the Enigma: Benchmarking Humans and AIs on the Many Facets of Working Memory
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Ankur Sikarwar, Mengmi Zhang
    * Abstract: Working memory (WM), a fundamental cognitive process facilitating the temporary storage, integration, manipulation, and retrieval of information, plays a vital role in reasoning and decision-making tasks. Robust benchmark datasets that capture the multifaceted nature of WM are crucial for the effective development and evaluation of AI WM models. Here, we introduce a comprehensive Working Memory (WorM) benchmark dataset for this purpose. WorM comprises 10 tasks and a total of 1 million trials, assessing 4 functionalities, 3 domains, and 11 behavioral and neural characteristics of WM. We jointly trained and tested state-of-the-art recurrent neural networks and transformers on all these tasks. We also include human behavioral benchmarks as an upper bound for comparison. Our results suggest that AI models replicate some characteristics of WM in the brain, most notably primacy and recency effects, and neural clusters and correlates specialized for different domains and functionalities of WM. In the experiments, we also reveal some limitations in existing models to approximate human behavior. This dataset serves as a valuable resource for communities in cognitive psychology, neuroscience, and AI, offering a standardized framework to compare and enhance WM models, investigate WM's neural underpinnings, and develop WM models with human-like capabilities. Our source code and data are available at: https://github.com/ZhangLab-DeepNeuroCogLab/WorM

count=30
* Online Dominant and Anomalous Behavior Detection in Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Roshtkhari_Online_Dominant_and_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Roshtkhari_Online_Dominant_and_2013_CVPR_paper.pdf)]
    * Title: Online Dominant and Anomalous Behavior Detection in Videos
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Mehrsan Javan Roshtkhari, Martin D. Levine
    * Abstract: We present a novel approach for video parsing and simultaneous online learning of dominant and anomalous behaviors in surveillance videos. Dominant behaviors are those occurring frequently in videos and hence, usually do not attract much attention. They can be characterized by different complexities in space and time, ranging from a scene background to human activities. In contrast, an anomalous behavior is defined as having a low likelihood of occurrence. We do not employ any models of the entities in the scene in order to detect these two kinds of behaviors. In this paper, video events are learnt at each pixel without supervision using densely constructed spatio-temporal video volumes. Furthermore, the volumes are organized into large contextual graphs. These compositions are employed to construct a hierarchical codebook model for the dominant behaviors. By decomposing spatio-temporal contextual information into unique spatial and temporal contexts, the proposed framework learns the models of the dominant spatial and temporal events. Thus, it is ultimately capable of simultaneously modeling high-level behaviors as well as low-level spatial, temporal and spatio-temporal pixel level changes.

count=30
* Progressive Unsupervised Deep Transfer Learning for Forest Mapping in Satellite Image
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/html/Ahmed_Progressive_Unsupervised_Deep_Transfer_Learning_for_Forest_Mapping_in_Satellite_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/papers/Ahmed_Progressive_Unsupervised_Deep_Transfer_Learning_for_Forest_Mapping_in_Satellite_ICCVW_2021_paper.pdf)]
    * Title: Progressive Unsupervised Deep Transfer Learning for Forest Mapping in Satellite Image
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Nouman Ahmed, Sudipan Saha, Muhammad Shahzad, Muhammad Moazam Fraz, Xiao Xiang Zhu
    * Abstract: Automated forest mapping is important to understand our forests that play a key role in ecological system. However, efforts towards forest mapping is impeded by difficulty to collect labeled forest images that show large intraclass variation. Recently unsupervised learning has shown promising capability when exploiting limited labeled data. Motivated by this, we propose a progressive unsupervised deep transfer learning method for forest mapping. The proposed method exploits a pre-trained model that is subsequently fine-tuned over the target forest domain. We propose two different fine-tuning mechanism, one works in a totally unsupervised setting by jointly learning the parameters of CNN and the k-means based cluster assignments of the resulting features and the other one works in a semi-supervised setting by exploiting the extracted knearest neighbor based pseudo labels. The proposed progressive scheme is evaluated on publicly available EuroSAT dataset using the relevant base model trained on BigEarthNet labels. The results show that the proposed method greatly improves the forest regions classification accuracy as compared to the unsupervised baseline, nearly approaching the supervised classification approach.

count=29
* Unsupervised Human Action Detection by Action Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/html/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/papers/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.pdf)]
    * Title: Unsupervised Human Action Detection by Action Matching
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Basura Fernando, Sareh Shirazi, Stephen Gould
    * Abstract: We propose a new task of unsupervised action detection by action matching. Given two long videos, the objective is to temporally detect all pairs of matching video segments. A pair of video segments are matched if they share the same human action. The task is category independent---it does not matter what action is being performed---and no supervision is used to discover such video segments. Unsupervised action detection by action matching allows us to align videos in a meaningful manner. As such, it can be used to discover new action categories or as an action proposal technique within, say, an action detection pipeline. We solve this new task using an effective and efficient method. We use an unsupervised temporal encoding method and exploit the temporal consistency in human actions to obtain candidate action segments. We evaluate our method on this challenging task using three activity recognition benchmarks.

count=26
* Meta Learning on a Sequence of Imbalanced Domains With Difficulty Awareness
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Meta_Learning_on_a_Sequence_of_Imbalanced_Domains_With_Difficulty_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Meta_Learning_on_a_Sequence_of_Imbalanced_Domains_With_Difficulty_ICCV_2021_paper.pdf)]
    * Title: Meta Learning on a Sequence of Imbalanced Domains With Difficulty Awareness
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zhenyi Wang, Tiehang Duan, Le Fang, Qiuling Suo, Mingchen Gao
    * Abstract: Recognizing new objects by learning from a few labeled examples in an evolving environment is crucial to obtain excellent generalization ability for real-world machine learning systems. A typical setting across current meta learning algorithms assumes a stationary task distribution during meta training. In this paper, we explore a more practical and challenging setting where task distribution changes over time with domain shift. Particularly, we consider realistic scenarios where task distribution is highly imbalanced with domain labels unavailable in nature. We propose a kernel-based method for domain change detection and a difficulty-aware memory management mechanism that jointly considers the imbalanced domain size and domain importance to learn across domains continuously. Furthermore, we introduce an efficient adaptive task sampling method during meta training, which significantly reduces task gradient variance with theoretical guarantees. Finally, we propose a challenging benchmark with imbalanced domain sequences and varied domain difficulty. We have performed extensive evaluations on the proposed benchmark, demonstrating the effectiveness of our method.

count=25
* Detecting Anomalous Objects on Mobile Platforms
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.pdf)]
    * Title: Detecting Anomalous Objects on Mobile Platforms
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Wallace Lawson, Laura Hiatt, Keith Sullivan
    * Abstract: We present an approach where a robot patrols a fixed path through an environment, autonomously locating suspicious or anomalous objects. To learn, the robot patrols this environment building a dictionary describing what is present. The dictionary is built by clustering features from a deep neural network. The objects present vary depending on the scene, which means that an object that is anomalous in one scene may be completely normal in another. To reason about this, the robot uses a computational cognitive model to learn the dictionary elements that are typically found in each scene. Once the dictionary and model has been built, the robot can patrol the environment matching objects against the dictionary, and querying the model to find the most likely objects present and to determine which objects (if any) are anomalous. We demonstrate our approach by patrolling two indoor and one outdoor environments.

count=24
* Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Temporally_Consistent_Unbalanced_Optimal_Transport_for_Unsupervised_Action_Segmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Temporally_Consistent_Unbalanced_Optimal_Transport_for_Unsupervised_Action_Segmentation_CVPR_2024_paper.pdf)]
    * Title: Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ming Xu, Stephen Gould
    * Abstract: We propose a novel approach to the action segmentation task for long untrimmed videos based on solving an optimal transport problem. By encoding a temporal consistency prior into a Gromov-Wasserstein problem we are able to decode a temporally consistent segmentation from a noisy affinity/matching cost matrix between video frames and action classes. Unlike previous approaches our method does not require knowing the action order for a video to attain temporal consistency. Furthermore our resulting (fused) Gromov-Wasserstein problem can be efficiently solved on GPUs using a few iterations of projected mirror descent. We demonstrate the effectiveness of our method in an unsupervised learning setting where our method is used to generate pseudo-labels for self-training. We evaluate our segmentation approach and unsupervised learning pipeline on the Breakfast 50-Salads YouTube Instructions and Desktop Assembly datasets yielding state-of-the-art results for the unsupervised video action segmentation task.

count=18
* Geography-Aware Self-Supervised Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.pdf)]
    * Title: Geography-Aware Self-Supervised Learning
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell, Stefano Ermon
    * Abstract: Contrastive learning methods have significantly narrowed the gap between supervised and unsupervised learning on computer vision tasks. In this paper, we explore their application to geo-located datasets, e.g. remote sensing, where unlabeled data is often abundant but labeled data is scarce. We first show that due to their different characteristics, a non-trivial gap persists between contrastive and supervised learning on standard benchmarks. To close the gap, we propose novel training methods that exploit the spatio-temporal structure of remote sensing data. We leverage spatially aligned images over time to construct temporal positive pairs in contrastive learning and geo-location to design pre-text tasks. Our experiments show that our proposed method closes the gap between contrastive and supervised learning on image classification, object detection and semantic segmentation for remote sensing. Moreover, we demonstrate that the proposed method can also be applied to geo-tagged ImageNet images, improving downstream performance on various tasks.

count=17
* Space-Time Localization and Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Lee_Space-Time_Localization_and_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Space-Time_Localization_and_ICCV_2017_paper.pdf)]
    * Title: Space-Time Localization and Mapping
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Minhaeng Lee, Charless C. Fowlkes
    * Abstract: This paper addresses the problem of building a spatio-temporal model of the world from a stream of time-stamped data. Unlike traditional models for simultaneous localization and mapping (SLAM) and structure-from-motion (SfM) which focus on recovering a single rigid 3D model, we tackle the problem of mapping scenes in which dynamic components appear, move and disappear independently of each other over time. We introduce a simple generative probabilistic model of 4D structure which specifies location, spatial and temporal extent of rigid surface patches by local Gaussian mixtures. We fit this model to a time-stamped stream of input data using expectation-maximization to estimate the model structure parameters (mapping) and the alignment of the input data to the model (localization). By explicitly representing the temporal extent and observability of surfaces in a scene, our method yields superior localization and reconstruction relative to baselines that assume a static 3D scene. We carry out experiments on both synthetic RGB-D data streams as well as challenging real-world datasets, tracking scene dynamics in a human workspace over the course of several weeks.

count=16
* Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Warburg_Mapillary_Street-Level_Sequences_A_Dataset_for_Lifelong_Place_Recognition_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Warburg_Mapillary_Street-Level_Sequences_A_Dataset_for_Lifelong_Place_Recognition_CVPR_2020_paper.pdf)]
    * Title: Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Frederik Warburg,  Soren Hauberg,  Manuel Lopez-Antequera,  Pau Gargallo,  Yubin Kuang,  Javier Civera
    * Abstract: Lifelong place recognition is an essential and challenging task in computer vision with vast applications in robust localization and efficient large-scale 3D reconstruction. Progress is currently hindered by a lack of large, diverse, publicly available datasets. We contribute with Mapillary Street-Level Sequences (SLS), a large dataset for urban and suburban place recognition from image sequences. It contains more than 1.6 million images curated from the Mapillary collaborative mapping platform. The dataset is orders of magnitude larger than current data sources, and is designed to reflect the diversities of true lifelong learning. It features images from 30 major cities across six continents, hundreds of distinct cameras, and substantially different viewpoints and capture times, spanning all seasons over a nine year period. All images are geo-located with GPS and compass, and feature high-level attributes such as road type. We propose a set of benchmark tasks designed to push state-of-the-art performance and provide baseline studies. We show that current state-of-the-art methods still have a long way to go, and that the lack of diversity in existing datasets have prevented generalization to new environments. The dataset and benchmarks are available for academic research.

count=14
* Modeling Actions through State Changes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Fathi_Modeling_Actions_through_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Fathi_Modeling_Actions_through_2013_CVPR_paper.pdf)]
    * Title: Modeling Actions through State Changes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Alireza Fathi, James M. Rehg
    * Abstract: In this paper we present a model of action based on the change in the state of the environment. Many actions involve similar dynamics and hand-object relationships, but differ in their purpose and meaning. The key to differentiating these actions is the ability to identify how they change the state of objects and materials in the environment. We propose a weakly supervised method for learning the object and material states that are necessary for recognizing daily actions. Once these state detectors are learned, we can apply them to input videos and pool their outputs to detect actions. We further demonstrate that our method can be used to segment discrete actions from a continuous video of an activity. Our results outperform state-of-the-art action recognition and activity segmentation results.

count=14
* Word2Fun: Modelling Words as Functions for Diachronic Word Representation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/16a5cdae362b8d27a1d8f8c7b78b4330-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/16a5cdae362b8d27a1d8f8c7b78b4330-Paper.pdf)]
    * Title: Word2Fun: Modelling Words as Functions for Diachronic Word Representation
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Benyou Wang, Emanuele Di Buccio, Massimo Melucci
    * Abstract: Word meaning may change over time as a reflection of changes in human society. Therefore, modeling time in word representation is necessary for some diachronic tasks. Most existing diachronic word representation approaches train the embeddings separately for each pre-grouped time-stamped corpus and align these embeddings, e.g., by orthogonal projections, vector initialization, temporal referencing, and compass. However, not only does word meaning change in a short time, word meaning may also be subject to evolution over long timespans, thus resulting in a unified continuous process. A recent approach called `DiffTime' models semantic evolution as functions parameterized by multiple-layer nonlinear neural networks over time. In this paper, we will carry on this line of work by learning explicit functions over time for each word. Our approach, called `Word2Fun', reduces the space complexity from $\mathcal{O}(TVD)$ to $\mathcal{O}(kVD)$ where $k$ is a small constant ($k \ll T $). In particular, a specific instance based on polynomial functions could provably approximate any function modeling word evolution with a given negligible error thanks to the Weierstrass Approximation Theorem. The effectiveness of the proposed approach is evaluated in diverse tasks including time-aware word clustering, temporal analogy, and semantic change detection. Code at: {\url{https://github.com/wabyking/Word2Fun.git}}.

count=13
* Inverse Filtering for Hidden Markov Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf)]
    * Title: Inverse Filtering for Hidden Markov Models
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Robert Mattila, Cristian Rojas, Vikram Krishnamurthy, Bo Wahlberg
    * Abstract: This paper considers a number of related inverse filtering problems for hidden Markov models (HMMs). In particular, given a sequence of state posteriors and the system dynamics; i) estimate the corresponding sequence of observations, ii) estimate the observation likelihoods, and iii) jointly estimate the observation likelihoods and the observation sequence. We show how to avoid a computationally expensive mixed integer linear program (MILP) by exploiting the algebraic structure of the HMM filter using simple linear algebra operations, and provide conditions for when the quantities can be uniquely reconstructed. We also propose a solution to the more general case where the posteriors are noisily observed. Finally, the proposed inverse filtering algorithms are evaluated on real-world polysomnographic data used for automatic sleep segmentation.

count=12
* Multi-Source Multi-Modal Activity Recognition in Aerial Video Surveillance
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/html/Hammoud_Multi-Source_Multi-Modal_Activity_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/papers/Hammoud_Multi-Source_Multi-Modal_Activity_2014_CVPR_paper.pdf)]
    * Title: Multi-Source Multi-Modal Activity Recognition in Aerial Video Surveillance
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Riad I. Hammoud, Cem S. Sahin, Erik P. Blasch, Bradley J. Rhodes
    * Abstract: Recognizing activities in wide aerial/overhead imagery remains a challenging problem due in part to low-resolution video and cluttered scenes with a large number of moving objects. In the context of this research, we deal with two unsynchronized data sources collected in real-world operating scenarios: full-motion videos (FMV) and analyst call-outs (ACO) in the form of chat messages (voice-to-text) made by a human watching the streamed FMV from an aerial platform. We present a multi-source multi-modal activity/event recognition system for surveillance applications, consisting of: (1) detecting and tracking multiple dynamic targets from a moving platform, (2) representing FMV target tracks and chat messages as graphs of attributes, (3) associating FMV tracks and chat messages using a probabilistic graph-based matching approach, and (4) detecting spatial-temporal activity boundaries. We also present an activity pattern learning framework which uses the multi-source associated data as training to index a large archive of FMV videos. Finally, we describe a multi-intelligence user interface for querying an index of activities of interest (AOIs) by movement type and geo-location, and for playing-back a summary of associated text (ACO) and activity video segments of targets-of-interest (TOIs) (in both pixel and geo-coordinates). Such tools help the end-user to quickly search, browse, and prepare mission reports from multi-source data.

count=11
* Continuous Scene Representations for Embodied AI
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Gadre_Continuous_Scene_Representations_for_Embodied_AI_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Gadre_Continuous_Scene_Representations_for_Embodied_AI_CVPR_2022_paper.pdf)]
    * Title: Continuous Scene Representations for Embodied AI
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Samir Yitzhak Gadre, Kiana Ehsani, Shuran Song, Roozbeh Mottaghi
    * Abstract: We propose Continuous Scene Representations (CSR), a scene representation constructed by an embodied agent navigating within a space, where objects and their relationships are modeled by continuous valued embeddings. Our method captures feature relationships between objects, composes them into a graph structure on-the-fly, and situates an embodied agent within the representation. Our key insight is to embed pair-wise relationships between objects in a latent space. This allows for a richer representation compared to discrete relations (e.g., [support], [next-to]) commonly used for building scene representations. CSR can track objects as the agent moves in a scene, update the representation accordingly, and detect changes in room configurations. Using CSR, we outperform state-of-the-art approaches for the challenging downstream task of visual room rearrangement, without any task specific training. Moreover, we show the learned embeddings capture salient spatial details of the scene and show applicability to real world data. A summery video and code is available at https://prior.allenai.org/projects/csr.

count=11
* CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf)]
    * Title: CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Anthony Fuller, Koreen Millard, James Green
    * Abstract: A vital and rapidly growing application, remote sensing offers vast yet sparsely labeled, spatially aligned multimodal data; this makes self-supervised learning algorithms invaluable. We present CROMA: a framework that combines contrastive and reconstruction self-supervised objectives to learn rich unimodal and multimodal representations. Our method separately encodes masked-out multispectral optical and synthetic aperture radar samples—aligned in space and time—and performs cross-modal contrastive learning. Another encoder fuses these sensors, producing joint multimodal encodings that are used to predict the masked patches via a lightweight decoder. We show that these objectives are complementary when leveraged on spatially aligned multimodal data. We also introduce X- and 2D-ALiBi, which spatially biases our cross- and self-attention matrices. These strategies improve representations and allow our models to effectively extrapolate to images up to $17.6\times$ larger at test-time. CROMA outperforms the current SoTA multispectral model, evaluated on: four classification benchmarks—finetuning (avg.$\uparrow$ 1.8%), linear (avg.$\uparrow$ 2.4%) and nonlinear (avg.$\uparrow$ 1.4%) probing, $k$NN classification (avg.$\uparrow$ 3.5%), and $K$-means clustering (avg.$\uparrow$ 8.4%); and three segmentation benchmarks (avg.$\uparrow$ 6.4%). CROMA’s rich, optionally multimodal representations can be widely leveraged across remote sensing applications.

count=10
* Filmy Cloud Removal on Satellite Imagery With Multispectral Conditional Generative Adversarial Nets
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Enomoto_Filmy_Cloud_Removal_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Enomoto_Filmy_Cloud_Removal_CVPR_2017_paper.pdf)]
    * Title: Filmy Cloud Removal on Satellite Imagery With Multispectral Conditional Generative Adversarial Nets
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Kenji Enomoto, Ken Sakurada, Weimin Wang, Hiroshi Fukui, Masashi Matsuoka, Ryosuke Nakamura, Nobuo Kawaguchi
    * Abstract: This paper proposes a method for cloud removal from visible light RGB satellite images by extending the conditional Generative Adversarial Networks (cGANs) from RGB images to multispectral images. The networks are trained to output images that are close to the ground truth with the input images synthesized with clouds on the ground truth images. In the available dataset, the ratio of images of the forest or the sea is very high, which will cause bias of the training dataset if we uniformly sample from the dataset. Thus, we utilize the t-Distributed Stochastic Neighbor Embedding (t-SNE) to improve the bias problem of the training dataset. Finally, we confirm the feasibility of the proposed networks on the dataset of 4 bands images including three visible light bands and one near-infrared (NIR) band.

count=10
* Deep Anomaly Discovery From Unlabeled Videos via Normality Advantage and Self-Paced Refinement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.pdf)]
    * Title: Deep Anomaly Discovery From Unlabeled Videos via Normality Advantage and Self-Paced Refinement
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Guang Yu, Siqi Wang, Zhiping Cai, Xinwang Liu, Chuanfu Xu, Chengkun Wu
    * Abstract: While classic video anomaly detection (VAD) requires labeled normal videos for training, emerging unsupervised VAD (UVAD) aims to discover anomalies directly from fully unlabeled videos. However, existing UVAD methods still rely on shallow models to perform detection or initialization, and they are evidently inferior to classic VAD methods. This paper proposes a full deep neural network (DNN) based solution that can realize highly effective UVAD. First, we, for the first time, point out that deep reconstruction can be surprisingly effective for UVAD, which inspires us to unveil a property named "normality advantage", i.e., normal events will enjoy lower reconstruction loss when DNN learns to reconstruct unlabeled videos. With this property, we propose Localization based Reconstruction (LBR) as a strong UVAD baseline and a solid foundation of our solution. Second, we propose a novel self-paced refinement (SPR) scheme, which is synthesized into LBR to conduct UVAD. Unlike ordinary self-paced learning that injects more samples in an easy-to-hard manner, the proposed SPR scheme gradually drops samples so that suspicious anomalies can be removed from the learning process. In this way, SPR consolidates normality advantage and enables better UVAD in a more proactive way. Finally, we further design a variant solution that explicitly takes the motion cues into account. The solution evidently enhances the UVAD performance, and it sometimes even surpasses the best classic VAD methods. Experiments show that our solution not only significantly outperforms existing UVAD methods by a wide margin (5% to 9% AUROC), but also enables UVAD to catch up with the mainstream performance of classic VAD.

count=9
* Semantic-Aware Domain Generalized Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.pdf)]
    * Title: Semantic-Aware Domain Generalized Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Duo Peng, Yinjie Lei, Munawar Hayat, Yulan Guo, Wen Li
    * Abstract: Deep models trained on source domain lack generalization when evaluated on unseen target domains with different data distributions. The problem becomes even more pronounced when we have no access to target domain samples for adaptation. In this paper, we address domain generalized semantic segmentation, where a segmentation model is trained to be domain-invariant without using any target domain data. Existing approaches to tackle this problem standardize data into a unified distribution. We argue that while such a standardization promotes global normalization, the resulting features are not discriminative enough to get clear segmentation boundaries. To enhance separation between categories while simultaneously promoting domain invariance, we propose a framework including two novel modules: Semantic-Aware Normalization (SAN) and Semantic-Aware Whitening (SAW). Specifically, SAN focuses on category-level center alignment between features from different image styles, while SAW enforces distributed alignment for the already center-aligned features. With the help of SAN and SAW, we encourage both intraclass compactness and inter-class separability. We validate our approach through extensive experiments on widely-used datasets (i.e. GTAV, SYNTHIA, Cityscapes, Mapillary and BDDS). Our approach shows significant improvements over existing state-of-the-art on various backbone networks. Code is available at https://github.com/leolyj/SAN-SAW

count=9
* Meta-Learning for Few-Shot Land Cover Classification
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Russwurm_Meta-Learning_for_Few-Shot_Land_Cover_Classification_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Russwurm_Meta-Learning_for_Few-Shot_Land_Cover_Classification_CVPRW_2020_paper.pdf)]
    * Title: Meta-Learning for Few-Shot Land Cover Classification
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Marc Russwurm, Sherrie Wang, Marco Korner, David Lobell
    * Abstract: The representations of the Earth's surface vary from one geographic region to another. For instance, the appearance of urban areas differs between continents, and seasonality influences the appearance of vegetation. To capture the diversity within a single category, such as urban or vegetation, requires a large model capacity and, consequently, large datasets. In this work, we propose a different perspective and view this diversity as an inductive transfer learning problem where few data samples from one region allow a model to adapt to an unseen region. We evaluate the model-agnostic meta-learning (MAML) algorithm on classification and segmentation tasks using globally and regionally distributed datasets. We find that few-shot model adaptation outperforms pre-training with regular gradient descent and fine-tuning on the (1) Sen12MS dataset and (2) DeepGlobe dataset when the source domain and target domain differ. This indicates that model optimization with meta-learning may benefit tasks in the Earth sciences whose data show a high degree of diversity from region to region, while traditional gradient-based supervised learning remains suitable in the absence of a feature or label shift.

count=9
* On the Exploration of Local Significant Differences For Two-Sample Test
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/10fc83943b4540a9524af6fc67a23fef-Paper-Conference.pdf)]
    * Title: On the Exploration of Local Significant Differences For Two-Sample Test
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Zhijian Zhou, Jie Ni, Jia-He Yao, Wei Gao
    * Abstract: Recent years have witnessed increasing attentions on two-sample test with diverse real applications, while this work takes one more step on the exploration of local significant differences for two-sample test. We propose the ME$_\text{MaBiD}$, an effective test for two-sample testing, and the basic idea is to exploit local information by multiple Mahalanobis kernels and introduce bi-directional hypothesis for testing. On the exploration of local significant differences, we first partition the embedding space into several rectangle regions via a new splitting criterion, which is relevant to test power and data correlation. We then explore local significant differences based on our bi-directional masked $p$-value together with the ME$_\text{MaBiD}$ test. Theoretically, we present the asymptotic distribution and lower bounds of test power for our ME$_\text{MaBiD}$ test, and control the familywise error rate on the exploration of local significant differences. We finally conduct extensive experiments to validate the effectiveness of our proposed methods on two-sample test and the exploration of local significant differences.

count=7
* SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.pdf)]
    * Title: SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xin Guo, Jiangwei Lao, Bo Dang, Yingying Zhang, Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang, Kang Wu, Dingxiang Hu, Huimei He, Jian Wang, Jingdong Chen, Ming Yang, Yongjun Zhang, Yansheng Li
    * Abstract: Prior studies on Remote Sensing Foundation Model (RSFM) reveal immense potential towards a generic model for Earth Observation. Nevertheless these works primarily focus on a single modality without temporal and geo-context modeling hampering their capabilities for diverse tasks. In this study we present SkySense a generic billion-scale model pre-trained on a curated multi-modal Remote Sensing Imagery (RSI) dataset with 21.5 million temporal sequences. SkySense incorporates a factorized multi-modal spatiotemporal encoder taking temporal sequences of optical and Synthetic Aperture Radar (SAR) data as input. This encoder is pre-trained by our proposed Multi-Granularity Contrastive Learning to learn representations across different modal and spatial granularities. To further enhance the RSI representations by the geo-context clue we introduce Geo-Context Prototype Learning to learn region-aware prototypes upon RSI's multi-modal spatiotemporal features. To our best knowledge SkySense is the largest Multi-Modal RSFM to date whose modules can be flexibly combined or used individually to accommodate various tasks. It demonstrates remarkable generalization capabilities on a thorough evaluation encompassing 16 datasets over 7 tasks from single- to multi-modal static to temporal and classification to localization. SkySense surpasses 18 recent RSFMs in all test scenarios. Specifically it outperforms the latest models such as GFM SatLas and Scale-MAE by a large margin i.e. 2.76% 3.67% and 3.61% on average respectively. We will release the pre-trained weights to facilitate future research and Earth Observation applications.

count=7
* Tracing the Influence of Predecessors on Trajectory Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/ROAD%2B%2B/html/Liu_Tracing_the_Influence_of_Predecessors_on_Trajectory_Prediction_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/ROAD++/papers/Liu_Tracing_the_Influence_of_Predecessors_on_Trajectory_Prediction_ICCVW_2023_paper.pdf)]
    * Title: Tracing the Influence of Predecessors on Trajectory Prediction
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Mengmeng Liu, Hao Cheng, Michael Ying Yang
    * Abstract: In real-world traffic scenarios, agents such as pedestrians and car drivers often observe neighboring agents who exhibit similar behavior as examples and then mimic their actions to some extent in their own behavior. This information can serve as prior knowledge for trajectory prediction, which is unfortunately largely overlooked in current trajectory prediction models. This paper introduces a novel Predecessor-and-Successor (PnS) method that incorporates a predecessor tracing module to model the influence of predecessors (identified from concurrent neighboring agents) on the successor (target agent) within the same scene. The method utilizes the moving patterns of these predecessors to guide the predictor in trajectory prediction. PnS effectively aligns the motion encodings of the successor with multiple potential predecessors in a probabilistic manner, facilitating the decoding process. We demonstrate the effectiveness of PnS by integrating it into a graph-based predictor for pedestrian trajectory prediction on the ETH/UCY datasets, resulting in a new state-of-the-art performance. Furthermore, we replace the HD map-based scene-context module with our PnS method in a transformer-based predictor for vehicle trajectory prediction on the nuScenes dataset, showing that the predictor maintains good prediction performance even without relying on any map information.

count=6
* Temporal Action Segmentation From Timestamp Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Temporal_Action_Segmentation_From_Timestamp_Supervision_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Temporal_Action_Segmentation_From_Timestamp_Supervision_CVPR_2021_paper.pdf)]
    * Title: Temporal Action Segmentation From Timestamp Supervision
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Zhe Li, Yazan Abu Farha, Jurgen Gall
    * Abstract: Temporal action segmentation approaches have been very successful recently. However, annotating videos with frame-wise labels to train such models is very expensive and time consuming. While weakly supervised methods trained using only ordered action lists require less annotation effort, the performance is still worse than fully supervised approaches. In this paper, we propose to use timestamp supervision for the temporal action segmentation task. Timestamps require a comparable annotation effort to weakly supervised approaches, and yet provide a more supervisory signal. To demonstrate the effectiveness of timestamp supervision, we propose an approach to train a segmentation model using only timestamps annotations. Our approach uses the model output and the annotated timestamps to generate frame-wise labels by detecting the action changes. We further introduce a confidence loss that forces the predicted probabilities to monotonically decrease as the distance to the timestamps increases. This ensures that all and not only the most distinctive frames of an action are learned during training. The evaluation on four datasets shows that models trained with timestamps annotations achieve comparable performance to the fully supervised approaches.

count=6
* Extending Global-local View Alignment for Self-supervised Learning with Remote Sensing Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/WiCV/html/Wanyan_Extending_Global-local_View_Alignment_for_Self-supervised_Learning_with_Remote_Sensing_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/WiCV/papers/Wanyan_Extending_Global-local_View_Alignment_for_Self-supervised_Learning_with_Remote_Sensing_CVPRW_2024_paper.pdf)]
    * Title: Extending Global-local View Alignment for Self-supervised Learning with Remote Sensing Imagery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xinye Wanyan, Sachith Seneviratne, Shuchang Shen, Michael Kirley
    * Abstract: Since large number of high-quality remote sensing images are readily accessible exploiting the corpus of images with less manual annotation draws increasing attention. Self-supervised models acquire general feature representations by formulating a pretext task that generates pseudo-labels for massive unlabeled data to provide supervision for training. While prior studies have explored multiple self-supervised learning techniques in remote sensing domain pretext tasks based on local-global view alignment remain underexplored despite achieving state-of-the-art results on natural imagery. Inspired by DINO [6] which employs an effective representation learning structure with knowledge distillation based on global-local view alignment we formulate two pretext tasks for self-supervised learning on remote sensing imagery (SSLRS). Using these tasks we explore the effectiveness of positive temporal contrast as well as multi-sized views on SSLRS. We extend DINO and propose DINO-MC which uses local views of various sized crops instead of a single fixed size in order to alleviate the limited variation in object size observed in remote sensing imagery. Our experiments demonstrate that even when pre-trained on only 10% of the dataset DINO-MC performs on par or better than existing state-of-the-art SSLRS methods on multiple remote sensing tasks while using less computational resources. All codes models and results are released at https://github.com/WennyXY/DINO-MC.

count=6
* Robust Change Captioning
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Robust_Change_Captioning_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Robust_Change_Captioning_ICCV_2019_paper.pdf)]
    * Title: Robust Change Captioning
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Dong Huk Park,  Trevor Darrell,  Anna Rohrbach
    * Abstract: Describing what has changed in a scene can be useful to a user, but only if generated text focuses on what is semantically relevant. It is thus important to distinguish distractors (e.g. a viewpoint change) from relevant changes (e.g. an object has moved). We present a novel Dual Dynamic Attention Model (DUDA) to perform robust Change Captioning. Our model learns to distinguish distractors from semantic changes, localize the changes via Dual Attention over "before" and "after" images, and accurately describe them in natural language via Dynamic Speaker, by adaptively focusing on the necessary visual inputs (e.g. "before" or "after" image). To study the problem in depth, we collect a CLEVR-Change dataset, built off the CLEVR engine, with 5 types of scene changes. We benchmark a number of baselines on our dataset, and systematically study different change types and robustness to distractors. We show the superiority of our DUDA model in terms of both change captioning and localization. We also show that our approach is general, obtaining state-of-the-art results on the recent realistic Spot-the-Diff dataset which has no distractors.

count=6
* Dynamic Subtitles: A Multimodal Video Accessibility Enhancement Dedicated to Deaf and Hearing Impaired Users
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/ACVR/Tapu_Dynamic_Subtitles_A_Multimodal_Video_Accessibility_Enhancement_Dedicated_to_Deaf_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/ACVR/Tapu_Dynamic_Subtitles_A_Multimodal_Video_Accessibility_Enhancement_Dedicated_to_Deaf_ICCVW_2019_paper.pdf)]
    * Title: Dynamic Subtitles: A Multimodal Video Accessibility Enhancement Dedicated to Deaf and Hearing Impaired Users
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Ruxandra Tapu, Bogdan Mocanu, Titus Zaharia
    * Abstract: In this paper, we introduce a novel dynamic subtitle positioning system designed to increase the accessibility of the deaf and hearing impaired people to video documents. Our framework places the subtitle in the near vicinity of the active speaker in order to allow the viewer to follow the visual content while regarding the textual information. The proposed system is based on a multimodal fusion of text, audio and visual information in order to detect and recognize the identity of the active speaker. The experimental evaluation, performed on a large dataset of more than 30 videos, validates the methodology with average accuracy and recognition rates superior to 92%. The subjective evaluation demonstrates the effectiveness of our approach outperforming both conventional (static) subtitling and other state of the art techniques in terms of enhancement of the overall viewing experience and eyestrain reduction.

count=6
* Synthetic Examples Improve Generalization for Rare Classes
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Beery_Synthetic_Examples_Improve_Generalization_for_Rare_Classes_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Beery_Synthetic_Examples_Improve_Generalization_for_Rare_Classes_WACV_2020_paper.pdf)]
    * Title: Synthetic Examples Improve Generalization for Rare Classes
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Sara Beery,  Yang Liu,  Dan Morris,  Jim Piavis,  Ashish Kapoor,  Neel Joshi,  Markus Meister,  Pietro Perona
    * Abstract: The ability to detect and classify rare occurrences in images has important applications -- for example, counting rare and endangered species when studying biodiversity, or detecting infrequent traffic scenarios that pose a danger to self-driving cars. Few-shot learning is an open problem: current computer vision systems struggle to categorize objects they have seen only rarely during training, and collecting a sufficient number of training examples of rare events is often challenging and expensive, and sometimes outright impossible. We explore in depth an approach to this problem: complementing the few available training images with ad-hoc simulated data. Our testbed is animal species classification, which has a real-world long-tailed distribution. We analyze the effect of different axes of variation in simulation, such as pose, lighting, model, and simulation method, and we prescribe best practices for efficiently incorporating simulated data for real-world performance gain. Our experiments reveal that synthetic data can considerably reduce error rates for classes that are rare, that as the amount of simulated data is increased, accuracy on the target class improves, and that high variation of simulated data provides maximum performance gain.

count=6
* Implicit Neural Representation for Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.pdf)]
    * Title: Implicit Neural Representation for Change Detection
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Peter Naylor, Diego Di Carlo, Arianna Traviglia, Makoto Yamada, Marco Fiorucci
    * Abstract: Identifying changes in a pair of 3D aerial LiDAR point clouds, obtained during two distinct time periods over the same geographic region presents a significant challenge due to the disparities in spatial coverage and the presence of noise in the acquisition system. The most commonly used approaches to detecting changes in point clouds are based on supervised methods which necessitate extensive labelled data often unavailable in real-world applications. To address these issues, we propose an unsupervised approach that comprises two components: Implicit Neural Representation (INR) for continuous shape reconstruction and a Gaussian Mixture Model for categorising changes. INR offers a grid-agnostic representation for encoding bi-temporal point clouds, with unmatched spatial support that can be regularised to enhance high-frequency details and reduce noise. The reconstructions at each timestamp are compared at arbitrary spatial scales, leading to a significant increase in detection capabilities. We apply our method to a benchmark dataset comprising simulated LiDAR point clouds for urban sprawling. This dataset encompasses diverse challenging scenarios, varying in resolutions, input modalities and noise levels. This enables a comprehensive multi-scenario evaluation, comparing our method with the current state-of-the-art approach. We outperform the previous methods by a margin of 10% in the intersection over union metric. In addition, we put our techniques to practical use by applying them in a real-world scenario to identify instances of illicit excavation of archaeological sites and validate our results by comparing them with findings from field experts.

count=5
* StoryGraphs: Visualizing Character Interactions as a Timeline
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Tapaswi_StoryGraphs_Visualizing_Character_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Tapaswi_StoryGraphs_Visualizing_Character_2014_CVPR_paper.pdf)]
    * Title: StoryGraphs: Visualizing Character Interactions as a Timeline
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Makarand Tapaswi, Martin Bauml, Rainer Stiefelhagen
    * Abstract: We present a novel way to automatically summarize and represent the storyline of a TV episode by visualizing character interactions as a chart. We also propose a scene detection method that lends itself well to generate over-segmented scenes which is used to partition the video. The positioning of character lines in the chart is formulated as an optimization problem which trades between the aesthetics and functionality of the chart. Using automatic person identification, we present StoryGraphs for 3 diverse TV series encompassing a total of 22 episodes. We define quantitative criteria to evaluate StoryGraphs and also compare them against episode summaries to evaluate their ability to provide an overview of the episode.

count=5
* Geospatial Correspondences for Multimodal Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.pdf)]
    * Title: Geospatial Correspondences for Multimodal Registration
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Diego Marcos, Raffay Hamid, Devis Tuia
    * Abstract: The growing availability of very high resolution (<1 m/pixel) satellite and aerial images has opened up unprecedented opportunities to monitor and analyze the evolution of land-cover and land-use across the world. To do so, images of the same geographical areas acquired at different times and, potentially, with different sensors must be efficiently parsed to update maps and detect land-cover changes. However, a naive transfer of ground truth labels from one location in the source image to the corresponding location in the target image is not generally feasible, as these images are often only loosely registered (with up to +- 50m of non-uniform errors). Furthermore, land-cover changes in an area over time must be taken into account for an accurate ground truth transfer. To tackle these challenges, we propose a mid-level sensor-invariant representation that encodes image regions in terms of the spatial distribution of their spectral neighbors. We incorporate this representation in a Markov Random Field to simultaneously account for nonlinear mis-registrations and enforce locality priors to find matches between multi-sensor images. We show how our approach can be used to assist in several multimodal land-cover update and change detection problems.

count=5
* Moving Object Detection Under Discontinuous Change in Illumination Using Tensor Low-Rank and Invariant Sparse Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Shakeri_Moving_Object_Detection_Under_Discontinuous_Change_in_Illumination_Using_Tensor_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shakeri_Moving_Object_Detection_Under_Discontinuous_Change_in_Illumination_Using_Tensor_CVPR_2019_paper.pdf)]
    * Title: Moving Object Detection Under Discontinuous Change in Illumination Using Tensor Low-Rank and Invariant Sparse Decomposition
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Moein Shakeri,  Hong Zhang
    * Abstract: Although low-rank and sparse decomposition based methods have been successfully applied to the problem of moving object detection using structured sparsity-inducing norms, they are still vulnerable to significant illumination changes that arise in certain applications. We are interested in moving object detection in applications involving time-lapse image sequences for which current methods mistakenly group moving objects and illumination changes into foreground. Our method relies on the multilinear (tensor) data low-rank and sparse decomposition framework to address the weaknesses of existing methods. The key to our proposed method is to create first a set of prior maps that can characterize the changes in the image sequence due to illumination. We show that they can be detected by a k-support norm. To deal with concurrent, two types of changes, we employ two regularization terms, one for detecting moving objects and the other for accounting for illumination changes, in the tensor low-rank and sparse decomposition formulation. Through comprehensive experiments using challenging datasets, we show that our method demonstrates a remarkable ability to detect moving objects under discontinuous change in illumination, and outperforms the state-of-the-art solutions to this challenging problem.

count=5
* NestedVAE: Isolating Common Factors via Weak Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.pdf)]
    * Title: NestedVAE: Isolating Common Factors via Weak Supervision
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Matthew J. Vowels,  Necati Cihan Camgoz,  Richard Bowden
    * Abstract: Fair and unbiased machine learning is an important and active field of research, as decision processes are increasingly driven by models that learn from data. Unfortunately, any biases present in the data may be learned by the model, thereby inappropriately transferring that bias into the decision making process. We identify the connection between the task of bias reduction and that of isolating factors common between domains whilst encouraging domain specific invariance. To isolate the common factors we combine the theory of deep latent variable models with information bottleneck theory for scenarios whereby data may be naturally paired across domains and no additional supervision is required. The result is the Nested Variational AutoEncoder (NestedVAE). Two outer VAEs with shared weights attempt to reconstruct the input and infer a latent space, whilst a nested VAE attempts to reconstruct the latent representation of one image, from the latent representation of its paired image. In so doing, the nested VAE isolates the common latent factors/causes and becomes invariant to unwanted factors that are not shared between paired images. We also propose a new metric to provide a balanced method of evaluating consistency and classifier performance across domains which we refer to as the Adjusted Parity metric. An evaluation of NestedVAE on both domain and attribute invariance, change detection, and learning common factors for the prediction of biological sex demonstrates that NestedVAE significantly outperforms alternative methods.

count=5
* An Efficient Approach for Anomaly Detection in Traffic Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AICity/html/Doshi_An_Efficient_Approach_for_Anomaly_Detection_in_Traffic_Videos_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AICity/papers/Doshi_An_Efficient_Approach_for_Anomaly_Detection_in_Traffic_Videos_CVPRW_2021_paper.pdf)]
    * Title: An Efficient Approach for Anomaly Detection in Traffic Videos
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Keval Doshi, Yasin Yilmaz
    * Abstract: Due to its relevance in intelligent transportation systems, anomaly detection in traffic videos has recently received much interest. It remains a difficult problem due to a variety of factors influencing the video quality of a real-time traffic feed, such as temperature, perspective, lighting conditions, and so on. Even though state-of-the-art methods perform well on the available benchmark datasets, they need a large amount of external training data as well as substantial computational resources. In this paper, we propose an efficient approach for a video anomaly detection system which is capable of running at the edge devices, e.g., on a roadside camera. The proposed approach comprises a pre-processing module that detects changes in the scene and removes the corrupted frames, a two-stage background modelling module and a two-stage object detector. Finally, a backtracking anomaly detection algorithm computes a similarity statistic and decides on the onset time of the anomaly. We also propose a sequential change detection algorithm that can quickly adapt to a new scene and detect changes in the similarity statistic. Experimental results on the Track 4 test set of the 2021 AI City Challenge show the efficacy of the proposed framework as we achieve an F1-score of 0.9157 along with 8.4027 root mean square error (RMSE) and are ranked fourth in the competition.

count=5
* Augmentation Invariance and Adaptive Sampling in Semantic Segmentation of Agricultural Aerial Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/AgriVision/html/Tavera_Augmentation_Invariance_and_Adaptive_Sampling_in_Semantic_Segmentation_of_Agricultural_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/AgriVision/papers/Tavera_Augmentation_Invariance_and_Adaptive_Sampling_in_Semantic_Segmentation_of_Agricultural_CVPRW_2022_paper.pdf)]
    * Title: Augmentation Invariance and Adaptive Sampling in Semantic Segmentation of Agricultural Aerial Images
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Antonio Tavera, Edoardo Arnaudo, Carlo Masone, Barbara Caputo
    * Abstract: In this paper, we investigate the problem of Semantic Segmentation for agricultural aerial imagery. We observe that the existing methods used for this task are designed without considering two characteristics of the aerial data: (i) the top-down perspective implies that the model cannot rely on a fixed semantic structure of the scene, because the same scene may be experienced with different rotations of the sensor; (ii) there can be a strong imbalance in the distribution of semantic classes because the relevant objects of the scene may appear at extremely different scales (e.g., a field of crops and a small vehicle). We propose a solution to these problems based on two ideas: (i) we use together a set of suitable augmentation and a consistency loss to guide the model to learn semantic representations that are invariant to the photometric and geometric shifts typical of the top-down perspective (Augmentation Invariance); (ii) we use a sampling method (Adaptive Sampling) that selects the training images based on a measure of pixel-wise distribution of classes and actual network confidence. With an extensive set of experiments conducted on the Agriculture-Vision dataset, we demonstrate that our proposed strategies improve the performance of the current state-of-the-art method.

count=5
* Rescan: Inductive Instance Segmentation for Indoor RGBD Scans
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Halber_Rescan_Inductive_Instance_Segmentation_for_Indoor_RGBD_Scans_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Halber_Rescan_Inductive_Instance_Segmentation_for_Indoor_RGBD_Scans_ICCV_2019_paper.pdf)]
    * Title: Rescan: Inductive Instance Segmentation for Indoor RGBD Scans
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Maciej Halber,  Yifei Shi,  Kai Xu,  Thomas Funkhouser
    * Abstract: In depth-sensing applications ranging from home robotics to AR/VR, it will be common to acquire 3D scans of interior spaces repeatedly at sparse time intervals (e.g., as part of regular daily use). We propose an algorithm that analyzes these "rescans" to infer a temporal model of a scene with semantic instance information. Our algorithm operates inductively by using the temporal model resulting from past observations to infer an instance segmentation of a new scan, which is then used to update the temporal model. The model contains object instance associations across time and thus can be used to track individual objects, even though there are only sparse observations. During experiments with a new benchmark for the new task, our algorithm outperforms alternate approaches based on state-of-the-art networks for semantic instance segmentation.

count=5
* SSL4EO-L: Datasets and Foundation Models for Landsat Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/bbf7ee04e2aefec136ecf60e346c2e61-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/bbf7ee04e2aefec136ecf60e346c2e61-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: SSL4EO-L: Datasets and Foundation Models for Landsat Imagery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Adam Stewart, Nils Lehmann, Isaac Corley, Yi Wang, Yi-Chia Chang, Nassim Ait Ait Ali Braham, Shradha Sehgal, Caleb Robinson, Arindam Banerjee
    * Abstract: The Landsat program is the longest-running Earth observation program in history, with 50+ years of data acquisition by 8 satellites. The multispectral imagery captured by sensors onboard these satellites is critical for a wide range of scientific fields. Despite the increasing popularity of deep learning and remote sensing, the majority of researchers still use decision trees and random forests for Landsat image analysis due to the prevalence of small labeled datasets and lack of foundation models. In this paper, we introduce SSL4EO-L, the first ever dataset designed for Self-Supervised Learning for Earth Observation for the Landsat family of satellites (including 3 sensors and 2 product levels) and the largest Landsat dataset in history (5M image patches). Additionally, we modernize and re-release the L7 Irish and L8 Biome cloud detection datasets, and introduce the first ML benchmark datasets for Landsats 4–5 TM and Landsat 7 ETM+ SR. Finally, we pre-train the first foundation models for Landsat imagery using SSL4EO-L and evaluate their performance on multiple semantic segmentation tasks. All datasets and model weights are available via the TorchGeo library, making reproducibility and experimentation easy, and enabling scientific advancements in the burgeoning field of remote sensing for a multitude of downstream applications.

count=4
* Ensemble Video Object Cut in Highly Dynamic Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Ren_Ensemble_Video_Object_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ren_Ensemble_Video_Object_2013_CVPR_paper.pdf)]
    * Title: Ensemble Video Object Cut in Highly Dynamic Scenes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Xiaobo Ren, Tony X. Han, Zhihai He
    * Abstract: We consider video object cut as an ensemble of framelevel background-foreground object classifiers which fuses information across frames and refine their segmentation results in a collaborative and iterative manner. Our approach addresses the challenging issues of modeling of background with dynamic textures and segmentation of foreground objects from cluttered scenes. We construct patch-level bagof-words background models to effectively capture the background motion and texture dynamics. We propose a foreground salience graph (FSG) to characterize the similarity of an image patch to the bag-of-words background models in the temporal domain and to neighboring image patches in the spatial domain. We incorporate this similarity information into a graph-cut energy minimization framework for foreground object segmentation. The background-foreground classification results at neighboring frames are fused together to construct a foreground probability map to update the graph weights. The resulting object shapes at neighboring frames are also used as constraints to guide the energy minimization process during graph cut. Our extensive experimental results and performance comparisons over a diverse set of challenging videos with dynamic scenes, including the new Change Detection Challenge Dataset, demonstrate that the proposed ensemble video object cut method outperforms various state-ofthe-art algorithms.

count=4
* Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.pdf)]
    * Title: Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah
    * Abstract: Anomaly detection in video is a challenging computer vision problem. Due to the lack of anomalous events at training time, anomaly detection requires the design of learning methods without full supervision. In this paper, we approach anomalous event detection in video through self-supervised and multi-task learning at the object level. We first utilize a pre-trained detector to detect objects. Then, we train a 3D convolutional neural network to produce discriminative anomaly-specific information by jointly learning multiple proxy tasks: three self-supervised and one based on knowledge distillation. The self-supervised tasks are: (i) discrimination of forward/backward moving objects (arrow of time), (ii) discrimination of objects in consecutive/intermittent frames (motion irregularity) and (iii) reconstruction of object-specific appearance information. The knowledge distillation task takes into account both classification and detection information, generating large prediction discrepancies between teacher and student models when anomalies occur. To the best of our knowledge, we are the first to approach anomalous event detection in video as a multi-task learning problem, integrating multiple self-supervised and knowledge distillation proxy tasks in a single architecture. Our lightweight architecture outperforms the state-of-the-art methods on three benchmarks: Avenue, ShanghaiTech and UCSD Ped2. Additionally, we perform an ablation study demonstrating the importance of integrating self-supervised learning and normality-specific distillation in a multi-task learning setting.

count=4
* WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kumar_WildlifeMapper_Aerial_Image_Analysis_for_Multi-Species_Detection_and_Identification_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Kumar_WildlifeMapper_Aerial_Image_Analysis_for_Multi-Species_Detection_and_Identification_CVPR_2024_paper.pdf)]
    * Title: WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Satish Kumar, Bowen Zhang, Chandrakanth Gudavalli, Connor Levenson, Lacey Hughey, Jared A. Stabach, Irene Amoke, Gordon Ojwang, Joseph Mukeka, Stephen Mwiu, Joseph Ogutu, Howard Frederick, B.S. Manjunath
    * Abstract: We introduce WildlifeMapper (WM) a flexible model designed to detect locate and identify multiple species in aerial imagery. It addresses the limitations of traditional labor-intensive wildlife population assessments that are central to advancing environmental conservation efforts worldwide. While a number of methods exist to automate this process they are often limited in their ability to generalize to different species or landscapes due to the dominance of homogeneous backgrounds and/or poorly captured local image structures. WM introduces two novel modules that help to capture the local structure and context of objects of interest to accurately localize and identify them achieving a state-of-the-art (SOTA) detection rate of 0.56 mAP. Further we introduce a large aerial imagery dataset with more than 11k Images and 28k annotations verified by trained experts. WM also achieves SOTA performance on 3 other publicly available aerial survey datasets collected across 4 different countries improving mAP by 42%. Source code and trained models are available at Github

count=4
* Vehicle Re-identification with Learned Representation and Spatial Verification and Abnormality Detection with Multi-Adaptive Vehicle Detectors for Traffic Video Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/AI_City/Nguyen_Vehicle_Re-identification_with_Learned_Representation_and_Spatial_Verification_and_Abnormality_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI City/Nguyen_Vehicle_Re-identification_with_Learned_Representation_and_Spatial_Verification_and_Abnormality_CVPRW_2019_paper.pdf)]
    * Title: Vehicle Re-identification with Learned Representation and Spatial Verification and Abnormality Detection with Multi-Adaptive Vehicle Detectors for Traffic Video Analysis
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Khac-Tuan Nguyen,  Trung-Hieu Hoang,  Minh-Triet Tran,  Trung-Nghia Le,  Ngoc-Minh Bui,  Trong-Le Do,  Viet-Khoa Vo-Ho,  Quoc-An Luong,  Mai-Khiem Tran,  Thanh-An Nguyen,  Thanh-Dat Truong,  Vinh-Tiep Nguyen,  Minh N. Do
    * Abstract: Traffic flow analysis is essential for intelligent transportation systems. In this paper, we propose methods for two challenging problems in traffic flow analysis: vehicle re-identification and abnormal event detection. For the first problem, we propose to combine learned high-level features for vehicle instance representation with hand-crafted local features for spatial verification. For the second problem, we propose to use multiple adaptive vehicle detectors for anomaly proposal and use heuristics properties extracted from anomaly proposals to determine anomaly events. Experiments on the datasets of traffic flow analysis from AI City Challenge 2019 show that our methods achieve mAP of 0.4008 for vehicle re-identification in Track 2, and can detect abnormal events with very high accuracy (F1 = 0.9429) in Track 3.

count=4
* Seasonal Contrast: Unsupervised Pre-Training From Uncurated Remote Sensing Data
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.pdf)]
    * Title: Seasonal Contrast: Unsupervised Pre-Training From Uncurated Remote Sensing Data
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Oscar Mañas, Alexandre Lacoste, Xavier Giró-i-Nieto, David Vazquez, Pau Rodríguez
    * Abstract: Remote sensing and automatic earth monitoring are key to solve global-scale challenges such as disaster prevention, land use monitoring, or tackling climate change. Although there exist vast amounts of remote sensing data, most of it remains unlabeled and thus inaccessible for supervised learning algorithms. Transfer learning approaches can reduce the data requirements of deep learning algorithms. However, most of these methods are pre-trained on ImageNet and their generalization to remote sensing imagery is not guaranteed due to the domain gap. In this work, we propose Seasonal Contrast (SeCo), an effective pipeline to leverage unlabeled data for in-domain pre-training of remote sensing representations. The SeCo pipeline is composed of two parts. First, a principled procedure to gather large-scale, unlabeled and uncurated remote sensing datasets containing images from multiple Earth locations at different timestamps. Second, a self-supervised algorithm that takes advantage of time and position invariance to learn transferable representations for remote sensing applications. We empirically show that models trained with SeCo achieve better performance than their ImageNet pre-trained counterparts and state-of-the-art self-supervised learning methods on multiple downstream tasks. The datasets and models in SeCo will be made public to facilitate transfer learning and enable rapid progress in remote sensing applications.

count=4
* SiamSTA: Spatio-Temporal Attention Based Siamese Tracker for Tracking UAVs
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/html/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/papers/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.pdf)]
    * Title: SiamSTA: Spatio-Temporal Attention Based Siamese Tracker for Tracking UAVs
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Bo Huang, Junjie Chen, Tingfa Xu, Ying Wang, Shenwang Jiang, Yuncheng Wang, Lei Wang, Jianan Li
    * Abstract: With the growing threat of unmanned aerial vehicle (UAV) intrusion, anti-UAV techniques are becoming increasingly demanding. Object tracking, especially in thermal infrared (TIR) videos, though provides a promising solution, struggles with challenges like small scale and fast movement that commonly occur in anti-UAV scenarios. To mitigate this, we propose a simple yet effective spatio-temporal attention based Siamese network, dubbed SiamSTA, to track UAV robustly by performing reliable local tracking and wide-range re-detection alternatively. Concretely, tracking is carried out by posing spatial and temporal constraints on generating candidate proposals within local neighborhoods, hence eliminating background distractors to better perceive small targets. Complementarily, in case of target lost from local regions due to fast movement, a three-stage re-detection mechanism is introduced to re-detect targets from a global view by exploiting valuable motion cues through a correlation filter based on change detection. Finally, a state-aware switching policy is adopted to adaptively integrate local tracking and global re-detection and take their complementary strengths for robust tracking. Extensive experiments on the 1st and 2nd anti-UAV datasets well demonstrate the superiority of SiamSTA over other competing counterparts. Notably, SiamSTA is the foundation of the 1st-place winning entry in the 2nd Anti-UAV Challenge.

count=4
* 3D Semantic Label Transfer in Human-Robot Collaboration
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/html/Rozenberszki_3D_Semantic_Label_Transfer_in_Human-Robot_Collaboration_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/papers/Rozenberszki_3D_Semantic_Label_Transfer_in_Human-Robot_Collaboration_ICCVW_2021_paper.pdf)]
    * Title: 3D Semantic Label Transfer in Human-Robot Collaboration
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Dávid Rozenberszki, Gábor Sörös, Szilvia Szeier, András Lőrincz
    * Abstract: We tackle two practical problems in robotic scene understanding. First, the computational requirements of current semantic segmentation algorithms are prohibitive for typical robots. Second, the viewpoints of ground robots are quite different from typical human viewpoints of training datasets which may lead to misclassified objects from robot viewpoints. We present a system for sharing and reusing 3D semantic information between multiple agents with different viewpoints. We first co-localize all agents in the same coordinate system. Next, we create a 3D dense semantic model of the space from human viewpoints close to real time. Finally, by re-rendering the model's semantic labels (and/or depth maps) from the ground robots' own estimated viewpoints and sharing them over the network, we can give 3D semantic understanding to simpler agents. We evaluate the reconstruction quality and show how tiny robots can reuse knowledge about the space collected by more capable peers.

count=4
* Learned Event-Based Visual Perception for Improved Space Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Salvatore_Learned_Event-Based_Visual_Perception_for_Improved_Space_Object_Detection_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Salvatore_Learned_Event-Based_Visual_Perception_for_Improved_Space_Object_Detection_WACV_2022_paper.pdf)]
    * Title: Learned Event-Based Visual Perception for Improved Space Object Detection
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Nikolaus Salvatore, Justin Fletcher
    * Abstract: The detection of dim artificial Earth satellites using ground-based electro-optical sensors, particularly in the presence of background light, is technologically challenging. This perceptual task is foundational to our understanding of the space environment, and grows in importance as the number, variety, and dynamism of space objects increases. We present a hybrid image- and event-based architecture that leverages dynamic vision sensing technology to detect resident space objects in geosynchronous Earth orbit. Given the asynchronous, one-dimensional image data supplied by a dynamic vision sensor, our architecture applies conventional image feature extractors to integrated, two-dimensional frames in conjunction with point-cloud feature extractors, such as PointNet, in order to increase detection performance for dim objects in scenes with high background activity. In addition, an end-to-end event-based imaging simulator is developed to both produce data for model training as well as approximate the optimal sensor parameters for event-based sensing in the context of electro-optical telescope imagery. Experimental results confirm that the inclusion of point-cloud feature extractors increases recall for dim objects in the high-background regime.

count=4
* Semantic Segmentation in Aerial Imagery Using Multi-Level Contrastive Learning With Local Consistency
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Tang_Semantic_Segmentation_in_Aerial_Imagery_Using_Multi-Level_Contrastive_Learning_With_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Tang_Semantic_Segmentation_in_Aerial_Imagery_Using_Multi-Level_Contrastive_Learning_With_WACV_2023_paper.pdf)]
    * Title: Semantic Segmentation in Aerial Imagery Using Multi-Level Contrastive Learning With Local Consistency
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Maofeng Tang, Konstantinos Georgiou, Hairong Qi, Cody Champion, Marc Bosch
    * Abstract: Semantic segmentation in large-scale aerial images is an extremely challenging task. On one hand, the limited ground truth, as compared to the vast area the images cover, greatly hinders the development of supervised representation learning. On the other hand, the large footprint from remote sensing raises new challenges for semantic segmentation. In addition, the complex and ever changing image acquisition conditions further complicate the problem where domain shifting commonly occurs. In this paper, we exploit self-supervised contrastive learning (CL) methodologies for semantic segmentation in aerial imagery. In addition to performing CL at the feature level as most practices do, we add another level of contrastive learning, at the semantic level, taking advantage of the segmentation output from the downstream task. Further, we embed local mutual information in the semantic-level CL to enforce local consistency. This has largely enhanced the representation power at each pixel and improved the generalization capacity of the trained model. We refer to the proposed approach as multi-level contrastive learning with local consistency (mCL-LC). The experimental results on different benchmarks indicate that the proposed mCL-LC exhibits superior performance as compared to other state-of-the-art contrastive learning frameworks for the semantic segmentation task. mCL-LC also carries better generalization capacity especially when domain shifting exists.

count=4
* Do VSR Models Generalize Beyond LRS3?
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Djilali_Do_VSR_Models_Generalize_Beyond_LRS3_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Djilali_Do_VSR_Models_Generalize_Beyond_LRS3_WACV_2024_paper.pdf)]
    * Title: Do VSR Models Generalize Beyond LRS3?
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Yasser Abdelaziz Dahou Djilali, Sanath Narayan, Eustache LeBihan, Haithem Boussaid, Ebtesam Almazrouei, Merouane Debbah
    * Abstract: The Lip Reading Sentences-3 (LRS3) benchmark has primarily been the focus of intense research in visual speech recognition (VSR) during the last few years. As a result, there is an increased risk of overfitting to its excessively used test set, which is only one hour duration. To alleviate this issue, we build a new VSR test set by closely following the LRS3 dataset creation processes. We then evaluate and analyse the extent to which the current VSR models generalize to the new test data. We evaluate a broad range of publicly available VSR models and find significant drops in performance on our test set, compared to their corresponding LRS3 results. Our results suggest that the increase in word error rates is caused by the models' inability to generalize to slightly "harder" and more realistic lip sequences than those found in the LRS3 test set. Our new test benchmark will be made public in order to enable future research towards more robust VSR models.

count=4
* Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/b01153e7112b347d8ed54f317840d8af-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Utkarsh Mall, Bharath Hariharan, Kavita Bala
    * Abstract: Satellite imagery is increasingly available, high resolution, and temporally detailed. Changes in spatio-temporal datasets such as satellite images are particularly interesting as they reveal the many events and forces that shape our world. However, finding such interesting and meaningful change events from the vast data is challenging. In this paper, we present new datasets for such change events that include semantically meaningful events like road construction. Instead of manually annotating the very large corpus of satellite images, we introduce a novel unsupervised approach that takes a large spatio-temporal dataset from satellite images and finds interesting change events. To evaluate the meaningfulness on these datasets we create 2 benchmarks namely CaiRoad and CalFire which capture the events of road construction and forest fires. These new benchmarks can be used to evaluate semantic retrieval/classification performance. We explore these benchmarks qualitatively and quantitatively by using several methods and show that these new datasets are indeed challenging for many existing methods.

count=3
* EFEM: Equivariant Neural Field Expectation Maximization for 3D Object Segmentation Without Scene Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lei_EFEM_Equivariant_Neural_Field_Expectation_Maximization_for_3D_Object_Segmentation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Lei_EFEM_Equivariant_Neural_Field_Expectation_Maximization_for_3D_Object_Segmentation_CVPR_2023_paper.pdf)]
    * Title: EFEM: Equivariant Neural Field Expectation Maximization for 3D Object Segmentation Without Scene Supervision
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jiahui Lei, Congyue Deng, Karl Schmeckpeper, Leonidas Guibas, Kostas Daniilidis
    * Abstract: We introduce Equivariant Neural Field Expectation Maximization (EFEM), a simple, effective, and robust geometric algorithm that can segment objects in 3D scenes without annotations or training on scenes. We achieve such unsupervised segmentation by exploiting single object shape priors. We make two novel steps in that direction. First, we introduce equivariant shape representations to this problem to eliminate the complexity induced by the variation in object configuration. Second, we propose a novel EM algorithm that can iteratively refine segmentation masks using the equivariant shape prior. We collect a novel real dataset Chairs and Mugs that contains various object configurations and novel scenes in order to verify the effectiveness and robustness of our method. Experimental results demonstrate that our method achieves consistent and robust performance across different scenes where the (weakly) supervised methods may fail. Code and data available at https://www.cis.upenn.edu/ leijh/projects/efem

count=3
* Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.pdf)]
    * Title: Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Nicolae-C?t?lin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah
    * Abstract: We propose an efficient abnormal event detection model based on a lightweight masked auto-encoder (AE) applied at the video frame level. The novelty of the proposed model is threefold. First we introduce an approach to weight tokens based on motion gradients thus shifting the focus from the static background scene to the foreground objects. Second we integrate a teacher decoder and a student decoder into our architecture leveraging the discrepancy between the outputs given by the two decoders to improve anomaly detection. Third we generate synthetic abnormal events to augment the training videos and task the masked AE model to jointly reconstruct the original frames (without anomalies) and the corresponding pixel-level anomaly maps. Our design leads to an efficient and effective model as demonstrated by the extensive experiments carried out on four benchmarks: Avenue ShanghaiTech UBnormal and UCSD Ped2. The empirical results show that our model achieves an excellent trade-off between speed and accuracy obtaining competitive AUC scores while processing 1655 FPS. Hence our model is between 8 and 70 times faster than competing methods. We also conduct an ablation study to justify our design. Our code is freely available at: https://github.com/ristea/aed-mae.

count=3
* A Temporal Scheme for Fast Learning of Image-Patch Correspondences in Realistic Multi-camera Setups
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W17/html/Eisenbach_A_Temporal_Scheme_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W17/papers/Eisenbach_A_Temporal_Scheme_2013_CVPR_paper.pdf)]
    * Title: A Temporal Scheme for Fast Learning of Image-Patch Correspondences in Realistic Multi-camera Setups
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Jens Eisenbach, Christian Conrad, Rudolf Mester
    * Abstract: This paper addresses the problem of finding corresponding image patches in multi-camera video streams by means of an unsupervised learning method. We determine patchto-patch correspondence relations ('correspondence priors') merely using information from a temporal change detection. Correspondence priors are essential for geometric multi-camera calibration, but are useful also for further vision tasks such as object tracking and recognition. Since any change detection method with reasonably performance can be applied, our method can be used as an encapsulated processing module and be integrated into existing systems without major structural changes. The only assumption that is made is that relative orientation of pairs of cameras may be arbitrary, but fixed, and that the observed scene shows visual activity. Experimental results show the applicability of the presented approach in real world scenarios where the camera views show large differences in orientation and position. Furthermore we show that a classic spatial matching pipeline, e.g., based on SIFT will typically fail to determine correspondences in these kinds of scenarios.

count=3
* The Visual Object Tracking VOT2017 Challenge Results
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w28/html/Kristan_The_Visual_Object_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Kristan_The_Visual_Object_ICCV_2017_paper.pdf)]
    * Title: The Visual Object Tracking VOT2017 Challenge Results
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pflugfelder, Luka Cehovin Zajc, Tomas Vojir, Gustav Hager, Alan Lukezic, Abdelrahman Eldesokey, Gustavo Fernandez
    * Abstract: The Visual Object Tracking challenge VOT2017 is the fifth annual tracker benchmarking activity organized by the VOT initiative. Results of 51 trackers are presented; many are state-of-the-art published at major computer vision conferences or journals in recent years. The evaluation included the standard VOT and other popular methodologies and a new "real-time" experiment simulating a situation where a tracker processes images as if provided by a continuously running sensor. Performance of the tested trackers typically by far exceeds standard baselines. The source code for most of the trackers is publicly available from the VOT page. The VOT2017 goes beyond its predecessors by (i) improving the VOT public dataset and introducing a separate VOT2017 sequestered dataset, (ii) introducing a real-time tracking experiment and (iii) releasing a redesigned toolkit that supports complex experiments. The dataset, the evaluation kit and the results are publicly available at the challenge w ....

count=3
* Finding Time Together: Detection and Classification of Focused Interaction in Egocentric Video
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w34/html/individuals_s.banodundee.ac.uk_s.j.z.mckennadundee.ac.uk_j.n.zhangdundee.ac.uk_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w34/individuals_s.banodundee.ac.uk_s.j.z.mckennadundee.ac.uk_j.n.zhangdundee.ac.uk_ICCV_2017_paper.pdf)]
    * Title: Finding Time Together: Detection and Classification of Focused Interaction in Egocentric Video
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Sophia Bano, Stephen J. McKenna, Jianguo Zhang
    * Abstract: Focused interaction occurs when co-present individuals, having mutual focus of attention, interact by establishing face-to-face engagement and direct conversation. Face-to-face engagement is often not maintained throughout the entirety of a focused interaction. In this paper, we present an online method for automatic classification of unconstrained egocentric (first-person perspective) videos into segments having no focused interaction, focused interaction when the camera wearer is stationary and focused interaction when the camera wearer is moving. We extract features from both audio and video data streams and perform temporal segmentation by using support vector machines with linear and non-linear kernels. We provide empirical evidence that fusion of visual face track scores, camera motion profile and audio voice activity scores is an effective combination for focused interaction classification.

count=3
* An Anomaly Detection System via Moving Surveillance Robots With Human Collaboration
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/html/Zaheer_An_Anomaly_Detection_System_via_Moving_Surveillance_Robots_With_Human_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/papers/Zaheer_An_Anomaly_Detection_System_via_Moving_Surveillance_Robots_With_Human_ICCVW_2021_paper.pdf)]
    * Title: An Anomaly Detection System via Moving Surveillance Robots With Human Collaboration
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Muhammad Zaigham Zaheer, Arif Mahmood, M. Haris Khan, Marcella Astrid, Seung-Ik Lee
    * Abstract: Autonomous anomaly detection is a fundamental step in visual surveillance systems, and so we have witnessed great progress in the form of various promising algorithms. Nonetheless, majority of prior algorithms assume static surveillance cameras that severely restricts the coverage of the system unless the number of cameras is exponentially increased, consequently increasing both the installation and monitoring costs. In the current work we propose an anomaly detection system based on mobile surveillance cameras, i.e., moving robot which continuously navigates a target area. We compare the newly acquired test images with a database of normal images using geo-tags. For anomaly detection, a Siamese network is trained which analyses two input images for anomalies while ignoring the viewpoint differences. Further, our system is capable of updating the normal images database with human collaboration. Finally, we propose a new dataset that is captured by repeated visits of the robot over a target area. Our experiments demonstrate the effectiveness of the proposed system for anomaly detection using mobile surveillance robots.

count=3
* An Interactive Method for Adaptive Acquisition in Reflectance Transformation Imaging for Cultural Heritage
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/e-Heritage/html/Khawaja_An_Interactive_Method_for_Adaptive_Acquisition_in_Reflectance_Transformation_Imaging_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/e-Heritage/papers/Khawaja_An_Interactive_Method_for_Adaptive_Acquisition_in_Reflectance_Transformation_Imaging_ICCVW_2023_paper.pdf)]
    * Title: An Interactive Method for Adaptive Acquisition in Reflectance Transformation Imaging for Cultural Heritage
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Muhammad Arsalan Khawaja, Sony George, Franck Marzani, Jon Yngve Hardeberg, Alamin Mansouri
    * Abstract: This paper investigates the optimization of acquisition in Reflectance Transformation Imaging (RTI). Current methods for RTI acquisition are either computationally expensive or impractical, which leads to continued reliance on conventional classical methods like homogenous equally spaced methods in museums. We propose a methodology that is aimed at dynamic collaboration between automated analysis and cultural heritage expert knowledge to obtain optimized light positions. Our approach is cost-effective and adaptive to both linear and non-linear reflectance profile scenarios. The practical contribution of research in this field has a considerable impact on the cultural heritage context and beyond.

count=3
* The Seventh Visual Object Tracking VOT2019 Challenge Results
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/VOT/Kristan_The_Seventh_Visual_Object_Tracking_VOT2019_Challenge_Results_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Kristan_The_Seventh_Visual_Object_Tracking_VOT2019_Challenge_Results_ICCVW_2019_paper.pdf)]
    * Title: The Seventh Visual Object Tracking VOT2019 Challenge Results
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Matej Kristan, Jiri Matas, Ales Leonardis, Michael Felsberg, Roman Pflugfelder, Joni-Kristian Kamarainen, Luka Cehovin Zajc, Ondrej Drbohlav, Alan Lukezic, Amanda Berg, Abdelrahman Eldesokey, Jani Kapyla, Gustavo Fernandez, Abel Gonzalez-Garcia, Alireza Memarmoghadam, Andong Lu, Anfeng He, Anton Varfolomieiev, Antoni Chan, Ardhendu Shekhar Tripathi, Arnold Smeulders, Bala Suraj Pedasingu, Bao Xin Chen, Baopeng Zhang, Baoyuan Wu, Bi Li, Bin He, Bin Yan, Bing Bai, Bing Li, Bo Li, Byeong Hak Kim, Byeong Hak Ki
    * Abstract: The Visual Object Tracking challenge VOT2019 is the seventh annual tracker benchmarking activity organized by the VOT initiative. Results of 81 trackers are presented; many are state-of-the-art trackers published at major computer vision conferences or in journals in the recent years. The evaluation included the standard VOT and other popular methodologies for short-term tracking analysis as well as the standard VOT methodology for long-term tracking analysis. The VOT2019 challenge was composed of five challenges focusing on different tracking domains: (i) VOTST2019 challenge focused on short-term tracking in RGB, (ii) VOT-RT2019 challenge focused on "real-time" shortterm tracking in RGB, (iii) VOT-LT2019 focused on longterm tracking namely coping with target disappearance and reappearance. Two new challenges have been introduced: (iv) VOT-RGBT2019 challenge focused on short-term tracking in RGB and thermal imagery and (v) VOT-RGBD2019 challenge focused on long-term tracking in RGB and depth imagery. The VOT-ST2019, VOT-RT2019 and VOT-LT2019 datasets were refreshed while new datasets were introduced for VOT-RGBT2019 and VOT-RGBD2019. The VOT toolkit has been updated to support both standard shortterm, long-term tracking and tracking with multi-channel imagery. Performance of the tested trackers typically by far exceeds standard baselines. The source code for most of the trackers is publicly available from the VOT page. The dataset, the evaluation kit and the results are publicly available at the challenge website.

count=3
* Structure Learning with Side Information: Sample Complexity
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/e025b6279c1b88d3ec0eca6fcb6e6280-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/e025b6279c1b88d3ec0eca6fcb6e6280-Paper.pdf)]
    * Title: Structure Learning with Side Information: Sample Complexity
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Saurabh Sihag, Ali Tajer
    * Abstract: Graphical models encode the stochastic dependencies among random variables (RVs). The vertices represent the RVs, and the edges signify the conditional dependencies among the RVs. Structure learning is the process of inferring the edges by observing realizations of the RVs, and it has applications in a wide range of technological, social, and biological networks. Learning the structure of graphs when the vertices are treated in isolation from inferential information known about them is well-investigated. In a wide range of domains, however, often there exist additional inferred knowledge about the structure, which can serve as valuable side information. For instance, the gene networks that represent different subtypes of the same cancer share similar edges across all subtypes and also have exclusive edges corresponding to each subtype, rendering partially similar graphical models for gene expression in different cancer subtypes. Hence, an inferential decision regarding a gene network can serve as side information for inferring other related gene networks. When such side information is leveraged judiciously, it can translate to significant improvement in structure learning. Leveraging such side information can be abstracted as inferring structures of distinct graphical models that are {\sl partially} similar. This paper focuses on Ising graphical models, and considers the problem of simultaneously learning the structures of two {\sl partially} similar graphs, where any inference about the structure of one graph offers side information for the other graph. The bounded edge subclass of Ising models is considered, and necessary conditions (information-theoretic ), as well as sufficient conditions (algorithmic) for the sample complexity for achieving a bounded probability of error, are established. Furthermore, specific regimes are identified in which the necessary and sufficient conditions coincide, rendering the optimal sample complexity.

count=3
* Learning the Structure of Large Networked Systems Obeying Conservation Laws
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/5e0347e19c51cfd0f6fe52f371004dfc-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/5e0347e19c51cfd0f6fe52f371004dfc-Paper-Conference.pdf)]
    * Title: Learning the Structure of Large Networked Systems Obeying Conservation Laws
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Anirudh Rayas, Rajasekhar Anguluri, Gautam Dasarathy
    * Abstract: Many networked systems such as electric networks, the brain, and social networks of opinion dynamics are known to obey conservation laws. Examples of this phenomenon include the Kirchoff laws in electric networks and opinion consensus in social networks. Conservation laws in networked systems are modeled as balance equations of the form $X = B^\ast Y$, where the sparsity pattern of $B^\ast \in \mathbb{R}^{p\times p}$ captures the connectivity of the network on $p$ nodes, and $Y, X \in \mathbb{R}^p$ are vectors of ''potentials'' and ''injected flows'' at the nodes respectively. The node potentials $Y$ cause flows across edges which aim to balance out the potential difference, and the flows $X$ injected at the nodes are extraneous to the network dynamics. In several practical systems, the network structure is often unknown and needs to be estimated from data to facilitate modeling, management, and control. To this end, one has access to samples of the node potentials $Y$, but only the statistics of the node injections $X$. Motivated by this important problem, we study the estimation of the sparsity structure of the matrix $B^\ast$ from $n$ samples of $Y$ under the assumption that the node injections $X$ follow a Gaussian distribution with a known covariance $\Sigma_X$. We propose a new $\ell_{1}$-regularized maximum likelihood estimator for tackling this problem in the high-dimensional regime where the size of the network may be vastly larger than the number of samples $n$. We show that this optimization problem is convex in the objective and admits a unique solution. Under a new mutual incoherence condition, we establish sufficient conditions on the triple $(n,p,d)$ for which exact sparsity recovery of $B^\ast$ is possible with high probability; $d$ is the degree of the underlying graph. We also establish guarantees for the recovery of $B^\ast$ in the element-wise maximum, Frobenius, and operator norms. Finally, we complement these theoretical results with experimental validation of the performance of the proposed estimator on synthetic and real-world data.

count=3
* FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/353ca88f722cdd0c481b999428ae113a-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/353ca88f722cdd0c481b999428ae113a-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Anatol Garioud, Nicolas Gonthier, Loic Landrieu, Apolline De Wit, Marion Valette, Marc Poupée, Sebastien Giordano, boris Wattrelos
    * Abstract: We introduce the French Land cover from Aerospace ImageRy (FLAIR), an extensive dataset from the French National Institute of Geographical and Forest Information (IGN) that provides a unique and rich resource for large-scale geospatial analysis. FLAIR contains high-resolution aerial imagery with a ground sample distance of 20 cm and over 20 billion individually labeled pixels for precise land-cover classification. The dataset also integrates temporal and spectral data from optical satellite time series. FLAIR thus combines data with varying spatial, spectral, and temporal resolutions across over 817 km² of acquisitions representing the full landscape diversity of France. This diversity makes FLAIR a valuable resource for the development and evaluation of novel methods for large-scale land-cover semantic segmentation and raises significant challenges in terms of computer vision, data fusion, and geospatial analysis. We also provide powerful uni- and multi-sensor baseline models that can be employed to assess algorithm's performance and for downstream applications.

count=3
* On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/57587d8d6a7ede0e5302fc22d0878c53-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/57587d8d6a7ede0e5302fc22d0878c53-Paper-Conference.pdf)]
    * Title: On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Qi CHEN, Changjian Shui, Ligong Han, Mario Marchand
    * Abstract: We focus on Continual Meta-Learning (CML), which targets accumulating and exploiting meta-knowledge on a sequence of non-i.i.d. tasks. The primary challenge is to strike a balance between stability and plasticity, where a model should be stable to avoid catastrophic forgetting in previous tasks and plastic to learn generalizable concepts from new tasks. To address this, we formulate the CML objective as controlling the average excess risk upper bound of the task sequence, which reflects the trade-off between forgetting and generalization. Based on the objective, we introduce a unified theoretical framework for CML in both static and shifting environments, providing guarantees for various task-specific learning algorithms. Moreover, we first present a rigorous analysis of a bi-level trade-off in shifting environments. To approach the optimal trade-off, we propose a novel algorithm that dynamically adjusts the meta-parameter and its learning rate w.r.t environment change. Empirical evaluations on synthetic and real datasets illustrate the effectiveness of the proposed theory and algorithm.

count=3
* GEO-Bench: Toward Foundation Models for Earth Monitoring
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/a0644215d9cff6646fa334dfa5d29c5a-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: GEO-Bench: Toward Foundation Models for Earth Monitoring
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Björn Lütjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, Mehmet Gunturkun, Gabriel Huang, David Vazquez, Dava Newman, Yoshua Bengio, Stefano Ermon, Xiaoxiang Zhu
    * Abstract: Recent progress in self-supervision has shown that pre-training large neural networks on vast amounts of unsupervised data can lead to substantial increases in generalization to downstream tasks. Such models, recently coined foundation models, have been transformational to the field of natural language processing.Variants have also been proposed for image data, but their applicability to remote sensing tasks is limited.To stimulate the development of foundation models for Earth monitoring, we propose a benchmark comprised of six classification and six segmentation tasks, which were carefully curated and adapted to be both relevant to the field and well-suited for model evaluation. We accompany this benchmark with a robust methodology for evaluating models and reporting aggregated results to enable a reliable assessment of progress. Finally, we report results for 20 baselines to gain information about the performance of existing models.We believe that this benchmark will be a driver of progress across a variety of Earth monitoring tasks.

count=3
* CityRefer: Geography-aware 3D Visual Grounding Dataset on  City-scale Point Cloud Data
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/f4cef76305dcad4efd3537da087ff520-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/f4cef76305dcad4efd3537da087ff520-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: CityRefer: Geography-aware 3D Visual Grounding Dataset on  City-scale Point Cloud Data
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Taiki Miyanishi, Fumiya Kitamori, Shuhei Kurita, Jungdae Lee, Motoaki Kawanabe, Nakamasa Inoue
    * Abstract: City-scale 3D point cloud is a promising way to express detailed and complicated outdoor structures. It encompasses both the appearance and geometry features of segmented city components, including cars, streets, and buildings that can be utilized for attractive applications such as user-interactive navigation of autonomous vehicles and drones. However, compared to the extensive text annotations available for images and indoor scenes, the scarcity of text annotations for outdoor scenes poses a significant challenge for achieving these applications. To tackle this problem, we introduce the CityRefer dataset for city-level visual grounding. The dataset consists of 35k natural language descriptions of 3D objects appearing in SensatUrban city scenes and 5k landmarks labels synchronizing with OpenStreetMap. To ensure the quality and accuracy of the dataset, all descriptions and labels in the CityRefer dataset are manually verified. We also have developed a baseline system that can learn encoded language descriptions, 3D object instances, and geographical information about the city's landmarks to perform visual grounding on the CityRefer dataset. To the best of our knowledge, the CityRefer dataset is the largest city-level visual grounding dataset for localizing specific 3D objects.

count=2
* Human vs. Computer in Scene and Object Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Borji_Human_vs._Computer_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Borji_Human_vs._Computer_2014_CVPR_paper.pdf)]
    * Title: Human vs. Computer in Scene and Object Recognition
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Ali Borji, Laurent Itti
    * Abstract: Several decades of research in computer and primate vision have resulted in many models (some specialized for one problem, others more general) and invaluable experimental data. Here, to help focus research efforts onto the hardest unsolved problems, and bridge computer and human vision, we define a battery of 5 tests that measure the gap between human and machine performances in several dimensions (generalization across scene categories, generalization from images to edge maps and line drawings, invariance to rotation and scaling, local/global information with jumbled images, and object recognition performance). We measure model accuracy and the correlation between model and human error patterns. Experimenting over 7 datasets, where human data is available, and gauging 14 well-established models, we find that none fully resembles humans in all aspects, and we learn from each test which models and features are more promising in approaching humans in the tested dimension. Across all tests, we find that models based on local edge histograms consistently resemble humans more, while several scene statistics or "gist" models do perform well with both scenes and objects. While computer vision has long been inspired by human vision, we believe systematic efforts, such as this, will help better identify shortcomings of models and find new paths forward.

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
* Semantic Depth Map Fusion for Moving Vehicle Detection in Aerial Video
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w25/html/Poostchi_Semantic_Depth_Map_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w25/papers/Poostchi_Semantic_Depth_Map_CVPR_2016_paper.pdf)]
    * Title: Semantic Depth Map Fusion for Moving Vehicle Detection in Aerial Video
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Mahdieh Poostchi, Hadi Aliakbarpour, Raphael Viguier, Filiz Bunyak, Kannappan Palaniappan, Guna Seetharaman
    * Abstract: Automatic moving object detection and segmentation is one of the fundamental low-level tasks for many of the urban traffic surveillance applications. We develop an automatic moving vehicle detection system for aerial video based on semantic fusion of trace of the flux tensor and tall structures altitude mask. Trace of the flux tensor provides spatio-temporal information of moving edges including undesirable motion of tall structures caused by parallax effects. The parallax induced motions are filtered out by incorporating buildings altitude masks obtained from available dense 3D point clouds. Using a level-set based geodesic active contours framework, the coarse thresholded building depth masks evolved into the actual building boundaries. Experiments are carried out on a cropped 2kx2k region of interest for 200 frames from Albuquerque urban aerial imagery. An average precision of 83% and recall of 76% have been reported using an object-level detection performance evaluation method.

count=2
* HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Sironi_HATS_Histograms_of_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf)]
    * Title: HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Amos Sironi, Manuele Brambilla, Nicolas Bourdis, Xavier Lagorce, Ryad Benosman
    * Abstract: Event-based cameras have recently drawn the attention of the Computer Vision community thanks to their advantages in terms of high temporal resolution, low power consumption and high dynamic range, compared to traditional frame-based cameras. These properties make event-based cameras an ideal choice for autonomous vehicles, robot navigation or UAV vision, among others. However, the accuracy of event-based object classification algorithms, which is of crucial importance for any reliable system working in real-world conditions, is still far behind their frame-based counterparts. Two main reasons for this performance gap are: 1. The lack of effective low-level representations and architectures for event-based object classification and 2. The absence of large real-world event-based datasets. In this paper we address both problems. First, we introduce a novel event-based feature representation together with a new machine learning architecture. Compared to previous approaches, we use local memory units to efficiently leverage past temporal information and build a robust event-based representation. Second, we release the first large real-world event-based dataset for object classification. We compare our method to the state-of-the-art with extensive experiments, showing better classification performance and real-time computation.

count=2
* Now You Shake Me: Towards Automatic 4D Cinema
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_Now_You_Shake_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Now_You_Shake_CVPR_2018_paper.pdf)]
    * Title: Now You Shake Me: Towards Automatic 4D Cinema
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Yuhao Zhou, Makarand Tapaswi, Sanja Fidler
    * Abstract: We are interested in enabling automatic 4D cinema by parsing physical and special effects from untrimmed movies. These include effects such as physical interactions, water splashing, light, and shaking, and are grounded to either a character in the scene or the camera. We collect a new dataset referred to as the Movie4D dataset which annotates over 9K effects in 63 movies. We propose a Conditional Random Field model atop a neural network that brings together visual and audio information, as well as semantics in the form of person tracks. Our model further exploits correlations of effects between different characters in the clip as well as across movie threads. We propose effect detection and classification as two tasks, and present results along with ablation studies on our dataset, paving the way towards 4D cinema in everyone’s homes.

count=2
* Rare Event Detection Using Disentangled Representation Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Hamaguchi_Rare_Event_Detection_Using_Disentangled_Representation_Learning_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hamaguchi_Rare_Event_Detection_Using_Disentangled_Representation_Learning_CVPR_2019_paper.pdf)]
    * Title: Rare Event Detection Using Disentangled Representation Learning
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Ryuhei Hamaguchi,  Ken Sakurada,  Ryosuke Nakamura
    * Abstract: This paper presents a novel method for rare event detection from an image pair with class-imbalanced datasets. A straightforward approach for event detection tasks is to train a detection network from a large-scale dataset in an end-to-end manner. However, in many applications such as building change detection on satellite images, few positive samples are available for the training. Moreover, an image pair of scenes contains many trivial events, such as in illumination changes or background motions. These many trivial events and the class imbalance problem lead to false alarms for rare event detection. In order to overcome these difficulties, we propose a novel method to learn disentangled representations from only low-cost negative samples. The proposed method disentangles the different aspects in a pair of observations: variant and invariant factors that represent trivial events and image contents, respectively. The effectiveness of the proposed approach is verified by the quantitative evaluations on four change detection datasets, and the qualitative analysis shows that the proposed method can acquire the representations that disentangle rare events from trivial ones.

count=2
* An End-to-End Edge Aggregation Network for Moving Object Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Patil_An_End-to-End_Edge_Aggregation_Network_for_Moving_Object_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Patil_An_End-to-End_Edge_Aggregation_Network_for_Moving_Object_Segmentation_CVPR_2020_paper.pdf)]
    * Title: An End-to-End Edge Aggregation Network for Moving Object Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Prashant W. Patil,  Kuldeep M. Biradar,  Akshay Dudhane,  Subrahmanyam Murala
    * Abstract: Moving object segmentation in videos (MOS) is a highly demanding task for security-based applications like automated outdoor video surveillance. Most of the existing techniques proposed for MOS are highly depend on fine-tuning a model on the first frame(s) of test sequence or complicated training procedure, which leads to limited practical serviceability of the algorithm. In this paper, the inherent correlation learning-based edge extraction mechanism (EEM) and dense residual block (DRB) are proposed for the discriminative foreground representation. The multi-scale EEM module provides the efficient foreground edge related information (with the help of encoder) to the decoder through skip connection at subsequent scale. Further, the response of the optical flow encoder stream and the last EEM module are embedded in the bridge network. The bridge network comprises of multi-scale residual blocks with dense connections to learn the effective and efficient foreground relevant features. Finally, to generate accurate and consistent foreground object maps, a decoder block is proposed with skip connections from respective multi-scale EEM module feature maps and the subsequent down-sampled response of previous frame output. Specifically, the proposed network does not require any pre-trained models or fine-tuning of the parameters with the initial frame(s) of the test video. The performance of the proposed network is evaluated with different configurations like disjoint, cross-data, and global training-testing techniques. The ablation study is conducted to analyse each model of the proposed network. To demonstrate the effectiveness of the proposed framework, a comprehensive analysis on four benchmark video datasets is conducted. Experimental results show that the proposed approach outperforms the state-of-the-art methods for MOS.

count=2
* DyStaB: Unsupervised Object Segmentation via Dynamic-Static Bootstrapping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_DyStaB_Unsupervised_Object_Segmentation_via_Dynamic-Static_Bootstrapping_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_DyStaB_Unsupervised_Object_Segmentation_via_Dynamic-Static_Bootstrapping_CVPR_2021_paper.pdf)]
    * Title: DyStaB: Unsupervised Object Segmentation via Dynamic-Static Bootstrapping
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yanchao Yang, Brian Lai, Stefano Soatto
    * Abstract: We describe an unsupervised method to detect and segment portions of images of live scenes that, at some point in time, are seen moving as a coherent whole, which we refer to as objects. Our method first partitions the motion field by minimizing the mutual information between segments. Then, it uses the segments to learn object models that can be used for detection in a static image. Static and dynamic models are represented by deep neural networks trained jointly in a bootstrapping strategy, which enables extrapolation to previously unseen objects. While the training process requires motion, the resulting object segmentation network can be used on either static images or videos at inference time. As the volume of seen videos grows, more and more objects are seen moving, priming their detection, which then serves as a regularizer for new objects, turning our method into unsupervised continual learning to segment objects. Our models are compared to the state of the art in both video object segmentation and salient object detection. In the six benchmark datasets tested, our models compare favorably even to those using pixel-level supervision, despite requiring no manual annotation.

count=2
* Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.pdf)]
    * Title: Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Nicolae-Cătălin Ristea, Neelu Madan, Radu Tudor Ionescu, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B. Moeslund, Mubarak Shah
    * Abstract: Anomaly detection is commonly pursued as a one-class classification problem, where models can only learn from normal training samples, while being evaluated on both normal and abnormal test samples. Among the successful approaches for anomaly detection, a distinguished category of methods relies on predicting masked information (e.g. patches, future frames, etc.) and leveraging the reconstruction error with respect to the masked information as an abnormality score. Different from related methods, we propose to integrate the reconstruction-based functionality into a novel self-supervised predictive architectural building block. The proposed self-supervised block is generic and can easily be incorporated into various state-of-the-art anomaly detection methods. Our block starts with a convolutional layer with dilated filters, where the center area of the receptive field is masked. The resulting activation maps are passed through a channel attention module. Our block is equipped with a loss that minimizes the reconstruction error with respect to the masked area in the receptive field. We demonstrate the generality of our block by integrating it into several state-of-the-art frameworks for anomaly detection on image and video, providing empirical evidence that shows considerable performance improvements on MVTec AD, Avenue, and ShanghaiTech. We release our code as open source at: https://github.com/ristea/sspcab.

count=2
* Pix2map: Cross-Modal Retrieval for Inferring Street Maps From Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Pix2map_Cross-Modal_Retrieval_for_Inferring_Street_Maps_From_Images_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Pix2map_Cross-Modal_Retrieval_for_Inferring_Street_Maps_From_Images_CVPR_2023_paper.pdf)]
    * Title: Pix2map: Cross-Modal Retrieval for Inferring Street Maps From Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Xindi Wu, KwunFung Lau, Francesco Ferroni, Aljoša Ošep, Deva Ramanan
    * Abstract: Self-driving vehicles rely on urban street maps for autonomous navigation. In this paper, we introduce Pix2Map, a method for inferring urban street map topology directly from ego-view images, as needed to continually update and expand existing maps. This is a challenging task, as we need to infer a complex urban road topology directly from raw image data. The main insight of this paper is that this problem can be posed as cross-modal retrieval by learning a joint, cross-modal embedding space for images and existing maps, represented as discrete graphs that encode the topological layout of the visual surroundings. We conduct our experimental evaluation using the Argoverse dataset and show that it is indeed possible to accurately retrieve street maps corresponding to both seen and unseen roads solely from image data. Moreover, we show that our retrieved maps can be used to update or expand existing maps and even show proof-of-concept results for visual localization and image retrieval from spatial graphs.

count=2
* Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf)]
    * Title: Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, Yinghuan Shi
    * Abstract: In this work, we revisit the weak-to-strong consistency framework, popularized by FixMatch from semi-supervised classification, where the prediction of a weakly perturbed image serves as supervision for its strongly perturbed version. Intriguingly, we observe that such a simple pipeline already achieves competitive results against recent advanced works, when transferred to our segmentation scenario. Its success heavily relies on the manual design of strong data augmentations, however, which may be limited and inadequate to explore a broader perturbation space. Motivated by this, we propose an auxiliary feature perturbation stream as a supplement, leading to an expanded perturbation space. On the other, to sufficiently probe original image-level augmentations, we present a dual-stream perturbation technique, enabling two strong views to be simultaneously guided by a common weak view. Consequently, our overall Unified Dual-Stream Perturbations approach (UniMatch) surpasses all existing methods significantly across all evaluation protocols on the Pascal, Cityscapes, and COCO benchmarks. Its superiority is also demonstrated in remote sensing interpretation and medical image analysis. We hope our reproduced FixMatch and our results can inspire more future works. Code and logs are available at https://github.com/LiheYoung/UniMatch.

count=2
* Probability-Based Global Cross-Modal Upsampling for Pansharpening
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Probability-Based_Global_Cross-Modal_Upsampling_for_Pansharpening_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Probability-Based_Global_Cross-Modal_Upsampling_for_Pansharpening_CVPR_2023_paper.pdf)]
    * Title: Probability-Based Global Cross-Modal Upsampling for Pansharpening
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zeyu Zhu, Xiangyong Cao, Man Zhou, Junhao Huang, Deyu Meng
    * Abstract: Pansharpening is an essential preprocessing step for remote sensing image processing. Although deep learning (DL) approaches performed well on this task, current upsampling methods used in these approaches only utilize the local information of each pixel in the low-resolution multispectral (LRMS) image while neglecting to exploit its global information as well as the cross-modal information of the guiding panchromatic (PAN) image, which limits their performance improvement. To address this issue, this paper develops a novel probability-based global cross-modal upsampling (PGCU) method for pan-sharpening. Precisely, we first formulate the PGCU method from a probabilistic perspective and then design an efficient network module to implement it by fully utilizing the information mentioned above while simultaneously considering the channel specificity. The PGCU module consists of three blocks, i.e., information extraction (IE), distribution and expectation estimation (DEE), and fine adjustment (FA). Extensive experiments verify the superiority of the PGCU method compared with other popular upsampling methods. Additionally, experiments also show that the PGCU module can help improve the performance of existing SOTA deep learning pansharpening methods. The codes are available at https://github.com/Zeyu-Zhu/PGCU.

count=2
* APPLeNet: Visual Attention Parameterized Prompt Learning for Few-Shot Remote Sensing Image Generalization Using CLIP
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Jha_APPLeNet_Visual_Attention_Parameterized_Prompt_Learning_for_Few-Shot_Remote_Sensing_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Jha_APPLeNet_Visual_Attention_Parameterized_Prompt_Learning_for_Few-Shot_Remote_Sensing_CVPRW_2023_paper.pdf)]
    * Title: APPLeNet: Visual Attention Parameterized Prompt Learning for Few-Shot Remote Sensing Image Generalization Using CLIP
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Mainak Singha, Ankit Jha, Bhupendra Solanki, Shirsha Bose, Biplab Banerjee
    * Abstract: In recent years, the success of large-scale vision-language models (VLMs) such as CLIP has led to their increased usage in various computer vision tasks. These models enable zero-shot inference through carefully crafted instructional text prompts without task-specific supervision. However, the potential of VLMs for generalization tasks in remote sensing (RS) has not been fully realized. To address this research gap, we propose a novel image-conditioned prompt learning strategy called the Visual Attention Parameterized Prompts Learning Network (APPLeNet). APPLeNet emphasizes the importance of multi-scale feature learning in RS scene classification and disentangles visual style and content primitives for domain generalization tasks. To achieve this, APPLeNet combines visual content features obtained from different layers of the vision encoder and style properties obtained from feature statistics of domain-specific batches. An attention-driven injection module is further introduced to generate visual tokens from this information. We also introduce an anti-correlation regularizer to ensure discrimination among the token embeddings, as this visual information is combined with the textual tokens. To validate APPLeNet, we curated four available RS benchmarks and introduced experimental protocols and datasets for three domain generalization tasks. Our results consistently outperform the relevant literature.

count=2
* Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.pdf)]
    * Title: Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Olaf Wysocki, Yan Xia, Magdalena Wysocki, Eleonora Grilli, Ludwig Hoegner, Daniel Cremers, Uwe Stilla
    * Abstract: Reconstructing semantic 3D building models at the level of detail (LoD) 3 is a long-standing challenge. Unlike mesh-based models, they require watertight geometry and object-wise semantics at the facade level. The principal challenge of such demanding semantic 3D reconstruction is reliable facade-level semantic segmentation of 3D input data. We present a novel method, called Scan2LoD3, that accurately reconstructs semantic LoD3 building models by improving facade-level semantic 3D segmentation. To this end, we leverage laser physics and 3D building model priors to probabilistically identify model conflicts. These probabilistic physical conflicts propose locations of model openings: Their final semantics and shapes are inferred in a Bayesian network fusing multimodal probabilistic maps of conflicts, 3D point clouds, and 2D images. To fulfill demanding LoD3 requirements, we use the estimated shapes to cut openings in 3D building priors and fit semantic 3D objects from a library of facade objects. Extensive experiments on the TUM city campus datasets demonstrate the superior performance of the proposed Scan2LoD3 over the state-of-the-art methods in facade-level detection, semantic segmentation, and LoD3 building model reconstruction. We believe our method can foster the development of probability-driven semantic 3D reconstruction at LoD3 since not only the high-definition reconstruction but also reconstruction confidence becomes pivotal for various applications such as autonomous driving and urban simulations.

count=2
* Parcel3D: Shape Reconstruction From Single RGB Images for Applications in Transportation Logistics
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Naumann_Parcel3D_Shape_Reconstruction_From_Single_RGB_Images_for_Applications_in_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/VISION/papers/Naumann_Parcel3D_Shape_Reconstruction_From_Single_RGB_Images_for_Applications_in_CVPRW_2023_paper.pdf)]
    * Title: Parcel3D: Shape Reconstruction From Single RGB Images for Applications in Transportation Logistics
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Alexander Naumann, Felix Hertlein, Laura Dörr, Kai Furmans
    * Abstract: We focus on enabling damage and tampering detection in logistics and tackle the problem of 3D shape reconstruction of potentially damaged parcels. As input we utilize single RGB images, which corresponds to use-cases where only simple handheld devices are available, e.g. for postmen during delivery or clients on delivery. We present a novel synthetic dataset, named Parcel3D, that is based on the Google Scanned Objects (GSO) dataset and consists of more than 13,000 images of parcels with full 3D annotations. The dataset contains intact, i.e. cuboid-shaped, parcels and damaged parcels, which were generated in simulations. We work towards detecting mishandling of parcels by presenting a novel architecture called CubeRefine R-CNN, which combines estimating a 3D bounding box with an iterative mesh refinement. We benchmark our approach on Parcel3D and an existing dataset of cuboid-shaped parcels in real-world scenarios. Our results show, that while training on Parcel3D enables transfer to the real world, enabling reliable deployment in real-world scenarios is still challenging. CubeRefine R-CNN yields competitive performance in terms of Mesh AP and is the only model that directly enables deformation assessment by 3D mesh comparison and tampering detection by comparing viewpoint invariant parcel side surface representations. Dataset and code are available at https://a-nau.github.io/parcel3d.

count=2
* A Category Agnostic Model for Visual Rearrangment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_A_Category_Agnostic_Model_for_Visual_Rearrangment_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_A_Category_Agnostic_Model_for_Visual_Rearrangment_CVPR_2024_paper.pdf)]
    * Title: A Category Agnostic Model for Visual Rearrangment
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yuyi Liu, Xinhang Song, Weijie Li, Xiaohan Wang, Shuqiang Jiang
    * Abstract: This paper presents a novel category agnostic model for visual rearrangement task which can help an embodied agent to physically recover the shuffled scene configuration without any category concepts to the goal configuration. Previous methods usually follow a similar architecture completing the rearrangement task by aligning the scene changes of the goal and shuffled configuration according to the semantic scene graphs. However constructing scene graphs requires the inference of category labels which not only causes the accuracy drop of the entire task but also limits the application in real world scenario. In this paper we delve deep into the essence of visual rearrangement task and focus on the two most essential issues scene change detection and scene change matching. We utilize the movement and the protrusion of point cloud to accurately identify the scene changes and match these changes depending on the similarity of category agnostic appearance feature. Moreover to assist the agent to explore the environment more efficiently and comprehensively we propose a closer-aligned-retrace exploration policy aiming to observe more details of the scene at a closer distance. We conduct extensive experiments on AI2THOR Rearrangement Challenge based on RoomR dataset and a new multi-room multi-instance dataset MrMiR collected by us. The experimental results demonstrate the effectiveness of our proposed method.

count=2
* Tackling the Satellite Downlink Bottleneck with Federated Onboard Learning of Image Compression
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Space/html/Gomez_Tackling_the_Satellite_Downlink_Bottleneck_with_Federated_Onboard_Learning_of_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Space/papers/Gomez_Tackling_the_Satellite_Downlink_Bottleneck_with_Federated_Onboard_Learning_of_CVPRW_2024_paper.pdf)]
    * Title: Tackling the Satellite Downlink Bottleneck with Federated Onboard Learning of Image Compression
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Pablo Gómez, Gabriele Meoni
    * Abstract: Satellite data transmission is a crucial bottleneck for Earth observation applications. To overcome this problem we propose a novel solution that trains a neural network on board multiple satellites to compress raw data and only send down heavily compressed previews of the images while retaining the possibility of sending down selected losslessly compressed data. The neural network learns to encode and decode the data in an unsupervised fashion using distributed machine learning. By simulating and optimizing the learning process under realistic constraints such as thermal power and communication limitations we demonstrate the feasibility and effectiveness of our approach. For this we model a constellation of three satellites in a Sun-synchronous orbit. We use real raw multispectral data from Sentinel-2 and demonstrate the feasibility on space-proven hardware for the training. Our compression method outperforms JPEG compression on different image metrics achieving better compression ratios and image quality. We report key performance indicators of our method such as image quality compression ratio and benchmark training time on a Unibap iX10-100 processor. Our method has the potential to significantly increase the amount of satellite data collected that would typically be discarded (e.g. over oceans) and can potentially be extended to other applications even outside Earth observation. All code and data of the method are available online to enable rapid application of this approach.

count=2
* Image Vegetation Index Through a Cycle Generative Adversarial Network
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/PBVS/Suarez_Image_Vegetation_Index_Through_a_Cycle_Generative_Adversarial_Network_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Suarez_Image_Vegetation_Index_Through_a_Cycle_Generative_Adversarial_Network_CVPRW_2019_paper.pdf)]
    * Title: Image Vegetation Index Through a Cycle Generative Adversarial Network
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Patricia L. Suarez,  Angel D. Sappa,  Boris X. Vintimilla,  Riad I. Hammoud
    * Abstract: This paper proposes a novel approach to estimate the Normalized Difference Vegetation Index (NDVI) just from an RGB image. The NDVI values are obtained by using images from the visible spectral band together with a synthetic near infrared image obtained by a cycled GAN. The cycled GAN network is able to obtain a NIR image from a given gray scale image. It is trained by using unpaired set of gray scale and NIR images by using a U-net architecture and a multiple loss function (gray scale images are obtained from the provided RGB images). Then, the NIR image estimated with the proposed cycle generative adversarial network is used to compute the NDVI index. Experimental results are provided showing the validity of the proposed approach. Additionally, comparisons with previous approaches are also provided.

count=2
* HIDeGan: A Hyperspectral-Guided Image Dehazing GAN
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w14/Mehta_HIDeGan_A_Hyperspectral-Guided_Image_Dehazing_GAN_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w14/Mehta_HIDeGan_A_Hyperspectral-Guided_Image_Dehazing_GAN_CVPRW_2020_paper.pdf)]
    * Title: HIDeGan: A Hyperspectral-Guided Image Dehazing GAN
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Aditya Mehta, Harsh Sinha, Pratik Narang, Murari Mandal
    * Abstract: Haze removal in images captured from a diverse set of scenarios is a very challenging problem. The existing dehazing methods either reconstruct the transmission map or directly estimate the dehazed image in RGB color space. In this paper, we make a first attempt to propose a Hyperspectral-guided Image Dehazing Generative Adversarial Network (HIDEGAN). The HIDEGAN architecture is formulated by designing an enhanced version of CYCLEGAN named R2HCYCLE and an enhanced conditional GAN named H2RGAN. The R2HCYCLE makes use of the hyperspectral-image (HSI) in combination with cycle-consistency and skeleton losses in order to improve the quality of information recovery by analyzing the entire spectrum. The H2RGAN estimates the clean RGB image from the hazy hyperspectral image generated by the R2HCYCLE. The models designed for spatial-spectral-spatial mapping generate visually better haze-free images. To facilitate HSI generation, datasets from spectral reconstruction challenge at NTIRE 2018 and NTIRE 2020 are used. A comprehensive set of experiments were conducted on the D-Hazy, and the recent RESIDE-Standard (SOTS), RESIDE-b (OTS) and RESIDE-Standard (HSTS) datasets. The proposed HIDEGAN outperforms the existing state-of-the-art in all these datasets.

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
* Learning Slow Features for Behaviour Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Zafeiriou_Learning_Slow_Features_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Zafeiriou_Learning_Slow_Features_2013_ICCV_paper.pdf)]
    * Title: Learning Slow Features for Behaviour Analysis
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Lazaros Zafeiriou, Mihalis A. Nicolaou, Stefanos Zafeiriou, Symeon Nikitidis, Maja Pantic
    * Abstract: A recently introduced latent feature learning technique for time varying dynamic phenomena analysis is the socalled Slow Feature Analysis (SFA). SFA is a deterministic component analysis technique for multi-dimensional sequences that by minimizing the variance of the first order time derivative approximation of the input signal finds uncorrelated projections that extract slowly-varying features ordered by their temporal consistency and constancy. In this paper, we propose a number of extensions in both the deterministic and the probabilistic SFA optimization frameworks. In particular, we derive a novel deterministic SFA algorithm that is able to identify linear projections that extract the common slowest varying features of two or more sequences. In addition, we propose an Expectation Maximization (EM) algorithm to perform inference in a probabilistic formulation of SFA and similarly extend it in order to handle two and more time varying data sequences. Moreover, we demonstrate that the probabilistic SFA (EMSFA) algorithm that discovers the common slowest varying latent space of multiple sequences can be combined with dynamic time warping techniques for robust sequence timealignment. The proposed SFA algorithms were applied for facial behavior analysis demonstrating their usefulness and appropriateness for this task.

count=2
* A Scalable Architecture for Operational FMV Exploitation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w1/html/Thissell_A_Scalable_Architecture_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w1/papers/Thissell_A_Scalable_Architecture_ICCV_2015_paper.pdf)]
    * Title: A Scalable Architecture for Operational FMV Exploitation
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: William R. Thissell, Robert Czajkowski, Francis Schrenk, Timothy Selway, Anthony J. Ries, Shamoli Patel, Patricia L. McDermott, Rod Moten, Ron Rudnicki, Guna Seetharaman, Ilker Ersoy, Kannappan Palaniappan
    * Abstract: A scalable open systems and standards derived software ecosystem is described for computer vision analytics (CVA) assisted exploitation of full motion video (FMV). The ecosystem, referred to as the Advanced Video Activity Analytics (AVAA), has two instantiations, one for size, weight, and power (SWAP) constrained conditions, and the other for large to massive cloud based configurations. The architecture is designed to meet operational analyst requirements to increase their productivity and accuracy for exploiting FMV using local cluster or scalable cloud-based computing resources. CVAs are encapsulated within a software plug-in architecture and FMV processing pipelines are constructed by combining these plug-ins to accomplish analytical tasks and manage provenance of processing history. An example pipeline for real-time motion detection and moving object characterization using the flux tensor approach is presented. An example video ingest experiment is described. Quantitative and qualitative methods for human factors engineering (HFE) assessment to evaluate cognitive loads for alternative work flow design choices are discussed. This HFE process is used for validating that an AVAA system instantiation with candidate workflow pipelines meets CVA assisted FMV exploitation operational goals for specific analyst workflows. AVAA offers a new framework for video understanding at scale for large enterprise applications in the government and commercial sectors.

count=2
* Scene Categorization With Spectral Features
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Khan_Scene_Categorization_With_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Khan_Scene_Categorization_With_ICCV_2017_paper.pdf)]
    * Title: Scene Categorization With Spectral Features
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Salman H. Khan, Munawar Hayat, Fatih Porikli
    * Abstract: Spectral signatures of natural scenes were earlier found to be distinctive for different scene types with varying spatial envelope properties such as openness, naturalness, ruggedness, and symmetry. Recently, such handcrafted features have been outclassed by deep learning based representations. This paper proposes a novel spectral description of convolution features, implemented efficiently as a unitary transformation within deep network architectures. To the best of our knowledge, this is the first attempt to use deep learning based spectral features explicitly for image classification task. We show that the spectral transformation decorrelates convolutional activations, which reduces co-adaptation between feature detections, thus acts as an effective regularizer. Our approach achieves significant improvements on three large-scale scene-centric datasets (MIT-67, SUN-397, and Places-205). Furthermore, we evaluated the proposed approach on the attribute detection task where its superior performance manifests its relevance to semantically meaningful characteristics of natural scenes.

count=2
* Panoptic Segmentation of Satellite Image Time Series With Convolutional Temporal Attention Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Garnot_Panoptic_Segmentation_of_Satellite_Image_Time_Series_With_Convolutional_Temporal_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Garnot_Panoptic_Segmentation_of_Satellite_Image_Time_Series_With_Convolutional_Temporal_ICCV_2021_paper.pdf)]
    * Title: Panoptic Segmentation of Satellite Image Time Series With Convolutional Temporal Attention Networks
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Vivien Sainte Fare Garnot, Loic Landrieu
    * Abstract: Unprecedented access to multi-temporal satellite imagery has opened new perspectives for a variety of Earth observation tasks. Among them, pixel-precise panoptic segmentation of agricultural parcels has major economic and environmental implications. While researchers have explored this problem for single images, we argue that the complex temporal patterns of crop phenology are better addressed with temporal sequences of images. In this paper, we present the first end-to-end, single-stage method for panoptic segmentation of Satellite Image Time Series (SITS). This module can be combined with our novel image sequence encoding network which relies on temporal self-attention to extract rich and adaptive multi-scale spatio-temporal features. We also introduce PASTIS, the first open-access SITS dataset with panoptic annotations. We demonstrate the superiority of our encoder for semantic segmentation against multiple competing network architectures, and set up the first state-of-the-art of panoptic segmentation of SITS. Our implementation and the PASTIS dataset are publicly available at (link-upon-publication).

count=2
* Viewpoint-Agnostic Change Captioning With Cycle Consistency
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Viewpoint-Agnostic_Change_Captioning_With_Cycle_Consistency_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Viewpoint-Agnostic_Change_Captioning_With_Cycle_Consistency_ICCV_2021_paper.pdf)]
    * Title: Viewpoint-Agnostic Change Captioning With Cycle Consistency
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Hoeseong Kim, Jongseok Kim, Hyungseok Lee, Hyunsung Park, Gunhee Kim
    * Abstract: Change captioning is the task of identifying the change and describing it with a concise caption. Despite recent advancements, filtering out insignificant changes still remains as a challenge. Namely, images from different camera perspectives can cause issues; a mere change in viewpoint should be disregarded while still capturing the actual changes. In order to tackle this problem, we present a new Viewpoint-Agnostic change captioning network with Cycle Consistency (VACC) that requires only one image each for the before and after scene, without depending on any other information. We achieve this by devising a new difference encoder module which can encode viewpoint information and model the difference more effectively. In addition, we propose a cycle consistency module that can potentially improve the performance of any change captioning networks in general by matching the composite feature of the generated caption and before image with the after image feature. We evaluate the performance of our proposed model across three datasets for change captioning, including a novel dataset we introduce here that contains images with changes under extreme viewpoint shifts. Through our experiments, we show the excellence of our method with respect to the CIDEr, BLEU-4, METEOR and SPICE scores. Moreover, we demonstrate that attaching our proposed cycle consistency module yields a performance boost for existing change captioning networks, even with varying image encoding mechanisms.

count=2
* Complete Moving Object Detection in the Context of Robust Subspace Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/RSL-CV/Sultana_Complete_Moving_Object_Detection_in_the_Context_of_Robust_Subspace_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/RSL-CV/Sultana_Complete_Moving_Object_Detection_in_the_Context_of_Robust_Subspace_ICCVW_2019_paper.pdf)]
    * Title: Complete Moving Object Detection in the Context of Robust Subspace Learning
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Maryam Sultana, Arif Mahmood, Thierry Bouwmans, Soon Ki Jung
    * Abstract: Complete moving object detection plays a vital role in many applications of computer vision. For instance, depth estimation, scene understanding, object interaction, semantic segmentation, accident detection and avoidance in case of moving vehicles on a highway. However, it becomes challenging in the presence of dynamic backgrounds, camouflage, bootstrapping, varying illumination conditions, and noise. Over the past decade, robust subspace learning based methods addressed the moving objects detection problem with excellent performance. However, the moving objects detected by these methods are incomplete, unable to generate the occluded parts. Indeed, complete or occlusion-free moving object detection is still challenging for these methods. In the current work, we address this challenge by proposing a conditional Generative Adversarial Network (cGAN) conditioned on non-occluded moving object pixels during training. It therefore learns the subspace spanned by the moving objects covering all the dynamic variations and semantic information. While testing, our proposed Complete cGAN (CcGAN) is able to generate complete occlusion free moving objects in challenging conditions. The experimental evaluations of our proposed method are performed on SABS benchmark dataset and compared with 14 state-of-the-art methods, including both robust subspace and deep learning based methods. Our experiments demonstrate the superiority of our proposed model over both types of existing methods.

count=2
* Autoencoder-Based Background Reconstruction and Foreground Segmentation With Background Noise Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.pdf)]
    * Title: Autoencoder-Based Background Reconstruction and Foreground Segmentation With Background Noise Estimation
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Bruno Sauvalle, Arnaud de La Fortelle
    * Abstract: Even after decades of research, dynamic scene background reconstruction and foreground object segmentation are still considered as open problems due to various challenges such as illumination changes, camera movements, or background noise caused by air turbulence or moving trees. We propose in this paper to model the background of a frame sequence as a low dimensional manifold using an autoencoder and compare the reconstructed background provided by this autoencoder with the original image to compute the foreground/background segmentation masks. The main novelty of the proposed model is that the autoencoder is also trained to predict the background noise, which allows to compute for each frame a pixel-dependent threshold to perform the foreground segmentation. Although the proposed model does not use any temporal or motion information, it exceeds the state of the art for unsupervised background subtraction on the CDnet 2014 and LASIESTA datasets, with a significant improvement on videos where the camera is moving. It is also able to perform background reconstruction on some non-video image datasets.

count=2
* Flexible information routing in neural populations through stochastic comodulation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/516b38afeee70474b04881a633728b15-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/516b38afeee70474b04881a633728b15-Paper.pdf)]
    * Title: Flexible information routing in neural populations through stochastic comodulation
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Caroline Haimerl, Cristina Savin, Eero Simoncelli
    * Abstract: Humans and animals are capable of flexibly switching between a multitude of tasks, each requiring rapid, sensory-informed decision making. Incoming stimuli are processed by a hierarchy of neural circuits consisting of millions of neurons with diverse feature selectivity. At any given moment, only a small subset of these carry task-relevant information. In principle, downstream processing stages could identify the relevant neurons through supervised learning, but this would require many example trials. Such extensive learning periods are inconsistent with the observed flexibility of humans or animals, who can adjust to changes in task parameters or structure almost immediately. Here, we propose a novel solution based on functionally-targeted stochastic modulation. It has been observed that trial-to-trial neural activity is modulated by a shared, low-dimensional, stochastic signal that introduces task-irrelevant noise. Counter-intuitively this noise is preferentially targeted towards task-informative neurons, corrupting the encoded signal. However, we hypothesize that this modulation offers a solution to the identification problem, labeling task-informative neurons so as to facilitate decoding. We simulate an encoding population of spiking neurons whose rates are modulated by a shared stochastic signal, and show that a linear decoder with readout weights approximating neuron-specific modulation strength can achieve near-optimal accuracy. Such a decoder allows fast and flexible task-dependent information routing without relying on hardwired knowledge of the task-informative neurons (as in maximum likelihood) or unrealistically many supervised training trials (as in regression).

count=2
* Continuous Meta-Learning without Tasks
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/cc3f5463bc4d26bc38eadc8bcffbc654-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf)]
    * Title: Continuous Meta-Learning without Tasks
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: James Harrison, Apoorva Sharma, Chelsea Finn, Marco Pavone
    * Abstract: Meta-learning is a promising strategy for learning to efficiently learn using data gathered from a distribution of tasks. However, the meta-learning literature thus far has focused on the task segmented setting, where at train-time, offline data is assumed to be split according to the underlying task, and at test-time, the algorithms are optimized to learn in a single task. In this work, we enable the application of generic meta-learning algorithms to settings where this task segmentation is unavailable, such as continual online learning with unsegmented time series data. We present meta-learning via online changepoint analysis (MOCA), an approach which augments a meta-learning algorithm with a differentiable Bayesian changepoint detection scheme. The framework allows both training and testing directly on time series data without segmenting it into discrete tasks. We demonstrate the utility of this approach on three nonlinear meta-regression benchmarks as well as two meta-image-classification benchmarks.

count=2
* Local Spatiotemporal Representation Learning for Longitudinally-consistent Neuroimage Analysis
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/57da66da25d0ce77e0129b246f358851-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/57da66da25d0ce77e0129b246f358851-Paper-Conference.pdf)]
    * Title: Local Spatiotemporal Representation Learning for Longitudinally-consistent Neuroimage Analysis
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Mengwei Ren, Neel Dey, Martin Styner, Kelly Botteron, Guido Gerig
    * Abstract: Recent self-supervised advances in medical computer vision exploit the global and local anatomical self-similarity for pretraining prior to downstream tasks such as segmentation. However, current methods assume i.i.d. image acquisition, which is invalid in clinical study designs where follow-up longitudinal scans track subject-specific temporal changes. Further, existing self-supervised methods for medically-relevant image-to-image architectures exploit only spatial or temporal self-similarity and do so via a loss applied only at a single image-scale, with naive multi-scale spatiotemporal extensions collapsing to degenerate solutions. To these ends, this paper makes two contributions: (1) It presents a local and multi-scale spatiotemporal representation learning method for image-to-image architectures trained on longitudinal images. It exploits the spatiotemporal self-similarity of learned multi-scale intra-subject image features for pretraining and develops several feature-wise regularizations that avoid degenerate representations; (2) During finetuning, it proposes a surprisingly simple self-supervised segmentation consistency regularization to exploit intra-subject correlation. Benchmarked across various segmentation tasks, the proposed framework outperforms both well-tuned randomly-initialized baselines and current self-supervised techniques designed for both i.i.d. and longitudinal datasets. These improvements are demonstrated across both longitudinal neurodegenerative adult MRI and developing infant brain MRI and yield both higher performance and longitudinal consistency.

count=2
* D^2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/d2cc447db9e56c13b993c11b45956281-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/d2cc447db9e56c13b993c11b45956281-Paper-Conference.pdf)]
    * Title: D^2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Tianhao Wu, Fangcheng Zhong, Andrea Tagliasacchi, Forrester Cole, Cengiz Oztireli
    * Abstract: Given a monocular video, segmenting and decoupling dynamic objects while recovering the static environment is a widely studied problem in machine intelligence. Existing solutions usually approach this problem in the image domain, limiting their performance and understanding of the environment. We introduce Decoupled Dynamic Neural Radiance Field (D^2NeRF), a self-supervised approach that takes a monocular video and learns a 3D scene representation which decouples moving objects, including their shadows, from the static background. Our method represents the moving objects and the static background by two separate neural radiance fields with only one allowing for temporal changes. A naive implementation of this approach leads to the dynamic component taking over the static one as the representation of the former is inherently more general and prone to overfitting. To this end, we propose a novel loss to promote correct separation of phenomena. We further propose a shadow field network to detect and decouple dynamically moving shadows. We introduce a new dataset containing various dynamic objects and shadows and demonstrate that our method can achieve better performance than state-of-the-art approaches in decoupling dynamic and static 3D objects, occlusion and shadow removal, and image segmentation for moving objects. Project page: https://d2nerf.github.io/

count=2
* Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/2ab3163ee384cd46baa7f1abb2b1bf19-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/2ab3163ee384cd46baa7f1abb2b1bf19-Paper-Conference.pdf)]
    * Title: Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Katie Luo, Zhenzhen Liu, Xiangyu Chen, Yurong You, Sagie Benaim, Cheng Perng Phoo, Mark Campbell, Wen Sun, Bharath Hariharan, Kilian Q. Weinberger
    * Abstract: Recent advances in machine learning have shown that Reinforcement Learning from Human Feedback (RLHF) can improve machine learning models and align them with human preferences. Although very successful for Large Language Models (LLMs), these advancements have not had a comparable impact in research for autonomous vehicles—where alignment with human expectations can be imperative. In this paper, we propose to adapt similar RL-based methods to unsupervised object discovery, i.e. learning to detect objects from LiDAR points without any training labels. Instead of labels, we use simple heuristics to mimic human feedback. More explicitly, we combine multiple heuristics into a simple reward function that positively correlates its score with bounding box accuracy, i.e., boxes containing objects are scored higher than those without. We start from the detector’s own predictions to explore the space and reinforce boxes with high rewards through gradient updates. Empirically, we demonstrate that our approach is not only more accurate, but also orders of magnitudes faster to train compared to prior works on object discovery. Code is available at https://github.com/katieluo88/DRIFT.

count=2
* Disentangling Voice and Content with Self-Supervision for Speaker Recognition
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/9d276b0a087efdd2404f3295b26c24c1-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/9d276b0a087efdd2404f3295b26c24c1-Paper-Conference.pdf)]
    * Title: Disentangling Voice and Content with Self-Supervision for Speaker Recognition
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: TIANCHI LIU, Kong Aik Lee, Qiongqiong Wang, Haizhou Li
    * Abstract: For speaker recognition, it is difficult to extract an accurate speaker representation from speech because of its mixture of speaker traits and content. This paper proposes a disentanglement framework that simultaneously models speaker traits and content variability in speech. It is realized with the use of three Gaussian inference layers, each consisting of a learnable transition model that extracts distinct speech components. Notably, a strengthened transition model is specifically designed to model complex speech dynamics. We also propose a self-supervision method to dynamically disentangle content without the use of labels other than speaker identities. The efficacy of the proposed framework is validated via experiments conducted on the VoxCeleb and SITW datasets with 9.56\% and 8.24\% average reductions in EER and minDCF, respectively. Since neither additional model training nor data is specifically needed, it is easily applicable in practical use.

count=1
* Fully Transformer Network for Change Detection of Remote Sensing Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2022/html/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2022/papers/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.pdf)]
    * Title: Fully Transformer Network for Change Detection of Remote Sensing Images
    * Publisher: ACCV
    * Publication Date: `2022`
    * Authors: Tianyu Yan, Zifu Wan, Pingping Zhang
    * Abstract: Recently, change detection (CD) of remote sensing images have achieved great progress with the advances of deep learning. However, current methods generally deliver incomplete CD regions and irregular CD boundaries due to the limited representation ability of the extracted visual features. To relieve these issues, in this work we propose a novel learning framework named Fully Transformer Network (FTN) for remote sensing image CD, which improves the feature extraction from a global view and combines multi-level visual features in a pyramid manner. More specifically, the proposed framework first utilizes the advantages of Transformers in long-range dependency modeling. It can help to learn more discriminative global-level features and obtain complete CD regions. Then, we introduce a pyramid structure to aggregate multi-level visual features from Transformers for feature enhancement. The pyramid structure grafted with a Progressive Attention Module (PAM) can improve the feature representation ability with additional interdependencies through channel attentions. Finally, to better train the framework, we utilize the deeply-supervised learning with multiple boundaryaware loss functions. Extensive experiments demonstrate that our proposed method achieves a new state-of-the-art performance on four public CD benchmarks. For model reproduction, the source code is released at https://github.com/AI-Zhpp/FTN.

count=1
* OneDiff: A Generalist Model for Image Difference Captioning
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Hu_OneDiff_A_Generalist_Model_for_Image_Difference_Captioning_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Hu_OneDiff_A_Generalist_Model_for_Image_Difference_Captioning_ACCV_2024_paper.pdf)]
    * Title: OneDiff: A Generalist Model for Image Difference Captioning
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Erdong Hu, Longteng Guo, Tongtian Yue, Zijia Zhao, Shuning Xue, Jing Liu
    * Abstract: In computer vision, Image Difference Captioning (IDC) is crucial for accurately describing variations between closely related images. Traditional IDC methods often rely on specialist models, which restrict their applicability across varied contexts. This paper introduces the OneDiff model, a novel generalist approach that utilizes a robust vision-language model architecture, integrating a siamese image encoder with a Visual Delta Module. This innovative configuration allows for the precise detection and articulation of fine-grained differences between image pairs. OneDiff is trained through a dual-phase strategy, encompassing Coupled Sample Training and multi-task learning across a diverse array of data types, supported by our newly developed DiffCap Dataset. This dataset merges real-world and synthetic data, enhancing the training process and bolstering the models robustness. Extensive testing on diverse IDC benchmarks, such as Spot-the-Diff, Image-Editing-Request, and Birds-to-Words, shows that OneDiff consistently outperforms existing state-of-the-art models in accuracy and adaptability, achieving improvements of up to 97% CIDEr points in average. By setting a new benchmark in IDC, OneDiff paves the way for more versatile and effective applications in detecting and describing visual differences. The code, models, and data will be made publicly available.

count=1
* Moving Object Segmentation: All You Need Is SAM (and Flow)
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Xie_Moving_Object_Segmentation_All_You_Need_Is_SAM_and_Flow_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Xie_Moving_Object_Segmentation_All_You_Need_Is_SAM_and_Flow_ACCV_2024_paper.pdf)]
    * Title: Moving Object Segmentation: All You Need Is SAM (and Flow)
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Junyu Xie, Charig Yang, Weidi Xie, Andrew Zisserman
    * Abstract: The objective of this paper is motion segmentation -- discovering and segmenting the moving objects in a video. This is a much studied area with numerous careful, and sometimes complex, approaches and training schemes including: self-supervised learning, learning from synthetic datasets, object-centric representations, amodal representations, and many more. Our interest in this paper is to determine if the Segment Anything model (SAM) can contribute to this task. We investigate two models for combining SAM with optical flow that harness the segmentation power of SAM with the ability of flow to discover and group moving objects. In the first model, we adapt SAM to take optical flow, rather than RGB, as an input. In the second, SAM takes RGB as an input, and flow is used as a segmentation prompt. These surprisingly simple methods, without any further modifications, outperform all previous approaches by a considerable margin in both single and multi-object benchmarks. We also extend these frame-level segmentations to sequence-level segmentations that maintain object identity. Again, this simple model achieves outstanding performance across multiple moving object segmentation benchmarks.

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
* Background Subtraction Using Local SVD Binary Pattern
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/Guo_Background_Subtraction_Using_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/Guo_Background_Subtraction_Using_CVPR_2016_paper.pdf)]
    * Title: Background Subtraction Using Local SVD Binary Pattern
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Lili Guo, Dan Xu, Zhenping Qiang
    * Abstract: Background subtraction is a basic problem for change detection in videos and also the first step of high-level computer vision applications. Most background subtraction methods rely on color and texture feature. However, due to illuminations changes in different scenes and affections of noise pixels, those methods often resulted in high false positives in a complex environment. To solve this problem, we propose an adaptive background subtraction model which uses a novel Local SVD Binary Pattern (named LSBP) feature instead of simply depending on color intensity. This feature can describe the potential structure of the local regions in a given image, thus, it can enhance the robustness to illumination variation, noise, and shadows. We use a sample consensus model which is well suited for our LSBP feature. Experimental results on CDnet 2012 dataset demonstrate that our background subtraction method using LSBP feature is more effective than many state-of-the-art methods.

count=1
* Fast Image Gradients Using Binary Feature Convolutions
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/St-Charles_Fast_Image_Gradients_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/St-Charles_Fast_Image_Gradients_CVPR_2016_paper.pdf)]
    * Title: Fast Image Gradients Using Binary Feature Convolutions
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: The recent increase in popularity of binary feature descriptors has opened the door to new lightweight computer vision applications. Most research efforts thus far have been dedicated to the introduction of new large-scale binary features, which are primarily used for keypoint description and matching. In this paper, we show that the side products of small-scale binary feature computations can efficiently filter images and estimate image gradients. The improved efficiency of low-level operations can be especially useful in time-constrained applications. Through our experiments, we show that efficient binary feature convolutions can be used to mimic various image processing operations, and even outperform Sobel gradient estimation in the edge detection problem, both in terms of speed and F-Measure.

count=1
* Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Teutsch_Robust_Detection_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Teutsch_Robust_Detection_of_CVPR_2016_paper.pdf)]
    * Title: Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Michael Teutsch, Michael Grinberg
    * Abstract: Multiple object tracking in Wide Area Motion Imagery (WAMI) data is usually based on initial detections coming from background subtraction or frame differencing. However, these methods are prone to produce split and merged detections. Appearance based vehicle detection can be an alternative but is not well-suited for WAMI data since classifier models are of weak discriminative power for vehicles in top view at low resolution. We introduce a moving vehicle detection algorithm that combines 2-frame differencing with a vehicle appearance model to improve object detection. Our main contributions are (1) integration of robust vehicle detection with split/merge handling and (2) estimation of assignment likelihoods between object hypotheses in consecutive frames using an appearance based similarity measure. Without using any prior knowledge, we achieve state-of-the-art detection rates and produce tracklets that considerably simplify the data association problem for multiple object tracking.

count=1
* Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Makihara_Joint_Intensity_and_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Makihara_Joint_Intensity_and_CVPR_2017_paper.pdf)]
    * Title: Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yasushi Makihara, Atsuyuki Suzuki, Daigo Muramatsu, Xiang Li, Yasushi Yagi
    * Abstract: This paper describes a joint intensity metric learning method to improve the robustness of gait recognition with silhouette-based descriptors such as gait energy images. Because existing methods often use the difference of image intensities between a matching pair (e.g., the absolute difference of gait energies for the l_1-norm) to measure a dissimilarity, large intrasubject differences derived from covariate conditions (e.g., large gait energies caused by carried objects vs. small gait energies caused by the background), may wash out subtle intersubject differences (e.g., the difference of middle-level gait energies derived from motion differences). We therefore introduce a metric on joint intensity to mitigate the large intrasubject differences as well as leverage the subtle intersubject differences. More specifically, we formulate the joint intensity and spatial metric learning in a unified framework and alternately optimize it by linear or ranking support vector machines. Experiments using the OU-ISIR treadmill data set B with the largest clothing variation and large population data set with bag, b version containing carrying status in the wild demonstrate the effectiveness of the proposed method.

count=1
* 3D Human Pose Estimation From a Single Image via Distance Matrix Regression
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Moreno-Noguer_3D_Human_Pose_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moreno-Noguer_3D_Human_Pose_CVPR_2017_paper.pdf)]
    * Title: 3D Human Pose Estimation From a Single Image via Distance Matrix Regression
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Francesc Moreno-Noguer
    * Abstract: This paper addresses the problem of 3D human pose estimation from a single image. We follow a standard two-step pipeline by first detecting the 2D position of the N body joints, and then using these observations to infer 3D pose. For the first step, we use a recent CNN-based detector. For the second step, most existing approaches perform 2N-to-3N regression of the Cartesian joint coordinates. We show that more precise pose estimates can be obtained by representing both the 2D and 3D human poses using NxN distance matrices, and formulating the problem as a 2D-to-3D distance matrix regression. For learning such a regressor we leverage on simple Neural Network architectures, which by construction, enforce positivity and symmetry of the predicted matrices. The approach has also the advantage to naturally handle missing observations and allowing to hypothesize the position of non-observed joints. Quantitative results on Humaneva and Human3.6M datasets demonstrate consistent performance gains over state-of-the-art. Qualitative evaluation on the images in-the-wild of the LSP dataset, using the regressor learned on Human3.6M, reveals very promising generalization results.

count=1
* UntrimmedNets for Weakly Supervised Action Recognition and Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_UntrimmedNets_for_Weakly_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_UntrimmedNets_for_Weakly_CVPR_2017_paper.pdf)]
    * Title: UntrimmedNets for Weakly Supervised Action Recognition and Detection
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Limin Wang, Yuanjun Xiong, Dahua Lin, Luc Van Gool
    * Abstract: Current action recognition methods heavily rely on trimmed videos for model training. However, it is expensive and time-consuming to acquire a large-scale trimmed video dataset. This paper presents a new weakly supervised architecture, called UntrimmedNet, which is able to directly learn action recognition models from untrimmed videos without the requirement of temporal annotations of action instances. Our UntrimmedNet couples two important components, the classification module and the selection module, to learn the action models and reason about the temporal duration of action instances, respectively. These two components are implemented with feed-forward networks, and UntrimmedNet is therefore an end-to-end trainable architecture. We exploit the learned models for action recognition (WSR) and detection (WSD) on the untrimmed video datasets of THUMOS14 and ActivityNet. Although our UntrimmedNet only employs weak supervision, our method achieves performance superior or comparable to that of those strongly supervised approaches on these two datasets.

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
* Graph-Cut RANSAC
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.pdf)]
    * Title: Graph-Cut RANSAC
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Daniel Barath, Jiří Matas
    * Abstract: A novel method for robust estimation, called Graph-Cut RANSAC, GC-RANSAC in short, is introduced. To separate inliers and outliers, it runs the graph-cut algorithm in the local optimization (LO) step which is applied when a so-far-the-best model is found. The proposed LO step is conceptually simple, easy to implement, globally optimal and efficient. GC-RANSAC is shown experimentally, both on synthesized tests and real image pairs, to be more geometrically accurate than state-of-the-art methods on a range of problems, e.g. line fitting, homography, affine transformation, fundamental and essential matrix estimation. It runs in real-time for many problems at a speed approximately equal to that of the less accurate alternatives (in milliseconds on standard CPU).

count=1
* WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.pdf)]
    * Title: WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Tatjana Chavdarova, Pierre Baqué, Stéphane Bouquet, Andrii Maksai, Cijo Jose, Timur Bagautdinov, Louis Lettry, Pascal Fua, Luc Van Gool, François Fleuret
    * Abstract: People detection methods are highly sensitive to occlusions between pedestrians, which are extremely frequent in many situations where cameras have to be mounted at a limited height. The reduction of camera prices allows for the generalization of static multi-camera set-ups. Using joint visual information from multiple synchronized cameras gives the opportunity to improve detection performance. In this paper, we present a new large-scale and high-resolution dataset. It has been captured with seven static cameras in a public open area, and unscripted dense groups of pedestrians standing and walking. Together with the camera frames, we provide an accurate joint (extrinsic and intrinsic) calibration, as well as 7 series of 400 annotated frames for detection at a rate of 2 frames per second. This results in over 40,000 bounding boxes delimiting every person present in the area of interest, for a total of more than 300 individuals. We provide a series of benchmark results using baseline algorithms published over the recent months for multi-view detection with deep neural networks, and trajectory estimation using a non-Markovian model.

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
* Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Video_Rain_Streak_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Video_Rain_Streak_CVPR_2018_paper.pdf)]
    * Title: Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Minghan Li, Qi Xie, Qian Zhao, Wei Wei, Shuhang Gu, Jing Tao, Deyu Meng
    * Abstract: Videos captured by outdoor surveillance equipments sometimes contain unexpected rain streaks, which brings difficulty in subsequent video processing tasks. Rain streak removal from a video is thus an important topic in recent computer vision research. In this paper, we raise two intrinsic characteristics specifically possessed by rain streaks. Firstly, the rain streaks in a video contain repetitive local patterns sparsely scattered over different positions of the video. Secondly, the rain streaks are with multiscale configurations due to their occurrence on positions with different distances to the cameras. Based on such understanding, we specifically formulate both characteristics into a multiscale convolutional sparse coding (MS-CSC) model for the video rain streak removal task. Specifically, we use multiple convolutional filters convolved on the sparse feature maps to deliver the former characteristic, and further use multiscale filters to represent different scales of rain streaks. Such a new encoding manner makes the proposed method capable of properly extracting rain streaks from videos, thus getting fine video deraining effects. Experiments implemented on synthetic and real videos verify the superiority of the proposed method, as compared with the state-of-the-art ones along this research line, both visually and quantitatively.

count=1
* Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Qu_Unsupervised_Sparse_Dirichlet-Net_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qu_Unsupervised_Sparse_Dirichlet-Net_CVPR_2018_paper.pdf)]
    * Title: Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Ying Qu, Hairong Qi, Chiman Kwan
    * Abstract: In many computer vision applications, obtaining images of high resolution in both the spatial and spectral domains are equally important. However, due to hardware limitations, one can only expect to acquire images of high resolution in either the spatial or spectral domains. This paper focuses on hyperspectral image super-resolution (HSI-SR), where a hyperspectral image (HSI) with low spatial resolution (LR) but high spectral resolution is fused with a multispectral image (MSI) with high spatial resolution (HR) but low spectral resolution to obtain HR HSI. Existing deep learning-based solutions are all supervised that would need a large training set and the availability of HR HSI, which is unrealistic. Here, we make the first attempt to solving the HSI-SR problem using an unsupervised encoder-decoder architecture that carries the following uniquenesses. First, it is composed of two encoder-decoder networks, coupled through a shared decoder, in order to preserve the rich spectral information from the HSI network. Second, the network encourages the representations from both modalities to follow a sparse Dirichlet distribution which naturally incorporates the two physical constraints of HSI and MSI. Third, the angular difference between representations are minimized in order to reduce the spectral distortion. We refer to the proposed architecture as unsupervised Sparse Dirichlet-Net, or uSDN. Extensive experimental results demonstrate the superior performance of uSDN as compared to the state-of-the-art.

count=1
* Clothing Change Aware Person Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w41/html/Xue_Clothing_Change_Aware_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w41/Xue_Clothing_Change_Aware_CVPR_2018_paper.pdf)]
    * Title: Clothing Change Aware Person Identification
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Jia Xue, Zibo Meng, Karthik Katipally, Haibo Wang, Kees van Zon
    * Abstract: We develop a person identification approach - Clothing Change Aware Network (CCAN) for the task of clothing assisted person identification. CCAN concerns approaches that go beyond face recognition and particularly tackles the role of clothing to identification. Person identification is a rather challenging task when clothing appears changed under complex background information. With a pair of two person images as input, CCAN simultaneously performs a verification task to detect change in clothing and an identification task to predict person identity. When clothing from the pair of input images is detected to be different, CCAN automatically understates clothing information while emphasizing face, and vice versa. In practice, CCAN outperforms the way of equally stacking face and full body context features, and shows leading results on the People in Photos Album (PIPA) dataset.

count=1
* Efficient Online Multi-Person 2D Pose Tracking With Recurrent Spatio-Temporal Affinity Fields
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Raaj_Efficient_Online_Multi-Person_2D_Pose_Tracking_With_Recurrent_Spatio-Temporal_Affinity_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Raaj_Efficient_Online_Multi-Person_2D_Pose_Tracking_With_Recurrent_Spatio-Temporal_Affinity_CVPR_2019_paper.pdf)]
    * Title: Efficient Online Multi-Person 2D Pose Tracking With Recurrent Spatio-Temporal Affinity Fields
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yaadhav Raaj,  Haroon Idrees,  Gines Hidalgo,  Yaser Sheikh
    * Abstract: We present an online approach to efficiently and simultaneously detect and track 2D poses of multiple people in a video sequence. We build upon Part Affinity Field (PAF) representation designed for static images, and propose an architecture that can encode and predict Spatio-Temporal Affinity Fields (STAF) across a video sequence. In particular, we propose a novel temporal topology cross-linked across limbs which can consistently handle body motions of a wide range of magnitudes. Additionally, we make the overall approach recurrent in nature, where the network ingests STAF heatmaps from previous frames and estimates those for the current frame. Our approach uses only online inference and tracking, and is currently the fastest and the most accurate bottom-up approach that is runtime-invariant to the number of people in the scene and accuracy-invariant to input frame rate of camera. Running at ~30 fps on a single GPU at single scale, it achieves highly competitive results on the PoseTrack benchmarks.

count=1
* Did It Change? Learning to Detect Point-Of-Interest Changes for Proactive Map Updates
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Revaud_Did_It_Change_Learning_to_Detect_Point-Of-Interest_Changes_for_Proactive_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Revaud_Did_It_Change_Learning_to_Detect_Point-Of-Interest_Changes_for_Proactive_CVPR_2019_paper.pdf)]
    * Title: Did It Change? Learning to Detect Point-Of-Interest Changes for Proactive Map Updates
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jerome Revaud,  Minhyeok Heo,  Rafael S. Rezende,  Chanmi You,  Seong-Gyun Jeong
    * Abstract: Maps are an increasingly important tool in our daily lives, yet their rich semantic content still largely depends on manual input. Motivated by the broad availability of geo-tagged street-view images, we propose a new task aiming to make the map update process more proactive. We focus on automatically detecting changes of Points of Interest (POIs), specifically stores or shops of any kind, based on visual input. Faced with the lack of an appropriate benchmark, we build and release a large dataset, captured in two large shopping centers, that comprises 33K geo-localized images and 578 POIs. We then design a generic approach that compares two image sets captured in the same venue at different times and outputs POI changes as a ranked list of map locations. In contrast to logo or franchise recognition approaches, our system does not depend on an external franchise database. It is instead inspired by recent deep metric learning approaches that learn a similarity function fit to the task at hand. We compare various loss functions to learn a metric aligned with the POI change detection goal, and report promising results.

count=1
* Event Probability Mask (EPM) and Event Denoising Convolutional Neural Network (EDnCNN) for Neuromorphic Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Baldwin_Event_Probability_Mask_EPM_and_Event_Denoising_Convolutional_Neural_Network_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Baldwin_Event_Probability_Mask_EPM_and_Event_Denoising_Convolutional_Neural_Network_CVPR_2020_paper.pdf)]
    * Title: Event Probability Mask (EPM) and Event Denoising Convolutional Neural Network (EDnCNN) for Neuromorphic Cameras
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: R. Wes Baldwin,  Mohammed Almatrafi,  Vijayan Asari,  Keigo Hirakawa
    * Abstract: This paper presents a novel method for labeling real-world neuromorphic camera sensor data by calculating the likelihood of generating an event at each pixel within a short time window, which we refer to as "event probability mask" or EPM. Its applications include (i) objective benchmarking of event denoising performance, (ii) training convolutional neural networks for noise removal called "event denoising convolutional neural network" (EDnCNN), and (iii) estimating internal neuromorphic camera parameters. We provide the first dataset (DVSNOISE20) of real-world labeled neuromorphic camera events for noise removal.

count=1
* Large-Scale Localization Datasets in Crowded Indoor Spaces
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Large-Scale_Localization_Datasets_in_Crowded_Indoor_Spaces_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Large-Scale_Localization_Datasets_in_Crowded_Indoor_Spaces_CVPR_2021_paper.pdf)]
    * Title: Large-Scale Localization Datasets in Crowded Indoor Spaces
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Donghwan Lee, Soohyun Ryu, Suyong Yeon, Yonghan Lee, Deokhwa Kim, Cheolho Han, Yohann Cabon, Philippe Weinzaepfel, Nicolas Guerin, Gabriela Csurka, Martin Humenberger
    * Abstract: Estimating the precise location of a camera using visual localization enables interesting applications such as augmented reality or robot navigation. This is particularly useful in indoor environments where other localization technologies, such as GNSS, fail. Indoor spaces impose interesting challenges on visual localization algorithms: occlusions due to people, textureless surfaces, large viewpoint changes, low light, repetitive textures, etc. Existing indoor datasets are either comparably small or do only cover a subset of the mentioned challenges. In this paper, we introduce 5 new indoor datasets for visual localization in challenging real-world environments. They were captured in a large shopping mall and a large metro station in Seoul, South Korea, using a dedicated mapping platform consisting of 10 cameras and 2 laser scanners. In order to obtain accurate ground truth camera poses, we developed a robust LiDAR SLAM which provides initial poses that are then refined using a novel structure-from-motion based optimization. We present a benchmark of modern visual localization algorithms on these challenging datasets showing superior performance of structure-based methods using robust image features. The datasets are available at: https://naverlabs.com/datasets

count=1
* The Multi-Temporal Urban Development SpaceNet Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.pdf)]
    * Title: The Multi-Temporal Urban Development SpaceNet Dataset
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Adam Van Etten, Daniel Hogan, Jesus Martinez Manso, Jacob Shermeyer, Nicholas Weir, Ryan Lewis
    * Abstract: Satellite imagery analytics have numerous human development and disaster response applications, particularly when time series methods are involved. For example, quantifying population statistics is fundamental to 67 of the 231 United Nations Sustainable Development Goals Indicators, but the World Bank estimates that over 100 countries currently lack effective Civil Registration systems. To help address this deficit and develop novel computer vision methods for time series data, we present the Multi-Temporal Urban Development SpaceNet (MUDS, also known as SpaceNet 7) dataset. This open source dataset consists of medium resolution (4.0m) satellite imagery mosaics, which includes 24 images (one per month) covering >100 unique geographies, and comprises >40,000 km2 of imagery and exhaustive polygon labels of building footprints therein, totaling over 11M individual annotations. Each building is assigned a unique identifier (i.e. address), which permits tracking of individual objects over time. Label fidelity exceeds image resolution; this "omniscient labeling" is a unique feature of the dataset, and enables surprisingly precise algorithmic models to be crafted. We demonstrate methods to track building footprint construction (or demolition) over time, thereby directly assessing urbanization. Performance is measured with the newly developed SpaceNet Change and Object Tracking (SCOT) metric, which quantifies both object tracking as well as change detection. We demonstrate that despite the moderate resolution of the data, we are able to track individual building identifiers over time.

count=1
* Phenology Alignment Network: A Novel Framework for Cross-Regional Time Series Crop Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AgriVision/html/Wang_Phenology_Alignment_Network_A_Novel_Framework_for_Cross-Regional_Time_Series_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AgriVision/papers/Wang_Phenology_Alignment_Network_A_Novel_Framework_for_Cross-Regional_Time_Series_CVPRW_2021_paper.pdf)]
    * Title: Phenology Alignment Network: A Novel Framework for Cross-Regional Time Series Crop Classification
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Ziqiao Wang, Hongyan Zhang, Wei He, Liangpei Zhang
    * Abstract: Timely and accurate crop type classification plays an essential role in the study of agricultural application. However, large area or cross-regional crop classification confronts huge challenges owing to dramatic phenology discrepancy among training and test regions. In this work, we propose a novel framework to address these challenges based on deep recurrent network and unsupervised domain adaptation (DA). Specifically, we firstly propose a Temporal Spatial Network (TSNet) for pixelwise crop classification, which contains stacked RNN and self-attention module to adaptively extract multi-level features from crop samples under various planting conditions. To deal with the cross-regional challenge, an unsupervised DA-based framework named Phenology Alignment Network (PAN) is proposed. PAN consists of two branches of two identical TSNet pre-trained on source domain; one branch takes source samples while the other takes target samples as input. Through aligning the hierarchical deep features extracted from two branches, the discrepancy between two regions is decreased and the pre-trained model is adapted to the target domain without using target label information. As another contribution, a time series dataset based on Sentinel-2 was annotated containing winter crop samples collected on three study sites of China. Cross-regional experiments demonstrate that TSNet shows comparable accuracy to state-of-the-art methods, and PAN further improves the overall accuracy by 5.62%, and macro average F1 score by 0.094 unsupervisedly.

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
* QFabric: Multi-Task Change Detection Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.pdf)]
    * Title: QFabric: Multi-Task Change Detection Dataset
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Sagar Verma, Akash Panigrahi, Siddharth Gupta
    * Abstract: Detecting change through multi-image, multi-date remote sensing is essential to developing an understanding of global conditions. Despite recent advancements in remote sensing realized through deep learning, novel methods for accurate multi-image change detection remain unrealized. Recently, several promising methods have been proposed to address this topic, but a paucity of publicly available data limits the methods that can be assessed. In particular, there exists limited work on categorizing the nature and status of change across an observation period. This paper introduces the first labeled dataset available for such a task. We present an open-source change detection dataset, termed QFabric, with 450,000 change polygons annotated across 504 locations in 100 different cities covering a wide range of geographies and urban fabrics. QFabric is a temporal multi-task dataset with 6 change types and 9 change status classes. The geography and environment metadata around each polygon provides context that can be leveraged to build robust deep neural networks. We apply multiple benchmarks on our dataset for change detection, change type and status classification tasks. Project page: https://sagarverma.github.io/qfabric

count=1
* Progressive Attention on Multi-Level Dense Difference Maps for Generic Event Boundary Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Progressive_Attention_on_Multi-Level_Dense_Difference_Maps_for_Generic_Event_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Progressive_Attention_on_Multi-Level_Dense_Difference_Maps_for_Generic_Event_CVPR_2022_paper.pdf)]
    * Title: Progressive Attention on Multi-Level Dense Difference Maps for Generic Event Boundary Detection
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jiaqi Tang, Zhaoyang Liu, Chen Qian, Wayne Wu, Limin Wang
    * Abstract: Generic event boundary detection is an important yet challenging task in video understanding, which aims at detecting the moments where humans naturally perceive event boundaries. The main challenge of this task is perceiving various temporal variations of diverse event boundaries. To this end, this paper presents an effective and end-to-end learnable framework (DDM-Net). To tackle the diversity and complicated semantics of event boundaries, we make three notable improvements. First, we construct a feature bank to store multi-level features of space and time, prepared for difference calculation at multiple scales. Second, to alleviate inadequate temporal modeling of previous methods, we present dense difference maps (DDM) to comprehensively characterize the motion pattern. Finally, we exploit progressive attention on multi-level DDM to jointly aggregate appearance and motion clues. As a result, DDM-Net respectively achieves a significant boost of 14% and 8% on Kinetics-GEBD and TAPOS benchmark, and outperforms the top-1 winner solution of LOVEU Challenge@CVPR 2021 without bells and whistles. The state-of-the-art result demonstrates the effectiveness of richer motion representation and more sophisticated aggregation, in handling the diversity of generic event boundary detection. The code is made available at https://github.com/MCG-NJU/DDM.

count=1
* GeoEngine: A Platform for Production-Ready Geospatial Research
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Verma_GeoEngine_A_Platform_for_Production-Ready_Geospatial_Research_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Verma_GeoEngine_A_Platform_for_Production-Ready_Geospatial_Research_CVPR_2022_paper.pdf)]
    * Title: GeoEngine: A Platform for Production-Ready Geospatial Research
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Sagar Verma, Siddharth Gupta, Hal Shin, Akash Panigrahi, Shubham Goswami, Shweta Pardeshi, Natanael Exe, Ujwal Dutta, Tanka Raj Joshi, Nitin Bhojwani
    * Abstract: Geospatial machine learning has seen tremendous academic advancement, but its practical application has been constrained by difficulties with operationalizing performant and reliable solutions. Sourcing satellite imagery in real-world settings, handling terabytes of training data, and managing machine learning artifacts are a few of the challenges that have severely limited downstream innovation. In this paper we introduce the GeoEngine platform for reproducible and production-ready geospatial machine learning research. GeoEngine removes key technical hurdles to adopting computer vision and deep learning-based geospatial solutions at scale. It is the first end-to-end geospatial machine learning platform, simplifying access to insights locked behind petabytes of imagery. Backed by a rigorous research methodology, this geospatial framework empowers researchers with powerful abstractions for image sourcing, dataset development, model development, large scale training, and model deployment. In this paper we provide the GeoEngine architecture explaining our design rationale in detail. We provide several real-world use cases of image sourcing, dataset development, and model building that have helped different organisations build and deploy geospatial solutions.

count=1
* Segmenting Across Places: The Need for Fair Transfer Learning With Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/html/Zhang_Segmenting_Across_Places_The_Need_for_Fair_Transfer_Learning_With_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Zhang_Segmenting_Across_Places_The_Need_for_Fair_Transfer_Learning_With_CVPRW_2022_paper.pdf)]
    * Title: Segmenting Across Places: The Need for Fair Transfer Learning With Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Miao Zhang, Harvineet Singh, Lazarus Chok, Rumi Chunara
    * Abstract: The increasing availability of high-resolution satellite imagery has enabled the use of machine learning to support land-cover measurement and inform policy-making. However, labelling satellite images is expensive and is available for only some locations. This prompts the use of transfer learning to adapt models from data-rich locations to others. Given the potential for high-impact applications of satellite imagery across geographies, a systematic assessment of transfer learning implications is warranted. In this work, we consider the task of land-cover segmentation and study the fairness implications of transferring models across locations. We leverage a large satellite image segmentation benchmark with 5987 images from 18 districts (9 urban and 9 rural). Via fairness metrics we quantify disparities in model performance along two axes -- across urban-rural locations and across land-cover classes. Findings show that state-of-the-art models have better overall accuracy in rural areas compared to urban areas, through unsupervised domain adaptation methods transfer learning better to urban versus rural areas and enlarge fairness gaps. In analysis of reasons for these findings, we show that raw satellite images are overall more dissimilar between source and target districts for rural than for urban locations. This work highlights the need to conduct fairness analysis for satellite imagery segmentation models and motivates the development of methods for fair transfer learning in order not to introduce disparities between places, particularly urban and rural locations.

count=1
* Token Boosting for Robust Self-Supervised Visual Transformer Pre-Training
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Token_Boosting_for_Robust_Self-Supervised_Visual_Transformer_Pre-Training_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Token_Boosting_for_Robust_Self-Supervised_Visual_Transformer_Pre-Training_CVPR_2023_paper.pdf)]
    * Title: Token Boosting for Robust Self-Supervised Visual Transformer Pre-Training
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Tianjiao Li, Lin Geng Foo, Ping Hu, Xindi Shang, Hossein Rahmani, Zehuan Yuan, Jun Liu
    * Abstract: Learning with large-scale unlabeled data has become a powerful tool for pre-training Visual Transformers (VTs). However, prior works tend to overlook that, in real-world scenarios, the input data may be corrupted and unreliable. Pre-training VTs on such corrupted data can be challenging, especially when we pre-train via the masked autoencoding approach, where both the inputs and masked "ground truth" targets can potentially be unreliable in this case. To address this limitation, we introduce the Token Boosting Module (TBM) as a plug-and-play component for VTs that effectively allows the VT to learn to extract clean and robust features during masked autoencoding pre-training. We provide theoretical analysis to show how TBM improves model pre-training with more robust and generalizable representations, thus benefiting downstream tasks. We conduct extensive experiments to analyze TBM's effectiveness, and results on four corrupted datasets demonstrate that TBM consistently improves performance on downstream tasks.

count=1
* Change-Aware Sampling and Contrastive Learning for Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf)]
    * Title: Change-Aware Sampling and Contrastive Learning for Satellite Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Utkarsh Mall, Bharath Hariharan, Kavita Bala
    * Abstract: Automatic remote sensing tools can help inform many large-scale challenges such as disaster management, climate change, etc. While a vast amount of spatio-temporal satellite image data is readily available, most of it remains unlabelled. Without labels, this data is not very useful for supervised learning algorithms. Self-supervised learning instead provides a way to learn effective representations for various downstream tasks without labels. In this work, we leverage characteristics unique to satellite images to learn better self-supervised features. Specifically, we use the temporal signal to contrast images with long-term and short-term differences, and we leverage the fact that satellite images do not change frequently. Using these characteristics, we formulate a new loss contrastive loss called Change-Aware Contrastive (CACo) Loss. Further, we also present a novel method of sampling different geographical regions. We show that leveraging these properties leads to better performance on diverse downstream tasks. For example, we see a 6.5% relative improvement for semantic segmentation and an 8.5% relative improvement for change detection over the best-performing baseline with our method.

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
* Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.pdf)]
    * Title: Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ke Li, Di Wang, Zhangyuan Hu, Wenxuan Zhu, Shaofeng Li, Quan Wang
    * Abstract: Deep Convolutional Neural Networks (DCNNs) have achieved remarkable performance in synthetic aperture radar (SAR) object detection but this comes at the cost of tremendous computational resources partly due to extracting redundant features within a single convolutional layer. Recent works either delve into model compression methods or focus on the carefully-designed lightweight models both of which result in performance degradation. In this paper we propose an efficient convolution module for SAR object detection called SFS-Conv which increases feature diversity within each convolutional layer through a shunt-perceive-select strategy. Specifically we shunt input feature maps into space and frequency aspects. The former perceives the context of various objects by dynamically adjusting receptive field while the latter captures abundant frequency variations and textural features via fractional Gabor transformer. To adaptively fuse features from space and frequency aspects a parameter-free feature selection module is proposed to ensure that the most representative and distinctive information are preserved. With SFS-Conv we build a lightweight SAR object detection network called SFS-CNet. Experimental results show that SFS-CNet outperforms state-of-the-art (SoTA) models on a series of SAR object detection benchmarks while simultaneously reducing both the model size and computational cost.

count=1
* CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_CorrMatch_Label_Propagation_via_Correlation_Matching_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_CorrMatch_Label_Propagation_via_Correlation_Matching_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf)]
    * Title: CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Boyuan Sun, Yuqi Yang, Le Zhang, Ming-Ming Cheng, Qibin Hou
    * Abstract: This paper presents a simple but performant semi-supervised semantic segmentation approach called CorrMatch. Previous approaches mostly employ complicated training strategies to leverage unlabeled data but overlook the role of correlation maps in modeling the relationships between pairs of locations. We observe that the correlation maps not only enable clustering pixels of the same category easily but also contain good shape information which previous works have omitted. Motivated by these we aim to improve the use efficiency of unlabeled data by designing two novel label propagation strategies. First we propose to conduct pixel propagation by modeling the pairwise similarities of pixels to spread the high-confidence pixels and dig out more. Then we perform region propagation to enhance the pseudo labels with accurate class-agnostic masks extracted from the correlation maps. CorrMatch achieves great performance on popular segmentation benchmarks. Taking the DeepLabV3+ with ResNet-101 backbone as our segmentation model we receive a 76%+ mIoU score on the Pascal VOC 2012 dataset with only 92 annotated images. Code is available at https://github.com/BBBBchan/CorrMatch .

count=1
* High-fidelity Person-centric Subject-to-Image Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_High-fidelity_Person-centric_Subject-to-Image_Synthesis_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_High-fidelity_Person-centric_Subject-to-Image_Synthesis_CVPR_2024_paper.pdf)]
    * Title: High-fidelity Person-centric Subject-to-Image Synthesis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yibin Wang, Weizhong Zhang, Jianwei Zheng, Cheng Jin
    * Abstract: Current subject-driven image generation methods encounter significant challenges in person-centric image generation. The reason is that they learn the semantic scene and person generation by fine-tuning a common pre-trained diffusion which involves an irreconcilable training imbalance. Precisely to generate realistic persons they need to sufficiently tune the pre-trained model which inevitably causes the model to forget the rich semantic scene prior and makes scene generation over-fit to the training data. Moreover even with sufficient fine-tuning these methods can still not generate high-fidelity persons since joint learning of the scene and person generation also lead to quality compromise. In this paper we propose Face-diffuser an effective collaborative generation pipeline to eliminate the above training imbalance and quality compromise. Specifically we first develop two specialized pre-trained diffusion models i.e. Text-driven Diffusion Model (TDM) and Subject-augmented Diffusion Model (SDM) for scene and person generation respectively. The sampling process is divided into three sequential stages i.e. semantic scene construction subject-scene fusion and subject enhancement. The first and last stages are performed by TDM and SDM respectively. The subject-scene fusion stage that is the collaboration achieved through a novel and highly effective mechanism Saliency-adaptive Noise Fusion (SNF). Specifically it is based on our key observation that there exists a robust link between classifier-free guidance responses and the saliency of generated images. In each time step SNF leverages the unique strengths of each model and allows for the spatial blending of predicted noises from both models automatically in a saliency-aware manner all of which can be seamlessly integrated into the DDIM sampling process. Extensive experiments confirm the impressive effectiveness and robustness of the Face-diffuser in generating high-fidelity person images depicting multiple unseen persons with varying contexts. Code is available at https://github.com/CodeGoat24/Face-diffuser.

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
* Unmixing Diffusion for Self-Supervised Hyperspectral Image Denoising
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zeng_Unmixing_Diffusion_for_Self-Supervised_Hyperspectral_Image_Denoising_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zeng_Unmixing_Diffusion_for_Self-Supervised_Hyperspectral_Image_Denoising_CVPR_2024_paper.pdf)]
    * Title: Unmixing Diffusion for Self-Supervised Hyperspectral Image Denoising
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Haijin Zeng, Jiezhang Cao, Kai Zhang, Yongyong Chen, Hiep Luong, Wilfried Philips
    * Abstract: Hyperspectral images (HSIs) have extensive applications in various fields such as medicine agriculture and industry. Nevertheless acquiring high signal-to-noise ratio HSI poses a challenge due to narrow-band spectral filtering. Consequently the importance of HSI denoising is substantial especially for snapshot hyperspectral imaging technology. While most previous HSI denoising methods are supervised creating supervised training datasets for the diverse scenes hyperspectral cameras and scan parameters is impractical. In this work we present Diff-Unmix a self-supervised denoising method for HSI using diffusion denoising generative models. Specifically Diff-Unmix addresses the challenge of recovering noise-degraded HSI through a fusion of Spectral Unmixing and conditional abundance generation. Firstly it employs a learnable block-based spectral unmixing strategy complemented by a pure transformer-based backbone. Then we introduce a self-supervised generative diffusion network to enhance abundance maps from the spectral unmixing block. This network reconstructs noise-free Unmixing probability distributions effectively mitigating noise-induced degradations within these components. Finally the reconstructed HSI is reconstructed through unmixing reconstruction by blending the diffusion-adjusted abundance map with the spectral endmembers. Experimental results on both simulated and real-world noisy datasets show that Diff-Unmix achieves state-of-the-art performance.

count=1
* CUE-Net: Violence Detection Video Analytics with Spatial Cropping Enhanced UniformerV2 and Modified Efficient Additive Attention
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/ABAW/html/Senadeera_CUE-Net_Violence_Detection_Video_Analytics_with_Spatial_Cropping_Enhanced_UniformerV2_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/ABAW/papers/Senadeera_CUE-Net_Violence_Detection_Video_Analytics_with_Spatial_Cropping_Enhanced_UniformerV2_CVPRW_2024_paper.pdf)]
    * Title: CUE-Net: Violence Detection Video Analytics with Spatial Cropping Enhanced UniformerV2 and Modified Efficient Additive Attention
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Damith Chamalke Senadeera, Xiaoyun Yang, Dimitrios Kollias, Gregory Slabaugh
    * Abstract: In this paper we introduce CUE-Net a novel architecture designed for automated violence detection in video surveillance. As surveillance systems become more prevalent due to technological advances and decreasing costs the challenge of efficiently monitoring vast amounts of video data has intensified. CUE-Net addresses this challenge by combining spatial Cropping with an enhanced version of the UniformerV2 architecture integrating convolutional and self-attention mechanisms alongside a novel Modified Efficient Additive Attention mechanism (which reduces the quadratic time complexity of self-attention) to effectively and efficiently identify violent activities. This approach aims to overcome traditional challenges such as capturing distant or partially obscured subjects within video frames. By focusing on both local and global spatiotemporal features CUE-Net achieves state-of-the-art performance on the RWF-2000 and RLVS datasets surpassing existing methods.

count=1
* GeoLLM-Engine: A Realistic Environment for Building Geospatial Copilots
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Singh_GeoLLM-Engine_A_Realistic_Environment_for_Building_Geospatial_Copilots_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Singh_GeoLLM-Engine_A_Realistic_Environment_for_Building_Geospatial_Copilots_CVPRW_2024_paper.pdf)]
    * Title: GeoLLM-Engine: A Realistic Environment for Building Geospatial Copilots
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Simranjit Singh, Michael Fore, Dimitrios Stamoulis
    * Abstract: Geospatial Copilots unlock unprecedented potential for performing Earth Observation (EO) applications through natural language instructions. However existing agents rely on overly simplified single tasks and template-based prompts creating a disconnect with real-world scenarios. In this work we present GeoLLM-Engine an environment for tool-augmented agents with intricate tasks routinely executed by analysts on remote sensing platforms. We enrich our environment with geospatial API tools dynamic maps/UIs and external multimodal knowledge bases to properly gauge an agent's proficiency in interpreting realistic high-level natural language commands and its functional correctness in task completions. By alleviating overheads typically associated with human-in-the-loop benchmark curation we harness our massively parallel engine across 100 GPT-4-Turbo nodes scaling to over half a million diverse multi-tool tasks and across 1.1 million satellite images. By moving beyond traditional single-task image-caption paradigms we investigate state-of-the-art agents and prompting techniques against long-horizon prompts.

count=1
* Revisiting Pre-trained Remote Sensing Model Benchmarks: Resizing and Normalization Matters
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Corley_Revisiting_Pre-trained_Remote_Sensing_Model_Benchmarks_Resizing_and_Normalization_Matters_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Corley_Revisiting_Pre-trained_Remote_Sensing_Model_Benchmarks_Resizing_and_Normalization_Matters_CVPRW_2024_paper.pdf)]
    * Title: Revisiting Pre-trained Remote Sensing Model Benchmarks: Resizing and Normalization Matters
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Isaac Corley, Caleb Robinson, Rahul Dodhia, Juan M. Lavista Ferres, Peyman Najafirad
    * Abstract: Research in self-supervised learning (SSL) with natural images has progressed rapidly in recent years and is now increasingly being applied to and benchmarked with datasets containing remotely sensed imagery. A common benchmark case is to evaluate SSL pre-trained model embeddings on datasets of remotely sensed imagery with small patch sizes e.g. 32 x 32 pixels whereas standard SSL pre-training takes place with larger patch sizes e.g. 224 x 224. Furthermore pre-training methods tend to use different image normalization preprocessing steps depending on the dataset. In this paper we show across seven satellite and aerial imagery datasets of varying resolution that by simply following the preprocessing steps used in pre-training (precisely image sizing and normalization methods) one can achieve significant performance improvements when evaluating the extracted features on downstream tasks -- an important detail overlooked in previous work in this space. We show that by following these steps ImageNet pre-training remains a competitive baseline for satellite imagery based transfer learning tasks -- for example we find that these steps give +32.28 to overall accuracy on the So2Sat random split dataset and +11.16 on the EuroSAT dataset. Finally we report comprehensive benchmark results with a variety of simple baseline methods for each of the seven datasets forming an initial benchmark suite for remote sensing imagery.

count=1
* Guided Anisotropic Diffusion and Iterative Learning for Weakly Supervised Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/EarthVision/Daudt_Guided_Anisotropic_Diffusion_and_Iterative_Learning_for_Weakly_Supervised_Change_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/EarthVision/Daudt_Guided_Anisotropic_Diffusion_and_Iterative_Learning_for_Weakly_Supervised_Change_CVPRW_2019_paper.pdf)]
    * Title: Guided Anisotropic Diffusion and Iterative Learning for Weakly Supervised Change Detection
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Rodrigo Caye Daudt,  Bertrand Le Saux,  Alexandre Boulch,  Yann Gousseau
    * Abstract: Large scale datasets created from user labels or openly available data have become crucial to provide training data for large scale learning algorithms. While these datasets are easier to acquire, the data are frequently noisy and unreliable, which is motivating research on weakly supervised learning techniques. In this paper we propose an iterative learning method that extracts the useful information from a large scale change detection dataset generated from open vector data to train a fully convolutional network which surpasses the performance obtained by naive supervised learning. We also propose the guided anisotropic diffusion algorithm, which improves semantic segmentation results using the input images as guides to perform edge preserving filtering, and is used in conjunction with the iterative training method to improve results.

count=1
* Superpixel-Based 3D Building Model Refinement and Change Detection, Using VHR Stereo Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/WiCV/Gharibbafghi_Superpixel-Based_3D_Building_Model_Refinement_and_Change_Detection_Using_VHR_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WiCV/Gharibbafghi_Superpixel-Based_3D_Building_Model_Refinement_and_Change_Detection_Using_VHR_CVPRW_2019_paper.pdf)]
    * Title: Superpixel-Based 3D Building Model Refinement and Change Detection, Using VHR Stereo Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Zeinab Gharibbafghi,  Jiaojiao Tian,  Peter Reinartz
    * Abstract: Buildings are one of the main objects in urban remote sensing and photogrammetric computer vision applications using satellite data. In this paper a superpixel-based approach is presented to refine 3D building models from stereo satellite imagery. First, for each epoch in time, a multispectral very high resolution (VHR) satellite image is segmented using an efficient superpixel, called edge-based simple linear iterative clustering (ESLIC). The ESLIC algorithm segments the image utilizing the spectral and spatial information, as well as the statistical measures from the gray-level co-occurrence matrix (GLCM), simultaneously. Then the resulting superpixels are imposed on the corresponding 3D model of the scenes taken from each epoch. Since ESLIC has high capability of preserving edges in the image, normalized digital surface models (nDSMs) can be modified by averaging height values inside superpixels. These new normalized models for epoch 1 and epoch 2, are then used to detect the 3D change of each building in the scene.

count=1
* iTASK - Intelligent Traffic Analysis Software Kit
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w35/Tran_iTASK_-_Intelligent_Traffic_Analysis_Software_Kit_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Tran_iTASK_-_Intelligent_Traffic_Analysis_Software_Kit_CVPRW_2020_paper.pdf)]
    * Title: iTASK - Intelligent Traffic Analysis Software Kit
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Minh-Triet Tran, Tam V. Nguyen, Trung-Hieu Hoang, Trung-Nghia Le, Khac-Tuan Nguyen, Dat-Thanh Dinh, Thanh-An Nguyen, Hai-Dang Nguyen, Xuan-Nhat Hoang, Trong-Tung Nguyen, Viet-Khoa Vo-Ho, Trong-Le Do, Lam Nguyen, Minh-Quan Le, Hoang-Phuc Nguyen-Dinh, Trong-Thang Pham, Xuan-Vy Nguyen, E-Ro Nguyen, Quoc-Cuong Tran, Hung Tran, Hieu Dao, Mai-Khiem Tran, Quang-Thuc Nguyen, Tien-Phat Nguyen, The-Anh Vu-Le, Gia-Han Diep, Minh N. Do
    * Abstract: Traffic flow analysis is essential for intelligent transportation systems. In this paper, we introduce our Intelligent Traffic Analysis Software Kit (iTASK) to tackle three challenging problems: vehicle flow counting, vehicle re-identification, and abnormal event detection. For the first problem, we propose to real-time track vehicles moving along the desired direction in corresponding motion-of-interests (MOIs). For the second problem, we consider each vehicle as a document with multiple semantic words (i.e., vehicle attributes) and transform the given problem to classical document retrieval. For the last problem, we propose to forward and backward refine anomaly detection using GAN-based future prediction and backward tracking completely stalled vehicle or sudden-change direction, respectively. Experiments on the datasets of traffic flow analysis from AI City Challenge 2020 show our competitive results, namely, S1 score of 0.8297 for vehicle flow counting in Track 1, mAP score of 0.3882 for vehicle re-identification in Track 2, and S4 score of 0.9059 for anomaly detection in Track 4. All data and source code are publicly available on our project page.

count=1
* Calibrated Vehicle Paint Signatures for Simulating Hyperspectral Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Mulhollan_Calibrated_Vehicle_Paint_Signatures_for_Simulating_Hyperspectral_Imagery_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w6/Mulhollan_Calibrated_Vehicle_Paint_Signatures_for_Simulating_Hyperspectral_Imagery_CVPRW_2020_paper.pdf)]
    * Title: Calibrated Vehicle Paint Signatures for Simulating Hyperspectral Imagery
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zachary Mulhollan, Aneesh Rangnekar, Timothy Bauch, Matthew J. Hoffman, Anthony Vodacek
    * Abstract: We investigate a procedure for rapidly adding calibrated vehicle visible-near infrared (VNIR) paint signatures to an existing hyperspectral simulator - The Digital Imaging and Remote Sensing Image Generation (DIRSIG) model - to create more diversity in simulated urban scenes. The DIRSIG model can produce synthetic hyperspectral imagery with user-specified geometry, atmospheric conditions, and ground target spectra. To render an object pixel's spectral signature, DIRSIG uses a large database of reflectance curves for the corresponding object material and a bidirectional reflectance model to introduce s due to orientation and surface structure. However, this database contains only a few spectral curves for vehicle paints and generates new paint signatures by combining these curves internally. In this paper we demonstrate a method to rapidly generate multiple paint spectra, flying a drone carrying a pushbroom hyperspectral camera to image a university parking lot. We then process the images to convert them from the digital count space to spectral reflectance without the need of calibration panels in the scene, and port the paint signatures into DIRSIG for successful integration into the newly rendered sets of synthetic VNIR hyperspectral scenes.

count=1
* A Multi-sensor Fusion Framework in 3-D
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/html/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/papers/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.pdf)]
    * Title: A Multi-sensor Fusion Framework in 3-D
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Vishal Jain, Andrew C. Miller, Joseph L. Mundy
    * Abstract: The majority of existing image fusion techniques operate in the 2-d image domain which perform well for imagery of planar regions but fails in presence of any 3-d relief and provides inaccurate alignment of imagery from different sensors. A framework for multi-sensor image fusion in 3-d is proposed in this paper. The imagery from different sensors, specifically EO and IR, are fused in a common 3-d reference coordinate frame. A dense probabilistic and volumetric 3-d model is reconstructed from each of the sensors. The imagery is registered by aligning the 3-d models as the underlying 3-d structure in the images is the true invariant information. The image intensities are back-projected onto a 3-d model and every discretized location (voxel) of the 3-d model stores an array of intensities from different modalities. This 3-d model is forward-projected to produce a fused image of EO and IR from any viewpoint.

count=1
* Change Detection with Weightless Neural Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Gregorio_Change_Detection_with_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Gregorio_Change_Detection_with_2014_CVPR_paper.pdf)]
    * Title: Change Detection with Weightless Neural Networks
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Massimo De Gregorio, Maurizio Giordano
    * Abstract: In this paper a pixel'based Weightless Neural Network (WNN) method to face the problem of change detection in the field of view of a camera is proposed. The main features of the proposed method are 1) the dynamic adaptability to background change due to the WNN model adopted and 2) the introduction of pixel color histories to improve system behavior in videos characterized by (des)appearing of objects in video scene and/or sudden changes in lightning and background brightness and shape. The WNN approach is very simple and straightforward, and it gives high rank results in competition with other approaches applied to the ChangeDetection.net 2014 benchmark dataset.

count=1
* Flexible Background Subtraction With Self-Balanced Local Sensitivity
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf)]
    * Title: Flexible Background Subtraction With Self-Balanced Local Sensitivity
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: Most background subtraction approaches offer decent results in baseline scenarios, but adaptive and flexible solutions are still uncommon as many require scenario-specific parameter tuning to achieve optimal performance. In this paper, we introduce a new strategy to tackle this problem that focuses on balancing the inner workings of a non-parametric model based on pixel-level feedback loops. Pixels are modeled using a spatiotemporal feature descriptor for increased sensitivity. Using the video sequences and ground truth annotations of the 2012 and 2014 CVPR Change Detection Workshops, we demonstrate that our approach outperforms all previously ranked methods in the original dataset while achieving good results in the most recent one.

count=1
* Road Segmentation Using Multipass Single-Pol Synthetic Aperture Radar Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/Koch_Road_Segmentation_Using_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/Koch_Road_Segmentation_Using_2015_CVPR_paper.pdf)]
    * Title: Road Segmentation Using Multipass Single-Pol Synthetic Aperture Radar Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Mark W. Koch, Mary M. Moya, James G. Chow, Jeremy Goold, Rebecca Malinas
    * Abstract: Synthetic aperture radar (SAR) is a remote sensing technology that can truly operate 24/7. It's an all-weather system that can operate at any time except in the most extreme conditions. By making multiple passes over a wide area, a SAR can provide surveillance over a long time period. For high level processing it is convenient to segment and classify the SAR images into objects that identify various terrains and man-made structures that we call "static features." In this paper we concentrate on automatic road segmentation. This not only serves as a surrogate for finding other static features, but road detection in of itself is important for aligning SAR images with other data sources. In this paper we introduce a novel SAR image product that captures how different regions decorrelate at different rates. We also show how a modified Kolmogorov-Smirnov test can be used to model the static features even when the independent observation assumption is violated.

count=1
* Online Multimodal Video Registration Based on Shape Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/St-Charles_Online_Multimodal_Video_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/St-Charles_Online_Multimodal_Video_2015_CVPR_paper.pdf)]
    * Title: Online Multimodal Video Registration Based on Shape Matching
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: The registration of video sequences captured using different types of sensors often relies on dense feature matching methods, which are very costly. In this paper, we study the problem of "almost planar" scene registration (i.e. where the planar ground assumption is almost respected) in multimodal imagery using target shape information. We introduce a new strategy for robustly aligning scene elements based on the random sampling of shape contour correspondences and on the continuous update of our transformation model's parameters. We evaluate our solution on a public dataset and show its superiority by comparing it to a recently published method that targets the same problem. To make comparisons between such methods easier in the future, we provide our evaluation tools along with a full implementation of our solution online.

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
* Robust Kronecker-Decomposable Component Analysis for Low-Rank Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Bahri_Robust_Kronecker-Decomposable_Component_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Bahri_Robust_Kronecker-Decomposable_Component_ICCV_2017_paper.pdf)]
    * Title: Robust Kronecker-Decomposable Component Analysis for Low-Rank Modeling
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Mehdi Bahri, Yannis Panagakis, Stefanos Zafeiriou
    * Abstract: Dictionary learning and component analysis are part of one of the most well-studied and active research fields, at the intersection of signal and image processing, computer vision, and statistical machine learning. In dictionary learning, the current methods of choice are arguably K-SVD and its variants, which learn a dictionary (i.e., a decomposition) for sparse coding via Singular Value Decomposition. In robust component analysis, leading methods derive from Principal Component Pursuit (PCP), which recovers a low-rank matrix from sparse corruptions of unknown magnitude and support. However, K-SVD is sensitive to the presence of noise and outliers in the training set. Additionally, PCP does not provide a dictionary that respects the structure of the data (e.g., images), and requires expensive SVD computations when solved by convex relaxation. In this paper, we introduce a new robust decomposition of images by combining ideas from sparse dictionary learning and PCP. We propose a novel Kronecker-decomposable component analysis which is robust to gross corruption, can be used for low-rank modeling, and leverages separability to solve significantly smaller problems. We design an efficient learning algorithm by drawing links with a restricted form of tensor factorization. The effectiveness of the proposed approach is demonstrated on real-world applications, namely background subtraction and image denoising, by performing a thorough comparison with the current state of the art.

count=1
* Deep Occlusion Reasoning for Multi-Camera Multi-Target Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Baque_Deep_Occlusion_Reasoning_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Baque_Deep_Occlusion_Reasoning_ICCV_2017_paper.pdf)]
    * Title: Deep Occlusion Reasoning for Multi-Camera Multi-Target Detection
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Pierre Baque, Francois Fleuret, Pascal Fua
    * Abstract: People detection in 2D images has improved greatly in recent years. However, comparatively little of this progress has percolated into multi-camera multi-people tracking algorithms, whose performance still degrades severely when scenes become very crowded. In this work, we introduce a new architecture that combines Convolutional Neural Nets and Conditional Random Fields to explicitly resolve ambiguities. One of its key ingredients are high-order CRF terms that model potential occlusions and give our approach its robustness even when many people are present. Our model is trained end-to-end and we show that it outperforms several state-of-the-art algorithms on challenging scenes.

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
* Unmasking the Abnormal Events in Video
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Ionescu_Unmasking_the_Abnormal_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Ionescu_Unmasking_the_Abnormal_ICCV_2017_paper.pdf)]
    * Title: Unmasking the Abnormal Events in Video
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Radu Tudor Ionescu, Sorina Smeureanu, Bogdan Alexe, Marius Popescu
    * Abstract: We propose a novel framework for abnormal event detection in video that requires no training sequences. Our framework is based on unmasking, a technique previously used for authorship verification in text documents, which we adapt to our task. We iteratively train a binary classifier to distinguish between two consecutive video sequences while removing at each step the most discriminant features. Higher training accuracy rates of the intermediately obtained classifiers represent abnormal events. To the best of our knowledge, this is the first work to apply unmasking for a computer vision task. We compare our method with several state-of-the-art supervised and unsupervised methods on four benchmark data sets. The empirical results indicate that our abnormal event detection framework can achieve state-of-the-art results, while running in real-time at 20 frames per second.

count=1
* Moving Object Detection in Time-Lapse or Motion Trigger Image Sequences Using Low-Rank and Invariant Sparse Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Shakeri_Moving_Object_Detection_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Shakeri_Moving_Object_Detection_ICCV_2017_paper.pdf)]
    * Title: Moving Object Detection in Time-Lapse or Motion Trigger Image Sequences Using Low-Rank and Invariant Sparse Decomposition
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Moein Shakeri, Hong Zhang
    * Abstract: Low-rank and sparse representation based methods have attracted wide attention in background subtraction and moving object detection, where moving objects in the scene are modeled as pixel-wise sparse outliers. Since in real scenarios moving objects are also structurally sparse, recently researchers have attempted to extract moving objects using structured sparse outliers. Although existing methods with structured sparsity-inducing norms produce promising results, they are still vulnerable to various illumination changes that frequently occur in real environments, specifically for time-lapse image sequences where assumptions about sparsity between images such as group sparsity are not valid. In this paper, we first introduce a prior map obtained by illumination invariant representation of images. Next, we propose a low-rank and invariant sparse decomposition using the prior map to detect moving objects under significant illumination changes. Experiments on challenging benchmark datasets demonstrate the superior performance of our proposed method under complex illumination changes.

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
* Mutual Foreground Segmentation With Multispectral Stereo Pairs
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w6/html/St-Charles_Mutual_Foreground_Segmentation_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w6/St-Charles_Mutual_Foreground_Segmentation_ICCV_2017_paper.pdf)]
    * Title: Mutual Foreground Segmentation With Multispectral Stereo Pairs
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: Foreground-background segmentation of video sequences is a low-level process commonly used in machine vision, and highly valued in video content analysis and smart surveillance applications. Its efficacy relies on the contrast between objects observed by the sensor. In this work, we study how the combination of sensors operating in the long-wavelength infrared (LWIR) and visible spectra can improve the performance of foreground-background segmentation methods. As opposed to a classic visible spectrum stereo pair, this multispectral pair is more adequate for object segmentation since it reduces the odds of observing low-contrast regions simultaneously in both images. We show that by alternately minimizing stereo disparity and binary segmentation energies with dynamic priors, we can drastically improve the results of a traditional video segmentation approach applied to each sensor individually. Our implementation is freely available online for anyone wishing to recreate our results.

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
* Generic Event Boundary Detection: A Benchmark for Event Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Shou_Generic_Event_Boundary_Detection_A_Benchmark_for_Event_Segmentation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Shou_Generic_Event_Boundary_Detection_A_Benchmark_for_Event_Segmentation_ICCV_2021_paper.pdf)]
    * Title: Generic Event Boundary Detection: A Benchmark for Event Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Mike Zheng Shou, Stan Weixian Lei, Weiyao Wang, Deepti Ghadiyaram, Matt Feiszli
    * Abstract: This paper presents a novel task together with a new benchmark for detecting generic, taxonomy-free event boundaries that segment a whole video into chunks. Conventional work in temporal video segmentation and action detection focuses on localizing pre-defined action categories and thus does not scale to generic videos. Cognitive Science has known since last century that humans consistently segment videos into meaningful temporal chunks. This segmentation happens naturally, without pre-defined event categories and without being explicitly asked to do so. Here, we repeat these cognitive experiments on mainstream CV datasets; with our novel annotation guideline which addresses the complexities of taxonomy-free event boundary annotation, we introduce the task of Generic Event Boundary Detection (GEBD) and the new benchmark Kinetics-GEBD. We view GEBD as an important stepping stone towards understanding the video as a whole, and believe it has been previously neglected due to a lack of proper task definition and annotations. Through experiment and human study we demonstrate the value of the annotations. Further, we benchmark supervised and un-supervised GEBD approaches on the TAPOS dataset and our Kinetics-GEBD. We release our annotations and baseline codes at CVPR'21 LOVEU Challenge: https://sites.google.com/view/loveucvpr21.

count=1
* Background/Foreground Separation: Guided Attention Based Adversarial Modeling (GAAM) Versus Robust Subspace Learning Methods
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Sultana_BackgroundForeground_Separation_Guided_Attention_Based_Adversarial_Modeling_GAAM_Versus_Robust_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Sultana_BackgroundForeground_Separation_Guided_Attention_Based_Adversarial_Modeling_GAAM_Versus_Robust_ICCVW_2021_paper.pdf)]
    * Title: Background/Foreground Separation: Guided Attention Based Adversarial Modeling (GAAM) Versus Robust Subspace Learning Methods
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Maryam Sultana, Arif Mahmood, Thierry Bouwmans, Muhammad Haris Khan, Soon Ki Jung
    * Abstract: Background-Foreground separation and appearance generation is a fundamental step in many computer vision applications. Existing methods like Robust Subspace Learning (RSL) suffer performance degradation in the presence of challenges like bad weather, illumination variations, occlusion, dynamic backgrounds and intermittent object motion. In the current work we propose a more accurate deep neural network based model for background-foreground separation and complete appearance generation of the foreground objects. Our proposed model, Guided Attention based Adversarial Model (GAAM), can efficiently extract pixel-level boundaries of the foreground objects for improved appearance generation. Unlike RSL methods our model extracts the binary information of foreground objects labeled as attention map which guides our generator network to segment the foreground objects from the complex background information. Wide range of experiments performed on the benchmark CDnet2014 dataset demonstrate the excellent performance of our proposed model.

count=1
* Learning-Based Shadow Detection in Aerial Imagery Using Automatic Training Supervision From 3D Point Clouds
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/html/Ufuktepe_Learning-Based_Shadow_Detection_in_Aerial_Imagery_Using_Automatic_Training_Supervision_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/papers/Ufuktepe_Learning-Based_Shadow_Detection_in_Aerial_Imagery_Using_Automatic_Training_Supervision_ICCVW_2021_paper.pdf)]
    * Title: Learning-Based Shadow Detection in Aerial Imagery Using Automatic Training Supervision From 3D Point Clouds
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Deniz Kavzak Ufuktepe, Jaired Collins, Ekincan Ufuktepe, Joshua Fraser, Timothy Krock, Kannappan Palaniappan
    * Abstract: Shadows, motion parallax, and occlusions pose significant challenges to vision tasks in wide area motion imagery (WAMI) including object identification and tracking. Although there are many successful shadow detection approaches that work well in indoor scenes, close range outdoor scenes, and spaceborne satellite images, the methods tend to fail in intermediate altitude aerial WAMI. We propose an automatic shadow mask estimation approach for supervision without manual labeling to provide a large amount of training data for learning-based aerial shadow extraction. Analytical ground-truth shadow masks are generated using 3D point clouds combined with known solar angles. FSDNet, a deep network for shadow detection, is evaluated on aerial imagery. Preliminary results indicate that training using automated shadow mask supervision improves performance, and opens the door for developing new deep architectures for shadow detection and enhancement in WAMI.

count=1
* PanFlowNet: A Flow-Based Deep Network for Pan-Sharpening
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_PanFlowNet_A_Flow-Based_Deep_Network_for_Pan-Sharpening_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_PanFlowNet_A_Flow-Based_Deep_Network_for_Pan-Sharpening_ICCV_2023_paper.pdf)]
    * Title: PanFlowNet: A Flow-Based Deep Network for Pan-Sharpening
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Gang Yang, Xiangyong Cao, Wenzhe Xiao, Man Zhou, Aiping Liu, Xun Chen, Deyu Meng
    * Abstract: Pan-sharpening aims to generate a high-resolution multispectral (HRMS) image by integrating the spectral information of a low-resolution multispectral (LRMS) image with the texture details of a high-resolution panchromatic (PAN) image. It essentially inherits the ill-posed nature of the super-resolution (SR) task that diverse HRMS images can degrade into an LRMS image. However, existing deep learning-based methods recover only one HRMS image from the LRMS image and PAN image using a deterministic mapping, thus ignoring the diversity of the HRMS image. In this paper, to alleviate this ill-posed issue, we propose a flow-based pan-sharpening network (PanFlowNet) to directly learn the conditional distribution of HRMS image given LRMS image and PAN image instead of learning a deterministic mapping. Specifically, we first transform this unknown conditional distribution into a given Gaussian distribution by an invertible network, and the conditional distribution can thus be explicitly defined. Then, we design an invertible Conditional Affine Coupling Block (CACB) and further build the architecture of PanFlowNet by stacking a series of CACBs. Finally, the PanFlowNet is trained by maximizing the log-likelihood of the conditional distribution given a training set and can then be used to predict diverse HRMS images. The experimental results verify that the proposed PanFlowNet can generate various HRMS images given an LRMS image and a PAN image. Additionally, the experimental results on different kinds of satellite datasets also demonstrate the superiority of our PanFlowNet compared with other state-of-the-art methods both visually and quantitatively. Code is available at Github.

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
* MotionRec: A Unified Deep Framework for Moving Object Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Mandal_MotionRec_A_Unified_Deep_Framework_for_Moving_Object_Recognition_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Mandal_MotionRec_A_Unified_Deep_Framework_for_Moving_Object_Recognition_WACV_2020_paper.pdf)]
    * Title: MotionRec: A Unified Deep Framework for Moving Object Recognition
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Murari Mandal,  Lav Kush Kumar,  Mahipal Singh Saran,  Santosh Kumar vipparthi
    * Abstract: In this paper we present a novel deep learning framework to perform online moving object recognition (MOR) in streaming videos. The existing methods for moving object detection (MOD) only computes class-agnostic pixel-wise binary segmentation of video frames. On the other hand, the object detection techniques do not differentiate between static and moving objects. To the best of our knowledge, this is a first attempt for simultaneous localization and classification of moving objects in a video, i.e. MOR in a single-stage deep learning framework. We achieve this by labelling axis-aligned bounding boxes for moving objects which requires less computational resources than producing pixel-level estimates. In the proposed MotionRec, both temporal and spatial features are learned using past history and current frames respectively. First, the background is estimated with a temporal depth reductionist (TDR) block. Then the estimated background, current frame and temporal median of recent observations are assimilated to encode spatiotemporal motion saliency. Moreover, feature pyramids are generated from these motion saliency maps to perform regression and classification at multiple levels of feature abstractions. MotionRec works online at inference as it requires only few past frames for MOR. Moreover, it doesn't require predefined target initialization from user. We also annotated axis-aligned bounding boxes (42,614 objects (14,814 cars and 27,800 person) in 24,923 video frames of CDnet 2014 dataset) due to lack of available benchmark datasets for MOR. The performance is observed qualitatively and quantitatively in terms of mAP over a defined unseen test set. Experiments show that the proposed MotionRec significantly improves over strong baselines with RetinaNet architectures for MOR.

count=1
* Active Learning for Improved Semi-Supervised Semantic Segmentation in Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Desai_Active_Learning_for_Improved_Semi-Supervised_Semantic_Segmentation_in_Satellite_Images_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Desai_Active_Learning_for_Improved_Semi-Supervised_Semantic_Segmentation_in_Satellite_Images_WACV_2022_paper.pdf)]
    * Title: Active Learning for Improved Semi-Supervised Semantic Segmentation in Satellite Images
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Shasvat Desai, Debasmita Ghose
    * Abstract: Remote sensing data is crucial for applications ranging from monitoring forest fires and deforestation to tracking urbanization. Most of these tasks require dense pixel-level annotations for the model to parse visual information from limited labeled data available for these satellite images. Due to the dearth of high-quality labeled training data in this domain, there is a need to focus on semi-supervised techniques. These techniques generate pseudo-labels from a small set of labeled examples which are used to augment the labeled training set. This makes it necessary to have a highly representative and diverse labeled training set. Therefore, we propose to use an active learning-based sampling strategy to select a highly representative set of labeled training data. We demonstrate our proposed method's effectiveness on two existing semantic segmentation datasets containing satellite images: UC Merced Land Use Classification Dataset and DeepGlobe Land Cover Classification Dataset. We report a 27% improvement in mIoU with as little as 2% labeled data using active learning sampling strategies over randomly sampling the small set of labeled training data.

count=1
* Towards Interpretable Video Anomaly Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.pdf)]
    * Title: Towards Interpretable Video Anomaly Detection
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Keval Doshi, Yasin Yilmaz
    * Abstract: Most video anomaly detection approaches are based on data-intensive end-to-end trained neural networks, which extract spatiotemporal features from videos. The extracted feature representations in such approaches are not interpretable, which prevents the automatic identification of anomaly cause. To this end, we propose a novel framework which can explain the detected anomalous event in a surveillance video. In addition to monitoring objects independently, we also monitor the interactions between them to detect anomalous events and explain their root causes. Specifically, we demonstrate that the scene graphs obtained by monitoring the object interactions provide an interpretation for the context of the anomaly while performing competitively with respect to the recent state-of-the-art approaches. Moreover, the proposed interpretable method enables cross-domain adaptability (i.e., transfer learning in another surveillance scene), which is not feasible for most existing end-to-end methods due to the lack of sufficient labeled training data for every surveillance scene. The quick and reliable detection performance of the proposed method is evaluated both theoretically (through an asymptotic optimality proof) and empirically on the popular benchmark datasets.

count=1
* GAF-Net: Improving the Performance of Remote Sensing Image Fusion Using Novel Global Self and Cross Attention Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Jha_GAF-Net_Improving_the_Performance_of_Remote_Sensing_Image_Fusion_Using_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Jha_GAF-Net_Improving_the_Performance_of_Remote_Sensing_Image_Fusion_Using_WACV_2023_paper.pdf)]
    * Title: GAF-Net: Improving the Performance of Remote Sensing Image Fusion Using Novel Global Self and Cross Attention Learning
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Ankit Jha, Shirsha Bose, Biplab Banerjee
    * Abstract: The notion of self and cross-attention learning has been found to substantially boost the performance of remote sensing (RS) image fusion. However, while the self-attention models fail to incorporate the global context due to the limited size of the receptive fields, cross-attention learning may generate ambiguous features as the feature extractors for all the modalities are jointly trained. This results in the generation of redundant multi-modal features, thus limiting the fusion performance. To address these issues, we propose a novel fusion architecture called Global Attention based Fusion Network (GAF-Net), equipped with novel self and cross-attention learning techniques. We introduce the within-modality feature refinement module through global spectral-spatial attention learning using the query-key-value processing where both the global spatial and channel contexts are used to generate two channel attention masks. Since it is non-trivial to generate the cross-attention from within the fusion network, we propose to leverage two auxiliary tasks of modality-specific classification to produce highly discriminative cross-attention masks. Finally, to ensure non-redundancy, we propose to penalize the high correlation between attended modality-specific features. Our extensive experiments on five benchmark datasets, including optical, multispectral (MS), hyperspectral (HSI), light detection and ranging (LiDAR), synthetic aperture radar (SAR), and audio modalities establish the superiority of GAF-Net concerning the literature.

count=1
* The Change You Want To See
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Sachdeva_The_Change_You_Want_To_See_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Sachdeva_The_Change_You_Want_To_See_WACV_2023_paper.pdf)]
    * Title: The Change You Want To See
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Ragav Sachdeva, Andrew Zisserman
    * Abstract: We live in a dynamic world where things change all the time. Given two images of the same scene, being able to automatically detect the changes in them has practical applications in a variety of domains. In this paper, we tackle the change detection problem with the goal of detecting "object-level" changes in an image pair despite differences in their viewpoint and illumination. To this end, we make the following four contributions: (i) we propose a scalable methodology for obtaining a large-scale change detection training dataset by leveraging existing object segmentation benchmarks; (ii) we introduce a co-attention based novel architecture that is able to implicitly determine correspondences between an image pair and find changes in the form of bounding box predictions; (iii) we contribute four evaluation datasets that cover a variety of domains and transformations, including synthetic image changes, real surveillance images of a 3D scene, and synthetic 3D scenes with camera motion; (iv) we evaluate our model on these four datasets and demonstrate zero-shot and beyond training transformation generalization. The code, datasets and pre-trained model can be found at our project page: https://www.robots.ox.ac.uk/ vgg/research/cyws/

count=1
* Self-Pair: Synthesizing Changes From Single Source for Object Change Detection in Remote Sensing Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Seo_Self-Pair_Synthesizing_Changes_From_Single_Source_for_Object_Change_Detection_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Seo_Self-Pair_Synthesizing_Changes_From_Single_Source_for_Object_Change_Detection_WACV_2023_paper.pdf)]
    * Title: Self-Pair: Synthesizing Changes From Single Source for Object Change Detection in Remote Sensing Imagery
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Minseok Seo, Hakjin Lee, Yongjin Jeon, Junghoon Seo
    * Abstract: For change detection in remote sensing, constructing a training dataset for deep learning models is quite difficult due to the requirements of bi-temporal supervision. To overcome this issue, single-temporal supervision which treats change labels as the difference of two semantic masks has been proposed. This novel method trains a change detector using two spatially unrelated images with corresponding semantic labels. However, training with unpaired dataset shows not enough performance compared with other methods based on bi-temporal supervision. We suspect this phenomenon caused by ignorance of meaningful information in the actual bi-temporal pairs.In this paper, we emphasize that the change originates from the source image and show that manipulating the source image as an after-image is crucial to the performance of change detection. Our method achieves state-of-the-art performance in a large gap than existing methods.

count=1
* Semi-Supervised Scene Change Detection by Distillation From Feature-Metric Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Lee_Semi-Supervised_Scene_Change_Detection_by_Distillation_From_Feature-Metric_Alignment_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Lee_Semi-Supervised_Scene_Change_Detection_by_Distillation_From_Feature-Metric_Alignment_WACV_2024_paper.pdf)]
    * Title: Semi-Supervised Scene Change Detection by Distillation From Feature-Metric Alignment
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Seonhoon Lee, Jong-Hwan Kim
    * Abstract: Scene change detection (SCD) is a critical task for various applications, such as visual surveillance, anomaly detection, and mobile robotics. Recently, supervised methods for SCD have been developed for urban and indoor environments where input image pairs are typically unaligned due to differences in camera viewpoints. However, supervised SCD methods require pixel-wise change labels and alignment labels for the target domain, which can be both time-consuming and expensive to collect. To tackle this issue, we design an unsupervised loss with regularization methods based on the feature-metric alignment of input image pairs. The proposed unsupervised loss enables the SCD model to jointly learn the flow and the change maps on the target domain. In addition, we propose a semi-supervised learning method based on a distillation loss for the robustness of the SCD model. The proposed learning method is based on the student-teacher structure and incorporates the unsupervised loss of the unlabeled target data and the supervised loss of the labeled synthetic data. Our method achieves considerable performance improvement on the target domain through the proposed unsupervised and distillation loss, using only 10% of the target training dataset without using any labels of the target data.

count=1
* Precision and Recall for Time Series
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2018/file/8f468c873a32bb0619eaeb2050ba45d1-Paper.pdf)]
    * Title: Precision and Recall for Time Series
    * Publisher: NeurIPS
    * Publication Date: `2018`
    * Authors: Nesime Tatbul, Tae Jun Lee, Stan Zdonik, Mejbah Alam, Justin Gottschlich
    * Abstract: Classical anomaly detection is principally concerned with point-based anomalies, those anomalies that occur at a single point in time. Yet, many real-world anomalies are range-based, meaning they occur over a period of time. Motivated by this observation, we present a new mathematical model to evaluate the accuracy of time series classification algorithms. Our model expands the well-known Precision and Recall metrics to measure ranges, while simultaneously enabling customization support for domain-specific preferences.

count=1
* Trading robust representations for sample complexity through self-supervised visual experience
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2018/hash/c344336196d5ec19bd54fd14befdde87-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2018/file/c344336196d5ec19bd54fd14befdde87-Paper.pdf)]
    * Title: Trading robust representations for sample complexity through self-supervised visual experience
    * Publisher: NeurIPS
    * Publication Date: `2018`
    * Authors: Andrea Tacchetti, Stephen Voinea, Georgios Evangelopoulos
    * Abstract: Learning in small sample regimes is among the most remarkable features of the human perceptual system. This ability is related to robustness to transformations, which is acquired through visual experience in the form of weak- or self-supervision during development. We explore the idea of allowing artificial systems to learn representations of visual stimuli through weak supervision prior to downstream supervised tasks. We introduce a novel loss function for representation learning using unlabeled image sets and video sequences, and experimentally demonstrate that these representations support one-shot learning and reduce the sample complexity of multiple recognition tasks. We establish the existence of a trade-off between the sizes of weakly supervised, automatically obtained from video sequences, and fully supervised data sets. Our results suggest that equivalence sets other than class labels, which are abundant in unlabeled visual experience, can be used for self-supervised learning of semantically relevant image embeddings.

count=1
* Staying up to Date with Online Content Changes Using Reinforcement Learning for Scheduling
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/ad13a2a07ca4b7642959dc0c4c740ab6-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/ad13a2a07ca4b7642959dc0c4c740ab6-Paper.pdf)]
    * Title: Staying up to Date with Online Content Changes Using Reinforcement Learning for Scheduling
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Andrey Kolobov, Yuval Peres, Cheng Lu, Eric J. Horvitz
    * Abstract: From traditional Web search engines to virtual assistants and Web accelerators, services that rely on online information need to continually keep track of remote content changes by explicitly requesting content updates from remote sources (e.g., web pages). We propose a novel optimization objective for this setting that has several practically desirable properties, and efficient algorithms for it with optimality guarantees even in the face of mixed content change observability and initially unknown change model parameters. Experiments on 18.5M URLs crawled daily for 14 weeks show significant advantages of this approach over prior art.

count=1
* A Greek Parliament Proceedings Dataset for Computational Linguistics and Political Analysis
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/b96ce67b2f2d45e4ab315e13a6b5b9c5-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: A Greek Parliament Proceedings Dataset for Computational Linguistics and Political Analysis
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Konstantina Dritsa, Aikaterini Thoma, Ioannis Pavlopoulos, Panos Louridas
    * Abstract: Large, diachronic datasets of political discourse are hard to come across, especially for resource-lean languages such as Greek. In this paper, we introduce a curated dataset of the Greek Parliament Proceedings that extends chronologically from 1989 up to 2020. It consists of more than 1 million speeches with extensive meta-data, extracted from 5,355 parliamentary sitting record files. We explain how it was constructed and the challenges that had to be overcome. The dataset can be used for both computational linguistics and political analysis---ideally, combining the two. We present such an application, showing (i) how the dataset can be used to study the change of word usage through time, (ii) between significant historical events and political parties, (iii) by evaluating and employing algorithms for detecting semantic shifts.

count=1
* Detection and Localization of Changes in Conditional Distributions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/eb189151ced0ff808abafd16a51fec92-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/eb189151ced0ff808abafd16a51fec92-Paper-Conference.pdf)]
    * Title: Detection and Localization of Changes in Conditional Distributions
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Lizhen Nie, Dan Nicolae
    * Abstract: We study the change point problem that considers alterations in the conditional distribution of an inferential target on a set of covariates. This paired data scenario is in contrast to the standard setting where a sequentially observed variable is analyzed for potential changes in the marginal distribution. We propose new methodology for solving this problem, by starting from a simpler task that analyzes changes in conditional expectation, and generalizing the tools developed for that task to conditional distributions. Large sample properties of the proposed statistics are derived. In empirical studies, we illustrate the performance of the proposed method against baselines adapted from existing tools. Two real data applications are presented to demonstrate its potential.

count=1
* SIXO: Smoothing Inference with Twisted Objectives
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/fddc79681b2df2734c01444f9bc2a17e-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/fddc79681b2df2734c01444f9bc2a17e-Paper-Conference.pdf)]
    * Title: SIXO: Smoothing Inference with Twisted Objectives
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Dieterich Lawson, Allan Raventós, Andrew Warrington, Scott Linderman
    * Abstract: Sequential Monte Carlo (SMC) is an inference algorithm for state space models that approximates the posterior by sampling from a sequence of target distributions. The target distributions are often chosen to be the filtering distributions, but these ignore information from future observations, leading to practical and theoretical limitations in inference and model learning. We introduce SIXO, a method that instead learns target distributions that approximate the smoothing distributions, incorporating information from all observations. The key idea is to use density ratio estimation to fit functions that warp the filtering distributions into the smoothing distributions. We then use SMC with these learned targets to define a variational objective for model and proposal learning. SIXO yields provably tighter log marginal lower bounds and offers more accurate posterior inferences and parameter estimates in a variety of domains.

count=1
* Change point detection and inference in multivariate non-parametric models under mixing conditions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/42a0de6b8a1809ceba8fdad1661be06c-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/42a0de6b8a1809ceba8fdad1661be06c-Paper-Conference.pdf)]
    * Title: Change point detection and inference in multivariate non-parametric models under mixing conditions
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, OSCAR HERNAN MADRID PADILLA, Yi Yu
    * Abstract: This paper addresses the problem of localizing and inferring multiple change points, in non-parametric multivariate time series settings. Specifically, we consider a multivariate time series with potentially short-range dependence, whose underlying distributions have Hölder smooth densities and can change over time in a piecewise-constant manner. The change points, which correspond to the times when the distribution changes, are unknown. We present the limiting distributions of the change point estimators under the scenarios where the minimal jump size vanishes or remains constant. Such results have not been revealed in the literature in non-parametric change point settings. As byproducts, we develop a sharp estimator that can accurately localize the change points in multivariate non-parametric time series, and a consistent block-type long-run variance estimator. Numerical studies are provided to complement our theoretical findings.

