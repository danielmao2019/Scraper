count=17
* CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf)]
    * Title: CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Anthony Fuller, Koreen Millard, James Green
    * Abstract: A vital and rapidly growing application, remote sensing offers vast yet sparsely labeled, spatially aligned multimodal data; this makes self-supervised learning algorithms invaluable. We present CROMA: a framework that combines contrastive and reconstruction self-supervised objectives to learn rich unimodal and multimodal representations. Our method separately encodes masked-out multispectral optical and synthetic aperture radar samples—aligned in space and time—and performs cross-modal contrastive learning. Another encoder fuses these sensors, producing joint multimodal encodings that are used to predict the masked patches via a lightweight decoder. We show that these objectives are complementary when leveraged on spatially aligned multimodal data. We also introduce X- and 2D-ALiBi, which spatially biases our cross- and self-attention matrices. These strategies improve representations and allow our models to effectively extrapolate to images up to $17.6\times$ larger at test-time. CROMA outperforms the current SoTA multispectral model, evaluated on: four classification benchmarks—finetuning (avg.$\uparrow$ 1.8%), linear (avg.$\uparrow$ 2.4%) and nonlinear (avg.$\uparrow$ 1.4%) probing, $k$NN classification (avg.$\uparrow$ 3.5%), and $K$-means clustering (avg.$\uparrow$ 8.4%); and three segmentation benchmarks (avg.$\uparrow$ 6.4%). CROMA’s rich, optionally multimodal representations can be widely leveraged across remote sensing applications.

count=7
* Progressive Unsupervised Deep Transfer Learning for Forest Mapping in Satellite Image
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/html/Ahmed_Progressive_Unsupervised_Deep_Transfer_Learning_for_Forest_Mapping_in_Satellite_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/papers/Ahmed_Progressive_Unsupervised_Deep_Transfer_Learning_for_Forest_Mapping_in_Satellite_ICCVW_2021_paper.pdf)]
    * Title: Progressive Unsupervised Deep Transfer Learning for Forest Mapping in Satellite Image
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Nouman Ahmed, Sudipan Saha, Muhammad Shahzad, Muhammad Moazam Fraz, Xiao Xiang Zhu
    * Abstract: Automated forest mapping is important to understand our forests that play a key role in ecological system. However, efforts towards forest mapping is impeded by difficulty to collect labeled forest images that show large intraclass variation. Recently unsupervised learning has shown promising capability when exploiting limited labeled data. Motivated by this, we propose a progressive unsupervised deep transfer learning method for forest mapping. The proposed method exploits a pre-trained model that is subsequently fine-tuned over the target forest domain. We propose two different fine-tuning mechanism, one works in a totally unsupervised setting by jointly learning the parameters of CNN and the k-means based cluster assignments of the resulting features and the other one works in a semi-supervised setting by exploiting the extracted knearest neighbor based pseudo labels. The proposed progressive scheme is evaluated on publicly available EuroSAT dataset using the relevant base model trained on BigEarthNet labels. The results show that the proposed method greatly improves the forest regions classification accuracy as compared to the unsupervised baseline, nearly approaching the supervised classification approach.

count=6
* Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Temporally_Consistent_Unbalanced_Optimal_Transport_for_Unsupervised_Action_Segmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Temporally_Consistent_Unbalanced_Optimal_Transport_for_Unsupervised_Action_Segmentation_CVPR_2024_paper.pdf)]
    * Title: Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ming Xu, Stephen Gould
    * Abstract: We propose a novel approach to the action segmentation task for long untrimmed videos based on solving an optimal transport problem. By encoding a temporal consistency prior into a Gromov-Wasserstein problem we are able to decode a temporally consistent segmentation from a noisy affinity/matching cost matrix between video frames and action classes. Unlike previous approaches our method does not require knowing the action order for a video to attain temporal consistency. Furthermore our resulting (fused) Gromov-Wasserstein problem can be efficiently solved on GPUs using a few iterations of projected mirror descent. We demonstrate the effectiveness of our method in an unsupervised learning setting where our method is used to generate pseudo-labels for self-training. We evaluate our segmentation approach and unsupervised learning pipeline on the Breakfast 50-Salads YouTube Instructions and Desktop Assembly datasets yielding state-of-the-art results for the unsupervised video action segmentation task.

count=6
* DiSC: Differential Spectral Clustering of Features
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/a84953147312ea2e8b020e53a267321b-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/a84953147312ea2e8b020e53a267321b-Paper-Conference.pdf)]
    * Title: DiSC: Differential Spectral Clustering of Features
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Ram Dyuthi Sristi, Gal Mishne, Ariel Jaffe
    * Abstract: Selecting subsets of features that differentiate between two conditions is a key task in a broad range of scientific domains. In many applications, the features of interest form clusters with similar effects on the data at hand. To recover such clusters we develop DiSC, a data-driven approach for detecting groups of features that differentiate between conditions. For each condition, we construct a graph whose nodes correspond to the features and whose weights are functions of the similarity between them for that condition. We then apply a spectral approach to compute subsets of nodes whose connectivity pattern differs significantly between the condition-specific feature graphs. On the theoretical front, we analyze our approach with a toy example based on the stochastic block model. We evaluate DiSC on a variety of datasets, including MNIST, hyperspectral imaging, simulated scRNA-seq and task fMRI, and demonstrate that DiSC uncovers features that better differentiate between conditions compared to competing methods.

count=4
* Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Teutsch_Robust_Detection_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Teutsch_Robust_Detection_of_CVPR_2016_paper.pdf)]
    * Title: Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Michael Teutsch, Michael Grinberg
    * Abstract: Multiple object tracking in Wide Area Motion Imagery (WAMI) data is usually based on initial detections coming from background subtraction or frame differencing. However, these methods are prone to produce split and merged detections. Appearance based vehicle detection can be an alternative but is not well-suited for WAMI data since classifier models are of weak discriminative power for vehicles in top view at low resolution. We introduce a moving vehicle detection algorithm that combines 2-frame differencing with a vehicle appearance model to improve object detection. Our main contributions are (1) integration of robust vehicle detection with split/merge handling and (2) estimation of assignment likelihoods between object hypotheses in consecutive frames using an appearance based similarity measure. Without using any prior knowledge, we achieve state-of-the-art detection rates and produce tracklets that considerably simplify the data association problem for multiple object tracking.

count=3
* Articulated Motion Discovery Using Pairs of Trajectories
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Pero_Articulated_Motion_Discovery_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Pero_Articulated_Motion_Discovery_2015_CVPR_paper.pdf)]
    * Title: Articulated Motion Discovery Using Pairs of Trajectories
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Luca Del Pero, Susanna Ricco, Rahul Sukthankar, Vittorio Ferrari
    * Abstract: We propose an unsupervised approach for discovering characteristic motion patterns in videos of highly articulated objects performing natural, unscripted behaviors, such as tigers in the wild. We discover consistent patterns in a bottom-up manner by analyzing the relative displacements of large numbers of ordered trajectory pairs through time, such that each trajectory is attached to a different moving part on the object. The pairs of trajectories descriptor relies entirely on motion and is more discriminative than state-of-the-art features that employ single trajectories. Our method generates temporal video intervals, each automatically trimmed to one instance of the discovered behavior, and clusters them by type (e.g., running, turning head, drinking water). We present experiments on two datasets: dogs from YouTube-Objects and a new dataset of National Geographic tiger videos. Results confirm that our proposed descriptor outperforms existing appearance- and trajectory-based descriptors (e.g., HOG and DTFs) on both datasets and enables us to segment unconstrained animal video into intervals containing single behaviors.

count=3
* An Efficient Approach for Anomaly Detection in Traffic Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AICity/html/Doshi_An_Efficient_Approach_for_Anomaly_Detection_in_Traffic_Videos_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AICity/papers/Doshi_An_Efficient_Approach_for_Anomaly_Detection_in_Traffic_Videos_CVPRW_2021_paper.pdf)]
    * Title: An Efficient Approach for Anomaly Detection in Traffic Videos
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Keval Doshi, Yasin Yilmaz
    * Abstract: Due to its relevance in intelligent transportation systems, anomaly detection in traffic videos has recently received much interest. It remains a difficult problem due to a variety of factors influencing the video quality of a real-time traffic feed, such as temperature, perspective, lighting conditions, and so on. Even though state-of-the-art methods perform well on the available benchmark datasets, they need a large amount of external training data as well as substantial computational resources. In this paper, we propose an efficient approach for a video anomaly detection system which is capable of running at the edge devices, e.g., on a roadside camera. The proposed approach comprises a pre-processing module that detects changes in the scene and removes the corrupted frames, a two-stage background modelling module and a two-stage object detector. Finally, a backtracking anomaly detection algorithm computes a similarity statistic and decides on the onset time of the anomaly. We also propose a sequential change detection algorithm that can quickly adapt to a new scene and detect changes in the similarity statistic. Experimental results on the Track 4 test set of the 2021 AI City Challenge show the efficacy of the proposed framework as we achieve an F1-score of 0.9157 along with 8.4027 root mean square error (RMSE) and are ranked fourth in the competition.

count=3
* Improving Speaker Turn Embedding by Crossmodal Transfer Learning From Face Embedding
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w8/html/Le_Improving_Speaker_Turn_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w8/Le_Improving_Speaker_Turn_ICCV_2017_paper.pdf)]
    * Title: Improving Speaker Turn Embedding by Crossmodal Transfer Learning From Face Embedding
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Nam Le, Jean-Marc Odobez
    * Abstract: Learning speaker turn embeddings has shown considerable improvement in situations where conventional speaker modeling approaches fail. However, this improvement is relatively limited when compared to the gain observed in face embedding learning, which has proven very successful for face verification and clustering tasks. Assuming that face and voices from the same identities share some latent properties (like age, gender, ethnicity), we propose two transfer learning approaches to leverage the knowledge from the face domain learned from thousands of images and identities for tasks in the speaker domain. These approaches, namely target embedding transfer and clustering structure transfer, utilize the structure of the source face embedding space at different granularities to regularize the target speaker turn embedding space as optimizing terms. Our methods are evaluated on two public broadcast corpora and yield promising advances over competitive baselines in verification and audio clustering tasks, especially when dealing with short speaker utterances. The analysis of the results also gives insight into characteristics of the embedding spaces and shows their potential applications.

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
* NestedVAE: Isolating Common Factors via Weak Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.pdf)]
    * Title: NestedVAE: Isolating Common Factors via Weak Supervision
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Matthew J. Vowels,  Necati Cihan Camgoz,  Richard Bowden
    * Abstract: Fair and unbiased machine learning is an important and active field of research, as decision processes are increasingly driven by models that learn from data. Unfortunately, any biases present in the data may be learned by the model, thereby inappropriately transferring that bias into the decision making process. We identify the connection between the task of bias reduction and that of isolating factors common between domains whilst encouraging domain specific invariance. To isolate the common factors we combine the theory of deep latent variable models with information bottleneck theory for scenarios whereby data may be naturally paired across domains and no additional supervision is required. The result is the Nested Variational AutoEncoder (NestedVAE). Two outer VAEs with shared weights attempt to reconstruct the input and infer a latent space, whilst a nested VAE attempts to reconstruct the latent representation of one image, from the latent representation of its paired image. In so doing, the nested VAE isolates the common latent factors/causes and becomes invariant to unwanted factors that are not shared between paired images. We also propose a new metric to provide a balanced method of evaluating consistency and classifier performance across domains which we refer to as the Adjusted Parity metric. An evaluation of NestedVAE on both domain and attribute invariance, change detection, and learning common factors for the prediction of biological sex demonstrates that NestedVAE significantly outperforms alternative methods.

count=2
* Semantic-Aware Domain Generalized Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.pdf)]
    * Title: Semantic-Aware Domain Generalized Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Duo Peng, Yinjie Lei, Munawar Hayat, Yulan Guo, Wen Li
    * Abstract: Deep models trained on source domain lack generalization when evaluated on unseen target domains with different data distributions. The problem becomes even more pronounced when we have no access to target domain samples for adaptation. In this paper, we address domain generalized semantic segmentation, where a segmentation model is trained to be domain-invariant without using any target domain data. Existing approaches to tackle this problem standardize data into a unified distribution. We argue that while such a standardization promotes global normalization, the resulting features are not discriminative enough to get clear segmentation boundaries. To enhance separation between categories while simultaneously promoting domain invariance, we propose a framework including two novel modules: Semantic-Aware Normalization (SAN) and Semantic-Aware Whitening (SAW). Specifically, SAN focuses on category-level center alignment between features from different image styles, while SAW enforces distributed alignment for the already center-aligned features. With the help of SAN and SAW, we encourage both intraclass compactness and inter-class separability. We validate our approach through extensive experiments on widely-used datasets (i.e. GTAV, SYNTHIA, Cityscapes, Mapillary and BDDS). Our approach shows significant improvements over existing state-of-the-art on various backbone networks. Code is available at https://github.com/leolyj/SAN-SAW

count=2
* Self-Supervised Object Detection from Egocentric Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.pdf)]
    * Title: Self-Supervised Object Detection from Egocentric Videos
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Peri Akiva, Jing Huang, Kevin J Liang, Rama Kovvuri, Xingyu Chen, Matt Feiszli, Kristin Dana, Tal Hassner
    * Abstract: Understanding the visual world from the perspective of humans (egocentric) has been a long-standing challenge in computer vision. Egocentric videos exhibit high scene complexity and irregular motion flows compared to typical video understanding tasks. With the egocentric domain in mind, we address the problem of self-supervised, class-agnostic object detection, which aims to locate all objects in a given view, regardless of category, without any annotations or pre-training weights. Our method, self-supervised object Detection from Egocentric VIdeos (DEVI), generalizes appearance-based methods to learn features that are category-specific and invariant to viewing angles and illumination conditions from highly ambiguous environments in an end-to-end manner. Our approach leverages typical human behavior and its egocentric perception to sample diverse views of the same objects for our multi-view and scale-regression loss functions. With our learned cluster residual module, we are able to effectively describe multi-category patches for better complex scene understanding. DEVI provides a boost in performance on recent egocentric datasets, with performance gains up to 4.11% AP50, 0.11% AR1, 1.32% AR10, and 5.03% AR100, while significantly reducing model complexity. We also demonstrate competitive performance on out-of-domain datasets without additional training or fine-tuning.

count=2
* Inverse Filtering for Hidden Markov Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf)]
    * Title: Inverse Filtering for Hidden Markov Models
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Robert Mattila, Cristian Rojas, Vikram Krishnamurthy, Bo Wahlberg
    * Abstract: This paper considers a number of related inverse filtering problems for hidden Markov models (HMMs). In particular, given a sequence of state posteriors and the system dynamics; i) estimate the corresponding sequence of observations, ii) estimate the observation likelihoods, and iii) jointly estimate the observation likelihoods and the observation sequence. We show how to avoid a computationally expensive mixed integer linear program (MILP) by exploiting the algebraic structure of the HMM filter using simple linear algebra operations, and provide conditions for when the quantities can be uniquely reconstructed. We also propose a solution to the more general case where the posteriors are noisily observed. Finally, the proposed inverse filtering algorithms are evaluated on real-world polysomnographic data used for automatic sleep segmentation.

count=2
* Taming Local Effects in Graph-based Spatiotemporal Forecasting
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/ad58c61c71efd5436134a3ecc87da6ea-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/ad58c61c71efd5436134a3ecc87da6ea-Paper-Conference.pdf)]
    * Title: Taming Local Effects in Graph-based Spatiotemporal Forecasting
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Andrea Cini, Ivan Marisca, Daniele Zambon, Cesare Alippi
    * Abstract: Spatiotemporal graph neural networks have shown to be effective in time series forecasting applications, achieving better performance than standard univariate predictors in several settings. These architectures take advantage of a graph structure and relational inductive biases to learn a single (global) inductive model to predict any number of the input time series, each associated with a graph node. Despite the gain achieved in computational and data efficiency w.r.t. fitting a set of local models, relying on a single global model can be a limitation whenever some of the time series are generated by a different spatiotemporal stochastic process. The main objective of this paper is to understand the interplay between globality and locality in graph-based spatiotemporal forecasting, while contextually proposing a methodological framework to rationalize the practice of including trainable node embeddings in such architectures. We ascribe to trainable node embeddings the role of amortizing the learning of specialized components. Moreover, embeddings allow for 1) effectively combining the advantages of shared message-passing layers with node-specific parameters and 2) efficiently transferring the learned model to new node sets. Supported by strong empirical evidence, we provide insights and guidelines for specializing graph-based models to the dynamics of each time series and show how this aspect plays a crucial role in obtaining accurate predictions.

count=1
* Modeling Actions through State Changes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Fathi_Modeling_Actions_through_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Fathi_Modeling_Actions_through_2013_CVPR_paper.pdf)]
    * Title: Modeling Actions through State Changes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Alireza Fathi, James M. Rehg
    * Abstract: In this paper we present a model of action based on the change in the state of the environment. Many actions involve similar dynamics and hand-object relationships, but differ in their purpose and meaning. The key to differentiating these actions is the ability to identify how they change the state of objects and materials in the environment. We propose a weakly supervised method for learning the object and material states that are necessary for recognizing daily actions. Once these state detectors are learned, we can apply them to input videos and pool their outputs to detect actions. We further demonstrate that our method can be used to segment discrete actions from a continuous video of an activity. Our results outperform state-of-the-art action recognition and activity segmentation results.

count=1
* Large-Scale Damage Detection Using Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.pdf)]
    * Title: Large-Scale Damage Detection Using Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Lionel Gueguen, Raffay Hamid
    * Abstract: Satellite imagery is a valuable source of information for assessing damages in distressed areas undergoing a calamity, such as an earthquake or an armed conflict. However, the sheer amount of data required to be inspected for this assessment makes it impractical to do it manually. To address this problem, we present a semi-supervised learning framework for large-scale damage detection in satellite imagery. We present a comparative evaluation of our framework using over 88 million images collected from 4,665 square kilometers from 12 different locations around the world. To enable accurate and efficient damage detection, we introduce a novel use of hierarchical shape features in the bags-of-visual words setting. We analyze how practical factors such as sun, sensor-resolution, and satellite-angle differences impact the effectiveness of our proposed representation, and compare it to five alternative features in multiple learning settings. Finally, we demonstrate through a user-study that our semi-supervised framework results in a ten-fold reduction in human annotation time at a minimal loss in detection accuracy compared to an exhaustive manual inspection.

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
* Detecting Anomalous Objects on Mobile Platforms
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.pdf)]
    * Title: Detecting Anomalous Objects on Mobile Platforms
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Wallace Lawson, Laura Hiatt, Keith Sullivan
    * Abstract: We present an approach where a robot patrols a fixed path through an environment, autonomously locating suspicious or anomalous objects. To learn, the robot patrols this environment building a dictionary describing what is present. The dictionary is built by clustering features from a deep neural network. The objects present vary depending on the scene, which means that an object that is anomalous in one scene may be completely normal in another. To reason about this, the robot uses a computational cognitive model to learn the dictionary elements that are typically found in each scene. Once the dictionary and model has been built, the robot can patrol the environment matching objects against the dictionary, and querying the model to find the most likely objects present and to determine which objects (if any) are anomalous. We demonstrate our approach by patrolling two indoor and one outdoor environments.

count=1
* Temporal Vegetation Modelling Using Long Short-Term Memory Networks for Crop Identification From Medium-Resolution Multi-Spectral Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Russwurm_Temporal_Vegetation_Modelling_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Russwurm_Temporal_Vegetation_Modelling_CVPR_2017_paper.pdf)]
    * Title: Temporal Vegetation Modelling Using Long Short-Term Memory Networks for Crop Identification From Medium-Resolution Multi-Spectral Satellite Images
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Marc Russwurm, Marco Korner
    * Abstract: Land-cover classification is one of the key problems in earth observation and extensively investigated over the recent decades. Usually, approaches concentrate on single-time and multi- or hyperspectral reflectance space- or airborne sensor measurements observed. However, land-cover classes, e.g., crops, change their reflective characteristics over time complicating classification at one observation time. Contrary, these features change in a systematic and predictive manner, which can be utilized in a multi-temporal approach. We use long short-term memory (LSTM) networks to extract temporal characteristics from a sequence of Sentinel-2 observations. We compare the performance of LSTM and other network architectures and a SVM baseline to show the effectiveness of dynamic temporal feature extraction. A large test area combined with rich ground truth labels was used for training and evaluation. Our LSTM variant achieves state-of-the art performance opening potential for further research.

count=1
* Unsupervised Human Action Detection by Action Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/html/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/papers/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.pdf)]
    * Title: Unsupervised Human Action Detection by Action Matching
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Basura Fernando, Sareh Shirazi, Stephen Gould
    * Abstract: We propose a new task of unsupervised action detection by action matching. Given two long videos, the objective is to temporally detect all pairs of matching video segments. A pair of video segments are matched if they share the same human action. The task is category independent---it does not matter what action is being performed---and no supervision is used to discover such video segments. Unsupervised action detection by action matching allows us to align videos in a meaningful manner. As such, it can be used to discover new action categories or as an action proposal technique within, say, an action detection pipeline. We solve this new task using an effective and efficient method. We use an unsupervised temporal encoding method and exploit the temporal consistency in human actions to obtain candidate action segments. We evaluate our method on this challenging task using three activity recognition benchmarks.

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
* Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Video_Rain_Streak_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Video_Rain_Streak_CVPR_2018_paper.pdf)]
    * Title: Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Minghan Li, Qi Xie, Qian Zhao, Wei Wei, Shuhang Gu, Jing Tao, Deyu Meng
    * Abstract: Videos captured by outdoor surveillance equipments sometimes contain unexpected rain streaks, which brings difficulty in subsequent video processing tasks. Rain streak removal from a video is thus an important topic in recent computer vision research. In this paper, we raise two intrinsic characteristics specifically possessed by rain streaks. Firstly, the rain streaks in a video contain repetitive local patterns sparsely scattered over different positions of the video. Secondly, the rain streaks are with multiscale configurations due to their occurrence on positions with different distances to the cameras. Based on such understanding, we specifically formulate both characteristics into a multiscale convolutional sparse coding (MS-CSC) model for the video rain streak removal task. Specifically, we use multiple convolutional filters convolved on the sparse feature maps to deliver the former characteristic, and further use multiscale filters to represent different scales of rain streaks. Such a new encoding manner makes the proposed method capable of properly extracting rain streaks from videos, thus getting fine video deraining effects. Experiments implemented on synthetic and real videos verify the superiority of the proposed method, as compared with the state-of-the-art ones along this research line, both visually and quantitatively.

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
* Moving Object Detection Under Discontinuous Change in Illumination Using Tensor Low-Rank and Invariant Sparse Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Shakeri_Moving_Object_Detection_Under_Discontinuous_Change_in_Illumination_Using_Tensor_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shakeri_Moving_Object_Detection_Under_Discontinuous_Change_in_Illumination_Using_Tensor_CVPR_2019_paper.pdf)]
    * Title: Moving Object Detection Under Discontinuous Change in Illumination Using Tensor Low-Rank and Invariant Sparse Decomposition
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Moein Shakeri,  Hong Zhang
    * Abstract: Although low-rank and sparse decomposition based methods have been successfully applied to the problem of moving object detection using structured sparsity-inducing norms, they are still vulnerable to significant illumination changes that arise in certain applications. We are interested in moving object detection in applications involving time-lapse image sequences for which current methods mistakenly group moving objects and illumination changes into foreground. Our method relies on the multilinear (tensor) data low-rank and sparse decomposition framework to address the weaknesses of existing methods. The key to our proposed method is to create first a set of prior maps that can characterize the changes in the image sequence due to illumination. We show that they can be detected by a k-support norm. To deal with concurrent, two types of changes, we employ two regularization terms, one for detecting moving objects and the other for accounting for illumination changes, in the tensor low-rank and sparse decomposition formulation. Through comprehensive experiments using challenging datasets, we show that our method demonstrates a remarkable ability to detect moving objects under discontinuous change in illumination, and outperforms the state-of-the-art solutions to this challenging problem.

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
* DeepLocalization: Using Change Point Detection for Temporal Action Localization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Rahman_DeepLocalization_Using_Change_Point_Detection_for_Temporal_Action_Localization_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Rahman_DeepLocalization_Using_Change_Point_Detection_for_Temporal_Action_Localization_CVPRW_2024_paper.pdf)]
    * Title: DeepLocalization: Using Change Point Detection for Temporal Action Localization
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Mohammed Shaiqur Rahman, Ibne Farabi Shihab, Lynna Chu, Anuj Sharma
    * Abstract: In this study we introduce DeepLocalization an innovative framework devised for the real-time localization of actions tailored explicitly for monitoring driver behavior. Utilizing the power of advanced deep learning methodologies our objective is to tackle the critical issue of distracted driving--a significant factor contributing to road accidents. Our strategy employs a dual approach: leveraging Graph-Based Change-Point Detection for pinpointing actions in time alongside a Video Large Language Model (Video-LLM) for precisely categorizing activities. Through careful prompt engineering we customize the Video-LLM to adeptly handle driving activities' nuances ensuring its classification efficacy even with sparse data. Engineered to be lightweight our framework is optimized for consumer-grade GPUs making it vastly applicable in practical scenarios. We subjected our method to rigorous testing on the SynDD2 dataset a complex benchmark for distracted driving behaviors where it demonstrated commendable performance--achieving 57.5% accuracy in event classification and 51% in event detection. These outcomes underscore the substantial promise of DeepLocalization in accurately identifying diverse driver behaviors and their temporal occurrences all within the bounds of limited computational resources.

count=1
* Vehicle Re-identification with Learned Representation and Spatial Verification and Abnormality Detection with Multi-Adaptive Vehicle Detectors for Traffic Video Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/AI_City/Nguyen_Vehicle_Re-identification_with_Learned_Representation_and_Spatial_Verification_and_Abnormality_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI City/Nguyen_Vehicle_Re-identification_with_Learned_Representation_and_Spatial_Verification_and_Abnormality_CVPRW_2019_paper.pdf)]
    * Title: Vehicle Re-identification with Learned Representation and Spatial Verification and Abnormality Detection with Multi-Adaptive Vehicle Detectors for Traffic Video Analysis
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Khac-Tuan Nguyen,  Trung-Hieu Hoang,  Minh-Triet Tran,  Trung-Nghia Le,  Ngoc-Minh Bui,  Trong-Le Do,  Viet-Khoa Vo-Ho,  Quoc-An Luong,  Mai-Khiem Tran,  Thanh-An Nguyen,  Thanh-Dat Truong,  Vinh-Tiep Nguyen,  Minh N. Do
    * Abstract: Traffic flow analysis is essential for intelligent transportation systems. In this paper, we propose methods for two challenging problems in traffic flow analysis: vehicle re-identification and abnormal event detection. For the first problem, we propose to combine learned high-level features for vehicle instance representation with hand-crafted local features for spatial verification. For the second problem, we propose to use multiple adaptive vehicle detectors for anomaly proposal and use heuristics properties extracted from anomaly proposals to determine anomaly events. Experiments on the datasets of traffic flow analysis from AI City Challenge 2019 show that our methods achieve mAP of 0.4008 for vehicle re-identification in Track 2, and can detect abnormal events with very high accuracy (F1 = 0.9429) in Track 3.

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
* Meta-Learning for Few-Shot Land Cover Classification
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Russwurm_Meta-Learning_for_Few-Shot_Land_Cover_Classification_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Russwurm_Meta-Learning_for_Few-Shot_Land_Cover_Classification_CVPRW_2020_paper.pdf)]
    * Title: Meta-Learning for Few-Shot Land Cover Classification
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Marc Russwurm, Sherrie Wang, Marco Korner, David Lobell
    * Abstract: The representations of the Earth's surface vary from one geographic region to another. For instance, the appearance of urban areas differs between continents, and seasonality influences the appearance of vegetation. To capture the diversity within a single category, such as urban or vegetation, requires a large model capacity and, consequently, large datasets. In this work, we propose a different perspective and view this diversity as an inductive transfer learning problem where few data samples from one region allow a model to adapt to an unseen region. We evaluate the model-agnostic meta-learning (MAML) algorithm on classification and segmentation tasks using globally and regionally distributed datasets. We find that few-shot model adaptation outperforms pre-training with regular gradient descent and fine-tuning on the (1) Sen12MS dataset and (2) DeepGlobe dataset when the source domain and target domain differ. This indicates that model optimization with meta-learning may benefit tasks in the Earth sciences whose data show a high degree of diversity from region to region, while traditional gradient-based supervised learning remains suitable in the absence of a feature or label shift.

count=1
* Multi-Source Multi-Modal Activity Recognition in Aerial Video Surveillance
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/html/Hammoud_Multi-Source_Multi-Modal_Activity_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/papers/Hammoud_Multi-Source_Multi-Modal_Activity_2014_CVPR_paper.pdf)]
    * Title: Multi-Source Multi-Modal Activity Recognition in Aerial Video Surveillance
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Riad I. Hammoud, Cem S. Sahin, Erik P. Blasch, Bradley J. Rhodes
    * Abstract: Recognizing activities in wide aerial/overhead imagery remains a challenging problem due in part to low-resolution video and cluttered scenes with a large number of moving objects. In the context of this research, we deal with two unsynchronized data sources collected in real-world operating scenarios: full-motion videos (FMV) and analyst call-outs (ACO) in the form of chat messages (voice-to-text) made by a human watching the streamed FMV from an aerial platform. We present a multi-source multi-modal activity/event recognition system for surveillance applications, consisting of: (1) detecting and tracking multiple dynamic targets from a moving platform, (2) representing FMV target tracks and chat messages as graphs of attributes, (3) associating FMV tracks and chat messages using a probabilistic graph-based matching approach, and (4) detecting spatial-temporal activity boundaries. We also present an activity pattern learning framework which uses the multi-source associated data as training to index a large archive of FMV videos. Finally, we describe a multi-intelligence user interface for querying an index of activities of interest (AOIs) by movement type and geo-location, and for playing-back a summary of associated text (ACO) and activity video segments of targets-of-interest (TOIs) (in both pixel and geo-coordinates). Such tools help the end-user to quickly search, browse, and prepare mission reports from multi-source data.

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
* Geography-Aware Self-Supervised Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.pdf)]
    * Title: Geography-Aware Self-Supervised Learning
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell, Stefano Ermon
    * Abstract: Contrastive learning methods have significantly narrowed the gap between supervised and unsupervised learning on computer vision tasks. In this paper, we explore their application to geo-located datasets, e.g. remote sensing, where unlabeled data is often abundant but labeled data is scarce. We first show that due to their different characteristics, a non-trivial gap persists between contrastive and supervised learning on standard benchmarks. To close the gap, we propose novel training methods that exploit the spatio-temporal structure of remote sensing data. We leverage spatially aligned images over time to construct temporal positive pairs in contrastive learning and geo-location to design pre-text tasks. Our experiments show that our proposed method closes the gap between contrastive and supervised learning on image classification, object detection and semantic segmentation for remote sensing. Moreover, we demonstrate that the proposed method can also be applied to geo-tagged ImageNet images, improving downstream performance on various tasks.

count=1
* Word2Fun: Modelling Words as Functions for Diachronic Word Representation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/16a5cdae362b8d27a1d8f8c7b78b4330-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/16a5cdae362b8d27a1d8f8c7b78b4330-Paper.pdf)]
    * Title: Word2Fun: Modelling Words as Functions for Diachronic Word Representation
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Benyou Wang, Emanuele Di Buccio, Massimo Melucci
    * Abstract: Word meaning may change over time as a reflection of changes in human society. Therefore, modeling time in word representation is necessary for some diachronic tasks. Most existing diachronic word representation approaches train the embeddings separately for each pre-grouped time-stamped corpus and align these embeddings, e.g., by orthogonal projections, vector initialization, temporal referencing, and compass. However, not only does word meaning change in a short time, word meaning may also be subject to evolution over long timespans, thus resulting in a unified continuous process. A recent approach called `DiffTime' models semantic evolution as functions parameterized by multiple-layer nonlinear neural networks over time. In this paper, we will carry on this line of work by learning explicit functions over time for each word. Our approach, called `Word2Fun', reduces the space complexity from $\mathcal{O}(TVD)$ to $\mathcal{O}(kVD)$ where $k$ is a small constant ($k \ll T $). In particular, a specific instance based on polynomial functions could provably approximate any function modeling word evolution with a given negligible error thanks to the Weierstrass Approximation Theorem. The effectiveness of the proposed approach is evaluated in diverse tasks including time-aware word clustering, temporal analogy, and semantic change detection. Code at: {\url{https://github.com/wabyking/Word2Fun.git}}.

count=1
* On the Exploration of Local Significant Differences For Two-Sample Test
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/10fc83943b4540a9524af6fc67a23fef-Paper-Conference.pdf)]
    * Title: On the Exploration of Local Significant Differences For Two-Sample Test
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Zhijian Zhou, Jie Ni, Jia-He Yao, Wei Gao
    * Abstract: Recent years have witnessed increasing attentions on two-sample test with diverse real applications, while this work takes one more step on the exploration of local significant differences for two-sample test. We propose the ME$_\text{MaBiD}$, an effective test for two-sample testing, and the basic idea is to exploit local information by multiple Mahalanobis kernels and introduce bi-directional hypothesis for testing. On the exploration of local significant differences, we first partition the embedding space into several rectangle regions via a new splitting criterion, which is relevant to test power and data correlation. We then explore local significant differences based on our bi-directional masked $p$-value together with the ME$_\text{MaBiD}$ test. Theoretically, we present the asymptotic distribution and lower bounds of test power for our ME$_\text{MaBiD}$ test, and control the familywise error rate on the exploration of local significant differences. We finally conduct extensive experiments to validate the effectiveness of our proposed methods on two-sample test and the exploration of local significant differences.

