count=81
* MapFormer: Boosting Change Detection by Using Pre-change Information
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Bernhard_MapFormer_Boosting_Change_Detection_by_Using_Pre-change_Information_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Bernhard_MapFormer_Boosting_Change_Detection_by_Using_Pre-change_Information_ICCV_2023_paper.pdf)]
    * Title: MapFormer: Boosting Change Detection by Using Pre-change Information
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Maximilian Bernhard, Niklas Strauß, Matthias Schubert
    * Abstract: Change detection in remote sensing imagery is essential for a variety of applications such as urban planning, disaster management, and climate research. However, existing methods for identifying semantically changed areas overlook the availability of semantic information in the form of existing maps describing features of the earth's surface. In this paper, we leverage this information for change detection in bi-temporal images. We show that the simple integration of the additional information via concatenation of latent representations suffices to significantly outperform state-of-the-art change detection methods. Motivated by this observation, we propose the new task of Conditional Change Detection, where pre-change semantic information is used as input next to bi-temporal images. To fully exploit the extra information, we propose MapFormer, a novel architecture based on a multi-modal feature fusion module that allows for feature processing conditioned on the available semantic information. We further employ a supervised, cross-modal contrastive loss to guide the learning of visual representations. Our approach outperforms existing change detection methods by an absolute 11.7% and 18.4% in terms of binary change IoU on DynamicEarthNet and HRSCD, respectively. Furthermore, we demonstrate the robustness of our approach to the quality of the pre-change semantic information and the absence pre-change imagery. The code is available at https://github.com/mxbh/mapformer.

count=73
* Change Is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Change_Is_Everywhere_Single-Temporal_Supervised_Object_Change_Detection_in_Remote_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Change_Is_Everywhere_Single-Temporal_Supervised_Object_Change_Detection_in_Remote_ICCV_2021_paper.pdf)]
    * Title: Change Is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zhuo Zheng, Ailong Ma, Liangpei Zhang, Yanfei Zhong
    * Abstract: For high spatial resolution (HSR) remote sensing images, bitemporal supervised learning always dominates change detection using many pairwise labeled bitemporal images. However, it is very expensive and time-consuming to pairwise label large-scale bitemporal HSR remote sensing images. In this paper, we propose single-temporal supervised learning (STAR) for change detection from a new perspective of exploiting object changes in unpaired images as supervisory signals. STAR enables us to train a high-accuracy change detector only using unpaired labeled images and generalize to real-world bitemporal images. To evaluate the effectiveness of STAR, we design a simple yet effective change detector called ChangeStar, which can reuse any deep semantic segmentation architecture by the ChangeMixin module. The comprehensive experimental results show that ChangeStar outperforms the baseline with a large margin under single-temporal supervision and achieves superior performance under bitemporal supervision. Code is available at https://github.com/Z-Zheng/ChangeStar.

count=67
* Learning to Detect Fine-Grained Change Under Variant Imaging Conditions
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w42/html/Huang_Learning_to_Detect_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w42/Huang_Learning_to_Detect_ICCV_2017_paper.pdf)]
    * Title: Learning to Detect Fine-Grained Change Under Variant Imaging Conditions
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Rui Huang, Wei Feng, Zezheng Wang, Mingyuan Fan, Liang Wan, Jizhou Sun
    * Abstract: Fine-grained change detection under variant imaging conditions is an important and challenging task for high-value scene monitoring in culture heritage. In this paper, we show that after a simple coarse alignment of lighting and camera differences, fine-grained change detection can be reliably solved by a deep network model, which is specifically composed of three functional parts, i.e., camera pose correction network (PCN), fine-grained change detection network (FCDN), and detection confidence boosting. Since our model is properly pre-trained and fine-tuned on both general and specialized data, it exhibits very good generalization capability to produce high-quality minute change detection on real-world scenes under varied imaging conditions. Extensive experiments validate the superior effectiveness and reliability over state-of-the-art methods. We have achieved 67.41% relative F1-measure improvement over the best competitor on real-world benchmark dataset.

count=66
* Unsupervised Change Detection Based on Image Reconstruction Loss
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Noh_Unsupervised_Change_Detection_Based_on_Image_Reconstruction_Loss_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Noh_Unsupervised_Change_Detection_Based_on_Image_Reconstruction_Loss_CVPRW_2022_paper.pdf)]
    * Title: Unsupervised Change Detection Based on Image Reconstruction Loss
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Hyeoncheol Noh, Jingi Ju, Minseok Seo, Jongchan Park, Dong-Geol Choi
    * Abstract: To train the change detector, bi-temporal images taken at different times in the same area are used. However, collecting labeled bi-temporal images is expensive and time consuming. To solve this problem, various unsupervised change detection methods have been proposed, but they still require unlabeled bi-temporal images. In this paper, we propose unsupervised change detection based on image reconstruction loss using only unlabeled single temporal single image. The image reconstruction model is trained to reconstruct the original source image by receiving the source image and the photometrically transformed source image as a pair. During inference, the model receives bi-temporal images as the input, and tries to reconstruct one of the inputs. The changed region between bi-temporal images shows high reconstruction loss. Our change detector showed significant performance in various change detection benchmark datasets even though only a single temporal single source image was used. The code and trained models will be publicly available for reproducibility.

count=64
* Fine-Grained Change Detection of Misaligned Scenes With Varied Illuminations
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Feng_Fine-Grained_Change_Detection_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Feng_Fine-Grained_Change_Detection_ICCV_2015_paper.pdf)]
    * Title: Fine-Grained Change Detection of Misaligned Scenes With Varied Illuminations
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Wei Feng, Fei-Peng Tian, Qian Zhang, Nan Zhang, Liang Wan, Jizhou Sun
    * Abstract: Detecting fine-grained subtle changes among a scene is critically important in practice. Previous change detection methods, focusing on detecting large-scale significant changes, cannot do this well. This paper proposes a feasible end-to-end approach to this challenging problem. We start from active camera relocation that quickly relocates camera to nearly the same pose and position of the last time observation. To guarantee detection sensitivity and accuracy of minute changes, in an observation, we capture a group of images under multiple illuminations, which need only to be roughly aligned to the last time lighting conditions. Given two times observations, we formulate fine-grained change detection as a joint optimization problem of three related factors, i.e., normal-aware lighting difference, camera geometry correction flow, and real scene change mask. We solve the three factors in a coarse-to-fine manner and achieve reliable change decision by rank minimization. We build three real-world datasets to benchmark fine-grained change detection of misaligned scenes under varied multiple lighting conditions. Extensive experiments show the superior performance of our approach over state-of-the-art change detection methods and its ability to distinguish real scene changes from false ones caused by lighting variations.

count=59
* Dual Task Learning by Leveraging Both Dense Correspondence and Mis-Correspondence for Robust Change Detection With Imperfect Matches
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.pdf)]
    * Title: Dual Task Learning by Leveraging Both Dense Correspondence and Mis-Correspondence for Robust Change Detection With Imperfect Matches
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jin-Man Park, Ue-Hwan Kim, Seon-Hoon Lee, Jong-Hwan Kim
    * Abstract: Accurate change detection enables a wide range of tasks in visual surveillance, anomaly detection and mobile robotics. However, contemporary change detection approaches assume an ideal matching between the current and stored scenes, whereas only coarse matching is possible in real-world scenarios. Thus, contemporary approaches fail to show the reported performance in real-world settings. To overcome this limitation, we propose SimSaC. SimSaC concurrently conducts scene flow estimation and change detection and is able to detect changes with imperfect matches. To train SimSaC without additional manual labeling, we propose a training scheme with random geometric transformations and the cut-paste method. Moreover, we design an evaluation protocol which reflects performance in real-world settings. In designing the protocol, we collect a test benchmark dataset, which we claim as another contribution. Our comprehensive experiments verify that SimSaC displays robust performance even given imperfect matches and the performance margin compared to contemporary approaches is huge.

count=54
* Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/b01153e7112b347d8ed54f317840d8af-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Utkarsh Mall, Bharath Hariharan, Kavita Bala
    * Abstract: Satellite imagery is increasingly available, high resolution, and temporally detailed. Changes in spatio-temporal datasets such as satellite images are particularly interesting as they reveal the many events and forces that shape our world. However, finding such interesting and meaningful change events from the vast data is challenging. In this paper, we present new datasets for such change events that include semantically meaningful events like road construction. Instead of manually annotating the very large corpus of satellite images, we introduce a novel unsupervised approach that takes a large spatio-temporal dataset from satellite images and finds interesting change events. To evaluate the meaningfulness on these datasets we create 2 benchmarks namely CaiRoad and CalFire which capture the events of road construction and forest fires. These new benchmarks can be used to evaluate semantic retrieval/classification performance. We explore these benchmarks qualitatively and quantitatively by using several methods and show that these new datasets are indeed challenging for many existing methods.

count=52
* Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.pdf)]
    * Title: Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Maria Vakalopoulou, Konstantinos Karantzalos, Nikos Komodakis, Nikos Paragios
    * Abstract: In order to exploit the currently continuous streams of massive, multi-temporal, high-resolution remote sensing datasets there is an emerging need to address efficiently the image registration and change detection challenges. To this end, in this paper we propose a modular, scalable, metric free single shot change detection/registration method. The approach exploits a decomposed interconnected graphical model formulation where registration similarity constraints are relaxed in the presence of change detection. The deformation space is discretized, while efficient linear programming and duality principles are used to optimize a joint solution space where local consistency is imposed on the deformation and the detection space. Promising results on large scale experiments demonstrate the extreme potentials of our method.

count=52
* Self-Pair: Synthesizing Changes From Single Source for Object Change Detection in Remote Sensing Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Seo_Self-Pair_Synthesizing_Changes_From_Single_Source_for_Object_Change_Detection_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Seo_Self-Pair_Synthesizing_Changes_From_Single_Source_for_Object_Change_Detection_WACV_2023_paper.pdf)]
    * Title: Self-Pair: Synthesizing Changes From Single Source for Object Change Detection in Remote Sensing Imagery
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Minseok Seo, Hakjin Lee, Yongjin Jeon, Junghoon Seo
    * Abstract: For change detection in remote sensing, constructing a training dataset for deep learning models is quite difficult due to the requirements of bi-temporal supervision. To overcome this issue, single-temporal supervision which treats change labels as the difference of two semantic masks has been proposed. This novel method trains a change detector using two spatially unrelated images with corresponding semantic labels. However, training with unpaired dataset shows not enough performance compared with other methods based on bi-temporal supervision. We suspect this phenomenon caused by ignorance of meaningful information in the actual bi-temporal pairs.In this paper, we emphasize that the change originates from the source image and show that manipulating the source image as an after-image is crucial to the performance of change detection. Our method achieves state-of-the-art performance in a large gap than existing methods.

count=51
* QFabric: Multi-Task Change Detection Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.pdf)]
    * Title: QFabric: Multi-Task Change Detection Dataset
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Sagar Verma, Akash Panigrahi, Siddharth Gupta
    * Abstract: Detecting change through multi-image, multi-date remote sensing is essential to developing an understanding of global conditions. Despite recent advancements in remote sensing realized through deep learning, novel methods for accurate multi-image change detection remain unrealized. Recently, several promising methods have been proposed to address this topic, but a paucity of publicly available data limits the methods that can be assessed. In particular, there exists limited work on categorizing the nature and status of change across an observation period. This paper introduces the first labeled dataset available for such a task. We present an open-source change detection dataset, termed QFabric, with 450,000 change polygons annotated across 504 locations in 100 different cities covering a wide range of geographies and urban fabrics. QFabric is a temporal multi-task dataset with 6 change types and 9 change status classes. The geography and environment metadata around each polygon provides context that can be leveraged to build robust deep neural networks. We apply multiple benchmarks on our dataset for change detection, change type and status classification tasks. Project page: https://sagarverma.github.io/qfabric

count=50
* Fully Transformer Network for Change Detection of Remote Sensing Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2022/html/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2022/papers/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.pdf)]
    * Title: Fully Transformer Network for Change Detection of Remote Sensing Images
    * Publisher: ACCV
    * Publication Date: `2022`
    * Authors: Tianyu Yan, Zifu Wan, Pingping Zhang
    * Abstract: Recently, change detection (CD) of remote sensing images have achieved great progress with the advances of deep learning. However, current methods generally deliver incomplete CD regions and irregular CD boundaries due to the limited representation ability of the extracted visual features. To relieve these issues, in this work we propose a novel learning framework named Fully Transformer Network (FTN) for remote sensing image CD, which improves the feature extraction from a global view and combines multi-level visual features in a pyramid manner. More specifically, the proposed framework first utilizes the advantages of Transformers in long-range dependency modeling. It can help to learn more discriminative global-level features and obtain complete CD regions. Then, we introduce a pyramid structure to aggregate multi-level visual features from Transformers for feature enhancement. The pyramid structure grafted with a Progressive Attention Module (PAM) can improve the feature representation ability with additional interdependencies through channel attentions. Finally, to better train the framework, we utilize the deeply-supervised learning with multiple boundaryaware loss functions. Extensive experiments demonstrate that our proposed method achieves a new state-of-the-art performance on four public CD benchmarks. For model reproduction, the source code is released at https://github.com/AI-Zhpp/FTN.

count=47
* Guided Anisotropic Diffusion and Iterative Learning for Weakly Supervised Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/EarthVision/Daudt_Guided_Anisotropic_Diffusion_and_Iterative_Learning_for_Weakly_Supervised_Change_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/EarthVision/Daudt_Guided_Anisotropic_Diffusion_and_Iterative_Learning_for_Weakly_Supervised_Change_CVPRW_2019_paper.pdf)]
    * Title: Guided Anisotropic Diffusion and Iterative Learning for Weakly Supervised Change Detection
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Rodrigo Caye Daudt,  Bertrand Le Saux,  Alexandre Boulch,  Yann Gousseau
    * Abstract: Large scale datasets created from user labels or openly available data have become crucial to provide training data for large scale learning algorithms. While these datasets are easier to acquire, the data are frequently noisy and unreliable, which is motivating research on weakly supervised learning techniques. In this paper we propose an iterative learning method that extracts the useful information from a large scale change detection dataset generated from open vector data to train a fully convolutional network which surpasses the performance obtained by naive supervised learning. We also propose the guided anisotropic diffusion algorithm, which improves semantic segmentation results using the input images as guides to perform edge preserving filtering, and is used in conjunction with the iterative training method to improve results.

count=39
* Semi-Supervised Scene Change Detection by Distillation From Feature-Metric Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Lee_Semi-Supervised_Scene_Change_Detection_by_Distillation_From_Feature-Metric_Alignment_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Lee_Semi-Supervised_Scene_Change_Detection_by_Distillation_From_Feature-Metric_Alignment_WACV_2024_paper.pdf)]
    * Title: Semi-Supervised Scene Change Detection by Distillation From Feature-Metric Alignment
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Seonhoon Lee, Jong-Hwan Kim
    * Abstract: Scene change detection (SCD) is a critical task for various applications, such as visual surveillance, anomaly detection, and mobile robotics. Recently, supervised methods for SCD have been developed for urban and indoor environments where input image pairs are typically unaligned due to differences in camera viewpoints. However, supervised SCD methods require pixel-wise change labels and alignment labels for the target domain, which can be both time-consuming and expensive to collect. To tackle this issue, we design an unsupervised loss with regularization methods based on the feature-metric alignment of input image pairs. The proposed unsupervised loss enables the SCD model to jointly learn the flow and the change maps on the target domain. In addition, we propose a semi-supervised learning method based on a distillation loss for the robustness of the SCD model. The proposed learning method is based on the student-teacher structure and incorporates the unsupervised loss of the unlabeled target data and the supervised loss of the labeled synthetic data. Our method achieves considerable performance improvement on the target domain through the proposed unsupervised and distillation loss, using only 10% of the target training dataset without using any labels of the target data.

count=36
* CDnet 2014: An Expanded Change Detection Benchmark Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Wang_CDnet_2014_An_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Wang_CDnet_2014_An_2014_CVPR_paper.pdf)]
    * Title: CDnet 2014: An Expanded Change Detection Benchmark Dataset
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Yi Wang, Pierre-Marc Jodoin, Fatih Porikli, Janusz Konrad, Yannick Benezeth, Prakash Ishwar
    * Abstract: Change detection is one of the most important low-level tasks in video analytics. In 2012, we introduced the changedetection.net (CDnet) benchmark, a video dataset devoted to the evaluation of change and motion detection approaches. Here, we present the latest release of the CDnet dataset, which includes 22 additional videos (~70,000 pixel-wise annotated frames) spanning 5 new categories that incorporate challenges encountered in many surveillance settings. We describe these categories in detail and provide an overview of the results of more than a dozen methods submitted to the IEEE Change Detection Workshop 2014. We highlight strengths and weaknesses of these methods and identify remaining issues in change detection.

count=32
* A Novel Inspection System For Variable Data Printing Using Deep Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Haik_A_Novel_Inspection_System_For_Variable_Data_Printing_Using_Deep_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Haik_A_Novel_Inspection_System_For_Variable_Data_Printing_Using_Deep_WACV_2020_paper.pdf)]
    * Title: A Novel Inspection System For Variable Data Printing Using Deep Learning
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Oren Haik,  Oded Perry,  Eli Chen,  Peter Klammer
    * Abstract: We present a novel approach for inspecting variable data prints (VDP) with an ultra-low false alarm rate (0.005%) and potential applicability to other real-world problems. The system is based on a comparison between two images: a reference image and an image captured by low-cost scanners. The comparison task is challenging as low-cost imaging systems create artifacts that may erroneously be classified as true (genuine) defects. To address this challenge we introduce two new fusion methods, for change detection applications, which are both fast and efficient. The first is an early fusion method that combines the two input images into a single pseudo-color image. The second, called Change-Detection Single Shot Detector (CD-SSD) leverages the SSD by fusing features in the middle of the network. We demonstrate the effectiveness of the proposed deep learning-based approach with a large dataset from real-world printing scenarios. Finally, we evaluate our models on a different domain of aerial imagery change detection (AICD). Our best method clearly outperforms the state-of-the-art baseline on this dataset.

count=31
* TAMPAR: Visual Tampering Detection for Parcel Logistics in Postal Supply Chains
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Naumann_TAMPAR_Visual_Tampering_Detection_for_Parcel_Logistics_in_Postal_Supply_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Naumann_TAMPAR_Visual_Tampering_Detection_for_Parcel_Logistics_in_Postal_Supply_WACV_2024_paper.pdf)]
    * Title: TAMPAR: Visual Tampering Detection for Parcel Logistics in Postal Supply Chains
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Alexander Naumann, Felix Hertlein, Laura Dörr, Kai Furmans
    * Abstract: Due to the steadily rising amount of valuable goods in supply chains, tampering detection for parcels is becoming increasingly important. In this work, we focus on the use-case last-mile delivery, where only a single RGB image is taken and compared against a reference from an existing database to detect potential appearance changes that indicate tampering. We propose a tampering detection pipeline that utilizes keypoint detection to identify the eight corner points of a parcel. This permits applying a perspective transformation to create normalized fronto-parallel views for each visible parcel side surface. These viewpoint-invariant parcel side surface representations facilitate the identification of signs of tampering on parcels within the supply chain, since they reduce the problem to parcel side surface matching with pair-wise appearance change detection. Experiments with multiple classical and deep learning-based change detection approaches are performed on our newly collected TAMpering detection dataset for PARcels, called TAMPAR. We evaluate keypoint and change detection separately, as well as in a unified system for tampering detection. Our evaluation shows promising results for keypoint (Keypoint AP 75.76) and tampering detection (81% accuracy, F1-Score 0.83) on real images. Furthermore, a sensitivity analysis for tampering types, lens distortion and viewing angles is presented. Code and dataset are available at https://a-nau.github.io/tampar.

count=30
* Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Scalable_Multi-Temporal_Remote_Sensing_Change_Data_Generation_via_Simulating_Stochastic_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Scalable_Multi-Temporal_Remote_Sensing_Change_Data_Generation_via_Simulating_Stochastic_ICCV_2023_paper.pdf)]
    * Title: Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zhuo Zheng, Shiqi Tian, Ailong Ma, Liangpei Zhang, Yanfei Zhong
    * Abstract: Understanding the temporal dynamics of Earth's surface is a mission of multi-temporal remote sensing image analysis, significantly promoted by deep vision models with its fuel---labeled multi-temporal images. However, collecting, preprocessing, and annotating multi-temporal remote sensing images at scale is non-trivial since it is expensive and knowledge-intensive. In this paper, we present a scalable multi-temporal remote sensing change data generator via generative modeling, which is cheap and automatic, alleviating these problems. Our main idea is to simulate a stochastic change process over time. We consider the stochastic change process as a probabilistic semantic state transition, namely generative probabilistic change model (GPCM), which decouples the complex simulation problem into two more trackable sub-problems, i.e., change event simulation and semantic change synthesis. To solve these two problems, we present the change generator (Changen), a GAN-based GPCM, enabling controllable object change data generation, including customizable object property, and change event. The extensive experiments suggest that our Changen has superior generation capability, and the change detectors with Changen pre-training exhibit excellent transferability to real-world change datasets.

count=29
* Robust Change Captioning
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Robust_Change_Captioning_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Robust_Change_Captioning_ICCV_2019_paper.pdf)]
    * Title: Robust Change Captioning
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Dong Huk Park,  Trevor Darrell,  Anna Rohrbach
    * Abstract: Describing what has changed in a scene can be useful to a user, but only if generated text focuses on what is semantically relevant. It is thus important to distinguish distractors (e.g. a viewpoint change) from relevant changes (e.g. an object has moved). We present a novel Dual Dynamic Attention Model (DUDA) to perform robust Change Captioning. Our model learns to distinguish distractors from semantic changes, localize the changes via Dual Attention over "before" and "after" images, and accurately describe them in natural language via Dynamic Speaker, by adaptively focusing on the necessary visual inputs (e.g. "before" or "after" image). To study the problem in depth, we collect a CLEVR-Change dataset, built off the CLEVR engine, with 5 types of scene changes. We benchmark a number of baselines on our dataset, and systematically study different change types and robustness to distractors. We show the superiority of our DUDA model in terms of both change captioning and localization. We also show that our approach is general, obtaining state-of-the-art results on the recent realistic Spot-the-Diff dataset which has no distractors.

count=26
* Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.pdf)]
    * Title: Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Peri Akiva, Matthew Purri, Matthew Leotta
    * Abstract: Self-supervised learning aims to learn image feature representations without the usage of manually annotated labels. It is often used as a precursor step to obtain useful initial network weights which contribute to faster convergence and superior performance of downstream tasks. While self-supervision allows one to reduce the domain gap between supervised and unsupervised learning without the usage of labels, the self-supervised objective still requires a strong inductive bias to downstream tasks for effective transfer learning. In this work, we present our material and texture based self-supervision method named MATTER (MATerial and TExture Representation Learning), which is inspired by classical material and texture methods. Material and texture can effectively describe any surface, including its tactile properties, color, and specularity. By extension, effective representation of material and texture can describe other semantic classes strongly associated with said material and texture. MATTER leverages multi-temporal, spatially aligned remote sensing imagery over unchanged regions to learn invariance to illumination and viewing angle as a mechanism to achieve consistency of material and texture representation. We show that our self-supervision pre-training method allows for up to 24.22% and 6.33% performance increase in unsupervised and fine-tuned setups, and up to 76% faster convergence on change detection, land cover classification, and semantic segmentation tasks.

count=25
* WATCH: Wide-Area Terrestrial Change Hypercube
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Greenwell_WATCH_Wide-Area_Terrestrial_Change_Hypercube_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Greenwell_WATCH_Wide-Area_Terrestrial_Change_Hypercube_WACV_2024_paper.pdf)]
    * Title: WATCH: Wide-Area Terrestrial Change Hypercube
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Connor Greenwell, Jon Crall, Matthew Purri, Kristin Dana, Nathan Jacobs, Armin Hadzic, Scott Workman, Matt Leotta
    * Abstract: Monitoring Earth activity using data collected from multiple satellite imaging platforms in a unified way is a significant challenge, especially with large variability in image resolution, spectral bands, and revisit rates. Further, the availability of sensor data varies across time as new platforms are launched. In this work, we introduce an adaptable framework and network architecture capable of predicting on subsets of the available platforms, bands, or temporal ranges it was trained on. Our system, called WATCH, is highly general and can be applied to a variety of geospatial tasks. In this work, we analyze the performance of WATCH using the recent IARPA SMART public dataset and metrics. We focus primarily on the problem of broad area search for heavy construction sites. Experiments validate the robustness of WATCH during inference to limited sensor availability, as well the the ability to alter inference-time spatial or temporal sampling. WATCH is open source and available for use on this or other remote sensing problems. Code and model weights are available at: https://gitlab.kitware.com/computer-vision/geowatch

count=24
* DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.pdf)]
    * Title: DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Aysim Toker, Lukas Kondmann, Mark Weber, Marvin Eisenberger, Andrés Camero, Jingliang Hu, Ariadna Pregel Hoderlein, Çağlar Şenaras, Timothy Davis, Daniel Cremers, Giovanni Marchisio, Xiao Xiang Zhu, Laura Leal-Taixé
    * Abstract: Earth observation is a fundamental tool for monitoring the evolution of land use in specific areas of interest. Observing and precisely defining change, in this context, requires both time-series data and pixel-wise segmentations. To that end, we propose the DynamicEarthNet dataset that consists of daily, multi-spectral satellite observations of 75 selected areas of interest distributed over the globe with imagery from Planet Labs. These observations are paired with pixel-wise monthly semantic segmentation labels of 7 land use and land cover (LULC) classes. DynamicEarthNet is the first dataset that provides this unique combination of daily measurements and high-quality labels. In our experiments, we compare several established baselines that either utilize the daily observations as additional training data (semi-supervised learning) or multiple observations at once (spatio-temporal learning) as a point of reference for future research. Finally, we propose a new evaluation metric SCS that addresses the specific challenges associated with time-series semantic change segmentation. The data is available at: https://mediatum.ub.tum.de/1650201.

count=24
* SyntheWorld: A Large-Scale Synthetic Dataset for Land Cover Mapping and Building Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Song_SyntheWorld_A_Large-Scale_Synthetic_Dataset_for_Land_Cover_Mapping_and_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Song_SyntheWorld_A_Large-Scale_Synthetic_Dataset_for_Land_Cover_Mapping_and_WACV_2024_paper.pdf)]
    * Title: SyntheWorld: A Large-Scale Synthetic Dataset for Land Cover Mapping and Building Change Detection
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Jian Song, Hongruixuan Chen, Naoto Yokoya
    * Abstract: Synthetic datasets, recognized for their cost effectiveness, play a pivotal role in advancing computer vision tasks and techniques. However, when it comes to remote sensing image processing, the creation of synthetic datasets becomes challenging due to the demand for larger-scale and more diverse 3D models. This complexity is compounded by the difficulties associated with real remote sensing datasets, including limited data acquisition and high annotation costs, which amplifies the need for high-quality synthetic alternatives. To address this, we present SyntheWorld, a synthetic dataset unparalleled in quality, diversity, and scale. It includes 40,000 images with submeter-level pixels and fine-grained land cover annotations of eight categories, and it also provides 40,000 pairs of bitemporal image pairs with building change annotations for building change detection. We conduct experiments on multiple benchmark remote sensing datasets to verify the effectiveness of SyntheWorld and to investigate the conditions under which our synthetic data yield advantages. The dataset is available at https://github.com/JTRNEO/SyntheWorld.

count=23
* City-Scale Change Detection in Cadastral 3D Models Using Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Taneja_City-Scale_Change_Detection_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Taneja_City-Scale_Change_Detection_2013_CVPR_paper.pdf)]
    * Title: City-Scale Change Detection in Cadastral 3D Models Using Images
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Aparna Taneja, Luca Ballan, Marc Pollefeys
    * Abstract: In this paper, we propose a method to detect changes in the geometry of a city using panoramic images captured by a car driving around the city. We designed our approach to account for all the challenges involved in a large scale application of change detection, such as, inaccuracies in the input geometry, errors in the geo-location data of the images, as well as, the limited amount of information due to sparse imagery. We evaluated our approach on an area of 6 square kilometers inside a city, using 3420 images downloaded from Google StreetView. These images besides being publicly available, are also a good example of panoramic images captured with a driving vehicle, and hence demonstrating all the possible challenges resulting from such an acquisition. We also quantitatively compared the performance of our approach with respect to a ground truth, as well as to prior work. This evaluation shows that our approach outperforms the current state of the art.

count=23
* Spectral-360: A Physics-Based Technique for Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Sedky_Spectral-360_A_Physics-Based_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Sedky_Spectral-360_A_Physics-Based_2014_CVPR_paper.pdf)]
    * Title: Spectral-360: A Physics-Based Technique for Change Detection
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Mohamed Sedky, Mansour Moniri, Claude C. Chibelushi
    * Abstract: This paper presents and assesses a novel physics-based change detection technique, Spectral-360, which is based on the dichromatic color reflectance model. This approach, uses image formation models to computationally estimate, from the camera output, a consistent physics-based color descriptor of the spectral reflectance of surfaces visible in the image, and then to measure the similarity between the full-spectrum reflectance of the background and foreground pixels to segment the foreground from a static background. This method represents a new approach to change detection, using explicit hypotheses about the physics that create images. The assumptions which have been made are that diffuse-only-reflection is applicable, and the existence of a dominant illuminant. The objective evaluation performed using the 'changedetection.net 2014' dataset shows that our Spectral-360 method outperforms most state-of-the-art methods.

count=23
* The Change You Want To See
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Sachdeva_The_Change_You_Want_To_See_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Sachdeva_The_Change_You_Want_To_See_WACV_2023_paper.pdf)]
    * Title: The Change You Want To See
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Ragav Sachdeva, Andrew Zisserman
    * Abstract: We live in a dynamic world where things change all the time. Given two images of the same scene, being able to automatically detect the changes in them has practical applications in a variety of domains. In this paper, we tackle the change detection problem with the goal of detecting "object-level" changes in an image pair despite differences in their viewpoint and illumination. To this end, we make the following four contributions: (i) we propose a scalable methodology for obtaining a large-scale change detection training dataset by leveraging existing object segmentation benchmarks; (ii) we introduce a co-attention based novel architecture that is able to implicitly determine correspondences between an image pair and find changes in the form of bounding box predictions; (iii) we contribute four evaluation datasets that cover a variety of domains and transformations, including synthetic image changes, real surveillance images of a 3D scene, and synthetic 3D scenes with camera motion; (iv) we evaluate our model on these four datasets and demonstrate zero-shot and beyond training transformation generalization. The code, datasets and pre-trained model can be found at our project page: https://www.robots.ox.ac.uk/ vgg/research/cyws/

count=22
* A Category Agnostic Model for Visual Rearrangment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_A_Category_Agnostic_Model_for_Visual_Rearrangment_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_A_Category_Agnostic_Model_for_Visual_Rearrangment_CVPR_2024_paper.pdf)]
    * Title: A Category Agnostic Model for Visual Rearrangment
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yuyi Liu, Xinhang Song, Weijie Li, Xiaohan Wang, Shuqiang Jiang
    * Abstract: This paper presents a novel category agnostic model for visual rearrangement task which can help an embodied agent to physically recover the shuffled scene configuration without any category concepts to the goal configuration. Previous methods usually follow a similar architecture completing the rearrangement task by aligning the scene changes of the goal and shuffled configuration according to the semantic scene graphs. However constructing scene graphs requires the inference of category labels which not only causes the accuracy drop of the entire task but also limits the application in real world scenario. In this paper we delve deep into the essence of visual rearrangement task and focus on the two most essential issues scene change detection and scene change matching. We utilize the movement and the protrusion of point cloud to accurately identify the scene changes and match these changes depending on the similarity of category agnostic appearance feature. Moreover to assist the agent to explore the environment more efficiently and comprehensively we propose a closer-aligned-retrace exploration policy aiming to observe more details of the scene at a closer distance. We conduct extensive experiments on AI2THOR Rearrangement Challenge based on RoomR dataset and a new multi-room multi-instance dataset MrMiR collected by us. The experimental results demonstrate the effectiveness of our proposed method.

count=21
* Building Bridges across Spatial and Temporal Resolutions: Reference-Based Super-Resolution via Change Priors and Conditional Diffusion Model
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Dong_Building_Bridges_across_Spatial_and_Temporal_Resolutions_Reference-Based_Super-Resolution_via_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Dong_Building_Bridges_across_Spatial_and_Temporal_Resolutions_Reference-Based_Super-Resolution_via_CVPR_2024_paper.pdf)]
    * Title: Building Bridges across Spatial and Temporal Resolutions: Reference-Based Super-Resolution via Change Priors and Conditional Diffusion Model
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Runmin Dong, Shuai Yuan, Bin Luo, Mengxuan Chen, Jinxiao Zhang, Lixian Zhang, Weijia Li, Juepeng Zheng, Haohuan Fu
    * Abstract: Reference-based super-resolution (RefSR) has the potential to build bridges across spatial and temporal resolutions of remote sensing images. However existing RefSR methods are limited by the faithfulness of content reconstruction and the effectiveness of texture transfer in large scaling factors. Conditional diffusion models have opened up new opportunities for generating realistic high-resolution images but effectively utilizing reference images within these models remains an area for further exploration. Furthermore content fidelity is difficult to guarantee in areas without relevant reference information. To solve these issues we propose a change-aware diffusion model named Ref-Diff for RefSR using the land cover change priors to guide the denoising process explicitly. Specifically we inject the priors into the denoising model to improve the utilization of reference information in unchanged areas and regulate the reconstruction of semantically relevant content in changed areas. With this powerful guidance we decouple the semantics-guided denoising and reference texture-guided denoising processes to improve the model performance. Extensive experiments demonstrate the superior effectiveness and robustness of the proposed method compared with state-of-the-art RefSR methods in both quantitative and qualitative evaluations. The code and data are available at https://github.com/dongrunmin/RefDiff.

count=20
* Can We Speed up 3D Scanning? A Cognitive and Geometric Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w40/html/Vaiapury_Can_We_Speed_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w40/Vaiapury_Can_We_Speed_ICCV_2017_paper.pdf)]
    * Title: Can We Speed up 3D Scanning? A Cognitive and Geometric Analysis
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Karthikeyan Vaiapury, Balamuralidhar Purushothaman, Arpan Pal, Swapna Agarwal, Brojeshwar Bhowmick
    * Abstract: The paper propose a cognitive inspired change detection method for the detection and localization of shape variations on point clouds. A well defined pipeline is introduced by proposing a coarse to fine approach: i) shape segmentation, ii) fine segment registration using attention blocks. Shape segmentation is obtained using covariance based method and fine segment registration is carried out using gravitational registration algorithm. In particular the introduction of this partition-based approach using visual attention mechanism improves the speed of deformation detection and localization. Some results are shown on synthetic data of house and aircraft models. Experimental results shows that this simple yet effective approach designed with an eye to scalability can detect and localize the deformation in a faster manner. A real world car usecase is also presented with some preliminary promising results useful for auditing and insurance claim tasks.

count=19
* Implicit Neural Representation for Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.pdf)]
    * Title: Implicit Neural Representation for Change Detection
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Peter Naylor, Diego Di Carlo, Arianna Traviglia, Makoto Yamada, Marco Fiorucci
    * Abstract: Identifying changes in a pair of 3D aerial LiDAR point clouds, obtained during two distinct time periods over the same geographic region presents a significant challenge due to the disparities in spatial coverage and the presence of noise in the acquisition system. The most commonly used approaches to detecting changes in point clouds are based on supervised methods which necessitate extensive labelled data often unavailable in real-world applications. To address these issues, we propose an unsupervised approach that comprises two components: Implicit Neural Representation (INR) for continuous shape reconstruction and a Gaussian Mixture Model for categorising changes. INR offers a grid-agnostic representation for encoding bi-temporal point clouds, with unmatched spatial support that can be regularised to enhance high-frequency details and reduce noise. The reconstructions at each timestamp are compared at arbitrary spatial scales, leading to a significant increase in detection capabilities. We apply our method to a benchmark dataset comprising simulated LiDAR point clouds for urban sprawling. This dataset encompasses diverse challenging scenarios, varying in resolutions, input modalities and noise levels. This enables a comprehensive multi-scenario evaluation, comparing our method with the current state-of-the-art approach. We outperform the previous methods by a margin of 10% in the intersection over union metric. In addition, we put our techniques to practical use by applying them in a real-world scenario to identify instances of illicit excavation of archaeological sites and validate our results by comparing them with findings from field experts.

count=18
* Rare Event Detection Using Disentangled Representation Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Hamaguchi_Rare_Event_Detection_Using_Disentangled_Representation_Learning_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hamaguchi_Rare_Event_Detection_Using_Disentangled_Representation_Learning_CVPR_2019_paper.pdf)]
    * Title: Rare Event Detection Using Disentangled Representation Learning
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Ryuhei Hamaguchi,  Ken Sakurada,  Ryosuke Nakamura
    * Abstract: This paper presents a novel method for rare event detection from an image pair with class-imbalanced datasets. A straightforward approach for event detection tasks is to train a detection network from a large-scale dataset in an end-to-end manner. However, in many applications such as building change detection on satellite images, few positive samples are available for the training. Moreover, an image pair of scenes contains many trivial events, such as in illumination changes or background motions. These many trivial events and the class imbalance problem lead to false alarms for rare event detection. In order to overcome these difficulties, we propose a novel method to learn disentangled representations from only low-cost negative samples. The proposed method disentangles the different aspects in a pair of observations: variant and invariant factors that represent trivial events and image contents, respectively. The effectiveness of the proposed approach is verified by the quantitative evaluations on four change detection datasets, and the qualitative analysis shows that the proposed method can acquire the representations that disentangle rare events from trivial ones.

count=18
* Did It Change? Learning to Detect Point-Of-Interest Changes for Proactive Map Updates
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Revaud_Did_It_Change_Learning_to_Detect_Point-Of-Interest_Changes_for_Proactive_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Revaud_Did_It_Change_Learning_to_Detect_Point-Of-Interest_Changes_for_Proactive_CVPR_2019_paper.pdf)]
    * Title: Did It Change? Learning to Detect Point-Of-Interest Changes for Proactive Map Updates
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jerome Revaud,  Minhyeok Heo,  Rafael S. Rezende,  Chanmi You,  Seong-Gyun Jeong
    * Abstract: Maps are an increasingly important tool in our daily lives, yet their rich semantic content still largely depends on manual input. Motivated by the broad availability of geo-tagged street-view images, we propose a new task aiming to make the map update process more proactive. We focus on automatically detecting changes of Points of Interest (POIs), specifically stores or shops of any kind, based on visual input. Faced with the lack of an appropriate benchmark, we build and release a large dataset, captured in two large shopping centers, that comprises 33K geo-localized images and 578 POIs. We then design a generic approach that compares two image sets captured in the same venue at different times and outputs POI changes as a ranked list of map locations. In contrast to logo or franchise recognition approaches, our system does not depend on an external franchise database. It is instead inspired by recent deep metric learning approaches that learn a similarity function fit to the task at hand. We compare various loss functions to learn a metric aligned with the POI change detection goal, and report promising results.

count=18
* Change-Aware Sampling and Contrastive Learning for Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf)]
    * Title: Change-Aware Sampling and Contrastive Learning for Satellite Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Utkarsh Mall, Bharath Hariharan, Kavita Bala
    * Abstract: Automatic remote sensing tools can help inform many large-scale challenges such as disaster management, climate change, etc. While a vast amount of spatio-temporal satellite image data is readily available, most of it remains unlabelled. Without labels, this data is not very useful for supervised learning algorithms. Self-supervised learning instead provides a way to learn effective representations for various downstream tasks without labels. In this work, we leverage characteristics unique to satellite images to learn better self-supervised features. Specifically, we use the temporal signal to contrast images with long-term and short-term differences, and we leverage the fact that satellite images do not change frequently. Using these characteristics, we formulate a new loss contrastive loss called Change-Aware Contrastive (CACo) Loss. Further, we also present a novel method of sampling different geographical regions. We show that leveraging these properties leads to better performance on diverse downstream tasks. For example, we see a 6.5% relative improvement for semantic segmentation and an 8.5% relative improvement for change detection over the best-performing baseline with our method.

count=17
* The STVchrono Dataset: Towards Continuous Change Recognition in Time
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_The_STVchrono_Dataset_Towards_Continuous_Change_Recognition_in_Time_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_The_STVchrono_Dataset_Towards_Continuous_Change_Recognition_in_Time_CVPR_2024_paper.pdf)]
    * Title: The STVchrono Dataset: Towards Continuous Change Recognition in Time
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yanjun Sun, Yue Qiu, Mariia Khan, Fumiya Matsuzawa, Kenji Iwata
    * Abstract: Recognizing continuous changes offers valuable insights into past historical events supports current trend analysis and facilitates future planning. This knowledge is crucial for a variety of fields such as meteorology and agriculture environmental science urban planning and construction tourism and cultural preservation. Currently available datasets in the field of scene change understanding primarily concentrate on two main tasks: the detection of changed regions within a scene and the linguistic description of the change content. Existing datasets focus on recognizing discrete changes such as adding or deleting an object from two images and largely rely on artificially generated images. Consequently the existing change understanding methods primarily focus on identifying distinct object differences overlooking the importance of continuous gradual changes occurring over extended time intervals. To address the above issues we propose a novel benchmark dataset STVchrono targeting the localization and description of long-term continuous changes in real-world scenes. The dataset consists of 71900 photographs from Google Street View API taken over an 18-year span across 50 cities all over the world. Our STVchrono dataset is designed to support real-world continuous change recognition and description in both image pairs and extended image sequences while also enabling the segmentation of changed regions. We conduct experiments to evaluate state-of-the-art methods on continuous change description and segmentation as well as multimodal Large Language Models for describing changes. Our findings reveal that even the most advanced methods lag human performance emphasizing the need to adapt them to continuously changing real-world scenarios. We hope that our benchmark dataset will further facilitate the research of temporal change recognition in a dynamic world. The STVchrono dataset is available at STVchrono Dataset.

count=16
* Good at Captioning Bad at Counting: Benchmarking GPT-4V on Earth Observation Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Zhang_Good_at_Captioning_Bad_at_Counting_Benchmarking_GPT-4V_on_Earth_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Zhang_Good_at_Captioning_Bad_at_Counting_Benchmarking_GPT-4V_on_Earth_CVPRW_2024_paper.pdf)]
    * Title: Good at Captioning Bad at Counting: Benchmarking GPT-4V on Earth Observation Data
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chenhui Zhang,Sherrie Wang
    * Abstract: Large Vision-Language Models (VLMs) have demonstrated impressive performance on complex tasks involving visual input with natural language instructions. However it remains unclear to what extent capabilities on natural images transfer to Earth observation (EO) data which are predominantly satellite and aerial images less common in VLM training data. In this work we propose a comprehensive benchmark to gauge the progress of VLMs toward being useful tools for EO data by assessing their abilities on scene understanding localization and counting and change detection. Motivated by real-world applications our benchmark includes scenarios like urban monitoring disaster relief land use and conservation. We discover that although state-of-the-art VLMs like GPT-4V possess world knowledge that leads to strong performance on location understanding and image captioning their poor spatial reasoning limits usefulness on object localization and counting. Our benchmark leaderboard and evaluation suite are available at https://vleo.danielz.ch/. A full version of this paper is available at https://arxiv.org/abs/2401.17600.

count=15
* Ensemble Video Object Cut in Highly Dynamic Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Ren_Ensemble_Video_Object_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ren_Ensemble_Video_Object_2013_CVPR_paper.pdf)]
    * Title: Ensemble Video Object Cut in Highly Dynamic Scenes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Xiaobo Ren, Tony X. Han, Zhihai He
    * Abstract: We consider video object cut as an ensemble of framelevel background-foreground object classifiers which fuses information across frames and refine their segmentation results in a collaborative and iterative manner. Our approach addresses the challenging issues of modeling of background with dynamic textures and segmentation of foreground objects from cluttered scenes. We construct patch-level bagof-words background models to effectively capture the background motion and texture dynamics. We propose a foreground salience graph (FSG) to characterize the similarity of an image patch to the bag-of-words background models in the temporal domain and to neighboring image patches in the spatial domain. We incorporate this similarity information into a graph-cut energy minimization framework for foreground object segmentation. The background-foreground classification results at neighboring frames are fused together to construct a foreground probability map to update the graph weights. The resulting object shapes at neighboring frames are also used as constraints to guide the energy minimization process during graph cut. Our extensive experimental results and performance comparisons over a diverse set of challenging videos with dynamic scenes, including the new Change Detection Challenge Dataset, demonstrate that the proposed ensemble video object cut method outperforms various state-ofthe-art algorithms.

count=15
* Detecting Changes in 3D Structure of a Scene from Multi-view Images Captured by a Vehicle-Mounted Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Sakurada_Detecting_Changes_in_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Sakurada_Detecting_Changes_in_2013_CVPR_paper.pdf)]
    * Title: Detecting Changes in 3D Structure of a Scene from Multi-view Images Captured by a Vehicle-Mounted Camera
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Ken Sakurada, Takayuki Okatani, Koichiro Deguchi
    * Abstract: This paper proposes a method for detecting temporal changes of the three-dimensional structure of an outdoor scene from its multi-view images captured at two separate times. For the images, we consider those captured by a camera mounted on a vehicle running in a city street. The method estimates scene structures probabilistically, not deterministically, and based on their estimates, it evaluates the probability of structural changes in the scene, where the inputs are the similarity of the local image patches among the multi-view images. The aim of the probabilistic treatment is to maximize the accuracy of change detection, behind which there is our conjecture that although it is difficult to estimate the scene structures deterministically, it should be easier to detect their changes. The proposed method is compared with the methods that use multi-view stereo (MVS) to reconstruct the scene structures of the two time points and then differentiate them to detect changes. The experimental results show that the proposed method outperforms such MVS-based methods.

count=15
* Extending Global-local View Alignment for Self-supervised Learning with Remote Sensing Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/WiCV/html/Wanyan_Extending_Global-local_View_Alignment_for_Self-supervised_Learning_with_Remote_Sensing_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/WiCV/papers/Wanyan_Extending_Global-local_View_Alignment_for_Self-supervised_Learning_with_Remote_Sensing_CVPRW_2024_paper.pdf)]
    * Title: Extending Global-local View Alignment for Self-supervised Learning with Remote Sensing Imagery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xinye Wanyan, Sachith Seneviratne, Shuchang Shen, Michael Kirley
    * Abstract: Since large number of high-quality remote sensing images are readily accessible exploiting the corpus of images with less manual annotation draws increasing attention. Self-supervised models acquire general feature representations by formulating a pretext task that generates pseudo-labels for massive unlabeled data to provide supervision for training. While prior studies have explored multiple self-supervised learning techniques in remote sensing domain pretext tasks based on local-global view alignment remain underexplored despite achieving state-of-the-art results on natural imagery. Inspired by DINO [6] which employs an effective representation learning structure with knowledge distillation based on global-local view alignment we formulate two pretext tasks for self-supervised learning on remote sensing imagery (SSLRS). Using these tasks we explore the effectiveness of positive temporal contrast as well as multi-sized views on SSLRS. We extend DINO and propose DINO-MC which uses local views of various sized crops instead of a single fixed size in order to alleviate the limited variation in object size observed in remote sensing imagery. Our experiments demonstrate that even when pre-trained on only 10% of the dataset DINO-MC performs on par or better than existing state-of-the-art SSLRS methods on multiple remote sensing tasks while using less computational resources. All codes models and results are released at https://github.com/WennyXY/DINO-MC.

count=15
* Superpixel-Based 3D Building Model Refinement and Change Detection, Using VHR Stereo Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/WiCV/Gharibbafghi_Superpixel-Based_3D_Building_Model_Refinement_and_Change_Detection_Using_VHR_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WiCV/Gharibbafghi_Superpixel-Based_3D_Building_Model_Refinement_and_Change_Detection_Using_VHR_CVPRW_2019_paper.pdf)]
    * Title: Superpixel-Based 3D Building Model Refinement and Change Detection, Using VHR Stereo Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Zeinab Gharibbafghi,  Jiaojiao Tian,  Peter Reinartz
    * Abstract: Buildings are one of the main objects in urban remote sensing and photogrammetric computer vision applications using satellite data. In this paper a superpixel-based approach is presented to refine 3D building models from stereo satellite imagery. First, for each epoch in time, a multispectral very high resolution (VHR) satellite image is segmented using an efficient superpixel, called edge-based simple linear iterative clustering (ESLIC). The ESLIC algorithm segments the image utilizing the spectral and spatial information, as well as the statistical measures from the gray-level co-occurrence matrix (GLCM), simultaneously. Then the resulting superpixels are imposed on the corresponding 3D model of the scenes taken from each epoch. Since ESLIC has high capability of preserving edges in the image, normalized digital surface models (nDSMs) can be modified by averaging height values inside superpixels. These new normalized models for epoch 1 and epoch 2, are then used to detect the 3D change of each building in the scene.

count=15
* Describing and Localizing Multiple Changes With Transformers
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Qiu_Describing_and_Localizing_Multiple_Changes_With_Transformers_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiu_Describing_and_Localizing_Multiple_Changes_With_Transformers_ICCV_2021_paper.pdf)]
    * Title: Describing and Localizing Multiple Changes With Transformers
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yue Qiu, Shintaro Yamamoto, Kodai Nakashima, Ryota Suzuki, Kenji Iwata, Hirokatsu Kataoka, Yutaka Satoh
    * Abstract: Existing change captioning studies have mainly focused on a single change. However, detecting and describing multiple changed parts in image pairs is essential for enhancing adaptability to complex scenarios. We solve the above issues from three aspects: (i) We propose a simulation-based multi-change captioning dataset; (ii) We benchmark existing state-of-the-art methods of single change captioning on multi-change captioning; (iii) We further propose Multi-Change Captioning transformers (MCCFormers) that identify change regions by densely correlating different regions in image pairs and dynamically determines the related change regions with words in sentences. The proposed method obtained the highest scores on four conventional change captioning evaluation metrics for multi-change captioning. Additionally, our proposed method can separate attention maps for each change and performs well with respect to change localization. Moreover, the proposed framework outperformed the previous state-of-the-art methods on an existing change captioning benchmark, CLEVR-Change, by a large margin (+6.1 on BLEU-4 and +9.7 on CIDEr scores), indicating its general ability in change captioning tasks. The code and dataset are available at the project page.

count=15
* Towards Geospatial Foundation Models via Continual Pretraining
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.pdf)]
    * Title: Towards Geospatial Foundation Models via Continual Pretraining
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Matías Mendieta, Boran Han, Xingjian Shi, Yi Zhu, Chen Chen
    * Abstract: Geospatial technologies are becoming increasingly essential in our world for a wide range of applications, including agriculture, urban planning, and disaster response. To help improve the applicability and performance of deep learning models on these geospatial tasks, various works have begun investigating foundation models for this domain. Researchers have explored two prominent approaches for introducing such models in geospatial applications, but both have drawbacks in terms of limited performance benefit or prohibitive training cost. Therefore, in this work, we propose a novel paradigm for building highly effective geospatial foundation models with minimal resource cost and carbon impact. We first construct a compact yet diverse dataset from multiple sources to promote feature diversity, which we term GeoPile. Then, we investigate the potential of continual pretraining from large-scale ImageNet-22k models and propose a multi-objective continual pretraining paradigm, which leverages the strong representations of ImageNet while simultaneously providing the freedom to learn valuable in-domain features. Our approach outperforms previous state-of-the-art geospatial pretraining methods in an extensive evaluation on seven downstream datasets covering various tasks such as change detection, classification, multi-label classification, semantic segmentation, and super-resolution. Code is available at https://github.com/mmendiet/GFM.

count=15
* The Change You Want to See (Now in 3D)
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/html/Sachdeva_The_Change_You_Want_to_See_Now_in_3D_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/papers/Sachdeva_The_Change_You_Want_to_See_Now_in_3D_ICCVW_2023_paper.pdf)]
    * Title: The Change You Want to See (Now in 3D)
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ragav Sachdeva, Andrew Zisserman
    * Abstract: The goal of this paper is to detect what has changed, if anything, between two "in the wild" images of the same 3D scene acquired from different camera positions and at different temporal instances. The open-set nature of this problem, occlusions/dis-occlusions due to the shift in viewpoint, and the lack of suitable training datasets, presents substantial challenges in devising a solution. To address this problem, we contribute a change detection model that is trained entirely on synthetic data and is class-agnostic, yet it is performant out-of-the-box on real world images without requiring fine-tuning. Our solution entails a "register and difference" approach that leverages self-supervised frozen embeddings and feature differences, which allows the model to generalise to a wide variety of scenes and domains. The model is able to operate directly on two RGB images, without requiring access to ground truth camera intrinsics, extrinsics, depth maps, point clouds, or additional before-after images. Finally, we collect and release a new evaluation dataset consisting of real-world image pairs with human-annotated differences and demonstrate the efficacy of our method. The code, datasets and pre-trained model can be found at: https://github.com/ragavsachdeva/CYWS-3D

count=14
* Underwater Moving Object Detection Using an End-to-End Encoder-Decoder Architecture and GraphSage With Aggregator and Refactoring
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/WiCV/html/Kapoor_Underwater_Moving_Object_Detection_Using_an_End-to-End_Encoder-Decoder_Architecture_and_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/WiCV/papers/Kapoor_Underwater_Moving_Object_Detection_Using_an_End-to-End_Encoder-Decoder_Architecture_and_CVPRW_2023_paper.pdf)]
    * Title: Underwater Moving Object Detection Using an End-to-End Encoder-Decoder Architecture and GraphSage With Aggregator and Refactoring
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Meghna Kapoor, Suvam Patra, Badri Narayan Subudhi, Vinit Jakhetiya, Ankur Bansal
    * Abstract: Underwater environments are greatly affected by several factors, including low visibility, high turbidity, back-scattering, dynamic background, etc., and hence pose challenges in object detection. Several algorithms consider convolutional neural networks to extract deep features and then object detection using the same. However, the dependency on the kernel's size and the network's depth results in fading relationships of latent space features and also are unable to characterize the spatial-contextual bonding of the pixels. Hence, they are unable to procure satisfactory results in complex underwater scenarios. To re-establish this relationship, we propose a unique architecture for underwater object detection where U-Net architecture is considered with the ResNet-50 backbone. Further, the latent space features from the encoder are fed to the decoder through a GraphSage model. GraphSage-based model is explored to reweight the node relationship in non-euclidean space using different aggregator functions and hence characterize the spatio-contextual bonding among the pixels. Further, we explored the dependency on different aggregator functions: mean, max, and LSTM, to evaluate the model's performance. We evaluated the proposed model on two underwater benchmark databases: F4Knowledge and underwater change detection. The performance of the proposed model is evaluated against eleven state-of-the-art techniques in terms of both visual and quantitative evaluation measures.

count=14
* SiamSTA: Spatio-Temporal Attention Based Siamese Tracker for Tracking UAVs
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/html/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/papers/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.pdf)]
    * Title: SiamSTA: Spatio-Temporal Attention Based Siamese Tracker for Tracking UAVs
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Bo Huang, Junjie Chen, Tingfa Xu, Ying Wang, Shenwang Jiang, Yuncheng Wang, Lei Wang, Jianan Li
    * Abstract: With the growing threat of unmanned aerial vehicle (UAV) intrusion, anti-UAV techniques are becoming increasingly demanding. Object tracking, especially in thermal infrared (TIR) videos, though provides a promising solution, struggles with challenges like small scale and fast movement that commonly occur in anti-UAV scenarios. To mitigate this, we propose a simple yet effective spatio-temporal attention based Siamese network, dubbed SiamSTA, to track UAV robustly by performing reliable local tracking and wide-range re-detection alternatively. Concretely, tracking is carried out by posing spatial and temporal constraints on generating candidate proposals within local neighborhoods, hence eliminating background distractors to better perceive small targets. Complementarily, in case of target lost from local regions due to fast movement, a three-stage re-detection mechanism is introduced to re-detect targets from a global view by exploiting valuable motion cues through a correlation filter based on change detection. Finally, a state-aware switching policy is adopted to adaptively integrate local tracking and global re-detection and take their complementary strengths for robust tracking. Extensive experiments on the 1st and 2nd anti-UAV datasets well demonstrate the superiority of SiamSTA over other competing counterparts. Notably, SiamSTA is the foundation of the 1st-place winning entry in the 2nd Anti-UAV Challenge.

count=13
* BoT-FaceSORT: Bag-of-Tricks for Robust Multi-Face Tracking in Unconstrained Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Kim_BoT-FaceSORT_Bag-of-Tricks_for_Robust_Multi-Face_Tracking_in_Unconstrained_Videos_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Kim_BoT-FaceSORT_Bag-of-Tricks_for_Robust_Multi-Face_Tracking_in_Unconstrained_Videos_ACCV_2024_paper.pdf)]
    * Title: BoT-FaceSORT: Bag-of-Tricks for Robust Multi-Face Tracking in Unconstrained Videos
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Jonghyeon Kim, Chan-Yang Ju, Gun-Woo Kim, Dong-Ho Lee
    * Abstract: Multi-face tracking (MFT) is a subtask of multi-object tracking (MOT) that focuses on detecting and tracking multiple faces across video frames. Modern MOT trackers adopt the Kalman filter (KF), a linear model that estimates current motions based on previous observations. However, these KF-based trackers struggle to predict motions in unconstrained videos with frequent shot changes, occlusions, and appearance variations. To address these limitations, we propose BoT-FaceSORT, a novel MFT framework that integrates shot change detection, shared feature memory, and an adaptive cascade matching strategy for robust tracking. It detects shot changes by comparing the color histograms of adjacent frames and resets KF states to handle discontinuities. Additionally, we introduce MovieShot, a new benchmark of challenging movie clips to evaluate MFT performance in unconstrained scenarios. We also demonstrate the superior performance of our method compared to existing methods on three benchmarks, while an ablation study validates the effectiveness of each component in handling unconstrained videos.

count=13
* The Multi-Temporal Urban Development SpaceNet Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.pdf)]
    * Title: The Multi-Temporal Urban Development SpaceNet Dataset
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Adam Van Etten, Daniel Hogan, Jesus Martinez Manso, Jacob Shermeyer, Nicholas Weir, Ryan Lewis
    * Abstract: Satellite imagery analytics have numerous human development and disaster response applications, particularly when time series methods are involved. For example, quantifying population statistics is fundamental to 67 of the 231 United Nations Sustainable Development Goals Indicators, but the World Bank estimates that over 100 countries currently lack effective Civil Registration systems. To help address this deficit and develop novel computer vision methods for time series data, we present the Multi-Temporal Urban Development SpaceNet (MUDS, also known as SpaceNet 7) dataset. This open source dataset consists of medium resolution (4.0m) satellite imagery mosaics, which includes 24 images (one per month) covering >100 unique geographies, and comprises >40,000 km2 of imagery and exhaustive polygon labels of building footprints therein, totaling over 11M individual annotations. Each building is assigned a unique identifier (i.e. address), which permits tracking of individual objects over time. Label fidelity exceeds image resolution; this "omniscient labeling" is a unique feature of the dataset, and enables surprisingly precise algorithmic models to be crafted. We demonstrate methods to track building footprint construction (or demolition) over time, thereby directly assessing urbanization. Performance is measured with the newly developed SpaceNet Change and Object Tracking (SCOT) metric, which quantifies both object tracking as well as change detection. We demonstrate that despite the moderate resolution of the data, we are able to track individual building identifiers over time.

count=13
* Seasonal Contrast: Unsupervised Pre-Training From Uncurated Remote Sensing Data
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.pdf)]
    * Title: Seasonal Contrast: Unsupervised Pre-Training From Uncurated Remote Sensing Data
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Oscar Mañas, Alexandre Lacoste, Xavier Giró-i-Nieto, David Vazquez, Pau Rodríguez
    * Abstract: Remote sensing and automatic earth monitoring are key to solve global-scale challenges such as disaster prevention, land use monitoring, or tackling climate change. Although there exist vast amounts of remote sensing data, most of it remains unlabeled and thus inaccessible for supervised learning algorithms. Transfer learning approaches can reduce the data requirements of deep learning algorithms. However, most of these methods are pre-trained on ImageNet and their generalization to remote sensing imagery is not guaranteed due to the domain gap. In this work, we propose Seasonal Contrast (SeCo), an effective pipeline to leverage unlabeled data for in-domain pre-training of remote sensing representations. The SeCo pipeline is composed of two parts. First, a principled procedure to gather large-scale, unlabeled and uncurated remote sensing datasets containing images from multiple Earth locations at different timestamps. Second, a self-supervised algorithm that takes advantage of time and position invariance to learn transferable representations for remote sensing applications. We empirically show that models trained with SeCo achieve better performance than their ImageNet pre-trained counterparts and state-of-the-art self-supervised learning methods on multiple downstream tasks. The datasets and models in SeCo will be made public to facilitate transfer learning and enable rapid progress in remote sensing applications.

count=13
* Bandit Quickest Changepoint Detection
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/f3a4ff4839c56a5f460c88cce3666a2b-Paper.pdf)]
    * Title: Bandit Quickest Changepoint Detection
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Aditya Gopalan, Braghadeesh Lakshminarayanan, Venkatesh Saligrama
    * Abstract: Many industrial and security applications employ a suite of sensors for detecting abrupt changes in temporal behavior patterns. These abrupt changes typically manifest locally, rendering only a small subset of sensors informative. Continuous monitoring of every sensor can be expensive due to resource constraints, and serves as a motivation for the bandit quickest changepoint detection problem, where sensing actions (or sensors) are sequentially chosen, and only measurements corresponding to chosen actions are observed. We derive an information-theoretic lower bound on the detection delay for a general class of finitely parameterized probability distributions. We then propose a computationally efficient online sensing scheme, which seamlessly balances the need for exploration of different sensing options with exploitation of querying informative actions. We derive expected delay bounds for the proposed scheme and show that these bounds match our information-theoretic lower bounds at low false alarm rates, establishing optimality of the proposed method. We then perform a number of experiments on synthetic and real datasets demonstrating the effectiveness of our proposed method.

count=13
* A Greek Parliament Proceedings Dataset for Computational Linguistics and Political Analysis
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/b96ce67b2f2d45e4ab315e13a6b5b9c5-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: A Greek Parliament Proceedings Dataset for Computational Linguistics and Political Analysis
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Konstantina Dritsa, Aikaterini Thoma, Ioannis Pavlopoulos, Panos Louridas
    * Abstract: Large, diachronic datasets of political discourse are hard to come across, especially for resource-lean languages such as Greek. In this paper, we introduce a curated dataset of the Greek Parliament Proceedings that extends chronologically from 1989 up to 2020. It consists of more than 1 million speeches with extensive meta-data, extracted from 5,355 parliamentary sitting record files. We explain how it was constructed and the challenges that had to be overcome. The dataset can be used for both computational linguistics and political analysis---ideally, combining the two. We present such an application, showing (i) how the dataset can be used to study the change of word usage through time, (ii) between significant historical events and political parties, (iii) by evaluating and employing algorithms for detecting semantic shifts.

count=11
* Geospatial Correspondences for Multimodal Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.pdf)]
    * Title: Geospatial Correspondences for Multimodal Registration
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Diego Marcos, Raffay Hamid, Devis Tuia
    * Abstract: The growing availability of very high resolution (<1 m/pixel) satellite and aerial images has opened up unprecedented opportunities to monitor and analyze the evolution of land-cover and land-use across the world. To do so, images of the same geographical areas acquired at different times and, potentially, with different sensors must be efficiently parsed to update maps and detect land-cover changes. However, a naive transfer of ground truth labels from one location in the source image to the corresponding location in the target image is not generally feasible, as these images are often only loosely registered (with up to +- 50m of non-uniform errors). Furthermore, land-cover changes in an area over time must be taken into account for an accurate ground truth transfer. To tackle these challenges, we propose a mid-level sensor-invariant representation that encodes image regions in terms of the spatial distribution of their spectral neighbors. We incorporate this representation in a Markov Random Field to simultaneously account for nonlinear mis-registrations and enforce locality priors to find matches between multi-sensor images. We show how our approach can be used to assist in several multimodal land-cover update and change detection problems.

count=10
* Image Change Captioning by Learning From an Auxiliary Task
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Hosseinzadeh_Image_Change_Captioning_by_Learning_From_an_Auxiliary_Task_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Hosseinzadeh_Image_Change_Captioning_by_Learning_From_an_Auxiliary_Task_CVPR_2021_paper.pdf)]
    * Title: Image Change Captioning by Learning From an Auxiliary Task
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Mehrdad Hosseinzadeh, Yang Wang
    * Abstract: We tackle the challenging task of image change captioning. The goal is to describe the subtle difference between two very similar images by generating a sentence caption. While the recent methods mainly focus on proposing new model architectures for this problem, we instead focus on an alternative training scheme. Inspired by the success of multi-task learning, we formulate a training scheme that uses an auxiliary task to improve the training of the change captioning network. We argue that the task of composed query image retrieval is a natural choice as the auxiliary task. Given two almost similar images as the input, the primary network generates a caption describing the fine change between those two images. Next, the auxiliary network is provided with the generated caption and one of those two images. It then tries to pick the second image among a set of candidates. This forces the primary network to generate detailed and precise captions via having an extra supervision loss by the auxiliary network. Furthermore, we propose a new scheme for selecting a negative set of candidates for the retrieval task that can effectively improve the performance. We show that the proposed training strategy performs well on the task of change captioning on benchmark datasets.

count=10
* Static and Moving Object Detection Using Flux Tensor with Split Gaussian Models
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Wang_Static_and_Moving_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Wang_Static_and_Moving_2014_CVPR_paper.pdf)]
    * Title: Static and Moving Object Detection Using Flux Tensor with Split Gaussian Models
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Rui Wang, Filiz Bunyak, Guna Seetharaman, Kannappan Palaniappan
    * Abstract: In this paper, we present a moving object detection system named Flux Tensor with Split Gaussian models (FTSG) that exploits the benefits of fusing a motion computation method based on spatio-temporal tensor formulation, a novel foreground and background modeling scheme, and a multi-cue appearance comparison. This hybrid system can handle challenges such as shadows, illumination changes, dynamic background, stopped and removed objects. Extensive testing performed on the CVPR 2014 Change Detection benchmark dataset shows that FTSG outperforms state-ofthe-art methods.

count=10
* Meta Learning on a Sequence of Imbalanced Domains With Difficulty Awareness
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Meta_Learning_on_a_Sequence_of_Imbalanced_Domains_With_Difficulty_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Meta_Learning_on_a_Sequence_of_Imbalanced_Domains_With_Difficulty_ICCV_2021_paper.pdf)]
    * Title: Meta Learning on a Sequence of Imbalanced Domains With Difficulty Awareness
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zhenyi Wang, Tiehang Duan, Le Fang, Qiuling Suo, Mingchen Gao
    * Abstract: Recognizing new objects by learning from a few labeled examples in an evolving environment is crucial to obtain excellent generalization ability for real-world machine learning systems. A typical setting across current meta learning algorithms assumes a stationary task distribution during meta training. In this paper, we explore a more practical and challenging setting where task distribution changes over time with domain shift. Particularly, we consider realistic scenarios where task distribution is highly imbalanced with domain labels unavailable in nature. We propose a kernel-based method for domain change detection and a difficulty-aware memory management mechanism that jointly considers the imbalanced domain size and domain importance to learn across domains continuously. Furthermore, we introduce an efficient adaptive task sampling method during meta training, which significantly reduces task gradient variance with theoretical guarantees. Finally, we propose a challenging benchmark with imbalanced domain sequences and varied domain difficulty. We have performed extensive evaluations on the proposed benchmark, demonstrating the effectiveness of our method.

count=10
* Learning High-Density Regions for a Generalized Kolmogorov-Smirnov Test in High-Dimensional Data
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/6855456e2fe46a9d49d3d3af4f57443d-Paper.pdf)]
    * Title: Learning High-Density Regions for a Generalized Kolmogorov-Smirnov Test in High-Dimensional Data
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Assaf Glazer, Michael Lindenbaum, Shaul Markovitch
    * Abstract: We propose an efficient, generalized, nonparametric, statistical Kolmogorov-Smirnov test for detecting distributional change in high-dimensional data. To implement the test, we introduce a novel, hierarchical, minimum-volume sets estimator to represent the distributions to be tested. Our work is motivated by the need to detect changes in data streams, and the test is especially efficient in this context. We provide the theoretical foundations of our test and show its superiority over existing methods.

count=9
* ZBS: Zero-Shot Background Subtraction via Instance-Level Background Modeling and Foreground Selection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/An_ZBS_Zero-Shot_Background_Subtraction_via_Instance-Level_Background_Modeling_and_Foreground_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/An_ZBS_Zero-Shot_Background_Subtraction_via_Instance-Level_Background_Modeling_and_Foreground_CVPR_2023_paper.pdf)]
    * Title: ZBS: Zero-Shot Background Subtraction via Instance-Level Background Modeling and Foreground Selection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yongqi An, Xu Zhao, Tao Yu, Haiyun Guo, Chaoyang Zhao, Ming Tang, Jinqiao Wang
    * Abstract: Background subtraction (BGS) aims to extract all moving objects in the video frames to obtain binary foreground segmentation masks. Deep learning has been widely used in this field. Compared with supervised-based BGS methods, unsupervised methods have better generalization. However, previous unsupervised deep learning BGS algorithms perform poorly in sophisticated scenarios such as shadows or night lights, and they cannot detect objects outside the pre-defined categories. In this work, we propose an unsupervised BGS algorithm based on zero-shot object detection called Zero-shot Background Subtraction ZBS. The proposed method fully utilizes the advantages of zero-shot object detection to build the open-vocabulary instance-level background model. Based on it, the foreground can be effectively extracted by comparing the detection results of new frames with the background model. ZBS performs well for sophisticated scenarios, and it has rich and extensible categories. Furthermore, our method can easily generalize to other tasks, such as abandoned object detection in unseen environments. We experimentally show that ZBS surpasses state-of-the-art unsupervised BGS methods by 4.70% F-Measure on the CDnet 2014 dataset. The code is released at https://github.com/CASIA-IVA-Lab/ZBS.

count=9
* Flexible Background Subtraction With Self-Balanced Local Sensitivity
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf)]
    * Title: Flexible Background Subtraction With Self-Balanced Local Sensitivity
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: Most background subtraction approaches offer decent results in baseline scenarios, but adaptive and flexible solutions are still uncommon as many require scenario-specific parameter tuning to achieve optimal performance. In this paper, we introduce a new strategy to tackle this problem that focuses on balancing the inner workings of a non-parametric model based on pixel-level feedback loops. Pixels are modeled using a spatiotemporal feature descriptor for increased sensitivity. Using the video sequences and ground truth annotations of the 2012 and 2014 CVPR Change Detection Workshops, we demonstrate that our approach outperforms all previously ranked methods in the original dataset while achieving good results in the most recent one.

count=9
* An Anomaly Detection System via Moving Surveillance Robots With Human Collaboration
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/html/Zaheer_An_Anomaly_Detection_System_via_Moving_Surveillance_Robots_With_Human_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/papers/Zaheer_An_Anomaly_Detection_System_via_Moving_Surveillance_Robots_With_Human_ICCVW_2021_paper.pdf)]
    * Title: An Anomaly Detection System via Moving Surveillance Robots With Human Collaboration
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Muhammad Zaigham Zaheer, Arif Mahmood, M. Haris Khan, Marcella Astrid, Seung-Ik Lee
    * Abstract: Autonomous anomaly detection is a fundamental step in visual surveillance systems, and so we have witnessed great progress in the form of various promising algorithms. Nonetheless, majority of prior algorithms assume static surveillance cameras that severely restricts the coverage of the system unless the number of cameras is exponentially increased, consequently increasing both the installation and monitoring costs. In the current work we propose an anomaly detection system based on mobile surveillance cameras, i.e., moving robot which continuously navigates a target area. We compare the newly acquired test images with a database of normal images using geo-tags. For anomaly detection, a Siamese network is trained which analyses two input images for anomalies while ignoring the viewpoint differences. Further, our system is capable of updating the normal images database with human collaboration. Finally, we propose a new dataset that is captured by repeated visits of the robot over a target area. Our experiments demonstrate the effectiveness of the proposed system for anomaly detection using mobile surveillance robots.

count=9
* Word2Fun: Modelling Words as Functions for Diachronic Word Representation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/16a5cdae362b8d27a1d8f8c7b78b4330-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/16a5cdae362b8d27a1d8f8c7b78b4330-Paper.pdf)]
    * Title: Word2Fun: Modelling Words as Functions for Diachronic Word Representation
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Benyou Wang, Emanuele Di Buccio, Massimo Melucci
    * Abstract: Word meaning may change over time as a reflection of changes in human society. Therefore, modeling time in word representation is necessary for some diachronic tasks. Most existing diachronic word representation approaches train the embeddings separately for each pre-grouped time-stamped corpus and align these embeddings, e.g., by orthogonal projections, vector initialization, temporal referencing, and compass. However, not only does word meaning change in a short time, word meaning may also be subject to evolution over long timespans, thus resulting in a unified continuous process. A recent approach called `DiffTime' models semantic evolution as functions parameterized by multiple-layer nonlinear neural networks over time. In this paper, we will carry on this line of work by learning explicit functions over time for each word. Our approach, called `Word2Fun', reduces the space complexity from $\mathcal{O}(TVD)$ to $\mathcal{O}(kVD)$ where $k$ is a small constant ($k \ll T $). In particular, a specific instance based on polynomial functions could provably approximate any function modeling word evolution with a given negligible error thanks to the Weierstrass Approximation Theorem. The effectiveness of the proposed approach is evaluated in diverse tasks including time-aware word clustering, temporal analogy, and semantic change detection. Code at: {\url{https://github.com/wabyking/Word2Fun.git}}.

count=8
* Large-Scale Damage Detection Using Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.pdf)]
    * Title: Large-Scale Damage Detection Using Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Lionel Gueguen, Raffay Hamid
    * Abstract: Satellite imagery is a valuable source of information for assessing damages in distressed areas undergoing a calamity, such as an earthquake or an armed conflict. However, the sheer amount of data required to be inspected for this assessment makes it impractical to do it manually. To address this problem, we present a semi-supervised learning framework for large-scale damage detection in satellite imagery. We present a comparative evaluation of our framework using over 88 million images collected from 4,665 square kilometers from 12 different locations around the world. To enable accurate and efficient damage detection, we introduce a novel use of hierarchical shape features in the bags-of-visual words setting. We analyze how practical factors such as sun, sensor-resolution, and satellite-angle differences impact the effectiveness of our proposed representation, and compare it to five alternative features in multiple learning settings. Finally, we demonstrate through a user-study that our semi-supervised framework results in a ten-fold reduction in human annotation time at a minimal loss in detection accuracy compared to an exhaustive manual inspection.

count=8
* NestedVAE: Isolating Common Factors via Weak Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.pdf)]
    * Title: NestedVAE: Isolating Common Factors via Weak Supervision
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Matthew J. Vowels,  Necati Cihan Camgoz,  Richard Bowden
    * Abstract: Fair and unbiased machine learning is an important and active field of research, as decision processes are increasingly driven by models that learn from data. Unfortunately, any biases present in the data may be learned by the model, thereby inappropriately transferring that bias into the decision making process. We identify the connection between the task of bias reduction and that of isolating factors common between domains whilst encouraging domain specific invariance. To isolate the common factors we combine the theory of deep latent variable models with information bottleneck theory for scenarios whereby data may be naturally paired across domains and no additional supervision is required. The result is the Nested Variational AutoEncoder (NestedVAE). Two outer VAEs with shared weights attempt to reconstruct the input and infer a latent space, whilst a nested VAE attempts to reconstruct the latent representation of one image, from the latent representation of its paired image. In so doing, the nested VAE isolates the common latent factors/causes and becomes invariant to unwanted factors that are not shared between paired images. We also propose a new metric to provide a balanced method of evaluating consistency and classifier performance across domains which we refer to as the Adjusted Parity metric. An evaluation of NestedVAE on both domain and attribute invariance, change detection, and learning common factors for the prediction of biological sex demonstrates that NestedVAE significantly outperforms alternative methods.

count=8
* SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.pdf)]
    * Title: SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xin Guo, Jiangwei Lao, Bo Dang, Yingying Zhang, Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang, Kang Wu, Dingxiang Hu, Huimei He, Jian Wang, Jingdong Chen, Ming Yang, Yongjun Zhang, Yansheng Li
    * Abstract: Prior studies on Remote Sensing Foundation Model (RSFM) reveal immense potential towards a generic model for Earth Observation. Nevertheless these works primarily focus on a single modality without temporal and geo-context modeling hampering their capabilities for diverse tasks. In this study we present SkySense a generic billion-scale model pre-trained on a curated multi-modal Remote Sensing Imagery (RSI) dataset with 21.5 million temporal sequences. SkySense incorporates a factorized multi-modal spatiotemporal encoder taking temporal sequences of optical and Synthetic Aperture Radar (SAR) data as input. This encoder is pre-trained by our proposed Multi-Granularity Contrastive Learning to learn representations across different modal and spatial granularities. To further enhance the RSI representations by the geo-context clue we introduce Geo-Context Prototype Learning to learn region-aware prototypes upon RSI's multi-modal spatiotemporal features. To our best knowledge SkySense is the largest Multi-Modal RSFM to date whose modules can be flexibly combined or used individually to accommodate various tasks. It demonstrates remarkable generalization capabilities on a thorough evaluation encompassing 16 datasets over 7 tasks from single- to multi-modal static to temporal and classification to localization. SkySense surpasses 18 recent RSFMs in all test scenarios. Specifically it outperforms the latest models such as GFM SatLas and Scale-MAE by a large margin i.e. 2.76% 3.67% and 3.61% on average respectively. We will release the pre-trained weights to facilitate future research and Earth Observation applications.

count=8
* BSUV-Net: A Fully-Convolutional Neural Network for Background Subtraction of Unseen Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Tezcan_BSUV-Net_A_Fully-Convolutional_Neural_Network_for_Background_Subtraction_of_Unseen_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Tezcan_BSUV-Net_A_Fully-Convolutional_Neural_Network_for_Background_Subtraction_of_Unseen_WACV_2020_paper.pdf)]
    * Title: BSUV-Net: A Fully-Convolutional Neural Network for Background Subtraction of Unseen Videos
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Ozan Tezcan,  Prakash Ishwar,  Janusz Konrad
    * Abstract: Background subtraction is a basic task in computer vision and video processing often applied as a pre-processing step for object tracking, people recognition, etc. Recently, a number of successful background-subtraction algorithms have been proposed, however nearly all of the top-performing ones are supervised. Crucially, their success relies upon the availability of some annotated frames of the test video during training. Consequently, their performance on completely "unseen" videos is undocumented in the literature. In this work, we propose a new, supervised, background-subtraction algorithm for unseen videos (BSUV-Net) based on a fully-convolutional neural network. The input to our network consists of the current frame and two background frames captured at different time scales along with their semantic segmentation maps. In order to reduce the chance of overfitting, we also introduce a new data-augmentation technique which mitigates the impact of illumination difference between the background frames and the current frame. On the CDNet-2014 dataset, BSUV-Net outperforms state-of-the-art algorithms evaluated on unseen videos in terms of several metrics including F-measure, recall and precision.

count=8
* Density-Difference Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/f2fc990265c712c49d51a18a32b39f0c-Paper.pdf)]
    * Title: Density-Difference Estimation
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Masashi Sugiyama, Takafumi Kanamori, Taiji Suzuki, Marthinus Plessis, Song Liu, Ichiro Takeuchi
    * Abstract: We address the problem of estimating the difference between two probability densities. A naive approach is a two-step procedure of first estimating two densities separately and then computing their difference. However, such a two-step procedure does not necessarily work well because the first step is performed without regard to the second step and thus a small estimation error incurred in the first stage can cause a big error in the second stage. In this paper, we propose a single-shot procedure for directly estimating the density difference without separately estimating two densities. We derive a non-parametric finite-sample error bound for the proposed single-shot density-difference estimator and show that it achieves the optimal convergence rate. We then show how the proposed density-difference estimator can be utilized in L2-distance approximation. Finally, we experimentally demonstrate the usefulness of the proposed method in robust distribution comparison such as class-prior estimation and change-point detection.

count=7
* S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.pdf)]
    * Title: S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xuyang Li, Danfeng Hong, Jocelyn Chanussot
    * Abstract: In the expansive domain of computer vision a myriad of pre-trained models are at our disposal. However most of these models are designed for natural RGB images and prove inadequate for spectral remote sensing (RS) images. Spectral RS images have two main traits: (1) multiple bands capturing diverse feature information (2) spatial alignment and consistent spectral sequencing within the spatial-spectral dimension. In this paper we introduce Spatial-SpectralMAE (S2MAE) a specialized pre-trained architecture for spectral RS imagery. S2MAE employs a 3D transformer for masked autoencoder modeling integrating learnable spectral-spatial embeddings with a 90% masking ratio. The model efficiently captures local spectral consistency and spatial invariance using compact cube tokens demonstrating versatility to diverse input characteristics. This adaptability facilitates progressive pretraining on extensive spectral datasets. The effectiveness of S2MAE is validated through continuous pretraining on two sizable datasets totaling over a million training images. The pre-trained model is subsequently applied to three distinct downstream tasks with in-depth ablation studies conducted to emphasize its efficacy.

count=7
* A Temporal Scheme for Fast Learning of Image-Patch Correspondences in Realistic Multi-camera Setups
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W17/html/Eisenbach_A_Temporal_Scheme_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W17/papers/Eisenbach_A_Temporal_Scheme_2013_CVPR_paper.pdf)]
    * Title: A Temporal Scheme for Fast Learning of Image-Patch Correspondences in Realistic Multi-camera Setups
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Jens Eisenbach, Christian Conrad, Rudolf Mester
    * Abstract: This paper addresses the problem of finding corresponding image patches in multi-camera video streams by means of an unsupervised learning method. We determine patchto-patch correspondence relations ('correspondence priors') merely using information from a temporal change detection. Correspondence priors are essential for geometric multi-camera calibration, but are useful also for further vision tasks such as object tracking and recognition. Since any change detection method with reasonably performance can be applied, our method can be used as an encapsulated processing module and be integrated into existing systems without major structural changes. The only assumption that is made is that relative orientation of pairs of cameras may be arbitrary, but fixed, and that the observed scene shows visual activity. Experimental results show the applicability of the presented approach in real world scenarios where the camera views show large differences in orientation and position. Furthermore we show that a classic spatial matching pipeline, e.g., based on SIFT will typically fail to determine correspondences in these kinds of scenarios.

count=7
* Seamless Change Detection and Mosaicing for Aerial Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W11/html/T.M_Seamless_Change_Detection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W11/papers/T.M_Seamless_Change_Detection_2015_CVPR_paper.pdf)]
    * Title: Seamless Change Detection and Mosaicing for Aerial Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Nimisha T.M, A. N. Rajagopalan, Rangarajan Aravind
    * Abstract: The color appearance of an object can vary widely as a function of camera sensitivity and ambient illumination. In this paper, we discuss a methodology for seamless interfacing across imaging sensors and under varying illumination conditions for two very relevant problems in aerial imaging, namely, change detection and mosaicing. The proposed approach works by estimating surface reflectance which is an intrinsic property of the scene and is invariant to both camera and illumination. We advocate SIFT-based feature detection and matching in the reflectance domain followed by registration. We demonstrate that mosaicing and change detection when performed in the high-dimensional reflectance space yields better results as compared to operating in the 3-dimensional color space.

count=6
* Background Subtraction Using Local SVD Binary Pattern
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/Guo_Background_Subtraction_Using_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/Guo_Background_Subtraction_Using_CVPR_2016_paper.pdf)]
    * Title: Background Subtraction Using Local SVD Binary Pattern
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Lili Guo, Dan Xu, Zhenping Qiang
    * Abstract: Background subtraction is a basic problem for change detection in videos and also the first step of high-level computer vision applications. Most background subtraction methods rely on color and texture feature. However, due to illuminations changes in different scenes and affections of noise pixels, those methods often resulted in high false positives in a complex environment. To solve this problem, we propose an adaptive background subtraction model which uses a novel Local SVD Binary Pattern (named LSBP) feature instead of simply depending on color intensity. This feature can describe the potential structure of the local regions in a given image, thus, it can enhance the robustness to illumination variation, noise, and shadows. We use a sample consensus model which is well suited for our LSBP feature. Experimental results on CDnet 2012 dataset demonstrate that our background subtraction method using LSBP feature is more effective than many state-of-the-art methods.

count=6
* Exploring Real World Map Change Generalization of Prior-Informed HD Map Prediction Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/WAD/html/Bateman_Exploring_Real_World_Map_Change_Generalization_of_Prior-Informed_HD_Map_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/WAD/papers/Bateman_Exploring_Real_World_Map_Change_Generalization_of_Prior-Informed_HD_Map_CVPRW_2024_paper.pdf)]
    * Title: Exploring Real World Map Change Generalization of Prior-Informed HD Map Prediction Models
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Samuel M. Bateman, Ning Xu, H. Charles Zhao, Yael Ben Shalom, Vince Gong, Greg Long, Will Maddern
    * Abstract: Building and maintaining High-Definition (HD) maps represents a large barrier to autonomous vehicle deployment. This along with advances in modern online map detection models has sparked renewed interest in the online mapping problem. However effectively predicting online maps at a high enough quality to enable safe driverless deployments remains a significant challenge. Recent work on these models proposes training robust online mapping systems using low quality map priors with synthetic perturbations in an attempt to simulate out-of-date HD map priors. In this paper we investigate how models trained on these synthetically perturbed map priors generalize to performance on deployment-scale real world map changes. We present a large-scale experimental study to determine which synthetic perturbations are most useful in generalizing to real world HD map changes evaluated using multiple years of real-world autonomous driving data. We show there is still a substantial sim2real gap between synthetic prior perturbations and observed real-world changes which limits the utility of current prior-informed HD map prediction models.

count=6
* Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.pdf)]
    * Title: Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jhony H. Giraldo, Sajid Javed, Naoufel Werghi, Thierry Bouwmans
    * Abstract: Moving Object Detection (MOD) is a fundamental step for many computer vision applications. MOD becomes very challenging when a video sequence captured from a static or moving camera suffers from the challenges: camouflage, shadow, dynamic backgrounds, and lighting variations, to name a few. Deep learning methods have been successfully applied to address MOD with competitive performance. However, in order to handle the overfitting problem, deep learning methods require a large amount of labeled data which is a laborious task as exhaustive annotations are always not available. Moreover, some MOD deep learning methods show performance degradation in the presence of unseen video sequences because the testing and training splits of the same sequences are involved during the network learning process. In this work, we pose the problem of MOD as a node classification problem using Graph Convolutional Neural Networks (GCNNs). Our algorithm, dubbed as GraphMOD-Net, encompasses instance segmentation, background initialization, feature extraction, and graph construction. GraphMOD-Net is tested on unseen videos and outperforms state-of-the-art methods in unsupervised, semi-supervised, and supervised learning in several challenges of the Change Detection 2014 (CDNet2014) and UCSD background subtraction datasets.

count=6
* MotionRec: A Unified Deep Framework for Moving Object Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Mandal_MotionRec_A_Unified_Deep_Framework_for_Moving_Object_Recognition_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Mandal_MotionRec_A_Unified_Deep_Framework_for_Moving_Object_Recognition_WACV_2020_paper.pdf)]
    * Title: MotionRec: A Unified Deep Framework for Moving Object Recognition
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Murari Mandal,  Lav Kush Kumar,  Mahipal Singh Saran,  Santosh Kumar vipparthi
    * Abstract: In this paper we present a novel deep learning framework to perform online moving object recognition (MOR) in streaming videos. The existing methods for moving object detection (MOD) only computes class-agnostic pixel-wise binary segmentation of video frames. On the other hand, the object detection techniques do not differentiate between static and moving objects. To the best of our knowledge, this is a first attempt for simultaneous localization and classification of moving objects in a video, i.e. MOR in a single-stage deep learning framework. We achieve this by labelling axis-aligned bounding boxes for moving objects which requires less computational resources than producing pixel-level estimates. In the proposed MotionRec, both temporal and spatial features are learned using past history and current frames respectively. First, the background is estimated with a temporal depth reductionist (TDR) block. Then the estimated background, current frame and temporal median of recent observations are assimilated to encode spatiotemporal motion saliency. Moreover, feature pyramids are generated from these motion saliency maps to perform regression and classification at multiple levels of feature abstractions. MotionRec works online at inference as it requires only few past frames for MOR. Moreover, it doesn't require predefined target initialization from user. We also annotated axis-aligned bounding boxes (42,614 objects (14,814 cars and 27,800 person) in 24,923 video frames of CDnet 2014 dataset) due to lack of available benchmark datasets for MOR. The performance is observed qualitatively and quantitatively in terms of mAP over a defined unseen test set. Experiments show that the proposed MotionRec significantly improves over strong baselines with RetinaNet architectures for MOR.

count=6
* Active Learning for Improved Semi-Supervised Semantic Segmentation in Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Desai_Active_Learning_for_Improved_Semi-Supervised_Semantic_Segmentation_in_Satellite_Images_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Desai_Active_Learning_for_Improved_Semi-Supervised_Semantic_Segmentation_in_Satellite_Images_WACV_2022_paper.pdf)]
    * Title: Active Learning for Improved Semi-Supervised Semantic Segmentation in Satellite Images
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Shasvat Desai, Debasmita Ghose
    * Abstract: Remote sensing data is crucial for applications ranging from monitoring forest fires and deforestation to tracking urbanization. Most of these tasks require dense pixel-level annotations for the model to parse visual information from limited labeled data available for these satellite images. Due to the dearth of high-quality labeled training data in this domain, there is a need to focus on semi-supervised techniques. These techniques generate pseudo-labels from a small set of labeled examples which are used to augment the labeled training set. This makes it necessary to have a highly representative and diverse labeled training set. Therefore, we propose to use an active learning-based sampling strategy to select a highly representative set of labeled training data. We demonstrate our proposed method's effectiveness on two existing semantic segmentation datasets containing satellite images: UC Merced Land Use Classification Dataset and DeepGlobe Land Cover Classification Dataset. We report a 27% improvement in mIoU with as little as 2% labeled data using active learning sampling strategies over randomly sampling the small set of labeled training data.

count=6
* Detecting and Adapting to Irregular Distribution Shifts in Bayesian Online Learning
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/362387494f6be6613daea643a7706a42-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/362387494f6be6613daea643a7706a42-Paper.pdf)]
    * Title: Detecting and Adapting to Irregular Distribution Shifts in Bayesian Online Learning
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Aodong Li, Alex Boyd, Padhraic Smyth, Stephan Mandt
    * Abstract: We consider the problem of online learning in the presence of distribution shifts that occur at an unknown rate and of unknown intensity. We derive a new Bayesian online inference approach to simultaneously infer these distribution shifts and adapt the model to the detected changes by integrating ideas from change point detection, switching dynamical systems, and Bayesian online learning. Using a binary ‘change variable,’ we construct an informative prior such that--if a change is detected--the model partially erases the information of past model updates by tempering to facilitate adaptation to the new data distribution. Furthermore, the approach uses beam search to track multiple change-point hypotheses and selects the most probable one in hindsight. Our proposed method is model-agnostic, applicable in both supervised and unsupervised learning settings, suitable for an environment of concept drifts or covariate drifts, and yields improvements over state-of-the-art Bayesian online learning approaches.

count=6
* Spot the Difference: Detection of Topological Changes via Geometric Alignment
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/7867d6557b82ed3b5d61e6591a2a2fd3-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/7867d6557b82ed3b5d61e6591a2a2fd3-Paper.pdf)]
    * Title: Spot the Difference: Detection of Topological Changes via Geometric Alignment
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Per Steffen Czolbe, Aasa Feragen, Oswin Krause
    * Abstract: Geometric alignment appears in a variety of applications, ranging from domain adaptation, optimal transport, and normalizing flows in machine learning; optical flow and learned augmentation in computer vision and deformable registration within biomedical imaging. A recurring challenge is the alignment of domains whose topology is not the same; a problem that is routinely ignored, potentially introducing bias in downstream analysis. As a first step towards solving such alignment problems, we propose an unsupervised algorithm for the detection of changes in image topology. The model is based on a conditional variational auto-encoder and detects topological changes between two images during the registration step. We account for both topological changes in the image under spatial variation and unexpected transformations. Our approach is validated on two tasks and datasets: detection of topological changes in microscopy images of cells, and unsupervised anomaly detection brain imaging.

count=5
* Reconstructing Evolving Tree Structures in Time Lapse Sequences
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Glowacki_Reconstructing_Evolving_Tree_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Glowacki_Reconstructing_Evolving_Tree_2014_CVPR_paper.pdf)]
    * Title: Reconstructing Evolving Tree Structures in Time Lapse Sequences
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Przemyslaw Glowacki, Miguel Amavel Pinheiro, Engin Turetken, Raphael Sznitman, Daniel Lebrecht, Jan Kybic, Anthony Holtmaat, Pascal Fua
    * Abstract: We propose an approach to reconstructing tree structures that evolve over time in 2D images and 3D image stacks such as neuronal axons or plant branches. Instead of reconstructing structures in each image independently, we do so for all images simultaneously to take advantage of temporal-consistency constraints. We show that this problem can be formulated as a Quadratic Mixed Integer Program and solved efficiently. The outcome of our approach is a framework that provides substantial improvements in reconstructions over traditional single time-instance formulations. Furthermore, an added benefit of our approach is the ability to automatically detect places where significant changes have occurred over time, which is challenging when considering large amounts of data.

count=5
* Spatio-Temporal Self-Organizing Map Deep Network for Dynamic Object Detection From Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Du_Spatio-Temporal_Self-Organizing_Map_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Du_Spatio-Temporal_Self-Organizing_Map_CVPR_2017_paper.pdf)]
    * Title: Spatio-Temporal Self-Organizing Map Deep Network for Dynamic Object Detection From Videos
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yang Du, Chunfeng Yuan, Bing Li, Weiming Hu, Stephen Maybank
    * Abstract: In dynamic object detection, it is challenging to construct an effective model to sufficiently characterize the spatial-temporal properties of the background. This paper proposes a new Spatio-Temporal Self-Organizing Map (STSOM) deep network to detect dynamic objects in complex scenarios. The proposed approach has several contributions: First, a novel STSOM shared by all pixels in a video frame is presented to efficiently model complex background. We exploit the fact that the motions of complex background have the global variation in the space and the local variation in the time, to train STSOM using the whole frames and the sequence of a pixel over time to tackle the variance of complex background. Second, a Bayesian parameter estimation based method is presented to learn thresholds automatically for all pixels to filter out the background. Last, in order to model the complex background more accurately, we extend the single-layer STSOM to the deep network. Then the background is filtered out layer by layer. Experimental results on CDnet 2014 dataset demonstrate that the proposed STSOM deep network outperforms numerous recently proposed methods in the overall performance and in most categories of scenarios.

count=5
* Omnimatte: Associating Objects and Their Effects in Video
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lu_Omnimatte_Associating_Objects_and_Their_Effects_in_Video_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lu_Omnimatte_Associating_Objects_and_Their_Effects_in_Video_CVPR_2021_paper.pdf)]
    * Title: Omnimatte: Associating Objects and Their Effects in Video
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Erika Lu, Forrester Cole, Tali Dekel, Andrew Zisserman, William T. Freeman, Michael Rubinstein
    * Abstract: Computer vision has become increasingly better at segmenting objects in images and videos; however, scene effects related to the objects -- shadows, reflections, generated smoke, etc. -- are typically overlooked. Identifying such scene effects and associating them with the objects producing them is important for improving our fundamental understanding of visual scenes, and applications such as removing, duplicating, or enhancing objects in video. We take a step towards solving this novel problem of automatically associating objects with their effects in video. Given an ordinary video and a rough segmentation mask over time of one or more subjects of interest, we estimate an omnimatte for each subject -- an alpha matte and color image that includes the subject along with all its related time-varying scene elements. Our model is trained only on the input video in a self-supervised manner, without any manual labels, and is generic -- it produces omnimattes automatically for arbitrary objects and a variety of effects. We show results on real-world videos containing interactions between different types of subjects (cars, animals, people) and complex effects, ranging from semi-transparent smoke and reflections to fully opaque objects attached to the subject.

count=5
* Learning To Exploit Temporal Structure for Biomedical Vision-Language Processing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.pdf)]
    * Title: Learning To Exploit Temporal Structure for Biomedical Vision-Language Processing
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Shruthi Bannur, Stephanie Hyland, Qianchu Liu, Fernando Pérez-García, Maximilian Ilse, Daniel C. Castro, Benedikt Boecking, Harshita Sharma, Kenza Bouzid, Anja Thieme, Anton Schwaighofer, Maria Wetscherek, Matthew P. Lungren, Aditya Nori, Javier Alvarez-Valle, Ozan Oktay
    * Abstract: Self-supervised learning in vision--language processing (VLP) exploits semantic alignment between imaging and text modalities. Prior work in biomedical VLP has mostly relied on the alignment of single image and report pairs even though clinical notes commonly refer to prior images. This does not only introduce poor alignment between the modalities but also a missed opportunity to exploit rich self-supervision through existing temporal content in the data. In this work, we explicitly account for prior images and reports when available during both training and fine-tuning. Our approach, named BioViL-T, uses a CNN--Transformer hybrid multi-image encoder trained jointly with a text model. It is designed to be versatile to arising challenges such as pose variations and missing input images across time. The resulting model excels on downstream tasks both in single- and multi-image setups, achieving state-of-the-art (SOTA) performance on (I) progression classification, (II) phrase grounding, and (III) report generation, whilst offering consistent improvements on disease classification and sentence-similarity tasks. We release a novel multi-modal temporal benchmark dataset, CXR-T, to quantify the quality of vision--language representations in terms of temporal semantics. Our experimental results show the significant advantages of incorporating prior images and reports to make most use of the data.

count=5
* Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf)]
    * Title: Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, Yinghuan Shi
    * Abstract: In this work, we revisit the weak-to-strong consistency framework, popularized by FixMatch from semi-supervised classification, where the prediction of a weakly perturbed image serves as supervision for its strongly perturbed version. Intriguingly, we observe that such a simple pipeline already achieves competitive results against recent advanced works, when transferred to our segmentation scenario. Its success heavily relies on the manual design of strong data augmentations, however, which may be limited and inadequate to explore a broader perturbation space. Motivated by this, we propose an auxiliary feature perturbation stream as a supplement, leading to an expanded perturbation space. On the other, to sufficiently probe original image-level augmentations, we present a dual-stream perturbation technique, enabling two strong views to be simultaneously guided by a common weak view. Consequently, our overall Unified Dual-Stream Perturbations approach (UniMatch) surpasses all existing methods significantly across all evaluation protocols on the Pascal, Cityscapes, and COCO benchmarks. Its superiority is also demonstrated in remote sensing interpretation and medical image analysis. We hope our reproduced FixMatch and our results can inspire more future works. Code and logs are available at https://github.com/LiheYoung/UniMatch.

count=5
* A Framework for Semi-Automatic Collection of Temporal Satellite Imagery for Analysis of Dynamic Regions
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/html/Motlagh_A_Framework_for_Semi-Automatic_Collection_of_Temporal_Satellite_Imagery_for_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/papers/Motlagh_A_Framework_for_Semi-Automatic_Collection_of_Temporal_Satellite_Imagery_for_ICCVW_2021_paper.pdf)]
    * Title: A Framework for Semi-Automatic Collection of Temporal Satellite Imagery for Analysis of Dynamic Regions
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Nicholas Kashani Motlagh, Aswathnarayan Radhakrishnan, Jim Davis, Roman Ilin
    * Abstract: Analyzing natural and anthropogenic activities using remote sensing data has become a problem of increasing interest. However, this generally involves tediously labeling extensive imagery, perhaps on a global scale. The lack of a streamlined method to collect and label imagery over time makes it challenging to tackle these problems using popular, supervised deep learning approaches. We address this need by presenting a framework to semi-automatically collect and label dynamic regions in satellite imagery using crowd-sourced OpenStreetMap data and available satellite imagery resources. The generated labels can be quickly verified to ease the burden of full manual labeling. We leverage this framework for the ability to gather image sequences of areas that have label reclassification over time. One possible application of our framework is demonstrated to collect and classify construction vs. non-construction sites. Overall, the proposed framework can be adapted for similar change detection or classification tasks in various remote sensing applications.

count=5
* TransBlast: Self-Supervised Learning Using Augmented Subspace With Transformer for Background/Foreground Separation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Osman_TransBlast_Self-Supervised_Learning_Using_Augmented_Subspace_With_Transformer_for_BackgroundForeground_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Osman_TransBlast_Self-Supervised_Learning_Using_Augmented_Subspace_With_Transformer_for_BackgroundForeground_ICCVW_2021_paper.pdf)]
    * Title: TransBlast: Self-Supervised Learning Using Augmented Subspace With Transformer for Background/Foreground Separation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Islam Osman, Mohamed Abdelpakey, Mohamed S. Shehata
    * Abstract: Background/Foreground separation is a fundamental and challenging task of many computer vision applications. The F-measure performance of state-of-the-art models is limited due to the lack of fine details in the predicted output (i.e., the foreground object) and the limited labeled data. In this paper, we propose a background/foreground separation model based on a transformer that has a higher learning capacity than the convolutional neural networks. The model is trained using self-supervised learning to leverage the limited data and learn a strong object representation that is invariant to changes. The proposed method, dubbed TransBlast, reformulates the background/foreground separation problem in self-supervised learning using the augmented subspace loss function. The augmented subspace loss function consists of two components: 1) the cross-entropy loss function and 2) the subspace that depends on Singular Value Decomposition (SVD). The proposed model is evaluated using three benchmarks, namely CDNet, DAVIS, and SegTrackV2. The performance of TransBlast outperforms state-of-the-art background/foreground separation models in terms of F-measure.

count=5
* MSNet: A Multilevel Instance Segmentation Network for Natural Disaster Damage Assessment in Aerial Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Zhu_MSNet_A_Multilevel_Instance_Segmentation_Network_for_Natural_Disaster_Damage_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Zhu_MSNet_A_Multilevel_Instance_Segmentation_Network_for_Natural_Disaster_Damage_WACV_2021_paper.pdf)]
    * Title: MSNet: A Multilevel Instance Segmentation Network for Natural Disaster Damage Assessment in Aerial Videos
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Xiaoyu Zhu, Junwei Liang, Alexander Hauptmann
    * Abstract: In this paper, we study the problem of efficiently assessing building damage after natural disasters like hurricanes, floods or fires, through aerial video analysis. We make two main contributions. The first contribution is a new dataset, consisting of user-generated aerial videos from social media with annotations of instance-level building damage masks. This provides the first benchmark for quantitative evaluation of models to assess building damage using aerial videos. The second contribution is a new model, namely MSNet, which contains novel region proposal network designs and an unsupervised score refinement network for confidence score calibration in both bounding box and mask branches. We show that our model achieves state-of-the-art results compared to previous methods in our dataset.

count=5
* Autoencoder-Based Background Reconstruction and Foreground Segmentation With Background Noise Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.pdf)]
    * Title: Autoencoder-Based Background Reconstruction and Foreground Segmentation With Background Noise Estimation
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Bruno Sauvalle, Arnaud de La Fortelle
    * Abstract: Even after decades of research, dynamic scene background reconstruction and foreground object segmentation are still considered as open problems due to various challenges such as illumination changes, camera movements, or background noise caused by air turbulence or moving trees. We propose in this paper to model the background of a frame sequence as a low dimensional manifold using an autoencoder and compare the reconstructed background provided by this autoencoder with the original image to compute the foreground/background segmentation masks. The main novelty of the proposed model is that the autoencoder is also trained to predict the background noise, which allows to compute for each frame a pixel-dependent threshold to perform the foreground segmentation. Although the proposed model does not use any temporal or motion information, it exceeds the state of the art for unsupervised background subtraction on the CDnet 2014 and LASIESTA datasets, with a significant improvement on videos where the camera is moving. It is also able to perform background reconstruction on some non-video image datasets.

count=5
* SSL4EO-L: Datasets and Foundation Models for Landsat Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/bbf7ee04e2aefec136ecf60e346c2e61-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/bbf7ee04e2aefec136ecf60e346c2e61-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: SSL4EO-L: Datasets and Foundation Models for Landsat Imagery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Adam Stewart, Nils Lehmann, Isaac Corley, Yi Wang, Yi-Chia Chang, Nassim Ait Ait Ali Braham, Shradha Sehgal, Caleb Robinson, Arindam Banerjee
    * Abstract: The Landsat program is the longest-running Earth observation program in history, with 50+ years of data acquisition by 8 satellites. The multispectral imagery captured by sensors onboard these satellites is critical for a wide range of scientific fields. Despite the increasing popularity of deep learning and remote sensing, the majority of researchers still use decision trees and random forests for Landsat image analysis due to the prevalence of small labeled datasets and lack of foundation models. In this paper, we introduce SSL4EO-L, the first ever dataset designed for Self-Supervised Learning for Earth Observation for the Landsat family of satellites (including 3 sensors and 2 product levels) and the largest Landsat dataset in history (5M image patches). Additionally, we modernize and re-release the L7 Irish and L8 Biome cloud detection datasets, and introduce the first ML benchmark datasets for Landsats 4–5 TM and Landsat 7 ETM+ SR. Finally, we pre-train the first foundation models for Landsat imagery using SSL4EO-L and evaluate their performance on multiple semantic segmentation tasks. All datasets and model weights are available via the TorchGeo library, making reproducibility and experimentation easy, and enabling scientific advancements in the burgeoning field of remote sensing for a multitude of downstream applications.

count=5
* Decoding the Enigma: Benchmarking Humans and AIs on the Many Facets of Working Memory
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/ea8758dbe6cc5e6e1764c009acb4c31e-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/ea8758dbe6cc5e6e1764c009acb4c31e-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Decoding the Enigma: Benchmarking Humans and AIs on the Many Facets of Working Memory
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Ankur Sikarwar, Mengmi Zhang
    * Abstract: Working memory (WM), a fundamental cognitive process facilitating the temporary storage, integration, manipulation, and retrieval of information, plays a vital role in reasoning and decision-making tasks. Robust benchmark datasets that capture the multifaceted nature of WM are crucial for the effective development and evaluation of AI WM models. Here, we introduce a comprehensive Working Memory (WorM) benchmark dataset for this purpose. WorM comprises 10 tasks and a total of 1 million trials, assessing 4 functionalities, 3 domains, and 11 behavioral and neural characteristics of WM. We jointly trained and tested state-of-the-art recurrent neural networks and transformers on all these tasks. We also include human behavioral benchmarks as an upper bound for comparison. Our results suggest that AI models replicate some characteristics of WM in the brain, most notably primacy and recency effects, and neural clusters and correlates specialized for different domains and functionalities of WM. In the experiments, we also reveal some limitations in existing models to approximate human behavior. This dataset serves as a valuable resource for communities in cognitive psychology, neuroscience, and AI, offering a standardized framework to compare and enhance WM models, investigate WM's neural underpinnings, and develop WM models with human-like capabilities. Our source code and data are available at: https://github.com/ZhangLab-DeepNeuroCogLab/WorM

count=4
* 6D Dynamic Camera Relocalization From Single Reference Image
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Feng_6D_Dynamic_Camera_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Feng_6D_Dynamic_Camera_CVPR_2016_paper.pdf)]
    * Title: 6D Dynamic Camera Relocalization From Single Reference Image
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Wei Feng, Fei-Peng Tian, Qian Zhang, Jizhou Sun
    * Abstract: Dynamic relocalization of 6D camera pose from single reference image is a costly and challenging task that requires delicate hand-eye calibration and precision positioning platform to do 3D mechanical rotation and translation. In this paper, we show that high-quality camera relocalization can be achieved in a much less expensive way. Based on inexpensive platform with unreliable absolute repositioning accuracy (ARA), we propose a hand-eye calibration free strategy to actively relocate camera into the same 6D pose that produces the input reference image, by sequentially correcting 3D relative rotation and translation. We theoretically prove that, by this strategy, both rotational and translational relative pose can be effectively reduced to zero, with bounded unknown hand-eye pose displacement. To conquer 3D rotation and translation ambiguity, this theoretical strategy is further revised to a practical relocalization algorithm with faster convergence rate and more reliability by jointly adjusting 3D relative rotation and translation. Extensive experiments validate the effectiveness and superior accuracy of the proposed approach on laboratory tests and challenging real-world applications.

count=4
* GeoEngine: A Platform for Production-Ready Geospatial Research
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Verma_GeoEngine_A_Platform_for_Production-Ready_Geospatial_Research_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Verma_GeoEngine_A_Platform_for_Production-Ready_Geospatial_Research_CVPR_2022_paper.pdf)]
    * Title: GeoEngine: A Platform for Production-Ready Geospatial Research
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Sagar Verma, Siddharth Gupta, Hal Shin, Akash Panigrahi, Shubham Goswami, Shweta Pardeshi, Natanael Exe, Ujwal Dutta, Tanka Raj Joshi, Nitin Bhojwani
    * Abstract: Geospatial machine learning has seen tremendous academic advancement, but its practical application has been constrained by difficulties with operationalizing performant and reliable solutions. Sourcing satellite imagery in real-world settings, handling terabytes of training data, and managing machine learning artifacts are a few of the challenges that have severely limited downstream innovation. In this paper we introduce the GeoEngine platform for reproducible and production-ready geospatial machine learning research. GeoEngine removes key technical hurdles to adopting computer vision and deep learning-based geospatial solutions at scale. It is the first end-to-end geospatial machine learning platform, simplifying access to insights locked behind petabytes of imagery. Backed by a rigorous research methodology, this geospatial framework empowers researchers with powerful abstractions for image sourcing, dataset development, model development, large scale training, and model deployment. In this paper we provide the GeoEngine architecture explaining our design rationale in detail. We provide several real-world use cases of image sourcing, dataset development, and model building that have helped different organisations build and deploy geospatial solutions.

count=4
* Graph Representation for Order-Aware Visual Transformation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Qiu_Graph_Representation_for_Order-Aware_Visual_Transformation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Qiu_Graph_Representation_for_Order-Aware_Visual_Transformation_CVPR_2023_paper.pdf)]
    * Title: Graph Representation for Order-Aware Visual Transformation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yue Qiu, Yanjun Sun, Fumiya Matsuzawa, Kenji Iwata, Hirokatsu Kataoka
    * Abstract: This paper proposes a new visual reasoning formulation that aims at discovering changes between image pairs and their temporal orders. Recognizing scene dynamics and their chronological orders is a fundamental aspect of human cognition. The aforementioned abilities make it possible to follow step-by-step instructions, reason about and analyze events, recognize abnormal dynamics, and restore scenes to their previous states. However, it remains unclear how well current AI systems perform in these capabilities. Although a series of studies have focused on identifying and describing changes from image pairs, they mainly consider those changes that occur synchronously, thus neglecting potential orders within those changes. To address the above issue, we first propose a visual transformation graph structure for conveying order-aware changes. Then, we benchmarked previous methods on our newly generated dataset and identified the issues of existing methods for change order recognition. Finally, we show a significant improvement in order-aware change recognition by introducing a new model that explicitly associates different changes and then identifies changes and their orders in a graph representation.

count=4
* Generalized Event Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sundar_Generalized_Event_Cameras_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sundar_Generalized_Event_Cameras_CVPR_2024_paper.pdf)]
    * Title: Generalized Event Cameras
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Varun Sundar, Matthew Dutson, Andrei Ardelean, Claudio Bruschini, Edoardo Charbon, Mohit Gupta
    * Abstract: Event cameras capture the world at high time resolution and with minimal bandwidth requirements. However event streams which only encode changes in brightness do not contain sufficient scene information to support a wide variety of downstream tasks. In this work we design generalized event cameras that inherently preserve scene intensity in a bandwidth-efficient manner. We generalize event cameras in terms of when an event is generated and what information is transmitted. To implement our designs we turn to single-photon sensors that provide digital access to individual photon detections; this modality gives us the flexibility to realize a rich space of generalized event cameras. Our single-photon event cameras are capable of high-speed high-fidelity imaging at low readout rates. Consequently these event cameras can support plug-and-play downstream inference without capturing new event datasets or designing specialized event-vision models. As a practical implication our designs which involve lightweight and near-sensor-compatible computations provide a way to use single-photon sensors without exorbitant bandwidth costs.

count=4
* Detecting Out-Of-Distribution Earth Observation Images with Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Le_Bellier_Detecting_Out-Of-Distribution_Earth_Observation_Images_with_Diffusion_Models_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Le_Bellier_Detecting_Out-Of-Distribution_Earth_Observation_Images_with_Diffusion_Models_CVPRW_2024_paper.pdf)]
    * Title: Detecting Out-Of-Distribution Earth Observation Images with Diffusion Models
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Georges Le Bellier, Nicolas Audebert
    * Abstract: Earth Observation imagery can capture rare and unusual events such as disasters and major landscape changes whose visual appearance contrasts with the usual observations. Deep models trained on common remote sensing data will output drastically different features for these out-of-distribution samples compared to those closer to their training dataset. Detecting them could therefore help anticipate changes in the observations either geographical or environmental. In this work we show that the reconstruction error of diffusion models can effectively serve as unsupervised out-of-distribution detectors for remote sensing images using them as a plausibility score. Moreover we introduce ODEED a novel reconstruction-based scorer using the probability-flow ODE of diffusion models. We validate it experimentally on SpaceNet 8 with various scenarios such as classical OOD detection with geographical shift and near-OOD setups: pre/post-flood and non-flooded/flooded image recognition. We show that our ODEED scorer significantly outperforms other diffusion-based and discriminative baselines on the more challenging near-OOD scenarios of flood image detection where OOD images are close to the distribution tail. We aim to pave the way towards better use of generative models for anomaly detection in remote sensing.

count=4
* Change Detection with Weightless Neural Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Gregorio_Change_Detection_with_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Gregorio_Change_Detection_with_2014_CVPR_paper.pdf)]
    * Title: Change Detection with Weightless Neural Networks
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Massimo De Gregorio, Maurizio Giordano
    * Abstract: In this paper a pixel'based Weightless Neural Network (WNN) method to face the problem of change detection in the field of view of a camera is proposed. The main features of the proposed method are 1) the dynamic adaptability to background change due to the WNN model adopted and 2) the introduction of pixel color histories to improve system behavior in videos characterized by (des)appearing of objects in video scene and/or sudden changes in lightning and background brightness and shape. The WNN approach is very simple and straightforward, and it gives high rank results in competition with other approaches applied to the ChangeDetection.net 2014 benchmark dataset.

count=4
* Unmasking the Abnormal Events in Video
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Ionescu_Unmasking_the_Abnormal_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Ionescu_Unmasking_the_Abnormal_ICCV_2017_paper.pdf)]
    * Title: Unmasking the Abnormal Events in Video
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Radu Tudor Ionescu, Sorina Smeureanu, Bogdan Alexe, Marius Popescu
    * Abstract: We propose a novel framework for abnormal event detection in video that requires no training sequences. Our framework is based on unmasking, a technique previously used for authorship verification in text documents, which we adapt to our task. We iteratively train a binary classifier to distinguish between two consecutive video sequences while removing at each step the most discriminant features. Higher training accuracy rates of the intermediately obtained classifiers represent abnormal events. To the best of our knowledge, this is the first work to apply unmasking for a computer vision task. We compare our method with several state-of-the-art supervised and unsupervised methods on four benchmark data sets. The empirical results indicate that our abnormal event detection framework can achieve state-of-the-art results, while running in real-time at 20 frames per second.

count=4
* Space-Time Localization and Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Lee_Space-Time_Localization_and_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Space-Time_Localization_and_ICCV_2017_paper.pdf)]
    * Title: Space-Time Localization and Mapping
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Minhaeng Lee, Charless C. Fowlkes
    * Abstract: This paper addresses the problem of building a spatio-temporal model of the world from a stream of time-stamped data. Unlike traditional models for simultaneous localization and mapping (SLAM) and structure-from-motion (SfM) which focus on recovering a single rigid 3D model, we tackle the problem of mapping scenes in which dynamic components appear, move and disappear independently of each other over time. We introduce a simple generative probabilistic model of 4D structure which specifies location, spatial and temporal extent of rigid surface patches by local Gaussian mixtures. We fit this model to a time-stamped stream of input data using expectation-maximization to estimate the model structure parameters (mapping) and the alignment of the input data to the model (localization). By explicitly representing the temporal extent and observability of surfaces in a scene, our method yields superior localization and reconstruction relative to baselines that assume a static 3D scene. We carry out experiments on both synthetic RGB-D data streams as well as challenging real-world datasets, tracking scene dynamics in a human workspace over the course of several weeks.

count=4
* RIO: 3D Object Instance Re-Localization in Changing Indoor Environments
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.pdf)]
    * Title: RIO: 3D Object Instance Re-Localization in Changing Indoor Environments
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Johanna Wald,  Armen Avetisyan,  Nassir Navab,  Federico Tombari,  Matthias Niessner
    * Abstract: In this work, we introduce the task of 3D object instance re-localization (RIO): given one or multiple objects in an RGB-D scan, we want to estimate their corresponding 6DoF poses in another 3D scan of the same environment taken at a later point in time. We consider RIO a particularly important task in 3D vision since it enables a wide range of practical applications, including AI-assistants or robots that are asked to find a specific object in a 3D scene. To address this problem, we first introduce 3RScan, a novel dataset and benchmark, which features 1482 RGB-D scans of 478 environments across multiple time steps. Each scene includes several objects whose positions change over time, together with ground truth annotations of object instances and their respective 6DoF mappings among re-scans. Automatically finding 6DoF object poses leads to a particular challenging feature matching task due to varying partial observations and changes in the surrounding context. To this end, we introduce a new data-driven approach that efficiently finds matching features using a fully-convolutional 3D correspondence network operating on multiple spatial scales. Combined with a 6DoF pose optimization, our method outperforms state-of-the-art baselines on our newly-established benchmark, achieving an accuracy of 30.58%.

count=4
* Viewpoint-Agnostic Change Captioning With Cycle Consistency
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Viewpoint-Agnostic_Change_Captioning_With_Cycle_Consistency_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Viewpoint-Agnostic_Change_Captioning_With_Cycle_Consistency_ICCV_2021_paper.pdf)]
    * Title: Viewpoint-Agnostic Change Captioning With Cycle Consistency
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Hoeseong Kim, Jongseok Kim, Hyungseok Lee, Hyunsung Park, Gunhee Kim
    * Abstract: Change captioning is the task of identifying the change and describing it with a concise caption. Despite recent advancements, filtering out insignificant changes still remains as a challenge. Namely, images from different camera perspectives can cause issues; a mere change in viewpoint should be disregarded while still capturing the actual changes. In order to tackle this problem, we present a new Viewpoint-Agnostic change captioning network with Cycle Consistency (VACC) that requires only one image each for the before and after scene, without depending on any other information. We achieve this by devising a new difference encoder module which can encode viewpoint information and model the difference more effectively. In addition, we propose a cycle consistency module that can potentially improve the performance of any change captioning networks in general by matching the composite feature of the generated caption and before image with the after image feature. We evaluate the performance of our proposed model across three datasets for change captioning, including a novel dataset we introduce here that contains images with changes under extreme viewpoint shifts. Through our experiments, we show the excellence of our method with respect to the CIDEr, BLEU-4, METEOR and SPICE scores. Moreover, we demonstrate that attaching our proposed cycle consistency module yields a performance boost for existing change captioning networks, even with varying image encoding mechanisms.

count=4
* ZRG: A Dataset for Multimodal 3D Residential Rooftop Understanding
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Corley_ZRG_A_Dataset_for_Multimodal_3D_Residential_Rooftop_Understanding_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Corley_ZRG_A_Dataset_for_Multimodal_3D_Residential_Rooftop_Understanding_WACV_2024_paper.pdf)]
    * Title: ZRG: A Dataset for Multimodal 3D Residential Rooftop Understanding
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Isaac Corley, Jonathan Lwowski, Peyman Najafirad
    * Abstract: A crucial part of any home is the roof over our heads to protect us from the elements. In this paper we present the Zeitview Rooftop Geometry (ZRG) dataset for residential rooftop understanding. ZRG is a large-scale residential rooftop inspection dataset of over 20k properties from across the U.S. and includes high resolution aerial orthomosaics, digital surface models (DSM), colored point clouds, and 3D roof wireframe annotations. We provide an in-depth analysis and perform several experimental baselines including roof outline extraction, monocular height estimation, and planar roof structure extraction, to illustrate a few of the numerous applications unlocked by this dataset.

count=3
* Online Dominant and Anomalous Behavior Detection in Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Roshtkhari_Online_Dominant_and_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Roshtkhari_Online_Dominant_and_2013_CVPR_paper.pdf)]
    * Title: Online Dominant and Anomalous Behavior Detection in Videos
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Mehrsan Javan Roshtkhari, Martin D. Levine
    * Abstract: We present a novel approach for video parsing and simultaneous online learning of dominant and anomalous behaviors in surveillance videos. Dominant behaviors are those occurring frequently in videos and hence, usually do not attract much attention. They can be characterized by different complexities in space and time, ranging from a scene background to human activities. In contrast, an anomalous behavior is defined as having a low likelihood of occurrence. We do not employ any models of the entities in the scene in order to detect these two kinds of behaviors. In this paper, video events are learnt at each pixel without supervision using densely constructed spatio-temporal video volumes. Furthermore, the volumes are organized into large contextual graphs. These compositions are employed to construct a hierarchical codebook model for the dominant behaviors. By decomposing spatio-temporal contextual information into unique spatial and temporal contexts, the proposed framework learns the models of the dominant spatial and temporal events. Thus, it is ultimately capable of simultaneously modeling high-level behaviors as well as low-level spatial, temporal and spatio-temporal pixel level changes.

count=3
* StoryGraphs: Visualizing Character Interactions as a Timeline
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Tapaswi_StoryGraphs_Visualizing_Character_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Tapaswi_StoryGraphs_Visualizing_Character_2014_CVPR_paper.pdf)]
    * Title: StoryGraphs: Visualizing Character Interactions as a Timeline
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Makarand Tapaswi, Martin Bauml, Rainer Stiefelhagen
    * Abstract: We present a novel way to automatically summarize and represent the storyline of a TV episode by visualizing character interactions as a chart. We also propose a scene detection method that lends itself well to generate over-segmented scenes which is used to partition the video. The positioning of character lines in the chart is formulated as an optimization problem which trades between the aesthetics and functionality of the chart. Using automatic person identification, we present StoryGraphs for 3 diverse TV series encompassing a total of 22 episodes. We define quantitative criteria to evaluate StoryGraphs and also compare them against episode summaries to evaluate their ability to provide an overview of the episode.

count=3
* The TUM-DLR Multimodal Earth Observation Evaluation Benchmark
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w19/html/Koch_The_TUM-DLR_Multimodal_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w19/papers/Koch_The_TUM-DLR_Multimodal_CVPR_2016_paper.pdf)]
    * Title: The TUM-DLR Multimodal Earth Observation Evaluation Benchmark
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Tobias Koch, Pablo d'Angelo, Franz Kurz, Friedrich Fraundorfer, Peter Reinartz, Marco Korner
    * Abstract: We present a new dataset for development, benchmarking, and evaluation of remote sensing and earth observation approaches with special focus on converging perspectives. In order to provide data with different modalities, we observed the same scene using satellites, airplanes, unmanned aerial vehicles (UAV), and smartphones. The dataset is further complemented by ground-truth information and baseline results for different application scenarios. The provided data can be freely used by anybody interested in remote sensing and earth observation and will be continuously augmented and updated.

count=3
* Temporal Vegetation Modelling Using Long Short-Term Memory Networks for Crop Identification From Medium-Resolution Multi-Spectral Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Russwurm_Temporal_Vegetation_Modelling_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Russwurm_Temporal_Vegetation_Modelling_CVPR_2017_paper.pdf)]
    * Title: Temporal Vegetation Modelling Using Long Short-Term Memory Networks for Crop Identification From Medium-Resolution Multi-Spectral Satellite Images
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Marc Russwurm, Marco Korner
    * Abstract: Land-cover classification is one of the key problems in earth observation and extensively investigated over the recent decades. Usually, approaches concentrate on single-time and multi- or hyperspectral reflectance space- or airborne sensor measurements observed. However, land-cover classes, e.g., crops, change their reflective characteristics over time complicating classification at one observation time. Contrary, these features change in a systematic and predictive manner, which can be utilized in a multi-temporal approach. We use long short-term memory (LSTM) networks to extract temporal characteristics from a sequence of Sentinel-2 observations. We compare the performance of LSTM and other network architectures and a SVM baseline to show the effectiveness of dynamic temporal feature extraction. A large test area combined with rich ground truth labels was used for training and evaluation. Our LSTM variant achieves state-of-the art performance opening potential for further research.

count=3
* JA-POLS: A Moving-Camera Background Model via Joint Alignment and Partially-Overlapping Local Subspaces
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chelly_JA-POLS_A_Moving-Camera_Background_Model_via_Joint_Alignment_and_Partially-Overlapping_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chelly_JA-POLS_A_Moving-Camera_Background_Model_via_Joint_Alignment_and_Partially-Overlapping_CVPR_2020_paper.pdf)]
    * Title: JA-POLS: A Moving-Camera Background Model via Joint Alignment and Partially-Overlapping Local Subspaces
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Irit Chelly,  Vlad Winter,  Dor Litvak,  David Rosen,  Oren Freifeld
    * Abstract: Background models are widely used in computer vision. While successful Static-camera Background (SCB) models exist, Moving-camera Background (MCB) models are limited. Seemingly, there is a straightforward solution: 1) align the video frames; 2) learn an SCB model; 3) warp either original or previously-unseen frames toward the model. This approach, however, has drawbacks, especially when the accumulative camera motion is large and/or the video is long. Here we propose a purely-2D unsupervised modular method that systematically eliminates those issues. First, to estimate warps in the original video, we solve a joint-alignment problem while leveraging a certifiably-correct initialization. Next, we learn both multiple partially-overlapping local subspaces and how to predict alignments. Lastly, in test time, we warp a previously-unseen frame, based on the prediction, and project it on a subset of those subspaces to obtain a background/foreground separation. We show the method handles even large scenes with a relatively-free camera motion (provided the camera-to-scene distance does not change much) and that it not only yields State-of-the-Art results on the original video but also generalizes gracefully to previously-unseen videos of the same scene. Our code is available at https://github.com/BGU-CS-VIL/JA-POLS.

count=3
* Temporal Action Segmentation From Timestamp Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Temporal_Action_Segmentation_From_Timestamp_Supervision_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Temporal_Action_Segmentation_From_Timestamp_Supervision_CVPR_2021_paper.pdf)]
    * Title: Temporal Action Segmentation From Timestamp Supervision
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Zhe Li, Yazan Abu Farha, Jurgen Gall
    * Abstract: Temporal action segmentation approaches have been very successful recently. However, annotating videos with frame-wise labels to train such models is very expensive and time consuming. While weakly supervised methods trained using only ordered action lists require less annotation effort, the performance is still worse than fully supervised approaches. In this paper, we propose to use timestamp supervision for the temporal action segmentation task. Timestamps require a comparable annotation effort to weakly supervised approaches, and yet provide a more supervisory signal. To demonstrate the effectiveness of timestamp supervision, we propose an approach to train a segmentation model using only timestamps annotations. Our approach uses the model output and the annotated timestamps to generate frame-wise labels by detecting the action changes. We further introduce a confidence loss that forces the predicted probabilities to monotonically decrease as the distance to the timestamps increases. This ensures that all and not only the most distinctive frames of an action are learned during training. The evaluation on four datasets shows that models trained with timestamps annotations achieve comparable performance to the fully supervised approaches.

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
* Perceptual Loss for Robust Unsupervised Homography Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/IMW/html/Koguciuk_Perceptual_Loss_for_Robust_Unsupervised_Homography_Estimation_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/IMW/papers/Koguciuk_Perceptual_Loss_for_Robust_Unsupervised_Homography_Estimation_CVPRW_2021_paper.pdf)]
    * Title: Perceptual Loss for Robust Unsupervised Homography Estimation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Daniel Koguciuk, Elahe Arani, Bahram Zonooz
    * Abstract: Homography estimation is often an indispensable step in many computer vision tasks. The existing approaches, however, are not robust to illumination and/or larger viewpoint changes. In this paper, we propose bidirectional implicit Homography Estimation (biHomE) loss for unsupervised homography estimation. biHomE minimizes the distance in the feature space between the warped image from the source viewpoint and the corresponding image from the target viewpoint. Since we use a fixed pre-trained feature extractor and the only learnable component of our framework is the homography network, we effectively decouple the homography estimation from representation learning. We use an additional photometric distortion step in the synthetic COCO dataset generation to better represent the illumination variation of the real-world scenarios. We show that biHomE achieves state-of-the-art performance on synthetic COCO dataset, which is also comparable or better compared to supervised approaches. Furthermore, the empirical results demonstrate the robustness of our approach to illumination variation compared to existing methods.

count=3
* Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.pdf)]
    * Title: Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Nicolae-Cătălin Ristea, Neelu Madan, Radu Tudor Ionescu, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B. Moeslund, Mubarak Shah
    * Abstract: Anomaly detection is commonly pursued as a one-class classification problem, where models can only learn from normal training samples, while being evaluated on both normal and abnormal test samples. Among the successful approaches for anomaly detection, a distinguished category of methods relies on predicting masked information (e.g. patches, future frames, etc.) and leveraging the reconstruction error with respect to the masked information as an abnormality score. Different from related methods, we propose to integrate the reconstruction-based functionality into a novel self-supervised predictive architectural building block. The proposed self-supervised block is generic and can easily be incorporated into various state-of-the-art anomaly detection methods. Our block starts with a convolutional layer with dilated filters, where the center area of the receptive field is masked. The resulting activation maps are passed through a channel attention module. Our block is equipped with a loss that minimizes the reconstruction error with respect to the masked area in the receptive field. We demonstrate the generality of our block by integrating it into several state-of-the-art frameworks for anomaly detection on image and video, providing empirical evidence that shows considerable performance improvements on MVTec AD, Avenue, and ShanghaiTech. We release our code as open source at: https://github.com/ristea/sspcab.

count=3
* Probability-Based Global Cross-Modal Upsampling for Pansharpening
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Probability-Based_Global_Cross-Modal_Upsampling_for_Pansharpening_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Probability-Based_Global_Cross-Modal_Upsampling_for_Pansharpening_CVPR_2023_paper.pdf)]
    * Title: Probability-Based Global Cross-Modal Upsampling for Pansharpening
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zeyu Zhu, Xiangyong Cao, Man Zhou, Junhao Huang, Deyu Meng
    * Abstract: Pansharpening is an essential preprocessing step for remote sensing image processing. Although deep learning (DL) approaches performed well on this task, current upsampling methods used in these approaches only utilize the local information of each pixel in the low-resolution multispectral (LRMS) image while neglecting to exploit its global information as well as the cross-modal information of the guiding panchromatic (PAN) image, which limits their performance improvement. To address this issue, this paper develops a novel probability-based global cross-modal upsampling (PGCU) method for pan-sharpening. Precisely, we first formulate the PGCU method from a probabilistic perspective and then design an efficient network module to implement it by fully utilizing the information mentioned above while simultaneously considering the channel specificity. The PGCU module consists of three blocks, i.e., information extraction (IE), distribution and expectation estimation (DEE), and fine adjustment (FA). Extensive experiments verify the superiority of the PGCU method compared with other popular upsampling methods. Additionally, experiments also show that the PGCU module can help improve the performance of existing SOTA deep learning pansharpening methods. The codes are available at https://github.com/Zeyu-Zhu/PGCU.

count=3
* SatSynth: Augmenting Image-Mask Pairs through Diffusion Models for Aerial Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Toker_SatSynth_Augmenting_Image-Mask_Pairs_through_Diffusion_Models_for_Aerial_Semantic_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Toker_SatSynth_Augmenting_Image-Mask_Pairs_through_Diffusion_Models_for_Aerial_Semantic_CVPR_2024_paper.pdf)]
    * Title: SatSynth: Augmenting Image-Mask Pairs through Diffusion Models for Aerial Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Aysim Toker, Marvin Eisenberger, Daniel Cremers, Laura Leal-Taixé
    * Abstract: In recent years semantic segmentation has become a pivotal tool in processing and interpreting satellite imagery. Yet a prevalent limitation of supervised learning techniques remains the need for extensive manual annotations by experts. In this work we explore the potential of generative image diffusion to address the scarcity of annotated data in earth observation tasks. The main idea is to learn the joint data manifold of images and labels leveraging recent advancements in denoising diffusion probabilistic models. To the best of our knowledge we are the first to generate both images and corresponding masks for satellite segmentation. We find that the obtained pairs not only display high quality in fine-scale features but also ensure a wide sampling diversity. Both aspects are crucial for earth observation data where semantic classes can vary severely in scale and occurrence frequency. We employ the novel data instances for downstream segmentation as a form of data augmentation. In our experiments we provide comparisons to prior works based on discriminative diffusion models or GANs. We demonstrate that integrating generated samples yields significant quantitative improvements for satellite semantic segmentation -- both compared to baselines and when training only on the original data.

count=3
* A Comparison of Stereo and Multiview 3-D Reconstruction Using Cross-Sensor Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/Ozcanli_A_Comparison_of_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/Ozcanli_A_Comparison_of_2015_CVPR_paper.pdf)]
    * Title: A Comparison of Stereo and Multiview 3-D Reconstruction Using Cross-Sensor Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Ozge C. Ozcanli, Yi Dong, Joseph L. Mundy, Helen Webb, Riad Hammoud, Victor Tom
    * Abstract: High-resolution and accurate Digital Elevation Model (DEM) generation from satellite imagery is a challenging problem. In this work, a stereo 3-D reconstruction framework is outlined that is applicable to nonstereoscopic satellite image pairs that may be captured by different satellites. The orthographic height maps given by stereo reconstruction are compared to height maps given by a multiview approach based on Probabilistic Volumetric Representation (PVR). Height map qualities are measured in comparison to manually prepared ground-truth height maps in three sites from different parts of the world with urban, semi-urban and rural features. The results along with strengths and weaknesses of the two techniques are summarized.

count=3
* A Model-Based Approach to Finding Tracks in SAR CCD Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/Quach_A_Model-Based_Approach_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/Quach_A_Model-Based_Approach_2015_CVPR_paper.pdf)]
    * Title: A Model-Based Approach to Finding Tracks in SAR CCD Images
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Tu-Thach Quach, Rebecca Malinas, Mark W. Koch
    * Abstract: Combining multiple synthetic aperture radar (SAR) images taken at different times of the same scene produces coherent change detection (CCD) images that can detect small surface changes such as tire tracks. The resulting CCD images can be used in an automated approach to identify and label tracks. Existing techniques have limited success due to the noisy nature of these CCD images. In particular, existing techniques require some user cues and can only trace a single track. This paper presents an approach to automatically identify and label multiple tracks in CCD images. We use an explicit objective function that utilizes the Bayesian information criterion to find the simplest set of curves that explains the observed data. Experimental results show that it is capable of identifying tracks under various scenes and can correctly declare when no tracks are present.

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
* Rescan: Inductive Instance Segmentation for Indoor RGBD Scans
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Halber_Rescan_Inductive_Instance_Segmentation_for_Indoor_RGBD_Scans_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Halber_Rescan_Inductive_Instance_Segmentation_for_Indoor_RGBD_Scans_ICCV_2019_paper.pdf)]
    * Title: Rescan: Inductive Instance Segmentation for Indoor RGBD Scans
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Maciej Halber,  Yifei Shi,  Kai Xu,  Thomas Funkhouser
    * Abstract: In depth-sensing applications ranging from home robotics to AR/VR, it will be common to acquire 3D scans of interior spaces repeatedly at sparse time intervals (e.g., as part of regular daily use). We propose an algorithm that analyzes these "rescans" to infer a temporal model of a scene with semantic instance information. Our algorithm operates inductively by using the temporal model resulting from past observations to infer an instance segmentation of a new scan, which is then used to update the temporal model. The model contains object instance associations across time and thus can be used to track individual objects, even though there are only sparse observations. During experiments with a new benchmark for the new task, our algorithm outperforms alternate approaches based on state-of-the-art networks for semantic instance segmentation.

count=3
* Get Better 1 Pixel PCK: Ladder Scales Correspondence Flow Networks for Remote Sensing Image Matching in Higher Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/html/Chen_Get_Better_1_Pixel_PCK_Ladder_Scales_Correspondence_Flow_Networks_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/papers/Chen_Get_Better_1_Pixel_PCK_Ladder_Scales_Correspondence_Flow_Networks_ICCVW_2021_paper.pdf)]
    * Title: Get Better 1 Pixel PCK: Ladder Scales Correspondence Flow Networks for Remote Sensing Image Matching in Higher Resolution
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Weitao Chen, Zhibin Wang, Hao Li
    * Abstract: Recently, remote sensing image matching by deep learning reaches competitive performance evaluated by Probability of Correct Keypoints(PCK). Percentage of image size is often used as the threshold of PCK. Even though it can achieve a good 1% PCK in high resolution by regression of transformer parameters,the value will be reduced by using the absolute 1 pixel as threshold in the higher resolution. Inspired by the flow-based methods used in natural image matching tasks, we convert the transformer to correspondence flow and propose ladder scales correspondence flow networks(LSCFN) to get better 1 pixel PCK in higher resolution.Input images are resized to multi scales and then sent to network backbone to generate multi feature pyramids. These pyramids are linked and effectively pull up the highest resolution of original backbone just like a ladder when the global correlation scale is fixed.LSCFN regress correspondence flow in ladder scales by a dense cascade way.We build LSCFN-b and LSCFN-s based on the degree of semantic change between compared images. One with only global correlation is used for the big change, another with global and local correlation is used for the opposite one.The proposed LSCFN achieve state-of-the-art performance evaluated by 1% of image size PCK and absolute 1 pixel PCK on google earth dataset.

count=3
* JanusNet: Detection of Moving Objects From UAV Platforms
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/html/Zhao_JanusNet_Detection_of_Moving_Objects_From_UAV_Platforms_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/papers/Zhao_JanusNet_Detection_of_Moving_Objects_From_UAV_Platforms_ICCVW_2021_paper.pdf)]
    * Title: JanusNet: Detection of Moving Objects From UAV Platforms
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yuxiang Zhao, Khurram Shafique, Zeeshan Rasheed, Maoxu Li
    * Abstract: In this paper, we present JanusNet, an efficient CNN model that can perform online background subtraction and robustly detect moving targets using resource-constrained computational hardware on-board unmanned aerial vehicles (UAVs). Most of the existing work on background subtraction either assume that the camera is stationary or make limiting assumptions about the motion of the camera, the structure of the scene under observation, or the apparent motion of the background in video. JanusNet does not have these limitations and therefore, is applicable to a variety of UAV applications. JanusNet learns to extract and combine motion and appearance features to separate background and foreground to generate accurate pixel-wise masks of the moving objects. The network is trained using a simulated video dataset (generated using Unreal Engine 4) with ground-truth labels. Results on UCF Aerial and Kaggle Drone videos datasets show that the learned model transfers well to real UAV videos and can robustly detect moving targets in a wide variety of scenarios. Moreover, experiments on CDNet dataset demonstrate that even without explicitly assuming that the camera is stationary, the performance of JanusNet is comparable to traditional background subtraction methods.

count=3
* Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/html/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/papers/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.pdf)]
    * Title: Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Csaba Beleznai, Peter Gemeiner, Christian Zinner
    * Abstract: Reliable and timely detection of abandoned items in public places still represents an unsolved problem for automated visual surveillance. Typical surveilled scenarios are associated with high visual ambiguity such as shadows, occlusions, illumination changes and substantial clutter consisting of a mixture of dynamic and stationary objects. Motivated by these challenges we propose a reliable left item detection approach based on the combination of intensity and depth data from a passive stereo setup. The employed in-house developed stereo system consists of low-cost sensors and it is capable to perform detection in environments of up to 10m x 10m in size. The proposed algorithm is tested on a set of indoor sequences and compared to manually annotated ground truth data. Obtained results show that many failure modes of intensity-based approaches are absent and even small-sized objects such as a handbag can be reliably detected when left behind in a scene. The presented results display a very promising approach, which can robustly detect left luggage in dynamic environments at a close to real-time computational speed.

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
* TKD: Temporal Knowledge Distillation for Active Perception
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Bajestani_TKD_Temporal_Knowledge_Distillation_for_Active_Perception_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Bajestani_TKD_Temporal_Knowledge_Distillation_for_Active_Perception_WACV_2020_paper.pdf)]
    * Title: TKD: Temporal Knowledge Distillation for Active Perception
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Mohammad Farhadi Bajestani,  Yezhou Yang
    * Abstract: Deep neural network-based methods have been proved to achieve outstanding performance on object detection and classification tasks. Despite the significant performance improvement using the deep structures, they still require prohibitive runtime to process images and maintain the highest possible performance for real-time applications. Observing the phenomenon that human visual system (HVS) relies heavily on the temporal dependencies among frames from the visual input to conduct recognition efficiently, we propose a novel framework dubbed as TKD: temporal knowledge distillation. This framework distills the temporal knowledge from a heavy neural network-based model over selected video frames (the perception of the moments) to a light-weight model. To enable the distillation, we put forward two novel procedures: 1) a Long-short Term Memory (LSTM)-based key frame selection method; and 2) a novel teacher-bounded loss design. To validate our approach, we conduct comprehensive empirical evaluations using different object detection methods over multiple datasets including Youtube Objects and Hollywood scene dataset. Our results show consistent improvement in accuracy-speed trade-offs for object detection over the frames of the dynamic scene, compared to other modern object recognition methods. It can maintain the desired accuracy with the throughput of around 220 images per second. Implementation: https://github.com/mfarhadi/TKD-Cloud.

count=3
* Automatic Open-World Reliability Assessment
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Jafarzadeh_Automatic_Open-World_Reliability_Assessment_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Jafarzadeh_Automatic_Open-World_Reliability_Assessment_WACV_2021_paper.pdf)]
    * Title: Automatic Open-World Reliability Assessment
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Mohsen Jafarzadeh, Touqeer Ahmad, Akshay Raj Dhamija, Chunchun Li, Steve Cruz, Terrance E. Boult
    * Abstract: Image classification in the open-world must handle out-of-distribution (OOD) images. Systems should ideally reject OOD images, or they will map atop of known classes and reduce reliability. Using open-set classifiers that can reject OOD inputs can help. However, optimal accuracy of open-set classifiers depend on the frequency of OOD data. Thus, for either standard or open-set classifiers, it is important to be able to determine when the world changes and increasing OOD inputs will result in reduced system reliability. However, during operations, we cannot directly assess accuracy as there are no labels. Thus, the reliability assessment of these classifiers must be done by human operators, made more complex because networks are not 100% accurate, so some failures are to be expected. To automate this process, herein, we formalize the open-world recognition reliability problem and propose multiple automatic reliability assessment policies to address this new problem using only the distribution of reported scores/probability data. The distributional algorithms can be applied to both classic classifiers with SoftMax as well as the open-world Extreme Value Machine (EVM) to provide automated reliability assessment. We show that all of the new algorithms significantly outperform detection using the mean of SoftMax.

count=3
* Multi-Frame Recurrent Adversarial Network for Moving Object Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Patil_Multi-Frame_Recurrent_Adversarial_Network_for_Moving_Object_Segmentation_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Patil_Multi-Frame_Recurrent_Adversarial_Network_for_Moving_Object_Segmentation_WACV_2021_paper.pdf)]
    * Title: Multi-Frame Recurrent Adversarial Network for Moving Object Segmentation
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Prashant W. Patil, Akshay Dudhane, Subrahmanyam Murala
    * Abstract: Moving object segmentation (MOS) in different practical scenarios like weather degraded, dynamic background, etc. videos is a challenging and high demanding task for various computer vision applications. Existing supervised approaches achieve remarkable performance with complicated training or extensive fine-tuning or inappropriate training-testing data distribution. Also, the generalized effect of existing works with completely unseen data is difficult to identify. In this work, the recurrent feature sharing based generative adversarial network is proposed with unseen video analysis. The proposed network comprises of dilated convolution to extract the spatial features at multiple scales. Along with the temporally sampled multiple frames, previous frame output is considered as input to the network. As the motion is very minute between the two consecutive frames, the previous frame decoder features are shared with encoder features recurrently for current frame foreground segmentation. This recurrent feature sharing of different layers helps the encoder network to learn the hierarchical interactions between the motion and appearance based features. Also, the learning of the proposed network is concentrated in different ways, like disjoint and global training-testing for MOS. An extensive experimental analysis of the proposed network is carried out on two benchmark video datasets with seen and unseen MOS video. Qualitative and quantitative experimental study shows that the proposed network outperforms the existing methods.

count=3
* Do Adaptive Active Attacks Pose Greater Risk Than Static Attacks?
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Drenkow_Do_Adaptive_Active_Attacks_Pose_Greater_Risk_Than_Static_Attacks_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Drenkow_Do_Adaptive_Active_Attacks_Pose_Greater_Risk_Than_Static_Attacks_WACV_2023_paper.pdf)]
    * Title: Do Adaptive Active Attacks Pose Greater Risk Than Static Attacks?
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Nathan Drenkow, Max Lennon, I-Jeng Wang, Philippe Burlina
    * Abstract: In contrast to perturbation-based attacks, patch-based attacks are physically realizable, and are therefore increasingly studied. However, prior work neglects the possibility of adaptive attacks optimized for 3D pose. For the first time, to our knowledge, we consider the challenge of designing and evaluating attacks on image sequences using 3D optimization along entire 3D kinematic trajectories. In this context, we study a type of dynamic attack, referred to as "adaptive active attacks" (AAA), that takes into consideration the pose of the observer being targeted. To better address the threat and risk posed by AAA attacks, we develop several novel risk-based and trajectory-based metrics. These are designed to capture the risk of attack success for attacking earlier in the trajectory to derail autonomous driving systems as well as tradeoffs that may arise given the possibility of additional detection. We evaluate performance of white-box targeted attacks using a subset of ImageNet classes, and demonstrate, in aggregate, that AAA attacks can pose threats beyond static attacks in kinematic settings in situations of predominantly looming motion (i.,e., a prevalent use case in automated vehicular navigation). Results demonstrate that AAA attacks can exhibit targeted attack success exceeding 10% in aggregate, and for some specific classes, up to 15% over their static counterparts. However, taking into consideration the probability of detection by the defender shows a more nuanced risk pattern. These new insights are important for guiding future adversarial machine learning studies and suggest researchers should consider defense against novel threats posed by dynamic attacks for full trajectories and videos.

count=3
* Effective Restoration of Source Knowledge in Continual Test Time Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Niloy_Effective_Restoration_of_Source_Knowledge_in_Continual_Test_Time_Adaptation_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Niloy_Effective_Restoration_of_Source_Knowledge_in_Continual_Test_Time_Adaptation_WACV_2024_paper.pdf)]
    * Title: Effective Restoration of Source Knowledge in Continual Test Time Adaptation
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Fahim Faisal Niloy, Sk Miraj Ahmed, Dripta S. Raychaudhuri, Samet Oymak, Amit K. Roy-Chowdhury
    * Abstract: Traditional test-time adaptation (TTA) methods face significant challenges in adapting to dynamic environments characterized by continuously changing long-term target distributions. These challenges primarily stem from two factors: catastrophic forgetting of previously learned valuable source knowledge and gradual error accumulation caused by miscalibrated pseudo labels. To address these issues, this paper introduces an unsupervised domain change detection method that is capable of identifying domain shifts in dynamic environments and subsequently resets the model parameters to the original source pre-trained values. By restoring the knowledge from the source, it effectively corrects the negative consequences arising from the gradual deterioration of model parameters caused by ongoing shifts in the domain. Our method involves progressive estimation of global batch-norm statistics specific to each domain, while keeping track of changes in the statistics triggered by domain shifts. Importantly, our method is agnostic to the specific adaptation technique employed and thus, can be incorporated to existing TTA methods to enhance their performance in dynamic environments. We perform extensive experiments on benchmark datasets to demonstrate the superior performance of our method compared to state-of-the-art adaptation methods.

count=3
* Defense Against Adversarial Cloud Attack on Remote Sensing Salient Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Sun_Defense_Against_Adversarial_Cloud_Attack_on_Remote_Sensing_Salient_Object_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Sun_Defense_Against_Adversarial_Cloud_Attack_on_Remote_Sensing_Salient_Object_WACV_2024_paper.pdf)]
    * Title: Defense Against Adversarial Cloud Attack on Remote Sensing Salient Object Detection
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Huiming Sun, Lan Fu, Jinlong Li, Qing Guo, Zibo Meng, Tianyun Zhang, Yuewei Lin, Hongkai Yu
    * Abstract: Detecting the salient objects in a remote sensing image has wide applications. Many existing deep learning methods have been proposed for Salient Object Detection (SOD) in remote sensing images with remarkable results. However, the recent adversarial attack examples, generated by changing a few pixel values on the original image, could result in a collapse for the well-trained deep learning model. Different with existing methods adding perturbation to original images, we propose to jointly tune adversarial exposure and additive perturbation for attack and constrain image close to cloudy image as Adversarial Cloud. Cloud is natural and common in remote sensing images, however, camouflaging cloud based adversarial attack and defense for remote sensing images are not well studied before. Furthermore, we design DefenseNet as a learnable pre-processing to the adversarial cloudy images to preserve the performance of the deep learning based remote sensing SOD model, without tuning the already deployed deep SOD model. By considering both regular and generalized adversarial examples, the proposed DefenseNet can defend the proposed Adversarial Cloud in white-box setting and other attack methods in black-box setting. Experimental results on a synthesized benchmark from the public remote sensing dataset (EORSSD) show the promising defense against adversarial cloud attacks.

count=3
* Dynamic Mode Decomposition with Reproducing Kernels for Koopman Spectral Analysis
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2016/hash/1728efbda81692282ba642aafd57be3a-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2016/file/1728efbda81692282ba642aafd57be3a-Paper.pdf)]
    * Title: Dynamic Mode Decomposition with Reproducing Kernels for Koopman Spectral Analysis
    * Publisher: NeurIPS
    * Publication Date: `2016`
    * Authors: Yoshinobu Kawahara
    * Abstract: A spectral analysis of the Koopman operator, which is an infinite dimensional linear operator on an observable, gives a (modal) description of the global behavior of a nonlinear dynamical system without any explicit prior knowledge of its governing equations. In this paper, we consider a spectral analysis of the Koopman operator in a reproducing kernel Hilbert space (RKHS). We propose a modal decomposition algorithm to perform the analysis using finite-length data sequences generated from a nonlinear system. The algorithm is in essence reduced to the calculation of a set of orthogonal bases for the Krylov matrix in RKHS and the eigendecomposition of the projection of the Koopman operator onto the subspace spanned by the bases. The algorithm returns a decomposition of the dynamics into a finite number of modes, and thus it can be thought of as a feature extraction procedure for a nonlinear dynamical system. Therefore, we further consider applications in machine learning using extracted features with the presented analysis. We illustrate the method on the applications using synthetic and real-world data.

count=3
* Inverse Filtering for Hidden Markov Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf)]
    * Title: Inverse Filtering for Hidden Markov Models
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Robert Mattila, Cristian Rojas, Vikram Krishnamurthy, Bo Wahlberg
    * Abstract: This paper considers a number of related inverse filtering problems for hidden Markov models (HMMs). In particular, given a sequence of state posteriors and the system dynamics; i) estimate the corresponding sequence of observations, ii) estimate the observation likelihoods, and iii) jointly estimate the observation likelihoods and the observation sequence. We show how to avoid a computationally expensive mixed integer linear program (MILP) by exploiting the algebraic structure of the HMM filter using simple linear algebra operations, and provide conditions for when the quantities can be uniquely reconstructed. We also propose a solution to the more general case where the posteriors are noisily observed. Finally, the proposed inverse filtering algorithms are evaluated on real-world polysomnographic data used for automatic sleep segmentation.

count=3
* Generalization of Reinforcement Learners with Working and Episodic Memory
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/02ed812220b0705fabb868ddbf17ea20-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/02ed812220b0705fabb868ddbf17ea20-Paper.pdf)]
    * Title: Generalization of Reinforcement Learners with Working and Episodic Memory
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Meire Fortunato, Melissa Tan, Ryan Faulkner, Steven Hansen, Adrià Puigdomènech Badia, Gavin Buttimore, Charles Deck, Joel Z. Leibo, Charles Blundell
    * Abstract: Memory is an important aspect of intelligence and plays a role in many deep reinforcement learning models. However, little progress has been made in understanding when specific memory systems help more than others and how well they generalize. The field also has yet to see a prevalent consistent and rigorous approach for evaluating agent performance on holdout data. In this paper, we aim to develop a comprehensive methodology to test different kinds of memory in an agent and assess how well the agent can apply what it learns in training to a holdout set that differs from the training set along dimensions that we suggest are relevant for evaluating memory-specific generalization. To that end, we first construct a diverse set of memory tasks that allow us to evaluate test-time generalization across multiple dimensions. Second, we develop and perform multiple ablations on an agent architecture that combines multiple memory systems, observe its baseline models, and investigate its performance against the task suite.

count=3
* Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/466accbac9a66b805ba50e42ad715740-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/466accbac9a66b805ba50e42ad715740-Paper.pdf)]
    * Title: Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Vincent LE GUEN, Nicolas THOME
    * Abstract: This paper addresses the problem of time series forecasting for non-stationary signals and multiple future steps prediction. To handle this challenging task, we introduce DILATE (DIstortion Loss including shApe and TimE), a new objective function for training deep neural networks. DILATE aims at accurately predicting sudden changes, and explicitly incorporates two terms supporting precise shape and temporal change detection. We introduce a differentiable loss function suitable for training deep neural nets, and provide a custom back-prop implementation for speeding up optimization. We also introduce a variant of DILATE, which provides a smooth generalization of temporally-constrained Dynamic TimeWarping (DTW). Experiments carried out on various non-stationary datasets reveal the very good behaviour of DILATE compared to models trained with the standard Mean Squared Error (MSE) loss function, and also to DTW and variants. DILATE is also agnostic to the choice of the model, and we highlight its benefit for training fully connected networks as well as specialized recurrent architectures, showing its capacity to improve over state-of-the-art trajectory forecasting approaches.

count=3
* DiSC: Differential Spectral Clustering of Features
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/a84953147312ea2e8b020e53a267321b-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/a84953147312ea2e8b020e53a267321b-Paper-Conference.pdf)]
    * Title: DiSC: Differential Spectral Clustering of Features
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Ram Dyuthi Sristi, Gal Mishne, Ariel Jaffe
    * Abstract: Selecting subsets of features that differentiate between two conditions is a key task in a broad range of scientific domains. In many applications, the features of interest form clusters with similar effects on the data at hand. To recover such clusters we develop DiSC, a data-driven approach for detecting groups of features that differentiate between conditions. For each condition, we construct a graph whose nodes correspond to the features and whose weights are functions of the similarity between them for that condition. We then apply a spectral approach to compute subsets of nodes whose connectivity pattern differs significantly between the condition-specific feature graphs. On the theoretical front, we analyze our approach with a toy example based on the stochastic block model. We evaluate DiSC on a variety of datasets, including MNIST, hyperspectral imaging, simulated scRNA-seq and task fMRI, and demonstrate that DiSC uncovers features that better differentiate between conditions compared to competing methods.

count=2
* OneDiff: A Generalist Model for Image Difference Captioning
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Hu_OneDiff_A_Generalist_Model_for_Image_Difference_Captioning_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Hu_OneDiff_A_Generalist_Model_for_Image_Difference_Captioning_ACCV_2024_paper.pdf)]
    * Title: OneDiff: A Generalist Model for Image Difference Captioning
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Erdong Hu, Longteng Guo, Tongtian Yue, Zijia Zhao, Shuning Xue, Jing Liu
    * Abstract: In computer vision, Image Difference Captioning (IDC) is crucial for accurately describing variations between closely related images. Traditional IDC methods often rely on specialist models, which restrict their applicability across varied contexts. This paper introduces the OneDiff model, a novel generalist approach that utilizes a robust vision-language model architecture, integrating a siamese image encoder with a Visual Delta Module. This innovative configuration allows for the precise detection and articulation of fine-grained differences between image pairs. OneDiff is trained through a dual-phase strategy, encompassing Coupled Sample Training and multi-task learning across a diverse array of data types, supported by our newly developed DiffCap Dataset. This dataset merges real-world and synthetic data, enhancing the training process and bolstering the models robustness. Extensive testing on diverse IDC benchmarks, such as Spot-the-Diff, Image-Editing-Request, and Birds-to-Words, shows that OneDiff consistently outperforms existing state-of-the-art models in accuracy and adaptability, achieving improvements of up to 97% CIDEr points in average. By setting a new benchmark in IDC, OneDiff paves the way for more versatile and effective applications in detecting and describing visual differences. The code, models, and data will be made publicly available.

count=2
* Statistical Inference Models for Image Datasets With Systematic Variations
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Kim_Statistical_Inference_Models_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Kim_Statistical_Inference_Models_2015_CVPR_paper.pdf)]
    * Title: Statistical Inference Models for Image Datasets With Systematic Variations
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Won Hwa Kim, Barbara B. Bendlin, Moo K. Chung, Sterling C. Johnson, Vikas Singh
    * Abstract: Statistical analysis of longitudinal or cross sectionalbrain imaging data to identify effects of neurodegenerative diseases is a fundamental task in various studies in neuroscience. However, when there are systematic variations in the images due to parameters changes such as changes in the scanner protocol, hardware changes, or when combining data from multi-site studies, the statistical analysis becomes problematic. Motivated by this scenario, the goal of this paper is to develop a unified statistical solution to the problem of systematic variations in statistical image analysis. Based in part on recent literature in harmonic analysis on diffusion maps, we propose an algorithm which compares operators that are resilient to the systematic variations described above. These operators are derived from the empirical measurements of the image data and provide an efficient surrogate to capturing the actual changes across images. We also establish a connection between our method to the design of Wavelets in non-Euclidean space. To evaluate the proposed ideas, we present various experimental results on detecting changes in simulations as well as show how the method offers improved statistical power in the analysis of longitudinal real PIB-PET imaging data acquired from participants at risk for Alzheimer's disease(AD).

count=2
* Rolling Shutter Motion Deblurring
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Su_Rolling_Shutter_Motion_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Su_Rolling_Shutter_Motion_2015_CVPR_paper.pdf)]
    * Title: Rolling Shutter Motion Deblurring
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Shuochen Su, Wolfgang Heidrich
    * Abstract: Although motion blur and rolling shutter deformations are closely coupled artifacts in images taken with CMOS image sensors, the two phenomena have so far mostly been treated separately, with deblurring algorithms being unable to handle rolling shutter wobble, and rolling shutter algorithms being incapable of dealing with motion blur. We propose an approach that delivers sharp and undistorted output given a single rolling shutter motion blurred image. The key to achieving this is a global modeling of the camera motion trajectory, which enables each scanline of the image to be deblurred with the corresponding motion segment. We show the results of the proposed framework through experiments on synthetic and real data.

count=2
* From Bows to Arrows: Rolling Shutter Rectification of Urban Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Rengarajan_From_Bows_to_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Rengarajan_From_Bows_to_CVPR_2016_paper.pdf)]
    * Title: From Bows to Arrows: Rolling Shutter Rectification of Urban Scenes
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Vijay Rengarajan, Ambasamudram N. Rajagopalan, Rangarajan Aravind
    * Abstract: The rule of perspectivity that 'straight-lines-must-remain-straight' is easily inflected in CMOS cameras by distortions introduced by motion. Lines can be rendered as curves due to the row-wise exposure mechanism known as rolling shutter (RS). We solve the problem of correcting distortions arising from handheld cameras due to RS effect from a single image free from motion blur with special relevance to urban scenes. We develop a procedure to extract prominent curves from the RS image since this is essential for deciphering the varying row-wise motion. We pose an optimization problem with line desirability costs based on straightness, angle, and length, to resolve the geometric ambiguities while estimating the camera motion based on a rotation-only model assuming known camera intrinsic matrix. Finally, we rectify the RS image based on the estimated camera trajectory using inverse mapping. We show rectification results for RS images captured using mobile phone cameras. We also compare our single image method against existing video and nonblind RS rectification methods that typically require multiple images.

count=2
* Embedded Motion Detection via Neural Response Mixture Background Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/html/Shafiee_Embedded_Motion_Detection_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/papers/Shafiee_Embedded_Motion_Detection_CVPR_2016_paper.pdf)]
    * Title: Embedded Motion Detection via Neural Response Mixture Background Modeling
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Mohammad Javad Shafiee, Parthipan Siva, Paul Fieguth, Alexander Wong
    * Abstract: Recent studies have shown that deep neural networks (DNNs) can outperform state-of-the-art for a multitude of computer vision tasks. However, the ability to leverage DNNs for near real-time performance on embedded systems have been all but impossible so far without requiring specialized processors or GPUs. In this paper, we present a new motion detection algorithm that leverages the power of DNNs while maintaining low computational complexity needed for near real-time embedded performance without specialized hardware. The proposed Neural Response Mixture (NeRM) model leverages rich deep features extracted from the neural responses of an efficient, stochastically-formed deep neural network for constructing Gaussian mixture models to detect moving objects in a scene. NeRM was implemented on an embedded system on an Axis surveillance camera, and results demonstrated that the proposed NeRM approach can strong motion detection accuracy while operating at near real-time performance.

count=2
* Fast Image Gradients Using Binary Feature Convolutions
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/St-Charles_Fast_Image_Gradients_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/St-Charles_Fast_Image_Gradients_CVPR_2016_paper.pdf)]
    * Title: Fast Image Gradients Using Binary Feature Convolutions
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: The recent increase in popularity of binary feature descriptors has opened the door to new lightweight computer vision applications. Most research efforts thus far have been dedicated to the introduction of new large-scale binary features, which are primarily used for keypoint description and matching. In this paper, we show that the side products of small-scale binary feature computations can efficiently filter images and estimate image gradients. The improved efficiency of low-level operations can be especially useful in time-constrained applications. Through our experiments, we show that efficient binary feature convolutions can be used to mimic various image processing operations, and even outperform Sobel gradient estimation in the edge detection problem, both in terms of speed and F-Measure.

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
* Detecting Anomalous Objects on Mobile Platforms
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.pdf)]
    * Title: Detecting Anomalous Objects on Mobile Platforms
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Wallace Lawson, Laura Hiatt, Keith Sullivan
    * Abstract: We present an approach where a robot patrols a fixed path through an environment, autonomously locating suspicious or anomalous objects. To learn, the robot patrols this environment building a dictionary describing what is present. The dictionary is built by clustering features from a deep neural network. The objects present vary depending on the scene, which means that an object that is anomalous in one scene may be completely normal in another. To reason about this, the robot uses a computational cognitive model to learn the dictionary elements that are typically found in each scene. Once the dictionary and model has been built, the robot can patrol the environment matching objects against the dictionary, and querying the model to find the most likely objects present and to determine which objects (if any) are anomalous. We demonstrate our approach by patrolling two indoor and one outdoor environments.

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
* Minimum Delay Moving Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Lao_Minimum_Delay_Moving_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lao_Minimum_Delay_Moving_CVPR_2017_paper.pdf)]
    * Title: Minimum Delay Moving Object Detection
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Dong Lao, Ganesh Sundaramoorthi
    * Abstract: We present a general framework and method for detection of an object in a video based on apparent motion. The object moves relative to background motion at some unknown time in the video, and the goal is to detect and segment the object as soon it moves in an online manner. Due to unreliability of motion between frames, more than two frames are needed to reliably detect the object. Our method is designed to detect the object(s) with minimum delay, i.e., frames after the object moves, constraining the false alarms. Experiments on a new extensive dataset for moving object detection show that our method achieves less delay for all false alarm constraints than existing state-of-the-art.

count=2
* Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Makihara_Joint_Intensity_and_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Makihara_Joint_Intensity_and_CVPR_2017_paper.pdf)]
    * Title: Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yasushi Makihara, Atsuyuki Suzuki, Daigo Muramatsu, Xiang Li, Yasushi Yagi
    * Abstract: This paper describes a joint intensity metric learning method to improve the robustness of gait recognition with silhouette-based descriptors such as gait energy images. Because existing methods often use the difference of image intensities between a matching pair (e.g., the absolute difference of gait energies for the l_1-norm) to measure a dissimilarity, large intrasubject differences derived from covariate conditions (e.g., large gait energies caused by carried objects vs. small gait energies caused by the background), may wash out subtle intersubject differences (e.g., the difference of middle-level gait energies derived from motion differences). We therefore introduce a metric on joint intensity to mitigate the large intrasubject differences as well as leverage the subtle intersubject differences. More specifically, we formulate the joint intensity and spatial metric learning in a unified framework and alternately optimize it by linear or ranking support vector machines. Experiments using the OU-ISIR treadmill data set B with the largest clothing variation and large population data set with bag, b version containing carrying status in the wild demonstrate the effectiveness of the proposed method.

count=2
* Spatio-Temporal Alignment of Non-Overlapping Sequences From Independently Panning Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Safdarnejad_Spatio-Temporal_Alignment_of_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Safdarnejad_Spatio-Temporal_Alignment_of_CVPR_2017_paper.pdf)]
    * Title: Spatio-Temporal Alignment of Non-Overlapping Sequences From Independently Panning Cameras
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Seyed Morteza Safdarnejad, Xiaoming Liu
    * Abstract: This paper addresses the problem of spatio-temporal alignment of multiple video sequences. We identify and tackle a novel scenario of this problem referred to as Nonoverlapping Sequences (NOS). NOS are captured by multiple freely panning handheld cameras whose field of views (FOV) might have no direct spatial overlap. With the popularity of mobile sensors, NOS rise when multiple cooperative users capture a public event to create a panoramic video, or when consolidating multiple footages of an incident into a single video. To tackle this novel scenario, we first spatially align the sequences by reconstructing the background of each sequence and registering these backgrounds, even if the backgrounds are not overlapping. Given the spatial alignment, we temporally synchronize the sequences, such that the trajectories of moving objects (e.g., cars or pedestrians) are consistent across sequences. Experimental results demonstrate the performance of our algorithm in this novel and challenging scenario, quantitatively and qualitatively.

count=2
* Predicting Ground-Level Scene Layout From Aerial Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhai_Predicting_Ground-Level_Scene_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhai_Predicting_Ground-Level_Scene_CVPR_2017_paper.pdf)]
    * Title: Predicting Ground-Level Scene Layout From Aerial Imagery
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Menghua Zhai, Zachary Bessinger, Scott Workman, Nathan Jacobs
    * Abstract: We introduce a novel strategy for learning to extract semantically meaningful features from aerial imagery. Instead of manually labeling the aerial imagery, we propose to predict (noisy) semantic features automatically extracted from co-located ground imagery. Our network architecture takes an aerial image as input, extracts features using a convolutional neural network, and then applies an adaptive transformation to map these features into the ground-level perspective. We use an end-to-end learning approach to minimize the difference between the semantic segmentation extracted directly from the ground image and the semantic segmentation predicted solely based on the aerial image. We show that a model learned using this strategy, with no additional training, is already capable of rough semantic labeling of aerial imagery. Furthermore, we demonstrate that by finetuning this model we can achieve more accurate semantic segmentation than two baseline initialization strategies. We use our network to address the task of estimating the geolocation and geo-orientation of a ground image. Finally, we show how features extracted from an aerial image can be used to hallucinate a plausible ground-level panorama.

count=2
* Object State Recognition for Automatic AR-Based Maintenance Guidance
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w14/html/Dvorak_Object_State_Recognition_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w14/papers/Dvorak_Object_State_Recognition_CVPR_2017_paper.pdf)]
    * Title: Object State Recognition for Automatic AR-Based Maintenance Guidance
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Pavel Dvorak, Radovan Josth, Elisabetta Delponte
    * Abstract: This paper describes a component of an Augmented Reality (AR) based system focused on supporting workers in manufacturing and maintenance industry. Particularly, it describes a component responsible for verification of performed steps. Correct handling is crucial in both manufacturing and maintenance industries and deviations may cause problems in later stages of the production and assembly. The primary aim of such support systems is making the training of new employees faster and more efficient and reducing the error rate. We present a method for automatically recognizing an object's state with the objective of verifying a set of tasks performed by a user. The novelty of our approach is that the system can automatically recognize the state of the object and provide immediate feedback to the operator using an AR visualization enabling fully automatic step-by-step instructions.

count=2
* Joint Learning From Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Audebert_Joint_Learning_From_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Audebert_Joint_Learning_From_CVPR_2017_paper.pdf)]
    * Title: Joint Learning From Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Nicolas Audebert, Bertrand Le Saux, Sebastien Lefevre
    * Abstract: We investigate the use of OSM data for semantic labeling of EO images. Deep neural networks have been used in the past for remote sensing data classification from various sensors, including multispectral, hyperspectral, Radar and Lidar data. However, OSM is an abundant data source that has already been used as ground truth data, but rarely exploited as an input information layer. We study different use cases and deep network architectures to leverage this OSM data for semantic labeling of aerial and satellite images. Especially, we look into fusion based architectures and coarse-to-fine segmentation to include the OSM layer into multispectral-based deep fully convolutional networks. We illustrate how these methods can be used successfully on two public datasets: the ISPRS Potsdam and the DFC2017. We show that OSM data can efficiently be integrated into the vision-based deep learning models and that it significantly improves both the accuracy performance and the convergence.

count=2
* A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Lin_A_Prior-Less_Method_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lin_A_Prior-Less_Method_CVPR_2018_paper.pdf)]
    * Title: A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Chung-Ching Lin, Ying Hung
    * Abstract: This paper presents a prior-less method for tracking and clustering an unknown number of human faces and maintaining their individual identities in unconstrained videos. The key challenge is to accurately track faces with partial occlusion and drastic appearance changes in multiple shots resulting from significant variations of makeup, facial expression, head pose and illumination. To address this challenge, we propose a new multi-face tracking and re-identification algorithm, which provides high accuracy in face association in the entire video with automatic cluster number generation, and is robust to outliers. We develop a co-occurrence model of multiple body parts to seamlessly create face tracklets, and recursively link tracklets to construct a graph for extracting clusters. A Gaussian Process model is introduced to compensate the deep feature insufficiency, and is further used to refine the linking results. The advantages of the proposed algorithm are demonstrated using a variety of challenging music videos and newly introduced body-worn camera videos. The proposed method obtains significant improvements over the state of the art [51], while relying less on handling video-specific prior information to achieve high performance.

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
* Clothing Change Aware Person Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w41/html/Xue_Clothing_Change_Aware_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w41/Xue_Clothing_Change_Aware_CVPR_2018_paper.pdf)]
    * Title: Clothing Change Aware Person Identification
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Jia Xue, Zibo Meng, Karthik Katipally, Haibo Wang, Kees van Zon
    * Abstract: We develop a person identification approach - Clothing Change Aware Network (CCAN) for the task of clothing assisted person identification. CCAN concerns approaches that go beyond face recognition and particularly tackles the role of clothing to identification. Person identification is a rather challenging task when clothing appears changed under complex background information. With a pair of two person images as input, CCAN simultaneously performs a verification task to detect change in clothing and an identification task to predict person identity. When clothing from the pair of input images is detected to be different, CCAN automatically understates clothing information while emphasizing face, and vice versa. In practice, CCAN outperforms the way of equally stacking face and full body context features, and shows leading results on the People in Photos Album (PIPA) dataset.

count=2
* Crowd Activity Change Point Detection in Videos via Graph Stream Mining
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w6/html/Yang_Crowd_Activity_Change_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Yang_Crowd_Activity_Change_CVPR_2018_paper.pdf)]
    * Title: Crowd Activity Change Point Detection in Videos via Graph Stream Mining
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Meng Yang, Lida Rashidi, Sutharshan Rajasegarar, Christopher Leckie, Aravinda S. Rao, Marimuthu Palaniswami
    * Abstract: In recent years, there has been a growing interest in detecting anomalous behavioral patterns in video. In this work, we address this task by proposing a novel activity change point detection method to identify crowd movement anomalies for video surveillance. In our proposed novel framework, a hyperspherical clustering algorithm is utilized for the automatic identification of interesting regions, then the density of pedestrian flows between every pair of interesting regions over consecutive time intervals is monitored and represented as a sequence of adjacency matrices where the direction and density of flows are captured through a directed graph. Finally, we use graph edit distance as well as a cumulative sum test to detect change points in the graph sequence. We conduct experiments on four real-world video datasets: Dublin, New Orleans, Abbey Road and MCG Datasets. We observe that our proposed approach achieves a high F-measure, i.e., in the range [0.7, 1], for these datasets. The evaluation reveals that our proposed method can successfully detect the change points in all datasets at both global and local levels. Our results also demonstrate the efficiency and effectiveness of our proposed algorithm for change point detection and segmentation tasks.

count=2
* Event Probability Mask (EPM) and Event Denoising Convolutional Neural Network (EDnCNN) for Neuromorphic Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Baldwin_Event_Probability_Mask_EPM_and_Event_Denoising_Convolutional_Neural_Network_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Baldwin_Event_Probability_Mask_EPM_and_Event_Denoising_Convolutional_Neural_Network_CVPR_2020_paper.pdf)]
    * Title: Event Probability Mask (EPM) and Event Denoising Convolutional Neural Network (EDnCNN) for Neuromorphic Cameras
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: R. Wes Baldwin,  Mohammed Almatrafi,  Vijayan Asari,  Keigo Hirakawa
    * Abstract: This paper presents a novel method for labeling real-world neuromorphic camera sensor data by calculating the likelihood of generating an event at each pixel within a short time window, which we refer to as "event probability mask" or EPM. Its applications include (i) objective benchmarking of event denoising performance, (ii) training convolutional neural networks for noise removal called "event denoising convolutional neural network" (EDnCNN), and (iii) estimating internal neuromorphic camera parameters. We provide the first dataset (DVSNOISE20) of real-world labeled neuromorphic camera events for noise removal.

count=2
* Learning Geocentric Object Pose in Oblique Monocular Images
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Christie_Learning_Geocentric_Object_Pose_in_Oblique_Monocular_Images_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Christie_Learning_Geocentric_Object_Pose_in_Oblique_Monocular_Images_CVPR_2020_paper.pdf)]
    * Title: Learning Geocentric Object Pose in Oblique Monocular Images
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Gordon Christie,  Rodrigo Rene Rai Munoz Abujder,  Kevin Foster,  Shea Hagstrom,  Gregory D. Hager,  Myron Z. Brown
    * Abstract: An object's geocentric pose, defined as the height above ground and orientation with respect to gravity, is a powerful representation of real-world structure for object detection, segmentation, and localization tasks using RGBD images. For close-range vision tasks, height and orientation have been derived directly from stereo-computed depth and more recently from monocular depth predicted by deep networks. For long-range vision tasks such as Earth observation, depth cannot be reliably estimated with monocular images. Inspired by recent work in monocular height above ground prediction and optical flow prediction from static images, we develop an encoding of geocentric pose to address this challenge and train a deep network to compute the representation densely, supervised by publicly available airborne lidar. We exploit these attributes to rectify oblique images and remove observed object parallax to dramatically improve the accuracy of localization and to enable accurate alignment of multiple images taken from very different oblique viewpoints. We demonstrate the value of our approach by extending two large-scale public datasets for semantic segmentation in oblique satellite images. All of our data and code are publicly available.

count=2
* Joint Filtering of Intensity Images and Neuromorphic Events for High-Resolution Noise-Robust Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Joint_Filtering_of_Intensity_Images_and_Neuromorphic_Events_for_High-Resolution_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Joint_Filtering_of_Intensity_Images_and_Neuromorphic_Events_for_High-Resolution_CVPR_2020_paper.pdf)]
    * Title: Joint Filtering of Intensity Images and Neuromorphic Events for High-Resolution Noise-Robust Imaging
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zihao W. Wang,  Peiqi Duan,  Oliver Cossairt,  Aggelos Katsaggelos,  Tiejun Huang,  Boxin Shi
    * Abstract: We present a novel computational imaging system with high resolution and low noise. Our system consists of a traditional video camera which captures high-resolution intensity images, and an event camera which encodes high-speed motion as a stream of asynchronous binary events. To process the hybrid input, we propose a unifying framework that first bridges the two sensing modalities via a noise-robust motion compensation model, and then performs joint image filtering. The filtered output represents the temporal gradient of the captured space-time volume, which can be viewed as motion-compensated event frames with high resolution and low noise. Therefore, the output can be widely applied to many existing event-based algorithms that are highly dependent on spatial resolution and noise robustness. In experimental results performed on both publicly available datasets as well as our contributing RGB-DAVIS dataset, we show systematic performance improvement in applications such as high frame-rate video synthesis, feature/corner detection and tracking, as well as high dynamic range image reconstruction.

count=2
* An Efficient Approach for Anomaly Detection in Traffic Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AICity/html/Doshi_An_Efficient_Approach_for_Anomaly_Detection_in_Traffic_Videos_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AICity/papers/Doshi_An_Efficient_Approach_for_Anomaly_Detection_in_Traffic_Videos_CVPRW_2021_paper.pdf)]
    * Title: An Efficient Approach for Anomaly Detection in Traffic Videos
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Keval Doshi, Yasin Yilmaz
    * Abstract: Due to its relevance in intelligent transportation systems, anomaly detection in traffic videos has recently received much interest. It remains a difficult problem due to a variety of factors influencing the video quality of a real-time traffic feed, such as temperature, perspective, lighting conditions, and so on. Even though state-of-the-art methods perform well on the available benchmark datasets, they need a large amount of external training data as well as substantial computational resources. In this paper, we propose an efficient approach for a video anomaly detection system which is capable of running at the edge devices, e.g., on a roadside camera. The proposed approach comprises a pre-processing module that detects changes in the scene and removes the corrupted frames, a two-stage background modelling module and a two-stage object detector. Finally, a backtracking anomaly detection algorithm computes a similarity statistic and decides on the onset time of the anomaly. We also propose a sequential change detection algorithm that can quickly adapt to a new scene and detect changes in the similarity statistic. Experimental results on the Track 4 test set of the 2021 AI City Challenge show the efficacy of the proposed framework as we achieve an F1-score of 0.9157 along with 8.4027 root mean square error (RMSE) and are ranked fourth in the competition.

count=2
* Single View Geocentric Pose in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Christie_Single_View_Geocentric_Pose_in_the_Wild_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Christie_Single_View_Geocentric_Pose_in_the_Wild_CVPRW_2021_paper.pdf)]
    * Title: Single View Geocentric Pose in the Wild
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Gordon Christie, Kevin Foster, Shea Hagstrom, Gregory D. Hager, Myron Z. Brown
    * Abstract: Current methods for Earth observation tasks such as semantic mapping, map alignment, and change detection rely on near-nadir images; however, often the first available images in response to dynamic world events such as natural disasters are oblique. These tasks are much more difficult for oblique images due to observed object parallax. There has been recent success in learning to regress an object's geocentric pose, defined as height above ground and orientation with respect to gravity, by training with airborne lidar registered to satellite images. We present a model for this novel task that exploits affine invariance properties to outperform state of the art performance by a wide margin. We also address practical issues required to deploy this method in the wild for real-world applications. Our data and code are publicly available.

count=2
* Shadow Neural Radiance Fields for Multi-View Satellite Photogrammetry
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Derksen_Shadow_Neural_Radiance_Fields_for_Multi-View_Satellite_Photogrammetry_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Derksen_Shadow_Neural_Radiance_Fields_for_Multi-View_Satellite_Photogrammetry_CVPRW_2021_paper.pdf)]
    * Title: Shadow Neural Radiance Fields for Multi-View Satellite Photogrammetry
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Dawa Derksen, Dario Izzo
    * Abstract: We present a new generic method for shadow-aware multi-view satellite photogrammetry of Earth Observation scenes. Our proposed method, the Shadow Neural Radiance Field (S-NeRF) follows recent advances in implicit volumetric representation learning. For each scene, we train S-NeRF using very high spatial resolution optical images taken from known viewing angles. The learning requires no labels or shape priors: it is self-supervised by an image reconstruction loss. To accommodate for changing light source conditions both from a directional light source (the Sun) and a diffuse light source (the sky), we extend the NeRF approach in two ways. First, direct illumination from the Sun is modeled via a local light source visibility field. Second, indirect illumination from a diffuse light source is learned as a non-local color field as a function of the position of the Sun. Quantitatively, the combination of these factors reduces the altitude and color errors in shaded areas, compared to NeRF. The S-NeRF methodology not only performs novel view synthesis and full 3D shape estimation, it also enables shadow detection, albedo synthesis, and transient object filtering, without any explicit shape supervision.

count=2
* (ASNA) An Attention-Based Siamese-Difference Neural Network With Surrogate Ranking Loss Function for Perceptual Image Quality Assessment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Ayyoubzadeh_ASNA_An_Attention-Based_Siamese-Difference_Neural_Network_With_Surrogate_Ranking_Loss_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Ayyoubzadeh_ASNA_An_Attention-Based_Siamese-Difference_Neural_Network_With_Surrogate_Ranking_Loss_CVPRW_2021_paper.pdf)]
    * Title: (ASNA) An Attention-Based Siamese-Difference Neural Network With Surrogate Ranking Loss Function for Perceptual Image Quality Assessment
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Seyed Mehdi Ayyoubzadeh, Ali Royat
    * Abstract: Recently, deep convolutional neural networks (DCNN) that leverage the adversarial training framework for image restoration and enhancement have significantly improved the processed images' sharpness. Surprisingly, although these DCNNs produced crispier images than other methods visually, they may get a lower quality score when popular measures are employed for evaluating them. Therefore it is necessary to develop a quantitative metric to reflect their performances, which is well-aligned with the perceived quality of an image. Famous quantitative metrics such as Peak signal-to-noise ratio (PSNR), The structural similarity index measure (SSIM), and Perceptual Index (PI) are not well-correlated with the mean opinion score (MOS) for an image, especially for the neural networks trained with adversarial loss functions. This paper has proposed a convolutional neural network using an extension architecture of the traditional Siamese network so-called Siamese-Difference neural network. We have equipped this architecture with the spatial and channel-wise attention mechanism to increase our method's performance. Finally, we employed an auxiliary loss function to train our model. The suggested additional cost function surrogates ranking loss to increase Spearman's rank correlation coefficient while it is differentiable concerning the neural network parameters. Our method achieved superior performance in NTIRE 2021 Perceptual Image Quality Assessment Challenge. The implementations of our proposed method are publicly available.

count=2
* HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Bandara_HyperTransformer_A_Textural_and_Spectral_Feature_Fusion_Transformer_for_Pansharpening_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bandara_HyperTransformer_A_Textural_and_Spectral_Feature_Fusion_Transformer_for_Pansharpening_CVPR_2022_paper.pdf)]
    * Title: HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Wele Gedara Chaminda Bandara, Vishal M. Patel
    * Abstract: Pansharpening aims to fuse a registered high-resolution panchromatic image (PAN) with a low-resolution hyperspectral image (LR-HSI) to generate an enhanced HSI with high spectral and spatial resolution. Existing pansharpening approaches neglect using an attention mechanism to transfer HR texture features from PAN to LR-HSI features, resulting in spatial and spectral distortions. In this paper, we present a novel attention mechanism for pansharpening called HyperTransformer, in which features of LR-HSI and PAN are formulated as queries and keys in a transformer, respectively. HyperTransformer consists of three main modules, namely two separate feature extractors for PAN and HSI, a multi-head feature soft attention module, and a spatial-spectral feature fusion module. Such a network improves both spatial and spectral quality measures of the pansharpened HSI by learning cross-feature space dependencies and long-range details of PAN and LR-HSI. Furthermore, HyperTransformer can be utilized across multiple spatial scales at the backbone for obtaining improved performance. Extensive experiments conducted on three widely used datasets demonstrate that HyperTransformer achieves significant improvement over the state-of-the-art methods on both spatial and spectral quality measures. Implementation code and pre-trained weights can be accessed at https://github.com/wgcban/HyperTransformer.

count=2
* Deep Anomaly Discovery From Unlabeled Videos via Normality Advantage and Self-Paced Refinement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.pdf)]
    * Title: Deep Anomaly Discovery From Unlabeled Videos via Normality Advantage and Self-Paced Refinement
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Guang Yu, Siqi Wang, Zhiping Cai, Xinwang Liu, Chuanfu Xu, Chengkun Wu
    * Abstract: While classic video anomaly detection (VAD) requires labeled normal videos for training, emerging unsupervised VAD (UVAD) aims to discover anomalies directly from fully unlabeled videos. However, existing UVAD methods still rely on shallow models to perform detection or initialization, and they are evidently inferior to classic VAD methods. This paper proposes a full deep neural network (DNN) based solution that can realize highly effective UVAD. First, we, for the first time, point out that deep reconstruction can be surprisingly effective for UVAD, which inspires us to unveil a property named "normality advantage", i.e., normal events will enjoy lower reconstruction loss when DNN learns to reconstruct unlabeled videos. With this property, we propose Localization based Reconstruction (LBR) as a strong UVAD baseline and a solid foundation of our solution. Second, we propose a novel self-paced refinement (SPR) scheme, which is synthesized into LBR to conduct UVAD. Unlike ordinary self-paced learning that injects more samples in an easy-to-hard manner, the proposed SPR scheme gradually drops samples so that suspicious anomalies can be removed from the learning process. In this way, SPR consolidates normality advantage and enables better UVAD in a more proactive way. Finally, we further design a variant solution that explicitly takes the motion cues into account. The solution evidently enhances the UVAD performance, and it sometimes even surpasses the best classic VAD methods. Experiments show that our solution not only significantly outperforms existing UVAD methods by a wide margin (5% to 9% AUROC), but also enables UVAD to catch up with the mainstream performance of classic VAD.

count=2
* Cross-Dataset Learning for Generalizable Land Use Scene Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Gominski_Cross-Dataset_Learning_for_Generalizable_Land_Use_Scene_Classification_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Gominski_Cross-Dataset_Learning_for_Generalizable_Land_Use_Scene_Classification_CVPRW_2022_paper.pdf)]
    * Title: Cross-Dataset Learning for Generalizable Land Use Scene Classification
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Dimitri Gominski, Valérie Gouet-Brunet, Liming Chen
    * Abstract: Few-shot and cross-domain land use scene classification methods propose solutions to classify unseen classes or unseen visual distributions, but are hardly applicable to real-world situations due to restrictive assumptions. Few-shot methods involve episodic training on restrictive training subsets with small feature extractors, while cross-domain methods are only applied to common classes. The underlying challenge remains open: can we accurately classify new scenes on new datasets? In this paper, we propose a new framework for few-shot, cross-domain classification. Our retrieval-inspired approach exploits the interrelations in both the training and testing data to output class labels using compact descriptors. Results show that our method can accurately produce land-use predictions on unseen datasets and unseen classes, going beyond the traditional few-shot or cross-domain formulation, and allowing cross-dataset training.

count=2
* SpaceNet 8 - The Detection of Flooded Roads and Buildings
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Hansch_SpaceNet_8_-_The_Detection_of_Flooded_Roads_and_Buildings_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Hansch_SpaceNet_8_-_The_Detection_of_Flooded_Roads_and_Buildings_CVPRW_2022_paper.pdf)]
    * Title: SpaceNet 8 - The Detection of Flooded Roads and Buildings
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Ronny Hänsch, Jacob Arndt, Dalton Lunga, Matthew Gibb, Tyler Pedelose, Arnold Boedihardjo, Desiree Petrie, Todd M. Bacastow
    * Abstract: The frequency and intensity of natural disasters (i.e. wildfires, storms, floods) has increased over recent decades. Extreme weather can often be linked to climate change, and human population expansion and urbanization have led to a growing risk. In particular floods due to large amounts of rainfall are of rising severity and are causing loss of life, destruction of buildings and infrastructure, erosion of arable land, and environmental hazards around the world. Expanding urbanization along rivers and creeks often includes opening flood plains for building construction and river straightening and dredging speeding up the flow of water. In a flood event, rapid response is essential which requires knowledge which buildings are susceptible to flooding and which roads are still accessible. To this aim, SpaceNet 8 is the first remote sensing machine learning training dataset combining building footprint detection, road network extraction, and flood detection covering 850km 2, including 32k buildings and 1,300km roads of which 13% and 15% are flooded, respectively.

count=2
* AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning With Masked Autoencoders
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.pdf)]
    * Title: AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning With Masked Autoencoders
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Wele Gedara Chaminda Bandara, Naman Patel, Ali Gholami, Mehdi Nikkhah, Motilal Agrawal, Vishal M. Patel
    * Abstract: Masked Autoencoders (MAEs) learn generalizable representations for image, text, audio, video, etc., by reconstructing masked input data from tokens of the visible data. Current MAE approaches for videos rely on random patch, tube, or frame based masking strategies to select these tokens. This paper proposes AdaMAE, an adaptive masking strategy for MAEs that is end-to-end trainable. Our adaptive masking strategy samples visible tokens based on the semantic context using an auxiliary sampling network. This network estimates a categorical distribution over spacetime-patch tokens. The tokens that increase the expected reconstruction error are rewarded and selected as visible tokens, motivated by the policy gradient algorithm in reinforcement learning. We show that AdaMAE samples more tokens from the high spatiotemporal information regions, thereby allowing us to mask 95% of tokens, resulting in lower memory requirements and faster pre-training. We conduct ablation studies on the Something-Something v2 (SSv2) dataset to demonstrate the efficacy of our adaptive sampling approach and report state-of-the-art results of 70.0% and 81.7% in top-1 accuracy on SSv2 and Kinetics-400 action classification datasets with a ViT-Base backbone and 800 pre-training epochs. Code and pre-trained models are available at: https://github.com/wgcban/adamae.git

count=2
* Guided Depth Super-Resolution by Deep Anisotropic Diffusion
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Metzger_Guided_Depth_Super-Resolution_by_Deep_Anisotropic_Diffusion_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Metzger_Guided_Depth_Super-Resolution_by_Deep_Anisotropic_Diffusion_CVPR_2023_paper.pdf)]
    * Title: Guided Depth Super-Resolution by Deep Anisotropic Diffusion
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Nando Metzger, Rodrigo Caye Daudt, Konrad Schindler
    * Abstract: Performing super-resolution of a depth image using the guidance from an RGB image is a problem that concerns several fields, such as robotics, medical imaging, and remote sensing. While deep learning methods have achieved good results in this problem, recent work highlighted the value of combining modern methods with more formal frameworks. In this work we propose a novel approach which combines guided anisotropic diffusion with a deep convolutional network and advances the state of the art for guided depth super-resolution. The edge transferring/enhancing properties of the diffusion are boosted by the contextual reasoning capabilities of modern networks, and a strict adjustment step guarantees perfect adherence to the source image. We achieve unprecedented results in three commonly used benchmarks for guided depth super resolution. The performance gain compared to other methods is the largest at larger scales, such as x32 scaling. Code for the proposed method will be made available to promote reproducibility of our results.

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
* Few-Shot Depth Completion Using Denoising Diffusion Probabilistic Model
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Ran_Few-Shot_Depth_Completion_Using_Denoising_Diffusion_Probabilistic_Model_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Ran_Few-Shot_Depth_Completion_Using_Denoising_Diffusion_Probabilistic_Model_CVPRW_2023_paper.pdf)]
    * Title: Few-Shot Depth Completion Using Denoising Diffusion Probabilistic Model
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Weihang Ran, Wei Yuan, Ryosuke Shibasaki
    * Abstract: Generating dense depth maps from sparse LiDAR data is a challenging task, benefiting a lot of computer vision and photogrammetry tasks including autonomous driving, 3D point cloud generation, and aerial spatial awareness. Using RGB images as guidance to generate pixel-wise depth map is good, but these multi-modal data fusion networks always need numerous high-quality datasets like KITTI dataset to train on. Since this may be difficult in some cases, how to achieve few-shot learning with less train samples is worth discussing. So in this paper, we firstly proposed a few-shot learning paradigm for depth completion based on pre-trained denoising diffusion probabilistic model. To evaluate our model and other baselines, we constructed a smaller train set with only 12.5% samples from KITTI depth completion dataset to test their few-shot learning ability. Our model achieved the best on all metrics with a 5% improvement in RMSE compared to the second-place model.

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
* Weakly Misalignment-free Adaptive Feature Alignment for UAVs-based Multimodal Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Weakly_Misalignment-free_Adaptive_Feature_Alignment_for_UAVs-based_Multimodal_Object_Detection_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Weakly_Misalignment-free_Adaptive_Feature_Alignment_for_UAVs-based_Multimodal_Object_Detection_CVPR_2024_paper.pdf)]
    * Title: Weakly Misalignment-free Adaptive Feature Alignment for UAVs-based Multimodal Object Detection
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chen Chen, Jiahao Qi, Xingyue Liu, Kangcheng Bin, Ruigang Fu, Xikun Hu, Ping Zhong
    * Abstract: Visible-infrared (RGB-IR) image fusion has shown great potentials in object detection based on unmanned aerial vehicles (UAVs). However the weakly misalignment problem between multimodal image pairs limits its performance in object detection. Most existing methods often ignore the modality gap and emphasize a strict alignment resulting in an upper bound of alignment quality and an increase of implementation costs. To address these challenges we propose a novel method named Offset-guided Adaptive Feature Alignment (OAFA) which could adaptively adjust the relative positions between multimodal features. Considering the impact of modality gap on the cross-modality spatial matching a Cross-modality Spatial Offset Modeling (CSOM) module is designed to establish a common subspace to estimate the precise feature-level offsets. Then an Offset-guided Deformable Alignment and Fusion (ODAF) module is utilized to implicitly capture optimal fusion positions for detection task rather than conducting a strict alignment. Comprehensive experiments demonstrate that our method not only achieves state-of-the-art performance in the UAVs-based object detection task but also shows strong robustness to the weakly misalignment problem.

count=2
* WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kumar_WildlifeMapper_Aerial_Image_Analysis_for_Multi-Species_Detection_and_Identification_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Kumar_WildlifeMapper_Aerial_Image_Analysis_for_Multi-Species_Detection_and_Identification_CVPR_2024_paper.pdf)]
    * Title: WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Satish Kumar, Bowen Zhang, Chandrakanth Gudavalli, Connor Levenson, Lacey Hughey, Jared A. Stabach, Irene Amoke, Gordon Ojwang, Joseph Mukeka, Stephen Mwiu, Joseph Ogutu, Howard Frederick, B.S. Manjunath
    * Abstract: We introduce WildlifeMapper (WM) a flexible model designed to detect locate and identify multiple species in aerial imagery. It addresses the limitations of traditional labor-intensive wildlife population assessments that are central to advancing environmental conservation efforts worldwide. While a number of methods exist to automate this process they are often limited in their ability to generalize to different species or landscapes due to the dominance of homogeneous backgrounds and/or poorly captured local image structures. WM introduces two novel modules that help to capture the local structure and context of objects of interest to accurately localize and identify them achieving a state-of-the-art (SOTA) detection rate of 0.56 mAP. Further we introduce a large aerial imagery dataset with more than 11k Images and 28k annotations verified by trained experts. WM also achieves SOTA performance on 3 other publicly available aerial survey datasets collected across 4 different countries improving mAP by 42%. Source code and trained models are available at Github

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
* High-fidelity Person-centric Subject-to-Image Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_High-fidelity_Person-centric_Subject-to-Image_Synthesis_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_High-fidelity_Person-centric_Subject-to-Image_Synthesis_CVPR_2024_paper.pdf)]
    * Title: High-fidelity Person-centric Subject-to-Image Synthesis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yibin Wang, Weizhong Zhang, Jianwei Zheng, Cheng Jin
    * Abstract: Current subject-driven image generation methods encounter significant challenges in person-centric image generation. The reason is that they learn the semantic scene and person generation by fine-tuning a common pre-trained diffusion which involves an irreconcilable training imbalance. Precisely to generate realistic persons they need to sufficiently tune the pre-trained model which inevitably causes the model to forget the rich semantic scene prior and makes scene generation over-fit to the training data. Moreover even with sufficient fine-tuning these methods can still not generate high-fidelity persons since joint learning of the scene and person generation also lead to quality compromise. In this paper we propose Face-diffuser an effective collaborative generation pipeline to eliminate the above training imbalance and quality compromise. Specifically we first develop two specialized pre-trained diffusion models i.e. Text-driven Diffusion Model (TDM) and Subject-augmented Diffusion Model (SDM) for scene and person generation respectively. The sampling process is divided into three sequential stages i.e. semantic scene construction subject-scene fusion and subject enhancement. The first and last stages are performed by TDM and SDM respectively. The subject-scene fusion stage that is the collaboration achieved through a novel and highly effective mechanism Saliency-adaptive Noise Fusion (SNF). Specifically it is based on our key observation that there exists a robust link between classifier-free guidance responses and the saliency of generated images. In each time step SNF leverages the unique strengths of each model and allows for the spatial blending of predicted noises from both models automatically in a saliency-aware manner all of which can be seamlessly integrated into the DDIM sampling process. Extensive experiments confirm the impressive effectiveness and robustness of the Face-diffuser in generating high-fidelity person images depicting multiple unseen persons with varying contexts. Code is available at https://github.com/CodeGoat24/Face-diffuser.

count=2
* Fantastic Animals and Where to Find Them: Segment Any Marine Animal with Dual SAM
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Fantastic_Animals_and_Where_to_Find_Them_Segment_Any_Marine_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Fantastic_Animals_and_Where_to_Find_Them_Segment_Any_Marine_CVPR_2024_paper.pdf)]
    * Title: Fantastic Animals and Where to Find Them: Segment Any Marine Animal with Dual SAM
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Pingping Zhang, Tianyu Yan, Yang Liu, Huchuan Lu
    * Abstract: As an important pillar of underwater intelligence Marine Animal Segmentation (MAS) involves segmenting animals within marine environments. Previous methods don't excel in extracting long-range contextual features and overlook the connectivity between discrete pixels. Recently Segment Anything Model (SAM) offers a universal framework for general segmentation tasks. Unfortunately trained with natural images SAM does not obtain the prior knowledge from marine images. In addition the single-position prompt of SAM is very insufficient for prior guidance. To address these issues we propose a novel feature learning framework named Dual-SAM for high-performance MAS. To this end we first introduce a dual structure with SAM's paradigm to enhance feature learning of marine images. Then we propose a Multi-level Coupled Prompt (MCP) strategy to instruct comprehensive underwater prior information and enhance the multi-level features of SAM's encoder with adapters. Subsequently we design a Dilated Fusion Attention Module (DFAM) to progressively integrate multi-level features from SAM's encoder. Finally instead of directly predicting the masks of marine animals we propose a Criss-Cross Connectivity Prediction (C3P) paradigm to capture the inter-connectivity between discrete pixels. With dual decoders it generates pseudo-labels and achieves mutual supervision for complementary feature representations resulting in considerable improvements over previous techniques. Extensive experiments verify that our proposed method achieves state-of-the-art performances on five widely-used MAS datasets. The code is available at https://github.com/Drchip61/Dual SAM.

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
* DeepLocalization: Using Change Point Detection for Temporal Action Localization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Rahman_DeepLocalization_Using_Change_Point_Detection_for_Temporal_Action_Localization_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Rahman_DeepLocalization_Using_Change_Point_Detection_for_Temporal_Action_Localization_CVPRW_2024_paper.pdf)]
    * Title: DeepLocalization: Using Change Point Detection for Temporal Action Localization
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Mohammed Shaiqur Rahman, Ibne Farabi Shihab, Lynna Chu, Anuj Sharma
    * Abstract: In this study we introduce DeepLocalization an innovative framework devised for the real-time localization of actions tailored explicitly for monitoring driver behavior. Utilizing the power of advanced deep learning methodologies our objective is to tackle the critical issue of distracted driving--a significant factor contributing to road accidents. Our strategy employs a dual approach: leveraging Graph-Based Change-Point Detection for pinpointing actions in time alongside a Video Large Language Model (Video-LLM) for precisely categorizing activities. Through careful prompt engineering we customize the Video-LLM to adeptly handle driving activities' nuances ensuring its classification efficacy even with sparse data. Engineered to be lightweight our framework is optimized for consumer-grade GPUs making it vastly applicable in practical scenarios. We subjected our method to rigorous testing on the SynDD2 dataset a complex benchmark for distracted driving behaviors where it demonstrated commendable performance--achieving 57.5% accuracy in event classification and 51% in event detection. These outcomes underscore the substantial promise of DeepLocalization in accurately identifying diverse driver behaviors and their temporal occurrences all within the bounds of limited computational resources.

count=2
* UrbanSARFloods: Sentinel-1 SLC-Based Benchmark Dataset for Urban and Open-Area Flood Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Zhao_UrbanSARFloods_Sentinel-1_SLC-Based_Benchmark_Dataset_for_Urban_and_Open-Area_Flood_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Zhao_UrbanSARFloods_Sentinel-1_SLC-Based_Benchmark_Dataset_for_Urban_and_Open-Area_Flood_CVPRW_2024_paper.pdf)]
    * Title: UrbanSARFloods: Sentinel-1 SLC-Based Benchmark Dataset for Urban and Open-Area Flood Mapping
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jie Zhao, Zhitong Xiong, Xiao Xiang Zhu
    * Abstract: Due to its cloud-penetrating capability and independence from solar illumination satellite Synthetic Aperture Radar (SAR) is the preferred data source for large-scale flood mapping providing global coverage and including various land cover classes. However most studies on large-scale SAR-derived flood mapping using deep learning algorithms have primarily focused on flooded open areas utilizing available open-access datasets (e.g. Sen1Floods11) and with limited attention to urban floods. To address this gap we introduce UrbanSARFloods a floodwater dataset featuring pre-processed Sentinel-1 intensity data and interferometric coherence imagery acquired before and during flood events. It contains 8879 512 x 512 chips covering 807500 km2 across 20 land cover classes and 5 continents spanning 18 flood events. We used UrbanSARFloods to benchmark existing state-of-the-art convolutional neural networks (CNNs) for segmenting open and urban flood areas. Our findings indicate that prevalent approaches including the Weighted Cross-Entropy (WCE) loss and the application of transfer learning with pretrained models fall short in overcoming the obstacles posed by imbalanced data and the constraints of a small training dataset. Urban flood detection remains challenging. Future research should explore strategies for addressing imbalanced data challenges and investigate transfer learning's potential for SAR-based large-scale flood mapping. Besides expanding this dataset to include additional flood events holds promise for enhancing its utility and contributing to advancements in flood mapping techniques. The UrbanSARFloods dataset including training validation data and raw data can be found at https://github.com/jie666-6/UrbanSARFloods.

count=2
* Adapting the Segment Anything Model During Usage in Novel Situations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Schon_Adapting_the_Segment_Anything_Model_During_Usage_in_Novel_Situations_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Schon_Adapting_the_Segment_Anything_Model_During_Usage_in_Novel_Situations_CVPRW_2024_paper.pdf)]
    * Title: Adapting the Segment Anything Model During Usage in Novel Situations
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Robin Schön, Julian Lorenz, Katja Ludwig, Rainer Lienhart
    * Abstract: The interactive segmentation task consists in the creation of object segmentation masks based on user interactions. The most common way to guide a model towards producing a correct segmentation consists in clicks on the object and background. The recently published Segment Anything Model (SAM) supports a generalized version of the interactive segmentation problem and has been trained on an object segmentation dataset which contains 1.1B masks. Though being trained extensively and with the explicit purpose of serving as a foundation model we show significant limitations of SAM when being applied for interactive segmentation on novel domains or object types. On the used datasets SAM displays a failure rate FR30@90 of up to 72.6 %. Since we still want such foundation models to be immediately applicable we present a framework that can adapt SAM during immediate usage. For this we will leverage the user interactions and masks which are constructed during the interactive segmentation process. We use this information to generate pseudo-labels which we use to compute a loss function and optimize a part of the SAM model. The presented method causes a relative reduction of up to 48.1 % in the FR20@85 and 46.6 % in the FR30@90 metrics.

count=2
* Vehicle Re-identification with Learned Representation and Spatial Verification and Abnormality Detection with Multi-Adaptive Vehicle Detectors for Traffic Video Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/AI_City/Nguyen_Vehicle_Re-identification_with_Learned_Representation_and_Spatial_Verification_and_Abnormality_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI City/Nguyen_Vehicle_Re-identification_with_Learned_Representation_and_Spatial_Verification_and_Abnormality_CVPRW_2019_paper.pdf)]
    * Title: Vehicle Re-identification with Learned Representation and Spatial Verification and Abnormality Detection with Multi-Adaptive Vehicle Detectors for Traffic Video Analysis
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Khac-Tuan Nguyen,  Trung-Hieu Hoang,  Minh-Triet Tran,  Trung-Nghia Le,  Ngoc-Minh Bui,  Trong-Le Do,  Viet-Khoa Vo-Ho,  Quoc-An Luong,  Mai-Khiem Tran,  Thanh-An Nguyen,  Thanh-Dat Truong,  Vinh-Tiep Nguyen,  Minh N. Do
    * Abstract: Traffic flow analysis is essential for intelligent transportation systems. In this paper, we propose methods for two challenging problems in traffic flow analysis: vehicle re-identification and abnormal event detection. For the first problem, we propose to combine learned high-level features for vehicle instance representation with hand-crafted local features for spatial verification. For the second problem, we propose to use multiple adaptive vehicle detectors for anomaly proposal and use heuristics properties extracted from anomaly proposals to determine anomaly events. Experiments on the datasets of traffic flow analysis from AI City Challenge 2019 show that our methods achieve mAP of 0.4008 for vehicle re-identification in Track 2, and can detect abnormal events with very high accuracy (F1 = 0.9429) in Track 3.

count=2
* iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.pdf)]
    * Title: iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Syed Waqas Zamir,  Aditya Arora,  Akshita Gupta,  Salman  Khan,  Guolei Sun,  Fahad Shahbaz Khan,  Fan Zhu,  Ling Shao,  Gui-Song Xia,  Xiang Bai
    * Abstract: Existing Earth Vision datasets are either suitable for semantic segmentation or object detection. In this work, we introduce the first benchmark dataset for instance segmentation in aerial imagery that combines instance-level object detection and pixel-level segmentation tasks. In comparison to instance segmentation in natural scenes, aerial images present unique challenges e.g., huge number of instances per image, large object-scale variations and abundant tiny objects. Our large-scale and densely annotated Instance Segmentation in Aerial Images Dataset (IS-AID) comes with 655,451 object instances for 15 categories across 2,806 high-resolution images. Such precise per-pixel annotations for each instance ensure accurate localization that is essential for detailed scene analysis. Compared to existing small-scale aerial image based instance segmentation datasets, IS-AID contains 15x the number of object categories and 5x the number of instances. We benchmark our dataset using two popular instance segmentation approaches for natural images, namely Mask R-CNN and PANet. In our experiments we show that direct application of off-the-shelf Mask R-CNN and PANet on aerial images provide sub-optimal instance segmentation results, thus requiring specialized solutions from the research community.

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
* HIDeGan: A Hyperspectral-Guided Image Dehazing GAN
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w14/Mehta_HIDeGan_A_Hyperspectral-Guided_Image_Dehazing_GAN_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w14/Mehta_HIDeGan_A_Hyperspectral-Guided_Image_Dehazing_GAN_CVPRW_2020_paper.pdf)]
    * Title: HIDeGan: A Hyperspectral-Guided Image Dehazing GAN
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Aditya Mehta, Harsh Sinha, Pratik Narang, Murari Mandal
    * Abstract: Haze removal in images captured from a diverse set of scenarios is a very challenging problem. The existing dehazing methods either reconstruct the transmission map or directly estimate the dehazed image in RGB color space. In this paper, we make a first attempt to propose a Hyperspectral-guided Image Dehazing Generative Adversarial Network (HIDEGAN). The HIDEGAN architecture is formulated by designing an enhanced version of CYCLEGAN named R2HCYCLE and an enhanced conditional GAN named H2RGAN. The R2HCYCLE makes use of the hyperspectral-image (HSI) in combination with cycle-consistency and skeleton losses in order to improve the quality of information recovery by analyzing the entire spectrum. The H2RGAN estimates the clean RGB image from the hazy hyperspectral image generated by the R2HCYCLE. The models designed for spatial-spectral-spatial mapping generate visually better haze-free images. To facilitate HSI generation, datasets from spectral reconstruction challenge at NTIRE 2018 and NTIRE 2020 are used. A comprehensive set of experiments were conducted on the D-Hazy, and the recent RESIDE-Standard (SOTS), RESIDE-b (OTS) and RESIDE-Standard (HSTS) datasets. The proposed HIDEGAN outperforms the existing state-of-the-art in all these datasets.

count=2
* iTASK - Intelligent Traffic Analysis Software Kit
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w35/Tran_iTASK_-_Intelligent_Traffic_Analysis_Software_Kit_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Tran_iTASK_-_Intelligent_Traffic_Analysis_Software_Kit_CVPRW_2020_paper.pdf)]
    * Title: iTASK - Intelligent Traffic Analysis Software Kit
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Minh-Triet Tran, Tam V. Nguyen, Trung-Hieu Hoang, Trung-Nghia Le, Khac-Tuan Nguyen, Dat-Thanh Dinh, Thanh-An Nguyen, Hai-Dang Nguyen, Xuan-Nhat Hoang, Trong-Tung Nguyen, Viet-Khoa Vo-Ho, Trong-Le Do, Lam Nguyen, Minh-Quan Le, Hoang-Phuc Nguyen-Dinh, Trong-Thang Pham, Xuan-Vy Nguyen, E-Ro Nguyen, Quoc-Cuong Tran, Hung Tran, Hieu Dao, Mai-Khiem Tran, Quang-Thuc Nguyen, Tien-Phat Nguyen, The-Anh Vu-Le, Gia-Han Diep, Minh N. Do
    * Abstract: Traffic flow analysis is essential for intelligent transportation systems. In this paper, we introduce our Intelligent Traffic Analysis Software Kit (iTASK) to tackle three challenging problems: vehicle flow counting, vehicle re-identification, and abnormal event detection. For the first problem, we propose to real-time track vehicles moving along the desired direction in corresponding motion-of-interests (MOIs). For the second problem, we consider each vehicle as a document with multiple semantic words (i.e., vehicle attributes) and transform the given problem to classical document retrieval. For the last problem, we propose to forward and backward refine anomaly detection using GAN-based future prediction and backward tracking completely stalled vehicle or sudden-change direction, respectively. Experiments on the datasets of traffic flow analysis from AI City Challenge 2020 show our competitive results, namely, S1 score of 0.8297 for vehicle flow counting in Track 1, mAP score of 0.3882 for vehicle re-identification in Track 2, and S4 score of 0.9059 for anomaly detection in Track 4. All data and source code are publicly available on our project page.

count=2
* Any-Shot Sequential Anomaly Detection in Surveillance Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w54/Doshi_Any-Shot_Sequential_Anomaly_Detection_in_Surveillance_Videos_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Doshi_Any-Shot_Sequential_Anomaly_Detection_in_Surveillance_Videos_CVPRW_2020_paper.pdf)]
    * Title: Any-Shot Sequential Anomaly Detection in Surveillance Videos
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Keval Doshi, Yasin Yilmaz
    * Abstract: Anomaly detection in surveillance videos has been recently gaining attention. Even though the performance of state-of-the-art methods on publicly available data sets has been competitive, they demand a massive amount of training data. Also, they lack a concrete approach for continuously updating the trained model once new data is available. Furthermore, online decision making is an important but mostly neglected factor in this domain. Motivated by these research gaps, we propose an online anomaly detection method for surveillance videos using transfer learning and any-shot learning, which in turn significantly reduces the training complexity and provides a mechanism which can detect anomalies using only a few labeled nominal examples. Our proposed algorithm leverages the feature extraction power of neural network-based models for transfer learning, and the any-shot learning capability of statistical detection methods.

count=2
* Calibrated Vehicle Paint Signatures for Simulating Hyperspectral Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Mulhollan_Calibrated_Vehicle_Paint_Signatures_for_Simulating_Hyperspectral_Imagery_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w6/Mulhollan_Calibrated_Vehicle_Paint_Signatures_for_Simulating_Hyperspectral_Imagery_CVPRW_2020_paper.pdf)]
    * Title: Calibrated Vehicle Paint Signatures for Simulating Hyperspectral Imagery
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zachary Mulhollan, Aneesh Rangnekar, Timothy Bauch, Matthew J. Hoffman, Anthony Vodacek
    * Abstract: We investigate a procedure for rapidly adding calibrated vehicle visible-near infrared (VNIR) paint signatures to an existing hyperspectral simulator - The Digital Imaging and Remote Sensing Image Generation (DIRSIG) model - to create more diversity in simulated urban scenes. The DIRSIG model can produce synthetic hyperspectral imagery with user-specified geometry, atmospheric conditions, and ground target spectra. To render an object pixel's spectral signature, DIRSIG uses a large database of reflectance curves for the corresponding object material and a bidirectional reflectance model to introduce s due to orientation and surface structure. However, this database contains only a few spectral curves for vehicle paints and generates new paint signatures by combining these curves internally. In this paper we demonstrate a method to rapidly generate multiple paint spectra, flying a drone carrying a pushbroom hyperspectral camera to image a university parking lot. We then process the images to convert them from the digital count space to spectral reflectance without the need of calibration panels in the scene, and port the paint signatures into DIRSIG for successful integration into the newly rendered sets of synthetic VNIR hyperspectral scenes.

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
* A Fast Self-Tuning Background Subtraction Algorithm
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Wang_A_Fast_Self-Tuning_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Wang_A_Fast_Self-Tuning_2014_CVPR_paper.pdf)]
    * Title: A Fast Self-Tuning Background Subtraction Algorithm
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Bin Wang, Piotr Dudek
    * Abstract: In this paper, a fast pixel-level adapting background detection algorithm is presented. The proposed background model records not only each pixel's historical background values, but also estimates the efficacies of these values, based on the occurrence statistics. It is therefore capable of removing the least useful background values from the background model, selectively adapting to background changes with different timescales, and restraining the generation of ghosts. A further control process adjusts the individual decision threshold for each pixel, and reduces high frequency temporal noise, based on a measure of classification uncertainty in each pixel. Evaluation results based on the ChangeDetection.net database are presented in this paper. The results indicate that the proposed algorithm outperforms the majority of earlier state-of-the-art algorithms not only in terms of accuracy, but also in terms of processing speed.

count=2
* Integrating LIDAR Range Scans and Photographs with Temporal Changes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W19/html/Morago_Integrating_LIDAR_Range_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W19/papers/Morago_Integrating_LIDAR_Range_2014_CVPR_paper.pdf)]
    * Title: Integrating LIDAR Range Scans and Photographs with Temporal Changes
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Brittany Morago, Giang Bui, Ye Duan
    * Abstract: Registering 2D and 3D data is a rapidly growing research area. Motivating much of this work is the fact that 3D range scans and 2D imagery provide different, but complementing information about the same subject. Combining these two perspectives leads to the creation of accurate 3D models that are texture mapped with high resolution color information. Imagery can even be obtained on different days and in different seasons and registered together to show how a scene has changed with time. Finding correspondences among data captured with different cameras and containing content and temporal changes can be a challenging task. We address these difficulties by presenting a contextual approach for finding 2D matches, performing 2D-3D fusion by solving the projection matrix of a camera directly from its relationship to highly accurate range scan points, and minimizing an energy function based on gradient information in a 3D depth image.

count=2
* Online Multimodal Video Registration Based on Shape Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/St-Charles_Online_Multimodal_Video_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/St-Charles_Online_Multimodal_Video_2015_CVPR_paper.pdf)]
    * Title: Online Multimodal Video Registration Based on Shape Matching
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: The registration of video sequences captured using different types of sensors often relies on dense feature matching methods, which are very costly. In this paper, we study the problem of "almost planar" scene registration (i.e. where the planar ground assumption is almost respected) in multimodal imagery using target shape information. We introduce a new strategy for robustly aligning scene elements based on the random sampling of shape contour correspondences and on the continuous update of our transformation model's parameters. We evaluate our solution on a public dataset and show its superiority by comparing it to a recently published method that targets the same problem. To make comparisons between such methods easier in the future, we provide our evaluation tools along with a full implementation of our solution online.

count=2
* Rolling Shutter Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Punnappurath_Rolling_Shutter_Super-Resolution_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Punnappurath_Rolling_Shutter_Super-Resolution_ICCV_2015_paper.pdf)]
    * Title: Rolling Shutter Super-Resolution
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Abhijith Punnappurath, Vijay Rengarajan, A.N. Rajagopalan
    * Abstract: Classical multi-image super-resolution (SR) algorithms, designed for CCD cameras, assume that the motion among the images is global. But CMOS sensors that have increasingly started to replace their more expensive CCD counterparts in many applications do not respect this assumption if there is a motion of the camera relative to the scene during the exposure duration of an image because of the row-wise acquisition mechanism. In this paper, we study the hitherto unexplored topic of multi-image SR in CMOS cameras. We initially develop an SR observation model that accounts for the row-wise distortions called the ``rolling shutter'' (RS) effect observed in images captured using non-stationary CMOS cameras. We then propose a unified RS-SR framework to obtain an RS-free high-resolution image (and the row-wise motion) from distorted low-resolution images. We demonstrate the efficacy of the proposed scheme using synthetic data as well as real images captured using a hand-held CMOS camera. Quantitative and qualitative assessments reveal that our method significantly advances the state-of-the-art.

count=2
* Minimal Solvers for 3D Geometry From Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Minimal_Solvers_for_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Zheng_Minimal_Solvers_for_ICCV_2015_paper.pdf)]
    * Title: Minimal Solvers for 3D Geometry From Satellite Imagery
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Enliang Zheng, Ke Wang, Enrique Dunn, Jan-Michael Frahm
    * Abstract: We propose two novel minimal solvers which advance the state of the art in satellite imagery processing. Our methods are efficient and do not rely on the prior existence of complex inverse mapping functions to correlate 2D image coordinates and 3D terrain. Our first solver improves on the stereo correspondence problem for satellite imagery, in that we provide an exact image-to-object space mapping (where prior methods were inaccurate). Our second solver provides a novel mechanism for 3D point triangulation, which has improved robustness and accuracy over prior techniques. Given the usefulness and ubiquity of satellite imagery, our proposed methods allow for improved results in a variety of existing and future applications.

count=2
* Fast Structure from Motion for Sequential and Wide Area Motion Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w1/html/AliAkbarpour_Fast_Structure_from_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w1/papers/AliAkbarpour_Fast_Structure_from_ICCV_2015_paper.pdf)]
    * Title: Fast Structure from Motion for Sequential and Wide Area Motion Imagery
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Hadi AliAkbarpour, Kannappan Palaniappan, Guna Seetharaman
    * Abstract: We present a fast and efficient Structure-from-Motion (SfM) pipeline for refinement of camera parameters and 3D scene reconstruction given initial noisy camera metadata measurements. Examples include aerial Wide Area Motion Imagery (WAMI) which is typically acquired in a circular trajectory and other sequentially ordered multiview stereo imagery like Middlebury [??], Fountain [??] or body-worn videos [??]. Image frames are assumed (partially) ordered with approximate camera position and orientation information available from (imprecise) IMU and GPS sensors. In the proposed BA4S pipeline the noisy camera parameters or poses are directly used in a fast Bundle Adjustment (BA) optimization. Since the sequential ordering of the cameras is known, consecutive frame-to-frame matching is used to find a set of feature correspondences for the triangulation step of SfM. These putative correspondences are directly used in the BA optimization without any early-stage filtering (i.e. no RANSAC) using a statistical robust error function based on co-visibility, to deal with outliers (mismatches), which significantly speeds up our SfM pipeline by more than 100 times compared to VisualSfM.

count=2
* HDR Recovery Under Rolling Shutter Distortions
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w3/html/Gupta_HDR_Recovery_Under_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w3/papers/Gupta_HDR_Recovery_Under_ICCV_2015_paper.pdf)]
    * Title: HDR Recovery Under Rolling Shutter Distortions
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Sheetal B. Gupta, A. N. Rajagopalan, Gunasekaran Seetharaman
    * Abstract: Preserving the high dynamic irradiance of a scene is essential for many computer vision algorithms. In this paper, we develop a technique for high dynamic range (HDR) reconstruction from differently exposed frames captured with CMOS cameras which use a rolling shutter (RS) to good effect for reducing power consumption. However, because these sensors are exposed to the scene row-wise, any unintentional handshake poses a challenge for HDR reconstruction since each row experiences a different motion. We account for this motion in the irradiance domain by picking the correct warp for each row within a predefined search space. The RS effect is rectified and a clean frame is propagated from one exposure to another until we obtain rectified irradiance corresponding to all the exposures. The rectified irradiances are finally fused to yield an HDR map that is free from RS distortions.

count=2
* A Unified Model for Near and Remote Sensing
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Workman_A_Unified_Model_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Workman_A_Unified_Model_ICCV_2017_paper.pdf)]
    * Title: A Unified Model for Near and Remote Sensing
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Scott Workman, Menghua Zhai, David J. Crandall, Nathan Jacobs
    * Abstract: We propose a novel convolutional neural network architecture for estimating geospatial functions such as population density, land cover, or land use. In our approach, we combine overhead and ground-level images in an end-to-end trainable neural network, which uses kernel regression and density estimation to convert features extracted from the ground-level images into a dense feature map. The output of this network is a dense estimate of the geospatial function in the form of a pixel-level labeling of the overhead image. To evaluate our approach, we created a large dataset of overhead and ground-level images from a major urban area with three sets of labels: land use, building function, and building age. We find that our approach is more accurate for all tasks, in some cases dramatically so.

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
* Finding Time Together: Detection and Classification of Focused Interaction in Egocentric Video
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w34/html/individuals_s.banodundee.ac.uk_s.j.z.mckennadundee.ac.uk_j.n.zhangdundee.ac.uk_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w34/individuals_s.banodundee.ac.uk_s.j.z.mckennadundee.ac.uk_j.n.zhangdundee.ac.uk_ICCV_2017_paper.pdf)]
    * Title: Finding Time Together: Detection and Classification of Focused Interaction in Egocentric Video
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Sophia Bano, Stephen J. McKenna, Jianguo Zhang
    * Abstract: Focused interaction occurs when co-present individuals, having mutual focus of attention, interact by establishing face-to-face engagement and direct conversation. Face-to-face engagement is often not maintained throughout the entirety of a focused interaction. In this paper, we present an online method for automatic classification of unconstrained egocentric (first-person perspective) videos into segments having no focused interaction, focused interaction when the camera wearer is stationary and focused interaction when the camera wearer is moving. We extract features from both audio and video data streams and perform temporal segmentation by using support vector machines with linear and non-linear kernels. We provide empirical evidence that fusion of visual face track scores, camera motion profile and audio voice activity scores is an effective combination for focused interaction classification.

count=2
* Mutual Foreground Segmentation With Multispectral Stereo Pairs
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w6/html/St-Charles_Mutual_Foreground_Segmentation_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w6/St-Charles_Mutual_Foreground_Segmentation_ICCV_2017_paper.pdf)]
    * Title: Mutual Foreground Segmentation With Multispectral Stereo Pairs
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: Foreground-background segmentation of video sequences is a low-level process commonly used in machine vision, and highly valued in video content analysis and smart surveillance applications. Its efficacy relies on the contrast between objects observed by the sensor. In this work, we study how the combination of sensors operating in the long-wavelength infrared (LWIR) and visible spectra can improve the performance of foreground-background segmentation methods. As opposed to a classic visible spectrum stereo pair, this multispectral pair is more adequate for object segmentation since it reduces the odds of observing low-contrast regions simultaneously in both images. We show that by alternately minimizing stereo disparity and binary segmentation energies with dynamic priors, we can drastically improve the results of a traditional video segmentation approach applied to each sensor individually. Our implementation is freely available online for anyone wishing to recreate our results.

count=2
* Geography-Aware Self-Supervised Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.pdf)]
    * Title: Geography-Aware Self-Supervised Learning
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell, Stefano Ermon
    * Abstract: Contrastive learning methods have significantly narrowed the gap between supervised and unsupervised learning on computer vision tasks. In this paper, we explore their application to geo-located datasets, e.g. remote sensing, where unlabeled data is often abundant but labeled data is scarce. We first show that due to their different characteristics, a non-trivial gap persists between contrastive and supervised learning on standard benchmarks. To close the gap, we propose novel training methods that exploit the spatio-temporal structure of remote sensing data. We leverage spatially aligned images over time to construct temporal positive pairs in contrastive learning and geo-location to design pre-text tasks. Our experiments show that our proposed method closes the gap between contrastive and supervised learning on image classification, object detection and semantic segmentation for remote sensing. Moreover, we demonstrate that the proposed method can also be applied to geo-tagged ImageNet images, improving downstream performance on various tasks.

count=2
* Progressive Unsupervised Deep Transfer Learning for Forest Mapping in Satellite Image
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/html/Ahmed_Progressive_Unsupervised_Deep_Transfer_Learning_for_Forest_Mapping_in_Satellite_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/papers/Ahmed_Progressive_Unsupervised_Deep_Transfer_Learning_for_Forest_Mapping_in_Satellite_ICCVW_2021_paper.pdf)]
    * Title: Progressive Unsupervised Deep Transfer Learning for Forest Mapping in Satellite Image
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Nouman Ahmed, Sudipan Saha, Muhammad Shahzad, Muhammad Moazam Fraz, Xiao Xiang Zhu
    * Abstract: Automated forest mapping is important to understand our forests that play a key role in ecological system. However, efforts towards forest mapping is impeded by difficulty to collect labeled forest images that show large intraclass variation. Recently unsupervised learning has shown promising capability when exploiting limited labeled data. Motivated by this, we propose a progressive unsupervised deep transfer learning method for forest mapping. The proposed method exploits a pre-trained model that is subsequently fine-tuned over the target forest domain. We propose two different fine-tuning mechanism, one works in a totally unsupervised setting by jointly learning the parameters of CNN and the k-means based cluster assignments of the resulting features and the other one works in a semi-supervised setting by exploiting the extracted knearest neighbor based pseudo labels. The proposed progressive scheme is evaluated on publicly available EuroSAT dataset using the relevant base model trained on BigEarthNet labels. The results show that the proposed method greatly improves the forest regions classification accuracy as compared to the unsupervised baseline, nearly approaching the supervised classification approach.

count=2
* Background/Foreground Separation: Guided Attention Based Adversarial Modeling (GAAM) Versus Robust Subspace Learning Methods
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Sultana_BackgroundForeground_Separation_Guided_Attention_Based_Adversarial_Modeling_GAAM_Versus_Robust_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Sultana_BackgroundForeground_Separation_Guided_Attention_Based_Adversarial_Modeling_GAAM_Versus_Robust_ICCVW_2021_paper.pdf)]
    * Title: Background/Foreground Separation: Guided Attention Based Adversarial Modeling (GAAM) Versus Robust Subspace Learning Methods
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Maryam Sultana, Arif Mahmood, Thierry Bouwmans, Muhammad Haris Khan, Soon Ki Jung
    * Abstract: Background-Foreground separation and appearance generation is a fundamental step in many computer vision applications. Existing methods like Robust Subspace Learning (RSL) suffer performance degradation in the presence of challenges like bad weather, illumination variations, occlusion, dynamic backgrounds and intermittent object motion. In the current work we propose a more accurate deep neural network based model for background-foreground separation and complete appearance generation of the foreground objects. Our proposed model, Guided Attention based Adversarial Model (GAAM), can efficiently extract pixel-level boundaries of the foreground objects for improved appearance generation. Unlike RSL methods our model extracts the binary information of foreground objects labeled as attention map which guides our generator network to segment the foreground objects from the complex background information. Wide range of experiments performed on the benchmark CDnet2014 dataset demonstrate the excellent performance of our proposed model.

count=2
* Large Selective Kernel Network for Remote Sensing Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.pdf)]
    * Title: Large Selective Kernel Network for Remote Sensing Object Detection
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yuxuan Li, Qibin Hou, Zhaohui Zheng, Ming-Ming Cheng, Jian Yang, Xiang Li
    * Abstract: Recent research on remote sensing object detection has largely focused on improving the representation of oriented bounding boxes but has overlooked the unique prior knowledge presented in remote sensing scenarios. Such prior knowledge can be useful because tiny remote sensing objects may be mistakenly detected without referencing a sufficiently long-range context, which can vary for different objects. This paper considers these priors and proposes the lightweight Large Selective Kernel Network (LSKNet). LSKNet can dynamically adjust its large spatial receptive field to better model the ranging context of various objects in remote sensing scenarios. To our knowledge, large and selective kernel mechanisms have not been previously explored in remote sensing object detection. Without bells and whistles, our lightweight LSKNet sets new state-of-the-art scores on standard benchmarks, i.e., HRSC2016 (98.46% mAP), DOTA-v1.0 (81.85% mAP), and FAIR1M-v1.0 (47.87% mAP).

count=2
* Do VSR Models Generalize Beyond LRS3?
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Djilali_Do_VSR_Models_Generalize_Beyond_LRS3_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Djilali_Do_VSR_Models_Generalize_Beyond_LRS3_WACV_2024_paper.pdf)]
    * Title: Do VSR Models Generalize Beyond LRS3?
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Yasser Abdelaziz Dahou Djilali, Sanath Narayan, Eustache LeBihan, Haithem Boussaid, Ebtesam Almazrouei, Merouane Debbah
    * Abstract: The Lip Reading Sentences-3 (LRS3) benchmark has primarily been the focus of intense research in visual speech recognition (VSR) during the last few years. As a result, there is an increased risk of overfitting to its excessively used test set, which is only one hour duration. To alleviate this issue, we build a new VSR test set by closely following the LRS3 dataset creation processes. We then evaluate and analyse the extent to which the current VSR models generalize to the new test data. We evaluate a broad range of publicly available VSR models and find significant drops in performance on our test set, compared to their corresponding LRS3 results. Our results suggest that the increase in word error rates is caused by the models' inability to generalize to slightly "harder" and more realistic lip sequences than those found in the LRS3 test set. Our new test benchmark will be made public in order to enable future research towards more robust VSR models.

count=2
* Controlled Recognition Bounds for Visual Learning and Exploration
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/2a50e9c2d6b89b95bcb416d6857f8b45-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/2a50e9c2d6b89b95bcb416d6857f8b45-Paper.pdf)]
    * Title: Controlled Recognition Bounds for Visual Learning and Exploration
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Vasiliy Karasev, Alessandro Chiuso, Stefano Soatto
    * Abstract: We describe the tradeoff between the performance in a visual recognition problem and the control authority that the agent can exercise on the sensing process. We focus on the problem of “visual search” of an object in an otherwise known and static scene, propose a measure of control authority, and relate it to the expected risk and its proxy (conditional entropy of the posterior density). We show this analytically, as well as empirically by simulation using the simplest known model that captures the phenomenology of image formation, including scaling and occlusions. We show that a “passive” agent given a training set can provide no guarantees on performance beyond what is afforded by the priors, and that an “omnipotent” agent, capable of infinite control authority, can achieve arbitrarily good performance (asymptotically).

count=2
* Tracking Time-varying Graphical Structure
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2013/hash/233509073ed3432027d48b1a83f5fbd2-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2013/file/233509073ed3432027d48b1a83f5fbd2-Paper.pdf)]
    * Title: Tracking Time-varying Graphical Structure
    * Publisher: NeurIPS
    * Publication Date: `2013`
    * Authors: Erich Kummerfeld, David Danks
    * Abstract: Structure learning algorithms for graphical models have focused almost exclusively on stable environments in which the underlying generative process does not change; that is, they assume that the generating model is globally stationary. In real-world environments, however, such changes often occur without warning or signal. Real-world data often come from generating models that are only locally stationary. In this paper, we present LoSST, a novel, heuristic structure learning algorithm that tracks changes in graphical model structure or parameters in a dynamic, real-time manner. We show by simulation that the algorithm performs comparably to batch-mode learning when the generating graphical structure is globally stationary, and significantly better when it is only locally stationary.

count=2
* Trimmed Density Ratio Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/ea204361fe7f024b130143eb3e189a18-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/ea204361fe7f024b130143eb3e189a18-Paper.pdf)]
    * Title: Trimmed Density Ratio Estimation
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Song Liu, Akiko Takeda, Taiji Suzuki, Kenji Fukumizu
    * Abstract: Density ratio estimation is a vital tool in both machine learning and statistical community. However, due to the unbounded nature of density ratio, the estimation proceudre can be vulnerable to corrupted data points, which often pushes the estimated ratio toward infinity. In this paper, we present a robust estimator which automatically identifies and trims outliers. The proposed estimator has a convex formulation, and the global optimum can be obtained via subgradient descent. We analyze the parameter estimation error of this estimator under high-dimensional settings. Experiments are conducted to verify the effectiveness of the estimator.

count=2
* Limits on Testing Structural Changes in Ising Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/70431e77d378d760c3c5456519f06efe-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/70431e77d378d760c3c5456519f06efe-Paper.pdf)]
    * Title: Limits on Testing Structural Changes in Ising Models
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Aditya Gangrade, Bobak Nazer, Venkatesh Saligrama
    * Abstract: We present novel information-theoretic limits on detecting sparse changes in Isingmodels, a problem that arises in many applications where network changes canoccur due to some external stimuli. We show that the sample complexity fordetecting sparse changes, in a minimax sense, is no better than learning the entiremodel even in settings with local sparsity. This is a surprising fact in light of priorwork rooted in sparse recovery methods, which suggest that sample complexityin this context scales only with the number of network changes. To shed light onwhen change detection is easier than structured learning, we consider testing ofedge deletion in forest-structured graphs, and high-temperature ferromagnets ascase studies. We show for these that testing of small changes is similarly hard, buttesting oflargechanges is well-separated from structure learning. These resultsimply that testing of graphical models may not be amenable to concepts such asrestricted strong convexity leveraged for sparsity pattern recovery, and algorithmdevelopment instead should be directed towards detection of large changes.

count=2
* Locally private online change point detection
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/1c1d4df596d01da60385f0bb17a4a9e0-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/1c1d4df596d01da60385f0bb17a4a9e0-Paper.pdf)]
    * Title: Locally private online change point detection
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Tom Berrett, Yi Yu
    * Abstract: We study online change point detection problems under the constraint of local differential privacy (LDP) where, in particular, the statistician does not have access to the raw data. As a concrete problem, we study a multivariate nonparametric regression problem. At each time point $t$, the raw data are assumed to be of the form $(X_t, Y_t)$, where $X_t$ is a $d$-dimensional feature vector and $Y_t$ is a response variable. Our primary aim is to detect changes in the regression function $m_t(x)=\mathbb{E}(Y_t |X_t=x)$ as soon as the change occurs. We provide algorithms which respect the LDP constraint, which control the false alarm probability, and which detect changes with a minimal (minimax rate-optimal) delay. To quantify the cost of privacy, we also present the optimal rate in the benchmark, non-private setting. These non-private results are also new to the literature and thus are interesting \emph{per se}. In addition, we study the univariate mean online change point detection problem, under privacy constraints. This serves as the blueprint of studying more complicated private change point detection problems.

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
* On the Exploration of Local Significant Differences For Two-Sample Test
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/10fc83943b4540a9524af6fc67a23fef-Paper-Conference.pdf)]
    * Title: On the Exploration of Local Significant Differences For Two-Sample Test
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Zhijian Zhou, Jie Ni, Jia-He Yao, Wei Gao
    * Abstract: Recent years have witnessed increasing attentions on two-sample test with diverse real applications, while this work takes one more step on the exploration of local significant differences for two-sample test. We propose the ME$_\text{MaBiD}$, an effective test for two-sample testing, and the basic idea is to exploit local information by multiple Mahalanobis kernels and introduce bi-directional hypothesis for testing. On the exploration of local significant differences, we first partition the embedding space into several rectangle regions via a new splitting criterion, which is relevant to test power and data correlation. We then explore local significant differences based on our bi-directional masked $p$-value together with the ME$_\text{MaBiD}$ test. Theoretically, we present the asymptotic distribution and lower bounds of test power for our ME$_\text{MaBiD}$ test, and control the familywise error rate on the exploration of local significant differences. We finally conduct extensive experiments to validate the effectiveness of our proposed methods on two-sample test and the exploration of local significant differences.

count=2
* CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf)]
    * Title: CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Anthony Fuller, Koreen Millard, James Green
    * Abstract: A vital and rapidly growing application, remote sensing offers vast yet sparsely labeled, spatially aligned multimodal data; this makes self-supervised learning algorithms invaluable. We present CROMA: a framework that combines contrastive and reconstruction self-supervised objectives to learn rich unimodal and multimodal representations. Our method separately encodes masked-out multispectral optical and synthetic aperture radar samples—aligned in space and time—and performs cross-modal contrastive learning. Another encoder fuses these sensors, producing joint multimodal encodings that are used to predict the masked patches via a lightweight decoder. We show that these objectives are complementary when leveraged on spatially aligned multimodal data. We also introduce X- and 2D-ALiBi, which spatially biases our cross- and self-attention matrices. These strategies improve representations and allow our models to effectively extrapolate to images up to $17.6\times$ larger at test-time. CROMA outperforms the current SoTA multispectral model, evaluated on: four classification benchmarks—finetuning (avg.$\uparrow$ 1.8%), linear (avg.$\uparrow$ 2.4%) and nonlinear (avg.$\uparrow$ 1.4%) probing, $k$NN classification (avg.$\uparrow$ 3.5%), and $K$-means clustering (avg.$\uparrow$ 8.4%); and three segmentation benchmarks (avg.$\uparrow$ 6.4%). CROMA’s rich, optionally multimodal representations can be widely leveraged across remote sensing applications.

count=2
* Tracking Most Significant Shifts in Nonparametric Contextual Bandits
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/13b501c58ae3bfe9635a259f4414e943-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/13b501c58ae3bfe9635a259f4414e943-Paper-Conference.pdf)]
    * Title: Tracking Most Significant Shifts in Nonparametric Contextual Bandits
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Joe Suk, Samory Kpotufe
    * Abstract: We study nonparametric contextual bandits where Lipschitz mean reward functions may change over time.We first establish the minimax dynamic regret rate in this less understood setting in terms of number of changes $L$ and total-variation $V$, both capturing all changes in distribution over context space, and argue that state-of-the-art procedures are suboptimal in this setting.Next, we tend to the question of an _adaptivity_ for this setting, i.e. achieving the minimax rate without knowledge of $L$ or $V$. Quite importantly, we posit that the bandit problem, viewed locally at a given context $X_t$, should not be affected by reward changes in other parts of context space $\cal X$. We therefore propose a notion of _change_, which we term _experienced significant shifts_, that better accounts for locality, and thus counts considerably less changes than $L$ and $V$. Furthermore, similar to recent work on non-stationary MAB (Suk & Kpotufe, 2022), _experienced significant shifts_ only count the most _significant_ changes in mean rewards, e.g., severe best-arm changes relevant to observed contexts.Our main result is to show that this more tolerant notion of change can in fact be adapted to.

count=2
* Non-Stationary Bandits with Auto-Regressive Temporal Dependency
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/186a213d720568b31f9b59c085a23e5a-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/186a213d720568b31f9b59c085a23e5a-Paper-Conference.pdf)]
    * Title: Non-Stationary Bandits with Auto-Regressive Temporal Dependency
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Qinyi Chen, Negin Golrezaei, Djallel Bouneffouf
    * Abstract: Traditional multi-armed bandit (MAB) frameworks, predominantly examined under stochastic or adversarial settings, often overlook the temporal dynamics inherent in many real-world applications such as recommendation systems and online advertising. This paper introduces a novel non-stationary MAB framework that captures the temporal structure of these real-world dynamics through an auto-regressive (AR) reward structure. We propose an algorithm that integrates two key mechanisms: (i) an alternation mechanism adept at leveraging temporal dependencies to dynamically balance exploration and exploitation, and (ii) a restarting mechanism designed to discard out-of-date information. Our algorithm achieves a regret upper bound that nearly matches the lower bound, with regret measured against a robust dynamic benchmark. Finally, via a real-world case study on tourism demand prediction, we demonstrate both the efficacy of our algorithm and the broader applicability of our techniques to more complex, rapidly evolving time series.

count=2
* FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/353ca88f722cdd0c481b999428ae113a-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/353ca88f722cdd0c481b999428ae113a-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Anatol Garioud, Nicolas Gonthier, Loic Landrieu, Apolline De Wit, Marion Valette, Marc Poupée, Sebastien Giordano, boris Wattrelos
    * Abstract: We introduce the French Land cover from Aerospace ImageRy (FLAIR), an extensive dataset from the French National Institute of Geographical and Forest Information (IGN) that provides a unique and rich resource for large-scale geospatial analysis. FLAIR contains high-resolution aerial imagery with a ground sample distance of 20 cm and over 20 billion individually labeled pixels for precise land-cover classification. The dataset also integrates temporal and spectral data from optical satellite time series. FLAIR thus combines data with varying spatial, spectral, and temporal resolutions across over 817 km² of acquisitions representing the full landscape diversity of France. This diversity makes FLAIR a valuable resource for the development and evaluation of novel methods for large-scale land-cover semantic segmentation and raises significant challenges in terms of computer vision, data fusion, and geospatial analysis. We also provide powerful uni- and multi-sensor baseline models that can be employed to assess algorithm's performance and for downstream applications.

count=2
* On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/57587d8d6a7ede0e5302fc22d0878c53-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/57587d8d6a7ede0e5302fc22d0878c53-Paper-Conference.pdf)]
    * Title: On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Qi CHEN, Changjian Shui, Ligong Han, Mario Marchand
    * Abstract: We focus on Continual Meta-Learning (CML), which targets accumulating and exploiting meta-knowledge on a sequence of non-i.i.d. tasks. The primary challenge is to strike a balance between stability and plasticity, where a model should be stable to avoid catastrophic forgetting in previous tasks and plastic to learn generalizable concepts from new tasks. To address this, we formulate the CML objective as controlling the average excess risk upper bound of the task sequence, which reflects the trade-off between forgetting and generalization. Based on the objective, we introduce a unified theoretical framework for CML in both static and shifting environments, providing guarantees for various task-specific learning algorithms. Moreover, we first present a rigorous analysis of a bi-level trade-off in shifting environments. To approach the optimal trade-off, we propose a novel algorithm that dynamically adjusts the meta-parameter and its learning rate w.r.t environment change. Empirical evaluations on synthetic and real datasets illustrate the effectiveness of the proposed theory and algorithm.

count=2
* Non-stationary Experimental Design under Linear Trends
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/65e837e76a5308df3d5544aab6196e21-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/65e837e76a5308df3d5544aab6196e21-Paper-Conference.pdf)]
    * Title: Non-stationary Experimental Design under Linear Trends
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: David Simchi-Levi, Chonghuan Wang, Zeyu Zheng
    * Abstract: Experimentation has been critical and increasingly popular across various domains, such as clinical trials and online platforms, due to its widely recognized benefits. One of the primary objectives of classical experiments is to estimate the average treatment effect (ATE) to inform future decision-making. However, in healthcare and many other settings, treatment effects may be non-stationary, meaning that they can change over time, rendering the traditional experimental design inadequate and the classical static ATE uninformative. In this work, we address the problem of non-stationary experimental design under linear trends by considering two objectives: estimating the dynamic treatment effect and minimizing welfare loss within the experiment. We propose an efficient design that can be customized for optimal estimation error rate, optimal regret rate, or the Pareto optimal trade-off between the two objectives. We establish information-theoretical lower bounds that highlight the inherent challenge in estimating dynamic treatment effects and minimizing welfare loss, and also statistically reveal the fundamental trade-off between them.

count=1
* UTB180: A High-quality Benchmark for Underwater Tracking
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2022/html/Alawode_UTB180_A_High-quality_Benchmark_for_Underwater_Tracking_ACCV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2022/papers/Alawode_UTB180_A_High-quality_Benchmark_for_Underwater_Tracking_ACCV_2022_paper.pdf)]
    * Title: UTB180: A High-quality Benchmark for Underwater Tracking
    * Publisher: ACCV
    * Publication Date: `2022`
    * Authors: Basit Alawode, Yuhang Guo, Mehnaz Ummar, Naoufel Werghi, Jorge Dias, Ajmal Mian, Sajid Javed
    * Abstract: Deep learning methods have demonstrated encouraging performance on open-air visual object tracking (VOT) benchmarks, however, their strength remains unexplored on underwater video sequences due to the lack of challenging underwater VOT benchmarks. Apart from the challenges of open-air tracking, videos captured in underwater environments pose additional challenges for tracking such as low visibility, poor video quality, distortions in sharpness and contrast, reflections from suspended particles, and non-uniform lighting. We propose a new underwater tracking benchmark dataset (UTB180) consisting of 180 sequences to facilitate the development of underwater deep trackers. The sequences in UTB180 are selected from both underwater natural and online sources with over 58,000 annotated frames. Video-level attributes are also provided to facilitate the development of robust trackers for specific challenges. We benchmark 15 existing pre-trained State-Of-The-Art (SOTA) trackers on UTB180 and compare their performance on another publicly available underwater benchmark. The trackers consistently perform worse on UTB180 showing that it poses more challenging scenarios. Moreover, we show that fine-tuning five high-quality SOTA trackers on UTB180 still does not sufficiently boost their tracking performance. Our experiments show that the UTB180 sequences pose a major burden on the SOTA trackers as compared to their open-air tracking performance. The performance gap reveals the need for a dedicated end-to-end underwater deep tracker that takes into account the inherent properties of underwater environments. We believe that our proposed dataset will be of great value to the tracking community in advancing the state of the art in underwater VOT. Our dataset is publicly available on Kaggle.

count=1
* Chinese Character Component Segmentation Based on Character Structure Masks
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Li_Chinese_Character_Component_Segmentation_Based_on_Character_Structure_Masks_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Li_Chinese_Character_Component_Segmentation_Based_on_Character_Structure_Masks_ACCV_2024_paper.pdf)]
    * Title: Chinese Character Component Segmentation Based on Character Structure Masks
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Haiyan Li, Fang Yang
    * Abstract: To address the issue where rectangular anchor boxes in object detection-based Chinese character component segmentation cannot segment semi-enclosed Chinese characters, this paper proposes a method for segmenting Chinese character components based on Chinese character structure masks. This method utilizes a U-Net encoder with ResNet as the backbone network, transforming the segmentation of Chinese character components into the generation of Chinese character structure masks. First, this study proposes a Res-CBAM module, which leverages the structural features of Chinese characters by incorporating CBAM into the residual U-Net network, effectively solving the problem of incomplete segmentation of Chinese character components. Secondly, a vector-guided supervision mechanism is designed to guide the training process of the model by designing structure vectors of Chinese characters, effectively addressing the issue of component adhesion in Chinese characters. Experimental results demonstrate that compared to traditional object detection methods, this method can achieve fast and efficient segmentation in lightweight networks by training small datasets.

count=1
* Decoupled DETR For Few-shot Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Shangguan_Decoupled_DETR_For_Few-shot_Object_Detection_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Shangguan_Decoupled_DETR_For_Few-shot_Object_Detection_ACCV_2024_paper.pdf)]
    * Title: Decoupled DETR For Few-shot Object Detection
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Zeyu Shangguan, Lian Huai, Tong Liu, Yuyu Liu, Xingqun Jiang
    * Abstract: The efficient technique for dealing with severe data-hungry issues in object detection, known as Few-shot object detection (FSOD), has been widely explored. However, FSOD encounters some notable challenges such as the model's natural bias towards pre-training data and the inherent defects present in the existing models. In this paper, we introduce improved methods for the FSOD problem based on DETR structures: (i) To reduce bias from pre-training classes (i.e. many-shot base classes), we investigate the impact of decoupling the parameters of pre-training classes and fine-tuning classes (i.e. few-shot novel classes) in various ways. As a result, we propose a "base-novel categories decoupled DETR (DeDETR)" network for FSOD. (ii) To further improve the efficiency of the DETR's skip connection structure, we explore varied skip connection types in the DETR's encoder and decoder. Subsequently, we introduce a unified decoder module that dynamically blends decoder layers to generate the output feature. Our model's effectiveness is evaluated using PASCAL VOC and MSCOCO datasets. Our results indicate that our proposed module consistently improves performance by 5% to 10% in both fine-tuning and meta-learning frameworks and has surpassed the top scores achieved in recent studies.

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
* Modeling Actions through State Changes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Fathi_Modeling_Actions_through_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Fathi_Modeling_Actions_through_2013_CVPR_paper.pdf)]
    * Title: Modeling Actions through State Changes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Alireza Fathi, James M. Rehg
    * Abstract: In this paper we present a model of action based on the change in the state of the environment. Many actions involve similar dynamics and hand-object relationships, but differ in their purpose and meaning. The key to differentiating these actions is the ability to identify how they change the state of objects and materials in the environment. We propose a weakly supervised method for learning the object and material states that are necessary for recognizing daily actions. Once these state detectors are learned, we can apply them to input videos and pool their outputs to detect actions. We further demonstrate that our method can be used to segment discrete actions from a continuous video of an activity. Our results outperform state-of-the-art action recognition and activity segmentation results.

count=1
* Exploring Weak Stabilization for Motion Feature Extraction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Park_Exploring_Weak_Stabilization_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Park_Exploring_Weak_Stabilization_2013_CVPR_paper.pdf)]
    * Title: Exploring Weak Stabilization for Motion Feature Extraction
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Dennis Park, C. L. Zitnick, Deva Ramanan, Piotr Dollar
    * Abstract: We describe novel but simple motion features for the problem of detecting objects in video sequences. Previous approaches either compute optical flow or temporal differences on video frame pairs with various assumptions about stabilization. We describe a combined approach that uses coarse-scale flow and fine-scale temporal difference features. Our approach performs weak motion stabilization by factoring out camera motion and coarse object motion while preserving nonrigid motions that serve as useful cues for recognition. We show results for pedestrian detection and human pose estimation in video sequences, achieving state-of-the-art results in both. In particular, given a fixed detection rate our method achieves a five-fold reduction in false positives over prior art on the Caltech Pedestrian benchmark. Finally, we perform extensive diagnostic experiments to reveal what aspects of our system are crucial for good performance. Proper stabilization, long time-scale features, and proper normalization are all critical.

count=1
* Robust Real-Time Tracking of Multiple Objects by Volumetric Mass Densities
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Possegger_Robust_Real-Time_Tracking_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Possegger_Robust_Real-Time_Tracking_2013_CVPR_paper.pdf)]
    * Title: Robust Real-Time Tracking of Multiple Objects by Volumetric Mass Densities
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Horst Possegger, Sabine Sternig, Thomas Mauthner, Peter M. Roth, Horst Bischof
    * Abstract: Combining foreground images from multiple views by projecting them onto a common ground-plane has been recently applied within many multi-object tracking approaches. These planar projections introduce severe artifacts and constrain most approaches to objects moving on a common 2D ground-plane. To overcome these limitations, we introduce the concept of an occupancy volume exploiting the full geometry and the objects' center of mass and develop an efficient algorithm for 3D object tracking. Individual objects are tracked using the local mass density scores within a particle filter based approach, constrained by a Voronoi partitioning between nearby trackers. Our method benefits from the geometric knowledge given by the occupancy volume to robustly extract features and train classifiers on-demand, when volumetric information becomes unreliable. We evaluate our approach on several challenging real-world scenarios including the public APIDIS dataset. Experimental evaluations demonstrate significant improvements compared to state-of-theart methods, while achieving real-time performance.

count=1
* Human vs. Computer in Scene and Object Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Borji_Human_vs._Computer_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Borji_Human_vs._Computer_2014_CVPR_paper.pdf)]
    * Title: Human vs. Computer in Scene and Object Recognition
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Ali Borji, Laurent Itti
    * Abstract: Several decades of research in computer and primate vision have resulted in many models (some specialized for one problem, others more general) and invaluable experimental data. Here, to help focus research efforts onto the hardest unsolved problems, and bridge computer and human vision, we define a battery of 5 tests that measure the gap between human and machine performances in several dimensions (generalization across scene categories, generalization from images to edge maps and line drawings, invariance to rotation and scaling, local/global information with jumbled images, and object recognition performance). We measure model accuracy and the correlation between model and human error patterns. Experimenting over 7 datasets, where human data is available, and gauging 14 well-established models, we find that none fully resembles humans in all aspects, and we learn from each test which models and features are more promising in approaching humans in the tested dimension. Across all tests, we find that models based on local edge histograms consistently resemble humans more, while several scene statistics or "gist" models do perform well with both scenes and objects. While computer vision has long been inspired by human vision, we believe systematic efforts, such as this, will help better identify shortcomings of models and find new paths forward.

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
* Deep Sparse Representation for Robust Image Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Li_Deep_Sparse_Representation_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_Deep_Sparse_Representation_2015_CVPR_paper.pdf)]
    * Title: Deep Sparse Representation for Robust Image Registration
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Yeqing Li, Chen Chen, Fei Yang, Junzhou Huang
    * Abstract: The definition of the similarity measure is an essential component in image registration. In this paper, we propose a novel similarity measure for registration of two or more images. The proposed method is motivated by that the optimally registered images can be deeply sparsified in the gradient domain and frequency domain, with the separation of a sparse tensor of errors. One of the key advantages of the proposed similarity measure is its robustness to severe intensity distortions, which widely exist on medical images, remotely sensed images and natural photos due to the difference of acquisition modalities or illumination conditions. Two efficient algorithms are proposed to solve the batch image registration and pair registration problems in a unified framework. We validate our method on extensive challenging datasets. The experimental results demonstrate the robustness, accuracy and efficiency of our method over 9 traditional and state-of-the-art algorithms on synthetic images and a wide range of real-world applications.

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
* Solving Temporal Puzzles
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Dicle_Solving_Temporal_Puzzles_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Dicle_Solving_Temporal_Puzzles_CVPR_2016_paper.pdf)]
    * Title: Solving Temporal Puzzles
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Caglayan Dicle, Burak Yilmaz, Octavia Camps, Mario Sznaier
    * Abstract: Many physical phenomena, within short time windows, can be explained by low order differential relations. In a discrete world, these relations can be described using low order difference equations or equivalently low order auto regressive (AR) models. In this paper, based on this intuition, we propose an algorithm for solving time-sort temporal puzzles, defined as scrambled time series that need to be sorted out. We frame this highly combinatorial problem using a mixed-integer semi definite programming formulation and show how to turn it into a mixed-integer linear programming problem by using the recently introduced atomic norm framework. Our experiments show the effectiveness and generality of our approach in different scenarios.

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
* A Holistic Approach to Cross-Channel Image Noise Modeling and Its Application to Image Denoising
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Nam_A_Holistic_Approach_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Nam_A_Holistic_Approach_CVPR_2016_paper.pdf)]
    * Title: A Holistic Approach to Cross-Channel Image Noise Modeling and Its Application to Image Denoising
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Seonghyeon Nam, Youngbae Hwang, Yasuyuki Matsushita, Seon Joo Kim
    * Abstract: Modelling and analyzing noise in images is a fundamental task in many computer vision systems. Traditionally, noise has been modelled per color channel assuming that the color channels are independent. Although the color channels can be considered as mutually independent in camera RAW images, signals from different color channels get mixed during the imaging process inside the camera due to gamut mapping, tone-mapping, and compression. We show the influence of the in-camera imaging pipeline on noise and propose a new noise model in the 3D RGB space to accounts for the color channel mix-ups. A data-driven approach for determining the parameters of the new noise model is introduced as well as its application to image denoising. The experiments show that our noise model represents the noise in regular JPEG images more accurately compared to the previous models and is advantageous in image denoising.

count=1
* The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.pdf)]
    * Title: The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, Antonio M. Lopez
    * Abstract: Vision-based semantic segmentation in urban scenarios is a key functionality for autonomous driving. Recent revolutionary results of deep convolutional neural networks (DCNNs) foreshadow the advent of reliable classifiers to perform such visual tasks. However, DCNNs require learning of many parameters from raw images; thus, having a sufficient amount of diverse images with class annotations is needed. These annotations are obtained via cumbersome, human labour which is particularly challenging for semantic segmentation since pixel-level annotations are required. In this paper, we propose to use a virtual world to automatically generate realistic synthetic images with pixel-level annotations. Then, we address the question of how useful such data can be for semantic segmentation -- in particular, when using a DCNN paradigm. In order to answer this question we have generated a synthetic collection of diverse urban images, named SYNTHIA, with automatically generated class annotations. We use SYNTHIA in combination with publicly available real-world urban images with manually provided annotations. Then, we conduct experiments with DCNNs that show how the inclusion of SYNTHIA in the training stage significantly improves performance on the semantic segmentation task.

count=1
* Semantic 3D Reconstruction With Continuous Regularization and Ray Potentials Using a Visibility Consistency Constraint
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Savinov_Semantic_3D_Reconstruction_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Savinov_Semantic_3D_Reconstruction_CVPR_2016_paper.pdf)]
    * Title: Semantic 3D Reconstruction With Continuous Regularization and Ray Potentials Using a Visibility Consistency Constraint
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Nikolay Savinov, Christian Hane, Lubor Ladicky, Marc Pollefeys
    * Abstract: We propose an approach for dense semantic 3D reconstruction which uses a data term that is defined as potentials over viewing rays, combined with continuous surface area penalization. Our formulation is a convex relaxation which we augment with a crucial non-convex constraint that ensures exact handling of visibility. To tackle the non-convex minimization problem, we propose a majorize-minimize type strategy which converges to a critical point. We demonstrate the benefits of using the non-convex constraint experimentally. For the geometry-only case, we set a new state of the art on two datasets of the commonly used Middlebury multi-view stereo benchmark. Moreover, our general-purpose formulation directly reconstructs thin objects, which are usually treated with specialized algorithms. A qualitative evaluation on the dense semantic 3D reconstruction task shows that we improve significantly over previous methods.

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
* Embedded Vision System for Atmospheric Turbulence Mitigation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/html/Deshmukh_Embedded_Vision_System_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/papers/Deshmukh_Embedded_Vision_System_CVPR_2016_paper.pdf)]
    * Title: Embedded Vision System for Atmospheric Turbulence Mitigation
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Ajinkya Deshmukh, Gaurav Bhosale, Swarup Shanti, Karthik Reddy, Hemanthkumar P., Chandrasekhar A., Kirankumar P., Vijaysagar K.
    * Abstract: Outdoor surveillance systems that involve farfield operations often encounter atmospheric turbulence perturbations due to a series of randomized reflections and refraction effecting incoming light rays. The resulting distortions make it hard to discriminate between true moving objects and turbulence induced motion. Current algorithms are not effective in detecting true moving objects in the scene and also rely on computationally complex warping methods. In this paper, we describe a real time embedded solution connected with traditional cameras to both rectify turbulence distortions and reliably detect and track true moving targets. Our comparisons with other methods shows better turbulence rectification with less false and miss detections. FPGA-DSP based embedded realization of our algorithm achieves nearly 15x speed-up along with lesser memory requirement over a quad core PC implementation. The proposed system is suitable for persistence surveillance systems and optical sight devices.

count=1
* Non-Planar Infrared-Visible Registration for Uncalibrated Stereo Pairs
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w9/html/Nguyen_Non-Planar_Infrared-Visible_Registration_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w9/papers/Nguyen_Non-Planar_Infrared-Visible_Registration_CVPR_2016_paper.pdf)]
    * Title: Non-Planar Infrared-Visible Registration for Uncalibrated Stereo Pairs
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Dinh-Luan Nguyen, Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau
    * Abstract: Thermal infrared-visible video registration for non-planar scenes is a new area in visual surveillance. It allows the combination of information from two spectra for better human detection and segmentation. In this paper, we present a novel online framework for visible and thermal infrared registration in non-planar scenes that includes foreground segmentation, feature matching, rectification and disparity calculation. Our proposed approach is based on sparse correspondences of contour points. The key ideas of the proposed framework are the removal of spurious regions at the beginning of videos and a registration methodology for non-planar scenes. Besides, a new non-planar dataset with an associated evaluation protocol is also proposed as a standard assessment. We evaluate our method on both public planar and non-planar datasets. Experimental results reveal that the proposed method can not only successfully handle non-planar scenes but also gets state-of-the-art results on planar ones.

count=1
* A Wide-Field-Of-View Monocentric Light Field Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Dansereau_A_Wide-Field-Of-View_Monocentric_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dansereau_A_Wide-Field-Of-View_Monocentric_CVPR_2017_paper.pdf)]
    * Title: A Wide-Field-Of-View Monocentric Light Field Camera
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Donald G. Dansereau, Glenn Schuster, Joseph Ford, Gordon Wetzstein
    * Abstract: Light field (LF) capture and processing are important in an expanding range of computer vision applications, offering rich textural and depth information and simplification of conventionally complex tasks. Although LF cameras are commercially available, no existing device offers wide field-of-view (FOV) imaging. This is due in part to the limitations of fisheye lenses, for which a fundamentally constrained entrance pupil diameter severely limits depth sensitivity. In this work we describe a novel, compact optical design that couples a monocentric lens with multiple sensors using microlens arrays, allowing LF capture with an unprecedented FOV. Leveraging capabilities of the LF representation, we propose a novel method for efficiently coupling the spherical lens and planar sensors, replacing expensive and bulky fiber bundles. We construct a single-sensor LF camera prototype, rotating the sensor relative to a fixed main lens to emulate a wide-FOV multi-sensor scenario. Finally, we describe a processing toolchain, including a convenient spherical LF parameterization, and demonstrate depth estimation and post-capture refocus for indoor and outdoor panoramas with 15 x 15 x 1600 x 200 pixels (72 MPix) and a 138-degree FOV.

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
* Dynamic Time-Of-Flight
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Schober_Dynamic_Time-Of-Flight_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Schober_Dynamic_Time-Of-Flight_CVPR_2017_paper.pdf)]
    * Title: Dynamic Time-Of-Flight
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Michael Schober, Amit Adam, Omer Yair, Shai Mazor, Sebastian Nowozin
    * Abstract: Time-of-flight (TOF) depth cameras provide robust depth inference at low power requirements in a wide variety of consumer and industrial applications. These cameras reconstruct a single depth frame from a given set of infrared (IR) frames captured over a very short exposure period. Operating in this mode the camera essentially forgets all information previously captured - and performs depth inference from scratch for every frame. We challenge this practice and propose using previously captured information when inferring depth. An inherent problem we have to address is camera motion over this longer period of collecting observations. We derive a probabilistic framework combining a simple but robust model of camera and object motion, together with an observation model. This combination allows us to integrate information over multiple frames while remaining robust to rapid changes. Operating the camera in this manner has implications in terms of both computational efficiency and how information should be captured. We address these two issues and demonstrate a realtime TOF system with robust temporal integration that improves depth accuracy over strong baseline methods including adaptive spatio-temporal filters.

count=1
* Semantic Multi-View Stereo: Jointly Estimating Objects and Voxels
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Ulusoy_Semantic_Multi-View_Stereo_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ulusoy_Semantic_Multi-View_Stereo_CVPR_2017_paper.pdf)]
    * Title: Semantic Multi-View Stereo: Jointly Estimating Objects and Voxels
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Ali Osman Ulusoy, Michael J. Black, Andreas Geiger
    * Abstract: Dense 3D reconstruction from RGB images is a highly ill-posed problem due to occlusions, textureless or reflective surfaces, as well as other challenges. We propose object-level shape priors to address these ambiguities. Towards this goal, we formulate a probabilistic model that integrates multi-view image evidence with 3D shape information from multiple objects. Inference in this model yields a dense 3D reconstruction of the scene as well as the existence and precise 3D pose of the objects in it. Our approach is able to recover fine details not captured in the input shapes while defaulting to the input models in occluded regions where image evidence is weak. Due to its probabilistic nature, the approach is able to cope with the approximate geometry of the 3D models as well as input shapes that are not present in the scene. We evaluate the approach quantitatively on several challenging indoor and outdoor datasets.

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
* Turning an Urban Scene Video Into a Cinemagraph
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Yan_Turning_an_Urban_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Turning_an_Urban_CVPR_2017_paper.pdf)]
    * Title: Turning an Urban Scene Video Into a Cinemagraph
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Hang Yan, Yebin Liu, Yasutaka Furukawa
    * Abstract: This paper proposes an algorithm that turns a regular video capturing urban scenes into a high-quality endless animation, known as a Cinemagraph. The creation of a Cinemagraph usually requires a static camera in a carefully configured scene. The task becomes challenging for a regular video with a moving camera and objects. Our approach first warps an input video into the viewpoint of a reference camera. Based on the warped video, we propose effective temporal analysis algorithms to detect regions with static geometry and dynamic appearance, where geometric modeling is reliable and visually attractive animations can be created. Lastly, the algorithm applies a sequence of video processing techniques to produce a Cinemagraph movie. We have tested the proposed approach on numerous challenging real scenes. To our knowledge, this work is the first to automatically generate Cinemagraph animations from regular movies in the wild.

count=1
* Filmy Cloud Removal on Satellite Imagery With Multispectral Conditional Generative Adversarial Nets
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Enomoto_Filmy_Cloud_Removal_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Enomoto_Filmy_Cloud_Removal_CVPR_2017_paper.pdf)]
    * Title: Filmy Cloud Removal on Satellite Imagery With Multispectral Conditional Generative Adversarial Nets
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Kenji Enomoto, Ken Sakurada, Weimin Wang, Hiroshi Fukui, Masashi Matsuoka, Ryosuke Nakamura, Nobuo Kawaguchi
    * Abstract: This paper proposes a method for cloud removal from visible light RGB satellite images by extending the conditional Generative Adversarial Networks (cGANs) from RGB images to multispectral images. The networks are trained to output images that are close to the ground truth with the input images synthesized with clouds on the ground truth images. In the available dataset, the ratio of images of the forest or the sea is very high, which will cause bias of the training dataset if we uniformly sample from the dataset. Thus, we utilize the t-Distributed Stochastic Neighbor Embedding (t-SNE) to improve the bias problem of the training dataset. Finally, we confirm the feasibility of the proposed networks on the dataset of 4 bands images including three visible light bands and one near-infrared (NIR) band.

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
* Unsupervised Human Action Detection by Action Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/html/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/papers/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.pdf)]
    * Title: Unsupervised Human Action Detection by Action Matching
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Basura Fernando, Sareh Shirazi, Stephen Gould
    * Abstract: We propose a new task of unsupervised action detection by action matching. Given two long videos, the objective is to temporally detect all pairs of matching video segments. A pair of video segments are matched if they share the same human action. The task is category independent---it does not matter what action is being performed---and no supervision is used to discover such video segments. Unsupervised action detection by action matching allows us to align videos in a meaningful manner. As such, it can be used to discover new action categories or as an action proposal technique within, say, an action detection pipeline. We solve this new task using an effective and efficient method. We use an unsupervised temporal encoding method and exploit the temporal consistency in human actions to obtain candidate action segments. We evaluate our method on this challenging task using three activity recognition benchmarks.

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
* RayNet: Learning Volumetric 3D Reconstruction With Ray Potentials
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.pdf)]
    * Title: RayNet: Learning Volumetric 3D Reconstruction With Ray Potentials
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Despoina Paschalidou, Osman Ulusoy, Carolin Schmitt, Luc Van Gool, Andreas Geiger
    * Abstract: In this paper, we consider the problem of reconstructing a dense 3D model using images captured from different views. Recent methods based on convolutional neural networks (CNN) allow learning the entire task from data. However, they do not incorporate the physics of image formation such as perspective geometry and occlusion. Instead, classical approaches based on Markov Random Fields (MRF) with ray-potentials explicitly model these physical processes, but they cannot cope with large surface appearance variations across different viewpoints. In this paper, we propose RayNet, which combines the strengths of both frameworks. RayNet integrates a CNN that learns view-invariant feature representations with an MRF that explicitly encodes the physics of perspective projection and occlusion. We train RayNet end-to-end using empirical risk minimization. We thoroughly evaluate our approach on challenging real-world datasets and demonstrate its benefits over a piece-wise trained baseline, hand-crafted models as well as other learning-based approaches.

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
* Now You Shake Me: Towards Automatic 4D Cinema
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_Now_You_Shake_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Now_You_Shake_CVPR_2018_paper.pdf)]
    * Title: Now You Shake Me: Towards Automatic 4D Cinema
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Yuhao Zhou, Makarand Tapaswi, Sanja Fidler
    * Abstract: We are interested in enabling automatic 4D cinema by parsing physical and special effects from untrimmed movies. These include effects such as physical interactions, water splashing, light, and shaking, and are grounded to either a character in the scene or the camera. We collect a new dataset referred to as the Movie4D dataset which annotates over 9K effects in 63 movies. We propose a Conditional Random Field model atop a neural network that brings together visual and audio information, as well as semantics in the form of person tracks. Our model further exploits correlations of effects between different characters in the clip as well as across movie threads. We propose effect detection and classification as two tasks, and present results along with ablation studies on our dataset, paving the way towards 4D cinema in everyone’s homes.

count=1
* Automatic Large-Scale 3D Building Shape Refinement Using Conditional Generative Adversarial Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w36/html/Bittner_Automatic_Large-Scale_3D_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w36/Bittner_Automatic_Large-Scale_3D_CVPR_2018_paper.pdf)]
    * Title: Automatic Large-Scale 3D Building Shape Refinement Using Conditional Generative Adversarial Networks
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Ksenia Bittner, Marco Korner
    * Abstract: Three-dimensional realistic representations of buildings in urban environments have been increasingly applied as data sources in a growing number of remote sensing fields such as urban planning and city management, navigation, environmental simulation (i.e. flood, earthquake, air pollution), 3D change detection after events like natural disasters or conflicts, etc. With recent technological developments, it becomes possible to acquire high-quality 3D input data. There are two main ways to obtain elevation information: from active remote sensing systems, such as light detection and ranging (LIDAR), and from passive remote sensing systems, such as optical images, which allow the acquisition of stereo images for automatic digital surface models (DSMs) generation. Although airborne laser scanning provides very accurate DSMs, it is a costly method. On the other hand, the DSMs from stereo satellite imagery show a large coverage and lower costs. However, they are not as accurate as LIDAR DSMs. With respect to automatic 3D information extraction, the availability of accurate and detailed DSMs is a crucial issue for automatic 3D building model reconstruction. We present a novel methodology for generating a better-quality stereo DSM with refined buildings shapes using a deep learning framework. To this end, a conditional generative adversarial network (cGAN) is trained to generate accurate LIDAR DSM-like height images from noisy stereo DSMs.

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
* Speed Invariant Time Surface for Learning to Detect Corner Points With Event-Based Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Manderscheid_Speed_Invariant_Time_Surface_for_Learning_to_Detect_Corner_Points_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Manderscheid_Speed_Invariant_Time_Surface_for_Learning_to_Detect_Corner_Points_CVPR_2019_paper.pdf)]
    * Title: Speed Invariant Time Surface for Learning to Detect Corner Points With Event-Based Cameras
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jacques Manderscheid,  Amos Sironi,  Nicolas Bourdis,  Davide Migliore,  Vincent Lepetit
    * Abstract: We propose a learning approach to corner detection for event-based cameras that is stable even under fast and abrupt motions. Event-based cameras offer high temporal resolution, power efficiency, and high dynamic range. However, the properties of event-based data are very different compared to standard intensity images, and simple extensions of corner detection methods designed for these images do not perform well on event-based data. We first introduce an efficient way to compute a time surface that is invariant to the speed of the objects. We then show that we can train a Random Forest to recognize events generated by a moving corner from our time surface. Random Forests are also extremely efficient, and therefore a good choice to deal with the high capture frequency of event-based cameras ---our implementation processes up to 1.6Mev/s on a single CPU. Thanks to our time surface formulation and this learning approach, our method is significantly more robust to abrupt changes of direction of the corners compared to previous ones. Our method also naturally assigns a confidence score for the corners, which can be useful for postprocessing. Moreover, we introduce a high-resolution dataset suitable for quantitative evaluation and comparison of corner detection methods for event-based cameras. We call our approach SILC, for Speed Invariant Learned Corners, and compare it to the state-of-the-art with extensive experiments, showing better performance.

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
* EventNet: Asynchronous Recursive Event Processing
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Sekikawa_EventNet_Asynchronous_Recursive_Event_Processing_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sekikawa_EventNet_Asynchronous_Recursive_Event_Processing_CVPR_2019_paper.pdf)]
    * Title: EventNet: Asynchronous Recursive Event Processing
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yusuke Sekikawa,  Kosuke Hara,  Hideo Saito
    * Abstract: Event cameras are bio-inspired vision sensors that mimic retinas to asynchronously report per-pixel intensity changes rather than outputting an actual intensity image at regular intervals. This new paradigm of image sensor offers significant potential advantages; namely, sparse and non-redundant data representation. Unfortunately, however, most of the existing artificial neural network architectures, such as a CNN, require dense synchronous input data, and therefore, cannot make use of the sparseness of the data. We propose EventNet, a neural network designed for real-time processing of asynchronous event streams in a recursive and event-wise manner. EventNet models dependence of the output on tens of thousands of causal events recursively using a novel temporal coding scheme. As a result, at inference time, our network operates in an event-wise manner that is realized with very few sum-of-the-product operations---look-up table and temporal feature aggregation---which enables processing of 1 mega or more events per second on standard CPU. In experiments using real data, we demonstrated the real-time performance and robustness of our framework.

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
* Privacy Protection in Street-View Panoramas Using Depth and Multi-View Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Uittenbogaard_Privacy_Protection_in_Street-View_Panoramas_Using_Depth_and_Multi-View_Imagery_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Uittenbogaard_Privacy_Protection_in_Street-View_Panoramas_Using_Depth_and_Multi-View_Imagery_CVPR_2019_paper.pdf)]
    * Title: Privacy Protection in Street-View Panoramas Using Depth and Multi-View Imagery
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Ries Uittenbogaard,  Clint Sebastian,  Julien Vijverberg,  Bas Boom,  Dariu M. Gavrila,  Peter H.N. de With
    * Abstract: The current paradigm in privacy protection in street-view images is to detect and blur sensitive information. In this paper, we propose a framework that is an alternative to blurring, which automatically removes and inpaints moving objects (e.g. pedestrians, vehicles) in street-view imagery. We propose a novel moving object segmentation algorithm exploiting consistencies in depth across multiple street-view images that are later combined with the results of a segmentation network. The detected moving objects are removed and inpainted with information from other views, to obtain a realistic output image such that the moving object is not visible anymore. We evaluate our results on a dataset of 1000 images to obtain a peak noise-to-signal ratio (PSNR) and L 1 loss of 27.2 dB and 2.5%, respectively. To assess overall quality, we also report the results of a survey conducted on 35 professionals, asked to visually inspect the images whether object removal and inpainting had taken place. The inpainting dataset will be made publicly available for scientific benchmarking purposes at https://research.cyclomedia.com/.

count=1
* EV-Gait: Event-Based Robust Gait Recognition Using Dynamic Vision Sensors
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_EV-Gait_Event-Based_Robust_Gait_Recognition_Using_Dynamic_Vision_Sensors_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_EV-Gait_Event-Based_Robust_Gait_Recognition_Using_Dynamic_Vision_Sensors_CVPR_2019_paper.pdf)]
    * Title: EV-Gait: Event-Based Robust Gait Recognition Using Dynamic Vision Sensors
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yanxiang Wang,  Bowen Du,  Yiran Shen,  Kai Wu,  Guangrong Zhao,  Jianguo Sun,  Hongkai Wen
    * Abstract: In this paper, we introduce a new type of sensing modality, the Dynamic Vision Sensors (Event Cameras), for the task of gait recognition. Compared with the traditional RGB sensors, the event cameras have many unique advantages such as ultra low resources consumption, high temporal resolution and much larger dynamic range. However, those cameras only produce noisy and asynchronous events of intensity changes rather than frames, where conventional vision-based gait recognition algorithms can't be directly applied. To address this, we propose a new Event-based Gait Recognition (EV-Gait) approach, which exploits motion consistency to effectively remove noise, and uses a deep neural network to recognise gait from the event streams. To evaluate the performance of EV-Gait, we collect two event-based gait datasets, one from real-world experiments and the other by converting the publicly available RGB gait recognition benchmark CASIA-B. Extensive experiments show that EV-Gait can get nearly 96% recognition accuracy in the real-world settings, while on the CASIA-B benchmark it achieves comparable performance with state-of-the-art RGB-based gait recognition approaches.

count=1
* Counting Out Time: Class Agnostic Video Repetition Counting in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.pdf)]
    * Title: Counting Out Time: Class Agnostic Video Repetition Counting in the Wild
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Debidatta Dwibedi,  Yusuf Aytar,  Jonathan Tompson,  Pierre Sermanet,  Andrew Zisserman
    * Abstract: We present an approach for estimating the period with which an action is repeated in a video. The crux of the approach lies in constraining the period prediction module to use temporal self-similarity as an intermediate representation bottleneck that allows generalization to unseen repetitions in videos in the wild. We train this model, called RepNet, with a synthetic dataset that is generated from a large unlabeled video collection by sampling short clips of varying lengths and repeating them with different periods and counts. This combination of synthetic data and a powerful yet constrained model, allows us to predict periods in a class-agnostic fashion. Our model substantially exceeds the state of the art performance on existing periodicity (PERTUBE) and repetition counting (QUVA) benchmarks. We also collect a new challenging dataset called Countix ( 90 times larger than existing datasets) which captures the challenges of repetition counting in real-world videos. Project webpage: https://sites.google.com/view/repnet .

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
* Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.pdf)]
    * Title: Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Vivien Sainte Fare Garnot,  Loic Landrieu,  Sebastien Giordano,  Nesrine Chehata
    * Abstract: Satellite image time series, bolstered by their growing availability, are at the forefront of an extensive effort towards automated Earth monitoring by international institutions. In particular, large-scale control of agricultural parcels is an issue of major political and economic importance. In this regard, hybrid convolutional-recurrent neural architectures have shown promising results for the automated classification of satellite image time series. We propose an alternative approach in which the convolutional layers are advantageously replaced with encoders operating on unordered sets of pixels to exploit the typically coarse resolution of publicly available satellite images. We also propose to extract temporal features using a bespoke neural architecture based on self-attention instead of recurrent networks. We demonstrate experimentally that our method not only outperforms previous state-of-the-art approaches in terms of precision, but also significantly decreases processing time and memory requirements. Lastly, we release a large open-access annotated dataset as a benchmark for future work on satellite image time series.

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
* An End-to-End Edge Aggregation Network for Moving Object Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Patil_An_End-to-End_Edge_Aggregation_Network_for_Moving_Object_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Patil_An_End-to-End_Edge_Aggregation_Network_for_Moving_Object_Segmentation_CVPR_2020_paper.pdf)]
    * Title: An End-to-End Edge Aggregation Network for Moving Object Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Prashant W. Patil,  Kuldeep M. Biradar,  Akshay Dudhane,  Subrahmanyam Murala
    * Abstract: Moving object segmentation in videos (MOS) is a highly demanding task for security-based applications like automated outdoor video surveillance. Most of the existing techniques proposed for MOS are highly depend on fine-tuning a model on the first frame(s) of test sequence or complicated training procedure, which leads to limited practical serviceability of the algorithm. In this paper, the inherent correlation learning-based edge extraction mechanism (EEM) and dense residual block (DRB) are proposed for the discriminative foreground representation. The multi-scale EEM module provides the efficient foreground edge related information (with the help of encoder) to the decoder through skip connection at subsequent scale. Further, the response of the optical flow encoder stream and the last EEM module are embedded in the bridge network. The bridge network comprises of multi-scale residual blocks with dense connections to learn the effective and efficient foreground relevant features. Finally, to generate accurate and consistent foreground object maps, a decoder block is proposed with skip connections from respective multi-scale EEM module feature maps and the subsequent down-sampled response of previous frame output. Specifically, the proposed network does not require any pre-trained models or fine-tuning of the parameters with the initial frame(s) of the test video. The performance of the proposed network is evaluated with different configurations like disjoint, cross-data, and global training-testing techniques. The ablation study is conducted to analyse each model of the proposed network. To demonstrate the effectiveness of the proposed framework, a comprehensive analysis on four benchmark video datasets is conducted. Experimental results show that the proposed approach outperforms the state-of-the-art methods for MOS.

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
* Background Matting: The World Is Your Green Screen
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Sengupta_Background_Matting_The_World_Is_Your_Green_Screen_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sengupta_Background_Matting_The_World_Is_Your_Green_Screen_CVPR_2020_paper.pdf)]
    * Title: Background Matting: The World Is Your Green Screen
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Soumyadip Sengupta,  Vivek Jayaram,  Brian Curless,  Steven M. Seitz,  Ira Kemelmacher-Shlizerman
    * Abstract: We propose a method for creating a matte - the per-pixel foreground color and alpha - of a person by taking photos or videos in an everyday setting with a handheld camera. Most existing matting methods require a green screen background or a manually created trimap to produce a good matte. Automatic, trimap-free methods are appearing, but are not of comparable quality. In our trimap free approach, we ask the user to take an additional photo of the background without the subject at the time of capture. This step requires a small amount of foresight but is far less timeconsuming than creating a trimap. We train a deep network with an adversarial loss to predict the matte. We first train a matting network with a supervised loss on ground truth data with synthetic composites. To bridge the domain gap to real imagery with no labeling, we train another matting network guided by the first network and by a discriminator that judges the quality of composites. We demonstrate results on a wide variety of photos and videos and show significant improvement over the state of the art.

count=1
* Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Warburg_Mapillary_Street-Level_Sequences_A_Dataset_for_Lifelong_Place_Recognition_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Warburg_Mapillary_Street-Level_Sequences_A_Dataset_for_Lifelong_Place_Recognition_CVPR_2020_paper.pdf)]
    * Title: Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Frederik Warburg,  Soren Hauberg,  Manuel Lopez-Antequera,  Pau Gargallo,  Yubin Kuang,  Javier Civera
    * Abstract: Lifelong place recognition is an essential and challenging task in computer vision with vast applications in robust localization and efficient large-scale 3D reconstruction. Progress is currently hindered by a lack of large, diverse, publicly available datasets. We contribute with Mapillary Street-Level Sequences (SLS), a large dataset for urban and suburban place recognition from image sequences. It contains more than 1.6 million images curated from the Mapillary collaborative mapping platform. The dataset is orders of magnitude larger than current data sources, and is designed to reflect the diversities of true lifelong learning. It features images from 30 major cities across six continents, hundreds of distinct cameras, and substantially different viewpoints and capture times, spanning all seasons over a nine year period. All images are geo-located with GPS and compass, and feature high-level attributes such as road type. We propose a set of benchmark tasks designed to push state-of-the-art performance and provide baseline studies. We show that current state-of-the-art methods still have a long way to go, and that the lack of diversity in existing datasets have prevented generalization to new environments. The dataset and benchmarks are available for academic research.

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
* Transformation Driven Visual Reasoning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Hong_Transformation_Driven_Visual_Reasoning_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_Transformation_Driven_Visual_Reasoning_CVPR_2021_paper.pdf)]
    * Title: Transformation Driven Visual Reasoning
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Xin Hong, Yanyan Lan, Liang Pang, Jiafeng Guo, Xueqi Cheng
    * Abstract: This paper defines a new visual reasoning paradigm by introducing an important factor, i.e. transformation. The motivation comes from the fact that most existing visual reasoning tasks, such as CLEVR in VQA, are solely defined to test how well the machine understands the concepts and relations within static settings, like one image. We argue that this kind of state driven visual reasoning approach has limitations in reflecting whether the machine has the ability to infer the dynamics between different states, which has been shown as important as state-level reasoning for human cognition in Piaget's theory. To tackle this problem, we propose a novel transformation driven visual reasoning task. Given both the initial and final states, the target is to infer the corresponding single-step or multi-step transformation, represented as a triplet (object, attribute, value) or a sequence of triplets, respectively. Following this definition, a new dataset namely TRANCE is constructed on the basis of CLEVR, including three levels of settings, i.e. Basic (single-step transformation), Event (multi-step transformation), and View (multi-step transformation with variant views). Experimental results show that the state-of-the-art visual reasoning models perform well on Basic, but are still far from human-level intelligence on Event and View. We believe the proposed new paradigm will boost the development of machine visual reasoning. More advanced methods and real data need to be investigated in this direction. The resource of TVR is available at https://hongxin2019.github.io/TVR.

count=1
* Learning Compositional Representation for 4D Captures With Neural ODE
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.pdf)]
    * Title: Learning Compositional Representation for 4D Captures With Neural ODE
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Boyan Jiang, Yinda Zhang, Xingkui Wei, Xiangyang Xue, Yanwei Fu
    * Abstract: Learning based representation has become the key to the success of many computer vision systems. While many 3D representations have been proposed, it is still an unaddressed problem how to represent a dynamically changing 3D object. In this paper, we introduce a compositional representation for 4D captures, i.e. a deforming 3D object over a temporal span, that disentangles shape, initial state, and motion respectively. Each component is represented by a latent code via a trained encoder. To model the motion, a neural Ordinary Differential Equation (ODE) is trained to update the initial state conditioned on the learned motion code, and a decoder takes the shape code and the updated state code to reconstruct the 3D model at each time stamp. To this end, we propose an Identity Exchange Training (IET) strategy to encourage the network to learn effectively decoupling each component. Extensive experiments demonstrate that the proposed method outperforms existing state-of-the-art deep learning based methods on 4D reconstruction, and significantly improves on various tasks, including motion transfer and completion.

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
* From Shadow Generation To Shadow Removal
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_From_Shadow_Generation_To_Shadow_Removal_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_From_Shadow_Generation_To_Shadow_Removal_CVPR_2021_paper.pdf)]
    * Title: From Shadow Generation To Shadow Removal
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Zhihao Liu, Hui Yin, Xinyi Wu, Zhenyao Wu, Yang Mi, Song Wang
    * Abstract: Shadow removal is a computer-vision task that aims to restore the image content in shadow regions. While almost all recent shadow-removal methods require shadow-free images for training, in ECCV 2020 Le and Samaras introduces an innovative approach without this requirement by cropping patches with and without shadows from shadow images as training samples. However, it is still laborious and time-consuming to construct a large amount of such unpaired patches. In this paper, we propose a new G2R-ShadowNet which leverages shadow generation for weakly-supervised shadow removal by only using a set of shadow images and their corresponding shadow masks for training. The proposed G2R-ShadowNet consists of three sub-networks for shadow generation, shadow removal and refinement, respectively and they are jointly trained in an end-to-end fashion. In particular, the shadow generation sub-net stylises non-shadow regions to be shadow ones, leading to paired data for training the shadow-removal sub-net. Extensive experiments on the ISTD dataset and the Video Shadow Removal dataset show that the proposed G2R-ShadowNet achieves competitive performances against the current state of the arts and outperforms Le and Samaras' patch-based shadow-removal method.

count=1
* DyStaB: Unsupervised Object Segmentation via Dynamic-Static Bootstrapping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_DyStaB_Unsupervised_Object_Segmentation_via_Dynamic-Static_Bootstrapping_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_DyStaB_Unsupervised_Object_Segmentation_via_Dynamic-Static_Bootstrapping_CVPR_2021_paper.pdf)]
    * Title: DyStaB: Unsupervised Object Segmentation via Dynamic-Static Bootstrapping
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yanchao Yang, Brian Lai, Stefano Soatto
    * Abstract: We describe an unsupervised method to detect and segment portions of images of live scenes that, at some point in time, are seen moving as a coherent whole, which we refer to as objects. Our method first partitions the motion field by minimizing the mutual information between segments. Then, it uses the segments to learn object models that can be used for detection in a static image. Static and dynamic models are represented by deep neural networks trained jointly in a bootstrapping strategy, which enables extrapolation to previously unseen objects. While the training process requires motion, the resulting object segmentation network can be used on either static images or videos at inference time. As the volume of seen videos grows, more and more objects are seen moving, priming their detection, which then serves as a regularizer for new objects, turning our method into unsupervised continual learning to segment objects. Our models are compared to the state of the art in both video object segmentation and salient object detection. In the six benchmark datasets tested, our models compare favorably even to those using pixel-level supervision, despite requiring no manual annotation.

count=1
* High-Speed Image Reconstruction Through Short-Term Plasticity for Spiking Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_High-Speed_Image_Reconstruction_Through_Short-Term_Plasticity_for_Spiking_Cameras_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_High-Speed_Image_Reconstruction_Through_Short-Term_Plasticity_for_Spiking_Cameras_CVPR_2021_paper.pdf)]
    * Title: High-Speed Image Reconstruction Through Short-Term Plasticity for Spiking Cameras
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yajing Zheng, Lingxiao Zheng, Zhaofei Yu, Boxin Shi, Yonghong Tian, Tiejun Huang
    * Abstract: Fovea, located in the centre of the retina, is specialized for high-acuity vision. Mimicking the sampling mechanism of the fovea, a retina-inspired camera, named spiking camera, is developed to record the external information with a sampling rate of 40,000 Hz, and outputs asynchronous binary spike streams. Although the temporal resolution of visual information is improved, how to reconstruct the scenes is still a challenging problem. In this paper, we present a novel high-speed image reconstruction model through the short-term plasticity (STP) mechanism of the brain. We derive the relationship between postsynaptic potential regulated by STP and the firing frequency of each pixel. By setting up the STP model at each pixel of the spiking camera, we can infer the scene radiance with the temporal regularity of the spike stream. Moreover, we show that STP can be used to distinguish the static and motion areas and further enhance the reconstruction results. The experimental results show that our methods achieve state-of-the-art performance in both image quality and computing time.

count=1
* MRSCAtt: A Spatio-Channel Attention-Guided Network for Mars Rover Image Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Chakravarthy_MRSCAtt_A_Spatio-Channel_Attention-Guided_Network_for_Mars_Rover_Image_Classification_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Chakravarthy_MRSCAtt_A_Spatio-Channel_Attention-Guided_Network_for_Mars_Rover_Image_Classification_CVPRW_2021_paper.pdf)]
    * Title: MRSCAtt: A Spatio-Channel Attention-Guided Network for Mars Rover Image Classification
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Anirudh S. Chakravarthy, Roshan Roy, Praveen Ravirathinam
    * Abstract: As the exploration of human beings pushes deeper into the galaxy, the classification of images from space and other planets is becoming an increasingly critical task. Image classification on these planetary images can be very challenging due to differences in hue, quality, illumination, and clarity when compared to images captured on Earth. In this work, we try to bridge this gap by developing a deep learning network, MRSCAtt (Mars Rover Spatial and Channel Attention), which jointly uses spatial and channel attention to accurately classify images. We use images taken by NASA's Curiosity rover on Mars as a dataset to show the superiority of our approach by achieving state-of-the-art results with 81.53% test set accuracy on the MSL Surface Dataset, outperforming other methods. To necessitate the use of spatial and channel attention, we perform an ablation study to show the effectiveness of each of the components. We further show robustness of our approach by validating with images taken aboard NASA's recently-landed Perseverance rover.

count=1
* DVS-OUTLAB: A Neuromorphic Event-Based Long Time Monitoring Dataset for Real-World Outdoor Scenarios
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EventVision/html/Bolten_DVS-OUTLAB_A_Neuromorphic_Event-Based_Long_Time_Monitoring_Dataset_for_Real-World_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EventVision/papers/Bolten_DVS-OUTLAB_A_Neuromorphic_Event-Based_Long_Time_Monitoring_Dataset_for_Real-World_CVPRW_2021_paper.pdf)]
    * Title: DVS-OUTLAB: A Neuromorphic Event-Based Long Time Monitoring Dataset for Real-World Outdoor Scenarios
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Tobias Bolten, Regina Pohle-Frohlich, Klaus D. Tonnies
    * Abstract: Neuromorphic vision sensors are biologically inspired devices which differ fundamentally from well known frame-based sensors. Even though developments in this research area are increasing, applications that rely entirely on event cameras are still relatively rare. This becomes particularly clear when considering real outdoor scenarios apart from laboratory conditions. One obstacle to the development of event-based vision applications in this context may be the lack of labeled datasets for algorithm development and evaluation. Therefore we describe a recording setting of a DVS-based long time monitoring of an urban public area and provide labeled DVS data that also contain effects of environmental outdoor influences recorded in this process. We also describe the processing chain used for label generation, as well as results from a performed denoising benchmark utilizing various spatio-temporal event stream filters. The dataset contains almost 7 hours of real world outdoor event-data with approx. 47k labeled regions of interest and can be downloaded at http://dnt.kr.hsnr.de/DVS-OUTLAB/

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
* ISDNet: Integrating Shallow and Deep Networks for Efficient Ultra-High Resolution Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_ISDNet_Integrating_Shallow_and_Deep_Networks_for_Efficient_Ultra-High_Resolution_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_ISDNet_Integrating_Shallow_and_Deep_Networks_for_Efficient_Ultra-High_Resolution_CVPR_2022_paper.pdf)]
    * Title: ISDNet: Integrating Shallow and Deep Networks for Efficient Ultra-High Resolution Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Shaohua Guo, Liang Liu, Zhenye Gan, Yabiao Wang, Wuhao Zhang, Chengjie Wang, Guannan Jiang, Wei Zhang, Ran Yi, Lizhuang Ma, Ke Xu
    * Abstract: The huge burden of computation and memory are two obstacles in ultra-high resolution image segmentation. To tackle these issues, most of the previous works follow the global-local refinement pipeline, which pays more attention to the memory consumption but neglects the inference speed. In comparison to the pipeline that partitions the large image into small local regions, we focus on inferring the whole image directly. In this paper, we propose ISDNet, a novel ultra-high resolution segmentation framework that integrates the shallow and deep networks in a new manner, which significantly accelerates the inference speed while achieving accurate segmentation. To further exploit the relationship between the shallow and deep features, we propose a novel Relational-Aware feature Fusion module, which ensures high performance and robustness of our framework. Extensive experiments on Deepglobe, Inria Aerial, and Cityscapes datasets demonstrate our performance is consistently superior to state-of-the-arts. Specifically, it achieves 73.30 mIoU with a speed of 27.70 FPS on Deepglobe, which is more accurate and 172 x faster than the recent competitor. Code available at https://github.com/cedricgsh/ISDNet.

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
* Modeling sRGB Camera Noise With Normalizing Flows
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Kousha_Modeling_sRGB_Camera_Noise_With_Normalizing_Flows_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Kousha_Modeling_sRGB_Camera_Noise_With_Normalizing_Flows_CVPR_2022_paper.pdf)]
    * Title: Modeling sRGB Camera Noise With Normalizing Flows
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Shayan Kousha, Ali Maleky, Michael S. Brown, Marcus A. Brubaker
    * Abstract: Noise modeling and reduction are fundamental tasks in low-level computer vision. They are particularly important for smartphone cameras relying on small sensors that exhibit visually noticeable noise. There has recently been renewed interest in using data-driven approaches to improve camera noise models via neural networks. These data-driven approaches target noise present in the raw-sensor image before it has been processed by the camera's image signal processor (ISP). Modeling noise in the RAW-rgb domain is useful for improving and testing the in-camera denoising algorithm; however, there are situations where the camera's ISP does not apply denoising or additional denoising is desired when the RAW-rgb domain image is no longer available. In such cases, the sensor noise propagates through the ISP to the final rendered image encoded in standard RGB (sRGB). The nonlinear steps on the ISP culminate in a significantly more complex noise distribution in the sRGB domain and existing raw-domain noise models are unable to capture the sRGB noise distribution. We propose a new sRGB-domain noise model based on normalizing flows that is capable of learning the complex noise distribution found in sRGB images under various ISO levels. Our normalizing flows-based approach outperforms other models by a large margin in noise modeling and synthesis tasks. We also show that image denoisers trained on noisy images synthesized with our noise model outperforms those trained with noise from baselines models.

count=1
* DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.pdf)]
    * Title: DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Mathias Parger, Chengcheng Tang, Christopher D. Twigg, Cem Keskin, Robert Wang, Markus Steinberger
    * Abstract: Convolutional neural network inference on video data requires powerful hardware for real-time processing. Given the inherent coherence across consecutive frames, large parts of a video typically change little. By skipping identical image regions and truncating insignificant pixel updates, computational redundancy can in theory be reduced significantly. However, these theoretical savings have been difficult to translate into practice, as sparse updates hamper computational consistency and memory access coherence; which are key for efficiency on real hardware. With DeltaCNN, we present a sparse convolutional neural network framework that enables sparse frame-by-frame updates to accelerate video inference in practice. We provide sparse implementations for all typical CNN layers and propagate sparse feature updates end-to-end - without accumulating errors over time. DeltaCNN is applicable to all convolutional neural networks without retraining. To the best of our knowledge, we are the first to significantly outperform the dense reference, cuDNN, in practical settings, achieving speedups of up to 7x with only marginal differences in accuracy.

count=1
* Semantic-Aware Domain Generalized Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.pdf)]
    * Title: Semantic-Aware Domain Generalized Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Duo Peng, Yinjie Lei, Munawar Hayat, Yulan Guo, Wen Li
    * Abstract: Deep models trained on source domain lack generalization when evaluated on unseen target domains with different data distributions. The problem becomes even more pronounced when we have no access to target domain samples for adaptation. In this paper, we address domain generalized semantic segmentation, where a segmentation model is trained to be domain-invariant without using any target domain data. Existing approaches to tackle this problem standardize data into a unified distribution. We argue that while such a standardization promotes global normalization, the resulting features are not discriminative enough to get clear segmentation boundaries. To enhance separation between categories while simultaneously promoting domain invariance, we propose a framework including two novel modules: Semantic-Aware Normalization (SAN) and Semantic-Aware Whitening (SAW). Specifically, SAN focuses on category-level center alignment between features from different image styles, while SAW enforces distributed alignment for the already center-aligned features. With the help of SAN and SAW, we encourage both intraclass compactness and inter-class separability. We validate our approach through extensive experiments on widely-used datasets (i.e. GTAV, SYNTHIA, Cityscapes, Mapillary and BDDS). Our approach shows significant improvements over existing state-of-the-art on various backbone networks. Code is available at https://github.com/leolyj/SAN-SAW

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
* Augmentation Invariance and Adaptive Sampling in Semantic Segmentation of Agricultural Aerial Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/AgriVision/html/Tavera_Augmentation_Invariance_and_Adaptive_Sampling_in_Semantic_Segmentation_of_Agricultural_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/AgriVision/papers/Tavera_Augmentation_Invariance_and_Adaptive_Sampling_in_Semantic_Segmentation_of_Agricultural_CVPRW_2022_paper.pdf)]
    * Title: Augmentation Invariance and Adaptive Sampling in Semantic Segmentation of Agricultural Aerial Images
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Antonio Tavera, Edoardo Arnaudo, Carlo Masone, Barbara Caputo
    * Abstract: In this paper, we investigate the problem of Semantic Segmentation for agricultural aerial imagery. We observe that the existing methods used for this task are designed without considering two characteristics of the aerial data: (i) the top-down perspective implies that the model cannot rely on a fixed semantic structure of the scene, because the same scene may be experienced with different rotations of the sensor; (ii) there can be a strong imbalance in the distribution of semantic classes because the relevant objects of the scene may appear at extremely different scales (e.g., a field of crops and a small vehicle). We propose a solution to these problems based on two ideas: (i) we use together a set of suitable augmentation and a consistency loss to guide the model to learn semantic representations that are invariant to the photometric and geometric shifts typical of the top-down perspective (Augmentation Invariance); (ii) we use a sampling method (Adaptive Sampling) that selects the training images based on a measure of pixel-wise distribution of classes and actual network confidence. With an extensive set of experiments conducted on the Agriculture-Vision dataset, we demonstrate that our proposed strategies improve the performance of the current state-of-the-art method.

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
* Hephaestus: A Large Scale Multitask Dataset Towards InSAR Understanding
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Bountos_Hephaestus_A_Large_Scale_Multitask_Dataset_Towards_InSAR_Understanding_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Bountos_Hephaestus_A_Large_Scale_Multitask_Dataset_Towards_InSAR_Understanding_CVPRW_2022_paper.pdf)]
    * Title: Hephaestus: A Large Scale Multitask Dataset Towards InSAR Understanding
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Nikolaos Ioannis Bountos, Ioannis Papoutsis, Dimitrios Michail, Andreas Karavias, Panagiotis Elias, Isaak Parcharidis
    * Abstract: Synthetic Aperture Radar (SAR) data and Interferometric SAR (InSAR) products in particular, are one of the largest sources of Earth Observation data. InSAR provides unique information on diverse geophysical processes and geology, and on the geotechnical properties of man-made structures. However, there are only a limited number of applications that exploit the abundance of InSAR data and deep learning methods to extract such knowledge. The main barrier has been the lack of a large curated and annotated InSAR dataset, which would be costly to create and would require an interdisciplinary team of experts experienced on InSAR data interpretation. In this work, we put the effort to create and make available the first of its kind, manually annotated dataset that consists of 19,919 individual Sentinel-1 interferograms acquired over 44 different volcanoes globally, which are split into 216,106 InSAR patches. The annotated dataset is designed to address different computer vision problems, including volcano state classification, semantic segmentation of ground deformation, detection and classification of atmospheric signals in InSAR imagery, interferogram captioning, text to InSAR generation, and InSAR image quality assessment.

count=1
* Prompt-RSVQA: Prompting Visual Context to a Language Model for Remote Sensing Visual Question Answering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Chappuis_Prompt-RSVQA_Prompting_Visual_Context_to_a_Language_Model_for_Remote_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Chappuis_Prompt-RSVQA_Prompting_Visual_Context_to_a_Language_Model_for_Remote_CVPRW_2022_paper.pdf)]
    * Title: Prompt-RSVQA: Prompting Visual Context to a Language Model for Remote Sensing Visual Question Answering
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Christel Chappuis, Valérie Zermatten, Sylvain Lobry, Bertrand Le Saux, Devis Tuia
    * Abstract: Remote sensing visual question answering (RSVQA) was recently proposed with the aim of interfacing natural language and vision to ease the access of information contained in Earth Observation data for a wide audience, which is granted by simple questions in natural language. The traditional vision/language interface is an embedding obtained by fusing features from two deep models, one processing the image and another the question. Despite the success of early VQA models, it remains difficult to control the adequacy of the visual information extracted by its deep model, which should act as a context regularizing the work of the language model. We propose to extract this context information with a visual model, convert it to text and inject it, i.e. prompt it, into a language model. The language model is therefore responsible to process the question with the visual context, and extract features, which are useful to find the answer. We study the effect of prompting with respect to a black-box visual extractor and discuss the importance of training a visual model producing accurate context.

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
* Detecting and Suppressing Marine Snow for Underwater Visual SLAM
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/IMW/html/Hodne_Detecting_and_Suppressing_Marine_Snow_for_Underwater_Visual_SLAM_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/IMW/papers/Hodne_Detecting_and_Suppressing_Marine_Snow_for_Underwater_Visual_SLAM_CVPRW_2022_paper.pdf)]
    * Title: Detecting and Suppressing Marine Snow for Underwater Visual SLAM
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Lars Martin Hodne, Eirik Leikvoll, Mauhing Yip, Andreas Langeland Teigen, Annette Stahl, Rudolf Mester
    * Abstract: Conventional SLAM methods which work very well in typical above-water situations, are based on detecting keypoints that are tracked between images, from which egomotion and the 3D structure of the scene are estimated. However, in underwater environments with marine snow -- small particles of organic matter which are carried by ocean currents throughout the water column -- keypoint detectors are prone to detect the marine snow particles. As the vast majority of SLAM front ends are sensitive against outliers, and the marine snow acts as severe "motion noise", failure of the regular egomotion and 3D structure estimation is expected. For this reason, we investigate the structure and appearance of marine snow and developed two schemes which classify keypoints into "marine snow" or "clean" based on either the image patches obtained from usual keypoint detectors or the descriptors computed from these patches. This way the subsequent SLAM pipeline is protected against 'false' keypoints. We quantitatively evaluate the performance of our marine snow classifier on both real underwater video scenes as well as on simulated underwater footage that contains marine snow. These simulated image sequences have been created by extracting real marine snow elements from real underwater footage, and subsequently overlaying these on "clean" underwater videos. Qualitative evaluation is also done on a nightime road sequence with snowfall to demonstrate applicability in other areas of autonomy. We furthermore evaluate the performance and the effect of marine snow detection & suppression by integrating the snow suppression module in a full SLAM pipeline based on the pySLAM system.

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
* Token Boosting for Robust Self-Supervised Visual Transformer Pre-Training
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Token_Boosting_for_Robust_Self-Supervised_Visual_Transformer_Pre-Training_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Token_Boosting_for_Robust_Self-Supervised_Visual_Transformer_Pre-Training_CVPR_2023_paper.pdf)]
    * Title: Token Boosting for Robust Self-Supervised Visual Transformer Pre-Training
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Tianjiao Li, Lin Geng Foo, Ping Hu, Xindi Shang, Hossein Rahmani, Zehuan Yuan, Jun Liu
    * Abstract: Learning with large-scale unlabeled data has become a powerful tool for pre-training Visual Transformers (VTs). However, prior works tend to overlook that, in real-world scenarios, the input data may be corrupted and unreliable. Pre-training VTs on such corrupted data can be challenging, especially when we pre-train via the masked autoencoding approach, where both the inputs and masked "ground truth" targets can potentially be unreliable in this case. To address this limitation, we introduce the Token Boosting Module (TBM) as a plug-and-play component for VTs that effectively allows the VT to learn to extract clean and robust features during masked autoencoding pre-training. We provide theoretical analysis to show how TBM improves model pre-training with more robust and generalizable representations, thus benefiting downstream tasks. We conduct extensive experiments to analyze TBM's effectiveness, and results on four corrupted datasets demonstrate that TBM consistently improves performance on downstream tasks.

count=1
* Ambiguous Medical Image Segmentation Using Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Rahman_Ambiguous_Medical_Image_Segmentation_Using_Diffusion_Models_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Rahman_Ambiguous_Medical_Image_Segmentation_Using_Diffusion_Models_CVPR_2023_paper.pdf)]
    * Title: Ambiguous Medical Image Segmentation Using Diffusion Models
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Aimon Rahman, Jeya Maria Jose Valanarasu, Ilker Hacihaliloglu, Vishal M. Patel
    * Abstract: Collective insights from a group of experts have always proven to outperform an individual's best diagnostic for clinical tasks. For the task of medical image segmentation, existing research on AI-based alternatives focuses more on developing models that can imitate the best individual rather than harnessing the power of expert groups. In this paper, we introduce a single diffusion model-based approach that produces multiple plausible outputs by learning a distribution over group insights. Our proposed model generates a distribution of segmentation masks by leveraging the inherent stochastic sampling process of diffusion using only minimal additional learning. We demonstrate on three different medical image modalities- CT, ultrasound, and MRI that our model is capable of producing several possible variants while capturing the frequencies of their occurrences. Comprehensive results show that our proposed approach outperforms existing state-of-the-art ambiguous segmentation networks in terms of accuracy while preserving naturally occurring variation. We also propose a new metric to evaluate the diversity as well as the accuracy of segmentation predictions that aligns with the interest of clinical practice of collective insights. Implementation code will be released publicly after the review process.

count=1
* APPLeNet: Visual Attention Parameterized Prompt Learning for Few-Shot Remote Sensing Image Generalization Using CLIP
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Jha_APPLeNet_Visual_Attention_Parameterized_Prompt_Learning_for_Few-Shot_Remote_Sensing_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Jha_APPLeNet_Visual_Attention_Parameterized_Prompt_Learning_for_Few-Shot_Remote_Sensing_CVPRW_2023_paper.pdf)]
    * Title: APPLeNet: Visual Attention Parameterized Prompt Learning for Few-Shot Remote Sensing Image Generalization Using CLIP
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Mainak Singha, Ankit Jha, Bhupendra Solanki, Shirsha Bose, Biplab Banerjee
    * Abstract: In recent years, the success of large-scale vision-language models (VLMs) such as CLIP has led to their increased usage in various computer vision tasks. These models enable zero-shot inference through carefully crafted instructional text prompts without task-specific supervision. However, the potential of VLMs for generalization tasks in remote sensing (RS) has not been fully realized. To address this research gap, we propose a novel image-conditioned prompt learning strategy called the Visual Attention Parameterized Prompts Learning Network (APPLeNet). APPLeNet emphasizes the importance of multi-scale feature learning in RS scene classification and disentangles visual style and content primitives for domain generalization tasks. To achieve this, APPLeNet combines visual content features obtained from different layers of the vision encoder and style properties obtained from feature statistics of domain-specific batches. An attention-driven injection module is further introduced to generate visual tokens from this information. We also introduce an anti-correlation regularizer to ensure discrimination among the token embeddings, as this visual information is combined with the textual tokens. To validate APPLeNet, we curated four available RS benchmarks and introduced experimental protocols and datasets for three domain generalization tasks. Our results consistently outperform the relevant literature.

count=1
* Multi-Modal Multi-Objective Contrastive Learning for Sentinel-1/2 Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.pdf)]
    * Title: Multi-Modal Multi-Objective Contrastive Learning for Sentinel-1/2 Imagery
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jonathan Prexl, Michael Schmitt
    * Abstract: The field of spaceborne Earth observation offers, due to constant monitoring of the Earth's surface, a huge amount of unlabeled data. At the same time, for many applications, there still exists a shortage of high-quality labelled datasets. This is one of the major bottlenecks for progress in developing globally applicable deep learning models for analysing the dynamics of our planet from space. In recent years self-supervised representation learning revealed itself to state a very powerful way of incorporating unlabeled data into the typical supervised machine learning workflow. Still, many questions on how to adapt commonly used approaches to domain-specific properties of Earth observation data remain. In this work, we introduce and study approaches to incorporate multi-modal Earth observation data into a contrastive self-supervised learning framework by forcing inter- and intra-modality similarity in the loss function. Further, we introduce a batch-sampling strategy that leverages the geo-coding of the imagery in order to obtain harder negative pairs for the contrastive learning problem. We show through extensive experiments that various domain-specific downstream problems are benefitting from the above-mentioned contributions.

count=1
* Masked Vision Transformers for Hyperspectral Image Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Scheibenreif_Masked_Vision_Transformers_for_Hyperspectral_Image_Classification_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Scheibenreif_Masked_Vision_Transformers_for_Hyperspectral_Image_Classification_CVPRW_2023_paper.pdf)]
    * Title: Masked Vision Transformers for Hyperspectral Image Classification
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Linus Scheibenreif, Michael Mommert, Damian Borth
    * Abstract: Transformer architectures have become state-of-the-art models in computer vision and natural language processing. To a significant degree, their success can be attributed to self-supervised pre-training on large scale unlabeled datasets. This work investigates the use of self-supervised masked image reconstruction to advance transformer models for hyperspectral remote sensing imagery. To facilitate self-supervised pre-training, we build a large dataset of unlabeled hyperspectral observations from the EnMAP satellite and systematically investigate modifications of the vision transformer architecture to optimally leverage the characteristics of hyperspectral data. We find significant improvements in accuracy on different land cover classification tasks over both standard vision and sequence transformers using (i) blockwise patch embeddings, (ii) spatial-spectral self-attention, (iii) spectral positional embeddings and (iv) masked self-supervised pre-training. The resulting model outperforms standard transformer architectures by +5% accuracy on a labeled subset of our EnMAP data and by +15% on Houston2018 hyperspectral dataset, making it competitive with a strong 3D convolutional neural network baseline. In an ablation study on label-efficiency based on the Houston2018 dataset, self-supervised pre-training significantly improves transformer accuracy when little labeled training data is available. The self-supervised model outperforms randomly initialized transformers and the 3D convolutional neural network by +7-8% when only 0.1-10% of the training labels are available.

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
* Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.pdf)]
    * Title: Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Olaf Wysocki, Yan Xia, Magdalena Wysocki, Eleonora Grilli, Ludwig Hoegner, Daniel Cremers, Uwe Stilla
    * Abstract: Reconstructing semantic 3D building models at the level of detail (LoD) 3 is a long-standing challenge. Unlike mesh-based models, they require watertight geometry and object-wise semantics at the facade level. The principal challenge of such demanding semantic 3D reconstruction is reliable facade-level semantic segmentation of 3D input data. We present a novel method, called Scan2LoD3, that accurately reconstructs semantic LoD3 building models by improving facade-level semantic 3D segmentation. To this end, we leverage laser physics and 3D building model priors to probabilistically identify model conflicts. These probabilistic physical conflicts propose locations of model openings: Their final semantics and shapes are inferred in a Bayesian network fusing multimodal probabilistic maps of conflicts, 3D point clouds, and 2D images. To fulfill demanding LoD3 requirements, we use the estimated shapes to cut openings in 3D building priors and fit semantic 3D objects from a library of facade objects. Extensive experiments on the TUM city campus datasets demonstrate the superior performance of the proposed Scan2LoD3 over the state-of-the-art methods in facade-level detection, semantic segmentation, and LoD3 building model reconstruction. We believe our method can foster the development of probability-driven semantic 3D reconstruction at LoD3 since not only the high-definition reconstruction but also reconstruction confidence becomes pivotal for various applications such as autonomous driving and urban simulations.

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
* Long-Tailed Anomaly Detection with Learnable Class Names
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ho_Long-Tailed_Anomaly_Detection_with_Learnable_Class_Names_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ho_Long-Tailed_Anomaly_Detection_with_Learnable_Class_Names_CVPR_2024_paper.pdf)]
    * Title: Long-Tailed Anomaly Detection with Learnable Class Names
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chih-Hui Ho, Kuan-Chuan Peng, Nuno Vasconcelos
    * Abstract: Anomaly detection (AD) aims to identify defective images and localize their defects (if any). Ideally AD models should be able to detect defects over many image classes; without relying on hard-coded class names that can be uninformative or inconsistent across datasets; learn without anomaly supervision; and be robust to the long-tailed distributions of real-world applications. To address these challenges we formulate the problem of long-tailed AD by introducing several datasets with different levels of class imbalance and metrics for performance evaluation. We then propose a novel method LTAD to detect defects from multiple and long-tailed classes without relying on dataset class names. LTAD combines AD by reconstruction and semantic AD modules. AD by reconstruction is implemented with a transformer-based reconstruction module. Semantic AD is implemented with a binary classifier which relies on learned pseudo class names and a pretrained foundation model. These modules are learned over two phases. Phase 1 learns the pseudo-class names and a variational autoencoder (VAE) for feature synthesis that augments the training data to combat long-tails. Phase 2 then learns the parameters of the reconstruction and classification modules of LTAD. Extensive experiments using the proposed long-tailed datasets show that LTAD substantially outperforms the state-of-the-art methods for most forms of dataset imbalance. The long-tailed dataset split is available at https://zenodo.org/records/10854201

count=1
* SPIDeRS: Structured Polarization for Invisible Depth and Reflectance Sensing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ichikawa_SPIDeRS_Structured_Polarization_for_Invisible_Depth_and_Reflectance_Sensing_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ichikawa_SPIDeRS_Structured_Polarization_for_Invisible_Depth_and_Reflectance_Sensing_CVPR_2024_paper.pdf)]
    * Title: SPIDeRS: Structured Polarization for Invisible Depth and Reflectance Sensing
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tomoki Ichikawa, Shohei Nobuhara, Ko Nishino
    * Abstract: Can we capture shape and reflectance in stealth? Such capability would be valuable for many application domains in vision xR robotics and HCI. We introduce structured polarization for invisible depth and reflectance sensing (SPIDeRS) the first depth and reflectance sensing method using patterns of polarized light. The key idea is to modulate the angle of linear polarization (AoLP) of projected light at each pixel. The use of polarization makes it invisible and lets us recover not only depth but also directly surface normals and even reflectance. We implement SPIDeRS with a liquid crystal spatial light modulator (SLM) and a polarimetric camera. We derive a novel method for robustly extracting the projected structured polarization pattern from the polarimetric object appearance. We evaluate the effectiveness of SPIDeRS by applying it to a number of real-world objects. The results show that our method successfully reconstructs object shapes of various materials and is robust to diffuse reflection and ambient light. We also demonstrate relighting using recovered surface normals and reflectance. We believe SPIDeRS opens a new avenue of polarization use in visual sensing.

count=1
* Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Koch_Open3DSG_Open-Vocabulary_3D_Scene_Graphs_from_Point_Clouds_with_Queryable_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Koch_Open3DSG_Open-Vocabulary_3D_Scene_Graphs_from_Point_Clouds_with_Queryable_CVPR_2024_paper.pdf)]
    * Title: Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Sebastian Koch, Narunas Vaskevicius, Mirco Colosi, Pedro Hermosilla, Timo Ropinski
    * Abstract: Current approaches for 3D scene graph prediction rely on labeled datasets to train models for a fixed set of known object classes and relationship categories. We present Open3DSG an alternative approach to learn 3D scene graph prediction in an open world without requiring labeled scene graph data. We co-embed the features from a 3D scene graph prediction backbone with the feature space of powerful open world 2D vision language foundation models. This enables us to predict 3D scene graphs from 3D point clouds in a zero-shot manner by querying object classes from an open vocabulary and predicting the inter-object relationships from a grounded LLM with scene graph features and queried object classes as context. Open3DSG is the first 3D point cloud method to predict not only explicit open-vocabulary object classes but also open-set relationships that are not limited to a predefined label set making it possible to express rare as well as specific objects and relationships in the predicted 3D scene graph. Our experiments show that Open3DSG is effective at predicting arbitrary object classes as well as their complex inter-object relationships describing spatial supportive semantic and comparative relationships.

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
* Step Differences in Instructional Video
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Nagarajan_Step_Differences_in_Instructional_Video_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Nagarajan_Step_Differences_in_Instructional_Video_CVPR_2024_paper.pdf)]
    * Title: Step Differences in Instructional Video
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tushar Nagarajan, Lorenzo Torresani
    * Abstract: Comparing a user video to a reference how-to video is a key requirement for AR/VR technology delivering personalized assistance tailored to the user's progress. However current approaches for language-based assistance can only answer questions about a single video. We propose an approach that first automatically generates large amounts of visual instruction tuning data involving pairs of videos from HowTo100M by leveraging existing step annotations and accompanying narrations and then trains a video-conditioned language model to jointly reason across multiple raw videos. Our model achieves state-of-the-art performance at identifying differences between video pairs and ranking videos based on the severity of these differences and shows promising ability to perform general reasoning over multiple videos.

count=1
* Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Noman_Rethinking_Transformers_Pre-training_for_Multi-Spectral_Satellite_Imagery_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Noman_Rethinking_Transformers_Pre-training_for_Multi-Spectral_Satellite_Imagery_CVPR_2024_paper.pdf)]
    * Title: Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Mubashir Noman, Muzammal Naseer, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Fahad Shahbaz Khan
    * Abstract: Recent advances in unsupervised learning have demonstrated the ability of large vision models to achieve promising results on downstream tasks by pre-training on large amount of unlabelled data. Such pre-training techniques have also been explored recently in the remote sensing domain due to the availability of large amount of unlabelled data. Different from standard natural image datasets remote sensing data is acquired from various sensor technologies and exhibit diverse range of scale variations as well as modalities. Existing satellite image pre-training methods either ignore the scale information present in the remote sensing imagery or restrict themselves to use only a single type of data modality. In this paper we re-visit transformers pre-training and leverage multi-scale information that is effectively utilized with multiple modalities. Our proposed approach named SatMAE++ performs multi-scale pre-training and utilizes convolution based upsampling blocks to reconstruct the image at higher scales making it extensible to include more scales. Compared to existing works the proposed SatMAE++ with multi-scale pre-training is equally effective for both optical as well as multi-spectral imagery. Extensive experiments on six datasets reveal the merits of proposed contributions leading to state-of-the-art performance on all datasets. SatMAE++ achieves mean average precision (mAP) gain of 2.5% for multi-label classification task on BigEarthNet dataset.

count=1
* HIR-Diff: Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.pdf)]
    * Title: HIR-Diff: Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Li Pang, Xiangyu Rui, Long Cui, Hongzhong Wang, Deyu Meng, Xiangyong Cao
    * Abstract: Hyperspectral image (HSI) restoration aims at recovering clean images from degraded observations and plays a vital role in downstream tasks. Existing model-based methods have limitations in accurately modeling the complex image characteristics with handcraft priors and deep learning-based methods suffer from poor generalization ability. To alleviate these issues this paper proposes an unsupervised HSI restoration framework with pre-trained diffusion model (HIR-Diff) which restores the clean HSIs from the product of two low-rank components i.e. the reduced image and the coefficient matrix. Specifically the reduced image which has a low spectral dimension lies in the image field and can be inferred from our improved diffusion model where a new guidance function with total variation (TV) prior is designed to ensure that the reduced image can be well sampled. The coefficient matrix can be effectively pre-estimated based on singular value decomposition (SVD) and rank-revealing QR (RRQR) factorization. Furthermore a novel exponential noise schedule is proposed to accelerate the restoration process (about 5xacceleration for denoising) with little performance decrease. Extensive experimental results validate the superiority of our method in both performance and speed on a variety of HSI restoration tasks including HSI denoising noisy HSI super-resolution and noisy HSI inpainting. The code is available at https://github.com/LiPang/HIRDiff.

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
* CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_CorrMatch_Label_Propagation_via_Correlation_Matching_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_CorrMatch_Label_Propagation_via_Correlation_Matching_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf)]
    * Title: CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Boyuan Sun, Yuqi Yang, Le Zhang, Ming-Ming Cheng, Qibin Hou
    * Abstract: This paper presents a simple but performant semi-supervised semantic segmentation approach called CorrMatch. Previous approaches mostly employ complicated training strategies to leverage unlabeled data but overlook the role of correlation maps in modeling the relationships between pairs of locations. We observe that the correlation maps not only enable clustering pixels of the same category easily but also contain good shape information which previous works have omitted. Motivated by these we aim to improve the use efficiency of unlabeled data by designing two novel label propagation strategies. First we propose to conduct pixel propagation by modeling the pairwise similarities of pixels to spread the high-confidence pixels and dig out more. Then we perform region propagation to enhance the pseudo labels with accurate class-agnostic masks extracted from the correlation maps. CorrMatch achieves great performance on popular segmentation benchmarks. Taking the DeepLabV3+ with ResNet-101 backbone as our segmentation model we receive a 76%+ mIoU score on the Pascal VOC 2012 dataset with only 92 annotated images. Code is available at https://github.com/BBBBchan/CorrMatch .

count=1
* Towards Progressive Multi-Frequency Representation for Image Warping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Towards_Progressive_Multi-Frequency_Representation_for_Image_Warping_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Towards_Progressive_Multi-Frequency_Representation_for_Image_Warping_CVPR_2024_paper.pdf)]
    * Title: Towards Progressive Multi-Frequency Representation for Image Warping
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jun Xiao, Zihang Lyu, Cong Zhang, Yakun Ju, Changjian Shui, Kin-Man Lam
    * Abstract: Image warping a classic task in computer vision aims to use geometric transformations to change the appearance of images. Recent methods learn the resampling kernels for warping through neural networks to estimate missing values in irregular grids which however fail to capture local variations in deformed content and produce images with distortion and less high-frequency details. To address this issue this paper proposes an effective method namely MFR to learn Multi-Frequency Representations from input images for image warping. Specifically we propose a progressive filtering network to learn image representations from different frequency subbands and generate deformable images in a coarse-to-fine manner. Furthermore we employ learnable Gabor wavelet filters to improve the model's capability to learn local spatial-frequency representations. Comprehensive experiments including homography transformation equirectangular to perspective projection and asymmetric image super-resolution demonstrate that the proposed MFR significantly outperforms state-of-the-art image warping methods. Our method also showcases superior generalization to out-of-distribution domains where the generated images are equipped with rich details and less distortion thereby high visual quality. The source code is available at https://github.com/junxiao01/MFR.

count=1
* Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Temporally_Consistent_Unbalanced_Optimal_Transport_for_Unsupervised_Action_Segmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Temporally_Consistent_Unbalanced_Optimal_Transport_for_Unsupervised_Action_Segmentation_CVPR_2024_paper.pdf)]
    * Title: Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ming Xu, Stephen Gould
    * Abstract: We propose a novel approach to the action segmentation task for long untrimmed videos based on solving an optimal transport problem. By encoding a temporal consistency prior into a Gromov-Wasserstein problem we are able to decode a temporally consistent segmentation from a noisy affinity/matching cost matrix between video frames and action classes. Unlike previous approaches our method does not require knowing the action order for a video to attain temporal consistency. Furthermore our resulting (fused) Gromov-Wasserstein problem can be efficiently solved on GPUs using a few iterations of projected mirror descent. We demonstrate the effectiveness of our method in an unsupervised learning setting where our method is used to generate pseudo-labels for self-training. We evaluate our segmentation approach and unsupervised learning pipeline on the Breakfast 50-Salads YouTube Instructions and Desktop Assembly datasets yielding state-of-the-art results for the unsupervised video action segmentation task.

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
* Tackling the Satellite Downlink Bottleneck with Federated Onboard Learning of Image Compression
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Space/html/Gomez_Tackling_the_Satellite_Downlink_Bottleneck_with_Federated_Onboard_Learning_of_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Space/papers/Gomez_Tackling_the_Satellite_Downlink_Bottleneck_with_Federated_Onboard_Learning_of_CVPRW_2024_paper.pdf)]
    * Title: Tackling the Satellite Downlink Bottleneck with Federated Onboard Learning of Image Compression
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Pablo Gómez, Gabriele Meoni
    * Abstract: Satellite data transmission is a crucial bottleneck for Earth observation applications. To overcome this problem we propose a novel solution that trains a neural network on board multiple satellites to compress raw data and only send down heavily compressed previews of the images while retaining the possibility of sending down selected losslessly compressed data. The neural network learns to encode and decode the data in an unsupervised fashion using distributed machine learning. By simulating and optimizing the learning process under realistic constraints such as thermal power and communication limitations we demonstrate the feasibility and effectiveness of our approach. For this we model a constellation of three satellites in a Sun-synchronous orbit. We use real raw multispectral data from Sentinel-2 and demonstrate the feasibility on space-proven hardware for the training. Our compression method outperforms JPEG compression on different image metrics achieving better compression ratios and image quality. We report key performance indicators of our method such as image quality compression ratio and benchmark training time on a Unibap iX10-100 processor. Our method has the potential to significantly increase the amount of satellite data collected that would typically be discarded (e.g. over oceans) and can potentially be extended to other applications even outside Earth observation. All code and data of the method are available online to enable rapid application of this approach.

count=1
* Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Okabayashi_Cross-sensor_super-resolution_of_irregularly_sampled_Sentinel-2_time_series_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Okabayashi_Cross-sensor_super-resolution_of_irregularly_sampled_Sentinel-2_time_series_CVPRW_2024_paper.pdf)]
    * Title: Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Aimi Okabayashi, Nicolas Audebert, Simon Donike, Charlotte Pelletier
    * Abstract: Satellite imaging generally presents a trade-off between the frequency of acquisitions and the spatial resolution of the images. Super-resolution is often advanced as a way to get the best of both worlds. In this work we investigate multi-image super-resolution of satellite image time series i.e. how multiple images of the same area acquired at different dates can help reconstruct a higher resolution observation. In particular we extend state-of-the-art deep single and multi-image super-resolution algorithms such as SRDiff and HighRes-net to deal with irregularly sampled Sentinel-2 time series. We introduce BreizhSR a new dataset for 4x super-resolution of Sentinel-2 time series using very high-resolution SPOT-6 imagery of Brittany a French region. We show that using multiple images significantly improves super-resolution performance and that a well-designed temporal positional encoding allows us to perform super-resolution for different times of the series. In addition we observe a trade-off between spectral fidelity and perceptual quality of the reconstructed HR images questioning future directions for super-resolution of Earth Observation data. The source code is available at https://github.com/aimiokab/MISR-S2.

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
* Generalized Few-Shot Meets Remote Sensing: Discovering Novel Classes in Land Cover Mapping via Hybrid Semantic Segmentation Framework
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/html/Li_Generalized_Few-Shot_Meets_Remote_Sensing_Discovering_Novel_Classes_in_Land_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Li_Generalized_Few-Shot_Meets_Remote_Sensing_Discovering_Novel_Classes_in_Land_CVPRW_2024_paper.pdf)]
    * Title: Generalized Few-Shot Meets Remote Sensing: Discovering Novel Classes in Land Cover Mapping via Hybrid Semantic Segmentation Framework
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zhuohong Li, Fangxiao Lu, Jiaqi Zou, Lei Hu, Hongyan Zhang
    * Abstract: Land-cover mapping is one of the vital applications in Earth observation aiming at classifying each pixel's land-cover type of remote-sensing images. As natural and human activities change the landscape the land-cover map needs to be rapidly updated. However discovering newly appeared land-cover types in existing classification systems is still a non-trivial task hindered by various scales of complex land objects and insufficient labeled data over a wide-span geographic area. In this paper we propose a generalized few-shot segmentation-based framework named SegLand to update novel classes in high-resolution land-cover mapping. Specifically the proposed framework is designed in three parts: (a) Data pre-processing: the base training set and the few-shot support sets of novel classes are analyzed and augmented; (b) Hybrid segmentation structure: Multiple base learners and a modified Projection onto Orthogonal Prototypes (POP) network are combined to enhance the base-class recognition and to dig novel classes from insufficient labels data; (c) Ultimate fusion: the semantic segmentation results of the base learners and POP network are reasonably fused. The proposed framework has won first place in the leaderboard of the OpenEarthMap Land Cover Mapping Few-Shot Challenge. Experiments demonstrate the superiority of the framework for automatically updating novel land-cover classes with limited labeled data.

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
* Seeing the Vibration from Fiber-Optic Cables: Rain Intensity Monitoring using Deep Frequency Filtering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Jiang_Seeing_the_Vibration_from_Fiber-Optic_Cables_Rain_Intensity_Monitoring_using_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Jiang_Seeing_the_Vibration_from_Fiber-Optic_Cables_Rain_Intensity_Monitoring_using_CVPRW_2024_paper.pdf)]
    * Title: Seeing the Vibration from Fiber-Optic Cables: Rain Intensity Monitoring using Deep Frequency Filtering
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zhuocheng Jiang, Yangmin Ding, Junhui Zhao, Yue Tian, Shaobo Han, Sarper Ozharar, Ting Wang, James M. Moore
    * Abstract: The various sensing technologies such as cameras LiDAR radar and satellites with advanced machine learning models offers a comprehensive approach to environmental perception and understanding. This paper introduces an innovative Distributed Fiber Optic Sensing (DFOS) technology utilizing the existing telecommunication infrastructure networks for rain intensity monitoring. DFOS enables a novel way to monitor weather condition and environmental changes provides real-time continuous and precise measurements over large areas and delivers comprehensive insights beyond the visible spectrum. We use rain intensity as an example to demonstrate the sensing capabilities of DFOS system. To enhance the rain sensing performance we introduce a Deep Phase-Magnitude Network (DFMN) divide the raw sensing data into phase and magnitude component allowing targeted feature learning on each component independently. Furthermore we propose a Phase Frequency learnable filter (PFLF) for the phase component filtering and conduct standard convolution layers on the magnitude component leveraging the inherent physical properties of optical fiber sensing. We formulate the phase-magnitude channel into a parallel network and subsequently fuse the features for a comprehensive analysis in the end. Experimental results on the collected fiber sensing data show that the proposed method performs favorably against the state-of-the-art approaches.

count=1
* Image Vegetation Index Through a Cycle Generative Adversarial Network
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/PBVS/Suarez_Image_Vegetation_Index_Through_a_Cycle_Generative_Adversarial_Network_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Suarez_Image_Vegetation_Index_Through_a_Cycle_Generative_Adversarial_Network_CVPRW_2019_paper.pdf)]
    * Title: Image Vegetation Index Through a Cycle Generative Adversarial Network
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Patricia L. Suarez,  Angel D. Sappa,  Boris X. Vintimilla,  Riad I. Hammoud
    * Abstract: This paper proposes a novel approach to estimate the Normalized Difference Vegetation Index (NDVI) just from an RGB image. The NDVI values are obtained by using images from the visible spectral band together with a synthetic near infrared image obtained by a cycled GAN. The cycled GAN network is able to obtain a NIR image from a given gray scale image. It is trained by using unpaired set of gray scale and NIR images by using a U-net architecture and a multiple loss function (gray scale images are obtained from the provided RGB images). Then, the NIR image estimated with the proposed cycle generative adversarial network is used to compute the NDVI index. Experimental results are provided showing the validity of the proposed approach. Additionally, comparisons with previous approaches are also provided.

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
* Residual Pixel Attention Network for Spectral Reconstruction From RGB Images
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Peng_Residual_Pixel_Attention_Network_for_Spectral_Reconstruction_From_RGB_Images_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Peng_Residual_Pixel_Attention_Network_for_Spectral_Reconstruction_From_RGB_Images_CVPRW_2020_paper.pdf)]
    * Title: Residual Pixel Attention Network for Spectral Reconstruction From RGB Images
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Hao Peng, Xiaomei Chen, Jie Zhao
    * Abstract: In recent years, hyperspectral reconstruction based on RGB imaging has made significant progress of deep learning, which greatly improves the accuracy of the reconstructed hyperspectral images. In this paper, we proposed a convolution neural network of the hyperspectral reconstruction from a single RGB image, called Residual Pixel Attention Network (RPAN). Specifically, we proposed a Pixel Attention (PA) module, which was applied to each pixel of all feature maps, to adaptively rescale pixel-wise features in all feature maps. The RPAN was trained on the hyperspectral dataset provided by NTIRE 2020 Spectral Reconstruction Challenge and compared with previous state-of-the-art method HSCNN+. The results showed our RPAN network had achieved superior performance in terms of MRAE and RMSE.

count=1
* Intelligent Scene Caching to Improve Accuracy for Energy-Constrained Embedded Vision
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Simpson_Intelligent_Scene_Caching_to_Improve_Accuracy_for_Energy-Constrained_Embedded_Vision_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Simpson_Intelligent_Scene_Caching_to_Improve_Accuracy_for_Energy-Constrained_Embedded_Vision_CVPRW_2020_paper.pdf)]
    * Title: Intelligent Scene Caching to Improve Accuracy for Energy-Constrained Embedded Vision
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Benjamin Simpson, Ekdeep Lubana, Yuchen Liu, Robert Dick
    * Abstract: We describe an efficient method of improving the performance of vision algorithms operating on video streams by reducing the amount of data captured and transferred from image sensors to analysis servers in a data-aware manner. The key concept is to combine guided, highly heterogeneous sampling with an intelligent Scene Cache. This enables the system to adapt to spatial and temporal patterns in the scene, thus reducing redundant data capture and processing. A software prototype of our framework running on a general-purpose embedded processor enables superior object detection accuracy (by 56%) at similar energy consumption (slight improvement of 4%) compared to an H.264 hardware accelerator.

count=1
* Representations, Metrics and Statistics for Shape Analysis of Elastic Graphs
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w50/Guo_Representations_Metrics_and_Statistics_for_Shape_Analysis_of_Elastic_Graphs_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w50/Guo_Representations_Metrics_and_Statistics_for_Shape_Analysis_of_Elastic_Graphs_CVPRW_2020_paper.pdf)]
    * Title: Representations, Metrics and Statistics for Shape Analysis of Elastic Graphs
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Xiaoyang Guo, Anuj Srivastava
    * Abstract: Past approaches for statistical shape analysis of objects have focused mainly on objects within the same topological classes, e.g. , scalar functions, Euclidean curves, or surfaces, etc. For objects that differ in more complex ways, the current literature offers only topological methods. This paper introduces a far-reaching geometric approach for analyzing shapes of graphical objects, such as road networks, blood vessels, brain fiber tracts, etc. It represents such objects, exhibiting differences in both geometries and topologies, as graphs made of curves with arbitrary shapes (edges) and connected at arbitrary junctions (nodes). To perform statistical analyses, one needs mathematical representations, metrics and other geometrical tools, such as geodesics, means, and covariances. This paper utilizes a quotient structure to develop efficient algorithms for computing these quantities, leading to useful statistical tools, including principal component analysis and analytical statistical testing and modeling of graphical shapes. The efficacy of this framework is demonstrated using various simulated as well as the real data from neurons and brain arterial networks.

count=1
* Learning to Detect Carried Objects with Minimal Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/html/Dondera_Learning_to_Detect_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/papers/Dondera_Learning_to_Detect_2013_CVPR_paper.pdf)]
    * Title: Learning to Detect Carried Objects with Minimal Supervision
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Radu Dondera, Vlad Morariu, Larry Davis
    * Abstract: We propose a learning-based method for detecting carried objects that generates candidate image regions from protrusion, color contrast and occlusion boundary cues, and uses a classifier to filter out the regions unlikely to be carried objects. The method achieves higher accuracy than state of the art, which can only detect protrusions from the human shape, and the discriminative model it builds for the silhouette context-based region features generalizes well. To reduce annotation effort, we investigate training the model in a Multiple Instance Learning framework where the only available supervision is "walk" and "carry" labels associated with intervals of human tracks, i.e., the spatial extent of carried objects is not annotated. We present an extension to the miSVM algorithm that uses knowledge of the fraction of positive instances in positive bags and that scales to training sets of hundreds of thousands of instances.

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
* Efficient and Automated Multimodal Satellite Data Registration Through MRFs and Linear Programming
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W06/html/Karantzalos_Efficient_and_Automated_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W06/papers/Karantzalos_Efficient_and_Automated_2014_CVPR_paper.pdf)]
    * Title: Efficient and Automated Multimodal Satellite Data Registration Through MRFs and Linear Programming
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Konstantinos Karantzalos, Aristeidis Sotiras, Nikos Paragios
    * Abstract: The accurate and automated registration of multimodal remote sensing data is of fundamental importance for numerous emerging geospatial environmental and engineering applications. However, the registration of very large multimodal, multitemporal, with different spatial resolutions data is, still, an open matter. To this end, we propose a generic and automated registration framework based on Markov Random Fields (MRFs) and efficient linear programming. The discrete optimization setting along with the introduced data-specific energy terms form a modular approach with respect to the similarity criterion allowing to fully exploit the spectral properties of multimodal remote sensing datasets. The proposed approach was validated both qualitatively and quantitatively demonstrating its potentials on very large (more than 100M pixels) multitemporal remote sensing datasets. In particular, in terms of spatial accuracy the geometry of the optical and radar data has been recovered with displacement errors of less than 2 and 3 pixels, respectively. In terms of computational efficiency the optical data term can converge after 7-8 minutes, while the radar data term after less than 15 minutes.

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
* Effective Semantic Pixel Labelling With Convolutional Networks and Conditional Random Fields
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Paisitkriangkrai_Effective_Semantic_Pixel_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Paisitkriangkrai_Effective_Semantic_Pixel_2015_CVPR_paper.pdf)]
    * Title: Effective Semantic Pixel Labelling With Convolutional Networks and Conditional Random Fields
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Sakrapee Paisitkriangkrai, Jamie Sherrah, Pranam Janney, Anton Van-Den Hengel
    * Abstract: Large amounts of available training data and increasing computing power have led to the recent success of deep convolutional neural networks (CNN) on a large number of applications. In this paper, we propose an effective semantic pixel labelling using CNN features, hand-crafted features and Conditional Random Fields (CRFs). Both CNN and hand-crafted features are applied to dense image patches to produce per-pixel class probabilities. The CRF infers a labelling that smooths regions while respecting the edges present in the imagery. The method is applied to the ISPRS 2D semantic labelling challenge dataset with competitive classification accuracy.

count=1
* Learning Slow Features for Behaviour Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Zafeiriou_Learning_Slow_Features_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Zafeiriou_Learning_Slow_Features_2013_ICCV_paper.pdf)]
    * Title: Learning Slow Features for Behaviour Analysis
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Lazaros Zafeiriou, Mihalis A. Nicolaou, Stefanos Zafeiriou, Symeon Nikitidis, Maja Pantic
    * Abstract: A recently introduced latent feature learning technique for time varying dynamic phenomena analysis is the socalled Slow Feature Analysis (SFA). SFA is a deterministic component analysis technique for multi-dimensional sequences that by minimizing the variance of the first order time derivative approximation of the input signal finds uncorrelated projections that extract slowly-varying features ordered by their temporal consistency and constancy. In this paper, we propose a number of extensions in both the deterministic and the probabilistic SFA optimization frameworks. In particular, we derive a novel deterministic SFA algorithm that is able to identify linear projections that extract the common slowest varying features of two or more sequences. In addition, we propose an Expectation Maximization (EM) algorithm to perform inference in a probabilistic formulation of SFA and similarly extend it in order to handle two and more time varying data sequences. Moreover, we demonstrate that the probabilistic SFA (EMSFA) algorithm that discovers the common slowest varying latent space of multiple sequences can be combined with dynamic time warping techniques for robust sequence timealignment. The proposed SFA algorithms were applied for facial behavior analysis demonstrating their usefulness and appropriateness for this task.

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
* Integrating Dashcam Views Through Inter-Video Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Chen_Integrating_Dashcam_Views_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_Integrating_Dashcam_Views_ICCV_2015_paper.pdf)]
    * Title: Integrating Dashcam Views Through Inter-Video Mapping
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Hsin-I Chen, Yi-Ling Chen, Wei-Tse Lee, Fan Wang, Bing-Yu Chen
    * Abstract: In this paper, an inter-video mapping approach is proposed to integrate video footages from two dashcams installed on a preceding and its following vehicle to provide the illusion that the driver of the following vehicle can see-through the preceding one. The key challenge is to adapt the perspectives of the two videos based on a small number of common features since a large portion of the common region in the video captured by the following vehicle is occluded by the preceding one. Inspired by the observation that images with the most similar viewpoints yield dense and high-quality matches, the proposed inter-video mapping estimates spatially-varying motions across two videos utilizing images of very similar contents. Specifically, we estimate frame-to-frame motions of each two consecutive images and incrementally add new views into a merged representation. In this way, long-rang motion estimation is achieved, and the observed perspective discrepancy between the two videos can be well approximated our motion estimation. Once the inter-video mapping is established, the correspondences can be updated incrementally, so the proposed method is suitable for on-line applications. Our experiments demonstrate the effectiveness of our approach on real-world challenging videos.

count=1
* Simultaneous Foreground Detection and Classification With Hybrid Features
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Kim_Simultaneous_Foreground_Detection_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Kim_Simultaneous_Foreground_Detection_ICCV_2015_paper.pdf)]
    * Title: Simultaneous Foreground Detection and Classification With Hybrid Features
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Jaemyun Kim, Adin Ramirez Rivera, Byungyong Ryu, Oksam Chae
    * Abstract: In this paper, we propose a hybrid background model that relies on edge and non-edge features of the image to produce the model. We encode these features into a coding scheme, that we called Local Hybrid Pattern (LHP), that selectively models edges and non-edges features of each pixel. Furthermore, we model each pixel with an adaptive code dictionary to represent the background dynamism, and update it by adding stable codes and discarding unstable ones. We weight each code in the dictionary to enhance its description of the pixel it models. The foreground is detected as the incoming codes that deviate from the dictionary. We can detect (as foreground or background) and classify (as edge or inner region) each pixel simultaneously. We tested our proposed method in existing databases with promising results.

count=1
* Robust and Optimal Sum-of-Squares-Based Point-to-Plane Registration of Image Sets and Structured Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Paudel_Robust_and_Optimal_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Paudel_Robust_and_Optimal_ICCV_2015_paper.pdf)]
    * Title: Robust and Optimal Sum-of-Squares-Based Point-to-Plane Registration of Image Sets and Structured Scenes
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Danda Pani Paudel, Adlane Habed, Cedric Demonceaux, Pascal Vasseur
    * Abstract: This paper deals with the problem of registering a known structured 3D scene and its metric Structure-from-Motion (SfM) counterpart. The proposed work relies on a prior plane segmentation of the 3D scene and aligns the data obtained from both modalities by solving the point-to-plane assignment problem. An inliers-maximization approach within a Branch-and-Bound (BnB) search scheme is adopted. For the first time in this paper, a Sum-of-Squares optimization theory framework is employed for identifying point-to-plane mismatches (i.e. outliers) with certainty. This allows us to iteratively build potential inliers sets and converge to the solution satisfied by the largest number of point-to-plane assignments. Furthermore, our approach is boosted by new plane visibility conditions which are also introduced in this paper. Using this framework, we solve the registration problem in two cases: (i) a set of putative point-to-plane correspondences (with possibly overwhelmingly many outliers) is given as input and (ii) no initial correspondences are given. In both cases, our approach yields outstanding results in terms of robustness and optimality.

count=1
* A Scalable Architecture for Operational FMV Exploitation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w1/html/Thissell_A_Scalable_Architecture_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w1/papers/Thissell_A_Scalable_Architecture_ICCV_2015_paper.pdf)]
    * Title: A Scalable Architecture for Operational FMV Exploitation
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: William R. Thissell, Robert Czajkowski, Francis Schrenk, Timothy Selway, Anthony J. Ries, Shamoli Patel, Patricia L. McDermott, Rod Moten, Ron Rudnicki, Guna Seetharaman, Ilker Ersoy, Kannappan Palaniappan
    * Abstract: A scalable open systems and standards derived software ecosystem is described for computer vision analytics (CVA) assisted exploitation of full motion video (FMV). The ecosystem, referred to as the Advanced Video Activity Analytics (AVAA), has two instantiations, one for size, weight, and power (SWAP) constrained conditions, and the other for large to massive cloud based configurations. The architecture is designed to meet operational analyst requirements to increase their productivity and accuracy for exploiting FMV using local cluster or scalable cloud-based computing resources. CVAs are encapsulated within a software plug-in architecture and FMV processing pipelines are constructed by combining these plug-ins to accomplish analytical tasks and manage provenance of processing history. An example pipeline for real-time motion detection and moving object characterization using the flux tensor approach is presented. An example video ingest experiment is described. Quantitative and qualitative methods for human factors engineering (HFE) assessment to evaluate cognitive loads for alternative work flow design choices are discussed. This HFE process is used for validating that an AVAA system instantiation with candidate workflow pipelines meets CVA assisted FMV exploitation operational goals for specific analyst workflows. AVAA offers a new framework for video understanding at scale for large enterprise applications in the government and commercial sectors.

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
* Scene Categorization With Spectral Features
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Khan_Scene_Categorization_With_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Khan_Scene_Categorization_With_ICCV_2017_paper.pdf)]
    * Title: Scene Categorization With Spectral Features
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Salman H. Khan, Munawar Hayat, Fatih Porikli
    * Abstract: Spectral signatures of natural scenes were earlier found to be distinctive for different scene types with varying spatial envelope properties such as openness, naturalness, ruggedness, and symmetry. Recently, such handcrafted features have been outclassed by deep learning based representations. This paper proposes a novel spectral description of convolution features, implemented efficiently as a unitary transformation within deep network architectures. To the best of our knowledge, this is the first attempt to use deep learning based spectral features explicitly for image classification task. We show that the spectral transformation decorrelates convolutional activations, which reduces co-adaptation between feature detections, thus acts as an effective regularizer. Our approach achieves significant improvements on three large-scale scene-centric datasets (MIT-67, SUN-397, and Places-205). Furthermore, we evaluated the proposed approach on the attribute detection task where its superior performance manifests its relevance to semantically meaningful characteristics of natural scenes.

count=1
* Going Unconstrained With Rolling Shutter Deblurring
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/R._Going_Unconstrained_With_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/R._Going_Unconstrained_With_ICCV_2017_paper.pdf)]
    * Title: Going Unconstrained With Rolling Shutter Deblurring
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Mahesh Mohan M. R., A. N. Rajagopalan, Gunasekaran Seetharaman
    * Abstract: Most present-day imaging devices are equipped with CMOS sensors. Motion blur is a common artifact in hand-held cameras. Because CMOS sensors mostly employ a rolling shutter (RS), the motion deblurring problem takes on a new dimension. Although few works have recently addressed this problem, they suffer from many constraints including heavy computational cost, need for precise sensor information, and inability to deal with wide-angle systems (which most cell-phone and drone cameras are) and irregular camera trajectory. In this work, we propose a model for RS blind motion deblurring that mitigates these issues significantly. Comprehensive comparisons with state-of-the-art methods reveal that our approach not only exhibits significant computational gains and unconstrained functionality but also leads to improved deblurring performance.

count=1
* Simultaneous Detection and Removal of High Altitude Clouds From an Image
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Sandhan_Simultaneous_Detection_and_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Sandhan_Simultaneous_Detection_and_ICCV_2017_paper.pdf)]
    * Title: Simultaneous Detection and Removal of High Altitude Clouds From an Image
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Tushar Sandhan, Jin Young Choi
    * Abstract: Interestingly, shape of the high-altitude clouds serves as a beacon for weather forecasting, so its detection is of vital importance. Besides these clouds often cause hindrance in an endeavor of satellites to inspect our world. Even thin clouds produce the undesired superposition of visual information, whose decomposition into the clear background and cloudy layer using a single satellite image is a highly ill-posed problem. In this work, we derive sophisticated image priors by thoroughly analyzing the properties of high-altitude clouds and geological images; and formulate a non-convex optimization scheme, which simultaneously detects and removes the clouds within a few seconds. Experimental results on real world RGB images demonstrate that the proposed method outperforms the other competitive methods by retaining the comprehensive background details and producing the precise shape of the cloudy layer.

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
* Panning and Jitter Invariant Incremental Principal Component Pursuit for Video Background Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w25/html/Chau_Panning_and_Jitter_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w25/Chau_Panning_and_Jitter_ICCV_2017_paper.pdf)]
    * Title: Panning and Jitter Invariant Incremental Principal Component Pursuit for Video Background Modeling
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Gustavo Chau, Paul Rodriguez
    * Abstract: Video background modeling is an important preprocessing stage for various applications and principal component pursuit (PCP) is among the state-of-the-art algorithms for this task. One of the main drawbacks of PCP is its sensitivity to jitter and camera movement. This problem has only been partially solved by a few methods devised for jitter or small transformations. However, such methods cannot handle the case of moving or panning cameras. We present a novel, fully incremental PCP algorithm, named incPCP-PTI, that is able to cope with panning scenarios and jitter by continuously aligning the low-rank component to the current reference frame of the camera. To the best of our knowledge, incPCP-PTI is the first low rank plus additive incremental matrix method capable of handling these scenarios. Results on synthetic videos and CDNET2014 videos show that incPCP-PTI is able to maintain a good performance in the detection of moving objects even when panning and jitter are present in a video

count=1
* Compressed Singular Value Decomposition for Image and Video Processing
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w25/html/Erichson_Compressed_Singular_Value_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w25/Erichson_Compressed_Singular_Value_ICCV_2017_paper.pdf)]
    * Title: Compressed Singular Value Decomposition for Image and Video Processing
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: N. Benjamin Erichson, Steven L. Brunton, J. Nathan Kutz
    * Abstract: We demonstrate a heuristic algorithm to compute the approximate low-rank singular value decomposition. The algorithm is inspired by ideas from compressed sensing and, in particular, is suitable for image and video processing applications. Specifically, our compressed singular value decomposition (cSVD) algorithm employs aggressive random test matrices to efficiently sketch the row space of the input matrix. The resulting compressed representation of the data enables the computation of an accurate approximation of the dominant high-dimensional left and right singular vectors. We benchmark cSVD against the current state-of-the-art randomized SVD and show a performance boost while attaining near similar relative errors. The cSVD is simple to implement as well as embarrassingly parallel, i.e, ideally suited for GPU computations and mobile platforms.

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
* Minimum Delay Object Detection From Video
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Lao_Minimum_Delay_Object_Detection_From_Video_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lao_Minimum_Delay_Object_Detection_From_Video_ICCV_2019_paper.pdf)]
    * Title: Minimum Delay Object Detection From Video
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Dong Lao,  Ganesh Sundaramoorthi
    * Abstract: We consider the problem of detecting objects, as they come into view, from videos in an online fashion. We provide the first real-time solution that is guaranteed to minimize the delay, i.e., the time between when the object comes in view and the declared detection time, subject to acceptable levels of detection accuracy. The method leverages modern CNN-based object detectors that operate on a single frame, to aggregate detection results over frames to provide reliable detection at a rate, specified by the user, in guaranteed minimal delay. To do this, we formulate the problem as a Quickest Detection problem, which provides the aforementioned guarantees. We derive our algorithms from this theory. We show in experiments, that with an overhead of just 50 fps, we can increase the number of correct detections and decrease the overall computational cost compared to running a modern single-frame detector.

count=1
* Generative Modeling for Small-Data Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Generative_Modeling_for_Small-Data_Object_Detection_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Generative_Modeling_for_Small-Data_Object_Detection_ICCV_2019_paper.pdf)]
    * Title: Generative Modeling for Small-Data Object Detection
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Lanlan Liu,  Michael Muelly,  Jia Deng,  Tomas Pfister,  Li-Jia Li
    * Abstract: This paper explores object detection in the small data regime, where only a limited number of annotated bounding boxes are available due to data rarity and annotation expense. This is a common challenge today with machine learning being applied to many new tasks where obtaining training data is more challenging, e.g. in medical images with rare diseases that doctors sometimes only see once in their life-time. In this work we explore this problem from a generative modeling perspective by learning to generate new images with associated bounding boxes, and using these for training an object detector. We show that simply training previously proposed generative models does not yield satisfactory performance due to them optimizing for image realism rather than object detection accuracy. To this end we develop a new model with a novel unrolling mechanism that jointly optimizes the generative model and a detector such that the generated images improve the performance of the detector. We show this method outperforms the state of the art on two challenging datasets, disease detection and small data pedestrian detection, improving the average precision on NIH Chest X-ray by a relative 20% and localization accuracy by a relative 50%.

count=1
* A Delay Metric for Video Object Detection: What Average Precision Fails to Tell
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Mao_A_Delay_Metric_for_Video_Object_Detection_What_Average_Precision_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mao_A_Delay_Metric_for_Video_Object_Detection_What_Average_Precision_ICCV_2019_paper.pdf)]
    * Title: A Delay Metric for Video Object Detection: What Average Precision Fails to Tell
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Huizi Mao,  Xiaodong Yang,  William J. Dally
    * Abstract: Average precision (AP) is a widely used metric to evaluate detection accuracy of image and video object detectors. In this paper, we analyze the object detection from video and point out that mAP alone is not sufficient to capture the temporal nature of video object detection. To tackle this problem, we propose a comprehensive metric, Average Delay (AD), to measure and compare detection delay. To facilitate delay evaluation, we carefully select a subset of ImageNet VID, which we name as ImageNet VIDT with an emphasis on complex trajectories. By extensively evaluating a wide range of detectors on VIDT, we show that most methods drastically increase the detection delay but still preserve mAP well. In other words, mAP is not sensitive enough to reflect the temporal characteristics of a video object detector. Our results suggest that video object detection methods should be evaluated with a delay metric, particularly for latency-critical applications such as autonomous vehicle perception.

count=1
* Occupancy Flow: 4D Reconstruction by Learning Particle Dynamics
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Niemeyer_Occupancy_Flow_4D_Reconstruction_by_Learning_Particle_Dynamics_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Niemeyer_Occupancy_Flow_4D_Reconstruction_by_Learning_Particle_Dynamics_ICCV_2019_paper.pdf)]
    * Title: Occupancy Flow: 4D Reconstruction by Learning Particle Dynamics
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Michael Niemeyer,  Lars Mescheder,  Michael Oechsle,  Andreas Geiger
    * Abstract: Deep learning based 3D reconstruction techniques have recently achieved impressive results. However, while state-of-the-art methods are able to output complex 3D geometry, it is not clear how to extend these results to time-varying topologies. Approaches treating each time step individually lack continuity and exhibit slow inference, while traditional 4D reconstruction methods often utilize a template model or discretize the 4D space at fixed resolution. In this work, we present Occupancy Flow, a novel spatio-temporal representation of time-varying 3D geometry with implicit correspondences. Towards this goal, we learn a temporally and spatially continuous vector field which assigns a motion vector to every point in space and time. In order to perform dense 4D reconstruction from images or sparse point clouds, we combine our method with a continuous 3D representation. Implicitly, our model yields correspondences over time, thus enabling fast inference while providing a sound physical description of the temporal dynamics. We show that our method can be used for interpolation and reconstruction tasks, and demonstrate the accuracy of the learned correspondences. We believe that Occupancy Flow is a promising new 4D representation which will be useful for a variety of spatio-temporal reconstruction tasks.

count=1
* Bridging the Domain Gap for Ground-to-Aerial Image Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Regmi_Bridging_the_Domain_Gap_for_Ground-to-Aerial_Image_Matching_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Regmi_Bridging_the_Domain_Gap_for_Ground-to-Aerial_Image_Matching_ICCV_2019_paper.pdf)]
    * Title: Bridging the Domain Gap for Ground-to-Aerial Image Matching
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Krishna Regmi,  Mubarak Shah
    * Abstract: The visual entities in cross-view (e.g. ground and aerial) images exhibit drastic domain changes due to the differences in viewpoints each set of images is captured from. Existing state-of-the-art methods address the problem by learning view-invariant images descriptors. We propose a novel method for solving this task by exploiting the gener- ative powers of conditional GANs to synthesize an aerial representation of a ground-level panorama query and use it to minimize the domain gap between the two views. The synthesized image being from the same view as the ref- erence (target) image, helps the network to preserve im- portant cues in aerial images following our Joint Feature Learning approach. We fuse the complementary features from a synthesized aerial image with the original ground- level panorama features to obtain a robust query represen- tation. In addition, we employ multi-scale feature aggre- gation in order to preserve image representations at dif- ferent scales useful for solving this complex task. Experi- mental results show that our proposed approach performs significantly better than the state-of-the-art methods on the challenging CVUSA dataset in terms of top-1 and top-1% retrieval accuracies. Furthermore, we evaluate the gen- eralization of the proposed method for urban landscapes on our newly collected cross-view localization dataset with geo-reference information.

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
* Panoptic Segmentation of Satellite Image Time Series With Convolutional Temporal Attention Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Garnot_Panoptic_Segmentation_of_Satellite_Image_Time_Series_With_Convolutional_Temporal_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Garnot_Panoptic_Segmentation_of_Satellite_Image_Time_Series_With_Convolutional_Temporal_ICCV_2021_paper.pdf)]
    * Title: Panoptic Segmentation of Satellite Image Time Series With Convolutional Temporal Attention Networks
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Vivien Sainte Fare Garnot, Loic Landrieu
    * Abstract: Unprecedented access to multi-temporal satellite imagery has opened new perspectives for a variety of Earth observation tasks. Among them, pixel-precise panoptic segmentation of agricultural parcels has major economic and environmental implications. While researchers have explored this problem for single images, we argue that the complex temporal patterns of crop phenology are better addressed with temporal sequences of images. In this paper, we present the first end-to-end, single-stage method for panoptic segmentation of Satellite Image Time Series (SITS). This module can be combined with our novel image sequence encoding network which relies on temporal self-attention to extract rich and adaptive multi-scale spatio-temporal features. We also introduce PASTIS, the first open-access SITS dataset with panoptic annotations. We demonstrate the superiority of our encoder for semantic segmentation against multiple competing network architectures, and set up the first state-of-the-art of panoptic segmentation of SITS. Our implementation and the PASTIS dataset are publicly available at (link-upon-publication).

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
* Do Image Classifiers Generalize Across Time?
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Shankar_Do_Image_Classifiers_Generalize_Across_Time_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Shankar_Do_Image_Classifiers_Generalize_Across_Time_ICCV_2021_paper.pdf)]
    * Title: Do Image Classifiers Generalize Across Time?
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Vaishaal Shankar, Achal Dave, Rebecca Roelofs, Deva Ramanan, Benjamin Recht, Ludwig Schmidt
    * Abstract: Vision models notoriously flicker when applied to videos: they correctly recognize objects in some frames, but fail on perceptually similar, nearby frames. In this work, we systematically analyze the robustness of image classifiers to such temporal perturbations in videos. To do so, we construct two new datasets, ImageNet-Vid-Robust and YTBB-Robust, containing a total of 57,897 images grouped into 3,139 sets of perceptually similar images. Our datasets were derived from ImageNet-Vid and YouTube-BB, respectively, and thoroughly re-annotated by human experts for image similarity. We evaluate a diverse array of classifiers pre-trained on ImageNet and show a median classification accuracy drop of 16 and 10 points, respectively, on our two datasets. Additionally, we evaluate three detection models and show that natural perturbations induce both classification as well as localization errors, leading to a median drop in detection mAP of 14 points. Our analysis demonstrates that perturbations occurring naturally in videos pose a substantial and realistic challenge to deploying convolutional neural networks in environments that require both reliable and low-latency predictions.

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
* Augmenting Depth Estimation With Geospatial Context
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Workman_Augmenting_Depth_Estimation_With_Geospatial_Context_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Workman_Augmenting_Depth_Estimation_With_Geospatial_Context_ICCV_2021_paper.pdf)]
    * Title: Augmenting Depth Estimation With Geospatial Context
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Scott Workman, Hunter Blanton
    * Abstract: Modern cameras are equipped with a wide array of sensors that enable recording the geospatial context of an image. Taking advantage of this, we explore depth estimation under the assumption that the camera is geocalibrated, a problem we refer to as geo-enabled depth estimation. Our key insight is that if capture location is known, the corresponding overhead viewpoint offers a valuable resource for understanding the scale of the scene. We propose an end-to-end architecture for depth estimation that uses geospatial context to infer a synthetic ground-level depth map from a co-located overhead image, then fuses it inside of an encoder/decoder style segmentation network. To support evaluation of our methods, we extend a recently released dataset with overhead imagery and corresponding height maps. Results demonstrate that integrating geospatial context significantly reduces error compared to baselines, both at close ranges and when evaluating at much larger distances than existing benchmarks consider.

count=1
* Dynamic Cross Feature Fusion for Remote Sensing Pansharpening
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Dynamic_Cross_Feature_Fusion_for_Remote_Sensing_Pansharpening_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Dynamic_Cross_Feature_Fusion_for_Remote_Sensing_Pansharpening_ICCV_2021_paper.pdf)]
    * Title: Dynamic Cross Feature Fusion for Remote Sensing Pansharpening
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Xiao Wu, Ting-Zhu Huang, Liang-Jian Deng, Tian-Jing Zhang
    * Abstract: Deep Convolution Neural Networks have been adopted for pansharpening and achieved state-of-the-art performance. However, most of the existing works mainly focus on single-scale feature fusion, which leads to failure in fully considering relationships of information between high-level semantics and low-level features, despite the network is deep enough. In this paper, we propose a dynamic cross feature fusion network (DCFNet) for pansharpening. Specifically, DCFNet contains multiple parallel branches, including a high-resolution branch served as the backbone, and the low-resolution branches progressively supplemented into the backbone. Thus our DCFNet can represent the overall information well. In order to enhance the relationships of inter-branches, dynamic cross feature transfers are embedded into multiple branches to obtain high-resolution representations. Then contextualized features will be learned to improve the fusion of information. Experimental results indicate that DCFNet significantly outperforms the prior arts in both quantitative indicators and visual qualities.

count=1
* 3D Semantic Label Transfer in Human-Robot Collaboration
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/html/Rozenberszki_3D_Semantic_Label_Transfer_in_Human-Robot_Collaboration_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/papers/Rozenberszki_3D_Semantic_Label_Transfer_in_Human-Robot_Collaboration_ICCVW_2021_paper.pdf)]
    * Title: 3D Semantic Label Transfer in Human-Robot Collaboration
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Dávid Rozenberszki, Gábor Sörös, Szilvia Szeier, András Lőrincz
    * Abstract: We tackle two practical problems in robotic scene understanding. First, the computational requirements of current semantic segmentation algorithms are prohibitive for typical robots. Second, the viewpoints of ground robots are quite different from typical human viewpoints of training datasets which may lead to misclassified objects from robot viewpoints. We present a system for sharing and reusing 3D semantic information between multiple agents with different viewpoints. We first co-localize all agents in the same coordinate system. Next, we create a 3D dense semantic model of the space from human viewpoints close to real time. Finally, by re-rendering the model's semantic labels (and/or depth maps) from the ground robots' own estimated viewpoints and sharing them over the network, we can give 3D semantic understanding to simpler agents. We evaluate the reconstruction quality and show how tiny robots can reuse knowledge about the space collected by more capable peers.

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
* Self-Supervised Object Detection from Egocentric Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.pdf)]
    * Title: Self-Supervised Object Detection from Egocentric Videos
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Peri Akiva, Jing Huang, Kevin J Liang, Rama Kovvuri, Xingyu Chen, Matt Feiszli, Kristin Dana, Tal Hassner
    * Abstract: Understanding the visual world from the perspective of humans (egocentric) has been a long-standing challenge in computer vision. Egocentric videos exhibit high scene complexity and irregular motion flows compared to typical video understanding tasks. With the egocentric domain in mind, we address the problem of self-supervised, class-agnostic object detection, which aims to locate all objects in a given view, regardless of category, without any annotations or pre-training weights. Our method, self-supervised object Detection from Egocentric VIdeos (DEVI), generalizes appearance-based methods to learn features that are category-specific and invariant to viewing angles and illumination conditions from highly ambiguous environments in an end-to-end manner. Our approach leverages typical human behavior and its egocentric perception to sample diverse views of the same objects for our multi-view and scale-regression loss functions. With our learned cluster residual module, we are able to effectively describe multi-category patches for better complex scene understanding. DEVI provides a boost in performance on recent egocentric datasets, with performance gains up to 4.11% AP50, 0.11% AR1, 1.32% AR10, and 5.03% AR100, while significantly reducing model complexity. We also demonstrate competitive performance on out-of-domain datasets without additional training or fine-tuning.

count=1
* Doppelgangers: Learning to Disambiguate Images of Similar Structures
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Doppelgangers_Learning_to_Disambiguate_Images_of_Similar_Structures_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Doppelgangers_Learning_to_Disambiguate_Images_of_Similar_Structures_ICCV_2023_paper.pdf)]
    * Title: Doppelgangers: Learning to Disambiguate Images of Similar Structures
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ruojin Cai, Joseph Tung, Qianqian Wang, Hadar Averbuch-Elor, Bharath Hariharan, Noah Snavely
    * Abstract: We consider the visual disambiguation task of determining whether a pair of visually similar images depict the same or distinct 3D surfaces (e.g., the same or opposite sides of a symmetric building). Illusory image matches, where two images observe distinct but visually similar 3D surfaces, can be challenging for humans to differentiate, and can also lead 3D reconstruction algorithms to produce erroneous results. We propose a learning-based approach to visual disambiguation, formulating it as a binary classification task on image pairs. To that end, we introduce a new dataset for this problem, Doppelgangers, which includes image pairs of similar structures with ground truth labels. We also design a network architecture that takes the spatial distribution of local keypoints and matches as input, allowing for better reasoning about both local and global cues. Our evaluation shows that our method can distinguish illusory matches in difficult cases, and can be integrated into SfM pipelines to produce correct, disambiguated 3D reconstructions. See our project page for our code, datasets, and more results: http://doppelgangers-3d.github.io/.

count=1
* FBLNet: FeedBack Loop Network for Driver Attention Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_FBLNet_FeedBack_Loop_Network_for_Driver_Attention_Prediction_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_FBLNet_FeedBack_Loop_Network_for_Driver_Attention_Prediction_ICCV_2023_paper.pdf)]
    * Title: FBLNet: FeedBack Loop Network for Driver Attention Prediction
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yilong Chen, Zhixiong Nan, Tao Xiang
    * Abstract: The problem of predicting driver attention from the driving perspective is gaining increasing research focus due to its remarkable significance for autonomous driving and assisted driving systems. The driving experience is extremely important for safe driving, a skilled driver is able to effortlessly predict oncoming danger (before it becomes salient) based on the driving experience and quickly pay attention to the corresponding zones. However, the nonobjective driving experience is difficult to model, so a mechanism simulating the driver experience accumulation procedure is absent in existing methods, and the current methods usually follow the technique line of saliency prediction methods to predict driver attention. In this paper, we propose a FeedBack Loop Network (FBLNet), which attempts to model the driving experience accumulation procedure. By over-and-over iterations, FBLNet generates the incremental knowledge that carries rich historically-accumulative and long-term temporal information. The incremental knowledge in our model is like the driving experience of humans. Under the guidance of the incremental knowledge, our model fuses the CNN feature and Transformer feature that are extracted from the input image to predict driver attention. Our model exhibits a solid advantage over existing methods, achieving an outstanding performance improvement on two driver attention benchmark datasets.

count=1
* Deep Geometry-Aware Camera Self-Calibration from Video
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Hagemann_Deep_Geometry-Aware_Camera_Self-Calibration_from_Video_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Hagemann_Deep_Geometry-Aware_Camera_Self-Calibration_from_Video_ICCV_2023_paper.pdf)]
    * Title: Deep Geometry-Aware Camera Self-Calibration from Video
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Annika Hagemann, Moritz Knorr, Christoph Stiller
    * Abstract: Accurate intrinsic calibration is essential for camera-based 3D perception, yet, it typically requires targets of well-known geometry. Here, we propose a camera self-calibration approach that infers camera intrinsics during application, from monocular videos in the wild. We propose to explicitly model projection functions and multi-view geometry, while leveraging the capabilities of deep neural networks for feature extraction and matching. To achieve this, we build upon recent research on integrating bundle adjustment into deep learning models, and introduce a self-calibrating bundle adjustment layer. The self-calibrating bundle adjustment layer optimizes camera intrinsics through classical Gauss-Newton steps and can be adapted to different camera models without re-training. As a specific realization, we implemented this layer within the deep visual SLAM system DROID-SLAM, and show that the resulting model, DroidCalib, yields state-of-the-art calibration accuracy across multiple public datasets. Our results suggest that the model generalizes to unseen environments and different camera models, including significant lens distortion. Thereby, the approach enables performing 3D perception tasks without prior knowledge about the camera. Code is available at https://github.com/boschresearch/droidcalib.

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
* OmnimatteRF: Robust Omnimatte with 3D Background Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_OmnimatteRF_Robust_Omnimatte_with_3D_Background_Modeling_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Lin_OmnimatteRF_Robust_Omnimatte_with_3D_Background_Modeling_ICCV_2023_paper.pdf)]
    * Title: OmnimatteRF: Robust Omnimatte with 3D Background Modeling
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Geng Lin, Chen Gao, Jia-Bin Huang, Changil Kim, Yipeng Wang, Matthias Zwicker, Ayush Saraf
    * Abstract: Video matting has broad applications, from adding interesting effects to casually captured movies to assisting video production professionals. Matting with associated effects such as shadows and reflections has also attracted increasing research activity, and methods like Omnimatte have been proposed to separate dynamic foreground objects of interest into their own layers. However, prior works represent video backgrounds as 2D image layers, limiting their capacity to express more complicated scenes, thus hindering application to real-world videos. In this paper, we propose a novel video matting method, OmnimatteRF, that combines dynamic 2D foreground layers and a 3D background model. The 2D layers preserve the details of the subjects, while the 3D background robustly reconstructs scenes in real-world videos. Extensive experiments demonstrate that our method reconstructs scenes with better quality on various videos.

count=1
* Under-Display Camera Image Restoration with Scattering Effect
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Song_Under-Display_Camera_Image_Restoration_with_Scattering_Effect_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Song_Under-Display_Camera_Image_Restoration_with_Scattering_Effect_ICCV_2023_paper.pdf)]
    * Title: Under-Display Camera Image Restoration with Scattering Effect
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Binbin Song, Xiangyu Chen, Shuning Xu, Jiantao Zhou
    * Abstract: The under-display camera (UDC) provides consumers with a full-screen visual experience without any obstruction due to notches or punched holes. However, the semi-transparent nature of the display inevitably introduces the severe degradation into UDC images. In this work, we address the UDC image restoration problem with the specific consideration of the scattering effect caused by the display. We explicitly model the scattering effect by treating the display as a piece of homogeneous scattering medium. With the physical model of the scattering effect, we improve the image formation pipeline for the image synthesis to construct a realistic UDC dataset with ground truths. To suppress the scattering effect for the eventual UDC image recovery, a two-branch restoration network is designed. More specifically, the scattering branch leverages global modeling capabilities of the channel-wise self-attention to estimate parameters of the scattering effect from degraded images. While the image branch exploits the local representation advantage of CNN to recover clear scenes, implicitly guided by the scattering branch. Extensive experiments are conducted on both real-world and synthesized data, demonstrating the superiority of the proposed method over the state-of-the-art UDC restoration techniques. The source code and dataset are available at https://github.com/NamecantbeNULL/SRUDC.

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
* XNet: Wavelet-Based Low and High Frequency Fusion Networks for Fully- and Semi-Supervised Semantic Segmentation of Biomedical Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_XNet_Wavelet-Based_Low_and_High_Frequency_Fusion_Networks_for_Fully-_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_XNet_Wavelet-Based_Low_and_High_Frequency_Fusion_Networks_for_Fully-_ICCV_2023_paper.pdf)]
    * Title: XNet: Wavelet-Based Low and High Frequency Fusion Networks for Fully- and Semi-Supervised Semantic Segmentation of Biomedical Images
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yanfeng Zhou, Jiaxing Huang, Chenlong Wang, Le Song, Ge Yang
    * Abstract: Fully- and semi-supervised semantic segmentation of biomedical images have been advanced with the development of deep neural networks (DNNs). So far, however, DNN models are usually designed to support one of these two learning schemes, unified models that support both fully- and semi-supervised segmentation remain limited. Furthermore, few fully-supervised models focus on the intrinsic low frequency (LF) and high frequency (HF) information of images to improve performance. Perturbations in consistency-based semi-supervised models are often artificially designed. They may introduce negative learning bias that are not beneficial for training. In this study, we propose a wavelet-based LF and HF fusion model XNet, which supports both fully- and semi-supervised semantic segmentation and outperforms state-of-the-art models in both fields. It emphasizes extracting LF and HF information for consistency training to alleviate the learning bias caused by artificial perturbations. Extensive experiments on two 2D and two 3D datasets demonstrate the effectiveness of our model. Code is available at https://github.com/Yanfeng-Zhou/XNet.

count=1
* Unlocking Comparative Plant Scoring with Siamese Neural Networks and Pairwise Pseudo Labelling
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/CVPPA/html/Hartley_Unlocking_Comparative_Plant_Scoring_with_Siamese_Neural_Networks_and_Pairwise_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/CVPPA/papers/Hartley_Unlocking_Comparative_Plant_Scoring_with_Siamese_Neural_Networks_and_Pairwise_ICCVW_2023_paper.pdf)]
    * Title: Unlocking Comparative Plant Scoring with Siamese Neural Networks and Pairwise Pseudo Labelling
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zane K. J. Hartley, Rob J. Lind, Nicholas Smith, Bob Collison, Andrew P. French
    * Abstract: Phenotypic assessment of plants for herbicide discovery is a complex visual task and involves the comparison of a non-treated plant to those treated with herbicides to assign a phytotoxicity score. It is often subjective and difficult to quantify by human observers. Employing novel computer vision approaches using neural networks in order to be non-subjective and truly quantitative offers advantages for data quality, leading to improved decision making. In this paper we present a deep learning approach for comparative plant assessment using Siamese neural networks, an architecture that takes pairs of images as inputs, and we overcome the hurdles of data collection by proposing a novel pseudo-labelling approach for combining different pairs of input images. We demonstrate a high level of accuracy with this method, comparable to human scoring, and present a series of experiments grading Amaranthus retroflexus weeds using our trained model.

count=1
* An Interactive Method for Adaptive Acquisition in Reflectance Transformation Imaging for Cultural Heritage
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/e-Heritage/html/Khawaja_An_Interactive_Method_for_Adaptive_Acquisition_in_Reflectance_Transformation_Imaging_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/e-Heritage/papers/Khawaja_An_Interactive_Method_for_Adaptive_Acquisition_in_Reflectance_Transformation_Imaging_ICCVW_2023_paper.pdf)]
    * Title: An Interactive Method for Adaptive Acquisition in Reflectance Transformation Imaging for Cultural Heritage
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Muhammad Arsalan Khawaja, Sony George, Franck Marzani, Jon Yngve Hardeberg, Alamin Mansouri
    * Abstract: This paper investigates the optimization of acquisition in Reflectance Transformation Imaging (RTI). Current methods for RTI acquisition are either computationally expensive or impractical, which leads to continued reliance on conventional classical methods like homogenous equally spaced methods in museums. We propose a methodology that is aimed at dynamic collaboration between automated analysis and cultural heritage expert knowledge to obtain optimized light positions. Our approach is cost-effective and adaptive to both linear and non-linear reflectance profile scenarios. The practical contribution of research in this field has a considerable impact on the cultural heritage context and beyond.

count=1
* Tracing the Influence of Predecessors on Trajectory Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/ROAD%2B%2B/html/Liu_Tracing_the_Influence_of_Predecessors_on_Trajectory_Prediction_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/ROAD++/papers/Liu_Tracing_the_Influence_of_Predecessors_on_Trajectory_Prediction_ICCVW_2023_paper.pdf)]
    * Title: Tracing the Influence of Predecessors on Trajectory Prediction
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Mengmeng Liu, Hao Cheng, Michael Ying Yang
    * Abstract: In real-world traffic scenarios, agents such as pedestrians and car drivers often observe neighboring agents who exhibit similar behavior as examples and then mimic their actions to some extent in their own behavior. This information can serve as prior knowledge for trajectory prediction, which is unfortunately largely overlooked in current trajectory prediction models. This paper introduces a novel Predecessor-and-Successor (PnS) method that incorporates a predecessor tracing module to model the influence of predecessors (identified from concurrent neighboring agents) on the successor (target agent) within the same scene. The method utilizes the moving patterns of these predecessors to guide the predictor in trajectory prediction. PnS effectively aligns the motion encodings of the successor with multiple potential predecessors in a probabilistic manner, facilitating the decoding process. We demonstrate the effectiveness of PnS by integrating it into a graph-based predictor for pedestrian trajectory prediction on the ETH/UCY datasets, resulting in a new state-of-the-art performance. Furthermore, we replace the HD map-based scene-context module with our PnS method in a transformer-based predictor for vehicle trajectory prediction on the nuScenes dataset, showing that the predictor maintains good prediction performance even without relying on any map information.

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
* Probabilistic Vehicle Reconstruction Using a Multi-Task CNN
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/CVRSUAD/Coenen_Probabilistic_Vehicle_Reconstruction_Using_a_Multi-Task_CNN_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Coenen_Probabilistic_Vehicle_Reconstruction_Using_a_Multi-Task_CNN_ICCVW_2019_paper.pdf)]
    * Title: Probabilistic Vehicle Reconstruction Using a Multi-Task CNN
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Max Coenen, Franz Rottensteiner
    * Abstract: The retrieval of the 3D pose and shape of objects from images is an ill-posed problem. A common way to object reconstruction is to match entities such as keypoints, edges, or contours of a deformable 3D model, used as shape prior, to their corresponding entities inferred from the image. However, such approaches are highly sensitive to model initialisation, imprecise keypoint localisations and/or illumination conditions. In this paper, we present a probabilistic approach for shape-aware 3D vehicle reconstruction from stereo images that leverages the outputs of a novel multi-task CNN. Specifically, we train a CNN that outputs probability distributions for the vehicle's orientation and for both, vehicle keypoints and wireframe edges. Together with 3D stereo information we integrate the predicted distributions into a common probabilistic framework. We believe that the CNN-based detection of wireframe edges reduces the sensitivity to illumination conditions and object contrast and that using the raw probability maps instead of inferring keypoint positions reduces the sensitivity to keypoint localisation errors. We show that our method achieves state-of-the-art results, evaluating our method on the challenging KITTI benchmark and on our own new 'Stereo-Vehicle' dataset.

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
* Complete Moving Object Detection in the Context of Robust Subspace Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/RSL-CV/Sultana_Complete_Moving_Object_Detection_in_the_Context_of_Robust_Subspace_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/RSL-CV/Sultana_Complete_Moving_Object_Detection_in_the_Context_of_Robust_Subspace_ICCVW_2019_paper.pdf)]
    * Title: Complete Moving Object Detection in the Context of Robust Subspace Learning
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Maryam Sultana, Arif Mahmood, Thierry Bouwmans, Soon Ki Jung
    * Abstract: Complete moving object detection plays a vital role in many applications of computer vision. For instance, depth estimation, scene understanding, object interaction, semantic segmentation, accident detection and avoidance in case of moving vehicles on a highway. However, it becomes challenging in the presence of dynamic backgrounds, camouflage, bootstrapping, varying illumination conditions, and noise. Over the past decade, robust subspace learning based methods addressed the moving objects detection problem with excellent performance. However, the moving objects detected by these methods are incomplete, unable to generate the occluded parts. Indeed, complete or occlusion-free moving object detection is still challenging for these methods. In the current work, we address this challenge by proposing a conditional Generative Adversarial Network (cGAN) conditioned on non-occluded moving object pixels during training. It therefore learns the subspace spanned by the moving objects covering all the dynamic variations and semantic information. While testing, our proposed Complete cGAN (CcGAN) is able to generate complete occlusion free moving objects in challenging conditions. The experimental evaluations of our proposed method are performed on SABS benchmark dataset and compared with 14 state-of-the-art methods, including both robust subspace and deep learning based methods. Our experiments demonstrate the superiority of our proposed model over both types of existing methods.

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
* RethNet: Object-by-Object Learning for Detecting Facial Skin Problems
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/VRMI/Bekmirzaev_RethNet_Object-by-Object_Learning_for_Detecting_Facial_Skin_Problems_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Bekmirzaev_RethNet_Object-by-Object_Learning_for_Detecting_Facial_Skin_Problems_ICCVW_2019_paper.pdf)]
    * Title: RethNet: Object-by-Object Learning for Detecting Facial Skin Problems
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Shohrukh Bekmirzaev, Seoyoung Oh, Sangwook Yo
    * Abstract: Semantic segmentation is a hot topic in computer vision where the most challenging tasks of object detection and recognition have been handling by the success of semantic segmentation approaches. We propose a concept of objectby-object learning technique to detect 11 types of facial skin lesions using semantic segmentation methods. Detecting individual skin lesion in a dense group is a challenging task, because of ambiguities in the appearance of the visual data. We observe that there exist co-occurrent visual relations between object classes (e.g., wrinkle and age spot, or papule and whitehead, etc.). In fact, rich contextual information significantly helps to handle the issue. Therefore, we propose REthinker blocks that are composed of the locally constructed convLSTM/Conv3D layers and SE module as a one-shot attention mechanism whose responsibility is to increase network's sensitivity in the local and global contextual representation that supports to capture ambiguously appeared objects and co-occurrence interactions between object classes. Experiments show that our proposed model reached MIoU of 79.46% on the test of a prepared dataset, representing a 15.34% improvement over Deeplab v3+ (MIoU of 64.12%).

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
* Behind the Scenes: What Moving Targets Reveal about Static Scene Geometry
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/html/Taylor_Behind_the_Scenes_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/papers/Taylor_Behind_the_Scenes_2013_ICCV_paper.pdf)]
    * Title: Behind the Scenes: What Moving Targets Reveal about Static Scene Geometry
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Geoffrey Taylor, Fei Mai
    * Abstract: Reasoning about 3D scene structure is an important component of visual scene understanding. Often, reasoning proceeds from low-level cues without resorting to full 3D reconstruction. However, existing geometric cues may require multiple viewpoints, supervised training, constraints on scene structure or information from auxiliary sensors. To address these limitations, this paper demonstrates how geometric context for a single static camera can be recovered from the location and shape of moving foreground targets. In particular, we propose methods to compute the likelihood of a static occlusion boundary and floor region at each pixel. Importantly, these cues do not require supervised training, or prior knowledge of camera geometry or scene structure. Finally, we show how the proposed geometric cues can be used to infer an ordinal depth map and demonstrate its use in compositing with correct occlusion handling.

count=1
* Synthetic Examples Improve Generalization for Rare Classes
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Beery_Synthetic_Examples_Improve_Generalization_for_Rare_Classes_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Beery_Synthetic_Examples_Improve_Generalization_for_Rare_Classes_WACV_2020_paper.pdf)]
    * Title: Synthetic Examples Improve Generalization for Rare Classes
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Sara Beery,  Yang Liu,  Dan Morris,  Jim Piavis,  Ashish Kapoor,  Neel Joshi,  Markus Meister,  Pietro Perona
    * Abstract: The ability to detect and classify rare occurrences in images has important applications -- for example, counting rare and endangered species when studying biodiversity, or detecting infrequent traffic scenarios that pose a danger to self-driving cars. Few-shot learning is an open problem: current computer vision systems struggle to categorize objects they have seen only rarely during training, and collecting a sufficient number of training examples of rare events is often challenging and expensive, and sometimes outright impossible. We explore in depth an approach to this problem: complementing the few available training images with ad-hoc simulated data. Our testbed is animal species classification, which has a real-world long-tailed distribution. We analyze the effect of different axes of variation in simulation, such as pose, lighting, model, and simulation method, and we prescribe best practices for efficiently incorporating simulated data for real-world performance gain. Our experiments reveal that synthetic data can considerably reduce error rates for classes that are rare, that as the amount of simulated data is increased, accuracy on the target class improves, and that high variation of simulated data provides maximum performance gain.

count=1
* Single Satellite Optical Imagery Dehazing using SAR Image Prior Based on conditional Generative Adversarial Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.pdf)]
    * Title: Single Satellite Optical Imagery Dehazing using SAR Image Prior Based on conditional Generative Adversarial Networks
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Binghui Huang,  Li Zhi,  Chao Yang,  Fuchun Sun,  Yixu Song
    * Abstract: Satellite image dehazing aims at precisely retrieving the real situations of the obscured parts from the hazy remote sensing (RS) images, which is a challenging task since the hazy regions contain both ground features and haze components. Many approaches of removing haze focus on processing multi-spectral or RGB images, whereas few of them utilize multi-sensor data. The multi-sensor data fusion is significant to provide auxiliary information since RGB images are sensitive to atmospheric conditions. In this paper, a dataset called SateHaze1k is established and composed of 1200 pairs clear Synthetic Aperture Radar (SAR), hazy RGB, and corresponding ground truth images, which are divided into three degrees of the haze, i.e. thin, moderate, and thick fog. Moreover, we propose a novel fusion dehazing method to directly restore the haze-free RS images by using an end-to-end conditional generative adversarial network(cGAN). The proposed network combines the information of both RGB and SAR images to eliminate the image blurring. Besides, the dilated residual blocks of the generator can also sufficiently improve the dehazing effects. Our experiments demonstrate that the proposed method, which fuses the information of different sensors applied to the cloudy conditions, can achieve more precise results than other baseline models.

count=1
* StressNet: Detecting Stress in Thermal Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Kumar_StressNet_Detecting_Stress_in_Thermal_Videos_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Kumar_StressNet_Detecting_Stress_in_Thermal_Videos_WACV_2021_paper.pdf)]
    * Title: StressNet: Detecting Stress in Thermal Videos
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Satish Kumar, A S M Iftekhar, Michael Goebel, Tom Bullock, Mary H. MacLean, Michael B. Miller, Tyler Santander, Barry Giesbrecht, Scott T. Grafton, B.S. Manjunath
    * Abstract: Precise measurement of physiological signals is critical for the effective monitoring of human vital signs. Recent developments in computer vision have demonstrated that signals such as pulse rate and respiration rate can be extracted from digital video of humans, increasing the possibility of contact-less monitoring. This paper presents a novel approach to obtaining physiological signals and classifying stress states from thermal video. The proposed net-work "StressNet", features a hybrid emission representation model that models the direct emission and absorption of heat by the skin and underlying blood vessels. This results in an information-rich feature representation of the face, which is used by spatio-temporal networks for recon-structing the ISTI ( Initial Systolic Time Interval : a measure of change in cardiac sympathetic activity that is considered to be a quantitative index of stress in humans). The recon-structed ISTI signal is fed to a stress-detection model to detect and classify the individual's stress state (i.e. stress or no stress). A detailed evaluation demonstrates that Stress-Net achieves a mean square error of 5.845 ms for predicting the ISTI signal and an average precision of 0.842 for stress detection.

count=1
* Vid2Int: Detecting Implicit Intention From Long Dialog Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Xu_Vid2Int_Detecting_Implicit_Intention_From_Long_Dialog_Videos_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Xu_Vid2Int_Detecting_Implicit_Intention_From_Long_Dialog_Videos_WACV_2021_paper.pdf)]
    * Title: Vid2Int: Detecting Implicit Intention From Long Dialog Videos
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Xiaoli Xu, Yao Lu, Zhiwu Lu, Tao Xiang
    * Abstract: Detecting subtle intention such as deception and subtext of a person in a long dialog video, or implicit intention detection (IID), is a challenging problem. The transcript (textual cues) often reveals little, so audio-visual cues including voice tone as well as facial and body behaviour are the main focuses for automated IID. Contextual cues are also crucial, since a person's implicit intentions are often correlated and context-dependent when the person moves from one question-answer pair to the next. However, no such dataset exists which contains fine-grained question-answer pair (video segment) level annotation. The first contribution of this work is thus a new benchmark dataset, called Vid2Int-Deception to fill this gap. A novel multi-grain representation model is also proposed to capture the subtle movement changes of eyes, face, and body (relevant for inferring intention) from a long dialog video. Moreover, to model the temporal correlation between the implicit intentions across video segments, we propose a Video-to-Intention network (Vid2Int) based on attentive recurrent neural network (RNN). Extensive experiments show that our model achieves state-of-the-art results.

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
* Learned Event-Based Visual Perception for Improved Space Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Salvatore_Learned_Event-Based_Visual_Perception_for_Improved_Space_Object_Detection_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Salvatore_Learned_Event-Based_Visual_Perception_for_Improved_Space_Object_Detection_WACV_2022_paper.pdf)]
    * Title: Learned Event-Based Visual Perception for Improved Space Object Detection
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Nikolaus Salvatore, Justin Fletcher
    * Abstract: The detection of dim artificial Earth satellites using ground-based electro-optical sensors, particularly in the presence of background light, is technologically challenging. This perceptual task is foundational to our understanding of the space environment, and grows in importance as the number, variety, and dynamism of space objects increases. We present a hybrid image- and event-based architecture that leverages dynamic vision sensing technology to detect resident space objects in geosynchronous Earth orbit. Given the asynchronous, one-dimensional image data supplied by a dynamic vision sensor, our architecture applies conventional image feature extractors to integrated, two-dimensional frames in conjunction with point-cloud feature extractors, such as PointNet, in order to increase detection performance for dim objects in scenes with high background activity. In addition, an end-to-end event-based imaging simulator is developed to both produce data for model training as well as approximate the optimal sensor parameters for event-based sensing in the context of electro-optical telescope imagery. Experimental results confirm that the inclusion of point-cloud feature extractors increases recall for dim objects in the high-background regime.

count=1
* AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Zhang_AuxAdapt_Stable_and_Efficient_Test-Time_Adaptation_for_Temporally_Consistent_Video_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Zhang_AuxAdapt_Stable_and_Efficient_Test-Time_Adaptation_for_Temporally_Consistent_Video_WACV_2022_paper.pdf)]
    * Title: AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Yizhe Zhang, Shubhankar Borse, Hong Cai, Fatih Porikli
    * Abstract: In video segmentation, generating temporally consistent results across frames is as important as achieving frame-wise accuracy. Existing methods rely either on optical flow regularization or fine-tuning with test data to attain temporal consistency. However, optical flow is not always avail-able and reliable. Besides, it is expensive to compute. Fine-tuning the original model in test time is cost sensitive. This paper presents an efficient, intuitive, and unsupervised online adaptation method, AuxAdapt, for improving the temporal consistency of most neural network models. It does not require optical flow and only takes one pass of the video. Since inconsistency mainly arises from the model's uncertainty in its output, we propose an adaptation scheme where the model learns from its own segmentation decisions as it streams a video, which allows producing more confident and temporally consistent labeling for similarly-looking pixels across frames. For stability and efficiency, we leverage a small auxiliary segmentation network (AuxNet) to assist with this adaptation. More specifically, AuxNet readjusts the decision of the original segmentation network (Main-Net) by adding its own estimations to that of MainNet. At every frame, only AuxNet is updated via back-propagation while keeping MainNet fixed. We extensively evaluate our test-time adaptation approach on standard video benchmarks, including Cityscapes, CamVid, and KITTI. The results demonstrate that our approach provides label-wise accurate, temporally consistent, and computationally efficient adaptation (5+ folds overhead reduction comparing to state-of-the-art test-time adaptation methods).

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
* Ev-NeRF: Event Based Neural Radiance Field
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Hwang_Ev-NeRF_Event_Based_Neural_Radiance_Field_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Hwang_Ev-NeRF_Event_Based_Neural_Radiance_Field_WACV_2023_paper.pdf)]
    * Title: Ev-NeRF: Event Based Neural Radiance Field
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Inwoo Hwang, Junho Kim, Young Min Kim
    * Abstract: We present Ev-NeRF, a Neural Radiance Field derived from event data. While event cameras can measure subtle brightness changes in high frame rates, the measurements in low lighting or extreme motion suffer from significant domain discrepancy with complex noise. As a result, the performance of event-based vision tasks does not transfer to challenging environments, where the event cameras are expected to thrive over normal cameras. We find that the multi-view consistency of NeRF provides a powerful self-supervision signal for eliminating spurious measurements and extracting the consistent underlying structure despite highly noisy input. Instead of posed images of the original NeRF, the input to Ev-NeRF is the event measurements accompanied by the movements of the sensors. Using the loss function that reflects the measurement model of the sensor, Ev-NeRF creates an integrated neural volume that summarizes the unstructured and sparse data points captured for about 2-4 seconds. The generated neural volume can also produce intensity images from novel views with reasonable depth estimates, which can serve as a high-quality input to various vision-based tasks. Our results show that Ev-NeRF achieves competitive performance for intensity image reconstruction under extreme noise conditions and high-dynamic-range imaging.

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
* Self-Supervised Relative Pose With Homography Model-Fitting in the Loop
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Muller_Self-Supervised_Relative_Pose_With_Homography_Model-Fitting_in_the_Loop_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Muller_Self-Supervised_Relative_Pose_With_Homography_Model-Fitting_in_the_Loop_WACV_2023_paper.pdf)]
    * Title: Self-Supervised Relative Pose With Homography Model-Fitting in the Loop
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Bruce R. Muller, William A. P. Smith
    * Abstract: We propose a self-supervised method for relative pose estimation for road scenes. By exploiting the approximate planarity of the local ground plane, we can extract a self-supervision signal via cross-projection between images using a homography derived from estimated ground-relative pose. We augment cross-projected perceptual loss by including classical image alignment in the network training loop. We use pretrained semantic segmentation and optical flow to extract ground plane correspondences between approximately aligned images and RANSAC to find the best fitting homography. By decomposing to ground-relative pose, we obtain pseudo labels that can be used for direct supervision. We show that this extremely simple geometric model is competitive for visual odometry with much more complex self-supervised methods that must learn depth estimation in conjunction with relative pose. Code and result videos: github.com/brucemuller/homographyVO.

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
* The Background Also Matters: Background-Aware Motion-Guided Objects Discovery
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Kara_The_Background_Also_Matters_Background-Aware_Motion-Guided_Objects_Discovery_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kara_The_Background_Also_Matters_Background-Aware_Motion-Guided_Objects_Discovery_WACV_2024_paper.pdf)]
    * Title: The Background Also Matters: Background-Aware Motion-Guided Objects Discovery
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Sandra Kara, Hejer Ammar, Florian Chabot, Quoc-Cuong Pham
    * Abstract: Recent works have shown that objects discovery can largely benefit from the inherent motion information in video data. However, these methods lack a proper background processing, resulting in an over-segmentation of the non-object regions into random segments. This is a critical limitation given the unsupervised setting, where object segments and noise are not distinguishable. To address this limitation we propose BMOD, a Background-aware Motion-guided Objects Discovery method. Concretely, we leverage masks of moving objects extracted from optical flow and design a learning mechanism to extend them to the true foreground composed of both moving and static objects. The background, a complementary concept of the learned foreground class, is then isolated in the object discovery process. This enables a joint learning of the objects discovery task and the object/non-object separation. The conducted experiments on synthetic and real-world datasets show that integrating our background handling with various cutting-edge methods brings each time a considerable improvement. Specifically, we improve the objects discovery performance with a large margin, while establishing a strong baseline for object/non-object separation.

count=1
* Spatio-Temporal Filter Analysis Improves 3D-CNN for Action Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Kobayashi_Spatio-Temporal_Filter_Analysis_Improves_3D-CNN_for_Action_Classification_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kobayashi_Spatio-Temporal_Filter_Analysis_Improves_3D-CNN_for_Action_Classification_WACV_2024_paper.pdf)]
    * Title: Spatio-Temporal Filter Analysis Improves 3D-CNN for Action Classification
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Takumi Kobayashi, Jiaxing Ye
    * Abstract: As 2D-CNNs are growing in image recognition literature, 3D-CNNs are enthusiastically applied to video action recognition. While spatio-temporal (3D) convolution successfully stems from spatial (2D) convolution, it is still unclear how the convolution works for encoding temporal motion patterns in 3D-CNNs. In this paper, we shed light on the mechanism of feature extraction through analyzing the spatio-temporal filters from a temporal viewpoint. The analysis not only describes characteristics of the two action datasets, Something-Something-v2 (SSv2) and Kinetics-400, but also reveals how temporal dynamics are characterized through stacked spatio-temporal convolutions. Based on the analysis, we propose methods to improve temporal feature extraction, covering temporal filter representation and temporal data augmentation. The proposed method contributes to enlarging temporal receptive field of 3D-CNN without touching its fundamental architecture, thus keeping the computation cost. In the experiments on action classification using SSv2 and Kinetics-400, it produces favorable performance improvement of 3D-CNNs.

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
* ICF-SRSR: Invertible Scale-Conditional Function for Self-Supervised Real-World Single Image Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Neshatavar_ICF-SRSR_Invertible_Scale-Conditional_Function_for_Self-Supervised_Real-World_Single_Image_Super-Resolution_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Neshatavar_ICF-SRSR_Invertible_Scale-Conditional_Function_for_Self-Supervised_Real-World_Single_Image_Super-Resolution_WACV_2024_paper.pdf)]
    * Title: ICF-SRSR: Invertible Scale-Conditional Function for Self-Supervised Real-World Single Image Super-Resolution
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Reyhaneh Neshatavar, Mohsen Yavartanoo, Sanghyun Son, Kyoung Mu Lee
    * Abstract: Single image super-resolution (SISR) is a challenging ill-posed problem that aims to up-sample a given low-resolution (LR) image to a high-resolution (HR) counterpart. Due to the difficulty in obtaining real LR-HR training pairs, recent approaches are trained on simulated LR images degraded by simplified down-sampling operators, e.g., bicubic. Such an approach can be problematic in practice due to the large gap between the synthesized and real-world LR images. To alleviate the issue, we propose a novel Invertible scale-Conditional Function (ICF), which can scale an input image and then restore the original input with different scale conditions. Using the proposed ICF, we construct a novel self-supervised SISR framework (ICF-SRSR) to handle the real-world SR task without using any paired/unpaired training data. Furthermore, our ICF-SRSR can generate realistic and feasible LR-HR pairs, which can make existing supervised SISR networks more robust. Extensive experiments demonstrate the effectiveness of our method in handling SISR in a fully self-supervised manner. Our ICF-SRSR demonstrates superior performance compared to the existing methods trained on synthetic paired images in real-world scenarios and exhibits comparable performance compared to state-of-the-art supervised/unsupervised methods on public benchmark datasets. The code is available from this link.

count=1
* ENIGMA-51: Towards a Fine-Grained Understanding of Human Behavior in Industrial Scenarios
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Ragusa_ENIGMA-51_Towards_a_Fine-Grained_Understanding_of_Human_Behavior_in_Industrial_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Ragusa_ENIGMA-51_Towards_a_Fine-Grained_Understanding_of_Human_Behavior_in_Industrial_WACV_2024_paper.pdf)]
    * Title: ENIGMA-51: Towards a Fine-Grained Understanding of Human Behavior in Industrial Scenarios
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Francesco Ragusa, Rosario Leonardi, Michele Mazzamuto, Claudia Bonanno, Rosario Scavo, Antonino Furnari, Giovanni Maria Farinella
    * Abstract: ENIGMA-51 is a new egocentric dataset acquired in an industrial scenario by 19 subjects who followed instructions to complete the repair of electrical boards using industrial tools (e.g., electric screwdriver) and equipments (e.g., oscilloscope). The 51 egocentric video sequences are densely annotated with a rich set of labels that enable the systematic study of human behavior in the industrial domain. We provide benchmarks on four tasks related to human behavior: 1) untrimmed temporal detection of human-object interactions, 2) egocentric human-object interaction detection, 3) short-term object interaction anticipation and 4) natural language understanding of intents and entities. Baseline results show that the ENIGMA-51 dataset poses a challenging benchmark to study human behavior in industrial scenarios. We publicly release the dataset at https://iplab.dmi.unict.it/ENIGMA-51.

count=1
* Neuronal Spike Generation Mechanism as an Oversampling, Noise-shaping A-to-D converter
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/36660e59856b4de58a219bcf4e27eba3-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/36660e59856b4de58a219bcf4e27eba3-Paper.pdf)]
    * Title: Neuronal Spike Generation Mechanism as an Oversampling, Noise-shaping A-to-D converter
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Dmitri Chklovskii, Daniel Soudry
    * Abstract: We explore the hypothesis that the neuronal spike generation mechanism is an analog-to-digital converter, which rectifies low-pass filtered summed synaptic currents and encodes them into spike trains linearly decodable in post-synaptic neurons. To digitally encode an analog current waveform, the sampling rate of the spike generation mechanism must exceed its Nyquist rate. Such oversampling is consistent with the experimental observation that the precision of the spike-generation mechanism is an order of magnitude greater than the cut-off frequency of dendritic low-pass filtering. To achieve additional reduction in the error of analog-to-digital conversion, electrical engineers rely on noise-shaping. If noise-shaping were used in neurons, it would introduce correlations in spike timing to reduce low-frequency (up to Nyquist) transmission error at the cost of high-frequency one (from Nyquist to sampling rate). Using experimental data from three different classes of neurons, we demonstrate that biological neurons utilize noise-shaping. We also argue that rectification by the spike-generation mechanism may improve energy efficiency and carry out de-noising. Finally, the zoo of ion channels in neurons may be viewed as a set of predictors, various subsets of which are activated depending on the statistics of the input current.

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
* Balancing Suspense and Surprise: Timely Decision Making with Endogenous Information Acquisition
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2016/hash/fe70c36866add1572a8e2b96bfede7bf-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2016/file/fe70c36866add1572a8e2b96bfede7bf-Paper.pdf)]
    * Title: Balancing Suspense and Surprise: Timely Decision Making with Endogenous Information Acquisition
    * Publisher: NeurIPS
    * Publication Date: `2016`
    * Authors: Ahmed M. Alaa, Mihaela van der Schaar
    * Abstract: We develop a Bayesian model for decision-making under time pressure with endogenous information acquisition. In our model, the decision-maker decides when to observe (costly) information by sampling an underlying continuous-time stochastic process (time series) that conveys information about the potential occurrence/non-occurrence of an adverse event which will terminate the decision-making process. In her attempt to predict the occurrence of the adverse event, the decision-maker follows a policy that determines when to acquire information from the time series (continuation), and when to stop acquiring information and make a final prediction (stopping). We show that the optimal policy has a "rendezvous" structure, i.e. a structure in which whenever a new information sample is gathered from the time series, the optimal "date" for acquiring the next sample becomes computable. The optimal interval between two information samples balances a trade-off between the decision maker’s "surprise", i.e. the drift in her posterior belief after observing new information, and "suspense", i.e. the probability that the adverse event occurs in the time interval between two information samples. Moreover, we characterize the continuation and stopping regions in the decision-maker’s state-space, and show that they depend not only on the decision-maker’s beliefs, but also on the "context", i.e. the current realization of the time series.

count=1
* Learning a Multi-View Stereo Machine
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/9c838d2e45b2ad1094d42f4ef36764f6-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/9c838d2e45b2ad1094d42f4ef36764f6-Paper.pdf)]
    * Title: Learning a Multi-View Stereo Machine
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Abhishek Kar, Christian Häne, Jitendra Malik
    * Abstract: We present a learnt system for multi-view stereopsis. In contrast to recent learning based methods for 3D reconstruction, we leverage the underlying 3D geometry of the problem through feature projection and unprojection along viewing rays. By formulating these operations in a differentiable manner, we are able to learn the system end-to-end for the task of metric 3D reconstruction. End-to-end learning allows us to jointly reason about shape priors while conforming to geometric constraints, enabling reconstruction from much fewer images (even a single image) than required by classical approaches as well as completion of unseen surfaces. We thoroughly evaluate our approach on the ShapeNet dataset and demonstrate the benefits over classical approaches and recent learning based methods.

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
* Infinite-Horizon Gaussian Processes
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2018/hash/b865367fc4c0845c0682bd466e6ebf4c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2018/file/b865367fc4c0845c0682bd466e6ebf4c-Paper.pdf)]
    * Title: Infinite-Horizon Gaussian Processes
    * Publisher: NeurIPS
    * Publication Date: `2018`
    * Authors: Arno Solin, James Hensman, Richard E. Turner
    * Abstract: Gaussian processes provide a flexible framework for forecasting, removing noise, and interpreting long temporal datasets. State space modelling (Kalman filtering) enables these non-parametric models to be deployed on long datasets by reducing the complexity to linear in the number of data points. The complexity is still cubic in the state dimension m which is an impediment to practical application. In certain special cases (Gaussian likelihood, regular spacing) the GP posterior will reach a steady posterior state when the data are very long. We leverage this and formulate an inference scheme for GPs with general likelihoods, where inference is based on single-sweep EP (assumed density filtering). The infinite-horizon model tackles the cubic cost in the state dimensionality and reduces the cost in the state dimension m to O(m^2) per data point. The model is extended to online-learning of hyperparameters. We show examples for large finite-length modelling problems, and present how the method runs in real-time on a smartphone on a continuous data stream updated at 100 Hz.

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
* AGEM: Solving Linear Inverse Problems via Deep Priors and Sampling
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/49182f81e6a13cf5eaa496d51fea6406-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/49182f81e6a13cf5eaa496d51fea6406-Paper.pdf)]
    * Title: AGEM: Solving Linear Inverse Problems via Deep Priors and Sampling
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Bichuan Guo, Yuxing Han, Jiangtao Wen
    * Abstract: In this paper we propose to use a denoising autoencoder (DAE) prior to simultaneously solve a linear inverse problem and estimate its noise parameter. Existing DAE-based methods estimate the noise parameter empirically or treat it as a tunable hyper-parameter. We instead propose autoencoder guided EM, a probabilistically sound framework that performs Bayesian inference with intractable deep priors. We show that efficient posterior sampling from the DAE can be achieved via Metropolis-Hastings, which allows the Monte Carlo EM algorithm to be used. We demonstrate competitive results for signal denoising, image deblurring and image devignetting. Our method is an example of combining the representation power of deep learning with uncertainty quantification from Bayesian statistics.

count=1
* Flexible information routing in neural populations through stochastic comodulation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/516b38afeee70474b04881a633728b15-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/516b38afeee70474b04881a633728b15-Paper.pdf)]
    * Title: Flexible information routing in neural populations through stochastic comodulation
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Caroline Haimerl, Cristina Savin, Eero Simoncelli
    * Abstract: Humans and animals are capable of flexibly switching between a multitude of tasks, each requiring rapid, sensory-informed decision making. Incoming stimuli are processed by a hierarchy of neural circuits consisting of millions of neurons with diverse feature selectivity. At any given moment, only a small subset of these carry task-relevant information. In principle, downstream processing stages could identify the relevant neurons through supervised learning, but this would require many example trials. Such extensive learning periods are inconsistent with the observed flexibility of humans or animals, who can adjust to changes in task parameters or structure almost immediately. Here, we propose a novel solution based on functionally-targeted stochastic modulation. It has been observed that trial-to-trial neural activity is modulated by a shared, low-dimensional, stochastic signal that introduces task-irrelevant noise. Counter-intuitively this noise is preferentially targeted towards task-informative neurons, corrupting the encoded signal. However, we hypothesize that this modulation offers a solution to the identification problem, labeling task-informative neurons so as to facilitate decoding. We simulate an encoding population of spiking neurons whose rates are modulated by a shared stochastic signal, and show that a linear decoder with readout weights approximating neuron-specific modulation strength can achieve near-optimal accuracy. Such a decoder allows fast and flexible task-dependent information routing without relying on hardwired knowledge of the task-informative neurons (as in maximum likelihood) or unrealistically many supervised training trials (as in regression).

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
* Structure Learning with Side Information: Sample Complexity
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/e025b6279c1b88d3ec0eca6fcb6e6280-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/e025b6279c1b88d3ec0eca6fcb6e6280-Paper.pdf)]
    * Title: Structure Learning with Side Information: Sample Complexity
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Saurabh Sihag, Ali Tajer
    * Abstract: Graphical models encode the stochastic dependencies among random variables (RVs). The vertices represent the RVs, and the edges signify the conditional dependencies among the RVs. Structure learning is the process of inferring the edges by observing realizations of the RVs, and it has applications in a wide range of technological, social, and biological networks. Learning the structure of graphs when the vertices are treated in isolation from inferential information known about them is well-investigated. In a wide range of domains, however, often there exist additional inferred knowledge about the structure, which can serve as valuable side information. For instance, the gene networks that represent different subtypes of the same cancer share similar edges across all subtypes and also have exclusive edges corresponding to each subtype, rendering partially similar graphical models for gene expression in different cancer subtypes. Hence, an inferential decision regarding a gene network can serve as side information for inferring other related gene networks. When such side information is leveraged judiciously, it can translate to significant improvement in structure learning. Leveraging such side information can be abstracted as inferring structures of distinct graphical models that are {\sl partially} similar. This paper focuses on Ising graphical models, and considers the problem of simultaneously learning the structures of two {\sl partially} similar graphs, where any inference about the structure of one graph offers side information for the other graph. The bounded edge subclass of Ising models is considered, and necessary conditions (information-theoretic ), as well as sufficient conditions (algorithmic) for the sample complexity for achieving a bounded probability of error, are established. Furthermore, specific regimes are identified in which the necessary and sufficient conditions coincide, rendering the optimal sample complexity.

count=1
* Continuous Meta-Learning without Tasks
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/cc3f5463bc4d26bc38eadc8bcffbc654-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf)]
    * Title: Continuous Meta-Learning without Tasks
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: James Harrison, Apoorva Sharma, Chelsea Finn, Marco Pavone
    * Abstract: Meta-learning is a promising strategy for learning to efficiently learn using data gathered from a distribution of tasks. However, the meta-learning literature thus far has focused on the task segmented setting, where at train-time, offline data is assumed to be split according to the underlying task, and at test-time, the algorithms are optimized to learn in a single task. In this work, we enable the application of generic meta-learning algorithms to settings where this task segmentation is unavailable, such as continual online learning with unsegmented time series data. We present meta-learning via online changepoint analysis (MOCA), an approach which augments a meta-learning algorithm with a differentiable Bayesian changepoint detection scheme. The framework allows both training and testing directly on time series data without segmenting it into discrete tasks. We demonstrate the utility of this approach on three nonlinear meta-regression benchmarks as well as two meta-image-classification benchmarks.

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
* Neural Tangent Kernel Maximum Mean Discrepancy
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/348a38cd25abeab0e440f37510e9b1fa-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/348a38cd25abeab0e440f37510e9b1fa-Paper.pdf)]
    * Title: Neural Tangent Kernel Maximum Mean Discrepancy
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Xiuyuan Cheng, Yao Xie
    * Abstract: We present a novel neural network Maximum Mean Discrepancy (MMD) statistic by identifying a new connection between neural tangent kernel (NTK) and MMD. This connection enables us to develop a computationally efficient and memory-efficient approach to compute the MMD statistic and perform NTK based two-sample tests towards addressing the long-standing challenge of memory and computational complexity of the MMD statistic, which is essential for online implementation to assimilating new samples. Theoretically, such a connection allows us to understand the NTK test statistic properties, such as the Type-I error and testing power for performing the two-sample test, by adapting existing theories for kernel MMD. Numerical experiments on synthetic and real-world datasets validate the theory and demonstrate the effectiveness of the proposed NTK-MMD statistic.

count=1
* Learning the Structure of Large Networked Systems Obeying Conservation Laws
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/5e0347e19c51cfd0f6fe52f371004dfc-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/5e0347e19c51cfd0f6fe52f371004dfc-Paper-Conference.pdf)]
    * Title: Learning the Structure of Large Networked Systems Obeying Conservation Laws
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Anirudh Rayas, Rajasekhar Anguluri, Gautam Dasarathy
    * Abstract: Many networked systems such as electric networks, the brain, and social networks of opinion dynamics are known to obey conservation laws. Examples of this phenomenon include the Kirchoff laws in electric networks and opinion consensus in social networks. Conservation laws in networked systems are modeled as balance equations of the form $X = B^\ast Y$, where the sparsity pattern of $B^\ast \in \mathbb{R}^{p\times p}$ captures the connectivity of the network on $p$ nodes, and $Y, X \in \mathbb{R}^p$ are vectors of ''potentials'' and ''injected flows'' at the nodes respectively. The node potentials $Y$ cause flows across edges which aim to balance out the potential difference, and the flows $X$ injected at the nodes are extraneous to the network dynamics. In several practical systems, the network structure is often unknown and needs to be estimated from data to facilitate modeling, management, and control. To this end, one has access to samples of the node potentials $Y$, but only the statistics of the node injections $X$. Motivated by this important problem, we study the estimation of the sparsity structure of the matrix $B^\ast$ from $n$ samples of $Y$ under the assumption that the node injections $X$ follow a Gaussian distribution with a known covariance $\Sigma_X$. We propose a new $\ell_{1}$-regularized maximum likelihood estimator for tackling this problem in the high-dimensional regime where the size of the network may be vastly larger than the number of samples $n$. We show that this optimization problem is convex in the objective and admits a unique solution. Under a new mutual incoherence condition, we establish sufficient conditions on the triple $(n,p,d)$ for which exact sparsity recovery of $B^\ast$ is possible with high probability; $d$ is the degree of the underlying graph. We also establish guarantees for the recovery of $B^\ast$ in the element-wise maximum, Frobenius, and operator norms. Finally, we complement these theoretical results with experimental validation of the performance of the proposed estimator on synthetic and real-world data.

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
* Robust Lipschitz Bandits to Adversarial Corruptions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/238f3b98bbe998b4f2234443907fe663-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/238f3b98bbe998b4f2234443907fe663-Paper-Conference.pdf)]
    * Title: Robust Lipschitz Bandits to Adversarial Corruptions
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Yue Kang, Cho-Jui Hsieh, Thomas Chun Man Lee
    * Abstract: Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.

count=1
* Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/2ab3163ee384cd46baa7f1abb2b1bf19-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/2ab3163ee384cd46baa7f1abb2b1bf19-Paper-Conference.pdf)]
    * Title: Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Katie Luo, Zhenzhen Liu, Xiangyu Chen, Yurong You, Sagie Benaim, Cheng Perng Phoo, Mark Campbell, Wen Sun, Bharath Hariharan, Kilian Q. Weinberger
    * Abstract: Recent advances in machine learning have shown that Reinforcement Learning from Human Feedback (RLHF) can improve machine learning models and align them with human preferences. Although very successful for Large Language Models (LLMs), these advancements have not had a comparable impact in research for autonomous vehicles—where alignment with human expectations can be imperative. In this paper, we propose to adapt similar RL-based methods to unsupervised object discovery, i.e. learning to detect objects from LiDAR points without any training labels. Instead of labels, we use simple heuristics to mimic human feedback. More explicitly, we combine multiple heuristics into a simple reward function that positively correlates its score with bounding box accuracy, i.e., boxes containing objects are scored higher than those without. We start from the detector’s own predictions to explore the space and reinforce boxes with high rewards through gradient updates. Empirically, we demonstrate that our approach is not only more accurate, but also orders of magnitudes faster to train compared to prior works on object discovery. Code is available at https://github.com/katieluo88/DRIFT.

count=1
* Change point detection and inference in multivariate non-parametric models under mixing conditions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/42a0de6b8a1809ceba8fdad1661be06c-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/42a0de6b8a1809ceba8fdad1661be06c-Paper-Conference.pdf)]
    * Title: Change point detection and inference in multivariate non-parametric models under mixing conditions
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, OSCAR HERNAN MADRID PADILLA, Yi Yu
    * Abstract: This paper addresses the problem of localizing and inferring multiple change points, in non-parametric multivariate time series settings. Specifically, we consider a multivariate time series with potentially short-range dependence, whose underlying distributions have Hölder smooth densities and can change over time in a piecewise-constant manner. The change points, which correspond to the times when the distribution changes, are unknown. We present the limiting distributions of the change point estimators under the scenarios where the minimal jump size vanishes or remains constant. Such results have not been revealed in the literature in non-parametric change point settings. As byproducts, we develop a sharp estimator that can accurately localize the change points in multivariate non-parametric time series, and a consistent block-type long-run variance estimator. Numerical studies are provided to complement our theoretical findings.

count=1
* Adapting to Continuous Covariate Shift via Online Density Ratio Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/5cad96c4433955a2e76749ee74a424f5-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/5cad96c4433955a2e76749ee74a424f5-Paper-Conference.pdf)]
    * Title: Adapting to Continuous Covariate Shift via Online Density Ratio Estimation
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Yu-Jie Zhang, Zhen-Yu Zhang, Peng Zhao, Masashi Sugiyama
    * Abstract: Dealing with distribution shifts is one of the central challenges for modern machine learning. One fundamental situation is the covariate shift, where the input distributions of data change from the training to testing stages while the input-conditional output distribution remains unchanged. In this paper, we initiate the study of a more challenging scenario --- continuous covariate shift --- in which the test data appear sequentially, and their distributions can shift continuously. Our goal is to adaptively train the predictor such that its prediction risk accumulated over time can be minimized. Starting with the importance-weighted learning, we theoretically show the method works effectively if the time-varying density ratios of test and train inputs can be accurately estimated. However, existing density ratio estimation methods would fail due to data scarcity at each time step. To this end, we propose an online density ratio estimation method that can appropriately reuse historical information. Our method is proven to perform well by enjoying a dynamic regret bound, which finally leads to an excess risk guarantee for the predictor. Empirical results also validate the effectiveness.

count=1
* Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/677c8dc72c99482507323f313faf4738-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/677c8dc72c99482507323f313faf4738-Paper-Conference.pdf)]
    * Title: Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen, Fenglong Ma, Ting Wang
    * Abstract: Pre-trained language models (PLMs) have demonstrated remarkable performance as few-shot learners. However, their security risks under such settings are largely unexplored. In this work, we conduct a pilot study showing that PLMs as few-shot learners are highly vulnerable to backdoor attacks while existing defenses are inadequate due to the unique challenges of few-shot scenarios. To address such challenges, we advocate MDP, a novel lightweight, pluggable, and effective defense for PLMs as few-shot learners. Specifically, MDP leverages the gap between the masking-sensitivity of poisoned and clean samples: with reference to the limited few-shot data as distributional anchors, it compares the representations of given samples under varying masking and identifies poisoned samples as ones with significant variations. We show analytically that MDP creates an interesting dilemma for the attacker to choose between attack effectiveness and detection evasiveness. The empirical evaluation using benchmark datasets and representative attacks validates the efficacy of MDP. The code of MDP is publicly available.

count=1
* Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/6b8c6f846c3575e1d1ad496abea28826-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/6b8c6f846c3575e1d1ad496abea28826-Paper-Conference.pdf)]
    * Title: Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Congyue Deng, Jiahui Lei, William B Shen, Kostas Daniilidis, Leonidas J. Guibas
    * Abstract: Equivariance has gained strong interest as a desirable network property that inherently ensures robust generalization. However, when dealing with complex systems such as articulated objects or multi-object scenes, effectively capturing inter-part transformations poses a challenge, as it becomes entangled with the overall structure and local transformations. The interdependence of part assignment and per-part group action necessitates a novel equivariance formulation that allows for their co-evolution. In this paper, we present Banana, a Banach fixed-point network for equivariant segmentation with inter-part equivariance by construction. Our key insight is to iteratively solve a fixed-point problem, where point-part assignment labels and per-part SE(3)-equivariance co-evolve simultaneously. We provide theoretical derivations of both per-step equivariance and global convergence, which induces an equivariant final convergent state. Our formulation naturally provides a strict definition of inter-part equivariance that generalizes to unseen inter-part configurations. Through experiments conducted on both articulated objects and multi-object scans, we demonstrate the efficacy of our approach in achieving strong generalization under inter-part transformations, even when confronted with substantial changes in pointcloud geometry and topology.

count=1
* Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/7abbcb05a5d55157ede410bb718e32d7-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/7abbcb05a5d55157ede410bb718e32d7-Paper-Conference.pdf)]
    * Title: Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Models
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Yule Wang, Zijing Wu, Chengrui Li, Anqi Wu
    * Abstract: In the field of behavior-related brain computation, it is necessary to align raw neural signals against the drastic domain shift among them. A foundational framework within neuroscience research posits that trial-based neural population activities rely on low-dimensional latent dynamics, thus focusing on the latter greatly facilitates the alignment procedure. Despite this field's progress, existing methods ignore the intrinsic spatio-temporal structure during the alignment phase. Hence, their solutions usually lead to poor quality in latent dynamics structures and overall performance. To tackle this problem, we propose an alignment method ERDiff, which leverages the expressivity of the diffusion model to preserve the spatio-temporal structure of latent dynamics. Specifically, the latent dynamics structures of the source domain are first extracted by a diffusion model. Then, under the guidance of this diffusion model, such structures are well-recovered through a maximum likelihood alignment procedure in the target domain. We first demonstrate the effectiveness of our proposed method on a synthetic dataset. Then, when applied to neural recordings from the non-human primate motor cortex, under both cross-day and inter-subject settings, our method consistently manifests its capability of preserving the spatio-temporal structure of latent dynamics and outperforms existing approaches in alignment goodness-of-fit and neural decoding performance.

count=1
* Perception Test: A Diagnostic Benchmark for Multimodal Video Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/8540fba4abdc7f9f7a7b1cc6cd60e409-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Perception Test: A Diagnostic Benchmark for Multimodal Video Models
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, joseph heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fréchette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, Joao Carreira
    * Abstract: We propose a novel multimodal video benchmark - the Perception Test - to evaluate the perception and reasoning skills of pre-trained multimodal models (e.g. Flamingo, BEiT-3, or GPT-4). Compared to existing benchmarks that focus on computational tasks (e.g. classification, detection or tracking), the Perception Test focuses on skills (Memory, Abstraction, Physics, Semantics) and types of reasoning (descriptive, explanatory, predictive, counterfactual) across video, audio, and text modalities, to provide a comprehensive and efficient evaluation tool. The benchmark probes pre-trained models for their transfer capabilities, in a zero-shot / few-shot or limited finetuning regime. For these purposes, the Perception Test introduces 11.6k real-world videos, 23s average length, designed to show perceptually interesting situations, filmed by around 100 participants worldwide. The videos are densely annotated with six types of labels (multiple-choice and grounded video question-answers, object and point tracks, temporal action and sound segments), enabling both language and non-language evaluations. The fine-tuning and validation splits of the benchmark are publicly available (CC-BY license), in addition to a challenge server with a held-out test split. Human baseline results compared to state-of-the-art video QA models show a significant gap in performance (91.4% vs 45.8%), suggesting that there is significant room for improvement in multimodal video understanding.Dataset, baselines code, and challenge server are available at https://github.com/deepmind/perception_test

count=1
* VidChapters-7M: Video Chapters at Scale
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/9b5c3e00d6ed30aad7adac9e7a664de1-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/9b5c3e00d6ed30aad7adac9e7a664de1-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: VidChapters-7M: Video Chapters at Scale
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid
    * Abstract: Segmenting untrimmed videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters-7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title. We benchmark both simple baselines as well as state-of-the-art video-language models on these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset.

count=1
* Disentangling Voice and Content with Self-Supervision for Speaker Recognition
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/9d276b0a087efdd2404f3295b26c24c1-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/9d276b0a087efdd2404f3295b26c24c1-Paper-Conference.pdf)]
    * Title: Disentangling Voice and Content with Self-Supervision for Speaker Recognition
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: TIANCHI LIU, Kong Aik Lee, Qiongqiong Wang, Haizhou Li
    * Abstract: For speaker recognition, it is difficult to extract an accurate speaker representation from speech because of its mixture of speaker traits and content. This paper proposes a disentanglement framework that simultaneously models speaker traits and content variability in speech. It is realized with the use of three Gaussian inference layers, each consisting of a learnable transition model that extracts distinct speech components. Notably, a strengthened transition model is specifically designed to model complex speech dynamics. We also propose a self-supervision method to dynamically disentangle content without the use of labels other than speaker identities. The efficacy of the proposed framework is validated via experiments conducted on the VoxCeleb and SITW datasets with 9.56\% and 8.24\% average reductions in EER and minDCF, respectively. Since neither additional model training nor data is specifically needed, it is easily applicable in practical use.

count=1
* GEO-Bench: Toward Foundation Models for Earth Monitoring
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/a0644215d9cff6646fa334dfa5d29c5a-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: GEO-Bench: Toward Foundation Models for Earth Monitoring
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Björn Lütjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, Mehmet Gunturkun, Gabriel Huang, David Vazquez, Dava Newman, Yoshua Bengio, Stefano Ermon, Xiaoxiang Zhu
    * Abstract: Recent progress in self-supervision has shown that pre-training large neural networks on vast amounts of unsupervised data can lead to substantial increases in generalization to downstream tasks. Such models, recently coined foundation models, have been transformational to the field of natural language processing.Variants have also been proposed for image data, but their applicability to remote sensing tasks is limited.To stimulate the development of foundation models for Earth monitoring, we propose a benchmark comprised of six classification and six segmentation tasks, which were carefully curated and adapted to be both relevant to the field and well-suited for model evaluation. We accompany this benchmark with a robust methodology for evaluating models and reporting aggregated results to enable a reliable assessment of progress. Finally, we report results for 20 baselines to gain information about the performance of existing models.We believe that this benchmark will be a driver of progress across a variety of Earth monitoring tasks.

count=1
* Taming Local Effects in Graph-based Spatiotemporal Forecasting
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/ad58c61c71efd5436134a3ecc87da6ea-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/ad58c61c71efd5436134a3ecc87da6ea-Paper-Conference.pdf)]
    * Title: Taming Local Effects in Graph-based Spatiotemporal Forecasting
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Andrea Cini, Ivan Marisca, Daniele Zambon, Cesare Alippi
    * Abstract: Spatiotemporal graph neural networks have shown to be effective in time series forecasting applications, achieving better performance than standard univariate predictors in several settings. These architectures take advantage of a graph structure and relational inductive biases to learn a single (global) inductive model to predict any number of the input time series, each associated with a graph node. Despite the gain achieved in computational and data efficiency w.r.t. fitting a set of local models, relying on a single global model can be a limitation whenever some of the time series are generated by a different spatiotemporal stochastic process. The main objective of this paper is to understand the interplay between globality and locality in graph-based spatiotemporal forecasting, while contextually proposing a methodological framework to rationalize the practice of including trainable node embeddings in such architectures. We ascribe to trainable node embeddings the role of amortizing the learning of specialized components. Moreover, embeddings allow for 1) effectively combining the advantages of shared message-passing layers with node-specific parameters and 2) efficiently transferring the learned model to new node sets. Supported by strong empirical evidence, we provide insights and guidelines for specializing graph-based models to the dynamics of each time series and show how this aspect plays a crucial role in obtaining accurate predictions.

count=1
* Gaussian Partial Information Decomposition: Bias Correction and Application to High-dimensional Data
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/ec0bff8bf4b11e36f874790046dfdb65-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/ec0bff8bf4b11e36f874790046dfdb65-Paper-Conference.pdf)]
    * Title: Gaussian Partial Information Decomposition: Bias Correction and Application to High-dimensional Data
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Praveen Venkatesh, Corbett Bennett, Sam Gale, Tamina Ramirez, Greggory Heller, Severine Durand, Shawn Olsen, Stefan Mihalas
    * Abstract: Recent advances in neuroscientific experimental techniques have enabled us to simultaneously record the activity of thousands of neurons across multiple brain regions. This has led to a growing need for computational tools capable of analyzing how task-relevant information is represented and communicated between several brain regions. Partial information decompositions (PIDs) have emerged as one such tool, quantifying how much unique, redundant and synergistic information two or more brain regions carry about a task-relevant message. However, computing PIDs is computationally challenging in practice, and statistical issues such as the bias and variance of estimates remain largely unexplored. In this paper, we propose a new method for efficiently computing and estimating a PID definition on multivariate Gaussian distributions. We show empirically that our method satisfies an intuitive additivity property, and recovers the ground truth in a battery of canonical examples, even at high dimensionality. We also propose and evaluate, for the first time, a method to correct the bias in PID estimates at finite sample sizes. Finally, we demonstrate that our Gaussian PID effectively characterizes inter-areal interactions in the mouse brain, revealing higher redundancy between visual areas when a stimulus is behaviorally relevant.

count=1
* CityRefer: Geography-aware 3D Visual Grounding Dataset on  City-scale Point Cloud Data
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/f4cef76305dcad4efd3537da087ff520-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/f4cef76305dcad4efd3537da087ff520-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: CityRefer: Geography-aware 3D Visual Grounding Dataset on  City-scale Point Cloud Data
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Taiki Miyanishi, Fumiya Kitamori, Shuhei Kurita, Jungdae Lee, Motoaki Kawanabe, Nakamasa Inoue
    * Abstract: City-scale 3D point cloud is a promising way to express detailed and complicated outdoor structures. It encompasses both the appearance and geometry features of segmented city components, including cars, streets, and buildings that can be utilized for attractive applications such as user-interactive navigation of autonomous vehicles and drones. However, compared to the extensive text annotations available for images and indoor scenes, the scarcity of text annotations for outdoor scenes poses a significant challenge for achieving these applications. To tackle this problem, we introduce the CityRefer dataset for city-level visual grounding. The dataset consists of 35k natural language descriptions of 3D objects appearing in SensatUrban city scenes and 5k landmarks labels synchronizing with OpenStreetMap. To ensure the quality and accuracy of the dataset, all descriptions and labels in the CityRefer dataset are manually verified. We also have developed a baseline system that can learn encoded language descriptions, 3D object instances, and geographical information about the city's landmarks to perform visual grounding on the CityRefer dataset. To the best of our knowledge, the CityRefer dataset is the largest city-level visual grounding dataset for localizing specific 3D objects.

