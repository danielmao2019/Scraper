count=4
* Fully Transformer Network for Change Detection of Remote Sensing Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2022/html/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2022/papers/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.pdf)]
    * Title: Fully Transformer Network for Change Detection of Remote Sensing Images
    * Publisher: ACCV
    * Publication Date: `2022`
    * Authors: Tianyu Yan, Zifu Wan, Pingping Zhang
    * Abstract: Recently, change detection (CD) of remote sensing images have achieved great progress with the advances of deep learning. However, current methods generally deliver incomplete CD regions and irregular CD boundaries due to the limited representation ability of the extracted visual features. To relieve these issues, in this work we propose a novel learning framework named Fully Transformer Network (FTN) for remote sensing image CD, which improves the feature extraction from a global view and combines multi-level visual features in a pyramid manner. More specifically, the proposed framework first utilizes the advantages of Transformers in long-range dependency modeling. It can help to learn more discriminative global-level features and obtain complete CD regions. Then, we introduce a pyramid structure to aggregate multi-level visual features from Transformers for feature enhancement. The pyramid structure grafted with a Progressive Attention Module (PAM) can improve the feature representation ability with additional interdependencies through channel attentions. Finally, to better train the framework, we utilize the deeply-supervised learning with multiple boundaryaware loss functions. Extensive experiments demonstrate that our proposed method achieves a new state-of-the-art performance on four public CD benchmarks. For model reproduction, the source code is released at https://github.com/AI-Zhpp/FTN.

count=3
* DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.pdf)]
    * Title: DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Aysim Toker, Lukas Kondmann, Mark Weber, Marvin Eisenberger, Andrés Camero, Jingliang Hu, Ariadna Pregel Hoderlein, Çağlar Şenaras, Timothy Davis, Daniel Cremers, Giovanni Marchisio, Xiao Xiang Zhu, Laura Leal-Taixé
    * Abstract: Earth observation is a fundamental tool for monitoring the evolution of land use in specific areas of interest. Observing and precisely defining change, in this context, requires both time-series data and pixel-wise segmentations. To that end, we propose the DynamicEarthNet dataset that consists of daily, multi-spectral satellite observations of 75 selected areas of interest distributed over the globe with imagery from Planet Labs. These observations are paired with pixel-wise monthly semantic segmentation labels of 7 land use and land cover (LULC) classes. DynamicEarthNet is the first dataset that provides this unique combination of daily measurements and high-quality labels. In our experiments, we compare several established baselines that either utilize the daily observations as additional training data (semi-supervised learning) or multiple observations at once (spatio-temporal learning) as a point of reference for future research. Finally, we propose a new evaluation metric SCS that addresses the specific challenges associated with time-series semantic change segmentation. The data is available at: https://mediatum.ub.tum.de/1650201.

count=3
* Unsupervised Change Detection Based on Image Reconstruction Loss
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Noh_Unsupervised_Change_Detection_Based_on_Image_Reconstruction_Loss_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Noh_Unsupervised_Change_Detection_Based_on_Image_Reconstruction_Loss_CVPRW_2022_paper.pdf)]
    * Title: Unsupervised Change Detection Based on Image Reconstruction Loss
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Hyeoncheol Noh, Jingi Ju, Minseok Seo, Jongchan Park, Dong-Geol Choi
    * Abstract: To train the change detector, bi-temporal images taken at different times in the same area are used. However, collecting labeled bi-temporal images is expensive and time consuming. To solve this problem, various unsupervised change detection methods have been proposed, but they still require unlabeled bi-temporal images. In this paper, we propose unsupervised change detection based on image reconstruction loss using only unlabeled single temporal single image. The image reconstruction model is trained to reconstruct the original source image by receiving the source image and the photometrically transformed source image as a pair. During inference, the model receives bi-temporal images as the input, and tries to reconstruct one of the inputs. The changed region between bi-temporal images shows high reconstruction loss. Our change detector showed significant performance in various change detection benchmark datasets even though only a single temporal single source image was used. The code and trained models will be publicly available for reproducibility.

count=3
* Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/b01153e7112b347d8ed54f317840d8af-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Utkarsh Mall, Bharath Hariharan, Kavita Bala
    * Abstract: Satellite imagery is increasingly available, high resolution, and temporally detailed. Changes in spatio-temporal datasets such as satellite images are particularly interesting as they reveal the many events and forces that shape our world. However, finding such interesting and meaningful change events from the vast data is challenging. In this paper, we present new datasets for such change events that include semantically meaningful events like road construction. Instead of manually annotating the very large corpus of satellite images, we introduce a novel unsupervised approach that takes a large spatio-temporal dataset from satellite images and finds interesting change events. To evaluate the meaningfulness on these datasets we create 2 benchmarks namely CaiRoad and CalFire which capture the events of road construction and forest fires. These new benchmarks can be used to evaluate semantic retrieval/classification performance. We explore these benchmarks qualitatively and quantitatively by using several methods and show that these new datasets are indeed challenging for many existing methods.

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
* Change-Aware Sampling and Contrastive Learning for Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf)]
    * Title: Change-Aware Sampling and Contrastive Learning for Satellite Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Utkarsh Mall, Bharath Hariharan, Kavita Bala
    * Abstract: Automatic remote sensing tools can help inform many large-scale challenges such as disaster management, climate change, etc. While a vast amount of spatio-temporal satellite image data is readily available, most of it remains unlabelled. Without labels, this data is not very useful for supervised learning algorithms. Self-supervised learning instead provides a way to learn effective representations for various downstream tasks without labels. In this work, we leverage characteristics unique to satellite images to learn better self-supervised features. Specifically, we use the temporal signal to contrast images with long-term and short-term differences, and we leverage the fact that satellite images do not change frequently. Using these characteristics, we formulate a new loss contrastive loss called Change-Aware Contrastive (CACo) Loss. Further, we also present a novel method of sampling different geographical regions. We show that leveraging these properties leads to better performance on diverse downstream tasks. For example, we see a 6.5% relative improvement for semantic segmentation and an 8.5% relative improvement for change detection over the best-performing baseline with our method.

count=1
* Describing and Localizing Multiple Changes With Transformers
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Qiu_Describing_and_Localizing_Multiple_Changes_With_Transformers_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiu_Describing_and_Localizing_Multiple_Changes_With_Transformers_ICCV_2021_paper.pdf)]
    * Title: Describing and Localizing Multiple Changes With Transformers
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yue Qiu, Shintaro Yamamoto, Kodai Nakashima, Ryota Suzuki, Kenji Iwata, Hirokatsu Kataoka, Yutaka Satoh
    * Abstract: Existing change captioning studies have mainly focused on a single change. However, detecting and describing multiple changed parts in image pairs is essential for enhancing adaptability to complex scenarios. We solve the above issues from three aspects: (i) We propose a simulation-based multi-change captioning dataset; (ii) We benchmark existing state-of-the-art methods of single change captioning on multi-change captioning; (iii) We further propose Multi-Change Captioning transformers (MCCFormers) that identify change regions by densely correlating different regions in image pairs and dynamically determines the related change regions with words in sentences. The proposed method obtained the highest scores on four conventional change captioning evaluation metrics for multi-change captioning. Additionally, our proposed method can separate attention maps for each change and performs well with respect to change localization. Moreover, the proposed framework outperformed the previous state-of-the-art methods on an existing change captioning benchmark, CLEVR-Change, by a large margin (+6.1 on BLEU-4 and +9.7 on CIDEr scores), indicating its general ability in change captioning tasks. The code and dataset are available at the project page.

count=1
* SiamSTA: Spatio-Temporal Attention Based Siamese Tracker for Tracking UAVs
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/html/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/papers/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.pdf)]
    * Title: SiamSTA: Spatio-Temporal Attention Based Siamese Tracker for Tracking UAVs
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Bo Huang, Junjie Chen, Tingfa Xu, Ying Wang, Shenwang Jiang, Yuncheng Wang, Lei Wang, Jianan Li
    * Abstract: With the growing threat of unmanned aerial vehicle (UAV) intrusion, anti-UAV techniques are becoming increasingly demanding. Object tracking, especially in thermal infrared (TIR) videos, though provides a promising solution, struggles with challenges like small scale and fast movement that commonly occur in anti-UAV scenarios. To mitigate this, we propose a simple yet effective spatio-temporal attention based Siamese network, dubbed SiamSTA, to track UAV robustly by performing reliable local tracking and wide-range re-detection alternatively. Concretely, tracking is carried out by posing spatial and temporal constraints on generating candidate proposals within local neighborhoods, hence eliminating background distractors to better perceive small targets. Complementarily, in case of target lost from local regions due to fast movement, a three-stage re-detection mechanism is introduced to re-detect targets from a global view by exploiting valuable motion cues through a correlation filter based on change detection. Finally, a state-aware switching policy is adopted to adaptively integrate local tracking and global re-detection and take their complementary strengths for robust tracking. Extensive experiments on the 1st and 2nd anti-UAV datasets well demonstrate the superiority of SiamSTA over other competing counterparts. Notably, SiamSTA is the foundation of the 1st-place winning entry in the 2nd Anti-UAV Challenge.

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
* Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Scalable_Multi-Temporal_Remote_Sensing_Change_Data_Generation_via_Simulating_Stochastic_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Scalable_Multi-Temporal_Remote_Sensing_Change_Data_Generation_via_Simulating_Stochastic_ICCV_2023_paper.pdf)]
    * Title: Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zhuo Zheng, Shiqi Tian, Ailong Ma, Liangpei Zhang, Yanfei Zhong
    * Abstract: Understanding the temporal dynamics of Earth's surface is a mission of multi-temporal remote sensing image analysis, significantly promoted by deep vision models with its fuel---labeled multi-temporal images. However, collecting, preprocessing, and annotating multi-temporal remote sensing images at scale is non-trivial since it is expensive and knowledge-intensive. In this paper, we present a scalable multi-temporal remote sensing change data generator via generative modeling, which is cheap and automatic, alleviating these problems. Our main idea is to simulate a stochastic change process over time. We consider the stochastic change process as a probabilistic semantic state transition, namely generative probabilistic change model (GPCM), which decouples the complex simulation problem into two more trackable sub-problems, i.e., change event simulation and semantic change synthesis. To solve these two problems, we present the change generator (Changen), a GAN-based GPCM, enabling controllable object change data generation, including customizable object property, and change event. The extensive experiments suggest that our Changen has superior generation capability, and the change detectors with Changen pre-training exhibit excellent transferability to real-world change datasets.

count=1
* Implicit Neural Representation for Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.pdf)]
    * Title: Implicit Neural Representation for Change Detection
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Peter Naylor, Diego Di Carlo, Arianna Traviglia, Makoto Yamada, Marco Fiorucci
    * Abstract: Identifying changes in a pair of 3D aerial LiDAR point clouds, obtained during two distinct time periods over the same geographic region presents a significant challenge due to the disparities in spatial coverage and the presence of noise in the acquisition system. The most commonly used approaches to detecting changes in point clouds are based on supervised methods which necessitate extensive labelled data often unavailable in real-world applications. To address these issues, we propose an unsupervised approach that comprises two components: Implicit Neural Representation (INR) for continuous shape reconstruction and a Gaussian Mixture Model for categorising changes. INR offers a grid-agnostic representation for encoding bi-temporal point clouds, with unmatched spatial support that can be regularised to enhance high-frequency details and reduce noise. The reconstructions at each timestamp are compared at arbitrary spatial scales, leading to a significant increase in detection capabilities. We apply our method to a benchmark dataset comprising simulated LiDAR point clouds for urban sprawling. This dataset encompasses diverse challenging scenarios, varying in resolutions, input modalities and noise levels. This enables a comprehensive multi-scenario evaluation, comparing our method with the current state-of-the-art approach. We outperform the previous methods by a margin of 10% in the intersection over union metric. In addition, we put our techniques to practical use by applying them in a real-world scenario to identify instances of illicit excavation of archaeological sites and validate our results by comparing them with findings from field experts.

