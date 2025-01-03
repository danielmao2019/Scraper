count=7
* Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.pdf)]
    * Title: Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Vivien Sainte Fare Garnot,  Loic Landrieu,  Sebastien Giordano,  Nesrine Chehata
    * Abstract: Satellite image time series, bolstered by their growing availability, are at the forefront of an extensive effort towards automated Earth monitoring by international institutions. In particular, large-scale control of agricultural parcels is an issue of major political and economic importance. In this regard, hybrid convolutional-recurrent neural architectures have shown promising results for the automated classification of satellite image time series. We propose an alternative approach in which the convolutional layers are advantageously replaced with encoders operating on unordered sets of pixels to exploit the typically coarse resolution of publicly available satellite images. We also propose to extract temporal features using a bespoke neural architecture based on self-attention instead of recurrent networks. We demonstrate experimentally that our method not only outperforms previous state-of-the-art approaches in terms of precision, but also significantly decreases processing time and memory requirements. Lastly, we release a large open-access annotated dataset as a benchmark for future work on satellite image time series.

count=5
* Dual Task Learning by Leveraging Both Dense Correspondence and Mis-Correspondence for Robust Change Detection With Imperfect Matches
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.pdf)]
    * Title: Dual Task Learning by Leveraging Both Dense Correspondence and Mis-Correspondence for Robust Change Detection With Imperfect Matches
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jin-Man Park, Ue-Hwan Kim, Seon-Hoon Lee, Jong-Hwan Kim
    * Abstract: Accurate change detection enables a wide range of tasks in visual surveillance, anomaly detection and mobile robotics. However, contemporary change detection approaches assume an ideal matching between the current and stored scenes, whereas only coarse matching is possible in real-world scenarios. Thus, contemporary approaches fail to show the reported performance in real-world settings. To overcome this limitation, we propose SimSaC. SimSaC concurrently conducts scene flow estimation and change detection and is able to detect changes with imperfect matches. To train SimSaC without additional manual labeling, we propose a training scheme with random geometric transformations and the cut-paste method. Moreover, we design an evaluation protocol which reflects performance in real-world settings. In designing the protocol, we collect a test benchmark dataset, which we claim as another contribution. Our comprehensive experiments verify that SimSaC displays robust performance even given imperfect matches and the performance margin compared to contemporary approaches is huge.

count=3
* Unsupervised Change Detection Based on Image Reconstruction Loss
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Noh_Unsupervised_Change_Detection_Based_on_Image_Reconstruction_Loss_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Noh_Unsupervised_Change_Detection_Based_on_Image_Reconstruction_Loss_CVPRW_2022_paper.pdf)]
    * Title: Unsupervised Change Detection Based on Image Reconstruction Loss
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Hyeoncheol Noh, Jingi Ju, Minseok Seo, Jongchan Park, Dong-Geol Choi
    * Abstract: To train the change detector, bi-temporal images taken at different times in the same area are used. However, collecting labeled bi-temporal images is expensive and time consuming. To solve this problem, various unsupervised change detection methods have been proposed, but they still require unlabeled bi-temporal images. In this paper, we propose unsupervised change detection based on image reconstruction loss using only unlabeled single temporal single image. The image reconstruction model is trained to reconstruct the original source image by receiving the source image and the photometrically transformed source image as a pair. During inference, the model receives bi-temporal images as the input, and tries to reconstruct one of the inputs. The changed region between bi-temporal images shows high reconstruction loss. Our change detector showed significant performance in various change detection benchmark datasets even though only a single temporal single source image was used. The code and trained models will be publicly available for reproducibility.

count=2
* Fully Transformer Network for Change Detection of Remote Sensing Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2022/html/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2022/papers/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.pdf)]
    * Title: Fully Transformer Network for Change Detection of Remote Sensing Images
    * Publisher: ACCV
    * Publication Date: `2022`
    * Authors: Tianyu Yan, Zifu Wan, Pingping Zhang
    * Abstract: Recently, change detection (CD) of remote sensing images have achieved great progress with the advances of deep learning. However, current methods generally deliver incomplete CD regions and irregular CD boundaries due to the limited representation ability of the extracted visual features. To relieve these issues, in this work we propose a novel learning framework named Fully Transformer Network (FTN) for remote sensing image CD, which improves the feature extraction from a global view and combines multi-level visual features in a pyramid manner. More specifically, the proposed framework first utilizes the advantages of Transformers in long-range dependency modeling. It can help to learn more discriminative global-level features and obtain complete CD regions. Then, we introduce a pyramid structure to aggregate multi-level visual features from Transformers for feature enhancement. The pyramid structure grafted with a Progressive Attention Module (PAM) can improve the feature representation ability with additional interdependencies through channel attentions. Finally, to better train the framework, we utilize the deeply-supervised learning with multiple boundaryaware loss functions. Extensive experiments demonstrate that our proposed method achieves a new state-of-the-art performance on four public CD benchmarks. For model reproduction, the source code is released at https://github.com/AI-Zhpp/FTN.

count=2
* Guided Anisotropic Diffusion and Iterative Learning for Weakly Supervised Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/EarthVision/Daudt_Guided_Anisotropic_Diffusion_and_Iterative_Learning_for_Weakly_Supervised_Change_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/EarthVision/Daudt_Guided_Anisotropic_Diffusion_and_Iterative_Learning_for_Weakly_Supervised_Change_CVPRW_2019_paper.pdf)]
    * Title: Guided Anisotropic Diffusion and Iterative Learning for Weakly Supervised Change Detection
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Rodrigo Caye Daudt,  Bertrand Le Saux,  Alexandre Boulch,  Yann Gousseau
    * Abstract: Large scale datasets created from user labels or openly available data have become crucial to provide training data for large scale learning algorithms. While these datasets are easier to acquire, the data are frequently noisy and unreliable, which is motivating research on weakly supervised learning techniques. In this paper we propose an iterative learning method that extracts the useful information from a large scale change detection dataset generated from open vector data to train a fully convolutional network which surpasses the performance obtained by naive supervised learning. We also propose the guided anisotropic diffusion algorithm, which improves semantic segmentation results using the input images as guides to perform edge preserving filtering, and is used in conjunction with the iterative training method to improve results.

count=2
* Change Is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Change_Is_Everywhere_Single-Temporal_Supervised_Object_Change_Detection_in_Remote_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Change_Is_Everywhere_Single-Temporal_Supervised_Object_Change_Detection_in_Remote_ICCV_2021_paper.pdf)]
    * Title: Change Is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zhuo Zheng, Ailong Ma, Liangpei Zhang, Yanfei Zhong
    * Abstract: For high spatial resolution (HSR) remote sensing images, bitemporal supervised learning always dominates change detection using many pairwise labeled bitemporal images. However, it is very expensive and time-consuming to pairwise label large-scale bitemporal HSR remote sensing images. In this paper, we propose single-temporal supervised learning (STAR) for change detection from a new perspective of exploiting object changes in unpaired images as supervisory signals. STAR enables us to train a high-accuracy change detector only using unpaired labeled images and generalize to real-world bitemporal images. To evaluate the effectiveness of STAR, we design a simple yet effective change detector called ChangeStar, which can reuse any deep semantic segmentation architecture by the ChangeMixin module. The comprehensive experimental results show that ChangeStar outperforms the baseline with a large margin under single-temporal supervision and achieves superior performance under bitemporal supervision. Code is available at https://github.com/Z-Zheng/ChangeStar.

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
* Minimum Delay Moving Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Lao_Minimum_Delay_Moving_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lao_Minimum_Delay_Moving_CVPR_2017_paper.pdf)]
    * Title: Minimum Delay Moving Object Detection
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Dong Lao, Ganesh Sundaramoorthi
    * Abstract: We present a general framework and method for detection of an object in a video based on apparent motion. The object moves relative to background motion at some unknown time in the video, and the goal is to detect and segment the object as soon it moves in an online manner. Due to unreliability of motion between frames, more than two frames are needed to reliably detect the object. Our method is designed to detect the object(s) with minimum delay, i.e., frames after the object moves, constraining the false alarms. Experiments on a new extensive dataset for moving object detection show that our method achieves less delay for all false alarm constraints than existing state-of-the-art.

count=1
* Spatio-Temporal Alignment of Non-Overlapping Sequences From Independently Panning Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Safdarnejad_Spatio-Temporal_Alignment_of_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Safdarnejad_Spatio-Temporal_Alignment_of_CVPR_2017_paper.pdf)]
    * Title: Spatio-Temporal Alignment of Non-Overlapping Sequences From Independently Panning Cameras
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Seyed Morteza Safdarnejad, Xiaoming Liu
    * Abstract: This paper addresses the problem of spatio-temporal alignment of multiple video sequences. We identify and tackle a novel scenario of this problem referred to as Nonoverlapping Sequences (NOS). NOS are captured by multiple freely panning handheld cameras whose field of views (FOV) might have no direct spatial overlap. With the popularity of mobile sensors, NOS rise when multiple cooperative users capture a public event to create a panoramic video, or when consolidating multiple footages of an incident into a single video. To tackle this novel scenario, we first spatially align the sequences by reconstructing the background of each sequence and registering these backgrounds, even if the backgrounds are not overlapping. Given the spatial alignment, we temporally synchronize the sequences, such that the trajectories of moving objects (e.g., cars or pedestrians) are consistent across sequences. Experimental results demonstrate the performance of our algorithm in this novel and challenging scenario, quantitatively and qualitatively.

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
* Continuous Scene Representations for Embodied AI
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Gadre_Continuous_Scene_Representations_for_Embodied_AI_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Gadre_Continuous_Scene_Representations_for_Embodied_AI_CVPR_2022_paper.pdf)]
    * Title: Continuous Scene Representations for Embodied AI
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Samir Yitzhak Gadre, Kiana Ehsani, Shuran Song, Roozbeh Mottaghi
    * Abstract: We propose Continuous Scene Representations (CSR), a scene representation constructed by an embodied agent navigating within a space, where objects and their relationships are modeled by continuous valued embeddings. Our method captures feature relationships between objects, composes them into a graph structure on-the-fly, and situates an embodied agent within the representation. Our key insight is to embed pair-wise relationships between objects in a latent space. This allows for a richer representation compared to discrete relations (e.g., [support], [next-to]) commonly used for building scene representations. CSR can track objects as the agent moves in a scene, update the representation accordingly, and detect changes in room configurations. Using CSR, we outperform state-of-the-art approaches for the challenging downstream task of visual room rearrangement, without any task specific training. Moreover, we show the learned embeddings capture salient spatial details of the scene and show applicability to real world data. A summery video and code is available at https://prior.allenai.org/projects/csr.

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
* Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.pdf)]
    * Title: Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Maria Vakalopoulou, Konstantinos Karantzalos, Nikos Komodakis, Nikos Paragios
    * Abstract: In order to exploit the currently continuous streams of massive, multi-temporal, high-resolution remote sensing datasets there is an emerging need to address efficiently the image registration and change detection challenges. To this end, in this paper we propose a modular, scalable, metric free single shot change detection/registration method. The approach exploits a decomposed interconnected graphical model formulation where registration similarity constraints are relaxed in the presence of change detection. The deformation space is discretized, while efficient linear programming and duality principles are used to optimize a joint solution space where local consistency is imposed on the deformation and the detection space. Promising results on large scale experiments demonstrate the extreme potentials of our method.

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
* Moving Object Detection in Time-Lapse or Motion Trigger Image Sequences Using Low-Rank and Invariant Sparse Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Shakeri_Moving_Object_Detection_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Shakeri_Moving_Object_Detection_ICCV_2017_paper.pdf)]
    * Title: Moving Object Detection in Time-Lapse or Motion Trigger Image Sequences Using Low-Rank and Invariant Sparse Decomposition
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Moein Shakeri, Hong Zhang
    * Abstract: Low-rank and sparse representation based methods have attracted wide attention in background subtraction and moving object detection, where moving objects in the scene are modeled as pixel-wise sparse outliers. Since in real scenarios moving objects are also structurally sparse, recently researchers have attempted to extract moving objects using structured sparse outliers. Although existing methods with structured sparsity-inducing norms produce promising results, they are still vulnerable to various illumination changes that frequently occur in real environments, specifically for time-lapse image sequences where assumptions about sparsity between images such as group sparsity are not valid. In this paper, we first introduce a prior map obtained by illumination invariant representation of images. Next, we propose a low-rank and invariant sparse decomposition using the prior map to detect moving objects under significant illumination changes. Experiments on challenging benchmark datasets demonstrate the superior performance of our proposed method under complex illumination changes.

count=1
* TransBlast: Self-Supervised Learning Using Augmented Subspace With Transformer for Background/Foreground Separation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Osman_TransBlast_Self-Supervised_Learning_Using_Augmented_Subspace_With_Transformer_for_BackgroundForeground_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Osman_TransBlast_Self-Supervised_Learning_Using_Augmented_Subspace_With_Transformer_for_BackgroundForeground_ICCVW_2021_paper.pdf)]
    * Title: TransBlast: Self-Supervised Learning Using Augmented Subspace With Transformer for Background/Foreground Separation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Islam Osman, Mohamed Abdelpakey, Mohamed S. Shehata
    * Abstract: Background/Foreground separation is a fundamental and challenging task of many computer vision applications. The F-measure performance of state-of-the-art models is limited due to the lack of fine details in the predicted output (i.e., the foreground object) and the limited labeled data. In this paper, we propose a background/foreground separation model based on a transformer that has a higher learning capacity than the convolutional neural networks. The model is trained using self-supervised learning to leverage the limited data and learn a strong object representation that is invariant to changes. The proposed method, dubbed TransBlast, reformulates the background/foreground separation problem in self-supervised learning using the augmented subspace loss function. The augmented subspace loss function consists of two components: 1) the cross-entropy loss function and 2) the subspace that depends on Singular Value Decomposition (SVD). The proposed model is evaluated using three benchmarks, namely CDNet, DAVIS, and SegTrackV2. The performance of TransBlast outperforms state-of-the-art background/foreground separation models in terms of F-measure.

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
* Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Scalable_Multi-Temporal_Remote_Sensing_Change_Data_Generation_via_Simulating_Stochastic_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Scalable_Multi-Temporal_Remote_Sensing_Change_Data_Generation_via_Simulating_Stochastic_ICCV_2023_paper.pdf)]
    * Title: Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zhuo Zheng, Shiqi Tian, Ailong Ma, Liangpei Zhang, Yanfei Zhong
    * Abstract: Understanding the temporal dynamics of Earth's surface is a mission of multi-temporal remote sensing image analysis, significantly promoted by deep vision models with its fuel---labeled multi-temporal images. However, collecting, preprocessing, and annotating multi-temporal remote sensing images at scale is non-trivial since it is expensive and knowledge-intensive. In this paper, we present a scalable multi-temporal remote sensing change data generator via generative modeling, which is cheap and automatic, alleviating these problems. Our main idea is to simulate a stochastic change process over time. We consider the stochastic change process as a probabilistic semantic state transition, namely generative probabilistic change model (GPCM), which decouples the complex simulation problem into two more trackable sub-problems, i.e., change event simulation and semantic change synthesis. To solve these two problems, we present the change generator (Changen), a GAN-based GPCM, enabling controllable object change data generation, including customizable object property, and change event. The extensive experiments suggest that our Changen has superior generation capability, and the change detectors with Changen pre-training exhibit excellent transferability to real-world change datasets.

count=1
* MSNet: A Multilevel Instance Segmentation Network for Natural Disaster Damage Assessment in Aerial Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Zhu_MSNet_A_Multilevel_Instance_Segmentation_Network_for_Natural_Disaster_Damage_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Zhu_MSNet_A_Multilevel_Instance_Segmentation_Network_for_Natural_Disaster_Damage_WACV_2021_paper.pdf)]
    * Title: MSNet: A Multilevel Instance Segmentation Network for Natural Disaster Damage Assessment in Aerial Videos
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Xiaoyu Zhu, Junwei Liang, Alexander Hauptmann
    * Abstract: In this paper, we study the problem of efficiently assessing building damage after natural disasters like hurricanes, floods or fires, through aerial video analysis. We make two main contributions. The first contribution is a new dataset, consisting of user-generated aerial videos from social media with annotations of instance-level building damage masks. This provides the first benchmark for quantitative evaluation of models to assess building damage using aerial videos. The second contribution is a new model, namely MSNet, which contains novel region proposal network designs and an unsupervised score refinement network for confidence score calibration in both bounding box and mask branches. We show that our model achieves state-of-the-art results compared to previous methods in our dataset.

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
* Multi-View 3D Object Reconstruction and Uncertainty Modelling With Neural Shape Prior
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Liao_Multi-View_3D_Object_Reconstruction_and_Uncertainty_Modelling_With_Neural_Shape_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Liao_Multi-View_3D_Object_Reconstruction_and_Uncertainty_Modelling_With_Neural_Shape_WACV_2024_paper.pdf)]
    * Title: Multi-View 3D Object Reconstruction and Uncertainty Modelling With Neural Shape Prior
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Ziwei Liao, Steven L. Waslander
    * Abstract: 3D object reconstruction is important for semantic scene understanding. It is challenging to reconstruct detailed 3D shapes from monocular images directly due to a lack of depth information, occlusion and noise. Most current methods generate deterministic object models without any awareness of the uncertainty of the reconstruction. We tackle this problem by leveraging a neural object representation which learns an object shape distribution from large dataset of 3d object models and maps it into a latent space. We propose a method to model uncertainty as part of the representation and define an uncertainty-aware encoder which generates latent codes with uncertainty directly from individual input images. Further, we propose a method to propagate the uncertainty in the latent code to SDF values and generate a 3d object mesh with local uncertainty for each mesh component. Finally, we propose an incremental fusion method under a Bayesian framework to fuse the latent codes from multi-view observations. We evaluate the system in both synthetic and real datasets to demonstrate the effectiveness of uncertainty-based fusion to improve 3D object reconstruction accuracy.

count=1
* Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/b01153e7112b347d8ed54f317840d8af-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Utkarsh Mall, Bharath Hariharan, Kavita Bala
    * Abstract: Satellite imagery is increasingly available, high resolution, and temporally detailed. Changes in spatio-temporal datasets such as satellite images are particularly interesting as they reveal the many events and forces that shape our world. However, finding such interesting and meaningful change events from the vast data is challenging. In this paper, we present new datasets for such change events that include semantically meaningful events like road construction. Instead of manually annotating the very large corpus of satellite images, we introduce a novel unsupervised approach that takes a large spatio-temporal dataset from satellite images and finds interesting change events. To evaluate the meaningfulness on these datasets we create 2 benchmarks namely CaiRoad and CalFire which capture the events of road construction and forest fires. These new benchmarks can be used to evaluate semantic retrieval/classification performance. We explore these benchmarks qualitatively and quantitatively by using several methods and show that these new datasets are indeed challenging for many existing methods.

