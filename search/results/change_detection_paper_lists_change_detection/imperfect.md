count=23
* Dual Task Learning by Leveraging Both Dense Correspondence and Mis-Correspondence for Robust Change Detection With Imperfect Matches
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.pdf)]
    * Title: Dual Task Learning by Leveraging Both Dense Correspondence and Mis-Correspondence for Robust Change Detection With Imperfect Matches
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jin-Man Park, Ue-Hwan Kim, Seon-Hoon Lee, Jong-Hwan Kim
    * Abstract: Accurate change detection enables a wide range of tasks in visual surveillance, anomaly detection and mobile robotics. However, contemporary change detection approaches assume an ideal matching between the current and stored scenes, whereas only coarse matching is possible in real-world scenarios. Thus, contemporary approaches fail to show the reported performance in real-world settings. To overcome this limitation, we propose SimSaC. SimSaC concurrently conducts scene flow estimation and change detection and is able to detect changes with imperfect matches. To train SimSaC without additional manual labeling, we propose a training scheme with random geometric transformations and the cut-paste method. Moreover, we design an evaluation protocol which reflects performance in real-world settings. In designing the protocol, we collect a test benchmark dataset, which we claim as another contribution. Our comprehensive experiments verify that SimSaC displays robust performance even given imperfect matches and the performance margin compared to contemporary approaches is huge.

count=3
* TAMPAR: Visual Tampering Detection for Parcel Logistics in Postal Supply Chains
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Naumann_TAMPAR_Visual_Tampering_Detection_for_Parcel_Logistics_in_Postal_Supply_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Naumann_TAMPAR_Visual_Tampering_Detection_for_Parcel_Logistics_in_Postal_Supply_WACV_2024_paper.pdf)]
    * Title: TAMPAR: Visual Tampering Detection for Parcel Logistics in Postal Supply Chains
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Alexander Naumann, Felix Hertlein, Laura Dörr, Kai Furmans
    * Abstract: Due to the steadily rising amount of valuable goods in supply chains, tampering detection for parcels is becoming increasingly important. In this work, we focus on the use-case last-mile delivery, where only a single RGB image is taken and compared against a reference from an existing database to detect potential appearance changes that indicate tampering. We propose a tampering detection pipeline that utilizes keypoint detection to identify the eight corner points of a parcel. This permits applying a perspective transformation to create normalized fronto-parallel views for each visible parcel side surface. These viewpoint-invariant parcel side surface representations facilitate the identification of signs of tampering on parcels within the supply chain, since they reduce the problem to parcel side surface matching with pair-wise appearance change detection. Experiments with multiple classical and deep learning-based change detection approaches are performed on our newly collected TAMpering detection dataset for PARcels, called TAMPAR. We evaluate keypoint and change detection separately, as well as in a unified system for tampering detection. Our evaluation shows promising results for keypoint (Keypoint AP 75.76) and tampering detection (81% accuracy, F1-Score 0.83) on real images. Furthermore, a sensitivity analysis for tampering types, lens distortion and viewing angles is presented. Code and dataset are available at https://a-nau.github.io/tampar.

count=3
* Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/2ab3163ee384cd46baa7f1abb2b1bf19-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/2ab3163ee384cd46baa7f1abb2b1bf19-Paper-Conference.pdf)]
    * Title: Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Katie Luo, Zhenzhen Liu, Xiangyu Chen, Yurong You, Sagie Benaim, Cheng Perng Phoo, Mark Campbell, Wen Sun, Bharath Hariharan, Kilian Q. Weinberger
    * Abstract: Recent advances in machine learning have shown that Reinforcement Learning from Human Feedback (RLHF) can improve machine learning models and align them with human preferences. Although very successful for Large Language Models (LLMs), these advancements have not had a comparable impact in research for autonomous vehicles—where alignment with human expectations can be imperative. In this paper, we propose to adapt similar RL-based methods to unsupervised object discovery, i.e. learning to detect objects from LiDAR points without any training labels. Instead of labels, we use simple heuristics to mimic human feedback. More explicitly, we combine multiple heuristics into a simple reward function that positively correlates its score with bounding box accuracy, i.e., boxes containing objects are scored higher than those without. We start from the detector’s own predictions to explore the space and reinforce boxes with high rewards through gradient updates. Empirically, we demonstrate that our approach is not only more accurate, but also orders of magnitudes faster to train compared to prior works on object discovery. Code is available at https://github.com/katieluo88/DRIFT.

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
* Minimum Delay Object Detection From Video
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Lao_Minimum_Delay_Object_Detection_From_Video_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lao_Minimum_Delay_Object_Detection_From_Video_ICCV_2019_paper.pdf)]
    * Title: Minimum Delay Object Detection From Video
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Dong Lao,  Ganesh Sundaramoorthi
    * Abstract: We consider the problem of detecting objects, as they come into view, from videos in an online fashion. We provide the first real-time solution that is guaranteed to minimize the delay, i.e., the time between when the object comes in view and the declared detection time, subject to acceptable levels of detection accuracy. The method leverages modern CNN-based object detectors that operate on a single frame, to aggregate detection results over frames to provide reliable detection at a rate, specified by the user, in guaranteed minimal delay. To do this, we formulate the problem as a Quickest Detection problem, which provides the aforementioned guarantees. We derive our algorithms from this theory. We show in experiments, that with an overhead of just 50 fps, we can increase the number of correct detections and decrease the overall computational cost compared to running a modern single-frame detector.

count=2
* Semi-Supervised Scene Change Detection by Distillation From Feature-Metric Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Lee_Semi-Supervised_Scene_Change_Detection_by_Distillation_From_Feature-Metric_Alignment_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Lee_Semi-Supervised_Scene_Change_Detection_by_Distillation_From_Feature-Metric_Alignment_WACV_2024_paper.pdf)]
    * Title: Semi-Supervised Scene Change Detection by Distillation From Feature-Metric Alignment
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Seonhoon Lee, Jong-Hwan Kim
    * Abstract: Scene change detection (SCD) is a critical task for various applications, such as visual surveillance, anomaly detection, and mobile robotics. Recently, supervised methods for SCD have been developed for urban and indoor environments where input image pairs are typically unaligned due to differences in camera viewpoints. However, supervised SCD methods require pixel-wise change labels and alignment labels for the target domain, which can be both time-consuming and expensive to collect. To tackle this issue, we design an unsupervised loss with regularization methods based on the feature-metric alignment of input image pairs. The proposed unsupervised loss enables the SCD model to jointly learn the flow and the change maps on the target domain. In addition, we propose a semi-supervised learning method based on a distillation loss for the robustness of the SCD model. The proposed learning method is based on the student-teacher structure and incorporates the unsupervised loss of the unlabeled target data and the supervised loss of the labeled synthetic data. Our method achieves considerable performance improvement on the target domain through the proposed unsupervised and distillation loss, using only 10% of the target training dataset without using any labels of the target data.

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
* Rolling Shutter Motion Deblurring
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Su_Rolling_Shutter_Motion_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Su_Rolling_Shutter_Motion_2015_CVPR_paper.pdf)]
    * Title: Rolling Shutter Motion Deblurring
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Shuochen Su, Wolfgang Heidrich
    * Abstract: Although motion blur and rolling shutter deformations are closely coupled artifacts in images taken with CMOS image sensors, the two phenomena have so far mostly been treated separately, with deblurring algorithms being unable to handle rolling shutter wobble, and rolling shutter algorithms being incapable of dealing with motion blur. We propose an approach that delivers sharp and undistorted output given a single rolling shutter motion blurred image. The key to achieving this is a global modeling of the camera motion trajectory, which enables each scanline of the image to be deblurred with the corresponding motion segment. We show the results of the proposed framework through experiments on synthetic and real data.

count=1
* JA-POLS: A Moving-Camera Background Model via Joint Alignment and Partially-Overlapping Local Subspaces
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chelly_JA-POLS_A_Moving-Camera_Background_Model_via_Joint_Alignment_and_Partially-Overlapping_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chelly_JA-POLS_A_Moving-Camera_Background_Model_via_Joint_Alignment_and_Partially-Overlapping_CVPR_2020_paper.pdf)]
    * Title: JA-POLS: A Moving-Camera Background Model via Joint Alignment and Partially-Overlapping Local Subspaces
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Irit Chelly,  Vlad Winter,  Dor Litvak,  David Rosen,  Oren Freifeld
    * Abstract: Background models are widely used in computer vision. While successful Static-camera Background (SCB) models exist, Moving-camera Background (MCB) models are limited. Seemingly, there is a straightforward solution: 1) align the video frames; 2) learn an SCB model; 3) warp either original or previously-unseen frames toward the model. This approach, however, has drawbacks, especially when the accumulative camera motion is large and/or the video is long. Here we propose a purely-2D unsupervised modular method that systematically eliminates those issues. First, to estimate warps in the original video, we solve a joint-alignment problem while leveraging a certifiably-correct initialization. Next, we learn both multiple partially-overlapping local subspaces and how to predict alignments. Lastly, in test time, we warp a previously-unseen frame, based on the prediction, and project it on a subset of those subspaces to obtain a background/foreground separation. We show the method handles even large scenes with a relatively-free camera motion (provided the camera-to-scene distance does not change much) and that it not only yields State-of-the-Art results on the original video but also generalizes gracefully to previously-unseen videos of the same scene. Our code is available at https://github.com/BGU-CS-VIL/JA-POLS.

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
* Learning Compositional Representation for 4D Captures With Neural ODE
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.pdf)]
    * Title: Learning Compositional Representation for 4D Captures With Neural ODE
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Boyan Jiang, Yinda Zhang, Xingkui Wei, Xiangyang Xue, Yanwei Fu
    * Abstract: Learning based representation has become the key to the success of many computer vision systems. While many 3D representations have been proposed, it is still an unaddressed problem how to represent a dynamically changing 3D object. In this paper, we introduce a compositional representation for 4D captures, i.e. a deforming 3D object over a temporal span, that disentangles shape, initial state, and motion respectively. Each component is represented by a latent code via a trained encoder. To model the motion, a neural Ordinary Differential Equation (ODE) is trained to update the initial state conditioned on the learned motion code, and a decoder takes the shape code and the updated state code to reconstruct the 3D model at each time stamp. To this end, we propose an Identity Exchange Training (IET) strategy to encourage the network to learn effectively decoupling each component. Extensive experiments demonstrate that the proposed method outperforms existing state-of-the-art deep learning based methods on 4D reconstruction, and significantly improves on various tasks, including motion transfer and completion.

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
* Online Multimodal Video Registration Based on Shape Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/St-Charles_Online_Multimodal_Video_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/St-Charles_Online_Multimodal_Video_2015_CVPR_paper.pdf)]
    * Title: Online Multimodal Video Registration Based on Shape Matching
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: The registration of video sequences captured using different types of sensors often relies on dense feature matching methods, which are very costly. In this paper, we study the problem of "almost planar" scene registration (i.e. where the planar ground assumption is almost respected) in multimodal imagery using target shape information. We introduce a new strategy for robustly aligning scene elements based on the random sampling of shape contour correspondences and on the continuous update of our transformation model's parameters. We evaluate our solution on a public dataset and show its superiority by comparing it to a recently published method that targets the same problem. To make comparisons between such methods easier in the future, we provide our evaluation tools along with a full implementation of our solution online.

count=1
* A Unified Model for Near and Remote Sensing
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Workman_A_Unified_Model_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Workman_A_Unified_Model_ICCV_2017_paper.pdf)]
    * Title: A Unified Model for Near and Remote Sensing
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Scott Workman, Menghua Zhai, David J. Crandall, Nathan Jacobs
    * Abstract: We propose a novel convolutional neural network architecture for estimating geospatial functions such as population density, land cover, or land use. In our approach, we combine overhead and ground-level images in an end-to-end trainable neural network, which uses kernel regression and density estimation to convert features extracted from the ground-level images into a dense feature map. The output of this network is a dense estimate of the geospatial function in the form of a pixel-level labeling of the overhead image. To evaluate our approach, we created a large dataset of overhead and ground-level images from a major urban area with three sets of labels: land use, building function, and building age. We find that our approach is more accurate for all tasks, in some cases dramatically so.

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
* Automatic Open-World Reliability Assessment
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Jafarzadeh_Automatic_Open-World_Reliability_Assessment_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Jafarzadeh_Automatic_Open-World_Reliability_Assessment_WACV_2021_paper.pdf)]
    * Title: Automatic Open-World Reliability Assessment
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Mohsen Jafarzadeh, Touqeer Ahmad, Akshay Raj Dhamija, Chunchun Li, Steve Cruz, Terrance E. Boult
    * Abstract: Image classification in the open-world must handle out-of-distribution (OOD) images. Systems should ideally reject OOD images, or they will map atop of known classes and reduce reliability. Using open-set classifiers that can reject OOD inputs can help. However, optimal accuracy of open-set classifiers depend on the frequency of OOD data. Thus, for either standard or open-set classifiers, it is important to be able to determine when the world changes and increasing OOD inputs will result in reduced system reliability. However, during operations, we cannot directly assess accuracy as there are no labels. Thus, the reliability assessment of these classifiers must be done by human operators, made more complex because networks are not 100% accurate, so some failures are to be expected. To automate this process, herein, we formalize the open-world recognition reliability problem and propose multiple automatic reliability assessment policies to address this new problem using only the distribution of reported scores/probability data. The distributional algorithms can be applied to both classic classifiers with SoftMax as well as the open-world Extreme Value Machine (EVM) to provide automated reliability assessment. We show that all of the new algorithms significantly outperform detection using the mean of SoftMax.

count=1
* Towards Interpretable Video Anomaly Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.pdf)]
    * Title: Towards Interpretable Video Anomaly Detection
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Keval Doshi, Yasin Yilmaz
    * Abstract: Most video anomaly detection approaches are based on data-intensive end-to-end trained neural networks, which extract spatiotemporal features from videos. The extracted feature representations in such approaches are not interpretable, which prevents the automatic identification of anomaly cause. To this end, we propose a novel framework which can explain the detected anomalous event in a surveillance video. In addition to monitoring objects independently, we also monitor the interactions between them to detect anomalous events and explain their root causes. Specifically, we demonstrate that the scene graphs obtained by monitoring the object interactions provide an interpretation for the context of the anomaly while performing competitively with respect to the recent state-of-the-art approaches. Moreover, the proposed interpretable method enables cross-domain adaptability (i.e., transfer learning in another surveillance scene), which is not feasible for most existing end-to-end methods due to the lack of sufficient labeled training data for every surveillance scene. The quick and reliable detection performance of the proposed method is evaluated both theoretically (through an asymptotic optimality proof) and empirically on the popular benchmark datasets.

