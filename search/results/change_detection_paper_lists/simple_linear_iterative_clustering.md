count=3
* Superpixels and Polygons Using Simple Non-Iterative Clustering
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2017/html/Achanta_Superpixels_and_Polygons_CVPR_2017_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2017/papers/Achanta_Superpixels_and_Polygons_CVPR_2017_paper.pdf)]
    * Title: Superpixels and Polygons Using Simple Non-Iterative Clustering
    * Year: `2017`
    * Authors: Radhakrishna Achanta, Sabine Susstrunk
    * Abstract: We present an improved version of the Simple Linear Iterative Clustering (SLIC) superpixel segmentation. Unlike SLIC, our algorithm is non-iterative, enforces connectivity from the start, requires lesser memory, and is faster. Relying on the superpixel boundaries obtained using our algorithm, we also present a polygonal partitioning algorithm. We demonstrate that our superpixels as well as the polygonal partitioning are superior to the respective state-of-the-art algorithms on quantitative benchmarks.
count=3
* Manifold SLIC: A Fast Method to Compute Content-Sensitive Superpixels
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_Manifold_SLIC_A_CVPR_2016_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Manifold_SLIC_A_CVPR_2016_paper.pdf)]
    * Title: Manifold SLIC: A Fast Method to Compute Content-Sensitive Superpixels
    * Year: `2016`
    * Authors: Yong-Jin Liu, Cheng-Chi Yu, Min-Jing Yu, Ying He
    * Abstract: Superpixels are perceptually meaningful atomic regions that can effectively capture image features. Among various methods for computing uniform superpixels, simple linear iterative clustering (SLIC) is popular due to its simplicity and high performance. In this paper, we extend SLIC to compute content-sensitive superpixels, i.e., small superpixels in content-dense regions (e.g., with high intensity or color variation) and large superpixels in content-sparse regions. Rather than the conventional SLIC method that clusters pixels in R5, we map the image I to a 2-dimensional manifold M in R5, whose area elements are a good measure of the content density in I. We propose an efficient method to compute restricted centroidal Voronoi tessellation (RCVT) --- a uniform tessellation --- on M, which induces the content-sensitive superpixels in I. Unlike other algorithms that characterize content-sensitivity by geodesic distances, manifold SLIC tackles the problem by measuring areas of Voronoi cells on M, which can be computed at a very low cost. As a result, it runs 10 times faster than the state-of-the-art content-sensitive superpixels algorithm. We evaluate manifold SLIC and seven representative methods on the BSDS500 benchmark and observe that our method outperforms the existing methods.
count=2
* A Video Representation Using Temporal Superpixels
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2013/html/Chang_A_Video_Representation_2013_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2013/papers/Chang_A_Video_Representation_2013_CVPR_paper.pdf)]
    * Title: A Video Representation Using Temporal Superpixels
    * Year: `2013`
    * Authors: Jason Chang, Donglai Wei, John W. Fisher III
    * Abstract: We develop a generative probabilistic model for temporally consistent superpixels in video sequences. In contrast to supervoxel methods, object parts in different frames are tracked by the same temporal superpixel. We explicitly model flow between frames with a bilateral Gaussian process and use this information to propagate superpixels in an online fashion. We consider four novel metrics to quantify performance of a temporal superpixel representation and demonstrate superior performance when compared to supervoxel methods.
count=1
* Understanding Video Transformers via Universal Concept Discovery
    [[abs-CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Kowal_Understanding_Video_Transformers_via_Universal_Concept_Discovery_CVPR_2024_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Kowal_Understanding_Video_Transformers_via_Universal_Concept_Discovery_CVPR_2024_paper.pdf)]
    * Title: Understanding Video Transformers via Universal Concept Discovery
    * Year: `2024`
    * Authors: Matthew Kowal, Achal Dave, Rares Ambrus, Adrien Gaidon, Konstantinos G. Derpanis, Pavel Tokmakov
    * Abstract: This paper studies the problem of concept-based interpretability of transformer representations for videos. Concretely we seek to explain the decision-making process of video transformers based on high-level spatiotemporal concepts that are automatically discovered. Prior research on concept-based interpretability has concentrated solely on image-level tasks. Comparatively video models deal with the added temporal dimension increasing complexity and posing challenges in identifying dynamic concepts over time. In this work we systematically address these challenges by introducing the first Video Transformer Concept Discovery (VTCD) algorithm. To this end we propose an efficient approach for unsupervised identification of units of video transformer representations - concepts and ranking their importance to the output of a model. The resulting concepts are highly interpretable revealing spatio-temporal reasoning mechanisms and object-centric representations in unstructured video models. Performing this analysis jointly over a diverse set of supervised and self-supervised representations we discover that some of these mechanism are universal in video transformers. Finally we show that VTCD can be used for fine-grained action recognition and video object segmentation.
count=1
* Region-Based Representations Revisited
    [[abs-CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Shlapentokh-Rothman_Region-Based_Representations_Revisited_CVPR_2024_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Shlapentokh-Rothman_Region-Based_Representations_Revisited_CVPR_2024_paper.pdf)]
    * Title: Region-Based Representations Revisited
    * Year: `2024`
    * Authors: Michal Shlapentokh-Rothman, Ansel Blume, Yao Xiao, Yuqun Wu, Sethuraman TV, Heyi Tao, Jae Yong Lee, Wilfredo Torres, Yu-Xiong Wang, Derek Hoiem
    * Abstract: We investigate whether region-based representations are effective for recognition. Regions were once a mainstay in recognition approaches but pixel and patch-based features are now used almost exclusively. We show that recent class-agnostic segmenters like SAM can be effectively combined with strong unsupervised representations like DINOv2 and used for a wide variety of tasks including semantic segmentation object-based image retrieval and multi-image analysis. Once the masks and features are extracted these representations even with linear decoders enable competitive performance making them well suited to applications that require custom queries. The compactness of the representation also makes it well-suited to video analysis and other problems requiring inference across many images.
count=1
* CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition With Variational Alignment
    [[abs-CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.pdf)]
    * Title: CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition With Variational Alignment
    * Year: `2023`
    * Authors: Jiangbin Zheng, Yile Wang, Cheng Tan, Siyuan Li, Ge Wang, Jun Xia, Yidong Chen, Stan Z. Li
    * Abstract: Sign language recognition (SLR) is a weakly supervised task that annotates sign videos as textual glosses. Recent studies show that insufficient training caused by the lack of large-scale available sign datasets becomes the main bottleneck for SLR. Most SLR works thereby adopt pretrained visual modules and develop two mainstream solutions. The multi-stream architectures extend multi-cue visual features, yielding the current SOTA performances but requiring complex designs and might introduce potential noise. Alternatively, the advanced single-cue SLR frameworks using explicit cross-modal alignment between visual and textual modalities are simple and effective, potentially competitive with the multi-cue framework. In this work, we propose a novel contrastive visual-textual transformation for SLR, CVT-SLR, to fully explore the pretrained knowledge of both the visual and language modalities. Based on the single-cue cross-modal alignment framework, we propose a variational autoencoder (VAE) for pretrained contextual knowledge while introducing the complete pretrained language module. The VAE implicitly aligns visual and textual modalities while benefiting from pretrained contextual knowledge as the traditional contextual module. Meanwhile, a contrastive cross-modal alignment algorithm is designed to explicitly enhance the consistency constraints. Extensive experiments on public datasets (PHOENIX-2014 and PHOENIX-2014T) demonstrate that our proposed CVT-SLR consistently outperforms existing single-cue methods and even outperforms SOTA multi-cue methods.
count=1
* GridShift: A Faster Mode-Seeking Algorithm for Image Segmentation and Object Tracking
    [[abs-CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Kumar_GridShift_A_Faster_Mode-Seeking_Algorithm_for_Image_Segmentation_and_Object_CVPR_2022_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content/CVPR2022/papers/Kumar_GridShift_A_Faster_Mode-Seeking_Algorithm_for_Image_Segmentation_and_Object_CVPR_2022_paper.pdf)]
    * Title: GridShift: A Faster Mode-Seeking Algorithm for Image Segmentation and Object Tracking
    * Year: `2022`
    * Authors: Abhishek Kumar, Oladayo S. Ajani, Swagatam Das, Rammohan Mallipeddi
    * Abstract: In machine learning, MeanShift is one of the popular clustering algorithms. It iteratively moves each data point to the weighted mean of its neighborhood data points. The computational cost required for finding neighborhood data points for each one is quadratic to the number of data points. Therefore, it is very slow for large-scale datasets. To address this issue, we propose a mode-seeking algorithm, GridShift, with faster computing and principally based on MeanShift that uses a grid-based approach. To speed up, GridShift employs a grid-based approach for neighbor search, which is linear to the number of data points. In addition, GridShift moves the active grid cells (grid cells associated with at least one data point) in place of data points towards the higher density, which provides more speed up. The runtime of GridShift is linear to the number of active grid cells and exponential to the number of features. Therefore, it is ideal for large-scale low-dimensional applications such as object tracking and image segmentation. Through extensive experiments, we showcase the superior performance of GridShift compared to other MeanShift-based algorithms and state-of-the-art algorithms in terms of accuracy and runtime on benchmark datasets, image segmentation. Finally, we provide a new object-tracking algorithm based on GridShift and show promising results for object tracking compared to camshift and MeanShift++.
count=1
* Learning a Weakly-Supervised Video Actor-Action Segmentation Model With a Wise Selection
    [[abs-CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Learning_a_Weakly-Supervised_Video_Actor-Action_Segmentation_Model_With_a_Wise_CVPR_2020_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_a_Weakly-Supervised_Video_Actor-Action_Segmentation_Model_With_a_Wise_CVPR_2020_paper.pdf)]
    * Title: Learning a Weakly-Supervised Video Actor-Action Segmentation Model With a Wise Selection
    * Year: `2020`
    * Authors: Jie Chen,  Zhiheng Li,  Jiebo Luo,  Chenliang Xu
    * Abstract: We address weakly-supervised video actor-action segmentation (VAAS), which extends general video object segmentation (VOS) to additionally consider action labels of the actors. The most successful methods on VOS synthesize a pool of pseudo-annotations (PAs) and then refine them iteratively. However, they face challenges as to how to select from a massive amount of PAs high-quality ones, how to set an appropriate stop condition for weakly-supervised training, and how to initialize PAs pertaining to VAAS. To overcome these challenges, we propose a general Weakly-Supervised framework with a Wise Selection of training samples and model evaluation criterion (WS^2). Instead of blindly trusting quality-inconsistent PAs, WS^2 employs a learning-based selection to select effective PAs and a novel region integrity criterion as a stopping condition for weakly-supervised training. In addition, a 3D-Conv GCAM is devised to adapt to the VAAS task. Extensive experiments show that WS^2 achieves state-of-the-art performance on both weakly-supervised VOS and VAAS tasks and is on par with the best fully-supervised method on VAAS.
count=1
* Transferring and Regularizing Prediction for Semantic Segmentation
    [[abs-CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Transferring_and_Regularizing_Prediction_for_Semantic_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Transferring_and_Regularizing_Prediction_for_Semantic_Segmentation_CVPR_2020_paper.pdf)]
    * Title: Transferring and Regularizing Prediction for Semantic Segmentation
    * Year: `2020`
    * Authors: Yiheng Zhang,  Zhaofan Qiu,  Ting Yao,  Chong-Wah Ngo,  Dong Liu,  Tao Mei
    * Abstract: Semantic segmentation often requires a large set of images with pixel-level annotations. In the view of extremely expensive expert labeling, recent research has shown that the models trained on photo-realistic synthetic data (e.g., computer games) with computer-generated annotations can be adapted to real images. Despite this progress, without constraining the prediction on real images, the models will easily overfit on synthetic data due to severe domain mismatch. In this paper, we novelly exploit the intrinsic properties of semantic segmentation to alleviate such problem for model transfer. Specifically, we present a Regularizer of Prediction Transfer (RPT) that imposes the intrinsic properties as constraints to regularize model transfer in an unsupervised fashion. These constraints include patch-level, cluster-level and context-level semantic prediction consistencies at different levels of image formation. As the transfer is label-free and data-driven, the robustness of prediction is addressed by selectively involving a subset of image regions for model regularization. Extensive experiments are conducted to verify the proposal of RPT on the transfer of models trained on GTA5 and SYNTHIA (synthetic data) to Cityscapes dataset (urban street scenes). RPT shows consistent improvements when injecting the constraints on several neural networks for semantic segmentation. More remarkably, when integrating RPT into the adversarial-based segmentation framework, we report to-date the best results: mIoU of 53.2%/51.7% when transferring from GTA5/SYNTHIA to Cityscapes, respectively.
count=1
* Topology Reconstruction of Tree-Like Structure in Images via Structural Similarity Measure and Dominant Set Clustering
    [[abs-CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Topology_Reconstruction_of_Tree-Like_Structure_in_Images_via_Structural_Similarity_CVPR_2019_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Topology_Reconstruction_of_Tree-Like_Structure_in_Images_via_Structural_Similarity_CVPR_2019_paper.pdf)]
    * Title: Topology Reconstruction of Tree-Like Structure in Images via Structural Similarity Measure and Dominant Set Clustering
    * Year: `2019`
    * Authors: Jianyang Xie,  Yitian Zhao,  Yonghuai Liu,  Pan Su,  Yifan Zhao,  Jun Cheng,  Yalin Zheng,  Jiang Liu
    * Abstract: The reconstruction and analysis of tree-like topological structures in the biomedical images is crucial for biologists and surgeons to understand biomedical conditions and plan surgical procedures. The underlying tree-structure topology reveals how different curvilinear components are anatomically connected to each other. Existing automated topology reconstruction methods have great difficulty in identifying the connectivity when two or more curvilinear components cross or bifurcate, due to their projection ambiguity, imaging noise and low contrast. In this paper, we propose a novel curvilinear structural similarity measure to guide a dominant-set clustering approach to address this indispensable issue. The novel similarity measure takes into account both intensity and geometric properties in representing the curvilinear structure locally and globally, and group curvilinear objects at crossover points into different connected branches by dominant-set clustering. The proposed method is applicable to different imaging modalities, and quantitative and qualitative results on retinal vessel, plant root, and neuronal network datasets show that our methodology is capable of advancing the current state-of-the-art techniques.
count=1
* Contour-Constrained Superpixels for Image and Video Processing
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2017/html/Lee_Contour-Constrained_Superpixels_for_CVPR_2017_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lee_Contour-Constrained_Superpixels_for_CVPR_2017_paper.pdf)]
    * Title: Contour-Constrained Superpixels for Image and Video Processing
    * Year: `2017`
    * Authors: Se-Ho Lee, Won-Dong Jang, Chang-Su Kim
    * Abstract: A novel contour-constrained superpixel (CCS) algorithm is proposed in this work. We initialize superpixels and regions in a regular grid and then refine the superpixel label of each region hierarchically from block to pixel levels. To make superpixel boundaries compatible with object contours, we propose the notion of contour pattern matching and formulate an objective function including the contour constraint. Furthermore, we extend the CCS algorithm to generate temporal superpixels for video processing. We initialize superpixel labels in each frame by transferring those in the previous frame and refine the labels to make superpixels temporally consistent as well as compatible with object contours. Experimental results demonstrate that the proposed algorithm provides better performance than the state-of-the-art superpixel methods.
count=1
* GraB: Visual Saliency via Novel Graph Model and Background Priors
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Wang_GraB_Visual_Saliency_CVPR_2016_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2016/papers/Wang_GraB_Visual_Saliency_CVPR_2016_paper.pdf)]
    * Title: GraB: Visual Saliency via Novel Graph Model and Background Priors
    * Year: `2016`
    * Authors: Qiaosong Wang, Wen Zheng, Robinson Piramuthu
    * Abstract: We propose an unsupervised bottom-up saliency detection approach by exploiting novel graph structure and background priors. The input image is represented as an undirected graph with superpixels as nodes. Feature vectors are extracted from each node to cover regional color, contrast and texture information. A novel graph model is proposed to effectively capture local and global saliency cues. To obtain more accurate saliency estimations, we optimize the saliency map by using a robust background measure. Comprehensive evaluations on benchmark datasets indicate that our algorithm universally surpasses state-of-the-art unsupervised solutions and performs favorably against supervised approaches.
count=1
* KL Divergence Based Agglomerative Clustering for Automated Vitiligo Grading
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/html/Gupta_KL_Divergence_Based_2015_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gupta_KL_Divergence_Based_2015_CVPR_paper.pdf)]
    * Title: KL Divergence Based Agglomerative Clustering for Automated Vitiligo Grading
    * Year: `2015`
    * Authors: Mithun Das Gupta, Srinidhi Srinivasa, Madhukara J., Meryl Antony
    * Abstract: In this paper we present a symmetric KL divergence based agglomerative clustering framework to segment multiple levels of depigmentation in Vitiligo images. The proposed framework starts with a simple merge cost based on symmetric KL divergence. We extend the recent body of work related to Bregman divergence based agglomerative clustering and prove that the symmetric KL divergence is an upper-bound for uni-modal Gaussian distributions. This leads to a very simple yet elegant method for bottomup agglomerative clustering. We introduce albedo and reflectance fields as features for the distance computations. We compare against other established methods to bring out possible pros and cons of the proposed method.
count=1
* Unconstrained Realtime Facial Performance Capture
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/html/Hsieh_Unconstrained_Realtime_Facial_2015_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/papers/Hsieh_Unconstrained_Realtime_Facial_2015_CVPR_paper.pdf)]
    * Title: Unconstrained Realtime Facial Performance Capture
    * Year: `2015`
    * Authors: Pei-Lun Hsieh, Chongyang Ma, Jihun Yu, Hao Li
    * Abstract: We introduce a realtime facial tracking system specifically designed for performance capture in unconstrained settings using a consumer-level RGB-D sensor. Our framework provides uninterrupted 3D facial tracking, even in the presence of extreme occlusions such as those caused by hair, hand-to-face gestures, and wearable accessories. Anyone's face can be instantly tracked and the users can be switched without an extra calibration step. During tracking, we explicitly segment face regions from any occluding parts by detecting outliers in the shape and appearance input using an exponentially smoothed and user-adaptive tracking model as prior. Our face segmentation combines depth and RGB input data and is also robust against illumination changes. To enable continuous and reliable facial feature tracking in the color channels, we synthesize plausible face textures in the occluded regions. Our tracking model is personalized on-the-fly by progressively refining the user's identity, expressions, and texture with reliable samples and temporal filtering. We demonstrate robust and high-fidelity facial tracking on a wide range of subjects with highly incomplete and largely occluded data. Our system works in everyday environments and is fully unobtrusive to the user, impacting consumer AR applications and surveillance.
count=1
* A Weighted Sparse Coding Framework for Saliency Detection
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/html/Li_A_Weighted_Sparse_2015_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_A_Weighted_Sparse_2015_CVPR_paper.pdf)]
    * Title: A Weighted Sparse Coding Framework for Saliency Detection
    * Year: `2015`
    * Authors: Nianyi Li, Bilin Sun, Jingyi Yu
    * Abstract: There is an emerging interest on using high-dimensional datasets beyond 2D images in saliency detection. Examples include 3D data based on stereo matching and Kinect sensors and more recently 4D light field data. However, these techniques adopt very different solution frameworks, in both type of features and procedures on using them. In this paper, we present a unified saliency detection framework for handling heterogenous types of input data. Our approach builds dictionaries using data-specific features. Specifically, we first select a group of potential background superpixels to build a primitive non-saliency dictionary. We then prune the outliers in the dictionary and test on the remaining superpixels to iteratively refine the dictionary. Comprehensive experiments show that our approach universally outperforms the state-of-the-art solution on all 2D, 3D and 4D data.
count=1
* Robust Saliency Detection via Regularized Random Walks Ranking
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/html/Li_Robust_Saliency_Detection_2015_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_Robust_Saliency_Detection_2015_CVPR_paper.pdf)]
    * Title: Robust Saliency Detection via Regularized Random Walks Ranking
    * Year: `2015`
    * Authors: Changyang Li, Yuchen Yuan, Weidong Cai, Yong Xia, David Dagan Feng
    * Abstract: In the field of saliency detection, many graph-based algorithms heavily depend on the accuracy of the pre-processed superpixel segmentation, which leads to significant sacrifice of detail information from the input image. In this paper, we propose a novel bottom-up saliency detection approach that takes advantage of both region-based features and image details. To provide more accurate saliency estimations, we first optimize the image boundary selection by the proposed erroneous boundary removal. By taking the image details and region-based estimations into account, we then propose the regularized random walks ranking to formulate pixel-wised saliency maps from the superpixel-based background and foreground saliency estimations. Experiment results on two public datasets indicate the significantly improved accuracy and robustness of the proposed algorithm in comparison with 12 state-of-the-art saliency detection approaches.
count=1
* Saliency Detection via Cellular Automata
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/html/Qin_Saliency_Detection_via_2015_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/papers/Qin_Saliency_Detection_via_2015_CVPR_paper.pdf)]
    * Title: Saliency Detection via Cellular Automata
    * Year: `2015`
    * Authors: Yao Qin, Huchuan Lu, Yiqun Xu, He Wang
    * Abstract: In this paper, we introduce Cellular Automata--a dynamic evolution model to intuitively detect the salient object. First, we construct a background-based map using color and space contrast with the clustered boundary seeds. Then, a novel propagation mechanism dependent on Cellular Automata is proposed to exploit the intrinsic relevance of similar regions through interactions with neighbors. Impact factor matrix and coherence matrix are constructed to balance the influential power towards each cell's next state. The saliency values of all cells will be renovated simultaneously according to the proposed updating rule. It's surprising to find out that parallel evolution can improve all the existing methods to a similar level regardless of their original results. Finally, we present an integration algorithm in the Bayesian framework to take advantage of multiple saliency maps. Extensive experiments on six public datasets demonstrate that the proposed algorithm outperforms state-of-the-art methods.
count=1
* Salient Object Detection via Bootstrap Learning
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/html/Tong_Salient_Object_Detection_2015_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/papers/Tong_Salient_Object_Detection_2015_CVPR_paper.pdf)]
    * Title: Salient Object Detection via Bootstrap Learning
    * Year: `2015`
    * Authors: Na Tong, Huchuan Lu, Xiang Ruan, Ming-Hsuan Yang
    * Abstract: We propose a bootstrap learning algorithm for salient object detection in which both weak and strong models are exploited. First, a weak saliency map is constructed based on image priors to generate training samples for a strong model. Second, a strong classifier based on samples directly from an input image is learned to detect salient pixels. Results from multiscale saliency maps are integrated to further improve the detection performance. Extensive experiments on five benchmark datasets demonstrate that the proposed bootstrap learning algorithm performs favorably against the state-of-the-art saliency detection methods. Furthermore, we show that the proposed bootstrap learning approach can be easily applied to other bottom-up saliency models for significant improvement.
count=1
* Automatic Feature Learning for Robust Shadow Detection
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2014/html/Khan_Automatic_Feature_Learning_2014_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2014/papers/Khan_Automatic_Feature_Learning_2014_CVPR_paper.pdf)]
    * Title: Automatic Feature Learning for Robust Shadow Detection
    * Year: `2014`
    * Authors: Salman Hameed Khan, Mohammed Bennamoun, Ferdous Sohel, Roberto Togneri
    * Abstract: We present a practical framework to automatically detect shadows in real world scenes from a single photograph. Previous works on shadow detection put a lot of effort in designing shadow variant and invariant hand-crafted features. In contrast, our framework automatically learns the most relevant features in a supervised manner using multiple convolutional deep neural networks (ConvNets). The 7-layer network architecture of each ConvNet consists of alternating convolution and sub-sampling layers. The proposed framework learns features at the super-pixel level and along the object boundaries. In both cases, features are extracted using a context aware window centered at interest points. The predicted posteriors based on the learned features are fed to a conditional random field model to generate smooth shadow contours. Our proposed framework consistently performed better than the state-of-the-art on all major shadow databases collected under a variety of conditions.
count=1
* Voxel Cloud Connectivity Segmentation - Supervoxels for Point Clouds
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2013/html/Papon_Voxel_Cloud_Connectivity_2013_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2013/papers/Papon_Voxel_Cloud_Connectivity_2013_CVPR_paper.pdf)]
    * Title: Voxel Cloud Connectivity Segmentation - Supervoxels for Point Clouds
    * Year: `2013`
    * Authors: Jeremie Papon, Alexey Abramov, Markus Schoeler, Florentin Worgotter
    * Abstract: Unsupervised over-segmentation of an image into regions of perceptually similar pixels, known as superpixels, is a widely used preprocessing step in segmentation algorithms. Superpixel methods reduce the number of regions that must be considered later by more computationally expensive algorithms, with a minimal loss of information. Nevertheless, as some information is inevitably lost, it is vital that superpixels not cross object boundaries, as such errors will propagate through later steps. Existing methods make use of projected color or depth information, but do not consider three dimensional geometric relationships between observed data points which can be used to prevent superpixels from crossing regions of empty space. We propose a novel over-segmentation algorithm which uses voxel relationships to produce over-segmentations which are fully consistent with the spatial geometry of the scene in three dimensional, rather than projective, space. Enforcing the constraint that segmented regions must have spatial connectivity prevents label flow across semantic object boundaries which might otherwise be violated. Additionally, as the algorithm works directly in 3D space, observations from several calibrated RGB+D cameras can be segmented jointly. Experiments on a large data set of human annotated RGB+D images demonstrate a significant reduction in occurrence of clusters crossing object boundaries, while maintaining speeds comparable to state-of-the-art 2D methods.
count=1
* Deep Edge-Aware Interactive Colorization Against Color-Bleeding Effects
    [[abs-ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Deep_Edge-Aware_Interactive_Colorization_Against_Color-Bleeding_Effects_ICCV_2021_paper.html)]
    [[pdf-ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Deep_Edge-Aware_Interactive_Colorization_Against_Color-Bleeding_Effects_ICCV_2021_paper.pdf)]
    * Title: Deep Edge-Aware Interactive Colorization Against Color-Bleeding Effects
    * Year: `2021`
    * Authors: Eungyeup Kim, Sanghyeon Lee, Jeonghoon Park, Somi Choi, Choonghyun Seo, Jaegul Choo
    * Abstract: Deep neural networks for automatic image colorization often suffer from the color-bleeding artifact, a problematic color spreading near the boundaries between adjacent objects. Such color-bleeding artifacts debase the reality of generated outputs, limiting the applicability of colorization models in practice. Although previous approaches have attempted to address this problem in an automatic manner, they tend to work only in limited cases where a high contrast of gray-scale values are given in an input image. Alternatively, leveraging user interactions would be a promising approach for solving this color-breeding artifacts. In this paper, we propose a novel edge-enhancing network for the regions of interest via simple user scribbles indicating where to enhance. In addition, our method requires a minimal amount of effort from users for their satisfactory enhancement. Experimental results demonstrate that our interactive edge-enhancing approach effectively improves the color-bleeding artifacts compared to the existing baselines across various datasets.
count=1
* View-Consistent 4D Light Field Superpixel Segmentation
    [[abs-ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Khan_View-Consistent_4D_Light_Field_Superpixel_Segmentation_ICCV_2019_paper.html)]
    [[pdf-ICCV](https://openaccess.thecvf.com/content_ICCV_2019/papers/Khan_View-Consistent_4D_Light_Field_Superpixel_Segmentation_ICCV_2019_paper.pdf)]
    * Title: View-Consistent 4D Light Field Superpixel Segmentation
    * Year: `2019`
    * Authors: Numair Khan,  Qian Zhang,  Lucas Kasser,  Henry Stone,  Min H. Kim,  James Tompkin
    * Abstract: Many 4D light field processing applications rely on superpixel segmentations, for which occlusion-aware view consistency is important. Yet, existing methods often enforce consistency by propagating clusters from a central view only, which can lead to inconsistent superpixels for non-central views. Our proposed approach combines an occlusion-aware angular segmentation in horizontal and vertical EPI spaces with an occlusion-aware clustering and propagation step across all views. Qualitative video demonstrations show that this helps to remove flickering and inconsistent boundary shapes versus the state-of-the-art approach, and quantitative metrics reflect these findings with improved boundary accuracy and view consistency scores.
count=1
* Semi-Supervised Normalized Cuts for Image Segmentation
    [[abs-ICCV](https://openaccess.thecvf.com/content_iccv_2015/html/Chew_Semi-Supervised_Normalized_Cuts_ICCV_2015_paper.html)]
    [[pdf-ICCV](https://openaccess.thecvf.com/content_iccv_2015/papers/Chew_Semi-Supervised_Normalized_Cuts_ICCV_2015_paper.pdf)]
    * Title: Semi-Supervised Normalized Cuts for Image Segmentation
    * Year: `2015`
    * Authors: Selene E. Chew, Nathan D. Cahill
    * Abstract: Since its introduction as a powerful graph-based method for image segmentation, the Normalized Cuts (NCuts) algorithm has been generalized to incorporate expert knowledge about how certain pixels or regions should be grouped, or how the resulting segmentation should be biased to be correlated with priors. Previous approaches incorporate hard must-link constraints on how certain pixels should be grouped as well as hard cannot-link constraints on how other pixels should be separated into different groups. In this paper, we reformulate NCuts to allow both sets of constraints to be handled in a soft manner, enabling the user to tune the degree to which the constraints are satisfied. An approximate spectral solution to the reformulated problem exists without requiring explicit construction of a large, dense matrix; hence, computation time is comparable to that of unconstrained NCuts. Using synthetic data and real imagery, we show that soft handling of constraints yields better results than unconstrained NCuts and enables more robust clustering and segmentation than is possible when the constraints are strictly enforced.
count=1
* Cluster-Based Point Set Saliency
    [[abs-ICCV](https://openaccess.thecvf.com/content_iccv_2015/html/Tasse_Cluster-Based_Point_Set_ICCV_2015_paper.html)]
    [[pdf-ICCV](https://openaccess.thecvf.com/content_iccv_2015/papers/Tasse_Cluster-Based_Point_Set_ICCV_2015_paper.pdf)]
    * Title: Cluster-Based Point Set Saliency
    * Year: `2015`
    * Authors: Flora Ponjou Tasse, Jiri Kosinka, Neil Dodgson
    * Abstract: We propose a cluster-based approach to point set saliency detection, a challenge since point sets lack topological information. A point set is first decomposed into small clusters, using fuzzy clustering. We evaluate cluster uniqueness and spatial distribution of each cluster and combine these values into a cluster saliency function. Finally, the probabilities of points belonging to each cluster are used to assign a saliency to each point. Our approach detects fine-scale salient features and uninteresting regions consistently have lower saliency values. We evaluate the proposed saliency model by testing our saliency-based keypoint detection against a 3D interest point detection benchmark. The evaluation shows that our method achieves a good balance between false positive and false negative error rates, without using any topological information.
count=1
* Saliency Detection via Dense and Sparse Reconstruction
    [[abs-ICCV](https://openaccess.thecvf.com/content_iccv_2013/html/Li_Saliency_Detection_via_2013_ICCV_paper.html)]
    [[pdf-ICCV](https://openaccess.thecvf.com/content_iccv_2013/papers/Li_Saliency_Detection_via_2013_ICCV_paper.pdf)]
    * Title: Saliency Detection via Dense and Sparse Reconstruction
    * Year: `2013`
    * Authors: Xiaohui Li, Huchuan Lu, Lihe Zhang, Xiang Ruan, Ming-Hsuan Yang
    * Abstract: In this paper, we propose a visual saliency detection algorithm from the perspective of reconstruction errors. The image boundaries are first extracted via superpixels as likely cues for background templates, from which dense and sparse appearance models are constructed. For each image region, we first compute dense and sparse reconstruction errors. Second, the reconstruction errors are propagated based on the contexts obtained from K-means clustering. Third, pixel-level saliency is computed by an integration of multi-scale reconstruction errors and refined by an object-biased Gaussian model. We apply the Bayes formula to integrate saliency measures based on dense and sparse reconstruction errors. Experimental results show that the proposed algorithm performs favorably against seventeen state-of-the-art methods in terms of precision and recall. In addition, the proposed algorithm is demonstrated to be more effective in highlighting salient objects uniformly and robust to background noise.
count=1
* Segment Any Point Cloud Sequences by Distilling Vision Foundation Models
    [[abs-NeurIPS](https://papers.nips.cc/paper_files/paper/2023/hash/753d9584b57ba01a10482f1ea7734a89-Abstract-Conference.html)]
    [[pdf-NeurIPS](https://papers.nips.cc/paper_files/paper/2023/file/753d9584b57ba01a10482f1ea7734a89-Paper-Conference.pdf)]
    * Title: Segment Any Point Cloud Sequences by Distilling Vision Foundation Models
    * Year: `2023`
    * Authors: Youquan Liu, Lingdong Kong, Jun CEN, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu
    * Abstract: Recent advancements in vision foundation models (VFMs) have opened up new possibilities for versatile and efficient visual perception. In this work, we introduce Seal, a novel framework that harnesses VFMs for segmenting diverse automotive point cloud sequences. Seal exhibits three appealing properties: i) Scalability: VFMs are directly distilled into point clouds, obviating the need for annotations in either 2D or 3D during pretraining. ii) Consistency: Spatial and temporal relationships are enforced at both the camera-to-LiDAR and point-to-segment regularization stages, facilitating cross-modal representation learning. iii) Generalizability: Seal enables knowledge transfer in an off-the-shelf manner to downstream tasks involving diverse point clouds, including those from real/synthetic, low/high-resolution, large/small-scale, and clean/corrupted datasets. Extensive experiments conducted on eleven different point cloud datasets showcase the effectiveness and superiority of Seal. Notably, Seal achieves a remarkable 45.0% mIoU on nuScenes after linear probing, surpassing random initialization by 36.9% mIoU and outperforming prior arts by 6.1% mIoU. Moreover, Seal demonstrates significant performance gains over existing methods across 20 different few-shot fine-tuning tasks on all eleven tested point cloud datasets. The code is available at this link.
count=1
* SmoothHess: ReLU Network Feature Interactions via Stein's Lemma
    [[abs-NeurIPS](https://papers.nips.cc/paper_files/paper/2023/hash/9ef5e965720193681fc8d16372ac4717-Abstract-Conference.html)]
    [[pdf-NeurIPS](https://papers.nips.cc/paper_files/paper/2023/file/9ef5e965720193681fc8d16372ac4717-Paper-Conference.pdf)]
    * Title: SmoothHess: ReLU Network Feature Interactions via Stein's Lemma
    * Year: `2023`
    * Authors: Max Torop, Aria Masoomi, Davin Hill, Kivanc Kose, Stratis Ioannidis, Jennifer Dy
    * Abstract: Several recent methods for interpretability model feature interactions by looking at the Hessian of a neural network. This poses a challenge for ReLU networks, which are piecewise-linear and thus have a zero Hessian almost everywhere. We propose SmoothHess, a method of estimating second-order interactions through Stein's Lemma. In particular, we estimate the Hessian of the network convolved with a Gaussian through an efficient sampling algorithm, requiring only network gradient calls. SmoothHess is applied post-hoc, requires no modifications to the ReLU network architecture, and the extent of smoothing can be controlled explicitly. We provide a non-asymptotic bound on the sample complexity of our estimation procedure. We validate the superior ability of SmoothHess to capture interactions on benchmark datasets and a real-world medical spirometry dataset.
