count=8
* A Semi-Supervised Approach for Ice-Water Classification Using Dual-Polarization SAR Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Li_A_Semi-Supervised_Approach_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Li_A_Semi-Supervised_Approach_2015_CVPR_paper.pdf)]
    * Title: A Semi-Supervised Approach for Ice-Water Classification Using Dual-Polarization SAR Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Fan Li, David A. Clausi, Lei Wang, Linlin Xu
    * Abstract: The daily interpretation of SAR sea ice imagery is very important for ship navigation and climate monitoring. Currently, the interpretation is still performed manually by ice analysts due to the complexity of data and the difficulty of creating fine-level ground truth. To overcome these problems, a semi-supervised approach for ice-water classification based on self-training is presented. The proposed algorithm integrates the spatial context model, region merging, and the self-training technique into a single framework. The backscatter intensity, texture, and edge strength features are incorporated in a CRF model using multi-modality Gaussian model as its unary classifier. Region merging is used to build a hierarchical data-adaptive structure to make the inference more efficient. Self-training is concatenated with region merging, so that the spatial location information of the original training samples can be used. Our algorithm has been tested on a large-scale RADARSAT-2 dual-polarization dataset over the Beaufort and Chukchi sea, and the classification results are significantly better than the supervised methods without self-training.

count=6
* Biologically-Constrained Graphs for Global Connectomics Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Matejek_Biologically-Constrained_Graphs_for_Global_Connectomics_Reconstruction_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Matejek_Biologically-Constrained_Graphs_for_Global_Connectomics_Reconstruction_CVPR_2019_paper.pdf)]
    * Title: Biologically-Constrained Graphs for Global Connectomics Reconstruction
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Brian Matejek,  Daniel Haehn,  Haidong Zhu,  Donglai Wei,  Toufiq Parag,  Hanspeter Pfister
    * Abstract: Most current state-of-the-art connectome reconstruction pipelines have two major steps: initial pixel-based segmentation with affinity prediction and watershed transform, and refined segmentation by merging over-segmented regions. These methods rely only on local context and are typically agnostic to the underlying biology. Since a few merge errors can lead to several incorrectly merged neuronal processes, these algorithms are currently tuned towards over-segmentation producing an overburden of costly proofreading. We propose a third step for connectomics reconstruction pipelines to refine an over-segmentation using both local and global context with an emphasis on adhering to the underlying biology. We first extract a graph from an input segmentation where nodes correspond to segment labels and edges indicate potential split errors in the over-segmentation. In order to increase throughput and allow for large-scale reconstruction, we employ biologically inspired geometric constraints based on neuron morphology to reduce the number of nodes and edges. Next, two neural networks learn these neuronal shapes to further aid the graph construction process. Lastly, we reformulate the region merging problem as a graph partitioning one to leverage global context. We demonstrate the performance of our approach on four real-world connectomics datasets with an average variation of information improvement of 21.3%.

count=2
* Hierarchical Video Representation with Trajectory Binary Partition Tree
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Palou_Hierarchical_Video_Representation_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Palou_Hierarchical_Video_Representation_2013_CVPR_paper.pdf)]
    * Title: Hierarchical Video Representation with Trajectory Binary Partition Tree
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Guillem Palou, Philippe Salembier
    * Abstract: As early stage of video processing, we introduce an iterative trajectory merging algorithm that produces a regionbased and hierarchical representation of the video sequence, called the Trajectory Binary Partition Tree (BPT). From this representation, many analysis and graph cut techniques can be used to extract partitions or objects that are useful in the context of specific applications. In order to define trajectories and to create a precise merging algorithm, color and motion cues have to be used. Both types of informations are very useful to characterize objects but present strong differences of behavior in the spatial and the temporal dimensions. On the one hand, scenes and objects are rich in their spatial color distributions, but these distributions are rather stable over time. Object motion, on the other hand, presents simple structures and low spatial variability but may change from frame to frame. The proposed algorithm takes into account this key difference and relies on different models and associated metrics to deal with color and motion information. We show that the proposed algorithm outperforms existing hierarchical video segmentation algorithms and provides more stable and precise regions.

count=2
* Joint Recovery of Dense Correspondence and Cosegmentation in Two Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Taniai_Joint_Recovery_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Taniai_Joint_Recovery_of_CVPR_2016_paper.pdf)]
    * Title: Joint Recovery of Dense Correspondence and Cosegmentation in Two Images
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Tatsunori Taniai, Sudipta N. Sinha, Yoichi Sato
    * Abstract: We propose a new technique to jointly recover cosegmentation and dense per-pixel correspondence in two images. Our method parameterizes the correspondence field using piecewise similarity transformations and recovers a mapping between the estimated common "foreground" regions in the two images allowing them to be precisely aligned. Our formulation is based on a hierarchical Markov random field model with segmentation and transformation labels. The hierarchical structure uses nested image regions to constrain inference across multiple scales. Unlike prior hierarchical methods which assume that the structure is given, our proposed iterative technique dynamically recovers the structure as a variable along with the labeling. This joint inference is performed in an energy minimization framework using iterated graph cuts. We evaluate our method on a new dataset of 400 image pairs with manually obtained ground truth, where it outperforms state-of-the-art methods designed specifically for either cosegmentation or correspondence estimation.

count=2
* Consensus-Based Image Segmentation via Topological Persistence
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w23/html/Ge_Consensus-Based_Image_Segmentation_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w23/papers/Ge_Consensus-Based_Image_Segmentation_CVPR_2016_paper.pdf)]
    * Title: Consensus-Based Image Segmentation via Topological Persistence
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Qian Ge, Edgar Lobaton
    * Abstract: Image segmentation is one of the most important low-level operation in image processing and computer vision. It is unlikely for a single algorithm with a fixed set of parameters to segment various images successfully due to variations between images. However, it can be observed that the desired boundaries are often detected more consistently than other ones in the output of state-of-the-art algorithms. In this paper, we propose a new approach to capture the consensus information from a segmentation set obtained by different algorithms. The present probability of a segment curve is estimated based on our probabilistic segmentation model. A connectivity probability map is constructed and persistent segments are extracted by applying topological persistence to the map. Finally, a robust segmentation is obtained with the detection of certain segment curves guaranteed. The experiments demonstrate our approach is able to consistently capture the curves present within the segmentation set.

count=2
* Joint Shape and Texture Based X-Ray Cargo Image Classification
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/html/Zhang_Joint_Shape_and_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/papers/Zhang_Joint_Shape_and_2014_CVPR_paper.pdf)]
    * Title: Joint Shape and Texture Based X-Ray Cargo Image Classification
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Jian Zhang, Li Zhang, Ziran Zhao, Yaohong Liu, Jianping Gu, Qiang Li, Duokun Zhang
    * Abstract: Security & Inspection X-Ray Systems is widely used by custom to accomplish some security missions by inspecting import-export cargo. Due to the specificity of cargo X-Ray image, such as overlap, viewpoint dependence, and variants of cargo categories, it couldn't be understood easily like natural ones by human. Even for experienced screeners, it's very difficult to judge cargo category and contraband. In this paper, cargo X-Ray image is described by joint shape and texture feature, which could reflect both cargo stacking mode and interior details. Classification performance is compared with the benchmark method by top hit 1, 3, 5 ratio, and it's demonstrated that good performance is achieved here. In addition, we also discuss X-Ray image property and explore some reasons why cargo classification under X-Ray is very difficult.

count=2
* EdgeFlow: Achieving Practical Interactive Segmentation With Edge-Guided Flow
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/html/Hao_EdgeFlow_Achieving_Practical_Interactive_Segmentation_With_Edge-Guided_Flow_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/papers/Hao_EdgeFlow_Achieving_Practical_Interactive_Segmentation_With_Edge-Guided_Flow_ICCVW_2021_paper.pdf)]
    * Title: EdgeFlow: Achieving Practical Interactive Segmentation With Edge-Guided Flow
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yuying Hao, Yi Liu, Zewu Wu, Lin Han, Yizhou Chen, Guowei Chen, Lutao Chu, Shiyu Tang, Zhiliang Yu, Zeyu Chen, Baohua Lai
    * Abstract: High-quality training data play a key role in image segmentation tasks. Usually, pixel-level annotations are expensive, laborious and time-consuming for the large volume of training data. To reduce labelling cost and improve segmentation quality, interactive segmentation methods have been proposed, which provide the result with just a few clicks. However, their performance does not meet the requirements of practical segmentation tasks in terms of speed and accuracy. In this work, we propose EdgeFlow, a novel architecture that fully utilizes interactive information of user clicks with edge-guided flow. Our method achieves state-of-the-art performance without any post-processing or iterative optimization scheme. Comprehensive experiments on benchmarks also demonstrate the superiority of our method. In addition, with the proposed method, we develop an efficient interactive segmentation tool for practical data annotation tasks. The source code and tool is avaliable at \href https://github.com/PaddlePaddle/PaddleSeg https://github.com/PaddlePaddle/PaddleSeg .

count=1
* Measures and Meta-Measures for the Supervised Evaluation of Image Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Pont-Tuset_Measures_and_Meta-Measures_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Pont-Tuset_Measures_and_Meta-Measures_2013_CVPR_paper.pdf)]
    * Title: Measures and Meta-Measures for the Supervised Evaluation of Image Segmentation
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Jordi Pont-Tuset, Ferran Marques
    * Abstract: This paper tackles the supervised evaluation of image segmentation algorithms. First, it surveys and structures the measures used to compare the segmentation results with a ground truth database; and proposes a new measure: the precision-recall for objects and parts. To compare the goodness of these measures, it defines three quantitative meta-measures involving six state of the art segmentation methods. The meta-measures consist in assuming some plausible hypotheses about the results and assessing how well each measure reflects these hypotheses. As a conclusion, this paper proposes the precision-recall curves for boundaries and for objects-and-parts as the tool of choice for the supervised evaluation of image segmentation. We make the datasets and code of all the measures publicly available.

count=1
* Towards Fast and Accurate Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Taylor_Towards_Fast_and_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Taylor_Towards_Fast_and_2013_CVPR_paper.pdf)]
    * Title: Towards Fast and Accurate Segmentation
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Camillo J. Taylor
    * Abstract: In this paper we explore approaches to accelerating segmentation and edge detection algorithms based on the gPb framework. The paper characterizes the performance of a simple but effective edge detection scheme which can be computed rapidly and offers performance that is competitive with the pB detector. The paper also describes an approach for computing a reduced order normalized cut that captures the essential features of the original problem but can be computed in less than half a second on a standard computing platform.

count=1
* SCALPEL: Segmentation Cascades with Localized Priors and Efficient Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Weiss_SCALPEL_Segmentation_Cascades_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Weiss_SCALPEL_Segmentation_Cascades_2013_CVPR_paper.pdf)]
    * Title: SCALPEL: Segmentation Cascades with Localized Priors and Efficient Learning
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: David Weiss, Ben Taskar
    * Abstract: We propose SCALPEL, a flexible method for object segmentation that integrates rich region-merging cues with midand high-level information about object layout, class, and scale into the segmentation process. Unlike competing approaches, SCALPEL uses a cascade of bottom-up segmentation models that is capable of learning to ignore boundaries early on, yet use them as a stopping criterion once the object has been mostly segmented. Furthermore, we show how such cascades can be learned efficiently. When paired with a novel method that generates better localized shape priors than our competitors, our method leads to a concise, accurate set of segmentation proposals; these proposals are more accurate on the PASCAL VOC2010 dataset than state-of-the-art methods that use re-ranking to filter much larger bags of proposals. The code for our algorithm is available online.

count=1
* Hierarchical Saliency Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Yan_Hierarchical_Saliency_Detection_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Yan_Hierarchical_Saliency_Detection_2013_CVPR_paper.pdf)]
    * Title: Hierarchical Saliency Detection
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Qiong Yan, Li Xu, Jianping Shi, Jiaya Jia
    * Abstract: When dealing with objects with complex structures, saliency detection confronts a critical problem namely that detection accuracy could be adversely affected if salient foreground or background in an image contains small-scale high-contrast patterns. This issue is common in natural images and forms a fundamental challenge for prior methods. We tackle it from a scale point of view and propose a multi-layer approach to analyze saliency cues. The final saliency map is produced in a hierarchical model. Different from varying patch sizes or downsizing images, our scale-based region handling is by finding saliency values optimally in a tree model. Our approach improves saliency detection on many images that cannot be handled well traditionally. A new dataset is also constructed.

count=1
* Multiscale Combinatorial Grouping
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Arbelaez_Multiscale_Combinatorial_Grouping_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Arbelaez_Multiscale_Combinatorial_Grouping_2014_CVPR_paper.pdf)]
    * Title: Multiscale Combinatorial Grouping
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Pablo Arbelaez, Jordi Pont-Tuset, Jonathan T. Barron, Ferran Marques, Jitendra Malik
    * Abstract: We propose a unified approach for bottom-up hierarchical image segmentation and object candidate generation for recognition, called Multiscale Combinatorial Grouping (MCG). For this purpose, we first develop a fast normalized cuts algorithm. We then propose a high-performance hierarchical segmenter that makes effective use of multiscale information. Finally, we propose a grouping strategy that combines our multiscale regions into highly-accurate object candidates by exploring efficiently their combinatorial space. We conduct extensive experiments on both the BSDS500 and on the PASCAL 2012 segmentation datasets, showing that MCG produces state-of-the-art contours, hierarchical regions and object candidates.

count=1
* Matching Bags of Regions in RGBD images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Jiang_Matching_Bags_of_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Jiang_Matching_Bags_of_2015_CVPR_paper.pdf)]
    * Title: Matching Bags of Regions in RGBD images
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Hao Jiang
    * Abstract: We study the new problem of matching regions between a pair of RGBD images given a large set of overlapping region proposals. These region proposals do not have a tree hierarchy and are treated as bags of regions. Matching RGBD images using bags of region candidates with unstructured relations is a challenging combinatorial problem. We propose a linear formulation, which optimizes the region selection and matching simultaneously so that the matched regions have similar color histogram, shape, and small overlaps, the selected regions have a small number and overall low concavity, and they tend to cover both of the images. We efficiently compute the lower bound by solving a sequence of min-cost bipartite matching problems via Lagrangian relaxation and we obtain the global optimum using branch and bound. Our experiments show that the proposed method is fast, accurate, and robust against cluttered scenes.

count=1
* Complexity-Adaptive Distance Metric for Object Proposals Generation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Xiao_Complexity-Adaptive_Distance_Metric_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Xiao_Complexity-Adaptive_Distance_Metric_2015_CVPR_paper.pdf)]
    * Title: Complexity-Adaptive Distance Metric for Object Proposals Generation
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Yao Xiao, Cewu Lu, Efstratios Tsougenis, Yongyi Lu, Chi-Keung Tang
    * Abstract: Distance metric plays a key role in grouping superpixels to produce object proposals for object detection. We observe that existing distance metrics work primarily for low complexity cases. In this paper, we develop a novel distance metric for grouping two superpixels in high-complexity scenarios. Combining them, a complexity-adaptive distance measure is produced that achieves improved grouping in different levels of complexity. Our extensive experimentation shows that our method can achieve good results in the PASCAL VOC 2012 dataset surpassing the latest state-of-the-art methods.

count=1
* Texture Complexity Based Redundant Regions Ranking for Object Proposal
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/Ke_Texture_Complexity_Based_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/Ke_Texture_Complexity_Based_CVPR_2016_paper.pdf)]
    * Title: Texture Complexity Based Redundant Regions Ranking for Object Proposal
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Wei Ke, Tianliang Zhang, Jie Chen, Fang Wan, Qixiang Ye, Zhenjun Han
    * Abstract: Object proposal has been successfully applied in recent visual object detection approaches and shown improved computational efficiency. The purpose of object proposal is to use as few as regions to cover as many as objects. In this paper, we propose a strategy named Texture Complexity based Redundant Regions Ranking (TCR) for object proposal. Our approach first produces rich but redundant regions using a color segmentation approach, i.e. Selective Search. It then uses Texture Complexity (TC) based on complete contour number and Local Binary Pattern (LBP) entropy to measure the objectness score of each region. By ranking based on the TC, it is expected that as many as true object regions are preserved, while the number of the regions is significantly reduced. Experimental results on the PASCAL VOC 2007 dataset show that the proposed TCR significantly improves the baseline approach by increasing AUC (area under recall curve) from 0.39 to 0.48.

count=1
* Unsupervised Segmentation of Cervical Cell Images Using Gaussian Mixture Model
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w27/html/Ragothaman_Unsupervised_Segmentation_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w27/papers/Ragothaman_Unsupervised_Segmentation_of_CVPR_2016_paper.pdf)]
    * Title: Unsupervised Segmentation of Cervical Cell Images Using Gaussian Mixture Model
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Srikanth Ragothaman, Sridharakumar Narasimhan, Madivala G. Basavaraj, Rajan Dewar
    * Abstract: Cervical cancer is one of the leading causes of cancer death in women. Screening at early stages using the popular Pap smear test has been demonstrated to reduce fatalities significantly. Cost effective, automated screening methods can significantly improve the adoption of these tests worldwide. Automated screening involves image analysis of cervical cells. Gaussian Mixture Models (GMM) are widely used in image processing for segmentation which is a crucial step in image analysis. In our proposed method, GMM is implemented to segment cell regions to identify cellular features such as nucleus, cytoplasm while addressing shortcomings of existing methods. This method is combined with shape based identification of nucleus to increase the accuracy of nucleus segmentation. This enables the algorithm to accurately trace the cells and nucleus contours from the pap smear images that contain cell clusters. The method also accounts for inconsistent staining, if any. The results that are presented shows that our proposed method performs well even in challenging conditions.

count=1
* A Non-Local Low-Rank Framework for Ultrasound Speckle Reduction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhu_A_Non-Local_Low-Rank_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_A_Non-Local_Low-Rank_CVPR_2017_paper.pdf)]
    * Title: A Non-Local Low-Rank Framework for Ultrasound Speckle Reduction
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Lei Zhu, Chi-Wing Fu, Michael S. Brown, Pheng-Ann Heng
    * Abstract: `Speckle' refers to the granular patterns that occur in ultrasound images due to wave interference. Speckle removal can greatly improve the visibility of the underlying structures in an ultrasound image and enhance subsequent post processing. We present a novel framework for speckle removal based on low-rank non-local filtering. Our approach works by first computing a guidance image that assists in the selection of candidate patches for non-local filtering in the face of significant speckles. The candidate patches are further refined using a low-rank minimization estimated using a truncated weighted nuclear norm (TWNN) and structured sparsity. We show that the proposed filtering framework produces results that outperform state-of-the-art methods both qualitatively and quantitatively. This framework also provides better segmentation results when used for pre-processing ultrasound images.

count=1
* Localization and Tracking in 4D Fluorescence Microscopy Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w44/html/Abousamra_Localization_and_Tracking_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w44/Abousamra_Localization_and_Tracking_CVPR_2018_paper.pdf)]
    * Title: Localization and Tracking in 4D Fluorescence Microscopy Imagery
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Shahira Abousamra, Shai Adar, Natalie Elia, Roy Shilkrot
    * Abstract: 3D fluorescence microscopy continues to pose challenging tasks with more experiments leading to identifying new physiological patterns in cells' life cycle and activity. It then falls on the hands of biologists to annotate this imagery which is laborious and time-consuming, especially with noisy images and hard to see and track patterns. Modeling of automation tasks that can handle depth-varying light conditions and noise, and other challenges inherent in 3D fluorescence microscopy often becomes complex and requires high processing power and memory. This paper presents an efficient methodology for the localization, classification, and tracking in fluorescence microscopy imagery by taking advantage of time sequential images in 4D data. We show the application of our proposed method on the challenging task of localizing and tracking microtubule fibers' bridge formation during the cell division of zebrafish embryos where we achieve 98% accuracy and 0.94 F1- score.

count=1
* Cross-Classification Clustering: An Efficient Multi-Object Tracking Technique for 3-D Instance Segmentation in Connectomics
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Meirovitch_Cross-Classification_Clustering_An_Efficient_Multi-Object_Tracking_Technique_for_3-D_Instance_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Meirovitch_Cross-Classification_Clustering_An_Efficient_Multi-Object_Tracking_Technique_for_3-D_Instance_CVPR_2019_paper.pdf)]
    * Title: Cross-Classification Clustering: An Efficient Multi-Object Tracking Technique for 3-D Instance Segmentation in Connectomics
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yaron Meirovitch,  Lu Mi,  Hayk Saribekyan,  Alexander Matveev,  David Rolnick,  Nir Shavit
    * Abstract: Pixel-accurate tracking of objects is a key element in many computer vision applications, often solved by iterated individual object tracking or instance segmentation followed by object matching. Here we introduce cross-classification clustering (3C), a technique that simultaneously tracks complex, interrelated objects in an image stack. The key idea in cross-classification is to efficiently turn a clustering problem into a classification problem by running a logarithmic number of independent classifications per image, letting the cross-labeling of these classifications uniquely classify each pixel to the object labels. We apply the 3C mechanism to achieve state-of-the-art accuracy in connectomics -- the nanoscale mapping of neural tissue from electron microscopy volumes. Our reconstruction system increases scalability by an order of magnitude over existing single-object tracking methods (such as flood-filling networks). This scalability is important for the deployment of connectomics pipelines, since currently the best performing techniques require computing infrastructures that are beyond the reach of most laboratories. Our algorithm may offer benefits in other domains that require pixel-accurate tracking of multiple objects, such as segmentation of videos and medical imagery.

count=1
* Super-BPD: Super Boundary-to-Pixel Direction for Fast Image Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wan_Super-BPD_Super_Boundary-to-Pixel_Direction_for_Fast_Image_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wan_Super-BPD_Super_Boundary-to-Pixel_Direction_for_Fast_Image_Segmentation_CVPR_2020_paper.pdf)]
    * Title: Super-BPD: Super Boundary-to-Pixel Direction for Fast Image Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Jianqiang Wan,  Yang Liu,  Donglai Wei,  Xiang Bai,  Yongchao Xu
    * Abstract: Image segmentation is a fundamental vision task and still remains a crucial step for many applications. In this paper, we propose a fast image segmentation method based on a novel super boundary-to-pixel direction (super-BPD) and a customized segmentation algorithm with super-BPD. Precisely, we define BPD on each pixel as a two-dimensional unit vector pointing from its nearest boundary to the pixel. In the BPD, nearby pixels from different regions have opposite directions departing from each other, and nearby pixels in the same region have directions pointing to the other or each other (i.e., around medial points). We make use of such property to partition image into super-BPDs, which are novel informative superpixels with robust direction similarity for fast grouping into segmentation regions. Extensive experimental results on BSDS500 and Pascal Context demonstrate the accuracy and efficiency of the proposed super-BPD in segmenting images. Specifically, we achieve comparable or superior performance with MCG while running at 25fps vs 0.07fps. Super-BPD also exhibits a noteworthy transferability to unseen scenes.

count=1
* Tilted Cross-Entropy (TCE): Promoting Fairness in Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/RCV/html/Szabo_Tilted_Cross-Entropy_TCE_Promoting_Fairness_in_Semantic_Segmentation_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/RCV/papers/Szabo_Tilted_Cross-Entropy_TCE_Promoting_Fairness_in_Semantic_Segmentation_CVPRW_2021_paper.pdf)]
    * Title: Tilted Cross-Entropy (TCE): Promoting Fairness in Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Attila Szabo, Hadi Jamali-Rad, Siva-Datta Mannava
    * Abstract: Traditional empirical risk minimization (ERM) for semantic segmentation can disproportionately advantage or disadvantage certain target classes in favor of an (unfair but) improved overall performance. Inspired by the recently introduced tilted ERM (TERM), we propose tilted cross-entropy (TCE) loss and adapt it to the semantic segmentation setting to minimize performance disparity among target classes and promote fairness. Through quantitative and qualitative performance analyses, we demonstrate that the proposed Stochastic TCE for semantic segmentation can offer improved overall fairness by efficiently minimizing the performance disparity among the target classes of Cityscapes.

count=1
* GASP, a Generalized Framework for Agglomerative Clustering of Signed Graphs and Its Application to Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Bailoni_GASP_a_Generalized_Framework_for_Agglomerative_Clustering_of_Signed_Graphs_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bailoni_GASP_a_Generalized_Framework_for_Agglomerative_Clustering_of_Signed_Graphs_CVPR_2022_paper.pdf)]
    * Title: GASP, a Generalized Framework for Agglomerative Clustering of Signed Graphs and Its Application to Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Alberto Bailoni, Constantin Pape, Nathan Hütsch, Steffen Wolf, Thorsten Beier, Anna Kreshuk, Fred A. Hamprecht
    * Abstract: We propose a theoretical framework that generalizes simple and fast algorithms for hierarchical agglomerative clustering to weighted graphs with both attractive and repulsive interactions between the nodes. This framework defines GASP, a Generalized Algorithm for Signed graph Partitioning, and allows us to explore many combinations of different linkage criteria and cannot-link constraints. We prove the equivalence of existing clustering methods to some of those combinations and introduce new algorithms for combinations that have not been studied before. We study both theoretical and empirical properties of these combinations and prove that some of these define an ultrametric on the graph. We conduct a systematic comparison of various instantiations of GASP on a large variety of both synthetic and existing signed clustering problems, in terms of accuracy but also efficiency and robustness to noise. Lastly, we show that some of the algorithms included in our framework, when combined with the predictions from a CNN model, result in a simple bottom-up instance segmentation pipeline. Going all the way from pixels to final segments with a simple procedure, we achieve state-of-the-art accuracy on the CREMI 2016 EM segmentation benchmark without requiring domain-specific superpixels.

count=1
* TAPS3D: Text-Guided 3D Textured Shape Generation From Pseudo Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_TAPS3D_Text-Guided_3D_Textured_Shape_Generation_From_Pseudo_Supervision_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_TAPS3D_Text-Guided_3D_Textured_Shape_Generation_From_Pseudo_Supervision_CVPR_2023_paper.pdf)]
    * Title: TAPS3D: Text-Guided 3D Textured Shape Generation From Pseudo Supervision
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jiacheng Wei, Hao Wang, Jiashi Feng, Guosheng Lin, Kim-Hui Yap
    * Abstract: In this paper, we investigate an open research task of generating controllable 3D textured shapes from the given textual descriptions. Previous works either require ground truth caption labeling or extensive optimization time. To resolve these issues, we present a novel framework, TAPS3D, to train a text-guided 3D shape generator with pseudo captions. Specifically, based on rendered 2D images, we retrieve relevant words from the CLIP vocabulary and construct pseudo captions using templates. Our constructed captions provide high-level semantic supervision for generated 3D shapes. Further, in order to produce fine-grained textures and increase geometry diversity, we propose to adopt low-level image regularization to enable fake-rendered images to align with the real ones. During the inference phase, our proposed model can generate 3D textured shapes from the given text without any additional optimization. We conduct extensive experiments to analyze each of our proposed components and show the efficacy of our framework in generating high-fidelity 3D textured and text-relevant shapes.

count=1
* Action Probability Calibration for Efficient Naturalistic Driving Action Localization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/AICity/html/Li_Action_Probability_Calibration_for_Efficient_Naturalistic_Driving_Action_Localization_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Li_Action_Probability_Calibration_for_Efficient_Naturalistic_Driving_Action_Localization_CVPRW_2023_paper.pdf)]
    * Title: Action Probability Calibration for Efficient Naturalistic Driving Action Localization
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Rongchang Li, Cong Wu, Linze Li, Zhongwei Shen, Tianyang Xu, Xiao-jun Wu, Xi Li, Jiwen Lu, Josef Kittler
    * Abstract: The task of naturalistic driving action localization carries significant safety implications, as it involves detecting and identifying possible distracting driving behaviors in untrimmed videos. Previous studies have demonstrated that action localization using a local snippet followed by probability-based post-processing, without any training cost or redundant structure, can outperform existing learning-based paradigms. However, the action probability is computed at the snippet-level, the input information near the boundaries is attenuated, and the snippet size is limited, which does not support the generation of more precise action boundaries. To tackle these challenges, we introduce an action probability calibration module that expands snippet-level action probability to the frame-level, based on a preset snippet position reliability, without incurring additional costs for probability prediction. The frame-level action probability and reliability enable the use of various snippet sizes and equal treatment for information of different temporal points. Additionally, based on the calibrated probability, we further design a category-customized filtering mechanism to eliminate the redundant action candidates. Our method ranks 2nd on the public leaderboard, and the code is available at https://github.com/RongchangLi/AICity2023_DrivingAction.

count=1
* Hierarchical Histogram Threshold Segmentation - Auto-terminating High-detail Oversegmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Chang_Hierarchical_Histogram_Threshold_Segmentation_-_Auto-terminating_High-detail_Oversegmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Chang_Hierarchical_Histogram_Threshold_Segmentation_-_Auto-terminating_High-detail_Oversegmentation_CVPR_2024_paper.pdf)]
    * Title: Hierarchical Histogram Threshold Segmentation - Auto-terminating High-detail Oversegmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Thomas V. Chang, Simon Seibt, Bartosz von Rymon Lipinski
    * Abstract: Superpixels play a crucial role in image processing by partitioning an image into clusters of pixels with similar visual attributes. This facilitates subsequent image processing tasks offering computational advantages over the manipulation of individual pixels. While numerous oversegmentation techniques have emerged in recent years many rely on predefined initialization and termination criteria. In this paper a novel top-down superpixel segmentation algorithm called Hierarchical Histogram Threshold Segmentation (HHTS) is introduced. It eliminates the need for initialization and implements auto-termination outperforming state-of-the-art methods w.r.t boundary recall. This is achieved by iteratively partitioning individual pixel segments into foreground and background and applying intensity thresholding across multiple color channels. The underlying iterative process constructs a superpixel hierarchy that adapts to local detail distributions until color information exhaustion. Experimental results demonstrate the superiority of the proposed approach in terms of boundary adherence while maintaining competitive runtime performance on the BSDS500 and NYUV2 datasets. Furthermore an application of HHTS in refining machine learning-based semantic segmentation masks produced by the Segment Anything Foundation Model (SAM) is presented.

count=1
* Scale and Rotation Invariant Approach to Tracking Human Body Part Regions in Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W19/html/Bo_Scale_and_Rotation_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W19/papers/Bo_Scale_and_Rotation_2013_CVPR_paper.pdf)]
    * Title: Scale and Rotation Invariant Approach to Tracking Human Body Part Regions in Videos
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Yihang Bo, Hao Jiang
    * Abstract: We propose a novel scale and rotation invariant method to track a human subject's body part regions in cluttered videos. The proposed method optimizes the assembly of body part region proposals with the spatial and temporal constraints of a human body plan. This approach is invariant to the object scale and rotation changes. To enable scale and rotation invariance, the human body part graph of the proposed method has to be loopy; efficiently optimizing the body part region assembly is a great challenge. We propose a dynamic programming method to solve the problem. We devise a method that finds N-best whole body configurations from loopy structures in each video frame using dynamic programming. The N-best configurations are then used to construct trellises with which we track human body part regions by finding shortest paths on the trellises. Our experiments on a variety of videos show that the proposed method is efficient, accurate and robust against object appearance variations, scale and rotation changes and background clutter.

count=1
* Edge-Weighted Centroid Voronoi Tessellation with Propagation of Consistency Constraint for 3D Grain Segmentation in Microscopic Superalloy Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/html/Zhou_Edge-Weighted_Centroid_Voronoi_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/papers/Zhou_Edge-Weighted_Centroid_Voronoi_2014_CVPR_paper.pdf)]
    * Title: Edge-Weighted Centroid Voronoi Tessellation with Propagation of Consistency Constraint for 3D Grain Segmentation in Microscopic Superalloy Images
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Youjie Zhou, Lili Ju, Yu Cao, Jarrell Waggoner, Yuewei Lin, Jeff Simmons, Song Wang
    * Abstract: 3D microstructures are important for material scientists to analyze physical properties of materials. While such microstructures are too small to be directly visible to human vision, modern microscopic and serial-sectioning techniques can provide their high-resolution 3D images in the form of a sequence of 2D image slices. In this paper, we propose an algorithm based on the Edge-Weighted Centroid Voronoi Tessellation which uses propagation of the inter-slice consistency constraint. It can segment a 3D superalloy image, slice by slice, to obtain the underlying grain microstructures. With the propagation of the consistency constraint, the proposed method can automatically match grain segments between slices. On each of the 2D image slices, stable structures identified from the previous slice can be well-preserved, with further refinement by clustering the pixels in terms of both intensity and spatial information. We tested the proposed algorithm on a 3D superalloy image consisting of 170 2D slices. Performance is evaluated against manually annotated ground-truth segmentation. The results show that the proposed method outperforms several state-of-the-art 2D, 3D, and propagation-based segmentation methods in terms of both segmentation accuracy and running time.

count=1
* Detection and Segmentation of 2D Curved Reflection Symmetric Structures
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Teo_Detection_and_Segmentation_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Teo_Detection_and_Segmentation_ICCV_2015_paper.pdf)]
    * Title: Detection and Segmentation of 2D Curved Reflection Symmetric Structures
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Ching L. Teo, Cornelia Fermuller, Yiannis Aloimonos
    * Abstract: Symmetry, as one of the key components of Gestalt theory, provides an important mid-level cue that serves as input to higher visual processes such as segmentation. In this work, we propose a complete approach that links the detection of curved reflection symmetries to produce symmetry-constrained segments of structures/regions in real images with clutter. For curved reflection symmetry detection, we leverage on patch-based symmetric features to train a Structured Random Forest classifier that detects multiscaled curved symmetries in 2D images. Next, using these curved symmetries, we modulate a novel symmetry-constrained foreground-background segmentation by their symmetry scores so that we enforce global symmetrical consistency in the final segmentation. This is achieved by imposing a pairwise symmetry prior that encourages symmetric pixels to have the same labels over a MRF-based representation of the input image edges, and the final segmentation is obtained via graph-cuts. Experimental results over four publicly available datasets containing annotated symmetric structures: 1) SYMMAX-300, 2) BSD-Parts, 3) Weizmann Horse and 4) NY-roads demonstrate the approach's applicability to different environments with state-of-the-art performance.

count=1
* Multiresolution Hierarchy Co-Clustering for Semantic Segmentation in Sequences With Small Variations
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Varas_Multiresolution_Hierarchy_Co-Clustering_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Varas_Multiresolution_Hierarchy_Co-Clustering_ICCV_2015_paper.pdf)]
    * Title: Multiresolution Hierarchy Co-Clustering for Semantic Segmentation in Sequences With Small Variations
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: David Varas, Monica Alfaro, Ferran Marques
    * Abstract: This paper presents a co-clustering technique that, given a collection of images and their hierarchies, clusters nodes from these hierarchies to obtain a coherent multiresolution representation of the image collection. We formalize the co-clustering as Quadratic Semi-Assignment Problem and solve it with a linear programming relaxation approach that makes effective use of information from hierarchies. Initially, we address the problem of generating an optimal, coherent partition per image and, afterwards, we extend this method to a multiresolution framework. Finally, we particularize this framework to an iterative multiresolution video segmentation algorithm in sequences with small variations. We evaluate the algorithm on the Video Occlusion/Object Boundary Detection Dataset, showing that it produces state-of-the-art results in these scenarios.

count=1
* TORNADO: A Spatio-Temporal Convolutional Regression Network for Video Action Proposal
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_TORNADO_A_Spatio-Temporal_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_TORNADO_A_Spatio-Temporal_ICCV_2017_paper.pdf)]
    * Title: TORNADO: A Spatio-Temporal Convolutional Regression Network for Video Action Proposal
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Hongyuan Zhu, Romain Vial, Shijian Lu
    * Abstract: Given a video clip, action proposal aims to quickly generate a number of spatio-temporal tubes that enclose candidate human activities. Recently, the regression-based object detectors and long-term recurrent convolutional network (LRCN) have demonstrated superior performance in human action detection and recognition. However, the regression-based detectors performs inference without considering the temporal context among neighboring frames, and the LRCN using global visual percepts lacks the capability to capture local temporal dynamics. In this paper, we present a novel framework called TORNADO for human action proposal detection in un-trimmed video clips. Specifically, we propose a spatial-temporal convolutional network that combines the advantages of regression-based detector and LRCN by empowering Convolutional LSTM with regression capability. Our approach consists of a temporal convolutional regression network (T-CRN) and a spatial regression network (S-CRN) which are trained end-to-end on both RGB and OpticalFlow streams. They fuse appearance, motion and temporal contexts to regress the bounding boxes of candidate human actions simultaneously in 28 FPS. The action proposals are constructed by solving dynamic programming with peak trimming of the generated action boxes. Extensive experiments on the challenging UCF-101 and UCF-Sports datasets show that our method achieves superior performance as compared with the state-of-the-arts.

count=1
* Computer Vision for the Visually Impaired: The Sound of Vision System
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w22/html/Caraiman_Computer_Vision_for_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w22/Caraiman_Computer_Vision_for_ICCV_2017_paper.pdf)]
    * Title: Computer Vision for the Visually Impaired: The Sound of Vision System
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Simona Caraiman, Anca Morar, Mateusz Owczarek, Adrian Burlacu, Dariusz Rzeszotarski, Nicolae Botezatu, Paul Herghelegiu, Florica Moldoveanu, Pawel Strumillo, Alin Moldoveanu
    * Abstract: This paper presents a computer vision based sensory substitution device for the visually impaired. Its main objective is to provide the users with a 3D representation of the environment around them, conveyed by means of the hearing and tactile senses. One of the biggest challenges for this system is to ensure pervasiveness, i.e., to be usable in any indoor or outdoor environments and in any illumination conditions. This work reveals both the hardware (3D acquisition system) and software (3D processing pipeline) used for developing this sensory substitution device and provides insight on its exploitation in various scenarios. Preliminary experiments with blind users revealed good usability results and provided valuable feedback for system improvement.

count=1
* Geometry Based Faceting of 3D Digitized Archaeological Fragments
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w42/html/ElNaghy_Geometry_Based_Faceting_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w42/ElNaghy_Geometry_Based_Faceting_ICCV_2017_paper.pdf)]
    * Title: Geometry Based Faceting of 3D Digitized Archaeological Fragments
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Hanan ElNaghy, Leo Dorst
    * Abstract: We present a robust pipeline for segmenting digital cultural heritage fragments into distinct facets, with few tunable yet archaeologically meaningful parameters. Given a terracotta broken artifact, digitally scanned in the form of irregularly sampled 3D mesh, our method first estimates the local angles of fractures by applying weighted eigenanalysis of the local neighborhoods. Using 3D fit of a quadratic polynomial, we estimate the directional derivative of the angle function along the maximum bending direction for accurate localization of the fracture lines across the mesh. Then, the salient fracture lines are detected and incidental possible gaps between them are closed in order to extract a set of closed facets. Finally, the facets are categorized into fracture and skin. The method is tested on two different datasets of the GRAVITATE project.

count=1
* MEDIRL: Predicting the Visual Attention of Drivers via Maximum Entropy Deep Inverse Reinforcement Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Baee_MEDIRL_Predicting_the_Visual_Attention_of_Drivers_via_Maximum_Entropy_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Baee_MEDIRL_Predicting_the_Visual_Attention_of_Drivers_via_Maximum_Entropy_ICCV_2021_paper.pdf)]
    * Title: MEDIRL: Predicting the Visual Attention of Drivers via Maximum Entropy Deep Inverse Reinforcement Learning
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Sonia Baee, Erfan Pakdamanian, Inki Kim, Lu Feng, Vicente Ordonez, Laura Barnes
    * Abstract: Inspired by human visual attention, we propose a novel inverse reinforcement learning formulation using Maximum Entropy Deep Inverse Reinforcement Learning (MEDIRL) for predicting the visual attention of drivers in accident-prone situations. MEDIRL predicts fixation locations that lead to maximal rewards by learning a task-sensitive reward function from eye fixation patterns recorded from attentive drivers. Additionally, we introduce EyeCar, a new driver attention dataset in accident-prone situations. We conduct comprehensive experiments to evaluate our proposed model on three common benchmarks: (DR(eye)VE, BDD-A, DADA-2000), and our EyeCar dataset. Results indicate that MEDIRL outperforms existing models for predicting attention and achieves state-of-the-art performance. We present extensive ablation studies to provide more insights into different features of our proposed model.

count=1
* Partition Speeds Up Learning Implicit Neural Representations Based on Exponential-Increase Hypothesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Partition_Speeds_Up_Learning_Implicit_Neural_Representations_Based_on_Exponential-Increase_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Partition_Speeds_Up_Learning_Implicit_Neural_Representations_Based_on_Exponential-Increase_ICCV_2023_paper.pdf)]
    * Title: Partition Speeds Up Learning Implicit Neural Representations Based on Exponential-Increase Hypothesis
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ke Liu, Feng Liu, Haishuai Wang, Ning Ma, Jiajun Bu, Bo Han
    * Abstract: Implicit neural representations (INRs) aim to learn a continuous function (i.e., a neural network) to represent an image, where the input and output of the function are pixel coordinates and RGB/Gray values, respectively. However, images tend to consist of many objects whose colors are not perfectly consistent, resulting in the challenge that image is actually a discontinuous piecewise function and cannot be well estimated by a continuous function. In this paper, we empirically investigate that if a neural network is enforced to fit a discontinuous piecewise function to reach a fixed small error, the time costs will increase exponentially with respect to the boundaries in the spatial domain of the target signal. We name this phenomenon the exponential-increase hypothesis. Under the exponential-increase hypothesis, learning INRs for images with many objects will converge very slowly. To address this issue, we first prove that partitioning a complex signal into several sub-regions and utilizing piecewise INRs to fit that signal can significantly speed up the convergence. Based on this fact, we introduce a simple partition mechanism to boost the performance of two INR methods for image reconstruction: one for learning INRs, and the other for learning-to-learn INRs. In both cases, we partition an image into different sub-regions and dedicate smaller networks for each part. In addition, we further propose two partition rules based on regular grids and semantic segmentation maps, respectively. Extensive experiments validate the effectiveness of the proposed partitioning methods in terms of learning INR for a single image (ordinary learning framework) and the learning-to-learn framework.

count=1
* Quantifying Statistical Significance of Neural Network-based Image Segmentation by Selective Inference
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/cd706106802dbea2068efd7031c3b420-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/cd706106802dbea2068efd7031c3b420-Paper-Conference.pdf)]
    * Title: Quantifying Statistical Significance of Neural Network-based Image Segmentation by Selective Inference
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Vo Nguyen Le Duy, Shogo Iwazaki, Ichiro Takeuchi
    * Abstract: Although a vast body of literature relates to image segmentation methods that use deep neural networks (DNNs), less attention has been paid to assessing the statistical reliability of segmentation results. In this study, we interpret the segmentation results as hypotheses driven by DNN (called DNN-driven hypotheses) and propose a method to quantify the reliability of these hypotheses within a statistical hypothesis testing framework. To this end, we introduce a conditional selective inference (SI) framework---a new statistical inference framework for data-driven hypotheses that has recently received considerable attention---to compute exact (non-asymptotic) valid p-values for the segmentation results. To use the conditional SI framework for DNN-based segmentation, we develop a new SI algorithm based on the homotopy method, which enables us to derive the exact (non-asymptotic) sampling distribution of DNN-driven hypothesis. We conduct several experiments to demonstrate the performance of the proposed method.

count=1
* Hierarchical Open-vocabulary Universal Image Segmentation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/43663f64775ae439ec52b64305d219d3-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/43663f64775ae439ec52b64305d219d3-Paper-Conference.pdf)]
    * Title: Hierarchical Open-vocabulary Universal Image Segmentation
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Xudong Wang, Shufan Li, Konstantinos Kallidromitis, Yusuke Kato, Kazuki Kozuka, Trevor Darrell
    * Abstract: Open-vocabulary image segmentation aims to partition an image into semantic regions according to arbitrary text descriptions. However, complex visual scenes can be naturally decomposed into simpler parts and abstracted at multiple lev4 els of granularity, introducing inherent segmentation ambiguity. Unlike existing methods that typically sidestep this ambiguity and treat it as an external factor, our approach actively incorporates a hierarchical representation encompassing different semantic-levels into the learning process. We propose a decoupled text-image fusion mechanism and representation learning modules for both “things” and “stuff”. Additionally, we systematically examine the differences that exist in the textual and visual features between these types of categories. Our resulting model, named HIPIE, tackles HIerarchical, oPen-vocabulary, and unIvErsal segmentation tasks within a unified framework. Benchmarked on diverse datasets, e.g., ADE20K,COCO, Pascal-VOC Part, and RefCOCO/RefCOCOg, HIPIE achieves the state-of14 the-art results at various levels of image comprehension, including semantic-level (e.g., semantic segmentation), instance-level (e.g., panoptic/referring segmentationand object detection), as well as part-level (e.g., part/subpart segmentation) tasks.

count=1
* Nearly Tight Bounds For Differentially Private Multiway Cut
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/4e8f257e054abd24c550d55e57cec274-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/4e8f257e054abd24c550d55e57cec274-Paper-Conference.pdf)]
    * Title: Nearly Tight Bounds For Differentially Private Multiway Cut
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Mina Dalirrooyfard, Slobodan Mitrovic, Yuriy Nevmyvaka
    * Abstract: Finding min $s$-$t$ cuts in graphs is a basic algorithmic tool, with applications in image segmentation, community detection, reinforcement learning, and data clustering. In this problem, we are given two nodes as terminals and the goal is to remove the smallest number of edges from the graph so that these two terminals are disconnected. We study the complexity of differential privacy for the min $s$-$t$ cut problem and show nearly tight lower and upper bounds where we achieve privacy at no cost for running time efficiency. We also develop a differentially private algorithm for the multiway $k$-cut problem, in which we are given $k$ nodes as terminals that we would like to disconnect. As a function of $k$, we obtain privacy guarantees that are exponentially more efficient than applying the advanced composition theorem to known algorithms for multiway $k$-cut. Finally, we empirically evaluate the approximation of our differentially private min $s$-$t$ cut algorithm and show that it almost matches the quality of the output of non-private ones.

count=1
* Annotator: A Generic Active Learning Baseline for LiDAR Semantic Segmentation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/976cc04f0cbaad7790ce0d665e44f90f-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/976cc04f0cbaad7790ce0d665e44f90f-Paper-Conference.pdf)]
    * Title: Annotator: A Generic Active Learning Baseline for LiDAR Semantic Segmentation
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Binhui Xie, Shuang Li, Qingju Guo, Chi Liu, Xinjing Cheng
    * Abstract: Active learning, a label-efficient paradigm, empowers models to interactively query an oracle for labeling new data. In the realm of LiDAR semantic segmentation, the challenges stem from the sheer volume of point clouds, rendering annotation labor-intensive and cost-prohibitive. This paper presents Annotator, a general and efficient active learning baseline, in which a voxel-centric online selection strategy is tailored to efficiently probe and annotate the salient and exemplar voxel girds within each LiDAR scan, even under distribution shift. Concretely, we first execute an in-depth analysis of several common selection strategies such as Random, Entropy, Margin, and then develop voxel confusion degree (VCD) to exploit the local topology relations and structures of point clouds. Annotator excels in diverse settings, with a particular focus on active learning (AL), active source-free domain adaptation (ASFDA), and active domain adaptation (ADA). It consistently delivers exceptional performance across LiDAR semantic segmentation benchmarks, spanning both simulation-to-real and real-to-real scenarios. Surprisingly, Annotator exhibits remarkable efficiency, requiring significantly fewer annotations, e.g., just labeling five voxels per scan in the SynLiDAR → SemanticKITTI task. This results in impressive performance, achieving 87.8% fully-supervised performance under AL, 88.5% under ASFDA, and 94.4% under ADA. We envision that Annotator will offer a simple, general, and efficient solution for label-efficient 3D applications.

