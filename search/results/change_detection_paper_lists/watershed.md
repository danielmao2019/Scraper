count=88
* Probabilistic Watershed: Sampling all spanning forests for seeded segmentation and semi-supervised learning
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/49af6c4e558a7569d80eee2e035e2bd7-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/49af6c4e558a7569d80eee2e035e2bd7-Paper.pdf)]
    * Title: Probabilistic Watershed: Sampling all spanning forests for seeded segmentation and semi-supervised learning
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Enrique Fita Sanmartin, Sebastian Damrich, Fred A. Hamprecht
    * Abstract: The seeded Watershed algorithm / minimax semi-supervised learning on a graph computes a minimum spanning forest which connects every pixel / unlabeled node to a seed / labeled node. We propose instead to consider all possible spanning forests and calculate, for every node, the probability of sampling a forest connecting a certain seed with that node. We dub this approach "Probabilistic Watershed". Leo Grady (2006) already noted its equivalence to the Random Walker / Harmonic energy minimization. We here give a simpler proof of this equivalence and establish the computational feasibility of the Probabilistic Watershed with Kirchhoff's matrix tree theorem. Furthermore, we show a new connection between the Random Walker probabilities and the triangle inequality of the effective resistance. Finally, we derive a new and intuitive interpretation of the Power Watershed.

count=52
* Learned Watershed: End-To-End Learning of Seeded Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Wolf_Learned_Watershed_End-To-End_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wolf_Learned_Watershed_End-To-End_ICCV_2017_paper.pdf)]
    * Title: Learned Watershed: End-To-End Learning of Seeded Segmentation
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Steffen Wolf, Lukas Schott, Ullrich Kothe, Fred Hamprecht
    * Abstract: Learned boundary maps are known to outperform hand-crafted ones as a basis for the watershed algorithm. We show, for the first time, how to train watershed computation jointly with boundary map prediction. The estimator for the merging priorities is cast as a neural network that is convolutional (over space) and recurrent (over iterations). The latter allows learning of complex shape priors. The method gives the best known seeded segmentation results on the CREMI segmentation challenge.

count=49
* Directed Probabilistic Watershed
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/a73d9b34d6f7c322fa3e34c633b1297d-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/a73d9b34d6f7c322fa3e34c633b1297d-Paper.pdf)]
    * Title: Directed Probabilistic Watershed
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Enrique Fita Sanmartin, Sebastian Damrich, Fred A. Hamprecht
    * Abstract: The Probabilistic Watershed is a semi-supervised learning algorithm applied on undirected graphs. Given a set of labeled nodes (seeds), it defines a Gibbs probability distribution over all possible spanning forests disconnecting the seeds. It calculates, for every node, the probability of sampling a forest connecting a certain seed with the considered node. We propose the "Directed Probabilistic Watershed", an extension of the Probabilistic Watershed algorithm to directed graphs. Building on the Probabilistic Watershed, we apply the Matrix Tree Theorem for directed graphs and define a Gibbs probability distribution over all incoming directed forests rooted at the seeds. Similar to the undirected case, this turns out to be equivalent to the Directed Random Walker. Furthermore, we show that in the limit case in which the Gibbs distribution has infinitely low temperature, the labeling of the Directed Probabilistic Watershed is equal to the one induced by the incoming directed forest of minimum cost. Finally, for illustration, we compare the empirical performance of the proposed method with other semi-supervised segmentation methods for directed graphs.

count=39
* Deep Watershed Transform for Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Bai_Deep_Watershed_Transform_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Bai_Deep_Watershed_Transform_CVPR_2017_paper.pdf)]
    * Title: Deep Watershed Transform for Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Min Bai, Raquel Urtasun
    * Abstract: Most contemporary approaches to instance segmentation use complex pipelines involving conditional random fields, recurrent neural networks, object proposals, or template matching schemes. In this paper, we present a simple yet powerful end-to-end convolutional neural network to tackle this task. Our approach combines intuitions from the classical watershed transform and modern deep learning to produce an energy map of the image where object instances are unambiguously represented as energy basins. We then perform a cut at a single energy level to directly yield connected components corresponding to object instances. Our model achieves more than double the performance over the state-of-the-art on the challenging Cityscapes Instance Level Segmentation task.

count=21
* Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Weigert_Star-convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Weigert_Star-convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.pdf)]
    * Title: Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Martin Weigert,  Uwe Schmidt,  Robert Haase,  Ko Sugawara,  Gene Myers
    * Abstract: Accurate detection and segmentation of cell nuclei in volumetric (3D) fluorescence microscopy datasets is an important step in many biomedical research projects. Although many automated methods for these tasks exist, they often struggle for images with low signal-to-noise ratios and/or dense packing of nuclei. It was recently shown for 2D microscopy images that these issues can be alleviated by training a neural network to directly predict a suitable shape representation (star-convex polygon) for cell nuclei. In this paper, we adopt and extend this approach to 3D volumes by using star-convex polyhedra to represent cell nuclei and similar shapes. To that end, we overcome the challenges of 1) finding parameter-efficient star-convex polyhedra representations that can faithfully describe cell nuclei shapes, 2) adapting to anisotropic voxel sizes often found in fluorescence microscopy datasets, and 3) efficiently computing intersections between pairs of star-convex polyhedra (required for non-maximum suppression). Although our approach is quite general, since star-convex polyhedra include common shapes like bounding boxes and spheres as special cases, our focus is on accurate detection and segmentation of cell nuclei. Finally, we demonstrate on two challenging datasets that our approach (StarDist-3D) leads to superior results when compared to classical and deep learning based methods.

count=20
* Extensions of Karger's Algorithm: Why They Fail in Theory and How They Are Useful in Practice
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Jenner_Extensions_of_Kargers_Algorithm_Why_They_Fail_in_Theory_and_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Jenner_Extensions_of_Kargers_Algorithm_Why_They_Fail_in_Theory_and_ICCV_2021_paper.pdf)]
    * Title: Extensions of Karger's Algorithm: Why They Fail in Theory and How They Are Useful in Practice
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Erik Jenner, Enrique Fita Sanmartín, Fred A. Hamprecht
    * Abstract: The minimum graph cut and minimum s-t-cut problems are important primitives in the modeling of combinatorial problems in computer science, including in computer vision and machine learning. Some of the most efficient algorithms for finding global minimum cuts are randomized algorithms based on Karger's groundbreaking contraction algorithm. Here, we study whether Karger's algorithm can be successfully generalized to other cut problems. We first prove that a wide class of natural generalizations of Karger's algorithm cannot efficiently solve the s-t-mincut or the normalized cut problem to optimality. However, we then present a simple new algorithm for seeded segmentation / graph-based semi-supervised learning that is closely based on Karger's original algorithm, showing that for these problems, extensions of Karger's algorithm can be useful. The new algorithm has linear asymptotic runtime and yields a potential that can be interpreted as the posterior probability of a sample belonging to a given seed / class. We clarify its relation to the random walker algorithm / harmonic energy minimization in terms of distributions over spanning forests. On classical problems from seeded image segmentation and graph-based semi-supervised learning on image data, the method performs at least as well as the random walker / harmonic energy minimization / Gaussian processes.

count=19
* Graph-Based Optimization with Tubularity Markov Tree for 3D Vessel Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Zhu_Graph-Based_Optimization_with_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Zhu_Graph-Based_Optimization_with_2013_CVPR_paper.pdf)]
    * Title: Graph-Based Optimization with Tubularity Markov Tree for 3D Vessel Segmentation
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Ning Zhu, Albert C.S. Chung
    * Abstract: In this paper, we propose a graph-based method for 3D vessel tree structure segmentation based on a new tubularity Markov tree model (TMT ), which works as both new energy function and graph construction method. With the help of power-watershed implementation [7], a global optimal segmentation can be obtained with low computational cost. Different with other graph-based vessel segmentation methods, the proposed method does not depend on any skeleton and ROI extraction method. The classical issues of the graph-based methods, such as shrinking bias and sensitivity to seed point location, can be solved with the proposed method thanks to vessel data fidelity obtained with TMT . The proposed method is compared with some classical graph-based image segmentation methods and two up-to-date 3D vessel segmentation methods, and is demonstrated to be more accurate than these methods for 3D vessel tree segmentation. Although the segmentation is done without ROI extraction, the computational cost for the proposed method is low (within 20 seconds for 256*256*144 image).

count=19
* End-To-End Learned Random Walker for Seeded Image Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Cerrone_End-To-End_Learned_Random_Walker_for_Seeded_Image_Segmentation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cerrone_End-To-End_Learned_Random_Walker_for_Seeded_Image_Segmentation_CVPR_2019_paper.pdf)]
    * Title: End-To-End Learned Random Walker for Seeded Image Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Lorenzo Cerrone,  Alexander Zeilmann,  Fred A. Hamprecht
    * Abstract: We present an end-to-end learned algorithm for seeded segmentation. Our method is based on the Random Walker algorithm, where we predict the edge weights of the un- derlying graph using a convolutional neural network. This can be interpreted as learning context-dependent diffusiv- ities for a linear diffusion process. After calculating the exact gradient for optimizing these diffusivities, we pro- pose simplifications that sparsely sample the gradient while still maintaining competitive results. The proposed method achieves the currently best results on the seeded CREMI neuron segmentation challenge.

count=17
* Learning To Correct Sloppy Annotations in Electron Microscopy Volumes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVMI/html/Chen_Learning_To_Correct_Sloppy_Annotations_in_Electron_Microscopy_Volumes_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVMI/papers/Chen_Learning_To_Correct_Sloppy_Annotations_in_Electron_Microscopy_Volumes_CVPRW_2023_paper.pdf)]
    * Title: Learning To Correct Sloppy Annotations in Electron Microscopy Volumes
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Minghao Chen, Mukesh Bangalore Renuka, Lu Mi, Jeff Lichtman, Nir Shavit, Yaron Meirovitch
    * Abstract: Connectomics deals with the problem of reconstructing neural circuitry from electron microscopy images at the synaptic level. Automatically reconstructing circuits from these volumes requires high fidelity 3-D instance segmentation, which yet appears to be a daunting task for current computer vision algorithms. Hence, to date, most datasets are not reconstructed by fully-automated methods. Even after painstaking proofreading, these methods still produce numerous small errors. In this paper, we propose an approach to accelerate manual reconstructions by learning to correct imperfect manual annotations. To achieve this, we designed a novel solution for the canonical problem of marker-based 2-D instance segmentation, reporting a new state-of-the-art for region-growing algorithms demonstrated on challenging electron microscopy image stacks. We use our marker-based instance segmentation algorithm to learn to correct all "sloppy" object annotations by reducing and expanding all annotations. Our correction algorithm results in high quality morphological reconstruction (near ground truth quality), while significantly cutting annotation time ( 8x) for several examples in connectomics. We demonstrate the accuracy of our approach on public connectomics benchmarks and on a set of large-scale neuron reconstruction problems, including on a new octopus dataset that cannot be automatically segmented at scale by existing algorithms.

count=16
* Anonymous and Copy-Robust Delegations for Liquid Democracy
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/dbb5180957513805ebeea787b8c66ac9-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/dbb5180957513805ebeea787b8c66ac9-Paper-Conference.pdf)]
    * Title: Anonymous and Copy-Robust Delegations for Liquid Democracy
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Markus Utke, Ulrike Schmidt-Kraepelin
    * Abstract: Liquid democracy with ranked delegations is a novel voting scheme that unites the practicability of representative democracy with the idealistic appeal of direct democracy: Every voter decides between casting their vote on a question at hand or delegating their voting weight to some other, trusted agent. Delegations are transitive, and since voters may end up in a delegation cycle, they are encouraged to indicate not only a single delegate, but a set of potential delegates and a ranking among them. Based on the delegation preferences of all voters, a delegation rule selects one representative per voter. Previous work has revealed a trade-off between two properties of delegation rules called anonymity and copy-robustness. To overcome this issue we study two fractional delegation rules: Mixed Borda branching, which generalizes a rule satisfying copy-robustness, and the random walk rule, which satisfies anonymity. Using the Markov chain tree theorem, we show that the two rules are in fact equivalent, and simultaneously satisfy generalized versions of the two properties. Combining the same theorem with Fulkerson's algorithm, we develop a polynomial-time algorithm for computing the outcome of the studied delegation rule. This algorithm is of independent interest, having applications in semi-supervised learning and graph theory.

count=15
* An Ensemble Learning and Slice Fusion Strategy for Three-Dimensional Nuclei Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/CVMI/html/Wu_An_Ensemble_Learning_and_Slice_Fusion_Strategy_for_Three-Dimensional_Nuclei_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/CVMI/papers/Wu_An_Ensemble_Learning_and_Slice_Fusion_Strategy_for_Three-Dimensional_Nuclei_CVPRW_2022_paper.pdf)]
    * Title: An Ensemble Learning and Slice Fusion Strategy for Three-Dimensional Nuclei Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Liming Wu, Alain Chen, Paul Salama, Kenneth W. Dunn, Edward J. Delp
    * Abstract: Automated microscopy image analysis is a fundamental step for digital pathology and computer aided diagnosis. Most existing deep learning methods typically require postprocessing to achieve instance segmentation and are computationally expensive when directly used with 3D microscopy volumes. Supervised learning methods generally need large amounts of ground truth annotations for training whereas manually annotating ground truth masks is laborious especially for a 3D volume. To address these issues, we propose an ensemble learning and slice fusion strategy for 3D nuclei instance segmentation that we call Ensemble Mask R-CNN (EMR-CNN) which uses different object detectors to generate nuclei segmentation masks for each 2D slice of a volume and propose a 2D ensemble fusion and a 2D to 3D slice fusion to merge these 2D segmentation masks into a 3D segmentation mask. Our method does not need any ground truth annotations for training and can inference on any large size volumes. Our proposed method was tested on a variety of microscopy volumes collected from multiple regions of organ tissues. The execution time and robustness analyses show that our method is practical and effective.

count=15
* Finding Berries: Segmentation and Counting of Cranberries Using Point Supervision and Shape Priors
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w5/Akiva_Finding_Berries_Segmentation_and_Counting_of_Cranberries_Using_Point_Supervision_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w5/Akiva_Finding_Berries_Segmentation_and_Counting_of_Cranberries_Using_Point_Supervision_CVPRW_2020_paper.pdf)]
    * Title: Finding Berries: Segmentation and Counting of Cranberries Using Point Supervision and Shape Priors
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Peri Akiva, Kristin Dana, Peter Oudemans, Michael Mars
    * Abstract: Precision agriculture has become a key factor for increasing crop yields by providing essential information to decision makers. In this work, we present a deep learning method for simultaneous segmentation and counting of cranberries to aid in yield estimation and sun exposure predictions. Notably, supervision is done using low cost center point annotations. The approach, named Triple-S Network, incorporates a three-part loss with shape priors to promote better fitting to objects of known shape typical in agricultural scenes. Our results improve overall segmentation performance by more than 6.74% and counting results by 22.91% when compared to state-of-the-art. To train and evaluate the network, we have collected the CRanberry Aerial Imagery Dataset (CRAID), the largest dataset of aerial drone imagery from cranberry fields. This dataset will be made publicly available.

count=13
* Self-Supervised Learning via Conditional Motion Propagation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhan_Self-Supervised_Learning_via_Conditional_Motion_Propagation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhan_Self-Supervised_Learning_via_Conditional_Motion_Propagation_CVPR_2019_paper.pdf)]
    * Title: Self-Supervised Learning via Conditional Motion Propagation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Xiaohang Zhan,  Xingang Pan,  Ziwei Liu,  Dahua Lin,  Chen Change Loy
    * Abstract: Intelligent agent naturally learns from motion. Various self-supervised algorithms have leveraged the motion cues to learn effective visual representations. The hurdle here is that motion is both ambiguous and complex, rendering previous works either suffer from degraded learning efficacy, or resort to strong assumptions on object motions. In this work, we design a new learning-from-motion paradigm to bridge these gaps. Instead of explicitly modeling the motion probabilities, we design the pretext task as a conditional motion propagation problem. Given an input image and several sparse flow guidance on it, our framework seeks to recover the full-image motion. Compared to other alternatives, our framework has several appealing properties: (1) Using sparse flow guidance during training resolves the inherent motion ambiguity, and thus easing feature learning. (2) Solving the pretext task of conditional motion propagation encourages the emergence of kinematically-sound representations that poss greater expressive power. Extensive experiments demonstrate that our framework learns structural and coherent features; and achieves state-of-the-art self-supervision performance on several downstream tasks including semantic segmentation, instance segmentation and human parsing. Furthermore, our framework is successfully extended to several useful applications such as semi-automatic pixel-level annotation.

count=13
* RCNN-SliceNet: A Slice and Cluster Approach for Nuclei Centroid Detection in Three-Dimensional Fluorescence Microscopy Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/CVMI/html/Wu_RCNN-SliceNet_A_Slice_and_Cluster_Approach_for_Nuclei_Centroid_Detection_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/CVMI/papers/Wu_RCNN-SliceNet_A_Slice_and_Cluster_Approach_for_Nuclei_Centroid_Detection_CVPRW_2021_paper.pdf)]
    * Title: RCNN-SliceNet: A Slice and Cluster Approach for Nuclei Centroid Detection in Three-Dimensional Fluorescence Microscopy Images
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Liming Wu, Shuo Han, Alain Chen, Paul Salama, Kenneth W. Dunn, Edward J. Delp
    * Abstract: Robust and accurate nuclei centroid detection is important for the understanding of biological structures in fluorescence microscopy images. Existing automated nuclei localization methods face three main challenges: (1) Most of object detection methods work only on 2D images and are difficult to extend to 3D volumes; (2) Segmentation-based models can be used on 3D volumes but it is computational expensive for large microscopy volumes and they have difficulty distinguishing different instances of objects; (3) Hand annotated ground truth is limited for 3D microscopy volumes. To address these issues, we present a scalable approach for nuclei centroid detection of 3D microscopy volumes. We describe the RCNN-SliceNet to detect 2D nuclei centroids for each slice of the volume from different directions and 3D agglomerative hierarchical clustering (AHC) is used to estimate the 3D centroids of nuclei in a volume. The model was trained with the synthetic microscopy data generated using Spatially Constrained Cycle-Consistent Adversarial Networks (SpCycleGAN) and tested on different types of real 3D microscopy data. Extensive experimental results demonstrate that our proposed method can accurately count and detect the nuclei centroids in a 3D microscopy volume.

count=12
* Laplacian Coordinates for Seeded Image Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Casaca_Laplacian_Coordinates_for_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Casaca_Laplacian_Coordinates_for_2014_CVPR_paper.pdf)]
    * Title: Laplacian Coordinates for Seeded Image Segmentation
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Wallace Casaca, Luis Gustavo Nonato, Gabriel Taubin
    * Abstract: Seed-based image segmentation methods have gained much attention lately, mainly due to their good performance in segmenting complex images with little user interaction. Such popularity leveraged the development of many new variations of seed-based image segmentation techniques, which vary greatly regarding mathematical formulation and complexity. Most existing methods in fact rely on complex mathematical formulations that typically do not guarantee unique solution for the segmentation problem while still being prone to be trapped in local minima. In this work we present a novel framework for seed-based image segmentation that is mathematically simple, easy to implement, and guaranteed to produce a unique solution. Moreover, the formulation holds an anisotropic behavior, that is, pixels sharing similar attributes are kept closer to each other while big jumps are naturally imposed on the boundary between image regions, thus ensuring better fitting on object boundaries. We show that the proposed framework outperform state-of-the-art techniques in terms of quantitative quality metrics as well as qualitative visual results.

count=12
* Deep Learning Based Corn Kernel Classification
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w5/Velesaca_Deep_Learning_Based_Corn_Kernel_Classification_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w5/Velesaca_Deep_Learning_Based_Corn_Kernel_Classification_CVPRW_2020_paper.pdf)]
    * Title: Deep Learning Based Corn Kernel Classification
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Henry O. Velesaca, Raul Mira, Patricia L. Suarez, Christian X. Larrea, Angel D. Sappa
    * Abstract: This paper presents a full pipeline to classify sample sets of corn kernels. The proposed approach follows a segmentation-classification scheme. The image segmentation is performed through a well known deep learning-based approach, the Mask R-CNN architecture, while the classification is performed through a novel-lightweight network specially designed for this task---good corn kernel, defective corn kernel and impurity categories are considered. As a second contribution, a carefully annotated multi-touching corn kernel dataset has been generated. This dataset has been used for training the segmentation and the classification modules. Quantitative evaluations have been performed and comparisons with other approaches are provided showing improvements with the proposed pipeline.

count=12
* Efficient Classifier Training to Minimize False Merges in Electron Microscopy Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Parag_Efficient_Classifier_Training_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Parag_Efficient_Classifier_Training_ICCV_2015_paper.pdf)]
    * Title: Efficient Classifier Training to Minimize False Merges in Electron Microscopy Segmentation
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Toufiq Parag, Dan C. Ciresan, Alessandro Giusti
    * Abstract: The prospect of neural reconstruction from Electron Microscopy (EM) images has been elucidated by the automatic segmentation algorithms. Although segmentation algorithms eliminate the necessity of tracing the neurons by hand, significant manual effort is still essential for correcting the mistakes they make. A considerable amount of human labor is also required for annotating groundtruth volumes for training the classifiers of a segmentation framework. It is critically important to diminish the dependence on human interaction in the overall reconstruction system. This study proposes a novel classifier training algorithm for EM segmentation aimed to reduce the amount of manual effort demanded by the groundtruth annotation and error refinement tasks. Instead of using an exhaustive pixel level groundtruth, an active learning algorithm is proposed for sparse labeling of pixel and boundaries of superpixels. Because over-segmentation errors are in general more tolerable and easier to correct than the under-segmentation errors, our algorithm is designed to prioritize minimization of false-merges over false-split mistakes. Our experiments on both 2D and 3D data suggest that the proposed method yields segmentation outputs that are more amenable to neural reconstruction than those of existing methods.

count=12
* Large Scale Labelled Video Data Augmentation for Semantic Segmentation in Driving Scenarios
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w3/html/Budvytis_Large_Scale_Labelled_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w3/Budvytis_Large_Scale_Labelled_ICCV_2017_paper.pdf)]
    * Title: Large Scale Labelled Video Data Augmentation for Semantic Segmentation in Driving Scenarios
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Ignas Budvytis, Patrick Sauer, Thomas Roddick, Kesar Breen, Roberto Cipolla
    * Abstract: In this paper we present an analysis of the effect of large scale video data augmentation for semantic segmentation in driving scenarios. Our work is motivated by a strong correlation between the high performance of most recent deep learning based methods and the availability of large volumes of ground truth labels. To generate additional labelled data, we make use of an occlusion-aware and uncertaintyenabled label propagation algorithm. As a result we increase the availability of high-resolution labelled frames by a factor of 20, yielding in a 6.8% to 10.8% rise in average classification accuracy and/or IoU scores for several semantic segmentation networks. Our key contributions include: (a) augmented CityScapes and CamVid datasets providing 56.2K and 6.5K additional labelled frames of object classes respectively, (b) detailed empirical analysis of the effect of the use of augmented data as well as (c) extension of proposed framework to instance segmentation.

count=11
* Locate n' Rotate: Two-stage Openable Part Detection with Geometric Foundation Model Priors
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Li_Locate_n_Rotate_Two-stage_Openable_Part_Detection_with_Geometric_Foundation_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Li_Locate_n_Rotate_Two-stage_Openable_Part_Detection_with_Geometric_Foundation_ACCV_2024_paper.pdf)]
    * Title: Locate n' Rotate: Two-stage Openable Part Detection with Geometric Foundation Model Priors
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Siqi Li, Xiaoxue Chen, Haoyu Cheng, Guyue Zhou, Hao Zhao, Guanzhong Tian
    * Abstract: Detecting the openable parts of articulated objects is crucial for downstream applications in intelligent robotics, such as pulling a drawer. This task poses a multitasking challenge due to the necessity of understanding object categories and motion. Most existing methods are either category-specific or trained on specific datasets, lacking generalization to unseen environments and objects. In this paper, we propose a Transformer-based Openable Part Detection (OPD) framework named Multi-feature Openable Part Detection (MOPD) that incorporates perceptual grouping and geometric priors, outperforming previous methods in performance. In the first stage of the framework, we introduce a perceptual grouping feature model that provides perceptual grouping feature priors for openable part detection, enhancing detection results through a cross-attention mechanism. In the second stage, a geometric understanding feature model offers geometric feature priors for predicting motion parameters. Compared to existing methods, our proposed approach shows better performance in both detection and motion parameter prediction. Codes and models are publicly available at https://github.com/lisiqi-zju/MOPD.

count=11
* Super-BPD: Super Boundary-to-Pixel Direction for Fast Image Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wan_Super-BPD_Super_Boundary-to-Pixel_Direction_for_Fast_Image_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wan_Super-BPD_Super_Boundary-to-Pixel_Direction_for_Fast_Image_Segmentation_CVPR_2020_paper.pdf)]
    * Title: Super-BPD: Super Boundary-to-Pixel Direction for Fast Image Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Jianqiang Wan,  Yang Liu,  Donglai Wei,  Xiang Bai,  Yongchao Xu
    * Abstract: Image segmentation is a fundamental vision task and still remains a crucial step for many applications. In this paper, we propose a fast image segmentation method based on a novel super boundary-to-pixel direction (super-BPD) and a customized segmentation algorithm with super-BPD. Precisely, we define BPD on each pixel as a two-dimensional unit vector pointing from its nearest boundary to the pixel. In the BPD, nearby pixels from different regions have opposite directions departing from each other, and nearby pixels in the same region have directions pointing to the other or each other (i.e., around medial points). We make use of such property to partition image into super-BPDs, which are novel informative superpixels with robust direction similarity for fast grouping into segmentation regions. Extensive experimental results on BSDS500 and Pascal Context demonstrate the accuracy and efficiency of the proposed super-BPD in segmenting images. Specifically, we achieve comparable or superior performance with MCG while running at 25fps vs 0.07fps. Super-BPD also exhibits a noteworthy transferability to unseen scenes.

count=11
* GASP, a Generalized Framework for Agglomerative Clustering of Signed Graphs and Its Application to Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Bailoni_GASP_a_Generalized_Framework_for_Agglomerative_Clustering_of_Signed_Graphs_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bailoni_GASP_a_Generalized_Framework_for_Agglomerative_Clustering_of_Signed_Graphs_CVPR_2022_paper.pdf)]
    * Title: GASP, a Generalized Framework for Agglomerative Clustering of Signed Graphs and Its Application to Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Alberto Bailoni, Constantin Pape, Nathan Hütsch, Steffen Wolf, Thorsten Beier, Anna Kreshuk, Fred A. Hamprecht
    * Abstract: We propose a theoretical framework that generalizes simple and fast algorithms for hierarchical agglomerative clustering to weighted graphs with both attractive and repulsive interactions between the nodes. This framework defines GASP, a Generalized Algorithm for Signed graph Partitioning, and allows us to explore many combinations of different linkage criteria and cannot-link constraints. We prove the equivalence of existing clustering methods to some of those combinations and introduce new algorithms for combinations that have not been studied before. We study both theoretical and empirical properties of these combinations and prove that some of these define an ultrametric on the graph. We conduct a systematic comparison of various instantiations of GASP on a large variety of both synthetic and existing signed clustering problems, in terms of accuracy but also efficiency and robustness to noise. Lastly, we show that some of the algorithms included in our framework, when combined with the predictions from a CNN model, result in a simple bottom-up instance segmentation pipeline. Going all the way from pixels to final segments with a simple procedure, we achieve state-of-the-art accuracy on the CREMI 2016 EM segmentation benchmark without requiring domain-specific superpixels.

count=11
* Enhancing Ki-67 Cell Segmentation with Dual U-Net Models: A Step Towards Uncertainty-Informed Active Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Anglada-Rotger_Enhancing_Ki-67_Cell_Segmentation_with_Dual_U-Net_Models_A_Step_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/papers/Anglada-Rotger_Enhancing_Ki-67_Cell_Segmentation_with_Dual_U-Net_Models_A_Step_CVPRW_2024_paper.pdf)]
    * Title: Enhancing Ki-67 Cell Segmentation with Dual U-Net Models: A Step Towards Uncertainty-Informed Active Learning
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: David Anglada-Rotger, Julia Sala, Ferran Marques, Philippe Salembier, Montse Pardàs
    * Abstract: The diagnosis and prognosis of breast cancer relies on histopathology image analysis where markers such as Ki-67 are increasingly important. The diagnosis using this marker is based on quantification of proliferation which implies counting of Ki-67 positive and negative tumoral cells excluding stromal cells. A common problem for automatic quantification of these images derives from overlapping and clustering of cells. We propose in this paper an automatic segmentation and classification system that overcomes this problem using two Convolutional Neural Networks (Dual U-Net) whose results are combined with a watershed algorithm. Taking into account that a major issue for the development of reliable neural networks is the availability of labeled databases we also introduce an approach for epistemic uncertainty estimation that can be used for active learning in instance segmentation applications. We use Monte Carlo Dropout within our networks to quantify the model's confidence across its predictions offering insights into areas of high uncertainty. Our results show how the postprocessed uncertainty maps can be used to refine ground truth annotations and to generate new labeled data with reduced annotation effort. To initialize the labeling and further reduce this effort we propose a tool for groundtruth generation which is based on candidate generation with maxtree. Candidates are filtered based on extracted features which can be adjusted for the specific image typology thereby facilitating precise model training and evaluation.

count=11
* Bots for Software-Assisted Analysis of Image-Based Transcriptomics
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Cicconet_Bots_for_Software-Assisted_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w1/Cicconet_Bots_for_Software-Assisted_ICCV_2017_paper.pdf)]
    * Title: Bots for Software-Assisted Analysis of Image-Based Transcriptomics
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Marcelo Cicconet, Daniel R. Hochbaum, David L. Richmond, Bernardo L. Sabatini
    * Abstract: We introduce software assistants -- bots -- for the task of analyzing image-based transcriptomic data. The key steps in this process are detecting nuclei, and counting associated puncta corresponding to labeled RNA. Our main release offers two algorithms for nuclei segmentation, and two for spot detection, to handle data of different complexities. For challenging nuclei segmentation cases, we enable the user to train a stacked Random Forest, which includes novel circularity features that leverage prior knowledge regarding nuclei shape for better instance segmentation. This machine learning model can be trained on a modern CPU-only computer, yet performs comparably with respect to a more hardware-demanding state-of-the-art deep learning approach, as demonstrated through experiments. While the primary motivation for the bots was image-based transcriptomics, we also demonstrate their applicability to the more general problem of scoring 'spots' in nuclei.

count=10
* CNN Based Yeast Cell Segmentation in Multi-Modal Fluorescent Microscopy Data
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/html/Aydin_CNN_Based_Yeast_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/papers/Aydin_CNN_Based_Yeast_CVPR_2017_paper.pdf)]
    * Title: CNN Based Yeast Cell Segmentation in Multi-Modal Fluorescent Microscopy Data
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Ali Selman Aydin, Abhinandan Dubey, Daniel Dovrat, Amir Aharoni, Roy Shilkrot
    * Abstract: We present a method for foreground segmentation of yeast cells in the presence of high-noise induced by intentional low illumination, where traditional approaches (e.g., threshold-based methods, specialized cell-segmentation methods) fail. To deal with these harsh conditions, we use a fully-convolutional semantic segmentation network based on the SegNet architecture. Our model is capable of segmenting patches extracted from yeast live-cell experiments with a mIOU score of 0.71 on unseen patches drawn from independent experiments. Further, we show that simultaneous multi-modal observations of bio-fluorescent markers can result in better segmentation performance than the DIC channel alone.

count=10
* TA-Net: Topology-Aware Network for Gland Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Wang_TA-Net_Topology-Aware_Network_for_Gland_Segmentation_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Wang_TA-Net_Topology-Aware_Network_for_Gland_Segmentation_WACV_2022_paper.pdf)]
    * Title: TA-Net: Topology-Aware Network for Gland Segmentation
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Haotian Wang, Min Xian, Aleksandar Vakanski
    * Abstract: Gland segmentation is a critical step to quantitatively assess the morphology of glands in histopathology image analysis. However, it is challenging to separate densely clustered glands accurately. Existing deep learning-based approaches attempted to use contour-based techniques to alleviate this issue but only achieved limited success. To address this challenge, we propose a novel topology-aware network (TA-Net) to accurately separate densely clustered and severely deformed glands. The proposed TA-Net has a multitask learning architecture and enhances the generalization of gland segmentation by learning shared representation from two tasks: instance segmentation and gland topology estimation. The proposed topology loss computes gland topology using gland skeletons and markers. It drives the network to generate segmentation results that comply with the true gland topology. We validate the proposed approach on the GlaS and CRAG datasets using three quantitative metrics, F1-score, object-level Dice coefficient, and object-level Hausdorff distance. Extensive experiments demonstrate that TA-Net achieves state-of-the-art performance on the two datasets. TA-Net outperforms other approaches in the presence of densely clustered glands.

count=9
* Building Detection From Satellite Imagery Using a Composite Loss Function
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Golovanov_Building_Detection_From_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Golovanov_Building_Detection_From_CVPR_2018_paper.pdf)]
    * Title: Building Detection From Satellite Imagery Using a Composite Loss Function
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Sergey Golovanov, Rauf Kurbanov, Aleksey Artamonov, Alex Davydow, Sergey Nikolenko
    * Abstract: In this paper, we present a LinkNet-based architecture with SE-ResNeXt-50 encoder and a novel training strategy that strongly relies on image preprocessing and incorporating distorted network outputs. The architecture combines a pre-trained convolutional encoder and a symmetric expanding path that enables precise localization. We show that such a network can be trained on plain RGB images with a composite loss function and achieves competitive results on the DeepGlobe challenge on building extraction from satellite images.

count=9
* Multi-Stage Multi-Recursive-Input Fully Convolutional Networks for Neuronal Boundary Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Shen_Multi-Stage_Multi-Recursive-Input_Fully_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Shen_Multi-Stage_Multi-Recursive-Input_Fully_ICCV_2017_paper.pdf)]
    * Title: Multi-Stage Multi-Recursive-Input Fully Convolutional Networks for Neuronal Boundary Detection
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Wei Shen, Bin Wang, Yuan Jiang, Yan Wang, Alan Yuille
    * Abstract: In the field of connectomics, neuroscientists seek to identify cortical connectivity comprehensively. Neuronal boundary detection from the Electron Microscopy (EM) images is often done to assist the automatic reconstruction of neuronal circuit. But the segmentation of EM images is a challenging problem, as it requires the detector to be able to detect both filament-like thin and blob-like thick membrane, while suppressing the ambiguous intracellular structure. In this paper, we propose multi-stage multi-recursiveinput fully convolutional networks to address this problem. The multiple recursive inputs for one stage, i.e., the multiple side outputs with different receptive field sizes learned from the lower stage, provide multi-scale contextual boundary information for the consecutive learning. This design is biologically-plausible, as it likes a human visual system to compare different possible segmentation solutions to address the ambiguous boundary issue. Our multi-stage networks are trained end-to-end. It achieves promising results on two public available EM segmentation datasets, the mouse piriform cortex dataset and the ISBI 2012 EM dataset.

count=8
* Discrete-Continuous Gradient Orientation Estimation for Faster Image Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Donoser_Discrete-Continuous_Gradient_Orientation_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Donoser_Discrete-Continuous_Gradient_Orientation_2014_CVPR_paper.pdf)]
    * Title: Discrete-Continuous Gradient Orientation Estimation for Faster Image Segmentation
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Michael Donoser, Dieter Schmalstieg
    * Abstract: The state-of-the-art in image segmentation builds hierarchical segmentation structures based on analyzing local feature cues in spectral settings. Due to their impressive performance, such segmentation approaches have become building blocks in many computer vision applications. Nevertheless, the main bottlenecks are still the computationally demanding processes of local feature processing and spectral analysis. In this paper, we demonstrate that based on a discrete-continuous optimization of oriented gradient signals, we are able to provide segmentation performance competitive to state-of-the-art on BSDS 500 (even without any spectral analysis) while reducing computation time by a factor of 40 and memory demands by a factor of 10.

count=8
* Fusion Moves for Correlation Clustering
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Beier_Fusion_Moves_for_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Beier_Fusion_Moves_for_2015_CVPR_paper.pdf)]
    * Title: Fusion Moves for Correlation Clustering
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Thorsten Beier, Fred A. Hamprecht, Jorg H. Kappes
    * Abstract: Correlation clustering, or multicut partitioning, is widely used in image segmentation for partitioning an undirected graph or image with positive and negative edge weights such that the sum of cut edge weights is minimized. Due to its NP-hardness, exact solvers do not scale and approximative solvers often give unsatisfactory results. We investigate scalable methods for correlation clustering. To this end we define fusion moves for the correlation clustering problem. Our algorithm iteratively fuses the current and a proposed partitioning which monotonously improves the partitioning and maintains a valid partitioning at all times. Furthermore, it scales to larger datasets, gives near optimal solutions, and at the same time shows a good anytime performance.

count=8
* Biologically-Constrained Graphs for Global Connectomics Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Matejek_Biologically-Constrained_Graphs_for_Global_Connectomics_Reconstruction_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Matejek_Biologically-Constrained_Graphs_for_Global_Connectomics_Reconstruction_CVPR_2019_paper.pdf)]
    * Title: Biologically-Constrained Graphs for Global Connectomics Reconstruction
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Brian Matejek,  Daniel Haehn,  Haidong Zhu,  Donglai Wei,  Toufiq Parag,  Hanspeter Pfister
    * Abstract: Most current state-of-the-art connectome reconstruction pipelines have two major steps: initial pixel-based segmentation with affinity prediction and watershed transform, and refined segmentation by merging over-segmented regions. These methods rely only on local context and are typically agnostic to the underlying biology. Since a few merge errors can lead to several incorrectly merged neuronal processes, these algorithms are currently tuned towards over-segmentation producing an overburden of costly proofreading. We propose a third step for connectomics reconstruction pipelines to refine an over-segmentation using both local and global context with an emphasis on adhering to the underlying biology. We first extract a graph from an input segmentation where nodes correspond to segment labels and edges indicate potential split errors in the over-segmentation. In order to increase throughput and allow for large-scale reconstruction, we employ biologically inspired geometric constraints based on neuron morphology to reduce the number of nodes and edges. Next, two neural networks learn these neuronal shapes to further aid the graph construction process. Lastly, we reformulate the region merging problem as a graph partitioning one to leverage global context. We demonstrate the performance of our approach on four real-world connectomics datasets with an average variation of information improvement of 21.3%.

count=7
* Guided Proofreading of Automatic Segmentations for Connectomics
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Haehn_Guided_Proofreading_of_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Haehn_Guided_Proofreading_of_CVPR_2018_paper.pdf)]
    * Title: Guided Proofreading of Automatic Segmentations for Connectomics
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Daniel Haehn, Verena Kaynig, James Tompkin, Jeff W. Lichtman, Hanspeter Pfister
    * Abstract: Automatic cell image segmentation methods in connectomics produce merge and split errors, which require correction through proofreading. Previous research has identified the visual search for these errors as the bottleneck in interactive proofreading. To aid error correction, we develop two classifiers that automatically recommend candidate merges and splits to the user. These classifiers use a convolutional neural network (CNN) that has been trained with errors in automatic segmentations against expert-labeled ground truth. Our classifiers detect potentially-erroneous regions by considering a large context region around a segmentation boundary. Corrections can then be performed by a user with yes/no decisions, which reduces variation of information 7.5x faster than previous proofreading methods. We also present a fully-automatic mode that uses a probability threshold to make merge/split decisions. Extensive experiments using the automatic approach and comparing performance of novice and expert users demonstrate that our method performs favorably against state-of-the-art proofreading methods on different connectomics datasets.

count=7
* Cross-Classification Clustering: An Efficient Multi-Object Tracking Technique for 3-D Instance Segmentation in Connectomics
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Meirovitch_Cross-Classification_Clustering_An_Efficient_Multi-Object_Tracking_Technique_for_3-D_Instance_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Meirovitch_Cross-Classification_Clustering_An_Efficient_Multi-Object_Tracking_Technique_for_3-D_Instance_CVPR_2019_paper.pdf)]
    * Title: Cross-Classification Clustering: An Efficient Multi-Object Tracking Technique for 3-D Instance Segmentation in Connectomics
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yaron Meirovitch,  Lu Mi,  Hayk Saribekyan,  Alexander Matveev,  David Rolnick,  Nir Shavit
    * Abstract: Pixel-accurate tracking of objects is a key element in many computer vision applications, often solved by iterated individual object tracking or instance segmentation followed by object matching. Here we introduce cross-classification clustering (3C), a technique that simultaneously tracks complex, interrelated objects in an image stack. The key idea in cross-classification is to efficiently turn a clustering problem into a classification problem by running a logarithmic number of independent classifications per image, letting the cross-labeling of these classifications uniquely classify each pixel to the object labels. We apply the 3C mechanism to achieve state-of-the-art accuracy in connectomics -- the nanoscale mapping of neural tissue from electron microscopy volumes. Our reconstruction system increases scalability by an order of magnitude over existing single-object tracking methods (such as flood-filling networks). This scalability is important for the deployment of connectomics pipelines, since currently the best performing techniques require computing infrastructures that are beyond the reach of most laboratories. Our algorithm may offer benefits in other domains that require pixel-accurate tracking of multiple objects, such as segmentation of videos and medical imagery.

count=7
* Large Scale High-Resolution Land Cover Mapping With Multi-Resolution Data
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Robinson_Large_Scale_High-Resolution_Land_Cover_Mapping_With_Multi-Resolution_Data_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Robinson_Large_Scale_High-Resolution_Land_Cover_Mapping_With_Multi-Resolution_Data_CVPR_2019_paper.pdf)]
    * Title: Large Scale High-Resolution Land Cover Mapping With Multi-Resolution Data
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Caleb Robinson,  Le Hou,  Kolya Malkin,  Rachel Soobitsky,  Jacob Czawlytko,  Bistra Dilkina,  Nebojsa Jojic
    * Abstract: In this paper we propose multi-resolution data fusion methods for deep learning-based high-resolution land cover mapping from aerial imagery. The land cover mapping problem, at country-level scales, is challenging for common deep learning methods due to the scarcity of high-resolution labels, as well as variation in geography and quality of input images. On the other hand, multiple satellite imagery and low-resolution ground truth label sources are widely available, and can be used to improve model training efforts. Our methods include: introducing low-resolution satellite data to smooth quality differences in high-resolution input, exploiting low-resolution labels with a dual loss function, and pairing scarce high-resolution labels with inputs from several points in time. We train models that are able to generalize from a portion of the Northeast United States, where we have high-resolution land cover labels, to the rest of the US. With these models, we produce the first high-resolution (1-meter) land cover map of the contiguous US, consisting of over 8 trillion pixels. We demonstrate the robustness and potential applications of this data in a case study with domain experts and develop a web application to share our results. This work is practically useful, and can be applied to other locations over the earth as high-resolution imagery becomes more widely available even as high-resolution labeled land cover data remains sparse.

count=7
* Mudslide: A Universal Nuclear Instance Segmentation Method
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Mudslide_A_Universal_Nuclear_Instance_Segmentation_Method_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Mudslide_A_Universal_Nuclear_Instance_Segmentation_Method_CVPR_2024_paper.pdf)]
    * Title: Mudslide: A Universal Nuclear Instance Segmentation Method
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jun Wang
    * Abstract: Nuclear instance segmentation has played a critical role in pathology image analysis. The main challenges arise from the difficulty in accurately segmenting densely overlapping instances and the high cost of precise mask-level annotations. Existing fully-supervised nuclear instance segmentation methods such as boundary-based methods struggle to capture differences between overlapping instances and thus fail in densely distributed blurry regions. They also face challenges transitioning to point supervision where annotations are simple and effective. Inspired by natural mudslides we propose a universal method called Mudslide that uses simple representations to characterize differences between different instances and can easily be extended from fully-supervised to point-supervised. oncretely we introduce a collapse field and leverage it to construct a force map and initial boundary enabling a distinctive representation for each instance. Each pixel is assigned a collapse force with distinct directions between adjacent instances. Starting from the initial boundary Mudslide executes a pixel-by-pixel collapse along various force directions. Pixels that collapse into the same region are considered as one instance concurrently accounting for both inter-instance distinctions and intra-instance coherence. Experiments on public datasets show superior performance in both fully-supervised and point-supervised tasks.

count=7
* Weakly Supervised Cell-Instance Segmentation With Two Types of Weak Labels by Single Instance Pasting
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Nishimura_Weakly_Supervised_Cell-Instance_Segmentation_With_Two_Types_of_Weak_Labels_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Nishimura_Weakly_Supervised_Cell-Instance_Segmentation_With_Two_Types_of_Weak_Labels_WACV_2023_paper.pdf)]
    * Title: Weakly Supervised Cell-Instance Segmentation With Two Types of Weak Labels by Single Instance Pasting
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Kazuya Nishimura, Ryoma Bise
    * Abstract: Cell instance segmentation that recognizes each cell boundary is an important task in cell image analysis. While deep learning-based methods have shown promising performances with a certain amount of training data, most of them require full annotations that show the boundary of each cell. Generating the annotation for cell segmentation is time-consuming and human labor. To reduce the annotation cost, we propose a weakly supervised segmentation method using two types of weak labels (one for cell type and one for nuclei position). Unlike general images, these two labels are easily obtained in phase-contrast images. The intercellular boundary, which is necessary for cell instance segmentation, cannot be directly obtained from these two weak labels, so to generate the boundary information, we propose a single instance pasting based on the copy-and-paste technique. First, we locate single-cell regions by counting cells and store them in a pool. Then, we generate the intercellular boundary by pasting the stored single-cell regions to the original image. Finally, we train a boundary estimation network with the generated labels and perform instance segmentation with the network. Our evaluation on a public dataset demonstrated that the proposed method achieves the best performance among the several weakly supervised methods we compared.

count=7
* Combinatorial Energy Learning for Image Segmentation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2016/hash/31857b449c407203749ae32dd0e7d64a-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2016/file/31857b449c407203749ae32dd0e7d64a-Paper.pdf)]
    * Title: Combinatorial Energy Learning for Image Segmentation
    * Publisher: NeurIPS
    * Publication Date: `2016`
    * Authors: Jeremy B. Maitin-Shepard, Viren Jain, Michal Januszewski, Peter Li, Pieter Abbeel
    * Abstract: We introduce a new machine learning approach for image segmentation that uses a neural network to model the conditional energy of a segmentation given an image. Our approach, combinatorial energy learning for image segmentation (CELIS) places a particular emphasis on modeling the inherent combinatorial nature of dense image segmentation problems. We propose efficient algorithms for learning deep neural networks to model the energy function, and for local optimization of this energy in the space of supervoxel agglomerations. We extensively evaluate our method on a publicly available 3-D microscopy dataset with 25 billion voxels of ground truth data. On an 11 billion voxel test set, we find that our method improves volumetric reconstruction accuracy by more than 20% as compared to two state-of-the-art baseline methods: graph-based segmentation of the output of a 3-D convolutional neural network trained to predict boundaries, as well as a random forest classifier trained to agglomerate supervoxels that were generated by a 3-D convolutional neural network.

count=6
* Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Barsellotti_Training-Free_Open-Vocabulary_Segmentation_with_Offline_Diffusion-Augmented_Prototype_Generation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Barsellotti_Training-Free_Open-Vocabulary_Segmentation_with_Offline_Diffusion-Augmented_Prototype_Generation_CVPR_2024_paper.pdf)]
    * Title: Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Luca Barsellotti, Roberto Amoroso, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara
    * Abstract: Open-vocabulary semantic segmentation aims at segmenting arbitrary categories expressed in textual form. Previous works have trained over large amounts of image-caption pairs to enforce pixel-level multimodal alignments. However captions provide global information about the semantics of a given image but lack direct localization of individual concepts. Further training on large-scale datasets inevitably brings significant computational costs. In this paper we propose FreeDA a training-free diffusion-augmented method for open-vocabulary semantic segmentation which leverages the ability of diffusion models to visually localize generated concepts and local-global similarities to match class-agnostic regions with semantic classes. Our approach involves an offline stage in which textual-visual reference embeddings are collected starting from a large set of captions and leveraging visual and semantic contexts. At test time these are queried to support the visual matching process which is carried out by jointly considering class-agnostic regions and global semantic similarities. Extensive analyses demonstrate that FreeDA achieves state-of-the-art performance on five datasets surpassing previous methods by more than 7.0 average points in terms of mIoU and without requiring any training. Our source code is available at https://aimagelab.github.io/freeda/.

count=6
* Low-power Continuous Remote Behavioral Localization with Event Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Hamann_Low-power_Continuous_Remote_Behavioral_Localization_with_Event_Cameras_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Hamann_Low-power_Continuous_Remote_Behavioral_Localization_with_Event_Cameras_CVPR_2024_paper.pdf)]
    * Title: Low-power Continuous Remote Behavioral Localization with Event Cameras
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Friedhelm Hamann, Suman Ghosh, Ignacio Juarez Martinez, Tom Hart, Alex Kacelnik, Guillermo Gallego
    * Abstract: Researchers in natural science need reliable methods for quantifying animal behavior. Recently numerous computer vision methods emerged to automate the process. However observing wild species at remote locations remains a challenging task due to difficult lighting conditions and constraints on power supply and data storage. Event cameras offer unique advantages for battery-dependent remote monitoring due to their low power consumption and high dynamic range capabilities. We use this novel sensor to quantify a behavior in Chinstrap penguins called ecstatic display. We formulate the problem as a temporal action detection task determining the start and end times of the behavior. For this purpose we recorded a colony of breeding penguins in Antarctica for several weeks and labeled event data on 16 nests. The developed method consists of a generator of candidate time intervals (proposals) and a classifier of the actions within them. The experiments show that the event cameras' natural response to motion is effective for continuous behavior monitoring and detection reaching a mean average precision (mAP) of 58% (which increases to 63% in good weather conditions). The results also demonstrate the robustness against various lighting conditions contained in the challenging dataset. The low-power capabilities of the event camera allow it to record significantly longer than with a conventional camera. This work pioneers the use of event cameras for remote wildlife observation opening new interdisciplinary opportunities. https:// tub-rip.github.io/ eventpenguins/

count=6
* Multi-Object Portion Tracking in 4D Fluorescence Microscopy Imagery With Deep Feature Maps
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/CVMI/Jiao_Multi-Object_Portion_Tracking_in_4D_Fluorescence_Microscopy_Imagery_With_Deep_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVMI/Jiao_Multi-Object_Portion_Tracking_in_4D_Fluorescence_Microscopy_Imagery_With_Deep_CVPRW_2019_paper.pdf)]
    * Title: Multi-Object Portion Tracking in 4D Fluorescence Microscopy Imagery With Deep Feature Maps
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yang Jiao,  Mo Weng,  Mei Yang
    * Abstract: 3D fluorescence microscopy of living organisms has increasingly become an essential and powerful tool in biomedical research and diagnosis. Exploding amount of imaging data has been collected whereas efficient and effective computational tools to extract information from them are still lagged behind. This largely is due to the challenges in analyzing biological data. Interesting biological structures are not only small but often are morphologically irregular and highly dynamic. Tracking cells in live organisms has been studied for years as a sophisticated mission in bioinformatics. However, existing tracking methods for cells are not effective in tracking subcellular structures, such as protein complexes, which feature in continuous morphological changes, such as split and merge, in addition to fast migration and complex motion. In this paper, we first define the problem of multi-object portion tracking to model protein object tracking process. A multi-object tracking method with portion matching is proposed based on 3D segmentation results. The proposed method distills deep feature maps from deep networks, then recognizes and matches objects' portions using extended search. Experimental results confirm that the proposed method achieves 2.98% higher on consistent tracking accuracy and 35.48% higher on event identification accuracy.

count=6
* Online Neural Cell Tracking Using Blob-Seed Segmentation and Optical Flow
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/CVMI/Yi_Online_Neural_Cell_Tracking_Using_Blob-Seed_Segmentation_and_Optical_Flow_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVMI/Yi_Online_Neural_Cell_Tracking_Using_Blob-Seed_Segmentation_and_Optical_Flow_CVPRW_2019_paper.pdf)]
    * Title: Online Neural Cell Tracking Using Blob-Seed Segmentation and Optical Flow
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jingru Yi,  Pengxiang Wu,  Qiaoying Huang,  Hui Qu,  Daniel J. Hoeppner,  Dimitris N. Metaxas
    * Abstract: Existing neural cell tracking methods generally use the morphology cell features for data association. However, these features are limited to the quality of cell segmentation and are prone to errors for mitosis determination. To overcome these issues, in this work we propose an online multi-object tracking method that leverages both cell appearance and motion features for data association. In particular, we propose a supervised blob-seed network (BSNet) to predict the cell appearance features and an unsupervised optical flow network (UnFlowNet) for capturing the cell motions. The data association is then solved using the Hungarian algorithm. Experimental evaluation shows that our approach achieves better performance than existing neural cell tracking methods.

count=6
* Edge-Weighted Centroid Voronoi Tessellation with Propagation of Consistency Constraint for 3D Grain Segmentation in Microscopic Superalloy Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/html/Zhou_Edge-Weighted_Centroid_Voronoi_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/papers/Zhou_Edge-Weighted_Centroid_Voronoi_2014_CVPR_paper.pdf)]
    * Title: Edge-Weighted Centroid Voronoi Tessellation with Propagation of Consistency Constraint for 3D Grain Segmentation in Microscopic Superalloy Images
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Youjie Zhou, Lili Ju, Yu Cao, Jarrell Waggoner, Yuewei Lin, Jeff Simmons, Song Wang
    * Abstract: 3D microstructures are important for material scientists to analyze physical properties of materials. While such microstructures are too small to be directly visible to human vision, modern microscopic and serial-sectioning techniques can provide their high-resolution 3D images in the form of a sequence of 2D image slices. In this paper, we propose an algorithm based on the Edge-Weighted Centroid Voronoi Tessellation which uses propagation of the inter-slice consistency constraint. It can segment a 3D superalloy image, slice by slice, to obtain the underlying grain microstructures. With the propagation of the consistency constraint, the proposed method can automatically match grain segments between slices. On each of the 2D image slices, stable structures identified from the previous slice can be well-preserved, with further refinement by clustering the pixels in terms of both intensity and spatial information. We tested the proposed algorithm on a 3D superalloy image consisting of 170 2D slices. Performance is evaluated against manually annotated ground-truth segmentation. The results show that the proposed method outperforms several state-of-the-art 2D, 3D, and propagation-based segmentation methods in terms of both segmentation accuracy and running time.

count=6
* Reinforcement Learning for Instance Segmentation with high-Level Priors
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/BIC/html/Hilt_Reinforcement_Learning_for_Instance_Segmentation_with_high-Level_Priors_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/BIC/papers/Hilt_Reinforcement_Learning_for_Instance_Segmentation_with_high-Level_Priors_ICCVW_2023_paper.pdf)]
    * Title: Reinforcement Learning for Instance Segmentation with high-Level Priors
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Paul Hilt, Maedeh Zarvandi, Edgar Kaziakhmedov, Sourabh Bhide, Maria Leptin, Constantin Pape, Anna Kreshuk
    * Abstract: Instance segmentation is a fundamental computer vision problem which remains challenging despite impressive recent advances due to deep learning-based methods. Given sufficient training data, fully supervised methods can yield excellent performance, but annotation of groundtruth remains a major bottleneck, especially for biomedical applications where it has to be performed by domain experts. The amount of labels required can be drastically reduced by using rules derived from prior knowledge to guide the segmentation. However, these rules are in general not differentiable and thus cannot be used with existing methods. Here, we revoke this requirement by using stateless actor critic reinforcement learning, which enables non-differentiable rewards. We formulate the instance segmentation problem as graph partitioning and the actor critic predicts the edge weights driven by the rewards, which are based on the conformity of segmented instances to high-level priors on object shape, position or size. The experiments on toy and real data demonstrate that a good set of priors is sufficient to reach excellent performance without any direct object-level supervision.

count=6
* Behind the Scenes: What Moving Targets Reveal about Static Scene Geometry
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/html/Taylor_Behind_the_Scenes_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/papers/Taylor_Behind_the_Scenes_2013_ICCV_paper.pdf)]
    * Title: Behind the Scenes: What Moving Targets Reveal about Static Scene Geometry
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Geoffrey Taylor, Fei Mai
    * Abstract: Reasoning about 3D scene structure is an important component of visual scene understanding. Often, reasoning proceeds from low-level cues without resorting to full 3D reconstruction. However, existing geometric cues may require multiple viewpoints, supervised training, constraints on scene structure or information from auxiliary sensors. To address these limitations, this paper demonstrates how geometric context for a single static camera can be recovered from the location and shape of moving foreground targets. In particular, we propose methods to compute the likelihood of a static occlusion boundary and floor region at each pixel. Importantly, these cues do not require supervised training, or prior knowledge of camera geometry or scene structure. Finally, we show how the proposed geometric cues can be used to infer an ordinal depth map and demonstrate its use in compositing with correct occlusion handling.

count=6
* Recursive Training of 2D-3D Convolutional Networks for Neuronal Boundary Prediction
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2015/hash/39dcaf7a053dc372fbc391d4e6b5d693-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2015/file/39dcaf7a053dc372fbc391d4e6b5d693-Paper.pdf)]
    * Title: Recursive Training of 2D-3D Convolutional Networks for Neuronal Boundary Prediction
    * Publisher: NeurIPS
    * Publication Date: `2015`
    * Authors: Kisuk Lee, Aleksandar Zlateski, Vishwanathan Ashwin, H. Sebastian Seung
    * Abstract: Efforts to automate the reconstruction of neural circuits from 3D electron microscopic (EM) brain images are critical for the field of connectomics. An important computation for reconstruction is the detection of neuronal boundaries. Images acquired by serial section EM, a leading 3D EM technique, are highly anisotropic, with inferior quality along the third dimension. For such images, the 2D max-pooling convolutional network has set the standard for performance at boundary detection. Here we achieve a substantial gain in accuracy through three innovations. Following the trend towards deeper networks for object recognition, we use a much deeper network than previously employed for boundary detection. Second, we incorporate 3D as well as 2D filters, to enable computations that use 3D context. Finally, we adopt a recursively trained architecture in which a first network generates a preliminary boundary map that is provided as input along with the original image to a second network that generates a final boundary map. Backpropagation training is accelerated by ZNN, a new implementation of 3D convolutional networks that uses multicore CPU parallelism for speed. Our hybrid 2D-3D architecture could be more generally applicable to other types of anisotropic 3D images, including video, and our recursive framework for any image labeling problem.

count=5
* Towards Fast and Accurate Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Taylor_Towards_Fast_and_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Taylor_Towards_Fast_and_2013_CVPR_paper.pdf)]
    * Title: Towards Fast and Accurate Segmentation
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Camillo J. Taylor
    * Abstract: In this paper we explore approaches to accelerating segmentation and edge detection algorithms based on the gPb framework. The paper characterizes the performance of a simple but effective edge detection scheme which can be computed rapidly and offers performance that is competitive with the pB detector. The paper also describes an approach for computing a reduced order normalized cut that captures the essential features of the original problem but can be computed in less than half a second on a standard computing platform.

count=5
* TernausNetV2: Fully Convolutional Network for Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Iglovikov_TernausNetV2_Fully_Convolutional_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Iglovikov_TernausNetV2_Fully_Convolutional_CVPR_2018_paper.pdf)]
    * Title: TernausNetV2: Fully Convolutional Network for Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Vladimir Iglovikov, Selim Seferbekov, Alexander Buslaev, Alexey Shvets
    * Abstract: The most common approaches to instance segmentation are complex and use two-stage networks with object proposals, conditional random-fields, template matching or recurrent neural networks. In this work we present TernausNetV2 - a simple fully convolutional network that allows extracting objects from a high-resolution satellite imagery on an instance level. The network has popular encoder-decoder type of architecture with skip connections but has a few essential modifications that allows using for semantic as well as for instance segmentation tasks. This approach is universal and allows to extend any network that has been successfully applied for semantic segmentation to perform instance segmentation task. In addition, we generalize network encoder that was pre-trained for RGB images to use additional input channels. It makes possible to use transfer learning from visual to a wider spectral range. For DeepGlobe-CVPR 2018 building detection sub-challenge, based on public leaderboard score, our approach shows superior performance in comparison to other methods.

count=5
* Enhancing Generic Segmentation With Learned Region Representations
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Isaacs_Enhancing_Generic_Segmentation_With_Learned_Region_Representations_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Isaacs_Enhancing_Generic_Segmentation_With_Learned_Region_Representations_CVPR_2020_paper.pdf)]
    * Title: Enhancing Generic Segmentation With Learned Region Representations
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Or Isaacs,  Oran Shayer,  Michael Lindenbaum
    * Abstract: Deep learning approaches to generic (non-semantic) segmentation have so far been indirect and relied on edge detection. This is in contrast to semantic segmentation, where DNNs are applied directly. We propose an alternative approach called Deep Generic Segmentation (DGS) and try to follow the path used for semantic segmentation. Our main contribution is a new method for learning a pixel-wise representation that reflects segment relatedness. This representation is combined with a CRF to yield the segmentation algorithm. We show that we are able to learn meaningful representations that improve segmentation quality and that the representations themselves achieve state-of-the-art segment similarity scores. The segmentation results are competitive and promising.

count=5
* VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_VGSE_Visually-Grounded_Semantic_Embeddings_for_Zero-Shot_Learning_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_VGSE_Visually-Grounded_Semantic_Embeddings_for_Zero-Shot_Learning_CVPR_2022_paper.pdf)]
    * Title: VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Wenjia Xu, Yongqin Xian, Jiuniu Wang, Bernt Schiele, Zeynep Akata
    * Abstract: Human-annotated attributes serve as powerful semantic embeddings in zero-shot learning. However, their annotation process is labor-intensive and needs expert supervision. Current unsupervised semantic embeddings, i.e., word embeddings, enable knowledge transfer between classes. However, word embeddings do not always reflect visual similarities and result in inferior zero-shot performance. We propose to discover semantic embeddings containing discriminative visual properties for zero-shot learning, without requiring any human annotation. Our model visually divides a set of images from seen classes into clusters of local image regions according to their visual similarity, and further imposes their class discrimination and semantic relatedness. To associate these clusters with previously unseen classes, we use external knowledge, e.g., word embeddings and propose a novel class relation discovery module. Through quantitative and qualitative evaluation, we demonstrate that our model discovers semantic embeddings that model the visual properties of both seen and unseen classes. Furthermore, we demonstrate on three benchmarks that our visually-grounded semantic embeddings further improve performance over word embeddings across various ZSL models by a large margin. Code is available at https://github.com/wenjiaXu/VGSE

count=5
* Hierarchical Histogram Threshold Segmentation - Auto-terminating High-detail Oversegmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Chang_Hierarchical_Histogram_Threshold_Segmentation_-_Auto-terminating_High-detail_Oversegmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Chang_Hierarchical_Histogram_Threshold_Segmentation_-_Auto-terminating_High-detail_Oversegmentation_CVPR_2024_paper.pdf)]
    * Title: Hierarchical Histogram Threshold Segmentation - Auto-terminating High-detail Oversegmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Thomas V. Chang, Simon Seibt, Bartosz von Rymon Lipinski
    * Abstract: Superpixels play a crucial role in image processing by partitioning an image into clusters of pixels with similar visual attributes. This facilitates subsequent image processing tasks offering computational advantages over the manipulation of individual pixels. While numerous oversegmentation techniques have emerged in recent years many rely on predefined initialization and termination criteria. In this paper a novel top-down superpixel segmentation algorithm called Hierarchical Histogram Threshold Segmentation (HHTS) is introduced. It eliminates the need for initialization and implements auto-termination outperforming state-of-the-art methods w.r.t boundary recall. This is achieved by iteratively partitioning individual pixel segments into foreground and background and applying intensity thresholding across multiple color channels. The underlying iterative process constructs a superpixel hierarchy that adapts to local detail distributions until color information exhaustion. Experimental results demonstrate the superiority of the proposed approach in terms of boundary adherence while maintaining competitive runtime performance on the BSDS500 and NYUV2 datasets. Furthermore an application of HHTS in refining machine learning-based semantic segmentation masks produced by the Segment Anything Foundation Model (SAM) is presented.

count=5
* Superpixel Estimation for Hyperspectral Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/html/Massoudifar_Superpixel_Estimation_for_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W04/papers/Massoudifar_Superpixel_Estimation_for_2014_CVPR_paper.pdf)]
    * Title: Superpixel Estimation for Hyperspectral Imagery
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Pegah Massoudifar, Anand Rangarajan, Paul Gader
    * Abstract: In the past decade, there has been a growing need for machine learning and computer vision components (segmentation, classification) in the hyperspectral imaging domain. Due to the complexity and size of hyperspectral imagery and the enormous number of wavelength channels, the need for combining compact representations with image segmentation and superpixel estimation has emerged in this area. Here, we present an approach to superpixel estimation in hyperspectral images by adapting the well known UCM approach to hyperspectral volumes. This approach benefits from the channel information at each pixel of the hyperspectral image while obtaining a compact representation of the hyperspectral volume using principal component analysis. Our experimental evaluation demonstrates that the additional information of spectral channels will substantially improve superpixel estimation from a single "monochromatic" channel. Furthermore, superpixel estimation performed on the compact hyperspectral representation outperforms the same when executed on the entire volume.

count=4
* Spectral Graph Reduction for Efficient Image and Streaming Video Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Galasso_Spectral_Graph_Reduction_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Galasso_Spectral_Graph_Reduction_2014_CVPR_paper.pdf)]
    * Title: Spectral Graph Reduction for Efficient Image and Streaming Video Segmentation
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Fabio Galasso, Margret Keuper, Thomas Brox, Bernt Schiele
    * Abstract: Computational and memory costs restrict spectral techniques to rather small graphs, which is a serious limitation especially in video segmentation. In this paper, we propose the use of a reduced graph based on superpixels. In contrast to previous work, the reduced graph is reweighted such that the resulting segmentation is equivalent, under certain assumptions, to that of the full graph. We consider equivalence in terms of the normalized cut and of its spectral clustering relaxation. The proposed method reduces runtime and memory consumption and yields on par results in image and video segmentation. Further, it enables an efficient data representation and update for a new streaming video segmentation approach that also achieves state-of-the-art performance.

count=4
* Localization and Tracking in 4D Fluorescence Microscopy Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w44/html/Abousamra_Localization_and_Tracking_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w44/Abousamra_Localization_and_Tracking_CVPR_2018_paper.pdf)]
    * Title: Localization and Tracking in 4D Fluorescence Microscopy Imagery
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Shahira Abousamra, Shai Adar, Natalie Elia, Roy Shilkrot
    * Abstract: 3D fluorescence microscopy continues to pose challenging tasks with more experiments leading to identifying new physiological patterns in cells' life cycle and activity. It then falls on the hands of biologists to annotate this imagery which is laborious and time-consuming, especially with noisy images and hard to see and track patterns. Modeling of automation tasks that can handle depth-varying light conditions and noise, and other challenges inherent in 3D fluorescence microscopy often becomes complex and requires high processing power and memory. This paper presents an efficient methodology for the localization, classification, and tracking in fluorescence microscopy imagery by taking advantage of time sequential images in 4D data. We show the application of our proposed method on the challenging task of localizing and tracking microtubule fibers' bridge formation during the cell division of zebrafish embryos where we achieve 98% accuracy and 0.94 F1- score.

count=4
* Multilayer Encoder-Decoder Network for 3D Nuclear Segmentation in Spheroid Models of Human Mammary Epithelial Cell Lines
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w44/html/Khoshdeli_Multilayer_Encoder-Decoder_Network_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w44/Khoshdeli_Multilayer_Encoder-Decoder_Network_CVPR_2018_paper.pdf)]
    * Title: Multilayer Encoder-Decoder Network for 3D Nuclear Segmentation in Spheroid Models of Human Mammary Epithelial Cell Lines
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Mina Khoshdeli, Garrett Winkelmaier, Bahram Parvin
    * Abstract: Nuclear segmentation is an important step in quantitative profiling of colony organization in 3D cell culture models. However, complexities arise from technical variations and biological heterogeneities. We proposed a new 3D segmentation model based on convolutional neural networks for 3D nuclear segmentation, which overcome the complexities associated with non-uniform staining, aberrations in cellular morphologies, and cells being in different states. The uniqueness of the method originates from (i) volumetric operations to capture all the three-dimensional features, and (ii) the encoder-decoder architecture, which enables segmentation of the spheroid models in one forward pass. The method is validated with four human mammary epithelial cell (HMEC) lines--each with a unique genetic makeup. The performance of the proposed method is compared with the previous methods and is shown that the deep learning model has a superior pixel-based segmentation, and an F1-score of 0.95 is reported.

count=4
* Character Region Awareness for Text Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.pdf)]
    * Title: Character Region Awareness for Text Detection
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Youngmin Baek,  Bado Lee,  Dongyoon Han,  Sangdoo Yun,  Hwalsuk Lee
    * Abstract: Scene text detection methods based on neural networks have emerged recently and have shown promising results. Previous methods trained with rigid word-level bounding boxes exhibit limitations in representing the text region in an arbitrary shape. In this paper, we propose a new scene text detection method to effectively detect text area by exploring each character and affinity between characters. To overcome the lack of individual character level annotations, our proposed framework exploits both the given character-level annotations for synthetic images and the estimated character-level ground-truths for real images acquired by the learned interim model. In order to estimate affinity between characters, the network is trained with the newly proposed representation for affinity. Extensive experiments on six benchmarks, including the TotalText and CTW-1500 datasets which contain highly curved texts in natural images, demonstrate that our character-level text detection significantly outperforms the state-of-the-art detectors. According to the results, our proposed method guarantees high flexibility in detecting complicated scene text images, such as arbitrarily-oriented, curved, or deformed texts.

count=4
* Learning without Exact Guidance: Updating Large-scale High-resolution Land Cover Maps from Low-resolution Historical Labels
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Learning_without_Exact_Guidance_Updating_Large-scale_High-resolution_Land_Cover_Maps_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Learning_without_Exact_Guidance_Updating_Large-scale_High-resolution_Land_Cover_Maps_CVPR_2024_paper.pdf)]
    * Title: Learning without Exact Guidance: Updating Large-scale High-resolution Land Cover Maps from Low-resolution Historical Labels
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zhuohong Li, Wei He, Jiepan Li, Fangxiao Lu, Hongyan Zhang
    * Abstract: Large-scale high-resolution (HR) land-cover mapping is a vital task to survey the Earth's surface and resolve many challenges facing humanity. However it is still a non-trivial task hindered by complex ground details various landforms and the scarcity of accurate training labels over a wide-span geographic area. In this paper we propose an efficient weakly supervised framework (Paraformer) to guide large-scale HR land-cover mapping with easy-access historical land-cover data of low resolution (LR). Specifically existing land-cover mapping approaches reveal the dominance of CNNs in preserving local ground details but still suffer from insufficient global modeling in various landforms. Therefore we design a parallel CNN-Transformer feature extractor in Paraformer consisting of a downsampling-free CNN branch and a Transformer branch to jointly capture local and global contextual information. Besides facing the spatial mismatch of training data a pseudo-label-assisted training (PLAT) module is adopted to reasonably refine LR labels for weakly supervised semantic segmentation of HR images. Experiments on two large-scale datasets demonstrate the superiority of Paraformer over other state-of-the-art methods for automatically updating HR land-cover maps from LR historical labels.

count=4
* Contour Detection and Characterization for Asynchronous Event Sensors
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Barranco_Contour_Detection_and_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Barranco_Contour_Detection_and_ICCV_2015_paper.pdf)]
    * Title: Contour Detection and Characterization for Asynchronous Event Sensors
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Francisco Barranco, Ching L. Teo, Cornelia Fermuller, Yiannis Aloimonos
    * Abstract: The bio-inspired, asynchronous event-based dynamic vision sensor records temporal changes in the luminance of the scene at high temporal resolution. Since events are only triggered at significant luminance changes, most events occur at the boundary of objects and their parts. The detection of these contours is an essential step for further interpretation of the scene. This paper presents an approach to learn the location of contours and their border ownership using Structured Random Forests on event-based features that encode motion, timing, texture, and spatial orientations. The classifier integrates elegantly information over time by utilizing the classification results previously computed. Finally, the contour detection and boundary assignment are demonstrated in a layer-segmentation of the scene. Experimental results demonstrate good performance in boundary detection and segmentation.

count=4
* CellTranspose: Few-Shot Domain Adaptation for Cellular Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Keaton_CellTranspose_Few-Shot_Domain_Adaptation_for_Cellular_Instance_Segmentation_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Keaton_CellTranspose_Few-Shot_Domain_Adaptation_for_Cellular_Instance_Segmentation_WACV_2023_paper.pdf)]
    * Title: CellTranspose: Few-Shot Domain Adaptation for Cellular Instance Segmentation
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Matthew R. Keaton, Ram J. Zaveri, Gianfranco Doretto
    * Abstract: Automated cellular instance segmentation is a process utilized for accelerating biological research for the past two decades, and recent advancements have produced higher quality results with less effort from the biologist. Most current endeavors focus on completely cutting the researcher out of the picture by generating highly generalized models. However, these models invariably fail when faced with novel data, distributed differently than the ones used for training. Rather than approaching the problem with methods that presume the availability of large amounts of target data and computing power for retraining, in this work we address the even greater challenge of designing an approach that requires minimal amounts of new annotated data as well as training time. We do so by designing specialized contrastive losses that leverage the few annotated samples very efficiently. A large set of results show that 3 to 5 annotations lead to models with accuracy that: 1) significantly mitigate the covariate shift effects; 2) matches or surpasses other adaptation methods; 3) even approaches methods that have been fully retrained on the target distribution. The adaptation training is only a few minutes, paving a path towards a balance between model performance, computing requirements and expert-level annotation needs.

count=4
* Fast Optimal Transport through Sliced Generalized Wasserstein Geodesics
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/6f1346bac8b02f76a631400e2799b24b-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/6f1346bac8b02f76a631400e2799b24b-Paper-Conference.pdf)]
    * Title: Fast Optimal Transport through Sliced Generalized Wasserstein Geodesics
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Guillaume Mahey, Laetitia Chapel, Gilles Gasso, Clément Bonet, Nicolas Courty
    * Abstract: Wasserstein distance (WD) and the associated optimal transport plan have been proven useful in many applications where probability measures are at stake. In this paper, we propose a new proxy of the squared WD, coined $\textnormal{min-SWGG}$, that is based on the transport map induced by an optimal one-dimensional projection of the two input distributions. We draw connections between $\textnormal{min-SWGG}$, and Wasserstein generalized geodesics in which the pivot measure is supported on a line. We notably provide a new closed form for the exact Wasserstein distance in the particular case of one of the distributions supported on a line allowing us to derive a fast computational scheme that is amenable to gradient descent optimization. We show that $\textnormal{min-SWGG}$, is an upper bound of WD and that it has a complexity similar to as Sliced-Wasserstein, with the additional feature of providing an associated transport plan. We also investigate some theoretical properties such as metricity, weak convergence, computational and topological properties. Empirical evidences support the benefits of $\textnormal{min-SWGG}$, in various contexts, from gradient flows, shape matching and image colorization, among others.

count=3
* Perceptual Organization and Recognition of Indoor Scenes from RGB-D Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Gupta_Perceptual_Organization_and_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Gupta_Perceptual_Organization_and_2013_CVPR_paper.pdf)]
    * Title: Perceptual Organization and Recognition of Indoor Scenes from RGB-D Images
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Saurabh Gupta, Pablo Arbelaez, Jitendra Malik
    * Abstract: We address the problems of contour detection, bottomup grouping and semantic segmentation using RGB-D data. We focus on the challenging setting of cluttered indoor scenes, and evaluate our approach on the recently introduced NYU-Depth V2 (NYUD2) dataset [27]. We propose algorithms for object boundary detection and hierarchical segmentation that generalize the gP b ucm approach of [2] by making effective use of depth information. We show that our system can label each contour with its type (depth, normal or albedo). We also propose a generic method for long-range amodal completion of surfaces and show its effectiveness in grouping. We then turn to the problem of semantic segmentation and propose a simple approach that classifies superpixels into the 40 dominant object categories in NYUD2. We use both generic and class-specific features to encode the appearance and geometry of objects. We also show how our approach can be used for scene classification, and how this contextual information in turn improves object recognition. In all of these tasks, we report significant improvements over the state-of-the-art.

count=3
* Moral Lineage Tracing
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Jug_Moral_Lineage_Tracing_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Jug_Moral_Lineage_Tracing_CVPR_2016_paper.pdf)]
    * Title: Moral Lineage Tracing
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Florian Jug, Evgeny Levinkov, Corinna Blasse, Eugene W. Myers, Bjoern Andres
    * Abstract: Lineage tracing, the tracking of living cells as they move and divide, is a central problem in biological image analysis. Solutions, called lineage forests, are key to understanding how the structure of multicellular organisms emerges. We propose an integer linear program (ILP) whose feasible solutions define, for every image in a sequence, a decomposition into cells (segmentation) and, across images, a lineage forest of cells (tracing). In this ILP, path-cut inequalities enforce the morality of lineages, i.e., the constraint that cells do not merge. To find feasible solutions of this NP-hard problem, with certified bounds to the global optimum, we define efficient separation procedures and apply these as part of a branch-and-cut algorithm. To show the effectiveness of this approach, we analyze feasible solutions for real microscopy data in terms of bounds and run-time, and by their weighted edit distance to lineage forests traced by humans.

count=3
* Unsupervised Segmentation of Cervical Cell Images Using Gaussian Mixture Model
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w27/html/Ragothaman_Unsupervised_Segmentation_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w27/papers/Ragothaman_Unsupervised_Segmentation_of_CVPR_2016_paper.pdf)]
    * Title: Unsupervised Segmentation of Cervical Cell Images Using Gaussian Mixture Model
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Srikanth Ragothaman, Sridharakumar Narasimhan, Madivala G. Basavaraj, Rajan Dewar
    * Abstract: Cervical cancer is one of the leading causes of cancer death in women. Screening at early stages using the popular Pap smear test has been demonstrated to reduce fatalities significantly. Cost effective, automated screening methods can significantly improve the adoption of these tests worldwide. Automated screening involves image analysis of cervical cells. Gaussian Mixture Models (GMM) are widely used in image processing for segmentation which is a crucial step in image analysis. In our proposed method, GMM is implemented to segment cell regions to identify cellular features such as nucleus, cytoplasm while addressing shortcomings of existing methods. This method is combined with shape based identification of nucleus to increase the accuracy of nucleus segmentation. This enables the algorithm to accurately trace the cells and nucleus contours from the pap smear images that contain cell clusters. The method also accounts for inconsistent staining, if any. The results that are presented shows that our proposed method performs well even in challenging conditions.

count=3
* Superpixels and Polygons Using Simple Non-Iterative Clustering
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Achanta_Superpixels_and_Polygons_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Achanta_Superpixels_and_Polygons_CVPR_2017_paper.pdf)]
    * Title: Superpixels and Polygons Using Simple Non-Iterative Clustering
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Radhakrishna Achanta, Sabine Susstrunk
    * Abstract: We present an improved version of the Simple Linear Iterative Clustering (SLIC) superpixel segmentation. Unlike SLIC, our algorithm is non-iterative, enforces connectivity from the start, requires lesser memory, and is faster. Relying on the superpixel boundaries obtained using our algorithm, we also present a polygonal partitioning algorithm. We demonstrate that our superpixels as well as the polygonal partitioning are superior to the respective state-of-the-art algorithms on quantitative benchmarks.

count=3
* Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)]
    * Title: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Alex Kendall, Yarin Gal, Roberto Cipolla
    * Abstract: Numerous deep learning applications benefit from multi-task learning with multiple regression and classification objectives. In this paper we make the observation that the performance of such systems is strongly dependent on the relative weighting between each task's loss. Tuning these weights by hand is a difficult and expensive process, making multi-task learning prohibitive in practice. We propose a principled approach to multi-task deep learning which weighs multiple loss functions by considering the homoscedastic uncertainty of each task. This allows us to simultaneously learn various quantities with different units or scales in both classification and regression settings. We demonstrate our model learning per-pixel depth regression, semantic and instance segmentation from a monocular input image. Perhaps surprisingly, we show our model can learn multi-task weightings and outperform separate models trained individually on each task.

count=3
* Recurrent Pixel Embedding for Instance Grouping
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Kong_Recurrent_Pixel_Embedding_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kong_Recurrent_Pixel_Embedding_CVPR_2018_paper.pdf)]
    * Title: Recurrent Pixel Embedding for Instance Grouping
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Shu Kong, Charless C. Fowlkes
    * Abstract: We introduce a differentiable, end-to-end trainable framework for solving pixel-level grouping problems such as instance segmentation consisting of two novel components. First, we regress pixels into a hyper-spherical embedding space so that pixels from the same group have high cosine similarity while those from different groups have similarity below a specified margin. We analyze the choice of embedding dimension and margin, relating them to theoretical results on the problem of distributing points uniformly on the sphere. Second, to group instances, we utilize a variant of mean-shift clustering, implemented as a recurrent neural network parameterized by kernel bandwidth. This recurrent grouping module is differentiable, enjoys convergent dynamics and probabilistic interpretability. Backpropagating the group-weighted loss through this module allows learning to focus on correcting embedding errors that won't be resolved during subsequent clustering. Our framework, while conceptually simple and theoretically abundant, is also practically effective and computationally efficient. We demonstrate substantial improvements over state-of-the-art instance segmentation for object proposal generation, as well as demonstrating the benefits of grouping loss for classification tasks such as boundary detection and semantic segmentation.

count=3
* Learning Deep Structured Active Contours End-to-End
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Marcos_Learning_Deep_Structured_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Marcos_Learning_Deep_Structured_CVPR_2018_paper.pdf)]
    * Title: Learning Deep Structured Active Contours End-to-End
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Diego Marcos, Devis Tuia, Benjamin Kellenberger, Lisa Zhang, Min Bai, Renjie Liao, Raquel Urtasun
    * Abstract: The world is covered with millions of buildings, and precisely knowing each instance's position and extents is vital to a multitude of applications. Recently, automated building footprint segmentation models have shown superior detection accuracy thanks to the usage of Convolutional Neural Networks (CNN). However, even the latest evolutions struggle to precisely delineating borders, which often leads to geometric distortions and inadvertent fusion of adjacent building instances. We propose to overcome this issue by exploiting the distinct geometric properties of buildings. To this end, we present Deep Structured Active Contours (DSAC), a novel framework that integrates priors and constraints into the segmentation process, such as continuous boundaries, smooth edges, and sharp corners. To do so, DSAC employs Active Contour Models (ACM), a family of constraint- and prior-based polygonal models. We learn ACM parameterizations per instance using a CNN, and show how to incorporate all components in a structured output model, making DSAC trainable end-to-end. We evaluate DSAC on three challenging building instance segmentation datasets, where it compares favorably against state-of-the-art. Code will be made available.

count=3
* Improved Extraction of Objects From Urine Microscopy Images With Unsupervised Thresholding and Supervised U-Net Techniques
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w44/html/Aziz_Improved_Extraction_of_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w44/Aziz_Improved_Extraction_of_CVPR_2018_paper.pdf)]
    * Title: Improved Extraction of Objects From Urine Microscopy Images With Unsupervised Thresholding and Supervised U-Net Techniques
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Abdul Aziz, Harshit Pande, Bharath Cheluvaraju, Tathagato Rai Dastidar
    * Abstract: We propose a novel unsupervised method for extracting objects from urine microscopy images and also applied U-net for extracting these objects. We fused these proposed methods with a known edge thresholding technique from an existing work on segmentation of urine microscopic images. Comparison between our proposed methods and the existing work showed that for certain object types the proposed unsupervised method with or without edge thresholding outperforms the other methods, while in other cases the U-net method with or without edge thresholding outperforms the other methods. Overall the proposed unsupervised method along with edge thresholding worked the best by extracting maximum number of objects and minimum number of artifacts. On a test dataset, the artifact to object ratio for the proposed unsupervised method was 0.71, which is significantly better than that of 1.26 for the existing work. The proposed unsupervised method along with edge thresholding extracted 3208 objects as compared to 1608 by the existing work. To the best of our knowledge this is the first application of Deep Learning for extraction of clinically significant objects in urine microscopy images.

count=3
* Hybrid Task Cascade for Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Hybrid_Task_Cascade_for_Instance_Segmentation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Hybrid_Task_Cascade_for_Instance_Segmentation_CVPR_2019_paper.pdf)]
    * Title: Hybrid Task Cascade for Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Kai Chen,  Jiangmiao Pang,  Jiaqi Wang,  Yu Xiong,  Xiaoxiao Li,  Shuyang Sun,  Wansen Feng,  Ziwei Liu,  Jianping Shi,  Wanli Ouyang,  Chen Change Loy,  Dahua Lin
    * Abstract: Cascade is a classic yet powerful architecture that has boosted performance on various tasks. However, how to introduce cascade to instance segmentation remains an open question. A simple combination of Cascade R-CNN and Mask R-CNN only brings limited gain. In exploring a more effective approach, we find that the key to a successful instance segmentation cascade is to fully leverage the reciprocal relationship between detection and segmentation. In this work, we propose a new framework, Hybrid Task Cascade (HTC), which differs in two important aspects: (1) instead of performing cascaded refinement on these two tasks separately, it interweaves them for a joint multi-stage processing; (2) it adopts a fully convolutional branch to provide spatial context, which can help distinguishing hard foreground from cluttered background. Overall, this framework can learn more discriminative features progressively while integrating complementary features together in each stage. Without bells and whistles, a single HTC obtains 38.4% and 1.5% improvement over a strong Cascade Mask R-CNN baseline on MSCOCO dataset. Moreover, our overall system achieves 48.6 mask AP on the test-challenge split, ranking 1st in the COCO 2018 Challenge Object Detection Task. Code is available at https://github.com/open-mmlab/mmdetection.

count=3
* Robust Histopathology Image Analysis: To Label or to Synthesize?
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Robust_Histopathology_Image_Analysis_To_Label_or_to_Synthesize_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Robust_Histopathology_Image_Analysis_To_Label_or_to_Synthesize_CVPR_2019_paper.pdf)]
    * Title: Robust Histopathology Image Analysis: To Label or to Synthesize?
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Le Hou,  Ayush Agarwal,  Dimitris Samaras,  Tahsin M. Kurc,  Rajarsi R. Gupta,  Joel H. Saltz
    * Abstract: Detection, segmentation and classification of nuclei are fundamental analysis operations in digital pathology. Existing state-of-the-art approaches demand extensive amount of supervised training data from pathologists and may still perform poorly in images from unseen tissue types. We propose an unsupervised approach for histopathology image segmentation that synthesizes heterogeneous sets of training image patches, of every tissue type. Although our synthetic patches are not always of high quality, we harness the motley crew of generated samples through a generally applicable importance sampling method. This proposed approach, for the first time, re-weighs the training loss over synthetic data so that the ideal (unbiased) generalization loss over the true data distribution is minimized. This enables us to use a random polygon generator to synthesize approximate cellular structures (i.e., nuclear masks) for which no real examples are given in many tissue types, and hence, GAN-based methods are not suited. In addition, we propose a hybrid synthesis pipeline that utilizes textures in real histopathology patches and GAN models, to tackle heterogeneity in tissue textures. Compared with existing state-of-the-art supervised models, our approach generalizes significantly better on cancer types without training data. Even in cancer types with training data, our approach achieves the same performance without supervision cost. We release code and segmentation results on over 5000 Whole Slide Images (WSI) in The Cancer Genome Atlas (TCGA) repository, a dataset that would be orders of magnitude larger than what is available today.

count=3
* PolarMask: Single Shot Instance Segmentation With Polar Representation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_PolarMask_Single_Shot_Instance_Segmentation_With_Polar_Representation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_PolarMask_Single_Shot_Instance_Segmentation_With_Polar_Representation_CVPR_2020_paper.pdf)]
    * Title: PolarMask: Single Shot Instance Segmentation With Polar Representation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Enze Xie,  Peize Sun,  Xiaoge Song,  Wenhai Wang,  Xuebo Liu,  Ding Liang,  Chunhua Shen,  Ping Luo
    * Abstract: In this paper, we introduce an anchor-box free and single shot instance segmentation method, which is conceptually simple, fully convolutional and can be used by easily embedding it into most off-the-shelf detection methods. Our method, termed PolarMask, formulates the instance segmentation problem as predicting contour of instance through instance center classification and dense distance regression in a polar coordinate. Moreover, we propose two effective approaches to deal with sampling high-quality center examples and optimization for dense distance regression, respectively, which can significantly improve the performance and simplify the training process. Without any bells and whistles, PolarMask achieves 32.9% in mask mAP with single-model and single-scale training/testing on the challenging COCO dataset. For the first time, we show that the complexity of instance segmentation, in terms of both design and computation complexity, can be the same as bounding box object detection and this much simpler and flexible instance segmentation framework can achieve competitive accuracy. We hope that the proposed PolarMask framework can serve as a fundamental and strong baseline for single shot instance segmentation task.

count=3
* VIP-DeepLab: Learning Visual Perception With Depth-Aware Video Panoptic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Qiao_VIP-DeepLab_Learning_Visual_Perception_With_Depth-Aware_Video_Panoptic_Segmentation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_VIP-DeepLab_Learning_Visual_Perception_With_Depth-Aware_Video_Panoptic_Segmentation_CVPR_2021_paper.pdf)]
    * Title: VIP-DeepLab: Learning Visual Perception With Depth-Aware Video Panoptic Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen
    * Abstract: In this paper, we present ViP-DeepLab, a unified model attempting to tackle the long-standing and challenging inverse projection problem in vision, which we model as restoring the point clouds from perspective image sequences while providing each point with instance-level semantic interpretations. Solving this problem requires the vision models to predict the spatial location, semantic class, and temporally consistent instance label for each 3D point. ViP-DeepLab approaches it by jointly performing monocular depth estimation and video panoptic segmentation. We name this joint task as Depth-aware Video Panoptic Segmentation, and propose a new evaluation metric along with two derived datasets for it, which will be made available to the public. On the individual sub-tasks, ViP-DeepLab also achieves state-of-the-art results, outperforming previous methods by 5.1% VPQ on Cityscapes-VPS, ranking 1st on the KITTI monocular depth estimation benchmark, and 1st on KITTI MOTS pedestrian. The datasets and the evaluation codes are made publicly available.

count=3
* MaX-DeepLab: End-to-End Panoptic Segmentation With Mask Transformers
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_MaX-DeepLab_End-to-End_Panoptic_Segmentation_With_Mask_Transformers_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_MaX-DeepLab_End-to-End_Panoptic_Segmentation_With_Mask_Transformers_CVPR_2021_paper.pdf)]
    * Title: MaX-DeepLab: End-to-End Panoptic Segmentation With Mask Transformers
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen
    * Abstract: We present MaX-DeepLab, the first end-to-end model for panoptic segmentation. Our approach simplifies the current pipeline that depends heavily on surrogate sub-tasks and hand-designed components, such as box detection, non-maximum suppression, thing-stuff merging, etc. Although these sub-tasks are tackled by area experts, they fail to comprehensively solve the target task. By contrast, our MaX-DeepLab directly predicts class-labeled masks with a mask transformer, and is trained with a panoptic quality inspired loss via bipartite matching. Our mask transformer employs a dual-path architecture that introduces a global memory path in addition to a CNN path, allowing direct communication with any CNN layers. As a result, MaX-DeepLab shows a significant 7.1% PQ gain in the box-free regime on the challenging COCO dataset, closing the gap between box-based and box-free methods for the first time. A small variant of MaX-DeepLab improves 3.0% PQ over DETR with similar parameters and M-Adds. Furthermore, MaX-DeepLab, without test time augmentation, achieves new state-of-the-art 51.3% PQ on COCO test-dev set.

count=3
* Clustering Plotted Data by Image Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Naous_Clustering_Plotted_Data_by_Image_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Naous_Clustering_Plotted_Data_by_Image_Segmentation_CVPR_2022_paper.pdf)]
    * Title: Clustering Plotted Data by Image Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Tarek Naous, Srinjay Sarkar, Abubakar Abid, James Zou
    * Abstract: Clustering is a popular approach to detecting patterns in unlabeled data. Existing clustering methods typically treat samples in a dataset as points in a metric space and compute distances to group together similar points. In this paper, we present a different way of clustering points in 2-dimensional space, inspired by how humans cluster data: by training neural networks to perform instance segmentation on plotted data. Our approach, Visual Clustering, has several advantages over traditional clustering algorithms: it is much faster than most existing clustering algorithms (making it suitable for very large datasets), it agrees strongly with human intuition for clusters, and it is by default hyperparameter free (although additional steps with hyperparameters can be introduced for more control of the algorithm). We describe the method and compare it to ten other clustering methods on synthetic data to illustrate its advantages and disadvantages. We then demonstrate how our approach can be extended to higher-dimensional data and illustrate its performance on real-world data. Our implementation of Visual Clustering is publicly available as a python package that can be installed and used on any dataset in a few lines of code. A demo on synthetic datasets is provided.

count=3
* DiSparse: Disentangled Sparsification for Multitask Model Compression
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_DiSparse_Disentangled_Sparsification_for_Multitask_Model_Compression_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_DiSparse_Disentangled_Sparsification_for_Multitask_Model_Compression_CVPR_2022_paper.pdf)]
    * Title: DiSparse: Disentangled Sparsification for Multitask Model Compression
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Xinglong Sun, Ali Hassani, Zhangyang Wang, Gao Huang, Humphrey Shi
    * Abstract: Despite the popularity of Model Compression and Multitask Learning, how to effectively compress a multitask model has been less thoroughly analyzed due to the challenging entanglement of tasks in the parameter space. In this paper, we propose DiSparse, a simple, effective, and first-of-its-kind multitask pruning and sparse training scheme. We consider each task independently by disentangling the importance measurement and take the unanimous decisions among all tasks when performing parameter pruning and selection. Our experimental results demonstrate superior performance on various configurations and settings compared to popular sparse training and pruning methods. Besides the effectiveness in compression, DiSparse also provides a powerful tool to the multitask learning community. Surprisingly, we even observed better performance than some dedicated multitask learning methods in several cases despite the high model sparsity enforced by DiSparse. We analyzed the pruning masks generated with DiSparse and observed strikingly similar sparse network architecture identified by each task even before the training starts. We also observe the existence of a "watershed" layer where the task relatedness sharply drops, implying no benefits in continued parameters sharing. Our code and models will be available at: https://github.com/SHI-Labs/DiSparse-Multitask-Model-Compression.

count=3
* CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_CMT-DeepLab_Clustering_Mask_Transformers_for_Panoptic_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_CMT-DeepLab_Clustering_Mask_Transformers_for_Panoptic_Segmentation_CVPR_2022_paper.pdf)]
    * Title: CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Qihang Yu, Huiyu Wang, Dahun Kim, Siyuan Qiao, Maxwell Collins, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen
    * Abstract: We propose Clustering Mask Transformer (CMT-DeepLab), a transformer-based framework for panoptic segmentation designed around clustering. It rethinks the existing transformer architectures used in segmentation and detection; CMT-DeepLab considers the object queries as cluster centers, which fill the role of grouping the pixels when applied to segmentation. The clustering is computed with an alternating procedure, by first assigning pixels to the clusters by their feature affinity, and then updating the cluster centers and pixel features. Together, these operations comprise the Clustering Mask Transformer (CMT) layer, which produces cross-attention that is denser and more consistent with the final segmentation task. CMT-DeepLab improves the performance over prior art significantly by 4.4% PQ, achieving a new state-of-the-art of 55.7% PQ on the COCO test-dev set.

count=3
* DoNet: Deep De-Overlapping Network for Cytology Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Jiang_DoNet_Deep_De-Overlapping_Network_for_Cytology_Instance_Segmentation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_DoNet_Deep_De-Overlapping_Network_for_Cytology_Instance_Segmentation_CVPR_2023_paper.pdf)]
    * Title: DoNet: Deep De-Overlapping Network for Cytology Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Hao Jiang, Rushan Zhang, Yanning Zhou, Yumeng Wang, Hao Chen
    * Abstract: Cell instance segmentation in cytology images has significant importance for biology analysis and cancer screening, while remains challenging due to 1) the extensive overlapping translucent cell clusters that cause the ambiguous boundaries, and 2) the confusion of mimics and debris as nuclei. In this work, we proposed a De-overlapping Network (DoNet) in a decompose-and-recombined strategy. A Dual-path Region Segmentation Module (DRM) explicitly decomposes the cell clusters into intersection and complement regions, followed by a Semantic Consistency-guided Recombination Module (CRM) for integration. To further introduce the containment relationship of the nucleus in the cytoplasm, we design a Mask-guided Region Proposal Strategy (MRP) that integrates the cell attention maps for inner-cell instance prediction. We validate the proposed approach on ISBI2014 and CPS datasets. Experiments show that our proposed DoNet significantly outperforms other state-of-the-art (SOTA) cell instance segmentation methods. The code is available at https://github.com/DeepDoNet/DoNet.

count=3
* An Ensemble Method With Edge Awareness for Abnormally Shaped Nuclei Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVMI/html/Han_An_Ensemble_Method_With_Edge_Awareness_for_Abnormally_Shaped_Nuclei_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVMI/papers/Han_An_Ensemble_Method_With_Edge_Awareness_for_Abnormally_Shaped_Nuclei_CVPRW_2023_paper.pdf)]
    * Title: An Ensemble Method With Edge Awareness for Abnormally Shaped Nuclei Segmentation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yue Han, Yang Lei, Viktor Shkolnikov, Daisy Xin, Alicia Auduong, Steven Barcelo, Jan Allebach, Edward J. Delp
    * Abstract: Abnormalities in biological cell nuclei shapes are correlated with cell cycle stages, disease states, and various external stimuli. There have been many deep learning approaches that are being used for nuclei segmentation and analysis. In recent years, transformers have performed better than CNN methods on many computer vision tasks. One problem with many deep learning nuclei segmentation methods is acquiring large amounts of annotated nuclei data, which is generally expensive to obtain. In this paper, we propose a Transformer and CNN hybrid ensemble processing method with edge awareness for accurately segmenting abnormally shaped nuclei. We call this method Hybrid Edge Mask R-CNN (HER-CNN), which uses Mask R-CNNs with the ResNet and the Swin-Transformer to segment abnormally shaped nuclei. We add an edge awareness loss to the mask prediction step of the Mask R-CNN to better distinguish the edge difference between the abnormally shaped nuclei and typical oval nuclei. We describe an ensemble processing strategy to combine or fuse individual segmentations from the CNN and the Transformer. We introduce the use of synthetic ground truth image generation to supplement the annotated training images due to the limited amount of data. Our proposed method is compared with other segmentation methods for segmenting abnormally shaped nuclei. We also include ablation studies to show the effectiveness of the edge awareness loss and the use of synthetic ground truth images.

count=3
* Robust Multiview Multimodal Driver Monitoring System Using Masked Multi-Head Self-Attention
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/MULA/html/Ma_Robust_Multiview_Multimodal_Driver_Monitoring_System_Using_Masked_Multi-Head_Self-Attention_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/MULA/papers/Ma_Robust_Multiview_Multimodal_Driver_Monitoring_System_Using_Masked_Multi-Head_Self-Attention_CVPRW_2023_paper.pdf)]
    * Title: Robust Multiview Multimodal Driver Monitoring System Using Masked Multi-Head Self-Attention
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yiming Ma, Victor Sanchez, Soodeh Nikan, Devesh Upadhyay, Bhushan Atote, Tanaya Guha
    * Abstract: Driver Monitoring Systems (DMSs) are crucial for safe hand-over actions in Level-2+ self-driving vehicles. State-of-the-art DMSs leverage multiple sensors mounted at different locations to monitor the driver and the vehicle's interior scene and employ decision-level fusion to integrate these heterogenous data. However, this fusion method may not fully utilize the complementarity of different data sources and may overlook their relative importance. To address these limitations, we propose a novel multiview multimodal driver monitoring system based on feature-level fusion through multi-head self-attention (MHSA). We demonstrate its effectiveness by comparing it against four alternative fusion strategies (Sum, Conv, SE, and AFF). We also present a novel GPU-friendly supervised contrastive learning framework SuMoCo to learn better representations. Furthermore, We fine-grained the test split of the DAD dataset to enable the multi-class recognition of drivers' activities. Experiments on this enhanced database demonstrate that 1) the proposed MHSA-based fusion method (AUC-ROC: 97.0%) outperforms all baselines and previous approaches, and 2) training MHSA with patch masking can improve its robustness against modality/view collapses. The code and annotations are publicly available.

count=3
* ArcticNet: A Deep Learning Solution to Classify the Arctic Area
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/DOAI/Jiang_ArcticNet_A_Deep_Learning_Solution_to_Classify_the_Arctic_Area_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Jiang_ArcticNet_A_Deep_Learning_Solution_to_Classify_the_Arctic_Area_CVPRW_2019_paper.pdf)]
    * Title: ArcticNet: A Deep Learning Solution to Classify the Arctic Area
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Ziyu Jiang,  Kate Von Ness,  Julie Loisel,  Zhangyang Wang
    * Abstract: Arctic environments are rapidly changing under the warming climate. Of particular interest are wetlands, a type of ecosystem that constitutes the most effective terrestrial long-term carbon store. As permafrost thaws, the carbon that was locked in these wetland soils for millennia becomes available for aerobic and anaerobic decomposition, which releases carbon dioxide CO2 and methane CH4, respectively, back to the atmosphere. As CO2 and CH4 are potent greenhouse gases, this transfer of carbon from the land to the atmosphere further contributes to global warming, thereby increasing the rate of permafrost degradation in a positive feedback loop. Therefore, monitoring Arctic wetland health and dynamics is a key scientific task that is also of importance for policy. However, the identification and delineation of these important wetland ecosystems, remain incomplete and often inaccurate. Mapping the extent of Arctic wetlands remains a challenge for the scientific community. Conventional, coarser remote sensing methods are inadequate at distinguishing the diverse and micro-topographically complex non-vascular vegetation that characterize Arctic wetlands, presenting the need for better identification methods. To tackle this challenging problem, we constructed and annotated the first-of-its-kind Arctic Wetland Dataset (AWD). Based on that, we present ArcticNet, a deep neural network that exploits the multi-spectral, high-resolution imagery captured from nanosatellites (Planet Dove CubeSats) with additional Digital Elevation Model (DEM) from the ArcticDEM project, to semantically label a Arctic study area into six types, in which three Arctic wetland functional types are included. We present multi-fold efforts to handle the arising challenges, including class imbalance, and the choice of fusion strategies. Preliminary results endorse the high promise of ArcticNet, achieving 93.12% in labelling a hold-out set of regions in our Arctic study area.

count=3
* Volumetric Semantic Segmentation Using Pyramid Context Features
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Barron_Volumetric_Semantic_Segmentation_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Barron_Volumetric_Semantic_Segmentation_2013_ICCV_paper.pdf)]
    * Title: Volumetric Semantic Segmentation Using Pyramid Context Features
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Jonathan T. Barron, Mark D. Biggin, Pablo Arbelaez, David W. Knowles, Soile V.E. Keranen, Jitendra Malik
    * Abstract: We present an algorithm for the per-voxel semantic segmentation of a three-dimensional volume. At the core of our algorithm is a novel "pyramid context" feature, a descriptive representation designed such that exact per-voxel linear classification can be made extremely efficient. This feature not only allows for efficient semantic segmentation but enables other aspects of our algorithm, such as novel learned features and a stacked architecture that can reason about self-consistency. We demonstrate our technique on 3D fluorescence microscopy data of Drosophila embryos for which we are able to produce extremely accurate semantic segmentations in a matter of minutes, and for which other algorithms fail due to the size and high-dimensionality of the data, or due to the difficulty of the task.

count=3
* SGN: Sequential Grouping Networks for Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_SGN_Sequential_Grouping_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_SGN_Sequential_Grouping_ICCV_2017_paper.pdf)]
    * Title: SGN: Sequential Grouping Networks for Instance Segmentation
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Shu Liu, Jiaya Jia, Sanja Fidler, Raquel Urtasun
    * Abstract: In this paper, we propose Sequential Grouping Networks (SGN) to tackle the problem of object instance segmentation. SGNs employ a sequence of neural networks, each solving a sub-grouping problem of increasing semantic complexity in order to gradually compose objects out of pixels. In particular, the first network aims to group pixels along each image row and column by predicting horizontal and vertical object breakpoints. These breakpoints are then used to create line segments. By exploiting two-directional information, the second network groups horizontal and vertical lines into connected components. Finally, the third network groups the connected components into object instances. Our experiments show that our SGN significantly outperforms state-of-the-art approaches in both, the Cityscapes dataset as well as PASCAL VOC.

count=3
* Leaf Counting With Deep Convolutional and Deconvolutional Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w29/html/Aich_Leaf_Counting_With_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w29/Aich_Leaf_Counting_With_ICCV_2017_paper.pdf)]
    * Title: Leaf Counting With Deep Convolutional and Deconvolutional Networks
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Shubhra Aich, Ian Stavness
    * Abstract: In this paper, we investigate the problem of counting rosette leaves from an RGB image, an important task in plant phenotyping. We propose a data-driven approach for this task generalized over different plant species and imaging setups. To accomplish this task, we use state-of-the-art deep learning architectures: a deconvolutional network for initial segmentation and a convolutional network for leaf counting. Evaluation is performed on the leaf counting challenge dataset at CVPPP-2017. Despite the small number of training samples in this dataset, as compared to typical deep learning image sets, we obtain satisfactory performance on segmenting leaves from the background as a whole and counting the number of leaves using simple data augmentation strategies. Comparative analysis is provided against methods evaluated on the previous competition datasets. Our framework achieves mean and standard deviation of absolute count difference of 1.62 and 2.30 averaged over all five test datasets.

count=3
* Interactive Class-Agnostic Object Counting
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Interactive_Class-Agnostic_Object_Counting_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_Interactive_Class-Agnostic_Object_Counting_ICCV_2023_paper.pdf)]
    * Title: Interactive Class-Agnostic Object Counting
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yifeng Huang, Viresh Ranjan, Minh Hoai
    * Abstract: We propose a novel framework for interactive class-agnostic object counting, where a human user can interactively provide feedback to improve the accuracy of a counter. Our framework consists of two main components: a user-friendly visualizer to gather feedback and an efficient mechanism to incorporate it. In each iteration, we produce a density map to show the current prediction result, and we segment it into non-overlapping regions with an easily verifiable number of objects. The user can provide feedback by selecting a region with obvious counting errors and specifying the range for the estimated number of objects within it. To improve the counting result, we develop a novel adaptation loss to force the visual counter to output the predicted count within the user-specified range. For effective and efficient adaptation, we propose a refinement module that can be used with any density-based visual counter, and only the parameters in the refinement module will be updated during adaptation. Our experiments on two challenging class-agnostic object counting benchmarks, FSCD-LVIS and FSC-147, show that our method can reduce the mean absolute error of multiple state-of-the-art visual counters by roughly 30% to 40% with minimal user input. Our project can be found at https://yifehuang97.github.io/ICACountProjectPage/.

count=3
* Frequency-aware GAN for Adversarial Manipulation Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Frequency-aware_GAN_for_Adversarial_Manipulation_Generation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Frequency-aware_GAN_for_Adversarial_Manipulation_Generation_ICCV_2023_paper.pdf)]
    * Title: Frequency-aware GAN for Adversarial Manipulation Generation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Peifei Zhu, Genki Osada, Hirokatsu Kataoka, Tsubasa Takahashi
    * Abstract: Image manipulation techniques have drawn growing concerns as manipulated images might cause morality and security problems. Various methods have been proposed to detect manipulations and achieved promising performance. However, these methods might be vulnerable to adversarial attacks. In this work, we design an Adversarial Manipulation Generation (AMG) task to explore the vulnerability of image manipulation detectors. We first propose an optimal loss function and extend existing attacks to generate adversarial examples. We observe that existing spatial attacks cause large degradation in image quality and find the loss of high-frequency detailed components might be its major reason. Inspired by this observation, we propose a novel adversarial attack that incorporates both spatial and frequency features into the GAN architecture to generate adversarial examples. We further design an encoder-decoder architecture with skip connections of high-frequency components to preserve fine details. We evaluated our method on three image manipulation detectors (FCN, ManTra-Net and MVSS-Net) with three benchmark datasets (DEFACTO, CASIAv2 and COVER). Experiments show that our method generates adversarial examples significantly fast (0.01s per image), preserves better image quality (PSNR 30% higher than spatial attacks) and achieves a high attack success rate. We also observe that the examples generated by AMG can fool both classification and segmentation models, which indicates better transferability among different tasks.

count=3
* Rapid Flood Inundation Forecast Using Fourier Neural Operator
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/AIHADR/html/Sun_Rapid_Flood_Inundation_Forecast_Using_Fourier_Neural_Operator_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/AIHADR/papers/Sun_Rapid_Flood_Inundation_Forecast_Using_Fourier_Neural_Operator_ICCVW_2023_paper.pdf)]
    * Title: Rapid Flood Inundation Forecast Using Fourier Neural Operator
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Alexander Y. Sun, Zhi Li, Wonhyun Lee, Qixing Huang, Bridget R. Scanlon, Clint Dawson
    * Abstract: Flood inundation forecast provides critical information for emergency planning before and during flood events. Real time flood inundation forecast tools are still lacking. High-resolution hydrodynamic modeling has become more accessible in recent years, however, predicting flood extents at the street and building levels in real-time is still computationally demanding. Here we present a hybrid process-based and data-driven machine learning (ML) approach for flood extent and inundation depth prediction. We used the Fourier neural operator (FNO), a highly efficient ML method, for surrogate modeling. The FNO model is demonstrated over an urban area in Houston (Texas, U.S.) by training using simulated water depths (in 15-min intervals) from six historical storm events and then tested over two holdout events. Results show FNO outperforms the baseline U-Net model. It maintains high predictability at all lead times tested (up to 3 hrs) and performs well when applying to new sites, suggesting strong generalization skill.

count=3
* Focus on Content not Noise: Improving Image Generation for Nuclei Segmentation by Suppressing Steganography in CycleGAN
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/BIC/html/Utz_Focus_on_Content_not_Noise_Improving_Image_Generation_for_Nuclei_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/BIC/papers/Utz_Focus_on_Content_not_Noise_Improving_Image_Generation_for_Nuclei_ICCVW_2023_paper.pdf)]
    * Title: Focus on Content not Noise: Improving Image Generation for Nuclei Segmentation by Suppressing Steganography in CycleGAN
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Jonas Utz, Tobias Weise, Maja Schlereth, Fabian Wagner, Mareike Thies, Mingxuan Gu, Stefan Uderhardt, Katharina Breininger
    * Abstract: Annotating nuclei in microscopy images for the training of neural networks is a laborious task that requires expert knowledge and suffers from inter-and intra-rater variability, especially in fluorescence microscopy. Generative networks such as CycleGAN can inverse the process and generate synthetic microscopy images for a given mask, thereby building a synthetic dataset. However, past works report content inconsistencies between the mask and generated image, partially due to CycleGAN minimizing its loss by hiding shortcut information for the image reconstruction in high frequencies rather than encoding the desired image content and learning the target task. In this work, we propose to remove the hidden shortcut information, called steganography, from generated images by employing a low pass filtering based on the DCT. We show that this increases coherence between generated images and cycled masks and evaluate synthetic datasets on a downstream nuclei segmentation task. Here we achieve an improvement of 5.4 points in the F1-score compared to a vanilla CycleGAN. Integrating advanced regularization techniques into the CycleGAN architecture may help mitigate steganography-related issues and produce more accurate synthetic datasets for nuclei segmentation.

count=3
* Deep Remote Sensing Methods for Methane Detection in Overhead Hyperspectral Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Kumar_Deep_Remote_Sensing_Methods_for_Methane_Detection_in_Overhead_Hyperspectral_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Kumar_Deep_Remote_Sensing_Methods_for_Methane_Detection_in_Overhead_Hyperspectral_WACV_2020_paper.pdf)]
    * Title: Deep Remote Sensing Methods for Methane Detection in Overhead Hyperspectral Imagery
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Satish Kumar,  Carlos Torres,  Oytun Ulutan,  Alana Ayasse,  Dar Roberts,  B. S. Manjunath
    * Abstract: Effective analysis of hyperspectral imagery is essential for gathering fast and actionable information of large areas affected by atmospheric and green house gases. Existing methods, which process hyperspectral data to detect amorphous gases such as CH4 require manual inspection from domain experts and annotation of massive datasets. These methods do not scale well and are prone to human errors due to the plumes' small pixel-footprint signature. The proposed Hyperspectral Mask-RCNN (H-mrcnn) uses principled statistics, signal processing, and deep neural networks to address these limitations. H-mrcnn introduces fast algorithms to analyze large-area hyper-spectral information and methods to autonomously represent and detect CH4 plumes. H-mrcnn processes information by match-filtering sliding windows of hyperspectral data across the spectral bands. This process produces information-rich features that are both effective plume representations and gas concentration analogs. The optimized matched-filtering stage processes spectral data, which is spatially sampled to train an ensemble of gas detectors. The ensemble outputs are fused to estimate a natural and accurate plume mask. Thorough evaluation demonstrates that H-mrcnn matches the manual and experience-dependent annotation process of experts by 85% (IOU). H-mrcnn scales to larger datasets, reduces the manual data processing and labeling time (12 times), and produces rapid actionable information about gas plumes.

count=3
* ShadowSense: Unsupervised Domain Adaptation and Feature Fusion for Shadow-Agnostic Tree Crown Detection From RGB-Thermal Drone Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Kapil_ShadowSense_Unsupervised_Domain_Adaptation_and_Feature_Fusion_for_Shadow-Agnostic_Tree_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kapil_ShadowSense_Unsupervised_Domain_Adaptation_and_Feature_Fusion_for_Shadow-Agnostic_Tree_WACV_2024_paper.pdf)]
    * Title: ShadowSense: Unsupervised Domain Adaptation and Feature Fusion for Shadow-Agnostic Tree Crown Detection From RGB-Thermal Drone Imagery
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Rudraksh Kapil, Seyed Mojtaba Marvasti-Zadeh, Nadir Erbilgin, Nilanjan Ray
    * Abstract: Accurate detection of individual tree crowns from remote sensing data poses a significant challenge due to the dense nature of forest canopy and the presence of diverse environmental variations, e.g., overlapping canopies, occlusions, and varying lighting conditions. Additionally, the lack of data for training robust models adds another limitation in effectively studying complex forest conditions. This paper presents a novel method for detecting shadowed tree crowns and provides a challenging dataset comprising roughly 50k paired RGB-thermal images to facilitate future research for illumination-invariant detection. The proposed method (ShadowSense) is entirely self-supervised, leveraging domain adversarial training without source domain annotations for feature extraction and foreground feature alignment for feature pyramid networks to adapt domain-invariant representations by focusing on visible foreground regions, respectively. It then fuses complementary information of both modalities to effectively improve upon the predictions of an RGB-trained detector and boost the overall accuracy. Extensive experiments demonstrate the superiority of the proposed method over both the baseline RGB-trained detector and state-of-the-art techniques that rely on unsupervised domain adaptation or early image fusion. Our code and data are available: https://github.com/rudrakshkapil/ShadowSense

count=3
* Stochastic Network Design in Bidirected Trees
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2014/hash/99c5e07b4d5de9d18c350cdf64c5aa3d-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2014/file/99c5e07b4d5de9d18c350cdf64c5aa3d-Paper.pdf)]
    * Title: Stochastic Network Design in Bidirected Trees
    * Publisher: NeurIPS
    * Publication Date: `2014`
    * Authors: xiaojian wu, Daniel R. Sheldon, Shlomo Zilberstein
    * Abstract: We investigate the problem of stochastic network design in bidirected trees. In this problem, an underlying phenomenon (e.g., a behavior, rumor, or disease) starts at multiple sources in a tree and spreads in both directions along its edges. Actions can be taken to increase the probability of propagation on edges, and the goal is to maximize the total amount of spread away from all sources. Our main result is a rounded dynamic programming approach that leads to a fully polynomial-time approximation scheme (FPTAS), that is, an algorithm that can find (1−ε)-optimal solutions for any problem instance in time polynomial in the input size and 1/ε. Our algorithm outperforms competing approaches on a motivating problem from computational sustainability to remove barriers in river networks to restore the health of aquatic ecosystems.

count=3
* Automatic Neuron Detection in Calcium Imaging Data Using Convolutional Networks
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2016/hash/0771fc6f0f4b1d7d1bb73bbbe14e0e31-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2016/file/0771fc6f0f4b1d7d1bb73bbbe14e0e31-Paper.pdf)]
    * Title: Automatic Neuron Detection in Calcium Imaging Data Using Convolutional Networks
    * Publisher: NeurIPS
    * Publication Date: `2016`
    * Authors: Noah Apthorpe, Alexander Riordan, Robert Aguilar, Jan Homann, Yi Gu, David Tank, H. Sebastian Seung
    * Abstract: Calcium imaging is an important technique for monitoring the activity of thousands of neurons simultaneously. As calcium imaging datasets grow in size, automated detection of individual neurons is becoming important. Here we apply a supervised learning approach to this problem and show that convolutional networks can achieve near-human accuracy and superhuman speed. Accuracy is superior to the popular PCA/ICA method based on precision and recall relative to ground truth annotation by a human expert. These results suggest that convolutional networks are an efficient and flexible tool for the analysis of large-scale calcium imaging data.

count=3
* Automated scalable segmentation of neurons from multispectral images
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2016/hash/7cce53cf90577442771720a370c3c723-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2016/file/7cce53cf90577442771720a370c3c723-Paper.pdf)]
    * Title: Automated scalable segmentation of neurons from multispectral images
    * Publisher: NeurIPS
    * Publication Date: `2016`
    * Authors: Uygar Sümbül, Douglas Roossien, Dawen Cai, Fei Chen, Nicholas Barry, John P. Cunningham, Edward Boyden, Liam Paninski
    * Abstract: Reconstruction of neuroanatomy is a fundamental problem in neuroscience. Stochastic expression of colors in individual cells is a promising tool, although its use in the nervous system has been limited due to various sources of variability in expression. Moreover, the intermingled anatomy of neuronal trees is challenging for existing segmentation algorithms. Here, we propose a method to automate the segmentation of neurons in such (potentially pseudo-colored) images. The method uses spatio-color relations between the voxels, generates supervoxels to reduce the problem size by four orders of magnitude before the final segmentation, and is parallelizable over the supervoxels. To quantify performance and gain insight, we generate simulated images, where the noise level and characteristics, the density of expression, and the number of fluorophore types are variable. We also present segmentations of real Brainbow images of the mouse hippocampus, which reveal many of the dendritic segments.

count=2
* Generic Image Segmentation in Fully Convolutional Networks by Superpixel Merging Map
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Huang_Generic_Image_Segmentation_in_Fully_Convolutional_Networks_by_Superpixel_Merging_ACCV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2020/papers/Huang_Generic_Image_Segmentation_in_Fully_Convolutional_Networks_by_Superpixel_Merging_ACCV_2020_paper.pdf)]
    * Title: Generic Image Segmentation in Fully Convolutional Networks by Superpixel Merging Map
    * Publisher: ACCV
    * Publication Date: `2020`
    * Authors: Jin-Yu Huang, Jian-Jiun Ding
    * Abstract: Recently, the Fully Convolutional Network (FCN) has been adopted in image segmentation. However, existing FCN-based segmentation algorithms were designed for semantic segmentation. Before learning-based algorithms were developed, many advanced generic segmentation algorithms are superpixel-based. However, due to the irregular shape and size of superpixels, it is hard to apply deep learning to superpixel-based image segmentation directly. In this paper, we combined the merits of the FCN and superpixels and proposed a highly accurate and extremely fast generic image segmentation algorithm. We treated image segmentation as multiple superpixel merging decision problems and determined whether the boundary between two adjacent superpixels should be kept. In other words, if the boundary of two adjacent superpixels should be deleted, then the two superpixels will be merged. The network applies the colors, the edge map, and the superpixel information to make decision about merging suprepixels. By solving all the superpixel-merging subproblems with just one forward pass, the FCN facilitates the speed of the whole segmentation process by a wide margin meanwhile gaining higher accuracy. Simulations show that the proposed algorithm has favorable runtime, meanwhile achieving highly accurate segmentation results. It outperforms state-of-the-art image segmentation methods, including feature-based and learning-based methods, in all metrics.

count=2
* Joint Spectral Correspondence for Disparate Image Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Bansal_Joint_Spectral_Correspondence_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Bansal_Joint_Spectral_Correspondence_2013_CVPR_paper.pdf)]
    * Title: Joint Spectral Correspondence for Disparate Image Matching
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Mayank Bansal, Kostas Daniilidis
    * Abstract: We address the problem of matching images with disparate appearance arising from factors like dramatic illumination (day vs. night), age (historic vs. new) and rendering style differences. The lack of local intensity or gradient patterns in these images makes the application of pixellevel descriptors like SIFT infeasible. We propose a novel formulation for detecting and matching persistent features between such images by analyzing the eigen-spectrum of the joint image graph constructed from all the pixels in the two images. We show experimental results of our approach on a public dataset of challenging image pairs and demonstrate significant performance improvements over state-of-the-art.

count=2
* Analysing the Structure of Collagen Fibres in SBFSEM Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w27/html/Almutairi_Analysing_the_Structure_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w27/papers/Almutairi_Analysing_the_Structure_CVPR_2016_paper.pdf)]
    * Title: Analysing the Structure of Collagen Fibres in SBFSEM Images
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Yassar Almutairi, Timothy Cootes, Karl Kadler
    * Abstract: Collagen fibres form important structures in tissue, and are essential for force transmission, scaffolding and cell addition. Each fibre is long and thin, and large numbers group together into complex networks of bundles, which are little studied as yet. Serial block-face scanning electron microscopy (SBFSEM) can be used to image tissues containing the fibres, but analysing the images manually is almost impossible - there can be over 30,000 fibres in each image slice, and many hundreds of individual image slices in a volume. We describe a system for automatically identifying and tracking the individual fibres, allowing analysis of their paths, how they form bundles and how individual fibres weave from one bundle to another.

count=2
* InstanceCut: From Edges to Instances With MultiCut
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Kirillov_InstanceCut_From_Edges_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Kirillov_InstanceCut_From_Edges_CVPR_2017_paper.pdf)]
    * Title: InstanceCut: From Edges to Instances With MultiCut
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Alexander Kirillov, Evgeny Levinkov, Bjoern Andres, Bogdan Savchynskyy, Carsten Rother
    * Abstract: This work addresses the task of instance-aware semantic segmentation. Our key motivation is to design a simple method with a new modelling-paradigm, which therefore has a different trade-off between advantages and disadvantages compared to known approaches. Our approach, we term InstanceCut, represents the problem by two output modalities: (i) an instance-agnostic semantic segmentation and (ii) all instance-boundaries. The former is computed from a standard convolutional neural network for semantic segmentation, and the latter is derived from a new instance-aware edge detection model. To reason globally about the optimal partitioning of an image into instances, we combine these two modalities into a novel MultiCut formulation. We evaluate our approach on the challenging CityScapes dataset. Despite the conceptual simplicity of our approach, we achieve the best result among all published methods, and perform particularly well for rare object classes.

count=2
* Microscopic Blood Smear Segmentation and Classification Using Deep Contour Aware CNN and Extreme Machine Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/html/Razzak_Microscopic_Blood_Smear_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/papers/Razzak_Microscopic_Blood_Smear_CVPR_2017_paper.pdf)]
    * Title: Microscopic Blood Smear Segmentation and Classification Using Deep Contour Aware CNN and Extreme Machine Learning
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Muhammad Imran Razzak, Saeeda Naz
    * Abstract: Recent advancement in genomic technologies has opened a new realm for early detection of diseases that shows potential to overcome the drawbacks of manual detection technologies. In this work, we have presented efficient contour aware segmentation approach based based on fully conventional network whereas for classification we have used extreme machine learning based on CNN features extracted from each segmented cell. We have evaluated system performance based on segmentation and classification on publicly available dataset. Experiment was conducted on 64000 blood cells and dataset is divided into 80% for training and 20% for testing. Segmentation results are compared with the manual segmentation and found that proposed approach provided with 98.12% and 98.16% for RBC and WBC respectively whereas classification accuracy is shown on publicly available dataset 94.71% and 98.68% for RBC \& its abnormalities detection and WBC respectively.

count=2
* MaskLab: Instance Segmentation by Refining Object Detection With Semantic and Direction Features
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_MaskLab_Instance_Segmentation_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_MaskLab_Instance_Segmentation_CVPR_2018_paper.pdf)]
    * Title: MaskLab: Instance Segmentation by Refining Object Detection With Semantic and Direction Features
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Liang-Chieh Chen, Alexander Hermans, George Papandreou, Florian Schroff, Peng Wang, Hartwig Adam
    * Abstract: In this work, we tackle the problem of instance segmentation, the task of simultaneously solving object detection and semantic segmentation. Towards this goal, we present a model, called MaskLab, which produces three outputs: box detection, semantic segmentation, and direction prediction. Building on top of the Faster-RCNN object detector, the predicted boxes provide accurate localization of object instances. Within each region of interest, MaskLab performs foreground/background segmentation by combining semantic and direction prediction. Semantic segmentation assists the model in distinguishing between objects of different semantic classes including background, while the direction prediction, estimating each pixel's direction towards its corresponding center, allows separating instances of the same semantic class. Moreover, we explore the effect of incorporating recent successful methods from both segmentation and detection (eg, atrous convolution and hypercolumn). Our proposed model is evaluated on the COCO instance segmentation benchmark and shows comparable performance with other state-of-art models.

count=2
* Matching Adversarial Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Mattyus_Matching_Adversarial_Networks_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mattyus_Matching_Adversarial_Networks_CVPR_2018_paper.pdf)]
    * Title: Matching Adversarial Networks
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Gellért Máttyus, Raquel Urtasun
    * Abstract: Generative Adversarial Nets (GANs) and Conditonal GANs (CGANs) show that using a trained network as loss function (discriminator) enables to synthesize highly structured outputs (e.g. natural images). However, applying a discriminator network as a universal loss function for common supervised tasks (e.g. semantic segmentation, line detection, depth estimation) is considerably less successful. We argue that the main difficulty of applying CGANs to supervised tasks is that the generator training consists of optimizing a loss function that does not depend directly on the ground truth labels. To overcome this, we propose to replace the discriminator with a matching network taking into account both the ground truth outputs as well as the generated examples. As a consequence, the generator loss function also depends on the targets of the training examples, thus facilitating learning. We demonstrate on three computer vision tasks that this approach can significantly outperform CGANs achieving comparable or superior results to task-specific solutions and results in stable training. Importantly, this is a general approach that does not require the use of task-specific loss functions.

count=2
* Image Segmentation by Deep Learning of Disjunctive Normal Shape Model Shape Representation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w10/html/Javanmardi_Image_Segmentation_by_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w10/Javanmardi_Image_Segmentation_by_CVPR_2018_paper.pdf)]
    * Title: Image Segmentation by Deep Learning of Disjunctive Normal Shape Model Shape Representation
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Mehran Javanmardi, Ricardo Bigolin Lanfredi, Mujdat Cetin, Tolga Tasdizen
    * Abstract: Segmenting images with low-quality, low signal to noise ratio has been a challenging task in computer vision. It has been shown that statistical prior information about the shape of the object to be segmented can be used to significantly mitigate this problem. However estimating the probability densities of the object shapes in the space of shapes can be difficult. This problem becomes more difficult when there is limited amount of training data or the testing images contain missing data. Most shape model based segmentation approaches tend to minimize an energy functional to segment the object. In this paper we propose a shape-based segmentation algorithm that utilizes convolutional neural networks to learn a posterior distribution of disjunction of conjunctions of half spaces to segment the object. This approach shows promising results on noisy and occluded data where it is able to accurately segment the objects. We show visual and quantitative results on datasets from several applications, demonstrating the effectiveness of the proposed approach. We should also note that inference with a CNN is computationally more efficient than density estimation and sampling approaches.

count=2
* Large Kernel Refine Fusion Net for Neuron Membrane Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w44/html/Liu_Large_Kernel_Refine_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w44/Liu_Large_Kernel_Refine_CVPR_2018_paper.pdf)]
    * Title: Large Kernel Refine Fusion Net for Neuron Membrane Segmentation
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Dongnan Liu, Donghao Zhang, Yang Song, Chaoyi Zhang, Heng Huang, Mei Chen, Weidong Cai
    * Abstract: 2D neuron membrane segmentation for Electron Microscopy (EM) images is a key step in the 3D neuron reconstruction task. Compared with the semantic segmentation tasks for general images, the boundary segmentation in EM images is more challenging. In EM segmentation tasks, we need not only to segment the ambiguous membrane boundaries from bubble-like noise in the images, but also to remove shadow-like intracellular structure. In order to address these problems, we propose a Large Kernel Refine Fusion Net, an encoder-decoder architecture with fusion of features at multiple resolution levels. We incorporate large convolutional blocks to ensure the valid receptive fields for the feature maps are large enough, which can reduce information loss. Our model can also process the background together with the membrane boundary by using residual cascade pooling blocks. In addition, the postprocessing method in our work is simple but effective for a final refinement of the output probability map. Our method was evaluated and achieved competitive performances on two EM membrane segmentation tasks: ISBI2012 EM segmentation challenge and mouse piriform cortex segmentation task.

count=2
* Improved Road Connectivity by Joint Learning of Orientation and Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Batra_Improved_Road_Connectivity_by_Joint_Learning_of_Orientation_and_Segmentation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Batra_Improved_Road_Connectivity_by_Joint_Learning_of_Orientation_and_Segmentation_CVPR_2019_paper.pdf)]
    * Title: Improved Road Connectivity by Joint Learning of Orientation and Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Anil Batra,  Suriya Singh,  Guan Pang,  Saikat Basu,  C.V. Jawahar,  Manohar Paluri
    * Abstract: Road network extraction from satellite images often produce fragmented road segments leading to road maps unfit for real applications. Pixel-wise classification fails to predict topologically correct and connected road masks due to the absence of connectivity supervision and difficulty in enforcing topological constraints. In this paper, we propose a connectivity task called Orientation Learning, motivated by the human behavior of annotating roads by tracing it at a specific orientation. We also develop a stacked multi-branch convolutional module to effectively utilize the mutual information between orientation learning and segmentation tasks. These contributions ensure that the model predicts topologically correct and connected road masks. We also propose Connectivity Refinement approach to further enhance the estimated road networks. The refinement model is pre-trained to connect and refine the corrupted ground-truth masks and later fine-tuned to enhance the predicted road masks. We demonstrate the advantages of our approach on two diverse road extraction datasets SpaceNet and DeepGlobe. Our approach improves over the state-of-the-art techniques by 9% and 7.5% in road topology metric on SpaceNet and DeepGlobe, respectively.

count=2
* DARNet: Deep Active Ray Network for Building Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Cheng_DARNet_Deep_Active_Ray_Network_for_Building_Segmentation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_DARNet_Deep_Active_Ray_Network_for_Building_Segmentation_CVPR_2019_paper.pdf)]
    * Title: DARNet: Deep Active Ray Network for Building Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Dominic Cheng,  Renjie Liao,  Sanja Fidler,  Raquel Urtasun
    * Abstract: In this paper, we propose a Deep Active Ray Network (DARNet) for automatic building segmentation. Taking an image as input, it first exploits a deep convolutional neural network (CNN) as the backbone to predict energy maps, which are further utilized to construct an energy function. A polygon-based contour is then evolved via minimizing the energy function, of which the minimum defines the final segmentation. Instead of parameterizing the contour using Euclidean coordinates, we adopt polar coordinates, i.e., rays, which not only prevents self-intersection but also simplifies the design of the energy function. Moreover, we propose a loss function that directly encourages the contours to match building boundaries. Our DARNet is trained end-to-end by back-propagating through the energy minimization and the backbone CNN, which makes the CNN adapt to the dynamics of the contour evolution. Experiments on three building instance segmentation datasets demonstrate our DARNet achieves either state-of-the-art or comparable performances to other competitors.

count=2
* Mask Scoring R-CNN
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf)]
    * Title: Mask Scoring R-CNN
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Zhaojin Huang,  Lichao Huang,  Yongchao Gong,  Chang Huang,  Xinggang Wang
    * Abstract: Letting a deep network be aware of the quality of its own predictions is an interesting yet important problem. In the task of instance segmentation, the confidence of instance classification is used as mask quality score in most instance segmentation frameworks. However, the mask quality, quantified as the IoU between the instance mask and its ground truth, is usually not well correlated with classification score. In this paper, we study this problem and propose Mask Scoring R-CNN which contains a network block to learn the quality of the predicted instance masks. The proposed network block takes the instance feature and the corresponding predicted mask together to regress the mask IoU. The mask scoring strategy calibrates the misalignment between mask quality and mask score, and improves instance segmentation performance by prioritizing more accurate mask predictions during COCO AP evaluation. By extensive evaluations on the COCO dataset, Mask Scoring R-CNN brings consistent and noticeable gain with different models and outperforms the state-of-the-art Mask R-CNN. We hope our simple and effective approach will provide a new direction for improving instance segmentation. The source code of our method is available at https://github.com/zjhuang22/maskscoring_rcnn.

count=2
* Object Instance Annotation With Deep Extreme Level Set Evolution
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Object_Instance_Annotation_With_Deep_Extreme_Level_Set_Evolution_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Object_Instance_Annotation_With_Deep_Extreme_Level_Set_Evolution_CVPR_2019_paper.pdf)]
    * Title: Object Instance Annotation With Deep Extreme Level Set Evolution
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Zian Wang,  David Acuna,  Huan Ling,  Amlan Kar,  Sanja Fidler
    * Abstract: In this paper, we tackle the task of interactive object segmentation. We revive the old ideas on level set segmentation which framed object annotation as curve evolution. Carefully designed energy functions ensured that the curve was well aligned with image boundaries, and generally "well behaved". The Level Set Method can handle objects with complex shapes and topological changes such as merging and splitting, thus able to deal with occluded objects and objects with holes. We propose Deep Extreme Level Set Evolution that combines powerful CNN models with level set optimization in an end-to-end fashion. Our method learns to predict evolution parameters conditioned on the image and evolves the predicted initial contour to produce the final result. We make our model interactive by incorporating user clicks on the extreme boundary points, following DEXTR. We show that our approach significantly outperforms DEXTR on the static Cityscapes dataset and the video segmentation benchmark DAVIS, and performs on par on PASCAL and SBD.

count=2
* MPM: Joint Representation of Motion and Position Map for Cell Tracking
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Hayashida_MPM_Joint_Representation_of_Motion_and_Position_Map_for_Cell_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hayashida_MPM_Joint_Representation_of_Motion_and_Position_Map_for_Cell_CVPR_2020_paper.pdf)]
    * Title: MPM: Joint Representation of Motion and Position Map for Cell Tracking
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Junya Hayashida,  Kazuya Nishimura,  Ryoma Bise
    * Abstract: Conventional cell tracking methods detect multiple cells in each frame (detection) and then associate the detection results in successive time-frames (association). Most cell tracking methods perform the association task independently from the detection task. However, there is no guarantee of preserving coherence between these tasks, and lack of coherence may adversely affect tracking performance. In this paper, we propose the Motion and Position Map (MPM) that jointly represents both detection and association for not only migration but also cell division. It guarantees coherence such that if a cell is detected, the corresponding motion flow can always be obtained. It is a simple but powerful method for multi-object tracking in dense environments. We compared the proposed method with current tracking methods under various conditions in real biological images and found that it outperformed the state-of-the-art (+5.2% improvement compared to the second-best).

count=2
* End-to-End 3D Point Cloud Instance Segmentation Without Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_End-to-End_3D_Point_Cloud_Instance_Segmentation_Without_Detection_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_End-to-End_3D_Point_Cloud_Instance_Segmentation_Without_Detection_CVPR_2020_paper.pdf)]
    * Title: End-to-End 3D Point Cloud Instance Segmentation Without Detection
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Haiyong Jiang,  Feilong Yan,  Jianfei Cai,  Jianmin Zheng,  Jun Xiao
    * Abstract: 3D instance segmentation plays a predominant role in environment perception of robotics and augmented reality. Many deep learning based methods have been presented recently for this task. These methods rely on either a detection branch to propose objects or a grouping step to assemble same-instance points. However, detection based methods do not ensure a consistent instance label for each point, while the grouping step requires parameter-tuning and is computationally expensive. In this paper, we introduce a novel framework to enable end-to-end instance segmentation without detection and a separate step of grouping. The core idea is to convert instance segmentation to a candidate assignment problem. At first, a set of instance candidates is sampled. Then we propose an assignment module for candidate assignment and a suppression module to eliminate redundant candidates. A mapping between instance labels and instance candidates is further sought to construct an instance grouping loss for the network training. Experimental results demonstrate that our method is more effective and efficient than previous approaches.

count=2
* PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.pdf)]
    * Title: PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Li Jiang,  Hengshuang Zhao,  Shaoshuai Shi,  Shu Liu,  Chi-Wing Fu,  Jiaya Jia
    * Abstract: Instance segmentation is an important task for scene understanding. Compared to the fully-developed 2D, 3D instance segmentation for point clouds have much room to improve. In this paper, we present PointGroup, a new end-to-end bottom-up architecture, specifically focused on better grouping the points by exploring the void space between objects. We design a two-branch network to extract point features and predict semantic labels and offsets, for shifting each point towards its respective instance centroid. A clustering component is followed to utilize both the original and offset-shifted point coordinate sets, taking advantage of their complementary strength. Further, we formulate the ScoreNet to evaluate the candidate instances, followed by the Non-Maximum Suppression (NMS) to remove duplicates. We conduct extensive experiments on two challenging datasets, ScanNet v2 and S3DIS, on which our method achieves the highest performance, 63.6% and 64.0%, compared to 54.9% and 54.4% achieved by former best solutions in terms of mAP with IoU threshold 0.5.

count=2
* Instance Segmentation of Biological Images Using Harmonic Embeddings
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Kulikov_Instance_Segmentation_of_Biological_Images_Using_Harmonic_Embeddings_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kulikov_Instance_Segmentation_of_Biological_Images_Using_Harmonic_Embeddings_CVPR_2020_paper.pdf)]
    * Title: Instance Segmentation of Biological Images Using Harmonic Embeddings
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Victor Kulikov,  Victor Lempitsky
    * Abstract: We present a new instance segmentation approach tailored to biological images, where instances may correspond to individual cells, organisms or plant parts. Unlike instance segmentation for user photographs or road scenes, in biological data object instances may be particularly densely packed, the appearance variation may be particularly low, the processing power may be restricted, while, on the other hand, the variability of sizes of individual instances may be limited. The proposed approach successfully addresses these peculiarities. Our approach describes each object instance using an expectation of a limited number of sine waves with frequencies and phases adjusted to particular object sizes and densities. At train time, a fully-convolutional network is learned to predict the object embeddings at each pixel using a simple pixelwise regression loss, while at test time the instances are recovered using clustering in the embedding space. In the experiments, we show that our approach outperforms previous embedding-based instance segmentation approaches on a number of biological datasets, achieving state-of-the-art on a popular CVPPP benchmark. This excellent performance is combined with computational efficiency that is needed for deployment to domain specialists. The source code of the approach is available at https://github.com/kulikovv/harmonic .

count=2
* PolyTransform: Deep Polygon Transformer for Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Liang_PolyTransform_Deep_Polygon_Transformer_for_Instance_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liang_PolyTransform_Deep_Polygon_Transformer_for_Instance_Segmentation_CVPR_2020_paper.pdf)]
    * Title: PolyTransform: Deep Polygon Transformer for Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Justin Liang,  Namdar Homayounfar,  Wei-Chiu Ma,  Yuwen Xiong,  Rui Hu,  Raquel Urtasun
    * Abstract: In this paper, we propose PolyTransform, a novel instance segmentation algorithm that produces precise, geometry-preserving masks by combining the strengths of prevailing segmentation approaches and modern polygon-based methods. In particular, we first exploit a segmentation network to generate instance masks. We then convert the masks into a set of polygons that are then fed to a deforming network that transforms the polygons such that they better fit the object boundaries. Our experiments on the challenging Cityscapes dataset show that our PolyTransform significantly improves the performance of the backbone instance segmentation network and ranks 1st on the Cityscapes test-set leaderboard. We also show impressive gains in the interactive annotation setting.

count=2
* Deep Snake for Real-Time Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Peng_Deep_Snake_for_Real-Time_Instance_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Peng_Deep_Snake_for_Real-Time_Instance_Segmentation_CVPR_2020_paper.pdf)]
    * Title: Deep Snake for Real-Time Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Sida Peng,  Wen Jiang,  Huaijin Pi,  Xiuli Li,  Hujun Bao,  Xiaowei Zhou
    * Abstract: This paper introduces a novel contour-based approach named deep snake for real-time instance segmentation. Unlike some recent methods that directly regress the coordinates of the object boundary points from an image, deep snake uses a neural network to iteratively deform an initial contour to match the object boundary, which implements the classic idea of snake algorithms with a learning-based approach. For structured feature learning on the contour, we propose to use circular convolution in deep snake, which better exploits the cycle-graph structure of a contour compared against generic graph convolution. Based on deep snake, we develop a two-stage pipeline for instance segmentation: initial contour proposal and contour deformation, which can handle errors in object localization. Experiments show that the proposed approach achieves competitive performances on the Cityscapes, KINS, SBD and COCO datasets while being efficient for real-time applications with a speed of 32.3 fps for 512 x 512 images on a 1080Ti GPU. The code is available at https://github.com/zju3dv/snake/.

count=2
* Pixel Consensus Voting for Panoptic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Pixel_Consensus_Voting_for_Panoptic_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Pixel_Consensus_Voting_for_Panoptic_Segmentation_CVPR_2020_paper.pdf)]
    * Title: Pixel Consensus Voting for Panoptic Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Haochen Wang,  Ruotian Luo,  Michael Maire,  Greg Shakhnarovich
    * Abstract: The core of our approach, Pixel Consensus Voting, is a framework for instance segmentation based on the generalized Hough transform. Pixels cast discretized, probabilistic votes for the likely regions that contain instance centroids. At the detected peaks that emerge in the voting heatmap, backprojection is applied to collect pixels and produce instance masks. Unlike a sliding window detector that densely enumerates object proposals, our method detects instances as a result of the consensus among pixel-wise votes. We implement vote aggregation and backprojection using native operators of a convolutional neural network. The discretization of centroid voting reduces the training of instance segmentation to pixel labeling, analogous and complementary to FCN-style semantic segmentation, leading to an efficient and unified architecture that jointly models things and stuff. We demonstrate the effectiveness of our pipeline on COCO and Cityscapes Panoptic Segmentation and obtain competitive results. Code will be open-sourced.

count=2
* Fast MSER
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Fast_MSER_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Fast_MSER_CVPR_2020_paper.pdf)]
    * Title: Fast MSER
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Hailiang Xu,  Siqi Xie,  Fan Chen
    * Abstract: Maximally Stable Extremal Regions (MSER) algorithms are based on the component tree and are used to detect invariant regions. OpenCV MSER, the most popular MSER implementation, uses a linked list to associate pixels with ERs. The data-structure of an ER contains the attributes of a head and a tail linked node, which makes OpenCV MSER hard to be performed in parallel using existing parallel component tree strategies. Besides, pixel extraction (i.e. extracting the pixels in MSERs) in OpenCV MSER is very slow. In this paper, we propose two novel MSER algorithms, called Fast MSER V1 and V2. They first divide an image into several spatial partitions, then construct sub-trees and doubly linked lists (for V1) or a labelled image (for V2) on the partitions in parallel. A novel sub-tree merging algorithm is used in V1 to merge the sub-trees into the final tree, and the doubly linked lists are also merged in the process. While V2 merges the sub-trees using an existing merging algorithm. Finally, MSERs are recognized, the pixels in them are extracted through two novel pixel extraction methods taking advantage of the fact that a lot of pixels in parent and child MSERs are duplicated. Both V1 and V2 outperform three open source MSER algorithms (28 and 26 times faster than OpenCV MSER), and reduce the memory of the pixels in MSERs by 78%.

count=2
* Deeply Shape-Guided Cascade for Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_Deeply_Shape-Guided_Cascade_for_Instance_Segmentation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_Deeply_Shape-Guided_Cascade_for_Instance_Segmentation_CVPR_2021_paper.pdf)]
    * Title: Deeply Shape-Guided Cascade for Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Hao Ding, Siyuan Qiao, Alan Yuille, Wei Shen
    * Abstract: The key to a successful cascade architecture for precise instance segmentation is to fully leverage the relationship between bounding box detection and mask segmentation across multiple stages. Although modern instance segmentation cascades achieve leading performance, they mainly make use of a unidirectional relationship, i.e., mask segmentation can benefit from iteratively refined bounding box detection. In this paper, we investigate an alternative direction, i.e., how to take the advantage of precise mask segmentation for bounding box detection in a cascade architecture. We propose a Deeply Shape-guided Cascade (DSC) for instance segmentation, which iteratively imposes the shape guidances extracted from mask prediction at previous stage on bounding box detection at current stage. It forms a bi-directional relationship between the two tasks by introducing three key components: (1) Initial shape guidance: A mask-supervised Region Proposal Network (mPRN) with the ability to generate class-agnostic masks; (2) Explicit shape guidance: A mask-guided region-of-interest (RoI) feature extractor, which employs mask segmentation at previous stage to focus feature extraction at current stage within a region aligned well with the shape of the instance-of-interest rather than a rectangular RoI; (3) Implicit shape guidance: A feature fusion operation which feeds intermediate mask features at previous stage to the bounding box head at current stage. Experimental results show that DSC outperforms the state-of-the-art instance segmentation cascade, Hybrid Task Cascade (HTC), by a large margin and achieves 51.8 box AP and 45.5 mask AP on COCO test-dev. The code is released at: https://github.com/hding2455/DSC.

count=2
* Representation Learning via Global Temporal Alignment and Cycle-Consistency
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Hadji_Representation_Learning_via_Global_Temporal_Alignment_and_Cycle-Consistency_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Hadji_Representation_Learning_via_Global_Temporal_Alignment_and_Cycle-Consistency_CVPR_2021_paper.pdf)]
    * Title: Representation Learning via Global Temporal Alignment and Cycle-Consistency
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Isma Hadji, Konstantinos G. Derpanis, Allan D. Jepson
    * Abstract: We introduce a weakly supervised method for representation learning based on aligning temporal sequences (e.g., videos) of the same process (e.g., human action). The main idea is to use the global temporal ordering of latent correspondences across sequence pairs as a supervisory signal. In particular, we propose a loss based on scoring the optimal sequence alignment to train an embedding network. Our loss is based on a novel probabilistic path finding view of dynamic time warping (DTW) that contains the following three key features: (i) the local path routing decisions are contrastive and differentiable, (ii) pairwise distances are cast as probabilities that are contrastive as well, and (iii) our formulation naturally admits a global cycle consistency loss that verifies correspondences. For evaluation, we consider the tasks of fine-grained action classification, few shot learning, and video synchronization. We report significant performance increases over previous methods. In addition, we report two applications of our temporal alignment framework, namely 3D pose reconstruction and fine-grained audio/visual retrieval.

count=2
* A2-FPN: Attention Aggregation Based Feature Pyramid Network for Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_A2-FPN_Attention_Aggregation_Based_Feature_Pyramid_Network_for_Instance_Segmentation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_A2-FPN_Attention_Aggregation_Based_Feature_Pyramid_Network_for_Instance_Segmentation_CVPR_2021_paper.pdf)]
    * Title: A2-FPN: Attention Aggregation Based Feature Pyramid Network for Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Miao Hu, Yali Li, Lu Fang, Shengjin Wang
    * Abstract: Learning pyramidal feature representations is crucial for recognizing object instances at different scales. Feature Pyramid Network (FPN) is the classic architecture to build a feature pyramid with high-level semantics throughout. However, intrinsic defects in feature extraction and fusion inhibit FPN from further aggregating more discriminative features. In this work, we propose Attention Aggregation based Feature Pyramid Network (A^2-FPN), to improve multi-scale feature learning through attention-guided feature aggregation. In feature extraction, it extracts discriminative features by collecting-distributing multi-level global context features, and mitigates the semantic information loss due to drastically reduced channels. In feature fusion, it aggregates complementary information from adjacent features to generate location-wise reassembly kernels for content-aware sampling, and employs channel-wise reweighting to enhance the semantic consistency before element-wise addition. A^2-FPN shows consistent gains on different instance segmentation frameworks. By replacing FPN with A^2-FPN in Mask R-CNN, our model boosts the performance by 2.1% and 1.6% mask AP when using ResNet-50 and ResNet-101 as backbone, respectively. Moreover, A^2-FPN achieves an improvement of 2.0% and 1.4% mask AP when integrated into the strong baselines such as Cascade Mask R-CNN and Hybrid Task Cascade.

count=2
* ColorRL: Reinforced Coloring for End-to-End Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Tuan_ColorRL_Reinforced_Coloring_for_End-to-End_Instance_Segmentation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Tuan_ColorRL_Reinforced_Coloring_for_End-to-End_Instance_Segmentation_CVPR_2021_paper.pdf)]
    * Title: ColorRL: Reinforced Coloring for End-to-End Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Tran Anh Tuan, Nguyen Tuan Khoa, Tran Minh Quan, Won-Ki Jeong
    * Abstract: Instance segmentation, the task of identifying and separating each individual object of interest in the image, is one of the actively studied research topics in computer vision. Although many feed-forward networks produce high-quality binary segmentation on different types of images, their final result heavily relies on the post-processing step, which separates instances from the binary mask. In comparison, the existing iterative methods extract a single object at a time using discriminative knowledge-based properties (e.g., shapes, boundaries, etc.) without relying on post-processing. However, they do not scale well with a large number of objects. To exploit the advantages of conventional sequential segmentation methods without impairing the scalability, we propose a novel iterative deep reinforcement learning agent that learns how to differentiate multiple objects in parallel. By constructing a relational graph between pixels, we design a reward function that encourages separating pixels of different objects and grouping pixels that belong to the same instance. We demonstrate that the proposed method can efficiently perform instance segmentation of many objects without heavy post-processing.

count=2
* RAMA: A Rapid Multicut Algorithm on GPU
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Abbas_RAMA_A_Rapid_Multicut_Algorithm_on_GPU_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Abbas_RAMA_A_Rapid_Multicut_Algorithm_on_GPU_CVPR_2022_paper.pdf)]
    * Title: RAMA: A Rapid Multicut Algorithm on GPU
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Ahmed Abbas, Paul Swoboda
    * Abstract: We propose a highly parallel primal-dual algorithm for the multicut (a.k.a. correlation clustering) problem, a classical graph clustering problem widely used in machine learning and computer vision. Our algorithm consists of three steps executed recursively: (1) Finding conflicted cycles that correspond to violated inequalities of the underlying multicut relaxation, (2) Performing message passing between the edges and cycles to optimize the Lagrange relaxation coming from the found violated cycles producing reduced costs and (3) Contracting edges with high reduced costs through matrix-matrix multiplications. Our algorithm produces primal solutions and lower bounds that estimate the distance to optimum. We implement our algorithm on GPUs and show resulting one to two orders-of-magnitudes improvements in execution speed without sacrificing solution quality compared to traditional sequential algorithms that run on CPUs. We can solve very large scale benchmark problems with up to O(10^8) variables in a few seconds with small primal-dual gaps. Our code is available at https://github.com/pawelswoboda/RAMA.

count=2
* Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Open-World_Instance_Segmentation_Exploiting_Pseudo_Ground_Truth_From_Learned_Pairwise_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Open-World_Instance_Segmentation_Exploiting_Pseudo_Ground_Truth_From_Learned_Pairwise_CVPR_2022_paper.pdf)]
    * Title: Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Weiyao Wang, Matt Feiszli, Heng Wang, Jitendra Malik, Du Tran
    * Abstract: Open-world instance segmentation is the task of grouping pixels into object instances without any pre-determined taxonomy. This is challenging, as state-of-the-art methods rely on explicit class semantics obtained from large labeled datasets, and out-of-domain evaluation performance drops significantly. Here we propose a novel approach for mask proposals, Generic Grouping Networks (GGNs), constructed without semantic supervision. Our approach combines a local measure of pixel affinity with instance-level mask supervision, producing a training regimen designed to make the model as generic as the data diversity allows. We introduce a method for predicting Pairwise Affinities (PA), a learned local relationship between pairs of pixels. PA generalizes very well to unseen categories. From PA we construct a large set of pseudo-ground-truth instance masks; combined with human-annotated instance masks we train GGNs and significantly outperform the SOTA on open-world instance segmentation on various benchmarks including COCO, LVIS, ADE20K, and UVO.

count=2
* Beyond mAP: Towards Better Evaluation of Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Jena_Beyond_mAP_Towards_Better_Evaluation_of_Instance_Segmentation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Jena_Beyond_mAP_Towards_Better_Evaluation_of_Instance_Segmentation_CVPR_2023_paper.pdf)]
    * Title: Beyond mAP: Towards Better Evaluation of Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Rohit Jena, Lukas Zhornyak, Nehal Doiphode, Pratik Chaudhari, Vivek Buch, James Gee, Jianbo Shi
    * Abstract: Correctness of instance segmentation constitutes counting the number of objects, correctly localizing all predictions and classifying each localized prediction. Average Precision is the de-facto metric used to measure all these constituents of segmentation. However, this metric does not penalize duplicate predictions in the high-recall range, and cannot distinguish instances that are localized correctly but categorized incorrectly. This weakness has inadvertently led to network designs that achieve significant gains in AP but also introduce a large number of false positives. We therefore cannot rely on AP to choose a model that provides an optimal tradeoff between false positives and high recall. To resolve this dilemma, we review alternative metrics in the literature and propose two new measures to explicitly measure the amount of both spatial and categorical duplicate predictions. We also propose a Semantic Sorting and NMS module to remove these duplicates based on a pixel occupancy matching scheme. Experiments show that modern segmentation networks have significant gains in AP, but also contain a considerable amount of duplicates. Our Semantic Sorting and NMS can be added as a plug-and-play module to mitigate hedged predictions and preserve AP.

count=2
* Gasformer: A Transformer-based Architecture for Segmenting Methane Emissions from Livestock in Optical Gas Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/html/Sarker_Gasformer_A_Transformer-based_Architecture_for_Segmenting_Methane_Emissions_from_Livestock_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/papers/Sarker_Gasformer_A_Transformer-based_Architecture_for_Segmenting_Methane_Emissions_from_Livestock_CVPRW_2024_paper.pdf)]
    * Title: Gasformer: A Transformer-based Architecture for Segmenting Methane Emissions from Livestock in Optical Gas Imaging
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Toqi Tahamid Sarker, Mohamed G Embaby, Khaled R Ahmed, Amer Abughazaleh
    * Abstract: Methane emissions from livestock particularly cattle significantly contribute to climate change. Effective methane emission mitigation strategies are crucial as the global population and demand for livestock products increase. We introduce Gasformer a novel semantic segmentation architecture for detecting low-flow rate methane emissions from livestock and controlled release experiments using optical gas imaging. We present two unique datasets captured with a FLIR GF77 OGI camera. Gasformer leverages a Mix Vision Transformer encoder and a Light-Ham decoder to generate multi-scale features and refine segmentation maps. Gasformer outperforms other state-of-the-art models on both datasets demonstrating its effectiveness in detecting and segmenting methane plumes in controlled and real-world scenarios. On the livestock dataset Gasformer achieves mIoU of 88.56% surpassing other state-of-the-art models. Materials are available at: github.com/toqitahamid/Gasformer.

count=2
* Enhanced Rotation-Equivariant U-Net for Nuclear Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/CVMI/Chidester_Enhanced_Rotation-Equivariant_U-Net_for_Nuclear_Segmentation_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVMI/Chidester_Enhanced_Rotation-Equivariant_U-Net_for_Nuclear_Segmentation_CVPRW_2019_paper.pdf)]
    * Title: Enhanced Rotation-Equivariant U-Net for Nuclear Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Benjamin Chidester,  That-Vinh Ton,  Minh-Triet Tran,  Jian Ma,  Minh N. Do
    * Abstract: Despite recent advances in deep learning, the crucial task of nuclear segmentation for computational pathology remains challenging. Recently, deep learning, and specifically U-Nets, have shown significant improvements for this task, but there is still room for improvement by further enhancing the design and training of U-Nets for nuclear segmentation. Specifically, we consider enforcing rotation equivariance in the network, the placement of residual blocks, and applying novel data augmentation designed specifically for histopathology images, and show the relative improvement and merit of each. Incorporating all of these enhancements in the design and training of a U-Net yields significantly improved segmentation results while still maintaining a speed of inference that is sufficient for real-world applications, in particular, analyzing whole-slide images (WSIs). Code for our enhanced U-Net is available at https://github.com/thatvinhton/G-U-Net.

count=2
* Leaf Segmentation by Functional Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/CVPPP/Chen_Leaf_Segmentation_by_Functional_Modeling_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVPPP/Chen_Leaf_Segmentation_by_Functional_Modeling_CVPRW_2019_paper.pdf)]
    * Title: Leaf Segmentation by Functional Modeling
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yuhao Chen,  Sriram Baireddy,  Enyu Cai,  Changye Yang,  Edward J. Delp
    * Abstract: The use of Unmanned Aerial Vehicles (UAVs) is a recent trend in field based plant phenotyping data collection. However, UAVs often provide low spatial resolution images when flying at high altitudes. This can be an issue when extracting individual leaves from these images. Leaf segmentation is even more challenging because of densely overlapping leaves. Segmentation of leaf instances in the UAV images can be used to measure various phenotypic traits such as leaf length, maximum leaf width, and leaf area index. Successful leaf segmentation accurately detects leaf edges. Popular deep neural network approaches have loss functions that do not consider the spatial accuracy of the segmentation near an object's edge. This paper proposes a shape-based leaf segmentation method that segments leaves using continuous functions and produces precise contours for the leaf edges. Experimental results prove the feasibility of the method and demonstrate better performance than the Mask R-CNN.

count=2
* A Semi-Supervised Approach for Ice-Water Classification Using Dual-Polarization SAR Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Li_A_Semi-Supervised_Approach_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Li_A_Semi-Supervised_Approach_2015_CVPR_paper.pdf)]
    * Title: A Semi-Supervised Approach for Ice-Water Classification Using Dual-Polarization SAR Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Fan Li, David A. Clausi, Lei Wang, Linlin Xu
    * Abstract: The daily interpretation of SAR sea ice imagery is very important for ship navigation and climate monitoring. Currently, the interpretation is still performed manually by ice analysts due to the complexity of data and the difficulty of creating fine-level ground truth. To overcome these problems, a semi-supervised approach for ice-water classification based on self-training is presented. The proposed algorithm integrates the spatial context model, region merging, and the self-training technique into a single framework. The backscatter intensity, texture, and edge strength features are incorporated in a CRF model using multi-modality Gaussian model as its unary classifier. Region merging is used to build a hierarchical data-adaptive structure to make the inference more efficient. Self-training is concatenated with region merging, so that the spatial location information of the original training samples can be used. Our algorithm has been tested on a large-scale RADARSAT-2 dual-polarization dataset over the Beaufort and Chukchi sea, and the classification results are significantly better than the supervised methods without self-training.

count=2
* Weakly Supervised Learning of Image Partitioning Using Decision Trees with Structured Split Criteria
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Straehle_Weakly_Supervised_Learning_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Straehle_Weakly_Supervised_Learning_2013_ICCV_paper.pdf)]
    * Title: Weakly Supervised Learning of Image Partitioning Using Decision Trees with Structured Split Criteria
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Christoph Straehle, Ullrich Koethe, Fred A. Hamprecht
    * Abstract: We propose a scheme that allows to partition an image into a previously unknown number of segments, using only minimal supervision in terms of a few must-link and cannotlink annotations. We make no use of regional data terms, learning instead what constitutes a likely boundary between segments. Since boundaries are only implicitly specified through cannot-link constraints, this is a hard and nonconvex latent variable problem. We address this problem in a greedy fashion using a randomized decision tree on features associated with interpixel edges. We use a structured purity criterion during tree construction and also show how a backtracking strategy can be used to prevent the greedy search from ending up in poor local optima. The proposed strategy is compared with prior art on natural images.

count=2
* Robust Image Segmentation Using Contour-Guided Color Palettes
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Fu_Robust_Image_Segmentation_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Fu_Robust_Image_Segmentation_ICCV_2015_paper.pdf)]
    * Title: Robust Image Segmentation Using Contour-Guided Color Palettes
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Xiang Fu, Chien-Yi Wang, Chen Chen, Changhu Wang, C.-C. Jay Kuo
    * Abstract: The contour-guided color palette (CCP) is proposed for robust image segmentation. It efficiently integrates contour and color cues of an image. To find representative colors of an image, color samples along long contours between regions, similar in spirit to machine learning methodology that focus on samples near decision boundaries, are collected followed by the mean-shift (MS) algorithm in the sampled color space to achieve an image-dependent color palette. This color palette provides a preliminary segmentation in the spatial domain, which is further fine-tuned by post-processing techniques such as leakage avoidance, fake boundary removal, and small region mergence. Segmentation performances of CCP and MS are compared and analyzed. While CCP offers an acceptable standalone segmentation result, it can be further integrated into the framework of layered spectral segmentation to produce a more robust segmentation. The superior performance of CCP-based segmentation algorithm is demonstrated by experiments on the Berkeley Segmentation Dataset.

count=2
* The Middle Child Problem: Revisiting Parametric Min-Cut and Seeds for Object Proposals
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Humayun_The_Middle_Child_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Humayun_The_Middle_Child_ICCV_2015_paper.pdf)]
    * Title: The Middle Child Problem: Revisiting Parametric Min-Cut and Seeds for Object Proposals
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Ahmad Humayun, Fuxin Li, James M. Rehg
    * Abstract: Object proposals have recently fueled the progress in detection performance. These proposals aim to provide category-agnostic localizations for all objects in an image. One way to generate proposals is to perform parametric min-cuts over seed locations. This paper demonstrates that standard parametric-cut models are ineffective in obtaining medium-sized objects, which we refer to as the middle child problem. We propose a new energy minimization framework incorporating geodesic distances between segments which solves this problem. In addition, we introduce a new superpixel merging algorithm which can generate a small set of seeds that reliably cover a large number of objects of all sizes. We call our method POISE--- "Proposals for Objects from Improved Seeds and Energies." POISE enables parametric min-cuts to reach their full potential. On PASCAL VOC it generates 2,640 segments with an average overlap of 0.81, whereas the closest competing methods require more than 4,200 proposals to reach the same accuracy. We show detailed quantitative comparisons against 5 state-of-the-art methods on PASCAL VOC and Microsoft COCO segmentation challenges.

count=2
* Geometric Mining: Scaling Geometric Hashing to Large Datasets
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w27/html/Gilbert_Geometric_Mining_Scaling_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w27/papers/Gilbert_Geometric_Mining_Scaling_ICCV_2015_paper.pdf)]
    * Title: Geometric Mining: Scaling Geometric Hashing to Large Datasets
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Andrew Gilbert, Richard Bowden
    * Abstract: It is known that relative feature location is important in representing objects, but assumptions that make learning tractable often simplify how structure is encoded e.g. spatial pooling or star models. For example, techniques such as spatial pyramid matching (SPM), in-conjunction with machine learning techniques perform well. However, there are limitations to such spatial encoding schemes which discard important information about the layout of features. In contrast, we propose to use the object itself to choose the basis of the features in an object centric approach. In doing so we return to the early work of geometric hashing but demonstrate how such approaches can be scaled-up to modern day object detection challenges in terms of both the number of examples and their variability. We apply a two stage process; initially filtering background features to localise the objects and then hashing the remaining pairwise features in an affine invariant model. During learning, we identify class-wise key feature predictors. We validate our detection and classification of objects on the PASCAL VOC'07 and '11 and CarDb datasets and compare with state of the art detectors and classifiers. Importantly we demonstrate how structure in features can be efficiently identified and how its inclusion can increase performance. This feature centric learning technique allows us to localise objects even without object annotation during training and the resultant segmentation provides accurate state of the art object localization, without the need for annotations.

count=2
* TorontoCity: Seeing the World With a Million Eyes
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Wang_TorontoCity_Seeing_the_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_TorontoCity_Seeing_the_ICCV_2017_paper.pdf)]
    * Title: TorontoCity: Seeing the World With a Million Eyes
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Shenlong Wang, Min Bai, Gellert Mattyus, Hang Chu, Wenjie Luo, Bin Yang, Justin Liang, Joel Cheverie, Sanja Fidler, Raquel Urtasun
    * Abstract: In this paper we introduce the TorontoCity benchmark, which covers the full greater Toronto area (GTA) with 712.5km2 of land, 8439km of road and around 400, 000 buildings. Our benchmark provides different perspectives of the world captured from airplanes, drones and cars driving around the city. Manually labeling such a large scale dataset is infeasible. Instead, we propose to utilize different sources of high-precision maps to create our ground truth. Towards this goal, we develop algorithms that allow us to align all data sources with the maps while requiring minimal human supervision. We have designed a wide variety of tasks including building height estimation (reconstruction), road centerline and curb extraction, building instance segmentation, building contour extraction (reorganization), semantic labeling and scene type classification (recognition). Our pilot study shows that most of these tasks are still difficult for modern convolutional neural networks.

count=2
* Temporal Action Detection With Structured Segment Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Zhao_Temporal_Action_Detection_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhao_Temporal_Action_Detection_ICCV_2017_paper.pdf)]
    * Title: Temporal Action Detection With Structured Segment Networks
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Yue Zhao, Yuanjun Xiong, Limin Wang, Zhirong Wu, Xiaoou Tang, Dahua Lin
    * Abstract: Detecting actions in untrimmed videos is an important yet challenging task. In this paper, we present the structured segment network (SSN), a novel framework which models the temporal structure of each action instance via a structured temporal pyramid. On top of the pyramid, we further introduce a decomposed discriminative model comprising two classifiers, respectively for classifying actions and determining completeness. This allows the framework to effectively distinguish positive proposals from background or incomplete ones, thus leading to both accurate recognition and localization. These components are integrated into a unified network that can be efficiently trained in an end-to-end fashion. Additionally, a simple yet effective temporal action proposal scheme, dubbed temporal actionness grouping (TAG) is devised to generate high quality action proposals. On two challenging benchmarks, THUMOS'14 and ActivityNet, our method remarkably outperforms previous state-of-the-art methods, demonstrating superior accuracy and strong adaptivity in handling actions with various temporal structures.

count=2
* Count-ception: Counting by Fully Convolutional Redundant Counting
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Cohen_Count-ception_Counting_by_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w1/Cohen_Count-ception_Counting_by_ICCV_2017_paper.pdf)]
    * Title: Count-ception: Counting by Fully Convolutional Redundant Counting
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Joseph Paul Cohen, Genevieve Boucher, Craig A. Glastonbury, Henry Z. Lo, Yoshua Bengio
    * Abstract: Counting objects in digital images is a process that should be replaced by machines. This tedious task is time consuming and prone to errors due to fatigue of human annotators. The goal is to have a system that takes as input an image and returns a count of the objects inside and justification for the prediction in the form of object localization. We repose a problem, originally posed by Lempitsky and Zisserman, to instead predict a count map which contains redundant counts based on the receptive field of a smaller regression network. The regression network predicts a count of the objects that exist inside this frame. By processing the image in a fully convolutional way each pixel is going to be accounted for some number of times, the number of windows which include it, which is the size of each window, (i.e., 32x32 = 1024). To recover the true count take the average over the redundant predictions. Our contribution is redundant counting instead of predicting a density map in order to average over errors. We also propose a novel deep neural network architecture adapted from the Inception family of networks called the Count-ception network. Together our approach results in a 20% relative improvement (2.9 to 2.3 MAE) over the state of the art method by Xie, Noble, and Zisserman in 2016.

count=2
* SSAP: Single-Shot Instance Segmentation With Affinity Pyramid
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Gao_SSAP_Single-Shot_Instance_Segmentation_With_Affinity_Pyramid_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gao_SSAP_Single-Shot_Instance_Segmentation_With_Affinity_Pyramid_ICCV_2019_paper.pdf)]
    * Title: SSAP: Single-Shot Instance Segmentation With Affinity Pyramid
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Naiyu Gao,  Yanhu Shan,  Yupei Wang,  Xin Zhao,  Yinan Yu,  Ming Yang,  Kaiqi Huang
    * Abstract: Recently, proposal-free instance segmentation has received increasing attention due to its concise and efficient pipeline. Generally, proposal-free methods generate instance-agnostic semantic segmentation labels and instance-aware features to group pixels into different object instances. However, previous methods mostly employ separate modules for these two sub-tasks and require multiple passes for inference. We argue that treating these two sub-tasks separately is suboptimal. In fact, employing multiple separate modules significantly reduces the potential for application. The mutual benefits between the two complementary sub-tasks are also unexplored. To this end, this work proposes a single-shot proposal-free instance segmentation method that requires only one single pass for prediction. Our method is based on a pixel-pair affinity pyramid, which computes the probability that two pixels belong to the same instance in a hierarchical manner. The affinity pyramid can also be jointly learned with the semantic class labeling and achieve mutual benefits. Moreover, incorporating with the learned affinity pyramid, a novel cascaded graph partition module is presented to sequentially generate instances from coarse to fine. Unlike previous time-consuming graph partition methods, this module achieves 5x speedup and 9% relative improvement on Average-Precision (AP). Our approach achieves new state of the art on the challenging Cityscapes dataset.

count=2
* ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Kuo_ShapeMask_Learning_to_Segment_Novel_Objects_by_Refining_Shape_Priors_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kuo_ShapeMask_Learning_to_Segment_Novel_Objects_by_Refining_Shape_Priors_ICCV_2019_paper.pdf)]
    * Title: ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Weicheng Kuo,  Anelia Angelova,  Jitendra Malik,  Tsung-Yi Lin
    * Abstract: Instance segmentation aims to detect and segment individual objects in a scene. Most existing methods rely on precise mask annotations of every category. However, it is difficult and costly to segment objects in novel categories because a large number of mask annotations is required. We introduce ShapeMask, which learns the intermediate concept of object shape to address the problem of generalization in instance segmentation to novel categories. ShapeMask starts with a bounding box detection and gradually refines it by first estimating the shape of the detected object through a collection of shape priors. Next, ShapeMask refines the coarse shape into an instance level mask by learning instance embeddings. The shape priors provide a strong cue for object-like prediction, and the instance embeddings model the instance specific appearance information. ShapeMask significantly outperforms the state-of-the-art by 6.4 and 3.8 AP when learning across categories, and obtains competitive performance in the fully supervised setting. It is also robust to inaccurate detections, decreased model capacity, and small training data. Moreover, it runs efficiently with 150ms inference time on a GPU and trains within 11 hours on TPUs. With a larger backbone model, ShapeMask increases the gap with state-of-the-art to 9.4 and 6.2 AP across categories. Code will be publicly available at: https://sites.google.com/view/shapemask/home.

count=2
* Multi-Level Bottom-Top and Top-Bottom Feature Fusion for Crowd Counting
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Sindagi_Multi-Level_Bottom-Top_and_Top-Bottom_Feature_Fusion_for_Crowd_Counting_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sindagi_Multi-Level_Bottom-Top_and_Top-Bottom_Feature_Fusion_for_Crowd_Counting_ICCV_2019_paper.pdf)]
    * Title: Multi-Level Bottom-Top and Top-Bottom Feature Fusion for Crowd Counting
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Vishwanath A. Sindagi,  Vishal M. Patel
    * Abstract: Crowd counting presents enormous challenges in the form of large variation in scales within images and across the dataset. These issues are further exacerbated in highly congested scenes. Approaches based on straightforward fusion of multi-scale features from a deep network seem to be obvious solutions to this problem. However, these fusion approaches do not yield significant improvements in the case of crowd counting in congested scenes. This is usually due to their limited abilities in effectively combining the multi-scale features for problems like crowd counting. To overcome this, we focus on how to efficiently leverage information present in different layers of the network. Specifically, we present a network that involves: (i) a multi-level bottom-top and top-bottom fusion (MBTTBF) method to combine information from shallower to deeper layers and vice versa at multiple levels, (ii) scale complementary feature extraction blocks (SCFB) involving cross-scale residual functions to explicitly enable flow of complementary features from adjacent conv layers along the fusion paths. Furthermore, in order to increase the effectiveness of the multi-scale fusion, we employ a principled way of generating scale-aware ground-truth density maps for training. Experiments conducted on three datasets that contain highly congested scenes (ShanghaiTech, UCF_CC_50, and UCF-QNRF) demonstrate that the proposed method is able to outperform several recent methods in all the datasets

count=2
* Social Fabric: Tubelet Compositions for Video Relation Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Social_Fabric_Tubelet_Compositions_for_Video_Relation_Detection_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Social_Fabric_Tubelet_Compositions_for_Video_Relation_Detection_ICCV_2021_paper.pdf)]
    * Title: Social Fabric: Tubelet Compositions for Video Relation Detection
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Shuo Chen, Zenglin Shi, Pascal Mettes, Cees G. M. Snoek
    * Abstract: This paper strives to classify and detect the relationship between object tubelets appearing within a video as a <subject-predicate-object> triplet. Where existing works treat object proposals or tubelets as single entities and model their relations a posteriori, we propose to classify and detect predicates for pairs of object tubelets a priori. We also propose Social Fabric: an encoding that represents a pair of object tubelets as a composition of interaction primitives. These primitives are learned over all relations, resulting in a compact representation able to localize and classify relations from the pool of co-occurring object tubelets across all timespans in a video. The encoding enables our two-stage network. In the first stage, we train Social Fabric to suggest proposals that are likely interacting. We use the Social Fabric in the second stage to simultaneously fine-tune and predict predicate labels for the tubelets. Experiments demonstrate the benefit of early video relation modeling, our encoding and the two-stage architecture, leading to a new state-of-the-art on two benchmarks. We also show how the encoding enables query-by-primitive-example to search for spatio-temporal video relations. Code: https://github.com/shanshuo/Social-Fabric.

count=2
* Robust Trust Region for Weakly Supervised Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Marin_Robust_Trust_Region_for_Weakly_Supervised_Segmentation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Marin_Robust_Trust_Region_for_Weakly_Supervised_Segmentation_ICCV_2021_paper.pdf)]
    * Title: Robust Trust Region for Weakly Supervised Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Dmitrii Marin, Yuri Boykov
    * Abstract: Acquisition of training data for the standard semantic segmentation is expensive if requiring that each pixel is labeled. Yet, current methods significantly deteriorate in weakly supervised settings, e.g. where a fraction of pixels is labeled or when only image-level tags are available. It has been shown that regularized losses---originally developed for unsupervised low-level segmentation and representing geometric priors on pixel labels---can considerably improve the quality of weakly supervised training. However, many common priors require optimization stronger than gradient descent. Thus, such regularizers have limited applicability in deep learning. We propose a new robust trust region approach for regularized losses improving the state-of-the-art results. Our approach can be seen as a higher-order generalization of the classic chain rule. It allows neural network optimization to use strong low-level solvers for the corresponding regularizers, including discrete ones.

count=2
* Generic Event Boundary Detection: A Benchmark for Event Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Shou_Generic_Event_Boundary_Detection_A_Benchmark_for_Event_Segmentation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Shou_Generic_Event_Boundary_Detection_A_Benchmark_for_Event_Segmentation_ICCV_2021_paper.pdf)]
    * Title: Generic Event Boundary Detection: A Benchmark for Event Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Mike Zheng Shou, Stan Weixian Lei, Weiyao Wang, Deepti Ghadiyaram, Matt Feiszli
    * Abstract: This paper presents a novel task together with a new benchmark for detecting generic, taxonomy-free event boundaries that segment a whole video into chunks. Conventional work in temporal video segmentation and action detection focuses on localizing pre-defined action categories and thus does not scale to generic videos. Cognitive Science has known since last century that humans consistently segment videos into meaningful temporal chunks. This segmentation happens naturally, without pre-defined event categories and without being explicitly asked to do so. Here, we repeat these cognitive experiments on mainstream CV datasets; with our novel annotation guideline which addresses the complexities of taxonomy-free event boundary annotation, we introduce the task of Generic Event Boundary Detection (GEBD) and the new benchmark Kinetics-GEBD. We view GEBD as an important stepping stone towards understanding the video as a whole, and believe it has been previously neglected due to a lack of proper task definition and annotations. Through experiment and human study we demonstrate the value of the annotations. Further, we benchmark supervised and un-supervised GEBD approaches on the TAPOS dataset and our Kinetics-GEBD. We release our annotations and baseline codes at CVPR'21 LOVEU Challenge: https://sites.google.com/view/loveucvpr21.

count=2
* ALBRT: Cellular Composition Prediction in Routine Histology Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/html/Dawood_ALBRT_Cellular_Composition_Prediction_in_Routine_Histology_Images_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/papers/Dawood_ALBRT_Cellular_Composition_Prediction_in_Routine_Histology_Images_ICCVW_2021_paper.pdf)]
    * Title: ALBRT: Cellular Composition Prediction in Routine Histology Images
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Muhammad Dawood, Kim Branson, Nasir M. Rajpoot, Fayyaz Minhas
    * Abstract: Cellular composition prediction, i.e., predicting the presence and counts of different types of cells in the tumor microenvironment from a digitized image of a Hematoxylin and Eosin (H&E) stained tissue section can be used for various tasks in computational pathology such as the analysis of cellular topology and interactions, subtype prediction, survival analysis, etc. In this work, we propose an image-based cellular composition predictor (ALBRT) which can accurately predict the presence and counts of different types of cells in a given image patch. ALBRT, by its contrastive-learning inspired design, learns a compact and rotation-invariant feature representation that is then used for cellular composition prediction of different cell types. It offers significant improvement over existing state-of-the-art approaches for cell classification and counting. The patch-level feature representation learned by ALBRT is transferrable for cellular composition analysis over novel datasets and can also be utilized for downstream prediction tasks in CPath as well. The code and the inference webserver for the proposed method are available at the URL: https://github.com/engrodawood/ALBRT.

count=2
* Class-Agnostic Segmentation Loss and Its Application to Salient Object Detection and Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/html/Sharma_Class-Agnostic_Segmentation_Loss_and_Its_Application_to_Salient_Object_Detection_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/papers/Sharma_Class-Agnostic_Segmentation_Loss_and_Its_Application_to_Salient_Object_Detection_ICCVW_2021_paper.pdf)]
    * Title: Class-Agnostic Segmentation Loss and Its Application to Salient Object Detection and Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Angira Sharma, Naeemullah Khan, Muhammad Mubashar, Ganesh Sundaramoorthi, Philip Torr
    * Abstract: In this paper we present a novel loss function, called class-agnostic segmentation (CAS) loss. With CAS loss the class descriptors are learned during training of the network. We don't require to define the label of a class a-priori, rather the CAS loss clusters regions with similar appearance together in a weakly-supervised manner. Furthermore, we show that the CAS loss function is sparse, bounded, and robust to class-imbalance. We first apply our CAS loss function with fully-convolutional ResNet101 and DeepLab-v3 architectures to the binary segmentation problem of salient object detection. We investigate the performance against the state-of-the-art methods in two settings of low and high-fidelity training data on seven salient object detection datasets. For low-fidelity training data (incorrect class label) class-agnostic segmentation loss outperforms the state-of-the-art methods on salient object detection datasets by staggering margins of around 50%. For high-fidelity training data (correct class labels) class-agnostic segmentation models perform as good as the state-of-the-art approaches while beating the state-of-the-art methods on most datasets. In order to show the utility of the loss function across different domains we then also test on general segmentation dataset, where class-agnostic segmentation loss outperforms competing losses by huge margins.

count=2
* Fine-Grain Prediction of Strawberry Freshness Using Subsurface Scattering
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LFFAI/html/Klotz_Fine-Grain_Prediction_of_Strawberry_Freshness_Using_Subsurface_Scattering_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/LFFAI/papers/Klotz_Fine-Grain_Prediction_of_Strawberry_Freshness_Using_Subsurface_Scattering_ICCVW_2021_paper.pdf)]
    * Title: Fine-Grain Prediction of Strawberry Freshness Using Subsurface Scattering
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jeremy Klotz, Vijay Rengarajan, Aswin C. Sankaranarayanan
    * Abstract: Predicting fruit freshness before any visible decay is invaluable in the food distribution chain, spanning producers, retailers, and consumers. In this work, we leverage subsurface scattering signatures associated with strawberry tissue to perform long-term edibility predictions. Specifically, we implement various active illumination techniques with a projector-camera system to measure a strawberry's subsurface scattering and predict the time when it is likely to be inedible. We propose a learning-based approach with captures under structured illumination to perform this prediction. We study the efficacy of our method by capturing a dataset of strawberries decaying naturally over time.

count=2
* Video Action Segmentation via Contextually Refined Temporal Keypoints
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Video_Action_Segmentation_via_Contextually_Refined_Temporal_Keypoints_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Video_Action_Segmentation_via_Contextually_Refined_Temporal_Keypoints_ICCV_2023_paper.pdf)]
    * Title: Video Action Segmentation via Contextually Refined Temporal Keypoints
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Borui Jiang, Yang Jin, Zhentao Tan, Yadong Mu
    * Abstract: Video action segmentation refers to the task of densely casting each video frame or short segment in an untrimmed video into some pre-specified action categories. Although recent years have witnessed a great promise in the development of action segmentation techniques.A large body of existing methods still rely on frame-wise segmentation, which tends to render fragmentary results (i.e., over-segmentation).To effectively address above issues, we here propose a video action segmentation model that implements the novel idea of Refined Temporal Keypoints (RTK) for overcoming caveats of existing methods.To act effectively, the proposed model initially seeks for high-quality, sparse temporal keypoints by extracting non-local cues from the video, rather than conducting frame-wise classification as in many competing methods.Afterwards, large improvements over the inital temporal keypoints are pin-pointed as contributions by further refining and re-assembling operations. In specific, we develop a graph matching module that aggregates structural information between different temporal keypoints by learning the corresponding relationship of the temporal source graphs and the annotated target graphs. The initial temporal keypoints are refined by the encoded structural information reusing the graph matching module.A few set of prior rules are harnessed for post-processing and re-assembling all temporal keypoints.The remaining temporal keypoiting going through all refinement are used to generate the final action segmentation results.We perform experiments on three popular datasets: 50salads, GTEA and Breakfast, and our methods significantly outperforms the current methods, particularly achieves the state-of-the-art F1@50 scores of 83.4%, 79.5%, and 60.5% on three datasets, respectively.

count=2
* ACTIS: Improving Data Efficiency by Leveraging Semi-Supervised Augmentation Consistency Training for Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/BIC/html/Rumberger_ACTIS_Improving_Data_Efficiency_by_Leveraging_Semi-Supervised_Augmentation_Consistency_Training_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/BIC/papers/Rumberger_ACTIS_Improving_Data_Efficiency_by_Leveraging_Semi-Supervised_Augmentation_Consistency_Training_ICCVW_2023_paper.pdf)]
    * Title: ACTIS: Improving Data Efficiency by Leveraging Semi-Supervised Augmentation Consistency Training for Instance Segmentation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Josef Lorenz Rumberger, Jannik Franzen, Peter Hirsch, Jan-Philipp Albrecht, Dagmar Kainmueller
    * Abstract: Segmenting objects like cells or nuclei in biomedical microscopy data is a standard task required for many downstream analyses. However, existing pre-trained models are continuously challenged by ever-evolving experimental setups and imaging platforms. On the other hand, training new models still requires a considerable number of annotated samples, rendering it infeasible for small to mid-sized experiments. To address this challenge, we propose a semi-supervised learning approach for instance segmentation that leverages a small number of annotated samples together with a larger number of unannotated samples. Our pipeline, Augmentation Consistency Training for Instance Segmentation (ACTIS), incorporates methods from consistency regularization and entropy minimization. In addition, we introduce a robust confidence-based loss masking scheme which we find empirically to work well on highly imbalanced class frequencies. We show that our model can surpass the performance of supervised models trained on more than twice as much annotated data. It achieves state-of-the-art results on three benchmark datasets in the biomedical domain, demonstrating its effectiveness for semi-supervised instance segmentation. Code: github.com/Kainmueller-Lab/ACTIS

count=2
* AI on the Bog: Monitoring and Evaluating Cranberry Crop Risk
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Akiva_AI_on_the_Bog_Monitoring_and_Evaluating_Cranberry_Crop_Risk_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Akiva_AI_on_the_Bog_Monitoring_and_Evaluating_Cranberry_Crop_Risk_WACV_2021_paper.pdf)]
    * Title: AI on the Bog: Monitoring and Evaluating Cranberry Crop Risk
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Peri Akiva, Benjamin Planche, Aditi Roy, Kristin Dana, Peter Oudemans, Michael Mars
    * Abstract: Machine vision for precision agriculture has attracted considerable research interest in recent years. The goal of this paper is to develop an end-end cranberry health monitoring system to enable and support real time cranberry over-heating assessment to facilitate informed decisions that may sustain the economic viability of the farm. Toward this goal, we propose two main deep learning-based modules for: 1) cranberry fruit segmentation to delineate the exact fruit regions in the cranberry field image that are exposed to sun, 2) prediction of cloud coverage conditions to estimate the inner temperature of exposed cranberries We develop drone-based field data and ground-based sky data collection systems to collect video imagery at multiple time points for use in crop health analysis. Extensive evaluation on the data set shows that it is possible to predict exposed fruit's inner temperature with high accuracy (0.02% MAPE) when irradiance is predicted with 5.59-19.84% MAPE in the 5-20 minutes time horizon. With 62.54% mIoU for segmentation and 13.46 MAE for counting accuracies in exposed fruit identification, this system is capable of giving informed feedback to growers to take precautionary action (e.g., irrigation) in identified crop field regions with higher risk of sunburn in the near future. Though this novel system is applied for cranberry health monitoring, it represents a pioneering step forward in efficiency for farming and is useful in precision agriculture beyond the problem of cranberry overheating.

count=2
* The Devil Is in the Boundary: Exploiting Boundary Representation for Basis-Based Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Kim_The_Devil_Is_in_the_Boundary_Exploiting_Boundary_Representation_for_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Kim_The_Devil_Is_in_the_Boundary_Exploiting_Boundary_Representation_for_WACV_2021_paper.pdf)]
    * Title: The Devil Is in the Boundary: Exploiting Boundary Representation for Basis-Based Instance Segmentation
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Myungchul Kim, Sanghyun Woo, Dahun Kim, In So Kweon
    * Abstract: Pursuing a more coherent scene understanding towards real-time vision applications, single-stage instance segmentation has recently gained popularity, achieving a simpler and more efficient design than its two-stage counterparts. Besides, its global mask representation often leads to superior accuracy to the two-stage Mask R-CNN which has been dominant thus far. Despite the promising advances in single-stage methods, finer delineation of instance boundaries still remains unexcavated. Indeed, boundary information provides a strong shape representation that can operate in synergy with the fully-convolutional mask features of the single-stage segmented. In this work, we propose Boundary Basis based Instance Segmentation(B2Inst) to learn a global boundary representation that can complement existing global-mask-based methods that are often lacking high-frequency details. Besides, we devise a unified quality measure of both mask and boundary and introduce a network block that learns to score the per-instance predictions of itself. When applied to the strongest baselines in single-stage instance segmentation, our B2Inst leads to consistent improvements and accurately parse out the instance boundaries in a scene. Regardless of being single-stage or two-stage frameworks, we outperform the existing state-of-the-art methods on the COCO dataset with the same ResNet-50 and ResNet-101 backbones.

count=2
* Solving Random Systems of Quadratic Equations via Truncated Generalized Gradient Flow
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2016/hash/5b8add2a5d98b1a652ea7fd72d942dac-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2016/file/5b8add2a5d98b1a652ea7fd72d942dac-Paper.pdf)]
    * Title: Solving Random Systems of Quadratic Equations via Truncated Generalized Gradient Flow
    * Publisher: NeurIPS
    * Publication Date: `2016`
    * Authors: Gang Wang, Georgios Giannakis
    * Abstract: This paper puts forth a novel algorithm, termed \emph{truncated generalized gradient flow} (TGGF), to solve for $\bm{x}\in\mathbb{R}^n/\mathbb{C}^n$ a system of $m$ quadratic equations $y_i=|\langle\bm{a}_i,\bm{x}\rangle|^2$, $i=1,2,\ldots,m$, which even for $\left\{\bm{a}_i\in\mathbb{R}^n/\mathbb{C}^n\right\}_{i=1}^m$ random is known to be \emph{NP-hard} in general. We prove that as soon as the number of equations $m$ is on the order of the number of unknowns $n$, TGGF recovers the solution exactly (up to a global unimodular constant) with high probability and complexity growing linearly with the time required to read the data $\left\{\left(\bm{a}_i;\,y_i\right)\right\}_{i=1}^m$. Specifically, TGGF proceeds in two stages: s1) A novel \emph{orthogonality-promoting} initialization that is obtained with simple power iterations; and, s2) a refinement of the initial estimate by successive updates of scalable \emph{truncated generalized gradient iterations}. The former is in sharp contrast to the existing spectral initializations, while the latter handles the rather challenging nonconvex and nonsmooth \emph{amplitude-based} cost function. Numerical tests demonstrate that: i) The novel orthogonality-promoting initialization method returns more accurate and robust estimates relative to its spectral counterparts; and ii) even with the same initialization, our refinement/truncation outperforms Wirtinger-based alternatives, all corroborating the superior performance of TGGF over state-of-the-art algorithms.

count=2
* Deep Variational Instance Segmentation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/3341f6f048384ec73a7ba2e77d2db48b-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/3341f6f048384ec73a7ba2e77d2db48b-Paper.pdf)]
    * Title: Deep Variational Instance Segmentation
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Jialin Yuan, Chao Chen, Fuxin Li
    * Abstract: Instance segmentation, which seeks to obtain both class and instance labels for each pixel in the input image, is a challenging task in computer vision. State-of- the-art algorithms often employ a search-based strategy, which first divides the output image with a regular grid and generate proposals at each grid cell, then the proposals are classified and boundaries refined. In this paper, we propose a novel algorithm that directly utilizes a fully convolutional network (FCN) to predict instance labels. Specifically, we propose a variational relaxation of instance segmentation as minimizing an optimization functional for a piecewise-constant segmentation problem, which can be used to train an FCN end-to-end. It extends the classical Mumford-Shah variational segmentation algorithm to be able to handle the permutation-invariant ground truth in instance segmentation. Experiments on PASCAL VOC 2012 and the MSCOCO 2017 dataset show that the proposed approach efficiently tackles the instance segmentation task.

count=2
* Uncovering the Topology of Time-Varying fMRI Data using Cubical Persistence
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/4d771504ddcd28037b4199740df767e6-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/4d771504ddcd28037b4199740df767e6-Paper.pdf)]
    * Title: Uncovering the Topology of Time-Varying fMRI Data using Cubical Persistence
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Bastian Rieck, Tristan Yates, Christian Bock, Karsten Borgwardt, Guy Wolf, Nicholas Turk-Browne, Smita Krishnaswamy
    * Abstract: Functional magnetic resonance imaging (fMRI) is a crucial technology for gaining insights into cognitive processes in humans. Data amassed from fMRI measurements result in volumetric data sets that vary over time. However, analysing such data presents a challenge due to the large degree of noise and person-to-person variation in how information is represented in the brain. To address this challenge, we present a novel topological approach that encodes each time point in an fMRI data set as a persistence diagram of topological features, i.e. high-dimensional voids present in the data. This representation naturally does not rely on voxel-by-voxel correspondence and is robust towards noise. We show that these time-varying persistence diagrams can be clustered to find meaningful groupings between participants, and that they are also useful in studying within-subject brain state trajectories of subjects performing a particular task. Here, we apply both clustering and trajectory analysis techniques to a group of participants watching the movie 'Partly Cloudy'. We observe significant differences in both brain state trajectories and overall topological activity between adults and children watching the same movie.

count=2
* Detecting Moments and Highlights in Videos via Natural Language Queries
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/62e0973455fd26eb03e91d5741a4a3bb-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/62e0973455fd26eb03e91d5741a4a3bb-Paper.pdf)]
    * Title: Detecting Moments and Highlights in Videos via Natural Language Queries
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Jie Lei, Tamara L Berg, Mohit Bansal
    * Abstract: Detecting customized moments and highlights from videos given natural language (NL) user queries is an important but under-studied topic. One of the challenges in pursuing this direction is the lack of annotated data. To address this issue, we present the Query-based Video Highlights (QVHighlights) dataset. It consists of over 10,000 YouTube videos, covering a wide range of topics, from everyday activities and travel in lifestyle vlog videos to social and political activities in news videos. Each video in the dataset is annotated with: (1) a human-written free-form NL query, (2) relevant moments in the video w.r.t. the query, and (3) five-point scale saliency scores for all query-relevant clips. This comprehensive annotation enables us to develop and evaluate systems that detect relevant moments as well as salient highlights for diverse, flexible user queries. We also present a strong baseline for this task, Moment-DETR, a transformer encoder-decoder model that views moment retrieval as a direct set prediction problem, taking extracted video and query representations as inputs and predicting moment coordinates and saliency scores end-to-end. While our model does not utilize any human prior, we show that it performs competitively when compared to well-engineered architectures. With weakly supervised pretraining using ASR captions, Moment-DETR substantially outperforms previous methods. Lastly, we present several ablations and visualizations of Moment-DETR. Data and code is publicly available at https://github.com/jayleicn/moment_detr.

count=2
* Multi-Scale Representation Learning on Proteins
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/d494020ff8ec181ef98ed97ac3f25453-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/d494020ff8ec181ef98ed97ac3f25453-Paper.pdf)]
    * Title: Multi-Scale Representation Learning on Proteins
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Vignesh Ram Somnath, Charlotte Bunne, Andreas Krause
    * Abstract: Proteins are fundamental biological entities mediating key roles in cellular function and disease. This paper introduces a multi-scale graph construction of a protein –HoloProt– connecting surface to structure and sequence. The surface captures coarser details of the protein, while sequence as primary component and structure –comprising secondary and tertiary components– capture finer details. Our graph encoder then learns a multi-scale representation by allowing each level to integrate the encoding from level(s) below with the graph at that level. We test the learned representation on different tasks, (i.) ligand binding affinity (regression), and (ii.) protein function prediction (classification).On the regression task, contrary to previous methods, our model performs consistently and reliably across different dataset splits, outperforming all baselines on most splits. On the classification task, it achieves a performance close to the top-performing model while using 10x fewer parameters. To improve the memory efficiency of our construction, we segment the multiplex protein surface manifold into molecular superpixels and substitute the surface with these superpixels at little to no performance loss.

count=1
* TSI: Temporal Scale Invariant Network for Action Proposal Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Liu_TSI_Temporal_Scale_Invariant_Network_for_Action_Proposal_Generation_ACCV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2020/papers/Liu_TSI_Temporal_Scale_Invariant_Network_for_Action_Proposal_Generation_ACCV_2020_paper.pdf)]
    * Title: TSI: Temporal Scale Invariant Network for Action Proposal Generation
    * Publisher: ACCV
    * Publication Date: `2020`
    * Authors: Shuming Liu, Xu Zhao, Haisheng Su, Zhilan Hu
    * Abstract: Despite the great progress in temporal action proposal generation, most state-of-the-art methods ignore the impact of action scales and the performance of short actions is still far from satisfaction. In this paper, we first analyze the sample imbalance issue in action proposal generation, and correspondingly devise a novel scale-invariant loss function to alleviate the insufficient learning of short actions. To further achieve proposal generation task, we adopt the pipeline of boundary evaluation and proposal completeness regression, and propose the Temporal Scale Invariant network. To better leverage the temporal context, boundary evaluation module generates action boundaries with high-precision-assured global branch and high-recall-assured local branch. Simultaneously, the proposal evaluation module is supervised with introduced scale-invariant loss, predicting accurate proposal completeness for different scales of actions. Comprehensive experiments are conducted on ActivityNet-1.3 and THUMOS14 benchmarks, where TSI achieves state-of-the-art performance. Especially, AUC performance of short actions is boosted from 36.53% to 39.63% compared with baseline.

count=1
* An Iterated L1 Algorithm for Non-smooth Non-convex Optimization in Computer Vision
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Ochs_An_Iterated_L1_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ochs_An_Iterated_L1_2013_CVPR_paper.pdf)]
    * Title: An Iterated L1 Algorithm for Non-smooth Non-convex Optimization in Computer Vision
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Peter Ochs, Alexey Dosovitskiy, Thomas Brox, Thomas Pock
    * Abstract: Natural image statistics indicate that we should use nonconvex norms for most regularization tasks in image processing and computer vision. Still, they are rarely used in practice due to the challenge to optimize them. Recently, iteratively reweighed 1 minimization has been proposed as a way to tackle a class of non-convex functions by solving a sequence of convex 2 1 problems. Here we extend the problem class to linearly constrained optimization of a Lipschitz continuous function, which is the sum of a convex function and a function being concave and increasing on the non-negative orthant (possibly non-convex and nonconcave on the whole space). This allows to apply the algorithm to many computer vision tasks. We show the effect of non-convex regularizers on image denoising, deconvolution, optical flow, and depth map fusion. Non-convexity is particularly interesting in combination with total generalized variation and learned image priors. Efficient optimization is made possible by some important properties that are shown to hold.

count=1
* Image Segmentation by Cascaded Region Agglomeration
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Ren_Image_Segmentation_by_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ren_Image_Segmentation_by_2013_CVPR_paper.pdf)]
    * Title: Image Segmentation by Cascaded Region Agglomeration
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Zhile Ren, Gregory Shakhnarovich
    * Abstract: We propose a hierarchical segmentation algorithm that starts with a very fine oversegmentation and gradually merges regions using a cascade of boundary classifiers. This approach allows the weights of region and boundary features to adapt to the segmentation scale at which they are applied. The stages of the cascade are trained sequentially, with asymetric loss to maximize boundary recall. On six segmentation data sets, our algorithm achieves best performance under most region-quality measures, and does it with fewer segments than the prior work. Our algorithm is also highly competitive in a dense oversegmentation (superpixel) regime under boundary-based measures.

count=1
* Nonparametric Scene Parsing with Adaptive Feature Relevance and Semantic Context
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Singh_Nonparametric_Scene_Parsing_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Singh_Nonparametric_Scene_Parsing_2013_CVPR_paper.pdf)]
    * Title: Nonparametric Scene Parsing with Adaptive Feature Relevance and Semantic Context
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Gautam Singh, Jana Kosecka
    * Abstract: This paper presents a nonparametric approach to semantic parsing using small patches and simple gradient, color and location features. We learn the relevance of individual feature channels at test time using a locally adaptive distance metric. To further improve the accuracy of the nonparametric approach, we examine the importance of the retrieval set used to compute the nearest neighbours using a novel semantic descriptor to retrieve better candidates. The approach is validated by experiments on several datasets used for semantic parsing demonstrating the superiority of the method compared to the state of art approaches.

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
* RIGOR: Reusing Inference in Graph Cuts for Generating Object Regions
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Humayun_RIGOR_Reusing_Inference_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Humayun_RIGOR_Reusing_Inference_2014_CVPR_paper.pdf)]
    * Title: RIGOR: Reusing Inference in Graph Cuts for Generating Object Regions
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Ahmad Humayun, Fuxin Li, James M. Rehg
    * Abstract: Popular figure-ground segmentation algorithms generate a pool of boundary-aligned segment proposals that can be used in subsequent object recognition engines. These algorithms can recover most image objects with high accuracy, but are usually computationally intensive since many graph cuts are computed with different enumerations of segment seeds. In this paper we propose an algorithm, RIGOR, for efficiently generating a pool of overlapping segment proposals in images. By precomputing a graph which can be used for parametric min-cuts over different seeds, we speed up the generation of the segment pool. In addition, we have made design choices that avoid extensive computations without losing performance. In particular, we demonstrate that the segmentation performance of our algorithm is slightly better than the state-of-the-art on the PASCAL VOC dataset, while being an order of magnitude faster.

count=1
* Pedestrian Detection in Low-resolution Imagery by Learning Multi-scale Intrinsic Motion Structures (MIMS)
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Zhu_Pedestrian_Detection_in_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Zhu_Pedestrian_Detection_in_2014_CVPR_paper.pdf)]
    * Title: Pedestrian Detection in Low-resolution Imagery by Learning Multi-scale Intrinsic Motion Structures (MIMS)
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Jiejie Zhu, Omar Javed, Jingen Liu, Qian Yu, Hui Cheng, Harpreet Sawhney
    * Abstract: Detecting pedestrians at a distance from large-format wide-area imagery is a challenging problem because of low ground sampling distance (GSD) and low frame rate of the imagery. In such a scenario, the approaches based on appearance cues alone mostly fail because pedestrians are only a few pixels in size. Frame-differencing and optical flow based approaches also give poor detection results due to noise, camera jitter and parallax in aerial videos. To overcome these challenges, we propose a novel approach to extract Multi-scale Intrinsic Motion Structure features from pedestrian's motion patterns for pedestrian detection. The MIMS feature encodes the intrinsic motion properties of an object, which are location, velocity and trajectory-shape invariant. The extracted MIMS representation is robust to noisy flow estimates. In this paper, we give a comparative evaluation of the proposed method and demonstrate that MIMS outperforms the state of the art approaches in identifying pedestrians from low resolution airborne videos.

count=1
* Multi-Instance Object Segmentation With Occlusion Handling
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Chen_Multi-Instance_Object_Segmentation_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Chen_Multi-Instance_Object_Segmentation_2015_CVPR_paper.pdf)]
    * Title: Multi-Instance Object Segmentation With Occlusion Handling
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Yi-Ting Chen, Xiaokai Liu, Ming-Hsuan Yang
    * Abstract: We present a multi-instance object segmentation algorithm to tackle occlusions. As an object is split into two parts by an occluder, it is nearly impossible to group the two separate regions into an instance by purely bottom-up schemes. To address this problem, we propose to incorporate top-down category specific reasoning and shape prediction through exemplars into an intuitive energy minimization framework. We perform extensive evaluations of our method on the challenging PASCAL VOC 2012 segmentation set. The proposed algorithm achieves favorable results on the joint detection and segmentation task against the state-of-the-art method both quantitatively and qualitatively.

count=1
* The S-Hock Dataset: Analyzing Crowds at the Stadium
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Conigliaro_The_S-Hock_Dataset_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Conigliaro_The_S-Hock_Dataset_2015_CVPR_paper.pdf)]
    * Title: The S-Hock Dataset: Analyzing Crowds at the Stadium
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Davide Conigliaro, Paolo Rota, Francesco Setti, Chiara Bassetti, Nicola Conci, Nicu Sebe, Marco Cristani
    * Abstract: The topic of crowd modeling in computer vision usually assumes a single generic typology of crowd, which is very simplistic. In this paper we adopt a taxonomy that is widely accepted in sociology, focusing on a particular category, the spectator crowd, which is formed by people "interested in watching something specific that they came to see". This can be found at the stadiums, amphitheaters, cinema, etc. In particular, we propose a novel dataset, the Spectators Hockey (S-Hock), which deals with 4 hockey matches during an international tournament. In the dataset, a massive annotation has been carried out, focusing on the spectators at different levels of details: at a higher level, people have been labeled depending on the team they are supporting and the fact that they know the people close to them; going to the lower levels, standard pose information has been considered (regarding the head, the body) but also fine grained actions such as hands on hips, clapping hands etc. The labeling focused on the game field also, permitting to relate what is going on in the match with the crowd behavior. This brought to more than 100 millions of annotations, useful for standard applications as people counting and head pose estimation but also for novel tasks as spectator categorization. For all of these we provide protocols and baseline results, encouraging further research.

count=1
* Hierarchically-Constrained Optical Flow
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Kennedy_Hierarchically-Constrained_Optical_Flow_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Kennedy_Hierarchically-Constrained_Optical_Flow_2015_CVPR_paper.pdf)]
    * Title: Hierarchically-Constrained Optical Flow
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Ryan Kennedy, Camillo J. Taylor
    * Abstract: This paper presents a novel approach to solving optical flow problems using a discrete, tree-structured MRF derived from a hierarchical segmentation of the image. Our method can be used to find globally optimal matching solutions even for problems involving very large motions. Experiments demonstrate that our approach is competitive on the MPI-Sintel dataset and that it can significantly outperform existing methods on problems involving large motions.

count=1
* Classifier Based Graph Construction for Video Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Khoreva_Classifier_Based_Graph_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Khoreva_Classifier_Based_Graph_2015_CVPR_paper.pdf)]
    * Title: Classifier Based Graph Construction for Video Segmentation
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Anna Khoreva, Fabio Galasso, Matthias Hein, Bernt Schiele
    * Abstract: Video segmentation has become an important and active research area with a large diversity of proposed approaches. Graph-based methods, enabling topperformance on recent benchmarks, consist of three essential components: 1. powerful features account for object appearance and motion similarities; 2. spatio-temporal neighborhoods of pixels or superpixels (the graph edges) are modeled using a combination of those features; 3. video segmentation is formulated as a graph partitioning problem. While a wide variety of features have been explored and various graph partition algorithms have been proposed, there is surprisingly little research on how to construct a graph to obtain the best video segmentation performance. This is the focus of our paper. We propose to combine features by means of a classifier, use calibrated classifier outputs as edge weights and define the graph topology by edge selection. By learning the graph (without changes to the graph partitioning method), we improve the results of the best performing video segmentation algorithm by 6% on the challenging VSB100 benchmark, while reducing its runtime by 55%, as the learnt graph is much sparser.

count=1
* Image Segmentation in Twenty Questions
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Rupprecht_Image_Segmentation_in_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Rupprecht_Image_Segmentation_in_2015_CVPR_paper.pdf)]
    * Title: Image Segmentation in Twenty Questions
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Christian Rupprecht, Loic Peter, Nassir Navab
    * Abstract: Consider the following scenario between a human user and the computer. Given an image, the user thinks of an object to be segmented within this picture, but is only allowed to provide binary inputs to the computer (yes or no). In these conditions, can the computer guess this hidden segmentation by asking well-chosen questions to the user? We introduce a strategy for the computer to increase the accuracy of its guess in a minimal number of questions. At each turn, the current belief about the answer is encoded in a Bayesian fashion via a probability distribution over the set of all possible segmentations. To efficiently handle this huge space, the distribution is approximated by sampling representative segmentations using an adapted version of the Metropolis-Hastings algorithm, whose proposal moves build on a geodesic distance transform segmentation method. Following a dichotomic search, the question halving the weighted set of samples is finally picked, and the provided answer is used to update the belief for the upcoming rounds. The performance of this strategy is assessed on three publicly available datasets with diverse visual properties. Our approach shows to be a tractable and very adaptive solution to this problem.

count=1
* Active Sample Selection and Correction Propagation on a Gradually-Augmented Graph
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Su_Active_Sample_Selection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Su_Active_Sample_Selection_2015_CVPR_paper.pdf)]
    * Title: Active Sample Selection and Correction Propagation on a Gradually-Augmented Graph
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Hang Su, Zhaozheng Yin, Takeo Kanade, Seungil Huh
    * Abstract: When data have a complex manifold structure or the characteristics of data evolve over time, it is unrealistic to expect a graph-based semi-supervised learning method to achieve flawless classification given a small number of initial annotations. To address this issue with minimal human interventions, we propose (i) a sample selection criterion used for \textit{active} query of informative samples by minimizing the expected prediction error, and (ii) an efficient {\it correction propagation} method that propagates human correction on selected samples over a {\it gradually-augmented graph} to unlabeled samples without rebuilding the affinity graph. Experimental results conducted on three real world datasets validate that our active sample selection and correction propagation algorithm quickly reaches high quality classification results with minimal human interventions.

count=1
* Bilateral Space Video Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Maerki_Bilateral_Space_Video_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Maerki_Bilateral_Space_Video_CVPR_2016_paper.pdf)]
    * Title: Bilateral Space Video Segmentation
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Nicolas Maerki, Federico Perazzi, Oliver Wang, Alexander Sorkine-Hornung
    * Abstract: In this work, we propose a novel approach to video segmentation that operates in bilateral space. We design a new energy on the vertices of a regularly sampled spatio-temporal bilateral grid, which can be solved efficiently using a standard graph cut label assignment. Using a bilateral formulation, the energy that we minimize implicitly approximates long-range, spatio-temporal connections between pixels while still containing only a small number of variables and only local graph edges. We compare to a number of recent methods, and show that our approach achieves state-of-the-art results on multiple benchmarks in a fraction of the runtime. Furthermore, our method scales linearly with image size, allowing for interactive feedback on real-world high resolution video.

count=1
* Affinity CNN: Learning Pixel-Centric Pairwise Relations for Figure/Ground Embedding
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Maire_Affinity_CNN_Learning_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Maire_Affinity_CNN_Learning_CVPR_2016_paper.pdf)]
    * Title: Affinity CNN: Learning Pixel-Centric Pairwise Relations for Figure/Ground Embedding
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Michael Maire, Takuya Narihira, Stella X. Yu
    * Abstract: Spectral embedding provides a framework for solving perceptual organization problems, including image segmentation and figure/ground organization. From an affinity matrix describing pairwise relationships between pixels, it clusters pixels into regions, and, using a complex-valued extension, orders pixels according to layer. We train a convolutional neural network (CNN) to directly predict the pairwise relationships that define this affinity matrix. Spectral embedding then resolves these predictions into a globally-consistent segmentation and figure/ground organization of the scene. Experiments demonstrate significant benefit to this direct coupling compared to prior works which use explicit intermediate stages, such as edge detection, on the pathway from image to affinities. Our results suggest spectral embedding as a powerful alternative to the conditional random field (CRF)-based globalization schemes typically coupled to deep neural networks.

count=1
* SimpleElastix: A User-Friendly, Multi-Lingual Library for Medical Image Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w15/html/Marstal_SimpleElastix_A_User-Friendly_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w15/papers/Marstal_SimpleElastix_A_User-Friendly_CVPR_2016_paper.pdf)]
    * Title: SimpleElastix: A User-Friendly, Multi-Lingual Library for Medical Image Registration
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Kasper Marstal, Floris Berendsen, Marius Staring, Stefan Klein
    * Abstract: In this paper we present SimpleElastix, an extension of SimpleITK designed to bring the Elastix medical image registration library to a wider audience. Elastix is a modular collection of robust C++ image registration algorithms that is widely used in the literature. However, its command-line interface introduces overhead during prototyping, experimental setup, and tuning of registration algorithms. By integrating Elastix with SimpleITK, Elastix can be used as a native library in Python, Java, R, Octave, Ruby, Lua, Tcl and C# on Linux, Mac and Windows. This allows Elastix to intregrate naturally with many development environments so the user can focus more on the registration problem and less on the underlying C++ implementation. As means of demonstration, we show how to register MR images of brains and natural pictures of faces using minimal amount of code. SimpleElastix is open source, licensed under the permissive Apache License Version 2.0 and available at https://github.com/kaspermarstal/SimpleElastix.

count=1
* Neuron Segmentation Based on CNN With Semi-Supervised Regularization
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w27/html/Xu_Neuron_Segmentation_Based_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w27/papers/Xu_Neuron_Segmentation_Based_CVPR_2016_paper.pdf)]
    * Title: Neuron Segmentation Based on CNN With Semi-Supervised Regularization
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Kun Xu, Hang Su, Jun Zhu, Ji-Song Guan, Bo Zhang
    * Abstract: Neuron segmentation in two-photon microscopy images is a critical step to investigate neural network activities in vivo. However, it still remains as a challenging problem due to the image qualities, which largely results from the non-linear imaging mechanism and 3D imaging diffusion. To address these issues, we proposed a novel framework by incorporating the convolutional neural network (CNN) with a semi-supervised regularization term, which reduces the human efforts in labeling without sacrificing the performance. Specifically, we generate a putative label for each unlabel sample regularized with a graph-smooth term, which are used as if they were true labels. A CNN model is therefore trained in a supervised fashion with labeled and unlabeled data simultaneously, which is used to detect neuron regions in 2D images. Afterwards, neuron segmentation in a 3D volume is conducted by associating the corresponding neuron regions in each image. Experiments on real-world datasets demonstrate that our approach outperforms neuron segmentation based on the graph-based semi-supervised learning, the supervised CNN and variants of the semi-supervised CNN.

count=1
* Hyperspectral Image Super-Resolution via Non-Local Sparse Tensor Factorization
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Dian_Hyperspectral_Image_Super-Resolution_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dian_Hyperspectral_Image_Super-Resolution_CVPR_2017_paper.pdf)]
    * Title: Hyperspectral Image Super-Resolution via Non-Local Sparse Tensor Factorization
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Renwei Dian, Leyuan Fang, Shutao Li
    * Abstract: Hyperspectral image(HSI)super-resolution, which fuses a low-resolution (LR) HSI with a high-resolution (HR) multispectral image (MSI), has recently attracted much attention. Most of the current HSI super-resolution approaches are based on matrix factorization, which unfolds the three-dimensional HSI as a matrix before processing. In general, the matrix data representation obtained after the matrix unfolding operation makes it hard to fully exploit the inherent HSI spatial-spectral structures. In this paper, a novel HSI super-resolution method based on non-local sparse tensor factorization (called as the NLSTF) is proposed. The sparse tensor factorization can directly decompose each cube of the HSI as a sparse core tensor and dictionaries of three modes, which reformulates the HSI super-resolution problem as the estimation of sparse core tensor and dictionaries for each cube. To further exploit the non-local spatial self-similarities of the HSI, similar cubes are grouped together, and they are assumed to share the same dictionaries. The dictionaries are learned from the LR-HSI and HR-MSI for each group, and corresponding sparse core tensors are estimated by spare coding on the learned dictionaries for each cube. Experimental results demonstrate the superiority of the proposed NLSTF approach over several state-of-the-art HSI super-resolution approaches.

count=1
* Visual-Inertial-Semantic Scene Representation for 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Dong_Visual-Inertial-Semantic_Scene_Representation_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dong_Visual-Inertial-Semantic_Scene_Representation_CVPR_2017_paper.pdf)]
    * Title: Visual-Inertial-Semantic Scene Representation for 3D Object Detection
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Jingming Dong, Xiaohan Fei, Stefano Soatto
    * Abstract: We describe a system to detect objects in three-dimensional space using video and inertial sensors (accelerometer and gyrometer), ubiquitous in modern mobile platforms from phones to drones. Inertials afford the ability to impose class-specific scale priors for objects, and provide a global orientation reference. A minimal sufficient representation, the posterior of semantic (identity) and syntactic (pose) attributes of objects in space, can be decomposed into a geometric term, which can be maintained by a localization-and-mapping filter, and a likelihood function, which can be approximated by a discriminatively-trained convolutional neural network The resulting system can process the video stream causally in real time, and provides a representation of objects in the scene that is persistent: Confidence in the presence of objects grows with evidence, and objects previously seen are kept in memory even when temporarily occluded, with their return into view automatically predicted to prime re-detection.

count=1
* Boundary-Aware Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Hayder_Boundary-Aware_Instance_Segmentation_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Hayder_Boundary-Aware_Instance_Segmentation_CVPR_2017_paper.pdf)]
    * Title: Boundary-Aware Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Zeeshan Hayder, Xuming He, Mathieu Salzmann
    * Abstract: We address the problem of instance-level semantic seg- mentation, which aims at jointly detecting, segmenting and classifying every individual object in an image. In this con- text, existing methods typically propose candidate objects, usually as bounding boxes, and directly predict a binary mask within each such proposal. As a consequence, they cannot recover from errors in the object candidate genera- tion process, such as too small or shifted boxes. In this paper, we introduce a novel object segment rep- resentation based on the distance transform of the object masks. We then design an object mask network (OMN) with a new residual-deconvolution architecture that infers such a representation and decodes it into the final binary object mask. This allows us to predict masks that go beyond the scope of the bounding boxes and are thus robust to inaccu- rate object candidates. We integrate our OMN into a Mul- titask Network Cascade framework, and learn the result- ing boundary-aware instance segmentation (BAIS) network in an end-to-end manner. Our experiments on the PAS- CAL VOC 2012 and the Cityscapes datasets demonstrate the benefits of our approach, which outperforms the state- of-the-art in both object proposal generation and instance segmentation.

count=1
* End-To-End Instance Segmentation With Recurrent Attention
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Ren_End-To-End_Instance_Segmentation_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ren_End-To-End_Instance_Segmentation_CVPR_2017_paper.pdf)]
    * Title: End-To-End Instance Segmentation With Recurrent Attention
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Mengye Ren, Richard S. Zemel
    * Abstract: While convolutional neural networks have gained impressive success recently in solving structured prediction problems such as semantic segmentation, it remains a challenge to differentiate individual object instances in the scene. Instance segmentation is very important in a variety of applications, such as autonomous driving, image captioning, and visual question answering. Techniques that combine large graphical models with low-level vision have been proposed to address this problem; however, we propose an end-to-end recurrent neural network (RNN) architecture with an attention mechanism to model a human-like counting process, and produce detailed instance segmentations. The network is jointly trained to sequentially produce regions of interest as well as a dominant object segmentation within each region. The proposed model achieves competitive results on the CVPPP, KITTI, and Cityscapes datasets.

count=1
* A Dual Ascent Framework for Lagrangean Decomposition of Combinatorial Problems
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Swoboda_A_Dual_Ascent_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Swoboda_A_Dual_Ascent_CVPR_2017_paper.pdf)]
    * Title: A Dual Ascent Framework for Lagrangean Decomposition of Combinatorial Problems
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Paul Swoboda, Jan Kuske, Bogdan Savchynskyy
    * Abstract: We propose a general dual ascent (message passing) framework for Lagrangean (dual) decomposition of combinatorial problems. Although methods of this type have shown their efficiency for a number of problems, so far there was no general algorithm applicable to multiple problem types. In this work, we propose such a general algorithm. It depends on several parameters, which can be used to optimize its performance in each particular setting. We demonstrate efficiency of our method on the graph matching and the multicut problems, where it outperforms state-of-the-art solvers including those based on the subgradient optimization and off-the-shelf linear programming solvers.

count=1
* A Message Passing Algorithm for the Minimum Cost Multicut Problem
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Swoboda_A_Message_Passing_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Swoboda_A_Message_Passing_CVPR_2017_paper.pdf)]
    * Title: A Message Passing Algorithm for the Minimum Cost Multicut Problem
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Paul Swoboda, Bjoern Andres
    * Abstract: We propose a dual decomposition and linear program relaxation of the NP-hard minimum cost multicut problem. Unlike other polyhedral relaxations of the multicut polytope, it is amenable to efficient optimization by message passing. Like other polyhedral relaxations, it can be tightened efficiently by cutting planes. We define an algorithm that alternates between message passing and efficient separation of cycle- and odd-wheel inequalities. This algorithm is more efficient than state-of-the-art algorithms based on linear programming, including algorithms written in the framework of leading commercial software, as we show in experiments with large instances of the problem from applications in computer vision, biomedical image analysis and data mining.

count=1
* Missing Modalities Imputation via Cascaded Residual Autoencoder
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Tran_Missing_Modalities_Imputation_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tran_Missing_Modalities_Imputation_CVPR_2017_paper.pdf)]
    * Title: Missing Modalities Imputation via Cascaded Residual Autoencoder
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Luan Tran, Xiaoming Liu, Jiayu Zhou, Rong Jin
    * Abstract: Affordable sensors lead to an increasing interest in acquiring and modeling data with multiple modalities. Learning from multiple modalities has shown to significantly improve performance in object recognition. However, in practice it is common that the sensing equipment experiences unforeseeable malfunction or configuration issues, leading to corrupted data with missing modalities. Most existing multi-modal learning algorithms could not handle missing modalities, and would discard either all modalities with missing values or all corrupted data. To leverage the valuable information in the corrupted data, we propose to impute the missing data by leveraging the relatedness among different modalities. Specifically, we propose a novel Cascaded Residual Autoencoder (CRA) to impute missing modalities. By stacking residual autoencoders, CRA grows iteratively to model the residual between the current prediction and original data. Extensive experiments demonstrate the superior performance of CRA on both the data imputation and the object recognition task on imputed data.

count=1
* Unsupervised Semantic Scene Labeling for Streaming Data
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Wigness_Unsupervised_Semantic_Scene_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wigness_Unsupervised_Semantic_Scene_CVPR_2017_paper.pdf)]
    * Title: Unsupervised Semantic Scene Labeling for Streaming Data
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Maggie Wigness, John G. Rogers III
    * Abstract: We introduce an unsupervised semantic scene labeling approach that continuously learns and adapts semantic models discovered within a data stream. While closely related to unsupervised video segmentation, our algorithm is not designed to be an early video processing strategy that produces coherent over-segmentations, but instead, to directly learn higher-level semantic concepts. This is achieved with an ensemble-based approach, where each learner clusters data from a local window in the data stream. Overlapping local windows are processed and encoded in a graph structure to create a label mapping across windows and reconcile the labelings to reduce unsupervised learning noise. Additionally, we iteratively learn a merging threshold criteria from observed data similarities to automatically determine the number of learned labels without human provided parameters. Experiments show that our approach semantically labels video streams with a high degree of accuracy, and achieves a better balance of under and over-segmentation entropy than existing video segmentation algorithms given similar numbers of label outputs.

count=1
* Semantic Instance Segmentation for Autonomous Driving
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/html/De_Brabandere_Semantic_Instance_Segmentation_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/papers/De_Brabandere_Semantic_Instance_Segmentation_CVPR_2017_paper.pdf)]
    * Title: Semantic Instance Segmentation for Autonomous Driving
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Bert De Brabandere, Davy Neven, Luc Van Gool
    * Abstract: Semantic instance segmentation remains a challenging task. In this work we propose to tackle the problem with a discriminative loss function, operating at the pixel level, that encourages a convolutional network to produce a representation of the image that can easily be clustered into instances with a simple post-processing step. Our approach of combining an off-the-shelf network with a principled loss function inspired by a metric learning objective is conceptually simple and distinct from recent efforts in instance segmentation and is well-suited for real-time applications. In contrast to previous works, our method does not rely on object proposals or recurrent mechanisms and is particularly well suited for tasks with complex occlusions. A key contribution of our work is to demonstrate that such a simple setup without bells and whistles is effective and can perform on-par with more complex methods. We achieve competitive performance on the Cityscapes segmentation benchmark.

count=1
* Nuclei Segmentation of Fluorescence Microscopy Images Using Three Dimensional Convolutional Neural Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/html/Ho_Nuclei_Segmentation_of_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/papers/Ho_Nuclei_Segmentation_of_CVPR_2017_paper.pdf)]
    * Title: Nuclei Segmentation of Fluorescence Microscopy Images Using Three Dimensional Convolutional Neural Networks
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: David Joon Ho, Chichen Fu, Paul Salama, Kenneth W. Dunn, Edward J. Delp
    * Abstract: Fluorescence microscopy enables one to visualize subcellular structures of living tissue or cells in three dimensions. This is especially true for two-photon microscopy using near-infrared light which can image deeper into tissue. To characterize and analyze biological structures, nuclei segmentation is a prerequisite step. Due to the complexity and size of the image data sets, manual segmentation is prohibitive. This paper describes a fully 3D nuclei segmentation method using three dimensional convolutional neural networks. To train the network, synthetic volumes with corresponding labeled volumes are automatically generated. Our results from multiple data sets demonstrate that our method can successfully segment nuclei in 3D.

count=1
* Efficient Interactive Annotation of Segmentation Datasets With Polygon-RNN++
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Acuna_Efficient_Interactive_Annotation_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Acuna_Efficient_Interactive_Annotation_CVPR_2018_paper.pdf)]
    * Title: Efficient Interactive Annotation of Segmentation Datasets With Polygon-RNN++
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: David Acuna, Huan Ling, Amlan Kar, Sanja Fidler
    * Abstract: Manually labeling datasets with object masks is extremely time consuming. In this work, we follow the idea of Polygon-RNN to produce polygonal annotations of objects interactively using humans-in-the-loop. We introduce several important improvements to the model: 1) we design a new CNN encoder architecture, 2) show how to effectively train the model with Reinforcement Learning, and 3) significantly increase the output resolution using a Graph Neural Network, allowing the model to accurately annotate high-resolution objects in images. Extensive evaluation on the Cityscapes dataset shows that our model, which we refer to as Polygon-RNN++, significantly outperforms the original model in both automatic (10% absolute and 16% relative improvement in mean IoU) and interactive modes (requiring 50% fewer clicks by annotators). We further analyze the cross-domain scenario in which our model is trained on one dataset, and used out of the box on datasets from varying domains. The results show that Polygon-RNN++ exhibits powerful generalization capabilities, achieving significant improvements over existing pixel-wise methods. Using simple online fine-tuning we further achieve a high reduction in annotation time for new datasets, moving a step closer towards an interactive annotation tool to be used in practice.

count=1
* Learning to Segment Every Thing
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Learning_to_Segment_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Learning_to_Segment_CVPR_2018_paper.pdf)]
    * Title: Learning to Segment Every Thing
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Ronghang Hu, Piotr Dollár, Kaiming He, Trevor Darrell, Ross Girshick
    * Abstract: Most methods for object instance segmentation require all training examples to be labeled with segmentation masks. This requirement makes it expensive to annotate new categories and has restricted instance segmentation models to ~100 well-annotated classes. The goal of this paper is to propose a new partially supervised training paradigm, together with a novel weight transfer function, that enables training instance segmentation models on a large set of categories all of which have box annotations, but only a small fraction of which have mask annotations. These contributions allow us to train Mask R-CNN to detect and segment 3000 visual concepts using box annotations from the Visual Genome dataset and mask annotations from the 80 classes in the COCO dataset. We evaluate our approach in a controlled study on the COCO dataset. This work is a first step towards instance segmentation models that have broad comprehension of the visual world.

count=1
* Jointly Localizing and Describing Events for Dense Video Captioning
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Jointly_Localizing_and_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Jointly_Localizing_and_CVPR_2018_paper.pdf)]
    * Title: Jointly Localizing and Describing Events for Dense Video Captioning
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Yehao Li, Ting Yao, Yingwei Pan, Hongyang Chao, Tao Mei
    * Abstract: Automatically describing a video with natural language is regarded as a fundamental challenge in computer vision. The problem nevertheless is not trivial especially when a video contains multiple events to be worthy of mention, which often happens in real videos. A valid question is how to temporally localize and then describe events, which is known as ``dense video captioning." In this paper, we present a novel framework for dense video captioning that unifies the localization of temporal event proposals and sentence generation of each proposal, by jointly training them in an end-to-end manner. To combine these two worlds, we integrate a new design, namely descriptiveness regression, into a single shot detection structure to infer the descriptive complexity of each detected proposal via sentence generation. This in turn adjusts the temporal locations of each event proposal. Our model differs from existing dense video captioning methods since we propose a joint and global optimization of detection and captioning, and the framework uniquely capitalizes on an attribute-augmented video captioning architecture. Extensive experiments are conducted on ActivityNet Captions dataset and our framework shows clear improvements when compared to the state-of-the-art techniques. More remarkably, we obtain a new record: METEOR of 12.96% on ActivityNet Captions official test set.

count=1
* Path Aggregation Network (PANet)
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Path_Aggregation_Network_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Path_Aggregation_Network_CVPR_2018_paper.pdf)]
    * Title: Path Aggregation Network for Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, Jiaya Jia
    * Abstract: The way that information propagates in neural networks is of great importance. In this paper, we propose Path Aggregation Network (PANet) aiming at boosting information flow in proposal-based instance segmentation framework. Specifically, we enhance the entire feature hierarchy with accurate localization signals in lower layers by bottom-up path augmentation, which shortens the information path between lower layers and topmost feature. We present adaptive feature pooling, which links feature grid and all feature levels to make useful information in each level propagate directly to following proposal subnetworks. A complementary branch capturing different views for each proposal is created to further improve mask prediction. These improvements are simple to implement, with subtle extra computational overhead. Yet they are useful and make our PANet reach the 1st place in the COCO 2017 Challenge Instance Segmentation task and the 2nd place in Object Detection task without large-batch training. PANet is also state-of-the-art on MVD and Cityscapes.

count=1
* Learning Superpixels With Segmentation-Aware Affinity Loss
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Tu_Learning_Superpixels_With_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tu_Learning_Superpixels_With_CVPR_2018_paper.pdf)]
    * Title: Learning Superpixels With Segmentation-Aware Affinity Loss
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Wei-Chih Tu, Ming-Yu Liu, Varun Jampani, Deqing Sun, Shao-Yi Chien, Ming-Hsuan Yang, Jan Kautz
    * Abstract: Superpixel segmentation has been widely used in many computer vision tasks. Existing superpixel algorithms are mainly based on hand-crafted features, which often fail to preserve weak object boundaries. In this work, we leverage deep neural networks to facilitate extracting superpixels from images. We show a simple integration of deep features with existing superpixel algorithms does not result in better performance as these features do not model segmentation. Instead, we propose a segmentation-aware affinity learning approach for superpixel segmentation. Specifically, we propose a new loss function that takes the segmentation error into account for affinity learning. We also develop the Pixel Affinity Net for affinity prediction. Extensive experimental results show that the proposed algorithm based on the learned segmentation-aware loss performs favorably against the state-of-the-art methods. We also demonstrate the use of the learned superpixels in numerous vision applications with consistent improvements.

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
* HSCNN+: Advanced CNN-Based Hyperspectral Recovery From RGB Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Shi_HSCNN_Advanced_CNN-Based_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Shi_HSCNN_Advanced_CNN-Based_CVPR_2018_paper.pdf)]
    * Title: HSCNN+: Advanced CNN-Based Hyperspectral Recovery From RGB Images
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Zhan Shi, Chang Chen, Zhiwei Xiong, Dong Liu, Feng Wu
    * Abstract: Hyperspectral recovery from a single RGB image has seen a great improvement with the development of deep convolutional neural networks (CNNs). In this paper, we propose two advanced CNNs for the hyperspectral reconstruction task, collectively called HSCNN+. We first develop a deep residual network named HSCNN-R, which comprises a number of residual blocks. The superior performance of this model comes from the modern architecture and optimization by removing the hand-crafted upsampling in HSCNN. Based on the promising results of HSCNN-R, we propose another distinct architecture that replaces the residual block by the dense block with a novel fusion scheme, leading to a new network named HSCNN-D. This model substantially deepens the network structure for a more accurate solution. Experimental results demonstrate that our proposed models significantly advance the state-of-the-art. In the NTIRE 2018 Spectral Reconstruction Challenge, our entries rank the 1st (HSCNN-D) and 2nd (HSCNN-R) places on both the "Clean" and "Real World" tracks.

count=1
* On the Iterative Refinement of Densely Connected Representation Levels for Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w14/html/Casanova_On_the_Iterative_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w14/Casanova_On_the_Iterative_CVPR_2018_paper.pdf)]
    * Title: On the Iterative Refinement of Densely Connected Representation Levels for Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Arantxa Casanova, Guillem Cucurull, Michal Drozdzal, Adriana Romero, Yoshua Bengio
    * Abstract: State-of-the-art semantic segmentation approaches increase the receptive field of their models by using either a downsampling path composed of poolings/strided convolutions or successive dilated convolutions. However, it is not clear which operation leads to best results. In this paper, we systematically study the differences introduced by distinct receptive field enlargement methods and their impact on the performance of a novel architecture, called Fully Convolutional DenseResNet (FC-DRN). FC-DRN has a densely connected backbone composed of residual networks. Following standard image segmentation architectures, receptive field enlargement operations that change the representation level are interleaved among residual networks. This allows the model to exploit the benefits of both residual and dense connectivity patterns, namely: gradient flow, iterative refinement of representations, multi-scale feature combination and deep supervision. In order to highlight the potential of our model, we test it on the challenging CamVid urban scene understanding benchmark and make the following observations: 1) downsampling operations outperform dilations when the model is trained from scratch, 2) dilations are useful during the finetuning step of the model, 3) coarser representations require less refinement steps, and 4) ResNets (by model construction) are good regularizers, since they can reduce the model capacity when needed. Finally, we compare our architecture to alternative methods and report state-of-the-art result on the Camvid dataset, with at least twice fewer parameters.

count=1
* 3D Cell Nuclear Morphology: Microscopy Imaging Dataset and Voxel-Based Morphometry Classification Results
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w44/html/Kalinin_3D_Cell_Nuclear_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w44/Kalinin_3D_Cell_Nuclear_CVPR_2018_paper.pdf)]
    * Title: 3D Cell Nuclear Morphology: Microscopy Imaging Dataset and Voxel-Based Morphometry Classification Results
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Alexandr A. Kalinin, Ari Allyn-Feuer, Alex Ade, Gordon-Victor Fon, Walter Meixner, David Dilworth, Jeffrey R. de Wet, Gerald A. Higgins, Gen Zheng, Amy Creekmore, John W. Wiley, James E. Verdone, Robert W. Veltri, Kenneth J. Pienta, Donald S. Coffey, Brian D. Athey, Ivo D. Dinov
    * Abstract: Cell deformation is regulated by complex underlying biological mechanisms associated with spatial and temporal morphological changes in the nucleus that are related to cell differentiation, development, proliferation, and disease. Thus, quantitative analysis of changes in size and shape of nuclear structures in 3D microscopic images is important not only for investigating nuclear organization, but also for detecting and treating pathological conditions such as cancer. While many efforts have been made to develop cell and nuclear shape characteristics in 2D or pseudo-3D, several studies have suggested that 3D morphometric measures provide better results for nuclear shape description and discrimination. A few methods have been proposed to classify cell and nuclear morphological phenotypes in 3D, however, there is a lack of publicly available 3D data for the evaluation and comparison of such algorithms. This limitation becomes of great importance when the ability to evaluate different approaches on benchmark data is needed for better dissemination of the current state of the art methods for bioimage analysis. To address this problem, we present a dataset containing two different cell collections, including original 3D microscopic images of cell nuclei and nucleoli. In addition, we perform a baseline evaluation of a number of popular classification algorithms using 2D and 3D voxel-based morphometric measures. To account for batch effects, while enabling calculations of AUROC and AUPR performance metrics, we propose a specific cross-validation scheme that we compare with commonly used k-fold cross-validation. Original and derived imaging data are made publicly available on the project web-page: http://www.socr.umich.edu/projects/3d-cell-morphometry/data.html.

count=1
* Devil Is in the Edges: Learning Semantic Boundaries From Noisy Annotations
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.pdf)]
    * Title: Devil Is in the Edges: Learning Semantic Boundaries From Noisy Annotations
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: David Acuna,  Amlan Kar,  Sanja Fidler
    * Abstract: We tackle the problem of semantic boundary prediction, which aims to identify pixels that belong to object(class) boundaries. We notice that relevant datasets consist of a significant level of label noise, reflecting the fact that precise annotations are laborious to get and thus annotators trade-off quality with efficiency. We aim to learn sharp and precise semantic boundaries by explicitly reasoning about annotation noise during training. We propose a simple new layer and loss that can be used with existing learning-based boundary detectors. Our layer/loss enforces the detector to predict a maximum response along the normal direction at an edge, while also regularizing its direction. We further reason about true object boundaries during training using a level set formulation, which allows the network to learn from misaligned labels in an end-to-end fashion. Experiments show that we improve over the CASENet backbone network by more than 4% in terms of MF(ODS) and 18.61% in terms of AP, outperforming all current state-of-the-art methods including those that deal with alignment. Furthermore, we show that our learned network can be used to significantly improve coarse segmentation labels, lending itself as an efficient way to label new data.

count=1
* Actor-Critic Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Araslanov_Actor-Critic_Instance_Segmentation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Araslanov_Actor-Critic_Instance_Segmentation_CVPR_2019_paper.pdf)]
    * Title: Actor-Critic Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Nikita Araslanov,  Constantin A. Rothkopf,  Stefan Roth
    * Abstract: Most approaches to visual scene analysis have emphasised parallel processing of the image elements. However, one area in which the sequential nature of vision is apparent, is that of segmenting multiple, potentially similar and partially occluded objects in a scene. In this work, we revisit the recurrent formulation of this challenging problem in the context of reinforcement learning. Motivated by the limitations of the global max-matching assignment of the ground-truth segments to the recurrent states, we develop an actor-critic approach in which the actor recurrently predicts one instance mask at a time and utilises the gradient from a concurrently trained critic network. We formulate the state, action, and the reward such as to let the critic model long-term effects of the current prediction and in- corporate this information into the gradient signal. Furthermore, to enable effective exploration in the inherently high-dimensional action space of instance masks, we learn a compact representation using a conditional variational auto-encoder. We show that our actor-critic model consistently provides accuracy benefits over the recurrent baseline on standard instance segmentation benchmarks.

count=1
* DeepCO3: Deep Instance Co-Segmentation by Co-Peak Search and Co-Saliency Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Hsu_DeepCO3_Deep_Instance_Co-Segmentation_by_Co-Peak_Search_and_Co-Saliency_Detection_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hsu_DeepCO3_Deep_Instance_Co-Segmentation_by_Co-Peak_Search_and_Co-Saliency_Detection_CVPR_2019_paper.pdf)]
    * Title: DeepCO3: Deep Instance Co-Segmentation by Co-Peak Search and Co-Saliency Detection
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Kuang-Jui Hsu,  Yen-Yu Lin,  Yung-Yu Chuang
    * Abstract: In this paper, we address a new task called instance co-segmentation. Given a set of images jointly covering object instances of a specific category, instance co-segmentation aims to identify all of these instances and segment each of them, i.e. generating one mask for each instance. This task is important since instance-level segmentation is preferable for humans and many vision applications. It is also challenging because no pixel-wise annotated training data are available and the number of instances in each image is unknown. We solve this task by dividing it into two sub-tasks, co-peak search and instance mask segmentation. In the former sub-task, we develop a CNN-based network to detect the co-peaks as well as co-saliency maps for a pair of images. A co-peak has two endpoints, one in each image, that are local maxima in the response maps and similar to each other. Thereby, the two endpoints are potentially covered by a pair of instances of the same category. In the latter subtask, we design a ranking function that takes the detected co-peaks and co-saliency maps as inputs and can select the object proposals to produce the final results. Our method for instance co-segmentation and its variant for object colocalization are evaluated on four datasets, and achieve favorable performance against the state-of-the-art methods. The source codes and the collected datasets are available at https://github.com/KuangJuiHsu/DeepCO3/

count=1
* SAIL-VOS: Semantic Amodal Instance Level Video Object Segmentation - A Synthetic Dataset and Baselines
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_SAIL-VOS_Semantic_Amodal_Instance_Level_Video_Object_Segmentation_-_A_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_SAIL-VOS_Semantic_Amodal_Instance_Level_Video_Object_Segmentation_-_A_CVPR_2019_paper.pdf)]
    * Title: SAIL-VOS: Semantic Amodal Instance Level Video Object Segmentation - A Synthetic Dataset and Baselines
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yuan-Ting Hu,  Hong-Shuo Chen,  Kexin Hui,  Jia-Bin Huang,  Alexander G. Schwing
    * Abstract: We introduce SAIL-VOS (Semantic Amodal Instance Level Video Object Segmentation), a new dataset aiming to stimulate semantic amodal segmentation research. Humans can effortlessly recognize partially occluded objects and reliably estimate their spatial extent beyond the visible. However, few modern computer vision techniques are capable of reasoning about occluded parts of an object. This is partly due to the fact that very few image datasets and no video dataset exist which permit development of those methods. To address this issue, we present a synthetic dataset extracted from the photo-realistic game GTA-V. Each frame is accompanied with densely annotated, pixel-accurate visible and amodal segmentation masks with semantic labels. More than 1.8M objects are annotated resulting in 100 times more annotations than existing datasets. We demonstrate the challenges of the dataset by quantifying the performance of several baselines. Data and additional material is available at http://sailvos.web.illinois.edu.

count=1
* Panoptic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf)]
    * Title: Panoptic Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Alexander Kirillov,  Kaiming He,  Ross Girshick,  Carsten Rother,  Piotr Dollar
    * Abstract: We propose and study a task we name panoptic segmentation (PS). Panoptic segmentation unifies the typically distinct tasks of semantic segmentation (assign a class label to each pixel) and instance segmentation (detect and segment each object instance). The proposed task requires generating a coherent scene segmentation that is rich and complete, an important step toward real-world vision systems. While early work in computer vision addressed related image/scene parsing tasks, these are not currently popular, possibly due to lack of appropriate metrics or associated recognition challenges. To address this, we propose a novel panoptic quality (PQ) metric that captures performance for all classes (stuff and things) in an interpretable and unified manner. Using the proposed metric, we perform a rigorous study of both human and machine performance for PS on three existing datasets, revealing interesting insights about the task. The aim of our work is to revive the interest of the community in a more unified view of image segmentation. For more analysis and up-to-date results, please check the arXiv version of the paper: \smallhttps://arxiv.org/abs/1801.00868 .

count=1
* Attention-Guided Unified Network for Panoptic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Attention-Guided_Unified_Network_for_Panoptic_Segmentation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Attention-Guided_Unified_Network_for_Panoptic_Segmentation_CVPR_2019_paper.pdf)]
    * Title: Attention-Guided Unified Network for Panoptic Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yanwei Li,  Xinze Chen,  Zheng Zhu,  Lingxi Xie,  Guan Huang,  Dalong Du,  Xingang Wang
    * Abstract: This paper studies panoptic segmentation, a recently proposed task which segments foreground (FG) objects at the instance level as well as background (BG) contents at the semantic level. Existing methods mostly dealt with these two problems separately, but in this paper, we reveal the underlying relationship between them, in particular, FG objects provide complementary cues to assist BG understanding. Our approach, named the Attention-guided Unified Network (AUNet), is a unified framework with two branches for FG and BG segmentation simultaneously. Two sources of attentions are added to the BG branch, namely, RPN and FG segmentation mask to provide object-level and pixel-level attentions, respectively. Our approach is generalized to different backbones with consistent accuracy gain in both FG and BG segmentation, and also sets new state-of-the-arts both in the MS-COCO (46.5% PQ) and Cityscapes (59.0% PQ) benchmarks.

count=1
* Fast Interactive Object Annotation With Curve-GCN
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Ling_Fast_Interactive_Object_Annotation_With_Curve-GCN_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ling_Fast_Interactive_Object_Annotation_With_Curve-GCN_CVPR_2019_paper.pdf)]
    * Title: Fast Interactive Object Annotation With Curve-GCN
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Huan Ling,  Jun Gao,  Amlan Kar,  Wenzheng Chen,  Sanja Fidler
    * Abstract: Manually labeling objects by tracing their boundaries is a laborious process. In Polygon-RNN++, the authors proposed Polygon-RNN that produces polygonal annotations in a recurrent manner using a CNN-RNN architecture, allowing interactive correction via humans-in-the-loop. We propose a new framework that alleviates the sequential nature of Polygon-RNN, by predicting all vertices simultaneously using a Graph Convolutional Network (GCN). Our model is trained end-to-end, and runs in real time. It supports object annotation by either polygons or splines, facilitating labeling efficiency for both line-based and curved objects. We show that Curve-GCN outperforms all existing approaches in automatic mode, including the powerful DeepLab, and is significantly more efficient in interactive mode than Polygon-RNN++. Our model runs at 29.3ms in automatic, and 2.6ms in interactive mode, making it 10x and 100x faster than Polygon-RNN++.

count=1
* An End-To-End Network for Panoptic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_An_End-To-End_Network_for_Panoptic_Segmentation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_An_End-To-End_Network_for_Panoptic_Segmentation_CVPR_2019_paper.pdf)]
    * Title: An End-To-End Network for Panoptic Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Huanyu Liu,  Chao Peng,  Changqian Yu,  Jingbo Wang,  Xu Liu,  Gang Yu,  Wei Jiang
    * Abstract: Panoptic segmentation, which needs to assign a category label to each pixel and segment each object instance simultaneously, is a challenging topic. Traditionally, the existing approaches utilize two independent models without sharing features, which makes the pipeline inefficient to implement. In addition, a heuristic method is usually employed to merge the results. However, the overlapping relationship between object instances is difficult to determine without sufficient context information during the merging process. To address the problems, we propose a novel end-to-end Occlusion Aware Network (OANet) for panoptic segmentation, which can efficiently and effectively predict both the instance and stuff segmentation in a single network. Moreover, we introduce a novel spatial ranking module to deal with the occlusion problem between the predicted instances. Extensive experiments have been done to validate the performance of our proposed method and promising results have been achieved on the COCO Panoptic benchmark.

count=1
* Content-Aware Multi-Level Guidance for Interactive Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Majumder_Content-Aware_Multi-Level_Guidance_for_Interactive_Instance_Segmentation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Majumder_Content-Aware_Multi-Level_Guidance_for_Interactive_Instance_Segmentation_CVPR_2019_paper.pdf)]
    * Title: Content-Aware Multi-Level Guidance for Interactive Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Soumajit Majumder,  Angela Yao
    * Abstract: In interactive instance segmentation, users give feedback to iteratively refine segmentation masks. The user-provided clicks are transformed into guidance maps which provide the network with necessary cues on the whereabouts of the object of interest. Guidance maps used in current systems are purely distance-based and are either too localized or non-informative. We propose a novel transformation of user clicks to generate content-aware guidance maps that leverage the hierarchical structural information present in an image. Using our guidance maps, even the most basic FCNs are able to outperform existing approaches that require state-of-the-art segmentation networks pre-trained on large scale segmentation datasets. We demonstrate the effectiveness of our proposed transformation strategy through comprehensive experimentation in which we significantly raise state-of-the-art on four standard interactive segmentation benchmarks.

count=1
* Beyond Gradient Descent for Regularized Segmentation Losses
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Marin_Beyond_Gradient_Descent_for_Regularized_Segmentation_Losses_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Marin_Beyond_Gradient_Descent_for_Regularized_Segmentation_Losses_CVPR_2019_paper.pdf)]
    * Title: Beyond Gradient Descent for Regularized Segmentation Losses
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Dmitrii Marin,  Meng Tang,  Ismail Ben Ayed,  Yuri Boykov
    * Abstract: The simplicity of gradient descent (GD) made it the default method for training ever-deeper and complex neural networks. Both loss functions and architectures are often explicitly tuned to be amenable to this basic local optimization. In the context of weakly-supervised CNN segmentation, we demonstrate a well-motivated loss function where an alternative optimizer (ADM) achieves the state-of-the-art while GD performs poorly. Interestingly, GD obtains its best result for a "smoother" tuning of the loss function. The results are consistent across different network architectures. Our loss is motivated by well-understood MRF/CRF regularization models in "shallow" segmentation and their known global solvers. Our work suggests that network design/training should pay more attention to optimization methods.

count=1
* Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Neven_Instance_Segmentation_by_Jointly_Optimizing_Spatial_Embeddings_and_Clustering_Bandwidth_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Neven_Instance_Segmentation_by_Jointly_Optimizing_Spatial_Embeddings_and_Clustering_Bandwidth_CVPR_2019_paper.pdf)]
    * Title: Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Davy Neven,  Bert De Brabandere,  Marc Proesmans,  Luc Van Gool
    * Abstract: Current state-of-the-art instance segmentation methods are not suited for real-time applications like autonomous driving, which require fast execution times at high accuracy. Although the currently dominant proposal-based methods have high accuracy, they are slow and generate masks at a fixed and low resolution. Proposal-free methods, by contrast, can generate masks at high resolution and are often faster, but fail to reach the same accuracy as the proposal-based methods. In this work we propose a new clustering loss function for proposal-free instance segmentation. The loss function pulls the spatial embeddings of pixels belonging to the same instance together and jointly learns an instance-specific clustering bandwidth, maximizing the intersection-over-union of the resulting instance mask. When combined with a fast architecture, the network can perform instance segmentation in real-time while maintaining a high accuracy. We evaluate our method on the challenging Cityscapes benchmark and achieve top results (5% improvement over Mask R-CNN) at more than 10 fps on 2MP images.

count=1
* Amodal Instance Segmentation With KINS Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Qi_Amodal_Instance_Segmentation_With_KINS_Dataset_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qi_Amodal_Instance_Segmentation_With_KINS_Dataset_CVPR_2019_paper.pdf)]
    * Title: Amodal Instance Segmentation With KINS Dataset
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Lu Qi,  Li Jiang,  Shu Liu,  Xiaoyong Shen,  Jiaya Jia
    * Abstract: Amodal instance segmentation, a new direction of instance segmentation, aims to segment each object instance involving its invisible, occluded parts to imitate human ability. This task requires to reason objects' complex structure. Despite important and futuristic, this task lacks data with large-scale and detailed annotations, due to the difficulty of correctly and consistently labeling invisible parts, which creates the huge barrier to explore the frontier of visual recognition. In this paper, we augment KITTI with more instance pixel-level annotation for 8 categories, which we call KITTI INStance dataset (KINS). We propose the network structure to reason invisible parts via a new multi-task framework with Multi-View Coding (MVC), which combines information in various recognition levels. Extensive experiments show that our MVC effectively improves both amodal and inmodal segmentation. The KINS dataset and our proposed method will be made publicly available.

count=1
* TACNet: Transition-Aware Context Network for Spatio-Temporal Action Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Song_TACNet_Transition-Aware_Context_Network_for_Spatio-Temporal_Action_Detection_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Song_TACNet_Transition-Aware_Context_Network_for_Spatio-Temporal_Action_Detection_CVPR_2019_paper.pdf)]
    * Title: TACNet: Transition-Aware Context Network for Spatio-Temporal Action Detection
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Lin Song,  Shiwei Zhang,  Gang Yu,  Hongbin Sun
    * Abstract: Current state-of-the-art approaches for spatio-temporal action detection have achieved impressive results but remain unsatisfactory for temporal extent detection. The main reason comes from that, there are some ambiguous states similar to the real actions which may be treated as target actions even by a well trained network. In this paper, we define these ambiguous samples as "transitional states", and propose a Transition-Aware Context Network (TACNet) to distinguish transitional states. The proposed TACNet includes two main components, i.e., temporal context detector and transition-aware classifier. The temporal context detector can extract long-term context information with constant time complexity by constructing a recurrent network. The transition-aware classifier can further distinguish transitional states by classifying action and transitional states simultaneously. Therefore, the proposed TACNet can substantially improve the performance of spatio-temporal action detection. We extensively evaluate the proposed TACNet on UCF101-24 and J-HMDB datasets. The experimental results demonstrate that TACNet obtains competitive performance on JHMDB and significantly outperforms the state-of-the-art methods on the untrimmed UCF101 24 in terms of both frame-mAP and video-mAP.

count=1
* DeepFlux for Skeletons in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_DeepFlux_for_Skeletons_in_the_Wild_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_DeepFlux_for_Skeletons_in_the_Wild_CVPR_2019_paper.pdf)]
    * Title: DeepFlux for Skeletons in the Wild
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yukang Wang,  Yongchao Xu,  Stavros Tsogkas,  Xiang Bai,  Sven Dickinson,  Kaleem Siddiqi
    * Abstract: Computing object skeletons in natural images is challenging, owing to large variations in object appearance and scale, and the complexity of handling background clutter. Many recent methods frame object skeleton detection as a binary pixel classification problem, which is similar in spirit to learning-based edge detection, as well as to semantic segmentation methods. In the present article, we depart from this strategy by training a CNN to predict a two-dimensional vector field, which maps each scene point to a candidate skeleton pixel, in the spirit of flux-based skeletonization algorithms. This "image context flux" representation has two major advantages over previous approaches. First, it explicitly encodes the relative position of skeletal pixels to semantically meaningful entities, such as the image points in their spatial context, and hence also the implied object boundaries. Second, since the skeleton detection context is a region-based vector field, it is better able to cope with object parts of large width. We evaluate the proposed method on three benchmark datasets for skeleton detection and two for symmetry detection, achieving consistently superior performance over state-of-the-art methods.

count=1
* Classifying, Segmenting, and Tracking Object Instances in Video with Mask Propagation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Bertasius_Classifying_Segmenting_and_Tracking_Object_Instances_in_Video_with_Mask_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bertasius_Classifying_Segmenting_and_Tracking_Object_Instances_in_Video_with_Mask_CVPR_2020_paper.pdf)]
    * Title: Classifying, Segmenting, and Tracking Object Instances in Video with Mask Propagation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Gedas Bertasius,  Lorenzo Torresani
    * Abstract: We introduce a method for simultaneously classifying, segmenting and tracking object instances in a video sequence. Our method, named MaskProp, adapts the popular Mask R-CNN to video by adding a mask propagation branch that propagates frame-level object instance masks from each video frame to all the other frames in a video clip. This allows our system to predict clip-level instance tracks with respect to the object instances segmented in the middle frame of the clip. Clip-level instance tracks generated densely for each frame in the sequence are finally aggregated to produce video-level object instance segmentation and classification. Our experiments demonstrate that our clip-level instance segmentation makes our approach robust to motion blur and object occlusions in video. MaskProp achieves the best reported accuracy on the YouTube-VIS dataset, outperforming the ICCV 2019 video instance segmentation challenge winner despite being much simpler and using orders of magnitude less labeled data (1.3M vs 1B images and 860K vs 14M bounding boxes). The project page is at: https://gberta.github.io/maskprop/.

count=1
* BANet: Bidirectional Aggregation Network With Occlusion Handling for Panoptic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_BANet_Bidirectional_Aggregation_Network_With_Occlusion_Handling_for_Panoptic_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_BANet_Bidirectional_Aggregation_Network_With_Occlusion_Handling_for_Panoptic_Segmentation_CVPR_2020_paper.pdf)]
    * Title: BANet: Bidirectional Aggregation Network With Occlusion Handling for Panoptic Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Yifeng Chen,  Guangchen Lin,  Songyuan Li,  Omar Bourahla,  Yiming Wu,  Fangfang Wang,  Junyi Feng,  Mingliang Xu,  Xi Li
    * Abstract: Panoptic segmentation aims to perform instance segmentation for foreground instances and semantic segmentation for background stuff simultaneously. The typical top-down pipeline concentrates on two key issues: 1) how to effectively model the intrinsic interaction between semantic segmentation and instance segmentation, and 2) how to properly handle occlusion for panoptic segmentation. Intuitively, the complementarity between semantic segmentation and instance segmentation can be leveraged to improve the performance. Besides, we notice that using detection/mask scores is insufficient for resolving the occlusion problem. Motivated by these observations, we propose a novel deep panoptic segmentation scheme based on a bidirectional learning pipeline. Moreover, we introduce a plug-and-play occlusion handling algorithm to deal with the occlusion between different object instances. The experimental results on COCO panoptic benchmark validate the effectiveness of our proposed method. Codes will be released soon at https://github.com/Mooonside/BANet.

count=1
* Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.pdf)]
    * Title: Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Bowen Cheng,  Maxwell D. Collins,  Yukun Zhu,  Ting Liu,  Thomas S. Huang,  Hartwig Adam,  Liang-Chieh Chen
    * Abstract: In this work, we introduce Panoptic-DeepLab, a simple, strong, and fast system for panoptic segmentation, aiming to establish a solid baseline for bottom-up methods that can achieve comparable performance of two-stage methods while yielding fast inference speed. In particular, Panoptic-DeepLab adopts the dual-ASPP and dual-decoder structures specific to semantic, and instance segmentation, respectively. The semantic segmentation branch is the same as the typical design of any semantic segmentation model (e.g., DeepLab), while the instance segmentation branch is class-agnostic, involving a simple instance center regression. As a result, our single Panoptic-DeepLab simultaneously ranks first at all three Cityscapes benchmarks, setting the new state-of-art of 84.2% mIoU, 39.0% AP, and 65.5% PQ on test set. Additionally, equipped with MobileNetV3, Panoptic-DeepLab runs nearly in real-time with a single 1025x2049 image (15.8 frames per second), while achieving a competitive performance on Cityscapes (54.1 PQ% on test set). On Mapillary Vistas test set, our ensemble of six models attains 42.7% PQ, outperforming the challenge winner in 2018 by a healthy margin of 1.5%. Finally, our Panoptic-DeepLab also performs on par with several top-down approaches on the challenging COCO dataset. For the first time, we demonstrate a bottom-up approach could deliver state-of-the-art results on panoptic segmentation.

count=1
* Deep Polarization Cues for Transparent Object Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Kalra_Deep_Polarization_Cues_for_Transparent_Object_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kalra_Deep_Polarization_Cues_for_Transparent_Object_Segmentation_CVPR_2020_paper.pdf)]
    * Title: Deep Polarization Cues for Transparent Object Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Agastya Kalra,  Vage Taamazyan,  Supreeth Krishna Rao,  Kartik Venkataraman,  Ramesh Raskar,  Achuta Kadambi
    * Abstract: Segmentation of transparent objects is a hard, open problem in computer vision. Transparent objects lack texture of their own, adopting instead the texture of scene background. This paper reframes the problem of transparent object segmentation into the realm of light polarization, i.e., the rotation of light waves. We use a polarization camera to capture multi-modal imagery and couple this with a unique deep learning backbone for processing polarization input data. Our method achieves instance segmentation on cluttered, transparent objects in various scene and background conditions, demonstrating an improvement over traditional image-based approaches. As an application we use this for robotic bin picking of transparent objects.

count=1
* Unsupervised Instance Segmentation in Microscopy Images via Panoptic Domain Adaptation and Task Re-Weighting
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Unsupervised_Instance_Segmentation_in_Microscopy_Images_via_Panoptic_Domain_Adaptation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Unsupervised_Instance_Segmentation_in_Microscopy_Images_via_Panoptic_Domain_Adaptation_CVPR_2020_paper.pdf)]
    * Title: Unsupervised Instance Segmentation in Microscopy Images via Panoptic Domain Adaptation and Task Re-Weighting
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Dongnan Liu,  Donghao Zhang,  Yang Song,  Fan Zhang,  Lauren O'Donnell,  Heng Huang,  Mei Chen,  Weidong Cai
    * Abstract: Unsupervised domain adaptation (UDA) for nuclei instance segmentation is important for digital pathology, as it alleviates the burden of labor-intensive annotation and domain shift across datasets. In this work, we propose a Cycle Consistency Panoptic Domain Adaptive Mask R-CNN (CyC-PDAM) architecture for unsupervised nuclei segmentation in histopathology images, by learning from fluorescence microscopy images. More specifically, we first propose a nuclei inpainting mechanism to remove the auxiliary generated objects in the synthesized images. Secondly, a semantic branch with a domain discriminator is designed to achieve panoptic-level domain adaptation. Thirdly, in order to avoid the influence of the source-biased features, we propose a task re-weighting mechanism to dynamically add trade-off weights for the task-specific loss functions. Experimental results on three datasets indicate that our proposed method outperforms state-of-the-art UDA methods significantly, and demonstrates a similar performance as fully supervised methods.

count=1
* Instance Shadow Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Instance_Shadow_Detection_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Instance_Shadow_Detection_CVPR_2020_paper.pdf)]
    * Title: Instance Shadow Detection
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Tianyu Wang,  Xiaowei Hu,  Qiong Wang,  Pheng-Ann Heng,  Chi-Wing Fu
    * Abstract: Instance shadow detection is a brand new problem, aiming to find shadow instances paired with object instances. To approach it, we first prepare a new dataset called SOBA, named after Shadow-OBject Association, with 3,623 pairs of shadow and object instances in 1,000 photos, each with individual labeled masks. Second, we design LISA, named after Light-guided Instance Shadow-object Association, an end-to-end framework to automatically predict the shadow and object instances, together with the shadow-object associations and light direction. Then, we pair up the predicted shadow and object instances, and match them with the predicted shadow-object associations to generate the final results. In our evaluations, we formulate a new metric named the shadow-object average precision to measure the performance of our results. Further, we conducted various experiments and demonstrate our method's applicability on light direction estimation and photo editing.

count=1
* Mask Encoding for Single Shot Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Mask_Encoding_for_Single_Shot_Instance_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Mask_Encoding_for_Single_Shot_Instance_Segmentation_CVPR_2020_paper.pdf)]
    * Title: Mask Encoding for Single Shot Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Rufeng Zhang,  Zhi Tian,  Chunhua Shen,  Mingyu You,  Youliang Yan
    * Abstract: To date, instance segmentation is dominated by two-stage methods, as pioneered by Mask R-CNN. In contrast, one-stage alternatives cannot compete with Mask R-CNN in mask AP, mainly due to the difficulty of compactly representing masks, making the design of one-stage methods very challenging. In this work, we propose a simple single-shot instance segmentation framework, termed mask encoding based instance segmentation (MEInst). Instead of predicting the two-dimensional mask directly, MEInst distills it into a compact and fixed-dimensional representation vector, which allows the instance segmentation task to be incorporated into one-stage bounding-box detectors and results in a simple yet efficient instance segmentation framework. The proposed one-stage MEInst achieves 36.4% in mask AP with single-model (ResNeXt-101-FPN backbone) and single-scale testing on the MS-COCO benchmark. We show that the much simpler and flexible one-stage instance segmentation method, can also achieve competitive performance. This framework can be easily adapted for other instance-level recognition tasks. Code is available at: git.io/AdelaiDet

count=1
* Learning Saliency Propagation for Semi-Supervised Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_Learning_Saliency_Propagation_for_Semi-Supervised_Instance_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Learning_Saliency_Propagation_for_Semi-Supervised_Instance_Segmentation_CVPR_2020_paper.pdf)]
    * Title: Learning Saliency Propagation for Semi-Supervised Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Yanzhao Zhou,  Xin Wang,  Jianbin Jiao,  Trevor Darrell,  Fisher Yu
    * Abstract: Instance segmentation is a challenging task for both modeling and annotation. Due to the high annotation cost, modeling becomes more difficult because of the limited amount of supervision. We aim to improve the accuracy of the existing instance segmentation models by utilizing a large amount of detection supervision. We propose ShapeProp, which learns to activate the salient regions within the object detection and propagate the areas to the whole instance through an iterative learnable message passing module. ShapeProp can benefit from more bounding box supervision to locate the instances more accurately and utilize the feature activations from the larger number of instances to achieve more accurate segmentation. We extensively evaluate ShapeProp on three datasets (MS COCO, PASCAL VOC, and BDD100k) with different supervision setups based on both two-stage (Mask R-CNN) and single-stage (RetinaMask) models. The results show our method establishes new states of the art for semi-supervised instance segmentation.

count=1
* Incremental Few-Shot Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Ganea_Incremental_Few-Shot_Instance_Segmentation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Ganea_Incremental_Few-Shot_Instance_Segmentation_CVPR_2021_paper.pdf)]
    * Title: Incremental Few-Shot Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Dan Andrei Ganea, Bas Boom, Ronald Poppe
    * Abstract: Few-shot instance segmentation methods are promising when labeled training data for novel classes is scarce. However, current approaches do not facilitate flexible addition of novel classes. They also require that examples of each class are provided at train and test time, which is memory intensive. In this paper, we address these limitations by presenting the first incremental approach to few-shot instance segmentation: iMTFA. We learn discriminative embeddings for object instances that are merged into class representatives. Storing embedding vectors rather than images effectively solves the memory overhead problem. We match these class embeddings at the RoI-level using cosine similarity. This allows us to add new classes without the need for further training or access to previous training data. In a series of experiments, we consistently outperform the current state-of-the-art. Moreover, the reduced memory requirements allow us to evaluate, for the first time, few-shot instance segmentation performance on all classes in COCO jointly.

count=1
* Towards Fast and Accurate Real-World Depth Super-Resolution: Benchmark Dataset and Baseline
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/He_Towards_Fast_and_Accurate_Real-World_Depth_Super-Resolution_Benchmark_Dataset_and_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/He_Towards_Fast_and_Accurate_Real-World_Depth_Super-Resolution_Benchmark_Dataset_and_CVPR_2021_paper.pdf)]
    * Title: Towards Fast and Accurate Real-World Depth Super-Resolution: Benchmark Dataset and Baseline
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Lingzhi He, Hongguang Zhu, Feng Li, Huihui Bai, Runmin Cong, Chunjie Zhang, Chunyu Lin, Meiqin Liu, Yao Zhao
    * Abstract: Depth maps obtained by commercial depth sensors are always in low-resolution, making it difficult to be used in various computer vision tasks. Thus, depth map super-resolution (SR) is a practical and valuable task, which upscales the depth map into high-resolution (HR) space. However, limited by the lack of real-world paired low-resolution (LR) and HR depth maps, most existing methods use downsampling to obtain paired training samples. To this end, we first construct a large-scale dataset named "RGB-D-D", which can greatly promote the study of depth map SR and even more depth-related real-world tasks. The "D-D" in our dataset represents the paired LR and HR depth maps captured from mobile phone and Lucid Helios respectively ranging from indoor scenes to challenging outdoor scenes. Besides, we provide a fast depth map super-resolution (FDSR) baseline, in which the high-frequency component adaptively decomposed from RGB image to guide the depth map SR. Extensive experiments on existing public datasets demonstrate the effectiveness and efficiency of our network compared with the state-of-the-art methods. Moreover, for the real-world LR depth maps, our algorithm can produce more accurate HR depth maps with clearer boundaries and to some extent correct the depth value errors.

count=1
* Deep Occlusion-Aware Instance Segmentation With Overlapping BiLayers
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Ke_Deep_Occlusion-Aware_Instance_Segmentation_With_Overlapping_BiLayers_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Ke_Deep_Occlusion-Aware_Instance_Segmentation_With_Overlapping_BiLayers_CVPR_2021_paper.pdf)]
    * Title: Deep Occlusion-Aware Instance Segmentation With Overlapping BiLayers
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Lei Ke, Yu-Wing Tai, Chi-Keung Tang
    * Abstract: Segmenting highly-overlapping objects is challenging, because typically no distinction is made between real object contours and occlusion boundaries. Unlike previous two-stage instance segmentation methods, we model image formation as composition of two overlapping layers, and propose Bilayer Convolutional Network (BCNet), where the top GCN layer detects the occluding objects (occluder) and the bottom GCN layer infers partially occluded instance (occludee). The explicit modeling of occlusion relationship with bilayer structure naturally decouples the boundaries of both the occluding and occluded instances, and considers the interaction between them during mask regression. We validate the efficacy of bilayer decoupling on both one-stage and two-stage object detectors with different backbones and network layer choices. Despite its simplicity, extensive experiments on COCO and KINS show that our occlusion-aware BCNet achieves large and consistent performance gain especially for heavy occlusion cases. Code is available at https://github.com/lkeab/BCNet.

count=1
* FAPIS: A Few-Shot Anchor-Free Part-Based Instance Segmenter
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Nguyen_FAPIS_A_Few-Shot_Anchor-Free_Part-Based_Instance_Segmenter_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Nguyen_FAPIS_A_Few-Shot_Anchor-Free_Part-Based_Instance_Segmenter_CVPR_2021_paper.pdf)]
    * Title: FAPIS: A Few-Shot Anchor-Free Part-Based Instance Segmenter
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Khoi Nguyen, Sinisa Todorovic
    * Abstract: This paper is about few-shot instance segmentation, where training and test image sets do not share the same object classes. We specify and evaluate a new few-shot anchor-free part-based instance segmenter (FAPIS). Our key novelty is in explicit modeling of latent object parts shared across training object classes, which is expected to facilitate our few-shot learning on new classes in testing. We specify a new anchor-free object detector aimed at scoring and regressing locations of foreground bounding boxes, as well as estimating relative importance of latent parts within each box. Also, we specify a new network for delineating and weighting latent parts for the final instance segmentation within every detected bounding box. Our evaluation on the benchmark COCO-20i dataset demonstrates that we significantly outperform the state of the art.

count=1
* Tuning IR-Cut Filter for Illumination-Aware Spectral Reconstruction From RGB
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Tuning_IR-Cut_Filter_for_Illumination-Aware_Spectral_Reconstruction_From_RGB_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Tuning_IR-Cut_Filter_for_Illumination-Aware_Spectral_Reconstruction_From_RGB_CVPR_2021_paper.pdf)]
    * Title: Tuning IR-Cut Filter for Illumination-Aware Spectral Reconstruction From RGB
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Bo Sun, Junchi Yan, Xiao Zhou, Yinqiang Zheng
    * Abstract: To reconstruct spectral signals from multi-channel observations, in particular trichromatic RGBs, has recently emerged as a promising alternative to traditional scanning-based spectral imager. It has been proven that the reconstruction accuracy relies heavily on the spectral response of the RGB camera in use. To improve accuracy, data-driven algorithms have been proposed to retrieve the best response curves of existing RGB cameras, or even to design brand new three-channel response curves. Instead, this paper explores the filter-array based color imaging mechanism of existing RGB cameras, and proposes to design the IR-cut filter properly for improved spectral recovery, which stands out as an in-between solution with better trade-off between reconstruction accuracy and implementation complexity. We further propose a deep learning based spectral reconstruction method, which allows to recover the illumination spectrum as well. Experiment results with both synthetic and real images under daylight illumination have shown the benefits of our IR-cut filter tuning method and our illumination-aware spectral reconstruction method.

count=1
* Look Closer To Segment Better: Boundary Patch Refinement for Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.pdf)]
    * Title: Look Closer To Segment Better: Boundary Patch Refinement for Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Chufeng Tang, Hang Chen, Xiao Li, Jianmin Li, Zhaoxiang Zhang, Xiaolin Hu
    * Abstract: Tremendous efforts have been made on instance segmentation but the mask quality is still not satisfactory. The boundaries of predicted instance masks are usually imprecise due to the low spatial resolution of feature maps and the imbalance problem caused by the extremely low proportion of boundary pixels. To address these issues, we propose a conceptually simple yet effective post-processing refinement framework to improve the boundary quality based on the results of any instance segmentation model, termed BPR. Following the idea of looking closer to segment boundaries better, we extract and refine a series of small boundary patches along the predicted instance boundaries. The refinement is accomplished by a boundary patch refinement network at higher resolution. The proposed BPR framework yields significant improvements over the Mask R-CNN baseline on Cityscapes benchmark, especially on the boundary-aware metrics. Moreover, by applying the BPR framework to the PolyTransform + SegFix baseline, we reached 1st place on the Cityscapes leaderboard.

count=1
* Self-Supervised Learning for Semi-Supervised Temporal Action Proposal
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Self-Supervised_Learning_for_Semi-Supervised_Temporal_Action_Proposal_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Self-Supervised_Learning_for_Semi-Supervised_Temporal_Action_Proposal_CVPR_2021_paper.pdf)]
    * Title: Self-Supervised Learning for Semi-Supervised Temporal Action Proposal
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Xiang Wang, Shiwei Zhang, Zhiwu Qing, Yuanjie Shao, Changxin Gao, Nong Sang
    * Abstract: Self-supervised learning presents a remarkable performance to utilize unlabeled data for various video tasks. In this paper, we focus on applying the power of self-supervised methods to improve semi-supervised action proposal generation. Particularly, we design a Self-supervised Semi-supervised Temporal Action Proposal (SSTAP) framework. The SSTAP contains two crucial branches, i.e., temporal-aware semi-supervised branch and relation-aware self-supervised branch. The semi-supervised branch improves the proposal model by introducing two temporal perturbations, i.e., temporal feature shift and temporal feature flip, in the mean teacher framework. The self-supervised branch defines two pretext tasks, including masked feature reconstruction and clip-order prediction, to learn the relation of temporal clues. By this means, SSTAP can better explore unlabeled videos, and improve the discriminative abilities of learned action features. We extensively evaluate the proposed SSTAP on THUMOS14 and ActivityNet v1.3 datasets. The experimental results demonstrate that SSTAP significantly outperforms state-of-the-art semi-supervised methods and even matches fully-supervised methods. The code will be released once this paper is accepted.

count=1
* Rethinking Text Segmentation: A Novel Dataset and a Text-Specific Refinement Approach
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_Rethinking_Text_Segmentation_A_Novel_Dataset_and_a_Text-Specific_Refinement_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Rethinking_Text_Segmentation_A_Novel_Dataset_and_a_Text-Specific_Refinement_CVPR_2021_paper.pdf)]
    * Title: Rethinking Text Segmentation: A Novel Dataset and a Text-Specific Refinement Approach
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Xingqian Xu, Zhifei Zhang, Zhaowen Wang, Brian Price, Zhonghao Wang, Humphrey Shi
    * Abstract: Text segmentation is a prerequisite in many real-world text-related tasks, e.g., text style transfer, and scene text removal. However, facing the lack of high-quality datasets and dedicated investigations, this critical prerequisite has been left as an assumption in many works, and has been largely overlooked by current research. To bridge this gap, we proposed TextSeg, a large-scale fine-annotated text dataset with six types of annotations: word- and character-wise bounding polygons, masks, and transcriptions. We also introduce Text Refinement Network (TexRNet), a novel text segmentation approach that adapts to the unique properties of text, e.g. non-convex boundary, diverse texture, etc., which often impose burdens on traditional segmentation models. In our TexRNet, we propose text-specific network designs to address such challenges, including key features pooling and attention-based similarity checking. We also introduce trimap and discriminator losses that show significant improvement in text segmentation. Extensive experiments are carried out on both our TextSeg dataset and other existing datasets. We demonstrate that TexRNet consistently improves text segmentation performance by nearly 2% compared to other state-of-the-art segmentation methods. Our dataset and code can be found at https://github.com/SHI-Labs/Rethinking-Text-Segmentation.

count=1
* Generalized Unsupervised Clustering of Hyperspectral Images of Geological Targets in the Near Infrared
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/PBVS/html/Gao_Generalized_Unsupervised_Clustering_of_Hyperspectral_Images_of_Geological_Targets_in_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/PBVS/papers/Gao_Generalized_Unsupervised_Clustering_of_Hyperspectral_Images_of_Geological_Targets_in_CVPRW_2021_paper.pdf)]
    * Title: Generalized Unsupervised Clustering of Hyperspectral Images of Geological Targets in the Near Infrared
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Angela F. Gao, Brandon Rasmussen, Peter Kulits, Eva L. Scheller, Rebecca Greenberger, Bethany L. Ehlmann
    * Abstract: The application of infrared hyperspectral imagery to geological problems is becoming more popular as data become more accessible and cost-effective. Clustering and classifying spectrally similar materials is often a first step in applications ranging from economic mineral exploration on Earth to planetary exploration on Mars. Semi-manual classification guided by expertly developed spectral parameters can be time consuming and biased, while supervised methods require abundant labeled data and can be difficult to generalize. Here we develop a fully unsupervised workflow for feature extraction and clustering informed by both expert spectral geologist input and quantitative metrics. Our pipeline uses a lightweight autoencoder followed by Gaussian mixture modeling to map the spectral diversity within any image. We validate the performance of our pipeline at submillimeter-scale with expert-labelled data from the Oman ophiolite drill core and evaluate performance at meters-scale with partially classified orbital data of Jezero Crater on Mars (the landing site for the Perseverance rover). We additionally examine the effects of various preprocessing techniques used in traditional analysis of hyperspectral imagery. This pipeline provides a fast and accurate clustering map of similar geological materials and consistently identifies and separates major mineral classes in both laboratory imagery and remote sensing imagery. We refer to our pipeline as ""Generalized Pipeline for Spectroscopic Unsupervised clustering of Minerals (GyPSUM)."

count=1
* Pointly-Supervised Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Pointly-Supervised_Instance_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Pointly-Supervised_Instance_Segmentation_CVPR_2022_paper.pdf)]
    * Title: Pointly-Supervised Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Bowen Cheng, Omkar Parkhi, Alexander Kirillov
    * Abstract: We propose an embarrassingly simple point annotation scheme to collect weak supervision for instance segmentation. In addition to bounding boxes, we collect binary labels for a set of points uniformly sampled inside each bounding box. We show that the existing instance segmentation models developed for full mask supervision can be seamlessly trained with point-based supervision collected via our scheme. Remarkably, Mask R-CNN trained on COCO, PASCAL VOC, Cityscapes, and LVIS with only 10 annotated random points per object achieves 94%-98% of its fully-supervised performance, setting a strong baseline for weakly-supervised instance segmentation. The new point annotation scheme is approximately 5 times faster than annotating full object masks, making high-quality instance segmentation more accessible in practice. Inspired by the point-based annotation form, we propose a modification to PointRend instance segmentation module. For each object, the new architecture, called Implicit PointRend, generates parameters for a function that makes the final point-level mask prediction. Implicit PointRend is more straightforward and uses a single point-level mask loss. Our experiments show that the new module is more suitable for the point-based supervision.

count=1
* Can Neural Nets Learn the Same Model Twice? Investigating Reproducibility and Double Descent From the Decision Boundary Perspective
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Somepalli_Can_Neural_Nets_Learn_the_Same_Model_Twice_Investigating_Reproducibility_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Somepalli_Can_Neural_Nets_Learn_the_Same_Model_Twice_Investigating_Reproducibility_CVPR_2022_paper.pdf)]
    * Title: Can Neural Nets Learn the Same Model Twice? Investigating Reproducibility and Double Descent From the Decision Boundary Perspective
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Gowthami Somepalli, Liam Fowl, Arpit Bansal, Ping Yeh-Chiang, Yehuda Dar, Richard Baraniuk, Micah Goldblum, Tom Goldstein
    * Abstract: We discuss methods for visualizing neural network decision boundaries and decision regions. We use these visualizations to investigate issues related to reproducibility and generalization in neural network training. We observe that changes in model architecture (and its associate inductive bias) cause visible changes in decision boundaries, while multiple runs with the same architecture yield results with strong similarities, especially in the case of wide architectures. We also use decision boundary methods to visualize double descent phenomena. We see that decision boundary reproducibility depends strongly on model width. Near the threshold of interpolation, neural network decision boundaries become fragmented into many small decision regions, and these regions are non-reproducible. Meanwhile, very narrows and very wide networks have high levels of reproducibility in their decision boundaries with relatively few decision regions. We discuss how our observations relate to the theory of double descent phenomena in convex models. Code is available at https://github.com/somepago/dbViz.

count=1
* Sparse Object-Level Supervision for Instance Segmentation With Pixel Embeddings
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wolny_Sparse_Object-Level_Supervision_for_Instance_Segmentation_With_Pixel_Embeddings_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wolny_Sparse_Object-Level_Supervision_for_Instance_Segmentation_With_Pixel_Embeddings_CVPR_2022_paper.pdf)]
    * Title: Sparse Object-Level Supervision for Instance Segmentation With Pixel Embeddings
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Adrian Wolny, Qin Yu, Constantin Pape, Anna Kreshuk
    * Abstract: Most state-of-the-art instance segmentation methods have to be trained on densely annotated images. While difficult in general, this requirement is especially daunting for biomedical images, where domain expertise is often required for annotation and no large public data collections are available for pre-training. We propose to address the dense annotation bottleneck by introducing a proposal-free segmentation approach based on non-spatial embeddings, which exploits the structure of the learned embedding space to extract individual instances in a differentiable way. The segmentation loss can then be applied directly to instances and the overall pipeline can be trained in a fully- or weakly supervised manner. We consider the challenging case of positive-unlabeled supervision, where a novel self-supervised consistency loss is introduced for the unlabeled parts of the training data. We evaluate the proposed method on 2D and 3D segmentation problems in different microscopy modalities as well as on the Cityscapes and CVPPP instance segmentation benchmarks, achieving state-of-the-art results on the latter.

count=1
* BTS: A Bi-Lingual Benchmark for Text Segmentation in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_BTS_A_Bi-Lingual_Benchmark_for_Text_Segmentation_in_the_Wild_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_BTS_A_Bi-Lingual_Benchmark_for_Text_Segmentation_in_the_Wild_CVPR_2022_paper.pdf)]
    * Title: BTS: A Bi-Lingual Benchmark for Text Segmentation in the Wild
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Xixi Xu, Zhongang Qi, Jianqi Ma, Honglun Zhang, Ying Shan, Xiaohu Qie
    * Abstract: As a prerequisite of many text-related tasks such as text erasing and text style transfer, text segmentation arouses more and more attention recently. Current researches mainly focus on only English characters and digits, while few work studies Chinese characters due to the lack of public large-scale and high-quality Chinese datasets, which limits the practical application scenarios of text segmentation. Different from English which has a limited alphabet of letters, Chinese has much more basic characters with complex structures, making the problem more difficult to deal with. To better analyze this problem, we propose the Bi-lingual Text Segmentation (BTS) dataset, a benchmark that covers various common Chinese scenes including 14,250 diverse and fine-annotated text images. BTS mainly focuses on Chinese characters, and also contains English words and digits. We also introduce Prior Guided Text Segmentation Network (PGTSNet), the first baseline to handle bi-lingual and complex-structured text segmentation. A plug-in text region highlighting module and a text perceptual discriminator are proposed in PGTSNet to supervise the model with text prior, and guide for more stable and finer text segmentation. A variation loss is also employed for suppressing background noise under complex scene. Extensive experiments are conducted not only to demonstrate the necessity and superiority of the proposed dataset BTS, but also to show the effectiveness of the proposed PGTSNet compared with a variety of state-of-the-art text segmentation methods.

count=1
* VRDFormer: End-to-End Video Visual Relation Detection With Transformers
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_VRDFormer_End-to-End_Video_Visual_Relation_Detection_With_Transformers_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_VRDFormer_End-to-End_Video_Visual_Relation_Detection_With_Transformers_CVPR_2022_paper.pdf)]
    * Title: VRDFormer: End-to-End Video Visual Relation Detection With Transformers
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Sipeng Zheng, Shizhe Chen, Qin Jin
    * Abstract: Visual relation understanding plays an essential role for holistic video understanding. Most previous works adopt a multi-stage framework for video visual relation detection (VidVRD), which cannot capture long-term spatiotemporal contexts in different stages and also suffers from inefficiency. In this paper, we propose a transformerbased framework called VRDFormer to unify these decoupling stages. Our model exploits a query-based approach to autoregressively generate relation instances. We specifically design static queries and recurrent queries to enable efficient object pair tracking with spatio-temporal contexts. The model is jointly trained with object pair detection and relation classification. Extensive experiments on two benchmark datasets, ImageNet-VidVRD and VidOR, demonstrate the effectiveness of the proposed VRDFormer, which achieves the state-of-the-art performance on both relation detection and relation tagging tasks.

count=1
* Transfer Learning From Synthetic In-Vitro Soybean Pods Dataset for In-Situ Segmentation of On-Branch Soybean Pods
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/AgriVision/html/Yang_Transfer_Learning_From_Synthetic_In-Vitro_Soybean_Pods_Dataset_for_In-Situ_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/AgriVision/papers/Yang_Transfer_Learning_From_Synthetic_In-Vitro_Soybean_Pods_Dataset_for_In-Situ_CVPRW_2022_paper.pdf)]
    * Title: Transfer Learning From Synthetic In-Vitro Soybean Pods Dataset for In-Situ Segmentation of On-Branch Soybean Pods
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Si Yang, Lihua Zheng, Xieyuanli Chen, Laura Zabawa, Man Zhang, Minjuan Wang
    * Abstract: The mature soybean plants are of complex architecture with pods frequently touching each other, posing a challenge for in-situ segmentation of on-branch soybean pods. Deep learning-based methods can achieve accurate training and strong generalization capabilities, but it demands massive labeled data, which is often a limitation, especially for agricultural applications. As lacking the labeled data to train an in-situ segmentation model for on-branch soybean pods, we propose a transfer learning from synthetic in-vitro soybean pods. First, we present a novel automated image generation method to rapidly generate a synthetic in-vitro soybean pods dataset with plenty of annotated samples. The in-vitro soybean pods samples are overlapped to simulate the frequently physically touching of on-branch soybean pods. Then, we design a two-step transfer learning. In the first step, we finetune an instance segmentation network pretrained by a source domain (MS COCO dataset) with a synthetic target domain (in-vitro soybean pods dataset). In the second step, transferring from simulation to reality is performed by finetuning on a few real-world mature soybean plant samples. The experimental results show the effectiveness of the proposed two-step transfer learning method, such that AP50 was 0.80 for the real-world mature soybean plant test dataset, which is higher than that of direct adaptation and its AP50 was 0.77. Furthermore, the visualizations of in-situ segmentation results of on-branch soybean pods show that our method performs better than other methods, especially when soybean pods overlap densely.

count=1
* Multi-Class Cell Detection Using Modified Self-Attention
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/CVMI/html/Sugimoto_Multi-Class_Cell_Detection_Using_Modified_Self-Attention_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/CVMI/papers/Sugimoto_Multi-Class_Cell_Detection_Using_Modified_Self-Attention_CVPRW_2022_paper.pdf)]
    * Title: Multi-Class Cell Detection Using Modified Self-Attention
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Tatsuhiko Sugimoto, Hiroaki Ito, Yuki Teramoto, Akihiko Yoshizawa, Ryoma Bise
    * Abstract: Multi-class cell detection (cancer or non-cancer) from a whole slide image (WSI) is an important task for pathological diagnosis. Cancer and non-cancer cells often have a similar appearance, so it is difficult even for experts to classify a cell from a patch image of individual cells. They usually identify the cell type not only on the basis of the appearance of a single cell but also on the context from the surrounding cells. For using such information, we propose a multi-class cell-detection method that introduces a modified self-attention to aggregate the surrounding image features of both classes. Experimental results demonstrate the effectiveness of the proposed method; our method achieved the best performance compared with a method, which simply use the standard self-attention method.

count=1
* Multi-Concept Customization of Text-to-Image Diffusion
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf)]
    * Title: Multi-Concept Customization of Text-to-Image Diffusion
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu
    * Abstract: While generative models produce high-quality images of concepts learned from a large-scale database, a user often wishes to synthesize instantiations of their own concepts (for example, their family, pets, or items). Can we teach a model to quickly acquire a new concept, given a few examples? Furthermore, can we compose multiple new concepts together? We propose Custom Diffusion, an efficient method for augmenting existing text-to-image models. We find that only optimizing a few parameters in the text-to-image conditioning mechanism is sufficiently powerful to represent new concepts while enabling fast tuning ( 6 minutes). Additionally, we can jointly train for multiple concepts or combine multiple fine-tuned models into one via closed-form constrained optimization. Our fine-tuned model generates variations of multiple new concepts and seamlessly composes them with existing concepts in novel settings. Our method outperforms or performs on par with several baselines and concurrent works in both qualitative and quantitative evaluations, while being memory and computationally efficient.

count=1
* SUDS: Scalable Urban Dynamic Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Turki_SUDS_Scalable_Urban_Dynamic_Scenes_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Turki_SUDS_Scalable_Urban_Dynamic_Scenes_CVPR_2023_paper.pdf)]
    * Title: SUDS: Scalable Urban Dynamic Scenes
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Haithem Turki, Jason Y. Zhang, Francesco Ferroni, Deva Ramanan
    * Abstract: We extend neural radiance fields (NeRFs) to dynamic large-scale urban scenes. Prior work tends to reconstruct single video clips of short durations (up to 10 seconds). Two reasons are that such methods (a) tend to scale linearly with the number of moving objects and input videos because a separate model is built for each and (b) tend to require supervision via 3D bounding boxes and panoptic labels, obtained manually or via category-specific models. As a step towards truly open-world reconstructions of dynamic cities, we introduce two key innovations: (a) we factorize the scene into three separate hash table data structures to efficiently encode static, dynamic, and far-field radiance fields, and (b) we make use of unlabeled target signals consisting of RGB images, sparse LiDAR, off-the-shelf self-supervised 2D descriptors, and most importantly, 2D optical flow. Operationalizing such inputs via photometric, geometric, and feature-metric reconstruction losses enables SUDS to decompose dynamic scenes into the static background, individual objects, and their motions. When combined with our multi-branch table representation, such reconstructions can be scaled to tens of thousands of objects across 1.2 million frames from 1700 videos spanning geospatial footprints of hundreds of kilometers, (to our knowledge) the largest dynamic NeRF built to date. We present qualitative initial results on a variety of tasks enabled by our representations, including novel-view synthesis of dynamic urban scenes, unsupervised 3D instance segmentation, and unsupervised 3D cuboid detection. To compare to prior work, we also evaluate on KITTI and Virtual KITTI 2, surpassing state-of-the-art methods that rely on ground truth 3D bounding box annotations while being 10x quicker to train.

count=1
* A Loopback Network for Explainable Microvascular Invasion Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_A_Loopback_Network_for_Explainable_Microvascular_Invasion_Classification_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_A_Loopback_Network_for_Explainable_Microvascular_Invasion_Classification_CVPR_2023_paper.pdf)]
    * Title: A Loopback Network for Explainable Microvascular Invasion Classification
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Shengxuming Zhang, Tianqi Shi, Yang Jiang, Xiuming Zhang, Jie Lei, Zunlei Feng, Mingli Song
    * Abstract: Microvascular invasion (MVI) is a critical factor for prognosis evaluation and cancer treatment. The current diagnosis of MVI relies on pathologists to manually find out cancerous cells from hundreds of blood vessels, which is time-consuming, tedious, and subjective. Recently, deep learning has achieved promising results in medical image analysis tasks. However, the unexplainability of black box models and the requirement of massive annotated samples limit the clinical application of deep learning based diagnostic methods. In this paper, aiming to develop an accurate, objective, and explainable diagnosis tool for MVI, we propose a Loopback Network (LoopNet) for classifying MVI efficiently. With the image-level category annotations of the collected Pathologic Vessel Image Dataset (PVID), LoopNet is devised to be composed binary classification branch and cell locating branch. The latter is devised to locate the area of cancerous cells, regular non-cancerous cells, and background. For healthy samples, the pseudo masks of cells supervise the cell locating branch to distinguish the area of regular non-cancerous cells and background. For each MVI sample, the cell locating branch predicts the mask of cancerous cells. Then the masked cancerous and non-cancerous areas of the same sample are inputted back to the binary classification branch separately. The loopback between two branches enables the category label to supervise the cell locating branch to learn the locating ability for cancerous areas. Experiment results show that the proposed LoopNet achieves 97.5% accuracy on MVI classification. Surprisingly, the proposed loopback mechanism not only enables LoopNet to predict the cancerous area but also facilitates the classification backbone to achieve better classification performance.

count=1
* Diversified and Multi-Class Controllable Industrial Defect Synthesis for Data Augmentation and Transfer
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Wei_Diversified_and_Multi-Class_Controllable_Industrial_Defect_Synthesis_for_Data_Augmentation_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/VISION/papers/Wei_Diversified_and_Multi-Class_Controllable_Industrial_Defect_Synthesis_for_Data_Augmentation_CVPRW_2023_paper.pdf)]
    * Title: Diversified and Multi-Class Controllable Industrial Defect Synthesis for Data Augmentation and Transfer
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jing Wei, Fei Shen, Chengkan Lv, Zhengtao Zhang, Feng Zhang, Huabin Yang
    * Abstract: Data augmentation is crucial to solve few-sample issues in industrial inspection based on deep learning. However, current industrial data augmentation methods have not yet demonstrated on-par ability in the synthesis of complex defects with pixel-level annotations. This paper proposes a new defect synthesis framework to fill the gap. Firstly, DCDGANc (Diversified and multi-class Controllable Defect Generation Adversarial Networks based on constant source images) is proposed to employ class labels to construct source inputs to control the category and random codes to generate diversified styles of defects. DCDGANc can generate defect content images with pure backgrounds, which avoids the influence of non-defect information and makes it easy to obtain binary masks by segmentation. Secondly, the Poisson blending is improved to avoid content loss when blending generated defect contents to the normal backgrounds. Finally, the complete defect samples and accurate pixel-level annotations are obtained by fine image processing. Experiments are conducted to verify the effectiveness of our work in wood, fabric, metal, and marble. The results show that our methods yield significant improvement in the segmentation performance of industrial products. Moreover, our work enables zero-shot inspection by facilitating defect transfer between datasets with different backgrounds but similar defects, which can greatly reduce the cost of data collection in industrial inspection.

count=1
* Cross-Dimension Affinity Distillation for 3D EM Neuron Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Cross-Dimension_Affinity_Distillation_for_3D_EM_Neuron_Segmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Cross-Dimension_Affinity_Distillation_for_3D_EM_Neuron_Segmentation_CVPR_2024_paper.pdf)]
    * Title: Cross-Dimension Affinity Distillation for 3D EM Neuron Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xiaoyu Liu, Miaomiao Cai, Yinda Chen, Yueyi Zhang, Te Shi, Ruobing Zhang, Xuejin Chen, Zhiwei Xiong
    * Abstract: Accurate 3D neuron segmentation from electron microscopy (EM) volumes is crucial for neuroscience research. However the complex neuron morphology often leads to over-merge and over-segmentation results. Recent advancements utilize 3D CNNs to predict a 3D affinity map with improved accuracy but suffer from two challenges: high computational cost and limited input size especially for practical deployment for large-scale EM volumes. To address these challenges we propose a novel method to leverage lightweight 2D CNNs for efficient neuron segmentation. Our method employs a 2D Y-shape network to generate two embedding maps from adjacent 2D sections which are then converted into an affinity map by measuring their embedding distance. While the 2D network better captures pixel dependencies inside sections with larger input sizes it overlooks inter-section dependencies. To overcome this we introduce a cross-dimension affinity distillation (CAD) strategy that transfers inter-section dependency knowledge from a 3D teacher network to the 2D student network by ensuring consistency between their output affinity maps. Additionally we design a feature grafting interaction (FGI) module to enhance knowledge transfer by grafting embedding maps from the 2D student onto those from the 3D teacher. Extensive experiments on multiple EM neuron segmentation datasets including a newly built one by ourselves demonstrate that our method achieves superior performance over state-of-the-art methods with only 1/20 inference latency.

count=1
* FISBe: A Real-World Benchmark Dataset for Instance Segmentation of Long-Range Thin Filamentous Structures
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Mais_FISBe_A_Real-World_Benchmark_Dataset_for_Instance_Segmentation_of_Long-Range_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Mais_FISBe_A_Real-World_Benchmark_Dataset_for_Instance_Segmentation_of_Long-Range_CVPR_2024_paper.pdf)]
    * Title: FISBe: A Real-World Benchmark Dataset for Instance Segmentation of Long-Range Thin Filamentous Structures
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Lisa Mais, Peter Hirsch, Claire Managan, Ramya Kandarpa, Josef Lorenz Rumberger, Annika Reinke, Lena Maier-Hein, Gudrun Ihrke, Dagmar Kainmueller
    * Abstract: Instance segmentation of neurons in volumetric light microscopy images of nervous systems enables groundbreaking research in neuroscience by facilitating joint functional and morphological analyses of neural circuits at cellular resolution. Yet said multi-neuron light microscopy data exhibits extremely challenging properties for the task of instance segmentation: Individual neurons have long-ranging thin filamentous and widely branching morphologies multiple neurons are tightly inter-weaved and partial volume effects uneven illumination and noise inherent to light microscopy severely impede local disentangling as well as long-range tracing of individual neurons. These properties reflect a current key challenge in machine learning research namely to effectively capture long-range dependencies in the data. While respective methodological research is buzzing to date methods are typically benchmarked on synthetic datasets. To address this gap we release the FlyLight Instance Segmentation Benchmark (FISBe) dataset the first publicly available multi-neuron light microscopy dataset with pixel-wise annotations. In addition we define a set of instance segmentation metrics for benchmarking that we designed to be meaningful with regard to downstream analyses. Lastly we provide three baselines to kick off a competition that we envision to both advance the field of machine learning regarding methodology for capturing long-range data dependencies and facilitate scientific discovery in basic neuroscience.

count=1
* InceptionNeXt
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_InceptionNeXt_When_Inception_Meets_ConvNeXt_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_InceptionNeXt_When_Inception_Meets_ConvNeXt_CVPR_2024_paper.pdf)]
    * Title: InceptionNeXt: When Inception Meets ConvNeXt
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Weihao Yu, Pan Zhou, Shuicheng Yan, Xinchao Wang
    * Abstract: Inspired by the long-range modeling ability of ViTs large-kernel convolutions are widely studied and adopted recently to enlarge the receptive field and improve model performance like the remarkable work ConvNeXt which employs 7x7 depthwise convolution. Although such depthwise operator only consumes a few FLOPs it largely harms the model efficiency on powerful computing devices due to the high memory access costs. For example ConvNeXt-T has similar FLOPs with ResNet-50 but only achieves 60% throughputs when trained on A100 GPUs with full precision. Although reducing the kernel size of ConvNeXt can improve speed it results in significant performance degradation which poses a challenging problem: How to speed up large-kernel-based CNN models while preserving their performance. To tackle this issue inspired by Inceptions we propose to decompose large-kernel depthwise convolution into four parallel branches along channel dimension i.e. small square kernel two orthogonal band kernels and an identity mapping. With this new Inception depthwise convolution we build a series of networks namely IncepitonNeXt which not only enjoy high throughputs but also maintain competitive performance. For instance InceptionNeXt-T achieves 1.6x higher training throughputs than ConvNeX-T as well as attains 0.2% top-1 accuracy improvement on ImageNet-1K. We anticipate InceptionNeXt can serve as an economical baseline for future architecture design to reduce carbon footprint.

count=1
* NOISe: Nuclei-Aware Osteoclast Instance Segmentation for Mouse-to-Human Domain Transfer
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/CVMI/html/Manne_NOISe_Nuclei-Aware_Osteoclast_Instance_Segmentation_for_Mouse-to-Human_Domain_Transfer_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/CVMI/papers/Manne_NOISe_Nuclei-Aware_Osteoclast_Instance_Segmentation_for_Mouse-to-Human_Domain_Transfer_CVPRW_2024_paper.pdf)]
    * Title: NOISe: Nuclei-Aware Osteoclast Instance Segmentation for Mouse-to-Human Domain Transfer
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Sai Kumar Reddy Manne, Brendan Martin, Tyler Roy, Ryan Neilson, Rebecca Peters, Meghana Chillara, Christine W. Lary, Katherine J. Motyl, Michael Wan
    * Abstract: Osteoclast cell image analysis plays a key role in osteoporosis research but it typically involves extensive manual image processing and hand annotations by a trained expert. In the last few years a handful of machine learning approaches for osteoclast image analysis have been developed but none have addressed the full instance segmentation task required to produce the same output as that of the human expert led process. Furthermore none of the prior fully automated algorithms have publicly available code pretrained models or annotated datasets inhibiting reproduction and extension of their work. We present a new dataset with 2x10^5 expert annotated mouse osteoclast masks together with a deep learning instance segmentation method which works for both in vitro mouse osteoclast cells on plastic tissue culture plates and human osteoclast cells on bone chips. To our knowledge this is the first work to automate the full osteoclast instance segmentation task. Our method achieves a performance of 0.82 mAP_0.5 (mean average precision at intersection-over-union threshold of 0.5) in cross validation for mouse osteoclasts. We present a novel nuclei-aware osteoclast instance segmentation training strategy (NOISe) based on the unique biology of osteoclasts to improve the model's generalizability and boost the mAP_0.5 from 0.60 to 0.82 on human osteoclasts. We publish our annotated mouse osteoclast image dataset instance segmentation models and code at github.com/michaelwwan/noise to enable reproducibility and to provide a public tool to accelerate osteoporosis research.

count=1
* Uncertainty Estimation for Tumor Prediction with Unlabeled Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/CVMI/html/Yun_Uncertainty_Estimation_for_Tumor_Prediction_with_Unlabeled_Data_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/CVMI/papers/Yun_Uncertainty_Estimation_for_Tumor_Prediction_with_Unlabeled_Data_CVPRW_2024_paper.pdf)]
    * Title: Uncertainty Estimation for Tumor Prediction with Unlabeled Data
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Juyoung Yun, Shahira Abousamra, Chen Li, Rajarsi Gupta, Tahsin Kurc, Dimitris Samaras, Alison Van Dyke, Joel Saltz, Chao Chen
    * Abstract: Estimating uncertainty of a neural network is crucial in providing transparency and trustworthiness. In this paper we focus on uncertainty estimation for digital pathology prediction models. To explore the large amount of unlabeled data in digital pathology we propose to adopt novel learning method that can fully exploit unlabeled data. The proposed method achieves superior performance compared with different baselines including the celebrated Monte-Carlo Dropout. Closeup inspection of uncertain regions reveal insight into the model and improves the trustworthiness of the models.

count=1
* ImplicitTerrain: a Continuous Surface Model for Terrain Data Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/INRV/html/Feng_ImplicitTerrain_a_Continuous_Surface_Model_for_Terrain_Data_Analysis_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/INRV/papers/Feng_ImplicitTerrain_a_Continuous_Surface_Model_for_Terrain_Data_Analysis_CVPRW_2024_paper.pdf)]
    * Title: ImplicitTerrain: a Continuous Surface Model for Terrain Data Analysis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Haoan Feng, Xin Xu, Leila De Floriani
    * Abstract: Digital terrain models (DTMs) are pivotal in remote sensing cartography and landscape management requiring accurate surface representation and topological information restoration. While topology analysis traditionally relies on smooth manifolds the absence of an easy-to-use continuous surface model for a large terrain results in a preference for discrete meshes. Structural representation based on topology provides a succinct surface description laying the foundation for many terrain analysis applications. However on discrete meshes numerical issues emerge and complex algorithms are designed to handle them. This paper brings the context of terrain data analysis back to the continuous world and introduces ImplicitTerrain an implicit neural representation (INR) approach for modeling high-resolution terrain continuously and differentiably. Our comprehensive experiments demonstrate superior surface fitting accuracy effective topological feature retrieval and various topographical feature extraction that are implemented over this compact representation in parallel. To our knowledge ImplicitTerrain pioneers a feasible continuous terrain surface modeling pipeline that provides a new research avenue for our community.

count=1
* IrrNet: Advancing Irrigation Mapping with Incremental Patch Size Training on Remote Sensing Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/html/Hoque_IrrNet_Advancing_Irrigation_Mapping_with_Incremental_Patch_Size_Training_on_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/papers/Hoque_IrrNet_Advancing_Irrigation_Mapping_with_Incremental_Patch_Size_Training_on_CVPRW_2024_paper.pdf)]
    * Title: IrrNet: Advancing Irrigation Mapping with Incremental Patch Size Training on Remote Sensing Imagery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Oishee Bintey Hoque, Samarth Swarup, Abhijin Adiga, Sayjro Kossi Nouwakpo, Madhav Marathe
    * Abstract: Irrigation mapping plays a crucial role in effective water management essential for preserving both water quality and quantity and is key to mitigating the global issue of water scarcity. The complexity of agricultural fields adorned with diverse irrigation practices especially when multiple systems coexist in close quarters poses a unique challenge. This complexity is further compounded by the nature of Landsat's remote sensing data where each pixel is rich with densely packed information complicating the task of accurate irrigation mapping. In this study we introduce an innovative approach that employs a progressive training method which strategically increases patch sizes throughout the training process utilizing datasets from Landsat 5 and 7 labeled with the WRLU dataset for precise labeling. This initial focus allows the model to capture detailed features progressively shifting to broader more general features as the patch size enlarges. Remarkably our method enhances the performance of existing state-of-the-art models by approximately 20%. Furthermore our analysis delves into the significance of incorporating various spectral bands into the model assessing their impact on performance. The findings reveal that additional bands are instrumental in enabling the model to discern finer details more effectively. This work sets a new standard for leveraging remote sensing imagery in irrigation mapping.

count=1
* Surface Parameterization and Registration for Statistical Multiscale Atlasing of Organ Development
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/BIC/Selka_Surface_Parameterization_and_Registration_for_Statistical_Multiscale_Atlasing_of_Organ_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/BIC/Selka_Surface_Parameterization_and_Registration_for_Statistical_Multiscale_Atlasing_of_Organ_CVPRW_2019_paper.pdf)]
    * Title: Surface Parameterization and Registration for Statistical Multiscale Atlasing of Organ Development
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Faical Selka,  Jasmine Burguet,  Eric Biot,  Thomas Blein,  Patrick Laufs,  Philippe Andrey
    * Abstract: During organ development, morphological and topological changes jointly occur at the cellular and tissue levels. Hence, the systematic and integrative quantification of cellular parameters during growth is essential to better understand organogenesis. We developed an atlasing strategy to quantitatively map cellular parameters during organ growth. Our approach is based on the computation of prototypical shapes, which are average shapes of individual organs at successive developmental stages, whereupon statistical descriptors of cellular parameters measured from individual organs are projected. We describe here the algorithmic pipeline we developed for 3D organ shape registration, based on the establishment of an organ-centered coordinate system and on the automatic parameterization of organ surface. Using our framework, dynamic developmental trajectories can be readily reconstructed using point-to-point interpolation between parameterized organ surfaces at different time points. We illustrate and validate our pipeline using 3D confocal images of developing plant leaves.

count=1
* Cell Image Segmentation by Integrating Pix2pixs for Each Class
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/CVMI/Tsuda_Cell_Image_Segmentation_by_Integrating_Pix2pixs_for_Each_Class_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVMI/Tsuda_Cell_Image_Segmentation_by_Integrating_Pix2pixs_for_Each_Class_CVPRW_2019_paper.pdf)]
    * Title: Cell Image Segmentation by Integrating Pix2pixs for Each Class
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Hiroki Tsuda,  Kazuhiro Hotta
    * Abstract: This paper presents a cell image segmentation method using Generative Adversarial Network (GAN) with multiple different roles. Pix2pix is a kind of GAN can be used for image segmentation. However, the accuracy is not sufficient because generator predicts multiple classes simultaneously. Thus, we propose to use multiple GANs with different roles. Each generator and discriminator has a specific role such as segmentation of cell membrane or nucleus. Since we assign each generator and discriminator to a different role, they can learn it efficiently. We evaluate the proposed method on the segmentation problem of cell images. The proposed method improved the segmentation accuracy in comparison to conventional pix2pix.

count=1
* In-Vehicle Occupancy Detection With Convolutional Networks on Thermal Images
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/PBVS/Nowruzi_In-Vehicle_Occupancy_Detection_With_Convolutional_Networks_on_Thermal_Images_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Nowruzi_In-Vehicle_Occupancy_Detection_With_Convolutional_Networks_on_Thermal_Images_CVPRW_2019_paper.pdf)]
    * Title: In-Vehicle Occupancy Detection With Convolutional Networks on Thermal Images
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Farzan Erlik Nowruzi,  Wassim A. El Ahmar,  Robert Laganiere,  Amir H. Ghods
    * Abstract: Counting people is a growing field of interest for researchers in recent years. In-vehicle passenger counting is an interesting problem in this domain that has several applications including High Occupancy Vehicle (HOV) lanes. In this paper, present a new in-vehicle thermal image dataset. We propose a tiny convolutional model to count on-board passengers and compare it to well known methods. We show that our model surpasses state-of-the-art methods in classification and has comparable performance in detection. Moreover, our model outperforms the state-of-the-art architectures in terms of speed, making it suitable for deployment on embedded platforms. We present the results of multiple deep learning models and thoroughly analyze them.

count=1
* Event Detection in Coarsely Annotated Sports Videos via Parallel Multi-Receptive Field 1D Convolutions
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w53/Vats_Event_Detection_in_Coarsely_Annotated_Sports_Videos_via_Parallel_Multi-Receptive_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w53/Vats_Event_Detection_in_Coarsely_Annotated_Sports_Videos_via_Parallel_Multi-Receptive_CVPRW_2020_paper.pdf)]
    * Title: Event Detection in Coarsely Annotated Sports Videos via Parallel Multi-Receptive Field 1D Convolutions
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Kanav Vats, Mehrnaz Fani, Pascale Walters, David A. Clausi, John Zelek
    * Abstract: In problems such as sports video analytics, it is difficult to obtain accurate frame-level annotations and exact event duration because of the lengthy videos and sheer volume of video data. This issue is even more pronounced in fast-paced sports such as ice hockey. Obtaining annotations on a coarse scale can be much more practical and time efficient. We propose the task of event detection in coarsely annotated videos. We introduce a multi-tower temporal convolutional network architecture for the proposed task. The network, with the help of multiple receptive fields, processes information at various temporal scales to account for the uncertainty with regard to the exact event location and duration. We demonstrate the effectiveness of the multi-receptive field architecture through appropriate ablation studies. The method is evaluated on two tasks - event detection in coarsely annotated hockey videos in the NHL dataset and event spotting in soccer on the SoccerNet dataset. The two datasets lack frame-level annotations and have very distinct event frequencies. Experimental results demonstrate the effectiveness of the network by obtaining a 55% average F1 score on the NHL dataset and by achieving competitive performance compared to the state of the art on the SoccerNet dataset. We believe our approach will help develop more practical pipelines for event detection in sports video.

count=1
* CTMC: Cell Tracking With Mitosis Detection Dataset Challenge
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Anjum_CTMC_Cell_Tracking_With_Mitosis_Detection_Dataset_Challenge_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w57/Anjum_CTMC_Cell_Tracking_With_Mitosis_Detection_Dataset_Challenge_CVPRW_2020_paper.pdf)]
    * Title: CTMC: Cell Tracking With Mitosis Detection Dataset Challenge
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Samreen Anjum, Danna Gurari
    * Abstract: While significant developments have been made in cell tracking algorithms, current datasets are still limited in size and diversity, especially for data-hungry generalized deep learning models. We introduce a new larger and more diverse cell tracking dataset in terms of number of sequences, length of sequences, and cell lines, accompanied with a public evaluation server and leaderboard to accelerate progress on this new challenging dataset. Our benchmarking of four top performing tracking algorithms highlights new challenges and opportunities to improve the state-of-the-art in cell tracking.

count=1
* A Non-invasive Method for Measuring Blood Flow Rate in Superficial Veins from a Single Thermal Image
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/html/Mahmoud_A_Non-invasive_Method_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/papers/Mahmoud_A_Non-invasive_Method_2013_CVPR_paper.pdf)]
    * Title: A Non-invasive Method for Measuring Blood Flow Rate in Superficial Veins from a Single Thermal Image
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Ali Mahmoud, Ahmed El-Barkouky, Heba Farag, James Graham, Aly Farag
    * Abstract: In this paper, we propose a thermal image based measurement technique for the volumetric flow rate of a liquid inside a thin tube. Our technique makes use of the convection heat transfer dependency between the flow rate and the temperature of the flowing liquid along the tube. The proposed method can be applied to diagnose superficial venous disease non-invasively by measuring the volumetric blood flow rate from a FLIR LWIR single thermal image.

count=1
* Dynamic Multi-vehicle Detection and Tracking from a Moving Platform
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/html/Lin_Dynamic_Multi-vehicle_Detection_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/papers/Lin_Dynamic_Multi-vehicle_Detection_2013_CVPR_paper.pdf)]
    * Title: Dynamic Multi-vehicle Detection and Tracking from a Moving Platform
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Chung-Ching Lin, Marilyn Wolf
    * Abstract: Recent work has successfully built the object classifier for object detection. Most approaches operate with a predefined class and require a model to be trained in advance. In this paper, we present a system with a novel approach for multi-vehicle detection and tracking by using a monocular camera on a moving platform. This approach requires no camera-intrinsic parameters or camera-motion parameters, which enable the system to be successfully implemented without prior training. In our approach, bottom-up segmentation is applied on the input images to get the superpixels. The scene is parsed into less segmented regions by merging similar superpixels. Then, the parsing results are utilized to estimate the road region and detect vehicles on the road by using the properties of superpixels. Finally, tracking is achieved and fed back to further guide vehicle detection in future frames. Experimental results show that the method demonstrates significant vehicle detecting and tracking performance without further restrictions and performs effectively in complex environments.

count=1
* GPS Refinement and Camera Orientation Estimation from a Single Image and a 2D Map
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W03/html/Chu_GPS_Refinement_and_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W03/papers/Chu_GPS_Refinement_and_2014_CVPR_paper.pdf)]
    * Title: GPS Refinement and Camera Orientation Estimation from a Single Image and a 2D Map
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Hang Chu, Andrew Gallagher, Tsuhan Chen
    * Abstract: A framework is presented for refining GPS location and estimate the camera orientation using a single urban building image, a 2D city map with building outlines, given a noisy GPS location. We propose to use tilt-invariant vertical building corner edges extracted from the building image. A location-orientation hypothesis, which we call an LOH, is a proposed map location from which an image of building corners would occur at the observed positions of corner edges in the photo. The noisy GPS location is refined and orientation is estimated using the computed LOHs. Experiments show the framework improves GPS accuracy significantly, generally produces reliable orientation estimation, and is computationally efficient.

count=1
* Improving Superpixel Boundaries Using Information Beyond the Visual Spectrum
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/Sullivan_Improving_Superpixel_Boundaries_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/Sullivan_Improving_Superpixel_Boundaries_2015_CVPR_paper.pdf)]
    * Title: Improving Superpixel Boundaries Using Information Beyond the Visual Spectrum
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Keith Sullivan, Wallace Lawson, Donald Sofge
    * Abstract: Superpixels enable a scene to be analyzed on a larger scale, by examining regions that have a high level of similarity. These regions can change depending on how similarity is measured. Color is a simple and effective measure, but it is adversely affected in environments where the boundary between objects and the surrounding environment are difficult to detect due to similar colors and/or shadows. We extend a common superpixel algorithm (SLIC) to include near-infrared intensity information and measured distance information to help oversegmentation in complex environments. We demonstrate the efficacy of our approach on two problems: object segmentation and scene segmentation.

count=1
* ICPIK: Inverse Kinematics Based Articulated-ICP
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W15/html/Fleishman_ICPIK_Inverse_Kinematics_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W15/papers/Fleishman_ICPIK_Inverse_Kinematics_2015_CVPR_paper.pdf)]
    * Title: ICPIK: Inverse Kinematics Based Articulated-ICP
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Shachar Fleishman, Mark Kliger, Alon Lerner, Gershom Kutliroff
    * Abstract: In this paper we address the problem of matching a kinematic model of an articulated body to a point cloud obtained from a consumer grade 3D sensor. We present the ICPIK algorithm - an Articulated Iterative Closest Point algorithm based on a solution to the Inverse Kinematic problem. The main virtue of the presented algorithm is its computational efficiency, achieved by relying on inverse-kinematics framework for analytical derivation of the Jacobian matrix, and the enforcement of kinematic constraints. We demonstrate the performance of the ICPIK algorithm by integrating it into a real-time hand tracking system. The presented algorithm achieves similar accuracy as state of the art methods, while significantly reducing computation time.

count=1
* Image Segmentation with Cascaded Hierarchical Models and Logistic Disjunctive Normal Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Seyedhosseini_Image_Segmentation_with_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Seyedhosseini_Image_Segmentation_with_2013_ICCV_paper.pdf)]
    * Title: Image Segmentation with Cascaded Hierarchical Models and Logistic Disjunctive Normal Networks
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Mojtaba Seyedhosseini, Mehdi Sajjadi, Tolga Tasdizen
    * Abstract: Contextual information plays an important role in solving vision problems such as image segmentation. However, extracting contextual information and using it in an effective way remains a difficult problem. To address this challenge, we propose a multi-resolution contextual framework, called cascaded hierarchical model (CHM), which learns contextual information in a hierarchical framework for image segmentation. At each level of the hierarchy, a classifier is trained based on downsampled input images and outputs of previous levels. Our model then incorporates the resulting multi-resolution contextual information into a classifier to segment the input image at original resolution. We repeat this procedure by cascading the hierarchical framework to improve the segmentation accuracy. Multiple classifiers are learned in the CHM; therefore, a fast and accurate classifier is required to make the training tractable. The classifier also needs to be robust against overfitting due to the large number of parameters learned during training. We introduce a novel classification scheme, called logistic disjunctive normal networks (LDNN), which consists of one adaptive layer of feature detectors implemented by logistic sigmoid functions followed by two fixed layers of logical units that compute conjunctions and disjunctions, respectively. We demonstrate that LDNN outperforms state-of-theart classifiers and can be used in the CHM to improve object segmentation performance.

count=1
* Adaptive Exponential Smoothing for Online Filtering of Pixel Prediction Maps
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Dang_Adaptive_Exponential_Smoothing_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Dang_Adaptive_Exponential_Smoothing_ICCV_2015_paper.pdf)]
    * Title: Adaptive Exponential Smoothing for Online Filtering of Pixel Prediction Maps
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Kang Dang, Jiong Yang, Junsong Yuan
    * Abstract: We propose an efficient online video filtering method, called adaptive exponential filtering (AES) to refine pixel prediction maps. Assuming each pixel is associated with a discriminative prediction score, the proposed AES applies exponentially decreasing weights over time to smooth the prediction score of each pixel, similar to classic exponential smoothing. However, instead of fixing the spatial pixel location to perform temporal filtering, we trace each pixel in the past frames by finding the optimal path that can bring the maximum exponential smoothing score, thus performing adaptive and non-linear filtering. Thanks to the pixel tracing, AES can better address object movements and avoid over-smoothing. To enable real-time filtering, we propose a linear-complexity dynamic programming scheme that can trace all pixels simultaneously. We apply the proposed filtering method to improve both saliency detection maps and scene parsing maps. The comparisons with average and exponential filtering, as well as state-of-the-art methods, validate that our AES can effectively refine the pixel prediction maps, without using the original video again.

count=1
* Multiple-Hypothesis Affine Region Estimation With Anisotropic LoG Filters
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Hasegawa_Multiple-Hypothesis_Affine_Region_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Hasegawa_Multiple-Hypothesis_Affine_Region_ICCV_2015_paper.pdf)]
    * Title: Multiple-Hypothesis Affine Region Estimation With Anisotropic LoG Filters
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Takahiro Hasegawa, Mitsuru Ambai, Kohta Ishikawa, Gou Koutaki, Yuji Yamauchi, Takayoshi Yamashita, Hironobu Fujiyoshi
    * Abstract: We propose a method for estimating multiple-hypothesis affine regions from a keypoint by using an anisotropic Laplacian-of-Gaussian (LoG) filter. Although conventional affine region detectors, such as Hessian/Harris-Affine, iterate to find an affine region that fits a given image patch, such iterative searching is adversely affected by an initial point. To avoid this problem, we allow multiple detections from a single keypoint. We demonstrate that the responses of all possible anisotropic LoG filters can be efficiently computed by factorizing them in a similar manner to spectral SIFT. A large number of LoG filters that are densely sampled in a parameter space are reconstructed by a weighted combination of a limited number of representative filters, called ``eigenfilters", by using singular value decomposition. Also, the reconstructed filter responses of the sampled parameters can be interpolated to a continuous representation by using a series of proper functions. This results in efficient multiple extrema searching in a continuous space. Experiments revealed that our method has higher repeatability than the conventional methods.

count=1
* StereoSnakes: Contour Based Consistent Object Extraction For Stereo Images
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Ju_StereoSnakes_Contour_Based_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Ju_StereoSnakes_Contour_Based_ICCV_2015_paper.pdf)]
    * Title: StereoSnakes: Contour Based Consistent Object Extraction For Stereo Images
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Ran Ju, Tongwei Ren, Gangshan Wu
    * Abstract: Consistent object extraction plays an essential role for stereo image editing with the population of stereoscopic 3D media. Most previous methods perform segmentation on entire images for both views using dense stereo correspondence constraints. We find that for such kind of methods the computation is highly redundant since the two views are near-duplicate. Besides, the consistency may be violated due to the imperfectness of current stereo matching algorithms. In this paper, we propose a contour based method which searches for consistent object contours instead of regions. It integrates both stereo correspondence and object boundary constraints into an energy minimization framework. The proposed method has several advantages compared to previous works. First, the searching space is restricted in object boundaries thus the efficiency significantly improved. Second, the discriminative power of object contours results in a more consistent segmentation. Furthermore, the proposed method can effortlessly extend existing single-image segmentation methods to work in stereo scenarios. The experiment on the Adobe benchmark shows superior extraction accuracy and significant improvement of efficiency of our method to state-of-the-art. We also demonstrate in a few applications how our method can be used as a basic tool for stereo image editing.

count=1
* Hyperspectral Super-Resolution by Coupled Spectral Unmixing
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Lanaras_Hyperspectral_Super-Resolution_by_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Lanaras_Hyperspectral_Super-Resolution_by_ICCV_2015_paper.pdf)]
    * Title: Hyperspectral Super-Resolution by Coupled Spectral Unmixing
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Charis Lanaras, Emmanuel Baltsavias, Konrad Schindler
    * Abstract: Hyperspectral cameras capture images with many narrow spectral channels, which densely sample the electromagnetic spectrum. The detailed spectral resolution is useful for many image analysis problems, but it comes at the cost of much lower spatial resolution. Hyperspectral super-resolution addresses this problem, by fusing a low-resolution hyperspectral image and a conventional high-resolution image into a product of both high spatial and high spectral resolution. In this paper, we propose a method which performs hyperspectral super-resolution by jointly unmixing the two input images into the pure reflectance spectra of the observed materials and the associated mixing coefficients. The formulation leads to a coupled matrix factorisation problem, with a number of useful constraints imposed by elementary physical properties of spectral mixing. In experiments with two benchmark datasets we show that the proposed approach delivers improved hyperspectral super-resolution.

count=1
* Learning to Combine Mid-Level Cues for Object Proposal Generation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Lee_Learning_to_Combine_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Lee_Learning_to_Combine_ICCV_2015_paper.pdf)]
    * Title: Learning to Combine Mid-Level Cues for Object Proposal Generation
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Tom Lee, Sanja Fidler, Sven Dickinson
    * Abstract: In recent years, region proposals have replaced sliding windows in support of object recognition, offering more discriminating shape and appearance information through improved localization. One powerful approach for generating region proposals is based on minimizing parametric energy functions with parametric maxflow. In this paper, we introduce Parametric Min-Loss (PML), a novel structured learning framework for parametric energy functions. While PML is generally applicable to different domains, we use it in the context of region proposals to learn to combine a set of mid-level grouping cues to yield a small set of object region proposals with high recall. Our learning framework accounts for multiple diverse outputs, and is complemented by diversification seeds based on image location and color. This approach casts perceptual grouping and cue combination in a novel structured learning framework which yields baseline improvements on VOC 2012 and COCO 2014.

count=1
* Projection Onto the Manifold of Elongated Structures for Accurate Extraction
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Sironi_Projection_Onto_the_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Sironi_Projection_Onto_the_ICCV_2015_paper.pdf)]
    * Title: Projection Onto the Manifold of Elongated Structures for Accurate Extraction
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Amos Sironi, Vincent Lepetit, Pascal Fua
    * Abstract: Detection of elongated structures in 2D images and 3D image stacks is a critical prerequisite in many applications and Machine Learning-based approaches have recently been shown to deliver superior performance. However, these methods essentially classify individual locations and do not explicitly model the strong relationship that exists between neighboring ones. As a result, isolated erroneous responses, discontinuities, and topological errors are present in the resulting score maps. We solve this problem by projecting patches of the score map to their nearest neighbors in a set of ground truth training patches. Our algorithm induces global spatial consistency on the classifier score map and returns results that are provably geometrically consistent. We apply our algorithm to challenging datasets in four different domains and show that it compares favorably to state-of-the-art methods.

count=1
* Piecewise Flat Embedding for Image Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Yu_Piecewise_Flat_Embedding_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Yu_Piecewise_Flat_Embedding_ICCV_2015_paper.pdf)]
    * Title: Piecewise Flat Embedding for Image Segmentation
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Yizhou Yu, Chaowei Fang, Zicheng Liao
    * Abstract: Image segmentation is a critical step in many computer vision tasks, including high-level visual recognition and scene understanding as well as low-level photo and video processing. In this paper, we propose a new nonlinear embedding, called piecewise flat embedding, for image segmentation. Based on the theory of sparse signal recovery, piecewise flat embedding attempts to identify segment boundaries while significantly suppressing variations within segments. We adopt an L1-regularized energy term in the formulation to promote sparse solutions. We further devise an effective two-stage numerical algorithm based on Bregman iterations to solve the proposed embedding. Piecewise flat embedding can be easily integrated into existing image segmentation frameworks, including segmentation based on spectral clustering and hierarchical segmentation based on contour detection. Experiments on BSDS500 indicate that segmentation algorithms incorporating this embedding can achieve significantly improved results in both frameworks.

count=1
* Solving Large Multicut Problems for Connectomics via Domain Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Pape_Solving_Large_Multicut_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w1/Pape_Solving_Large_Multicut_ICCV_2017_paper.pdf)]
    * Title: Solving Large Multicut Problems for Connectomics via Domain Decomposition
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Constantin Pape, Thorsten Beier, Peter Li, Viren Jain, Davi D. Bock, Anna Kreshuk
    * Abstract: In this contribution we demonstrate how a Multicut- based segmentation pipeline can be scaled up to datasets of hundreds of Gigabytes in size. Such datasets are preva- lent in connectomics, where neuron segmentation needs to be performed across very large electron microscopy image volumes. We show the advantages of a hierarchical block- wise scheme over local stitching strategies and evaluate the performance of different Multicut solvers for the segmenta- tion of the blocks in the hierarchy. We validate the accuracy of our algorithm on a small fully annotated dataset (5x5x5 mm) and demonstrate no significant loss in segmentation quality compared to solving the Multicut problem globally. We evaluate the scalability of the algorithm on a 95x60x60 mm image volume and show that solving the Multicut prob- lem is no longer the bottleneck of the segmentation pipeline.

count=1
* Towards a Spatio-Temporal Atlas of 3D Cellular Parameters During Leaf Morphogenesis
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Selka_Towards_a_Spatio-Temporal_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w1/Selka_Towards_a_Spatio-Temporal_ICCV_2017_paper.pdf)]
    * Title: Towards a Spatio-Temporal Atlas of 3D Cellular Parameters During Leaf Morphogenesis
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Faical Selka, Thomas Blein, Jasmine Burguet, Eric Biot, Patrick Laufs, Philippe Andrey
    * Abstract: Morphogenesis is a complex process that integrates several mechanisms from the molecular to the organ scales. In plants, division and growth are the two fundamental cellular mechanisms that drive morphogenesis. However, little is known about how these mechanisms are coordinated to establish functional tissue structure. A fundamental bottleneck is the current lack of techniques to systematically quantify the spatio-temporal evolution of 3D cell morphology during organ growth. Using leaf development as a relevant and challenging model to study morphogenesis, we developed a computational framework for cell analysis and quantification from 3D images and for the generation of 3D cell shape atlas. A remarkable feature of leaf morphogenesis being the formation of a laminar-like structure, we propose to automatically separate the cells corresponding to the leaf sides in the segmented leaves, by applying a clustering algorithm. The performance of the proposed pipeline was experimentally assessed on a dataset of 46 leaves in an early developmental state.

count=1
* Deep Convolutional Neural Networks for Detecting Cellular Changes Due to Malignancy
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Wieslander_Deep_Convolutional_Neural_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w1/Wieslander_Deep_Convolutional_Neural_ICCV_2017_paper.pdf)]
    * Title: Deep Convolutional Neural Networks for Detecting Cellular Changes Due to Malignancy
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Hakan Wieslander, Gustav Forslid, Ewert Bengtsson, Carolina Wahlby, Jan-Michael Hirsch, Christina Runow Stark, Sajith Kecheril Sadanandan
    * Abstract: Discovering cancer at an early stage is an effective way to increase the chance of survival. However, since most screening processes are done manually it is time inefficient and thus a costly process. One way of automizing the screening process could be to classify cells using Convolutional Neural Networks. Convolutional Neural Networks have been proven to be accurate for image classification tasks. Two datasets containing oral cells and two datasets containing cervical cells were used. For the cervical cancer dataset the cells were classified by medical experts as normal or abnormal. For the oral cell dataset we only used the diagnosis of the patient. All cells obtained from a patient with malignancy were thus considered malignant even though most of them looked normal. The performance was evaluated for two different network architectures, ResNet and VGG. For the oral datasets the accuracy varied between 78-82% correctly classified cells depending on the dataset and network. For the cervical datasets the accuracy varied between 84-86% correctly classified cells depending on the dataset and network. The results indicate a high potential for detecting abnormalities in oral cavity and in uterine cervix. ResNet was shown to be the preferable network, with a higher accuracy and a smaller standard deviation.

count=1
* In Defense of Shallow Learned Spectral Reconstruction From RGB Images
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w9/html/Aeschbacher_In_Defense_of_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w9/Aeschbacher_In_Defense_of_ICCV_2017_paper.pdf)]
    * Title: In Defense of Shallow Learned Spectral Reconstruction From RGB Images
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Jonas Aeschbacher, Jiqing Wu, Radu Timofte
    * Abstract: Very recent Galliani et al. proposed a method using a very deep CNN architecture or learned spectral reconstruction and showed large improvements over the recent sparse coding method of Arad et al. In this paper we defend the shallow learned spectral reconstruction methods by: (i) first, reimplementing Arad and showing that it can achieve significantly better results than those originally reported; (ii) second, introducing a novel shallow method based on A+ of Timofte et al. from super-resolution that substantially improves over Arad and, moreover, provides comparable performance to Galliani's very deep CNN method on three standard benchmarks (ICVL, CAVE, and NUS); and (iii) finally, arguing that the train and runtime efficiency as well as the clear relation between its parameters and the achieved performance makes from our shallow A+ a strong baseline for further research in learned spectral reconstruction from RGB images. Moreover, our shallow A+ (as well as Arad) requires and uses significantly smaller train data than Galliani (and generally the CNN approaches), is robust to overfitting and is easily deployable by fast training to newer cameras.

count=1
* Bottleneck Potentials in Markov Random Fields
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Abbas_Bottleneck_Potentials_in_Markov_Random_Fields_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Abbas_Bottleneck_Potentials_in_Markov_Random_Fields_ICCV_2019_paper.pdf)]
    * Title: Bottleneck Potentials in Markov Random Fields
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Ahmed Abbas,  Paul Swoboda
    * Abstract: We consider general discrete Markov Random Fields(MRFs) with additional bottleneck potentials which penalize the maximum (instead of the sum) over local potential value taken by the MRF-assignment. Bottleneck potentials or analogous constructions have been considered in (i) combinatorial optimization (e.g. bottleneck shortest path problem, the minimum bottleneck spanning tree problem, bottleneck function minimization in greedoids), (ii) inverse problems with L_ infinity -norm regularization and (iii) valued constraint satisfaction on the (min,max)-pre-semirings. Bottleneck potentials for general discrete MRFs are a natural generalization of the above direction of modeling work to Maximum-A-Posteriori (MAP) inference in MRFs. To this end we propose MRFs whose objective consists of two parts: terms that factorize according to (i) (min,+), i.e. potentials as in plain MRFs, and (ii) (min,max), i.e. bottleneck potentials. To solve the ensuing inference problem, we propose high-quality relaxations and efficient algorithms for solving them. We empirically show efficacy of our approach on large scale seismic horizon tracking problems.

count=1
* YOLACT: Real-Time Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.pdf)]
    * Title: YOLACT: Real-Time Instance Segmentation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Daniel Bolya,  Chong Zhou,  Fanyi Xiao,  Yong Jae Lee
    * Abstract: We present a simple, fully-convolutional model for real-time instance segmentation that achieves 29.8 mAP on MS COCO at 33.5 fps evaluated on a single Titan Xp, which is significantly faster than any previous competitive approach. Moreover, we obtain this result after training on only one GPU. We accomplish this by breaking instance segmentation into two parallel subtasks: (1) generating a set of prototype masks and (2) predicting per-instance mask coefficients. Then we produce instance masks by linearly combining the prototypes with the mask coefficients. We find that because this process doesn't depend on repooling, this approach produces very high-quality masks and exhibits temporal stability for free. Furthermore, we analyze the emergent behavior of our prototypes and show they learn to localize instances on their own in a translation variant manner, despite being fully-convolutional. Finally, we also propose Fast NMS, a drop-in 12 ms faster replacement for standard NMS that only has a marginal performance penalty.

count=1
* TensorMask: A Foundation for Dense Object Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_TensorMask_A_Foundation_for_Dense_Object_Segmentation_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_TensorMask_A_Foundation_for_Dense_Object_Segmentation_ICCV_2019_paper.pdf)]
    * Title: TensorMask: A Foundation for Dense Object Segmentation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Xinlei Chen,  Ross Girshick,  Kaiming He,  Piotr Dollar
    * Abstract: Sliding-window object detectors that generate bounding-box object predictions over a dense, regular grid have advanced rapidly and proven popular. In contrast, modern instance segmentation approaches are dominated by methods that first detect object bounding boxes, and then crop and segment these regions, as popularized by Mask R-CNN. In this work, we investigate the paradigm of dense sliding-window instance segmentation, which is surprisingly under-explored. Our core observation is that this task is fundamentally different than other dense prediction tasks such as semantic segmentation or bounding-box object detection, as the output at every spatial location is itself a geometric structure with its own spatial dimensions. To formalize this, we treat dense instance segmentation as a prediction task over 4D tensors and present a general framework called TensorMask that explicitly captures this geometry and enables novel operators on 4D tensors. We demonstrate that the tensor view leads to large gains over baselines that ignore this structure, and leads to results comparable to Mask R-CNN. These promising results suggest that TensorMask can serve as a foundation for novel advances in dense mask prediction and a more complete understanding of the task. Code will be made available.

count=1
* Learning Temporal Action Proposals With Fewer Labels
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Ji_Learning_Temporal_Action_Proposals_With_Fewer_Labels_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ji_Learning_Temporal_Action_Proposals_With_Fewer_Labels_ICCV_2019_paper.pdf)]
    * Title: Learning Temporal Action Proposals With Fewer Labels
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Jingwei Ji,  Kaidi Cao,  Juan Carlos Niebles
    * Abstract: Temporal action proposals are a common module in action detection pipelines today. Most current methods for training action proposal modules rely on fully supervised approaches that require large amounts of annotated temporal action intervals in long video sequences. The large cost and effort in annotation that this entails motivate us to study the problem of training proposal modules with less supervision. In this work, we propose a semi-supervised learning algorithm specifically designed for training temporal action proposal networks. When only a small number of labels are available, our semi-supervised method generates significantly better proposals than the fully-supervised counterpart and other strong semi-supervised baselines. We validate our method on two challenging action detection video datasets, ActivityNet v1.3 and THUMOS14. We show that our semi-supervised approach consistently matches or outperforms the fully supervised state-of-the-art approaches.

count=1
* 3D Instance Segmentation via Multi-Task Metric Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Lahoud_3D_Instance_Segmentation_via_Multi-Task_Metric_Learning_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lahoud_3D_Instance_Segmentation_via_Multi-Task_Metric_Learning_ICCV_2019_paper.pdf)]
    * Title: 3D Instance Segmentation via Multi-Task Metric Learning
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Jean Lahoud,  Bernard Ghanem,  Marc Pollefeys,  Martin R. Oswald
    * Abstract: We propose a novel method for instance label segmentation of dense 3D voxel grids. We target volumetric scene representations, which have been acquired with depth sensors or multi-view stereo methods and which have been processed with semantic 3D reconstruction or scene completion methods. The main task is to learn shape information about individual object instances in order to accurately separate them, including connected and incompletely scanned objects. We solve the 3D instance-labeling problem with a multi-task learning strategy. The first goal is to learn an abstract feature embedding, which groups voxels with the same instance label close to each other while separating clusters with different instance labels from each other. The second goal is to learn instance information by densely estimating directional information of the instance's center of mass for each voxel. This is particularly useful to find instance boundaries in the clustering post-processing step, as well as, for scoring the segmentation quality for the first goal. Both synthetic and real-world experiments demonstrate the viability and merits of our approach. In fact, it achieves state-of-the-art performance on the ScanNet 3D instance segmentation benchmark.

count=1
* BMN: Boundary-Matching Network for Temporal Action Proposal Generation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.pdf)]
    * Title: BMN: Boundary-Matching Network for Temporal Action Proposal Generation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Tianwei Lin,  Xiao Liu,  Xin Li,  Errui Ding,  Shilei Wen
    * Abstract: Temporal action proposal generation is an challenging and promising task which aims to locate temporal regions in real-world videos where action or event may occur. Current bottom-up proposal generation methods can generate proposals with precise boundary, but cannot efficiently generate adequately reliable confidence scores for retrieving proposals. To address these difficulties, we introduce the Boundary-Matching (BM) mechanism to evaluate confidence scores of densely distributed proposals, which denote a proposal as a matching pair of starting and ending boundaries and combine all densely distributed BM pairs into the BM confidence map. Based on BM mechanism, we propose an effective, efficient and end-to-end proposal generation method, named Boundary-Matching Network (BMN), which generates proposals with precise temporal boundaries as well as reliable confidence scores simultaneously. The two-branches of BMN are jointly trained in an unified framework. We conduct experiments on two challenging datasets: THUMOS-14 and ActivityNet-1.3, where BMN shows significant performance improvement with remarkable efficiency and generalizability. Further, combining with existing action classifier, BMN can achieve state-of-the-art temporal action detection performance.

count=1
* Weakly Supervised Temporal Action Localization Through Contrast Based Evaluation Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Weakly_Supervised_Temporal_Action_Localization_Through_Contrast_Based_Evaluation_Networks_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Weakly_Supervised_Temporal_Action_Localization_Through_Contrast_Based_Evaluation_Networks_ICCV_2019_paper.pdf)]
    * Title: Weakly Supervised Temporal Action Localization Through Contrast Based Evaluation Networks
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Ziyi Liu,  Le Wang,  Qilin Zhang,  Zhanning Gao,  Zhenxing Niu,  Nanning Zheng,  Gang Hua
    * Abstract: Weakly-supervised temporal action localization (WS-TAL) is a promising but challenging task with only video-level action categorical labels available during training. Without requiring temporal action boundary annotations in training data, WS-TAL could possibly exploit automatically retrieved video tags as video-level labels. However, such coarse video-level supervision inevitably incurs confusions, especially in untrimmed videos containing multiple action instances. To address this challenge, we propose the Contrast-based Localization EvaluAtioN Network (CleanNet) with our new action proposal evaluator, which provides pseudo-supervision by leveraging the temporal contrast in snippet-level action classification predictions. Essentially, the new action proposal evaluator enforces an additional temporal contrast constraint so that high-evaluation-score action proposals are more likely to coincide with true action instances. Moreover, the new action localization module is an integral part of CleanNet which enables end-to-end training. This is in contrast to many existing WS-TAL methods where action localization is merely a post-processing step. Experiments on THUMOS14 and ActivityNet datasets validate the efficacy of CleanNet against existing state-ofthe- art WS-TAL algorithms.

count=1
* Incremental Class Discovery for Semantic Segmentation With RGBD Sensing
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Nakajima_Incremental_Class_Discovery_for_Semantic_Segmentation_With_RGBD_Sensing_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nakajima_Incremental_Class_Discovery_for_Semantic_Segmentation_With_RGBD_Sensing_ICCV_2019_paper.pdf)]
    * Title: Incremental Class Discovery for Semantic Segmentation With RGBD Sensing
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Yoshikatsu Nakajima,  Byeongkeun Kang,  Hideo Saito,  Kris Kitani
    * Abstract: This work addresses the task of open world semantic segmentation using RGBD sensing to discover new semantic classes over time. Although there are many types of objects in the real-word, current semantic segmentation methods make a closed world assumption and are trained only to segment a limited number of object classes. Towards a more open world approach, we propose a novel method that incrementally learns new classes for image segmentation. The proposed system first segments each RGBD frame using both color and geometric information, and then aggregates that information to build a single segmented dense 3D map of the environment. The segmented 3D map representation is a key component of our approach as it is used to discover new object classes by identifying coherent regions in the 3D map that have no semantic label. The use of coherent region in the 3D map as a primitive element, rather than traditional elements such as surfels or voxels, also significantly reduces the computational complexity and memory use of our method. It thus leads to semi-real-time performance at 10.7 Hz when incrementally updating the dense 3D map at every frame. Through experiments on the NYUDv2 dataset, we demonstrate that the proposed method is able to correctly cluster objects of both known and unseen classes. We also show the quantitative comparison with the state-of-the-art supervised methods, the processing time of each step, and the influences of each component.

count=1
* Depth Completion From Sparse LiDAR Data With Depth-Normal Constraints
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Xu_Depth_Completion_From_Sparse_LiDAR_Data_With_Depth-Normal_Constraints_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Depth_Completion_From_Sparse_LiDAR_Data_With_Depth-Normal_Constraints_ICCV_2019_paper.pdf)]
    * Title: Depth Completion From Sparse LiDAR Data With Depth-Normal Constraints
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Yan Xu,  Xinge Zhu,  Jianping Shi,  Guofeng Zhang,  Hujun Bao,  Hongsheng Li
    * Abstract: Depth completion aims to recover dense depth maps from sparse depth measurements. It is of increasing importance for autonomous driving and draws increasing attention from the vision community. Most of the current competitive methods directly train a network to learn a mapping from sparse depth inputs to dense depth maps, which has difficulties in utilizing the 3D geometric constraints and handling the practical sensor noises. In this paper, to regularize the depth completion and improve the robustness against noise, we propose a unified CNN framework that 1) models the geometric constraints between depth and surface normal in a diffusion module and 2) predicts the confidence of sparse LiDAR measurements to mitigate the impact of noise. Specifically, our encoder-decoder backbone predicts the surface normal, coarse depth and confidence of LiDAR inputs simultaneously, which are subsequently inputted into our diffusion refinement module to obtain the final completion results. Extensive experiments on KITTI depth completion dataset and NYU-Depth-V2 dataset demonstrate that our method achieves state-of-the-art performance. Further ablation study and analysis give more insights into the proposed components and demonstrate the generalization capability and stability of our model.

count=1
* An Elastica Geodesic Approach With Convexity Shape Prior
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_An_Elastica_Geodesic_Approach_With_Convexity_Shape_Prior_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_An_Elastica_Geodesic_Approach_With_Convexity_Shape_Prior_ICCV_2021_paper.pdf)]
    * Title: An Elastica Geodesic Approach With Convexity Shape Prior
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Da Chen, Laurent D. Cohen, Jean-Marie Mirebeau, Xue-Cheng Tai
    * Abstract: The minimal geodesic models based on the Eikonal equations are capable of finding suitable solutions in various image segmentation scenarios. Existing geodesic-based segmentation approaches usually exploit the image features in conjunction with geometric regularization terms (such as curve length or elastica length) for computing geodesic paths. In this paper, we consider a more complicated problem: finding simple and closed geodesic curves which are imposed a convexity shape prior. The proposed approach relies on an orientation-lifting strategy, by which a planar curve can be mapped to an high-dimensional orientation space. The convexity shape prior serves as a constraint for the construction of local metrics. The geodesic curves in the lifted space then can be efficiently computed through the fast marching method. In addition, we introduce a way to incorporate region-based homogeneity features into the proposed geodesic model so as to solve the region-based segmentation issues with shape prior constraints.

count=1
* Hierarchical Aggregation for 3D Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Hierarchical_Aggregation_for_3D_Instance_Segmentation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Hierarchical_Aggregation_for_3D_Instance_Segmentation_ICCV_2021_paper.pdf)]
    * Title: Hierarchical Aggregation for 3D Instance Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Shaoyu Chen, Jiemin Fang, Qian Zhang, Wenyu Liu, Xinggang Wang
    * Abstract: Instance segmentation on point clouds is a fundamental task in 3D scene perception. In this work, we propose a concise clustering-based framework named HAIS, which makes full use of spatial relation of points and point sets. Considering clustering-based methods may result in over-segmentation or under-segmentation, we introduce the hierarchical aggregation to progressively generate instance proposals, i.e., point aggregation for preliminarily clustering points to sets and set aggregation for generating complete instances from sets. Once the complete 3D instances are obtained, a sub-network of intra-instance prediction is adopted for noisy points filtering and mask quality scoring. HAIS is fast (only 410ms per frame on Titan X) and does not require non-maximum suppression. It ranks 1st on the ScanNet v2 benchmark, achieving the highest 69.9% AP50 and surpassing previous state-of-the-art (SOTA) methods by a large margin. Besides, the SOTA results on the S3DIS dataset validate the good generalization ability. Code is available at https://github.com/hustvl/HAIS.

count=1
* Mutual-Complementing Framework for Nuclei Detection and Segmentation in Pathology Image
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_Mutual-Complementing_Framework_for_Nuclei_Detection_and_Segmentation_in_Pathology_Image_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_Mutual-Complementing_Framework_for_Nuclei_Detection_and_Segmentation_in_Pathology_Image_ICCV_2021_paper.pdf)]
    * Title: Mutual-Complementing Framework for Nuclei Detection and Segmentation in Pathology Image
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zunlei Feng, Zhonghua Wang, Xinchao Wang, Yining Mao, Thomas Li, Jie Lei, Yuexuan Wang, Mingli Song
    * Abstract: Detection and segmentation of nuclei are fundamental analysis operations in pathology images, the assessments derived from which serve as the gold standard for cancer diagnosis. Manual segmenting nuclei is expensive and time-consuming. What's more, accurate segmentation detection of nuclei can be challenging due to the large appearance variation, conjoined and overlapping nuclei, and serious degeneration of histological structures. Supervised methods highly rely on massive annotated samples. The existing two unsupervised methods are prone to failure on degenerated samples. This paper proposes a Mutual-Complementing Framework (MCF) for nuclei detection and segmentation in pathology images. Two branches of MCF are trained in the mutual-complementing manner, where the detection branch complements the pseudo mask of the segmentation branch, while the progressive trained segmentation branch complements the missing nucleus templates through calculating the mask residual between the predicted mask and detected result. In the detection branch, two response map fusion strategies and gradient direction based postprocessing are devised to obtain the optimal detection response. Furthermore, the confidence loss combined with the synthetic samples and self-finetuning is adopted to train the segmentation network with only high confidence areas. Extensive experiments demonstrate that MCF achieves comparable performance with only a few nucleus patches as supervision. Especially, MCF possesses good robustness (only dropping by about 6%) on degenerated samples, which are critical and common cases in clinical diagnosis.

count=1
* CDNet: Centripetal Direction Network for Nuclear Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/He_CDNet_Centripetal_Direction_Network_for_Nuclear_Instance_Segmentation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/He_CDNet_Centripetal_Direction_Network_for_Nuclear_Instance_Segmentation_ICCV_2021_paper.pdf)]
    * Title: CDNet: Centripetal Direction Network for Nuclear Instance Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Hongliang He, Zhongyi Huang, Yao Ding, Guoli Song, Lin Wang, Qian Ren, Pengxu Wei, Zhiqiang Gao, Jie Chen
    * Abstract: Nuclear instance segmentation is a challenging task due to a large number of touching and overlapping nuclei in pathological images. Existing methods cannot effectively recognize the accurate boundary owing to neglecting the relationship between pixels (e.g., direction information). In this paper, we propose a novel Centripetal Direction Network (CDNet) for nuclear instance segmentation. Specifically, we define the centripetal direction feature as a class of adjacent directions pointing to the nuclear center to represent the spatial relationship between pixels within the nucleus. These direction features are then used to construct a direction difference map to represent the similarity within instances and the differences between instances. Finally, we propose a direction-guided refinement module, which acts as a plug-and-play module to effectively integrate auxiliary tasks and aggregate the features of different branches. Experiments on MoNuSeg and CPM17 datasets show that CDNet is significantly better than the other methods and achieves the state-of-the-art performance. The code is available at https://github.com/honglianghe/CDNet.

count=1
* DiscoBox: Weakly Supervised Instance Segmentation and Semantic Correspondence From Box Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Lan_DiscoBox_Weakly_Supervised_Instance_Segmentation_and_Semantic_Correspondence_From_Box_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Lan_DiscoBox_Weakly_Supervised_Instance_Segmentation_and_Semantic_Correspondence_From_Box_ICCV_2021_paper.pdf)]
    * Title: DiscoBox: Weakly Supervised Instance Segmentation and Semantic Correspondence From Box Supervision
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Shiyi Lan, Zhiding Yu, Christopher Choy, Subhashree Radhakrishnan, Guilin Liu, Yuke Zhu, Larry S. Davis, Anima Anandkumar
    * Abstract: We introduce DiscoBox, a novel framework that jointly learns instance segmentation and semantic correspondence using bounding box supervision. Specifically, we propose a self-ensembling framework where instance segmentation and semantic correspondence are jointly guided by a structured teacher in addition to the bounding box supervision. The teacher is a structured energy model incorporating a pairwise potential and a cross-image potential to model the pairwise pixel relationships both within and across the boxes. Minimizing the teacher energy simultaneously yields refined object masks and dense correspondences between intra-class objects, which are taken as pseudo-labels to supervise the task network and provide positive/negative correspondence pairs for dense contrastive learning. We show a symbiotic relationship where the two tasks mutually benefit from each other. Our best model achieves 37.9% AP on COCO instance segmentation, surpassing prior weakly supervised methods and is competitive to supervised methods. We also obtain state of the art weakly supervised results on PASCAL VOC12 and PF-PASCAL with real-time inference.

count=1
* Field of Junctions: Extracting Boundary Structure at Low SNR
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Verbin_Field_of_Junctions_Extracting_Boundary_Structure_at_Low_SNR_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Verbin_Field_of_Junctions_Extracting_Boundary_Structure_at_Low_SNR_ICCV_2021_paper.pdf)]
    * Title: Field of Junctions: Extracting Boundary Structure at Low SNR
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Dor Verbin, Todd Zickler
    * Abstract: We introduce a bottom-up model for simultaneously finding many boundary elements in an image, including contours, corners and junctions. The model explains boundary shape in each small patch using a 'generalized M-junction' comprising M angles and a freely-moving vertex. Images are analyzed using non-convex optimization to cooperatively find M+2 junction values at every location, with spatial consistency being enforced by a novel regularizer that reduces curvature while preserving corners and junctions. The resulting 'field of junctions' is simultaneously a contour detector, corner/junction detector, and boundary-aware smoothing of regional appearance. Notably, its unified analysis of contours, corners, junctions and uniform regions allows it to succeed at high noise levels, where other methods for segmentation and boundary detection fail.

count=1
* Exploring Cross-Image Pixel Contrast for Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.pdf)]
    * Title: Exploring Cross-Image Pixel Contrast for Semantic Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Wenguan Wang, Tianfei Zhou, Fisher Yu, Jifeng Dai, Ender Konukoglu, Luc Van Gool
    * Abstract: Current semantic segmentation methods focus only on mining "local" context, i.e., dependencies between pixels within individual images, by context-aggregation modules (e.g., dilated convolution, neural attention) or structure-aware optimization criteria (e.g., IoU-like loss). However, they ignore "global" context of the training data, i.e., rich semantic relations between pixels across different images. Inspired by recent advance in unsupervised contrastive representation learning, we propose a pixel-wise contrastive algorithm for semantic segmentation in the fully supervised setting. The core idea is to enforce pixel embeddings belonging to a same semantic class to be more similar than embeddings from different classes. It raises a pixel-wise metric learning paradigm for semantic segmentation, by explicitly exploring the structures of labeled pixels, which were rarely explored before. Our method can be effortlessly incorporated into existing segmentation frameworks without extra overhead during testing. We experimentally show that, with famous segmentation models (i.e., DeepLabV3, HRNet, OCR) and backbones (i.e., ResNet, HRNet), our method brings performance improvements across diverse datasets (i.e., Cityscapes, PASCAL-Context, COCO-Stuff, CamVid). We expect this work will encourage our community to rethink the current de facto training paradigm in semantic segmentation.

count=1
* LeafMask: Towards Greater Accuracy on Leaf Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/html/Guo_LeafMask_Towards_Greater_Accuracy_on_Leaf_Segmentation_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/papers/Guo_LeafMask_Towards_Greater_Accuracy_on_Leaf_Segmentation_ICCVW_2021_paper.pdf)]
    * Title: LeafMask: Towards Greater Accuracy on Leaf Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Ruohao Guo, Liao Qu, Dantong Niu, Zhenbo Li, Jun Yue
    * Abstract: Leaf segmentation is the most direct and effective way for high-throughput plant phenotype data analysis and quantitative researches of complex traits. Currently, the primary goal of plant phenotyping is to raise the accuracy of the autonomous phenotypic measurement. In this work, we present the LeafMask neural network, a new end-to-end model to delineate each leaf region and count the number of leaves, with two main components: 1) the mask assembly module merging position-sensitive bases of each predicted box after non-maximum suppression (NMS) and corresponding coefficients to generate original masks; 2) the mask refining module elaborating leaf boundaries from the mask assembly module by the point selection strategy and predictor. In addition, we also design a novel and flexible multi-scale attention module for the dual attention-guided mask (DAG-Mask) branch to effectively enhance information expression and produce more accurate bases. Our main contribution is to generate the final improved masks by combining the mask assembly module with the mask refining module under the anchor-free instance segmentation paradigm. We validate our LeafMask through extensive experiments on Leaf Segmentation Challenge (LSC) dataset. Our proposed model achieves the 90.09% BestDice score outperforming other state-of-the-art approaches.

count=1
* Affine-Consistent Transformer for Multi-Class Cell Nuclei Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Affine-Consistent_Transformer_for_Multi-Class_Cell_Nuclei_Detection_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_Affine-Consistent_Transformer_for_Multi-Class_Cell_Nuclei_Detection_ICCV_2023_paper.pdf)]
    * Title: Affine-Consistent Transformer for Multi-Class Cell Nuclei Detection
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Junjia Huang, Haofeng Li, Xiang Wan, Guanbin Li
    * Abstract: Multi-class cell nuclei detection is a fundamental prerequisite in the diagnosis of histopathology. It is critical to efficiently locate and identify cells with diverse morphology and distributions in digital pathological images. Most existing methods take complex intermediate representations as learning targets and rely on inflexible post-refinements while paying less attention to various cell density and fields of view. In this paper, we propose a novel Affine-Consistent Transformer (AC-Former), which directly yields a sequence of nucleus positions and is trained collaboratively through two sub-networks, a global and a local network. The local branch learns to infer distorted input images of smaller scales while the global network outputs the large-scale predictions as extra supervision signals. We further introduce an Adaptive Affine Transformer (AAT) module, which can automatically learn the key spatial transformations to warp original images for local network training. The AAT module works by learning to capture the transformed image regions that are more valuable for training the model. Experimental results demonstrate that the proposed method significantly outperforms existing state-of-the-art algorithms on various benchmarks.

count=1
* Adaptive Superpixel for Active Learning in Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Kim_Adaptive_Superpixel_for_Active_Learning_in_Semantic_Segmentation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Adaptive_Superpixel_for_Active_Learning_in_Semantic_Segmentation_ICCV_2023_paper.pdf)]
    * Title: Adaptive Superpixel for Active Learning in Semantic Segmentation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Hoyoung Kim, Minhyeon Oh, Sehyun Hwang, Suha Kwak, Jungseul Ok
    * Abstract: Learning semantic segmentation requires pixel-wise annotations, which can be time-consuming and expensive. To reduce the annotation cost, we propose a superpixel-based active learning (AL) framework, which collects a dominant label per superpixel instead. To be specific, it consists of adaptive superpixel and sieving mechanisms, fully dedicated to AL. At each round of AL, we adaptively merge neighboring pixels of similar learned features into superpixels. We then query a selected subset of these superpixels using an acquisition function assuming no uniform superpixel size. This approach is more efficient than existing methods, which rely only on innate features such as RGB color and assume uniform superpixel sizes. Obtaining a dominant label per superpixel drastically reduces annotators' burden as it requires fewer clicks. However, it inevitably introduces noisy annotations due to mismatches between superpixel and ground truth segmentation. To address this issue, we further devise a sieving mechanism that identifies and excludes potentially noisy annotations from learning. Our experiments on both Cityscapes and PASCAL VOC datasets demonstrate the efficacy of adaptive superpixel and sieving mechanisms.

count=1
* Learning Cross-Representation Affinity Consistency for Sparsely Supervised Biomedical Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_Cross-Representation_Affinity_Consistency_for_Sparsely_Supervised_Biomedical_Instance_Segmentation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Learning_Cross-Representation_Affinity_Consistency_for_Sparsely_Supervised_Biomedical_Instance_Segmentation_ICCV_2023_paper.pdf)]
    * Title: Learning Cross-Representation Affinity Consistency for Sparsely Supervised Biomedical Instance Segmentation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Xiaoyu Liu, Wei Huang, Zhiwei Xiong, Shenglong Zhou, Yueyi Zhang, Xuejin Chen, Zheng-Jun Zha, Feng Wu
    * Abstract: Sparse instance-level supervision has recently been explored to address insufficient annotation in biomedical instance segmentation, which is easier to annotate crowded instances and better preserves instance completeness for 3D volumetric datasets compared to common semi-supervision.In this paper, we propose a sparsely supervised biomedical instance segmentation framework via cross-representation affinity consistency regularization. Specifically, we adopt two individual networks to enforce the perturbation consistency between an explicit affinity map and an implicit affinity map to capture both feature-level instance discrimination and pixel-level instance boundary structure. We then select the highly confident region of each affinity map as the pseudo label to supervise the other one for affinity consistency learning. To obtain the highly confident region, we propose a pseudo-label noise filtering scheme by integrating two entropy-based decision strategies. Extensive experiments on four biomedical datasets with sparse instance annotations show the state-of-the-art performance of our proposed framework. For the first time, we demonstrate the superiority of sparse instance-level supervision on 3D volumetric datasets, compared to common semi-supervision under the same annotation cost.

count=1
* Fast Full-frame Video Stabilization with Iterative Optimization
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Fast_Full-frame_Video_Stabilization_with_Iterative_Optimization_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Fast_Full-frame_Video_Stabilization_with_Iterative_Optimization_ICCV_2023_paper.pdf)]
    * Title: Fast Full-frame Video Stabilization with Iterative Optimization
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Weiyue Zhao, Xin Li, Zhan Peng, Xianrui Luo, Xinyi Ye, Hao Lu, Zhiguo Cao
    * Abstract: Video stabilization refers to the problem of transforming a shaky video into a visually pleasing one. The question of how to strike a good trade-off between visual quality and computational speed has remained one of the open challenges in video stabilization. Inspired by the analogy between wobbly frames and jigsaw puzzles, we propose an iterative optimization-based learning approach using synthetic datasets for video stabilization, which consists of two interacting submodules: motion trajectory smoothing and full-frame outpainting. First, we develop a two-level (coarse-to-fine) stabilizing algorithm based on the probabilistic flow field. The confidence map associated with the estimated optical flow is exploited to guide the search for shared regions through backpropagation. Second, we take a divide-and-conquer approach and propose a novel multiframe fusion strategy to render full-frame stabilized views. An important new insight brought about by our iterative optimization approach is that the target video can be interpreted as the fixed point of nonlinear mapping for video stabilization. We formulate video stabilization as a problem of minimizing the amount of jerkiness in motion trajectories, which guarantees convergence with the help of fixed-point theory. Extensive experimental results are reported to demonstrate the superiority of the proposed approach in terms of computational speed and visual quality. The code will be available on GitHub.

count=1
* PCTrans: Position-Guided Transformer with Query Contrast for Biological Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/BIC/html/Chen_PCTrans_Position-Guided_Transformer_with_Query_Contrast_for_Biological_Instance_Segmentation_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/BIC/papers/Chen_PCTrans_Position-Guided_Transformer_with_Query_Contrast_for_Biological_Instance_Segmentation_ICCVW_2023_paper.pdf)]
    * Title: PCTrans: Position-Guided Transformer with Query Contrast for Biological Instance Segmentation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Qi Chen, Wei Huang, Xiaoyu Liu, Jiacheng Li, Zhiwei Xiong
    * Abstract: Recently, query-based transformer gradually draws attention in segmentation tasks due to its powerful ability. Compared to instance segmentation in natural images, biological instance segmentation is more challenging due to high texture similarity, crowded objects and limited annotations. Therefore, it remains a pending issue to extract meaningful queries to model biological instances. In this paper, we analyze the problem when queries meet biological images and propose a novel Position-guided Transformer with query Contrast (PCTrans) for biological instance segmentation. PCTrans tackles the mentioned issue in two ways. First, for high texture similarity and crowded objects, we incorporate position information to guide query learning and mask prediction. This involves considering position similarity when learning queries and designing a dynamic mask head that takes instance position into account. Second, to learn more discriminative representation of the queries under limited annotated data, we further design two contrastive losses, namely Query Embedding Contrastive (QEC) loss and Mask Candidate Contrastive (MCC) loss. Experiments on two representative biological instance segmentation datasets demonstrate the superiority of PCTrans over existing methods. Code is available at https://github.com/qic999/PCTrans.

count=1
* 4-Connected Shift Residual Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Brown_4-Connected_Shift_Residual_Networks_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Brown_4-Connected_Shift_Residual_Networks_ICCVW_2019_paper.pdf)]
    * Title: 4-Connected Shift Residual Networks
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Andrew Brown, Pascal Mettes, Marcel Worring
    * Abstract: The shift operation was recently introduced as an alternative to spatial convolutions. The operation moves subsets of activations horizontally and/or vertically. Spatial convolutions are then replaced with shift operations followed by point-wise convolutions, significantly reducing computational costs. In this work, we investigate how shifts should best be applied to high accuracy CNNs. We apply shifts of two different neighbourhood groups to ResNet on ImageNet: the originally introduced 8-connected (8C) neighbourhood shift and the less well studied 4-connected (4C) neighbourhood shift. We find that when replacing ResNet's spatial convolutions with shifts, both shift neighbourhoods give equal ImageNet accuracy, showing the sufficiency of small neighbourhoods for large images. Interestingly, when incorporating shifts to all point-wise convolutions in residual networks, 4-connected shifts outperform 8-connected shifts. Such a 4-connected shift setup gives the same accuracy as full residual networks while reducing the number of parameters and FLOPs by over 40%. We then highlight that without spatial convolutions, ResNet's downsampling/upsampling bottleneck channel structure is no longer needed. We show a new, 4C shift-based residual network, much shorter than the original ResNet yet with a higher accuracy for the same computational cost. This network is the highest accuracy shift-based network yet shown, demonstrating the potential of shifting in deep neural networks.

count=1
* High Precision Localization of Bacterium and Scientific Visualization
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W06/html/Hosseini_High_Precision_Localization_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W06/papers/Hosseini_High_Precision_Localization_2013_ICCV_paper.pdf)]
    * Title: High Precision Localization of Bacterium and Scientific Visualization
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Mohammadreza Hosseini, Arcot Sowmya, Pascal Vallotton, Tomasz Bednarz
    * Abstract: Bacteria reproduce simply and rapidly by doubling their contents and then splitting in two. The majority of bacteria in the human body are countered by the human immune system, however some pathogenic bacteria survive and cause disease. A small variation of bacterium size has great impact on their ability to cope with the new environment. This variation is not traceable in bacterium images as they may be less than a pixel width. In this paper, we present a method for high precision localization and dimension estimation of bacteria in microscopic images. To create a safe environment for scientist to interact with bacteria images in sterile environment a Human Computer Interaction (HCI) system is developed using Creative Interactive Gesture Camera as a touchless input device to track a user's hand gestures and translate them into the natural field of view or point of focus as well. Experiments on simulated data, shows that our method can achieve more accurate estimation of bacterium dimension in comparing with stateof-the-art sub-pixel cell outlining tool. The visualization of augmented biological data speed up the extraction of useful information.

count=1
* DATNet: Dense Auxiliary Tasks for Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Levinshtein_DATNet_Dense_Auxiliary_Tasks_for_Object_Detection_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Levinshtein_DATNet_Dense_Auxiliary_Tasks_for_Object_Detection_WACV_2020_paper.pdf)]
    * Title: DATNet: Dense Auxiliary Tasks for Object Detection
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Alex Levinshtein,  Alborz Rezazadeh Sereshkeh,  Konstantinos Derpanis
    * Abstract: Beginning with R-CNN, there has been a rapid advancement in two-stage object detection approaches. While two-stage approaches remain the state-of-the-art in object detection, anchor-free single-stage methods have been gaining momentum. We believe that the strength of the former is in their region of interest (ROI) pooling stage, while the latter simplifies the learning problem by converting object detection into dense per-pixel prediction tasks. In this paper, we propose to combine the strengths of each approach in a new architecture. In particular, we first define several auxiliary tasks related to object detection and generate dense per-pixel predictions using a shared feature extraction backbone. As a consequence of this architecture, the shared backbone is trained using both the standard object detection losses and these per-pixel ones. Moreover, by combining the features from dense predictions with those from the backbone, we realize a more discriminative representation for subsequent downstream processing. In addition, we feed the fused features into a novel multi-scale ROI pooling layer, followed by per-ROI predictions. We refer to our architecture as the Dense Auxiliary Tasks Network (DATNet). We present an extensive set of evaluations of our method on the Pascal VOC and COCO datasets and show considerable accuracy improvements over comparable baselines.

count=1
* HistoNet: Predicting size histograms of object instances
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Sharma_HistoNet_Predicting_size_histograms_of_object_instances_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Sharma_HistoNet_Predicting_size_histograms_of_object_instances_WACV_2020_paper.pdf)]
    * Title: HistoNet: Predicting size histograms of object instances
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Kishan Sharma,  Moritz Gold,  Christian  Zurbruegg,  Laura Leal-Taixe,  Jan Dirk Wegner
    * Abstract: We propose to predict histograms of object sizes in crowded scenes directly without any explicit object instance segmentation. What makes this task challenging is the high density of objects (of the same category), which makes instance identification hard. Instead of explicitly segmenting object instances, we show that directly learning histograms of object sizes improves accuracy while using drastically less parameters. This is very useful for application scenarios where explicit, pixel-accurate instance segmentation is not needed, but their lies interest in the overall distribution of instance sizes. Our core applications are in biology, where we estimate the size distribution of soldier fly larvae, and medicine, where we estimate the size distribution of cancer cells as an intermediate step to calculate tumor cellularity score. Given an image with hundreds of small object instances, we output the total count and the size histogram. We also provide a new data set for this task, the FlyLarvae data set, which consists of 11,000 larvae instances labeled pixel-wise. Our method results in an overall improvement in the count and size distribution prediction as compared to state-of-the-art instance segmentation method Mask R-CNN.

count=1
* End-to-End Learning Improves Static Object Geo-Localization From Video
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Chaabane_End-to-End_Learning_Improves_Static_Object_Geo-Localization_From_Video_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Chaabane_End-to-End_Learning_Improves_Static_Object_Geo-Localization_From_Video_WACV_2021_paper.pdf)]
    * Title: End-to-End Learning Improves Static Object Geo-Localization From Video
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Mohamed Chaabane, Lionel Gueguen, Ameni Trabelsi, Ross Beveridge, Stephen O'Hara
    * Abstract: Accurately estimating the position of static objects, such as traffic lights, from the moving camera of a self-driving car is a challenging problem. In this work, we present a system that improves the localization of static objects by jointly-optimizing the components of the system via learning. Our system is comprised of networks that perform: 1) 5DoF object pose estimation from a single image, 2) association of objects between pairs of frames, and 3) multi-object tracking to produce the final geo-localization of the static objects within the scene. We evaluate our approach using a publicly-available data set, focusing on traffic lights due to data availability. For each component, we compare against contemporary alternatives and show significantly-improved performance. We also show that the end-to-end system performance is further improved via joint-training of the constituent models.

count=1
* Automatic Quantification of Plant Disease From Field Image Data Using Deep Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Garg_Automatic_Quantification_of_Plant_Disease_From_Field_Image_Data_Using_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Garg_Automatic_Quantification_of_Plant_Disease_From_Field_Image_Data_Using_WACV_2021_paper.pdf)]
    * Title: Automatic Quantification of Plant Disease From Field Image Data Using Deep Learning
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Kanish Garg, Swati Bhugra, Brejesh Lall
    * Abstract: Plant disease is a major factor in yield reduction. Thus, plant breeders currently rely on selecting disease-resistant plant cultivars, which involves disease severity rating of a large variety of cultivars. Traditional visual screening of these cultivars is an error-prone process, which necessitates the development of an automatic framework for disease quantification based on field-acquired images using unmanned aerial vehicles (UAVs) to augment the throughput. Since these images are impaired by complex backgrounds, uneven lighting, and densely overlapping leaves, state-of-the-art frameworks formulate the processing pipeline as a dichotomy problem (i.e. presence/absence of disease). However, additional information regarding accurate disease localization and quantification is crucial for breeders. This paper proposes a deep framework for simultaneous segmentation of individual leaf instances and corresponding diseased region using a unified feature map with a multi-task loss function for an end-to-end training. We test the framework on field maize dataset with Northern Leaf Blight (NLB) disease and the experimental results show a disease severity correlation of 73% with the manual ground truth data and run-time efficiency of 5fps.

count=1
* Active Latent Space Shape Model: A Bayesian Treatment of Shape Model Adaptation With an Application to Psoriatic Arthritis Radiographs
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Rambojun_Active_Latent_Space_Shape_Model_A_Bayesian_Treatment_of_Shape_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Rambojun_Active_Latent_Space_Shape_Model_A_Bayesian_Treatment_of_Shape_WACV_2021_paper.pdf)]
    * Title: Active Latent Space Shape Model: A Bayesian Treatment of Shape Model Adaptation With an Application to Psoriatic Arthritis Radiographs
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Adwaye Rambojun, William Tillett, Tony Shardlow, Neill D. F. Campbell
    * Abstract: Shape models have been used extensively to regularise segmentation of objects of interest in images, e.g. bones in medical x-ray radiographs, given supervised training examples. However, approaches usually adopt simple linear models that do not capture uncertainty and require extensive annotation effort to label a large number of set template landmarks for training. Conversely, supervised deep learning methods have been used on appearance directly (no explicit shape modelling) but these fail to capture detailed features that are clinically important. We present a supervised approach that combines both a non-linear generative shape model and a discriminative appearance-based convolutional neural network whilst quantifying uncertainty and relaxes the need for detailed, template based alignment for the training data. Our Bayesian framework couples the uncertainty from both the generator and the discriminator; our main contribution is the marginalisation of an intractable integral through the use of radial basis function approximations. We illustrate this model on the problem of segmenting bones from Psoriatic Arthritis hand radiographs and demonstrate that we can accurately measure the clinically important joint space gap between neighbouring bones.

count=1
* Consistent Cell Tracking in Multi-Frames With Spatio-Temporal Context by Object-Level Warping Loss
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Hayashida_Consistent_Cell_Tracking_in_Multi-Frames_With_Spatio-Temporal_Context_by_Object-Level_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Hayashida_Consistent_Cell_Tracking_in_Multi-Frames_With_Spatio-Temporal_Context_by_Object-Level_WACV_2022_paper.pdf)]
    * Title: Consistent Cell Tracking in Multi-Frames With Spatio-Temporal Context by Object-Level Warping Loss
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Junya Hayashida, Kazuya Nishimura, Ryoma Bise
    * Abstract: Multi-object tracking is essential in biomedical image analysis. Most multi-object tracking methods follow a tracking-by-detection approach that involves using object detectors and learning the appearance feature models of the detected regions for association. Although these methods can learn the appearance similarity features to identify the same objects among frames, they have difficulties identifying the same cells because cells have a similar appearance and their shapes change as they migrate. In addition, cells often partially overlap for several frames. In this case, even an expert biologist would require knowledge of the spatial-temporal context in order to identify individual cells. To tackle such difficult situations, we propose a cell-tracking method that can effectively use the spatial-temporal context in multiple frames by using long-term motion estimation and an object-level warping loss. We conducted experiments showing that the proposed method outperformed state-of-the-art methods under various conditions on real biological images.

count=1
* Contextual Proposal Network for Action Localization
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Hsieh_Contextual_Proposal_Network_for_Action_Localization_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Hsieh_Contextual_Proposal_Network_for_Action_Localization_WACV_2022_paper.pdf)]
    * Title: Contextual Proposal Network for Action Localization
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: He-Yen Hsieh, Ding-Jie Chen, Tyng-Luh Liu
    * Abstract: This paper investigates the problem of Temporal Action Proposal (TAP) generation, which aims to provide a set of high-quality video segments that potentially contain actions events locating in long untrimmed videos. Based on the goal to distill available contextual information, we introduce a Contextual Proposal Network (CPN) composing of two context-aware mechanisms. The first mechanism, i.e., feature enhancing, integrates the inception-like module with long-range attention to capture the multi-scale temporal contexts for yielding a robust video segment representation. The second mechanism, i.e., boundary scoring, employs the bi-directional recurrent neural networks (RNN) to capture bi-directional temporal contexts that explicitly model actionness, background, and confidence of proposals. While generating and scoring proposals, such bi-directional temporal contexts are helpful to retrieve high-quality proposals of low false positives for covering the video action instances. We conduct experiments on two challenging datasets of ActivityNet-1.3 and THUMOS-14 to demonstrate the effectiveness of the proposed Contextual Proposal Network (CPN). In particular, our method respectively surpasses state-of-the-art TAP methods by 1.54% AUC on ActivityNet-1.3 test split and by 0.61% AR@200 on THUMOS-14 dataset.

count=1
* HERS Superpixels: Deep Affinity Learning for Hierarchical Entropy Rate Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Peng_HERS_Superpixels_Deep_Affinity_Learning_for_Hierarchical_Entropy_Rate_Segmentation_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Peng_HERS_Superpixels_Deep_Affinity_Learning_for_Hierarchical_Entropy_Rate_Segmentation_WACV_2022_paper.pdf)]
    * Title: HERS Superpixels: Deep Affinity Learning for Hierarchical Entropy Rate Segmentation
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Hankui Peng, Angelica I. Aviles-Rivero, Carola-Bibiane Schönlieb
    * Abstract: Superpixels serve as a powerful preprocessing tool in many computer vision tasks. By using superpixel representation, the number of image primitives can be largely reduced by orders of magnitudes. The majority of superpixel methods use handcrafted features, which usually do not translate well into strong adherence to object boundaries. A few recent superpixel methods have introduced deep learning into the superpixel segmentation process. However, none of these methods is able to produce superpixels in near real-time, which is crucial to the applicability of a superpixel method in practice. In this work, we propose a two-stage graph-based framework for superpixel segmentation. In the first stage, we introduce an efficient Deep Affinity Learning (DAL) network that learns pairwise pixel affinities by aggregating multi-scale information. In the second stage, we propose a highly efficient superpixel method called Hierarchical Entropy Rate Segmentation (HERS). Using the learned affinities from the first stage, HERS builds a hierarchical tree structure that can produce any number of highly adaptive superpixels instantaneously. We demonstrate, through visual and numerical experiments, the effectiveness and efficiency of our method compared to various state-of-the-art superpixel methods.

count=1
* CountNet3D: A 3D Computer Vision Approach To Infer Counts of Occluded Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Jenkins_CountNet3D_A_3D_Computer_Vision_Approach_To_Infer_Counts_of_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Jenkins_CountNet3D_A_3D_Computer_Vision_Approach_To_Infer_Counts_of_WACV_2023_paper.pdf)]
    * Title: CountNet3D: A 3D Computer Vision Approach To Infer Counts of Occluded Objects
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Porter Jenkins, Kyle Armstrong, Stephen Nelson, Siddhesh Gotad, J. Stockton Jenkins, Wade Wilkey, Tanner Watts
    * Abstract: 3D scene understanding is an important problem that has experienced great progress in recent years, in large part due to the development of state-of-the-art methods for 3D object detection. However, the performance of 3D object detectors can suffer in scenarios where extreme occlusion of objects is present, or the number of object classes is large. In this paper, we study the problem of inferring 3D counts from densely packed scenes with heterogeneous objects. This problem has applications to important tasks such as inventory management or automatic crop yield estimation. We propose a novel regression-based method, CountNet3D, that uses mature 2D object detectors for finegrained classification and localization, and a PointNet backbone for geometric embedding. The network processes fused data from images and point clouds for end-to-end learning of counts. We perform experiments on a novel synthetic dataset for inventory management in retail, which we construct and make publicly available to the community. Our results show that regression-based 3D counting methods systematically outperform detection-based methods, and reveal that directly learning from raw point clouds greatly assists count estimation under extreme occlusion. Finally, we study the effectiveness of CountNet3D on a large dataset of real-world scenes where extreme occlusion is present and achieve an error rate of 11.01%.

count=1
* A Morphology Focused Diffusion Probabilistic Model for Synthesis of Histopathology Images
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Moghadam_A_Morphology_Focused_Diffusion_Probabilistic_Model_for_Synthesis_of_Histopathology_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Moghadam_A_Morphology_Focused_Diffusion_Probabilistic_Model_for_Synthesis_of_Histopathology_WACV_2023_paper.pdf)]
    * Title: A Morphology Focused Diffusion Probabilistic Model for Synthesis of Histopathology Images
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Puria Azadi Moghadam, Sanne Van Dalen, Karina C. Martin, Jochen Lennerz, Stephen Yip, Hossein Farahani, Ali Bashashati
    * Abstract: Visual microscopic study of diseased tissue by pathologists has been the cornerstone for cancer diagnosis and prognostication for more than a century. Recently, deep learning methods have made significant advances in the analysis and classification of tissue images. However, there has been limited work on the utility of such models in generating histopathology images. These synthetic images have several applications in pathology including utilities in education, proficiency testing, privacy, and data sharing. Recently, diffusion probabilistic models were introduced to generate high quality images. Here, for the first time, we investigate the potential use of such models along with prioritized morphology weighting and color normalization to synthesize high quality histopathology images of brain cancer. Our detailed results show that diffusion probabilistic models are capable of synthesizing a wide range of histopathology images and have superior performance compared to generative adversarial networks.

count=1
* FishTrack23: An Ensemble Underwater Dataset for Multi-Object Tracking
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Dawkins_FishTrack23_An_Ensemble_Underwater_Dataset_for_Multi-Object_Tracking_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Dawkins_FishTrack23_An_Ensemble_Underwater_Dataset_for_Multi-Object_Tracking_WACV_2024_paper.pdf)]
    * Title: FishTrack23: An Ensemble Underwater Dataset for Multi-Object Tracking
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Matthew Dawkins, Jack Prior, Bryon Lewis, Robin Faillettaz, Thompson Banez, Mary Salvi, Audrey Rollo, Julien Simon, Matthew Campbell, Matthew Lucero, Aashish Chaudhary, Benjamin Richards, Anthony Hoogs
    * Abstract: Tracking and classifying fish in optical underwater imagery presents several challenges which are encountered less frequently in terrestrial domains. Video may contain large schools comprised of many individuals, dynamic natural backgrounds, highly variable target scales, volatile collection conditions, and non-fish moving confusers including debris, marine snow, and other organisms. Additionally, there is a lack of large public datasets for algorithm evaluation available in this domain. The contributions of this paper is three fold. First, we present the FishTrack23 dataset which provides a large quantity of expert-annotated fish groundtruth tracks, in imagery and video collected across a range of different backgrounds, locations, collection conditions, and organizations. Approximately 850k bounding boxes across 26k tracks are included in the release of the ensemble, with potential for future growth in later releases. Second, we evaluate improvements upon baseline object detectors, trackers and classifiers on the dataset. Lastly, we integrate these methods into web and desktop interfaces to expedite annotation generation on new datasets.

count=1
* Deep Neural Networks Segment Neuronal Membranes in Electron Microscopy Images
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/459a4ddcb586f24efd9395aa7662bc7c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/459a4ddcb586f24efd9395aa7662bc7c-Paper.pdf)]
    * Title: Deep Neural Networks Segment Neuronal Membranes in Electron Microscopy Images
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Dan Ciresan, Alessandro Giusti, Luca Gambardella, Jürgen Schmidhuber
    * Abstract: We address a central problem of neuroanatomy, namely, the automatic segmentation of neuronal structures depicted in stacks of electron microscopy (EM) images. This is necessary to efficiently map 3D brain structure and connectivity. To segment {\em biological} neuron membranes, we use a special type of deep {\em artificial} neural network as a pixel classifier. The label of each pixel (membrane or non-membrane) is predicted from raw pixel values in a square window centered on it. The input layer maps each window pixel to a neuron. It is followed by a succession of convolutional and max-pooling layers which preserve 2D information and extract features with increasing levels of abstraction. The output layer produces a calibrated probability for each class. The classifier is trained by plain gradient descent on a $512 \times 512 \times 30$ stack with known ground truth, and tested on a stack of the same size (ground truth unknown to the authors) by the organizers of the ISBI 2012 EM Segmentation Challenge. Even without problem-specific post-processing, our approach outperforms competing techniques by a large margin in all three considered metrics, i.e. \emph{rand error}, \emph{warping error} and \emph{pixel error}. For pixel error, our approach is the only one outperforming a second human observer.

count=1
* Planar Ultrametrics for Image Segmentation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2015/hash/3416a75f4cea9109507cacd8e2f2aefc-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2015/file/3416a75f4cea9109507cacd8e2f2aefc-Paper.pdf)]
    * Title: Planar Ultrametrics for Image Segmentation
    * Publisher: NeurIPS
    * Publication Date: `2015`
    * Authors: Julian E. Yarkony, Charless Fowlkes
    * Abstract: We study the problem of hierarchical clustering on planar graphs. We formulate this in terms of finding the closest ultrametric to a specified set of distances and solve it using an LP relaxation that leverages minimum cost perfect matching as a subroutine to efficiently explore the space of planar partitions. We apply our algorithm to the problem of hierarchical image segmentation.

count=1
* An Error Detection and Correction Framework for Connectomics
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/4500e4037738e13c0c18db508e18d483-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/4500e4037738e13c0c18db508e18d483-Paper.pdf)]
    * Title: An Error Detection and Correction Framework for Connectomics
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Jonathan Zung, Ignacio Tartavull, Kisuk Lee, H. Sebastian Seung
    * Abstract: We define and study error detection and correction tasks that are useful for 3D reconstruction of neurons from electron microscopic imagery, and for image segmentation more generally. Both tasks take as input the raw image and a binary mask representing a candidate object. For the error detection task, the desired output is a map of split and merge errors in the object. For the error correction task, the desired output is the true object. We call this object mask pruning, because the candidate object mask is assumed to be a superset of the true object. We train multiscale 3D convolutional networks to perform both tasks. We find that the error-detecting net can achieve high accuracy. The accuracy of the error-correcting net is enhanced if its input object mask is ``advice'' (union of erroneous objects) from the error-detecting net.

count=1
* Towards Automatic Concept-based Explanations
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/77d2afcb31f6493e350fca61764efb9a-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/77d2afcb31f6493e350fca61764efb9a-Paper.pdf)]
    * Title: Towards Automatic Concept-based Explanations
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Amirata Ghorbani, James Wexler, James Y. Zou, Been Kim
    * Abstract: Interpretability has become an important topic of research as more machine learning (ML) models are deployed and widely used to make important decisions. Most of the current explanation methods provide explanations through feature importance scores, which identify features that are important for each individual input. However, how to systematically summarize and interpret such per sample feature importance scores itself is challenging. In this work, we propose principles and desiderata for \emph{concept} based explanation, which goes beyond per-sample features to identify higher level human-understandable concepts that apply across the entire dataset. We develop a new algorithm, ACE, to automatically extract visual concepts. Our systematic experiments demonstrate that \alg discovers concepts that are human-meaningful, coherent and important for the neural network's predictions.

count=1
* Ultrametric Fitting by Gradient Descent
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/b865367fc4c0845c0682bd466e6ebf4c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/b865367fc4c0845c0682bd466e6ebf4c-Paper.pdf)]
    * Title: Ultrametric Fitting by Gradient Descent
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Giovanni Chierchia, Benjamin Perret
    * Abstract: We study the problem of fitting an ultrametric distance to a dissimilarity graph in the context of hierarchical cluster analysis. Standard hierarchical clustering methods are specified procedurally, rather than in terms of the cost function to be optimized. We aim to overcome this limitation by presenting a general optimization framework for ultrametric fitting. Our approach consists of modeling the latter as a constrained optimization problem over the continuous space of ultrametrics. So doing, we can leverage the simple, yet effective, idea of replacing the ultrametric constraint with a min-max operation injected directly into the cost function. The proposed reformulation leads to an unconstrained optimization problem that can be efficiently solved by gradient descent methods. The flexibility of our framework allows us to investigate several cost functions, following the classic paradigm of combining a data fidelity term with a regularization. While we provide no theoretical guarantee to find the global optimum, the numerical results obtained over a number of synthetic and real datasets demonstrate the good performance of our approach with respect to state-of-the-art agglomerative algorithms. This makes us believe that the proposed framework sheds new light on the way to design a new generation of hierarchical clustering methods. Our code is made publicly available at https://github.com/PerretB/ultrametric-fitting.

count=1
* Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/e6e713296627dff6475085cc6a224464-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/e6e713296627dff6475085cc6a224464-Paper.pdf)]
    * Title: Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Cheng-Chun Hsu, Kuang-Jui Hsu, Chung-Chi Tsai, Yen-Yu Lin, Yung-Yu Chuang
    * Abstract: This paper presents a weakly supervised instance segmentation method that consumes training data with tight bounding box annotations. The major difficulty lies in the uncertain figure-ground separation within each bounding box since there is no supervisory signal about it. We address the difficulty by formulating the problem as a multiple instance learning (MIL) task, and generate positive and negative bags based on the sweeping lines of each bounding box. The proposed deep model integrates MIL into a fully supervised instance segmentation network, and can be derived by the objective consisting of two terms, i.e., the unary term and the pairwise term. The former estimates the foreground and background areas of each bounding box while the latter maintains the unity of the estimated object masks. The experimental results show that our method performs favorably against existing weakly supervised methods and even surpasses some fully supervised methods for instance segmentation on the PASCAL VOC dataset.

count=1
* Global Convergence and Variance Reduction for a Class of Nonconvex-Nonconcave Minimax Problems
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/0cc6928e741d75e7a92396317522069e-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/0cc6928e741d75e7a92396317522069e-Paper.pdf)]
    * Title: Global Convergence and Variance Reduction for a Class of Nonconvex-Nonconcave Minimax Problems
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Junchi Yang, Negar Kiyavash, Niao He
    * Abstract: Nonconvex minimax problems appear frequently in emerging machine learning applications, such as generative adversarial networks and adversarial learning. Simple algorithms such as the gradient descent ascent (GDA) are the common practice for solving these nonconvex games and receive lots of empirical success. Yet, it is known that these vanilla GDA algorithms with constant stepsize can potentially diverge even in the convex setting. In this work, we show that for a subclass of nonconvex-nonconcave objectives satisfying a so-called two-sided Polyak-{\L}ojasiewicz inequality, the alternating gradient descent ascent (AGDA) algorithm converges globally at a linear rate and the stochastic AGDA achieves a sublinear rate. We further develop a variance reduced algorithm that attains a provably faster rate than AGDA when the problem has the finite-sum structure.

count=1
* How Can I Explain This to You? An Empirical Study of Deep Neural Network Explanation Methods
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/2c29d89cc56cdb191c60db2f0bae796b-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/2c29d89cc56cdb191c60db2f0bae796b-Paper.pdf)]
    * Title: How Can I Explain This to You? An Empirical Study of Deep Neural Network Explanation Methods
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Jeya Vikranth Jeyakumar, Joseph Noor, Yu-Hsi Cheng, Luis Garcia, Mani Srivastava
    * Abstract: Explaining the inner workings of deep neural network models have received considerable attention in recent years. Researchers have attempted to provide human parseable explanations justifying why a model performed a specific classification. Although many of these toolkits are available for use, it is unclear which style of explanation is preferred by end-users, thereby demanding investigation. We performed a cross-analysis Amazon Mechanical Turk study comparing the popular state-of-the-art explanation methods to empirically determine which are better in explaining model decisions. The participants were asked to compare explanation methods across applications spanning image, text, audio, and sensory domains. Among the surveyed methods, explanation-by-example was preferred in all domains except text sentiment classification, where LIME's method of annotating input text was preferred. We highlight qualitative aspects of employing the studied explainability methods and conclude with implications for researchers and engineers that seek to incorporate explanations into user-facing deployments.

count=1
* Detecting Interactions from Neural Networks via Topological Analysis
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/473803f0f2ebd77d83ee60daaa61f381-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/473803f0f2ebd77d83ee60daaa61f381-Paper.pdf)]
    * Title: Detecting Interactions from Neural Networks via Topological Analysis
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Zirui Liu, Qingquan Song, Kaixiong Zhou, Ting-Hsiang Wang, Ying Shan, Xia Hu
    * Abstract: Detecting statistical interactions between input features is a crucial and challenging task. Recent advances demonstrate that it is possible to extract learned interactions from trained neural networks. It has also been observed that, in neural networks, any interacting features must follow a strongly weighted connection to common hidden units. Motivated by the observation, in this paper, we propose to investigate the interaction detection problem from a novel topological perspective by analyzing the connectivity in neural networks. Specially, we propose a new measure for quantifying interaction strength, based upon the well-received theory of persistent homology. Based on this measure, a Persistence Interaction Dection (PID) algorithm is developed to efficiently detect interactions. Our proposed algorithm is evaluated across a number of interaction detection tasks on several synthetic and real-world datasets with different hyperparameters. Experimental results validate that the PID algorithm outperforms the state-of-the-art baselines.

count=1
* Self-Supervised Visual Representation Learning from Hierarchical Grouping
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/c1502ae5a4d514baec129f72948c266e-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/c1502ae5a4d514baec129f72948c266e-Paper.pdf)]
    * Title: Self-Supervised Visual Representation Learning from Hierarchical Grouping
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Xiao Zhang, Michael Maire
    * Abstract: We create a framework for bootstrapping visual representation learning from a primitive visual grouping capability. We operationalize grouping via a contour detector that partitions an image into regions, followed by merging of those regions into a tree hierarchy. A small supervised dataset suffices for training this grouping primitive. Across a large unlabeled dataset, we apply this learned primitive to automatically predict hierarchical region structure. These predictions serve as guidance for self-supervised contrastive feature learning: we task a deep network with producing per-pixel embeddings whose pairwise distances respect the region hierarchy. Experiments demonstrate that our approach can serve as state-of-the-art generic pre-training, benefiting downstream tasks. We additionally explore applications to semantic region search and video-based object instance tracking.

count=1
* K-Net: Towards Unified Image Segmentation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/55a7cf9c71f1c9c495413f934dd1a158-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/55a7cf9c71f1c9c495413f934dd1a158-Paper.pdf)]
    * Title: K-Net: Towards Unified Image Segmentation
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Wenwei Zhang, Jiangmiao Pang, Kai Chen, Chen Change Loy
    * Abstract: Semantic, instance, and panoptic segmentations have been addressed using different and specialized frameworks despite their underlying connections. This paper presents a unified, simple, and effective framework for these essentially similar tasks. The framework, named K-Net, segments both instances and semantic categories consistently by a group of learnable kernels, where each kernel is responsible for generating a mask for either a potential instance or a stuff class. To remedy the difficulties of distinguishing various instances, we propose a kernel update strategy that enables each kernel dynamic and conditional on its meaningful group in the input image. K-Net can be trained in an end-to-end manner with bipartite matching, and its training and inference are naturally NMS-free and box-free. Without bells and whistles, K-Net surpasses all previous published state-of-the-art single-model results of panoptic segmentation on MS COCO test-dev split and semantic segmentation on ADE20K val split with 55.2% PQ and 54.3% mIoU, respectively. Its instance segmentation performance is also on par with Cascade Mask R-CNN on MS COCO with 60%-90% faster inference speeds. Code and models will be released at https://github.com/ZwwWayne/K-Net/.

count=1
* Combinatorial Optimization for Panoptic Segmentation: A Fully Differentiable Approach
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/83a368f54768f506b833130584455df4-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/83a368f54768f506b833130584455df4-Paper.pdf)]
    * Title: Combinatorial Optimization for Panoptic Segmentation: A Fully Differentiable Approach
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Ahmed Abbas, Paul Swoboda
    * Abstract: We propose a fully differentiable architecture for simultaneous semantic and instance segmentation (a.k.a. panoptic segmentation) consisting of a convolutional neural network and an asymmetric multiway cut problem solver. The latter solves a combinatorial optimization problem that elegantly incorporates semantic and boundary predictions to produce a panoptic labeling. Our formulation allows to directly maximize a smooth surrogate of the panoptic quality metric by backpropagating the gradient through the optimization problem. Experimental evaluation shows improvement by backpropagating through the optimization problem w.r.t. comparable approaches on Cityscapes and COCO datasets. Overall, our approach of combinatorial optimization for panoptic segmentation (COPS) shows the utility of using optimization in tandem with deep learning in a challenging large scale real-world problem and showcases benefits and insights into training such an architecture.

count=1
* Learning Equivariant Segmentation with Instance-Unique Querying
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/53a525a5f8910609263ffd130ef370b8-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/53a525a5f8910609263ffd130ef370b8-Paper-Conference.pdf)]
    * Title: Learning Equivariant Segmentation with Instance-Unique Querying
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Wenguan Wang, James Liang, Dongfang Liu
    * Abstract: Prevalent state-of-the-art instance segmentation methods fall into a query-based scheme, in which instance masks are derived by querying the image feature using a set of instance-aware embeddings. In this work, we devise a new training framework that boosts query-based models through discriminative query embedding learning. It explores two essential properties, namely dataset-level uniqueness and transformation equivariance, of the relation between queries and instances. First, our algorithm uses the queries to retrieve the corresponding instances from the whole training dataset, instead of only searching within individual scenes. As querying instances across scenes is more challenging, the segmenters are forced to learn more discriminative queries for effective instance separation. Second, our algorithm encourages both image (instance) representations and queries to be equivariant against geometric transformations, leading to more robust, instance-query matching. On top of four famous, query-based models (i.e., CondInst, SOLOv2, SOTR, and Mask2Former), our training algorithm provides significant performance gains (e.g., +1.6 – 3.2 AP) on COCO dataset. In addition, our algorithm promotes the performance of SOLOv2 by 2.7 AP, on LVISv1 dataset.

count=1
* NIS3D: A Completely Annotated Benchmark for Dense 3D Nuclei Image Segmentation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/0f2cd3d09a132757555b602e2dd43784-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/0f2cd3d09a132757555b602e2dd43784-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: NIS3D: A Completely Annotated Benchmark for Dense 3D Nuclei Image Segmentation
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Wei Zheng, Cheng Peng, Zeyuan Hou, Boyu Lyu, Mengfan Wang, Xuelong Mi, Shuoxuan Qiao, Yinan Wan, Guoqiang Yu
    * Abstract: 3D segmentation of nuclei images is a fundamental task for many biological studies. Despite the rapid advances of large-volume 3D imaging acquisition methods and the emergence of sophisticated algorithms to segment the nuclei in recent years, a benchmark with all cells completely annotated is still missing, making it hard to accurately assess and further improve the performance of the algorithms. The existing nuclei segmentation benchmarks either worked on 2D only or annotated a small number of 3D cells, perhaps due to the high cost of 3D annotation for large-scale data. To fulfill the critical need, we constructed NIS3D, a 3D, high cell density, large-volume, and completely annotated Nuclei Image Segmentation benchmark, assisted by our newly designed semi-automatic annotation software. NIS3D provides more than 22,000 cells across multiple most-used species in this area. Each cell is labeled by three independent annotators, so we can measure the variability of each annotation. A confidence score is computed for each cell, allowing more nuanced testing and performance comparison. A comprehensive review on the methods of segmenting 3D dense nuclei was conducted. The benchmark was used to evaluate the performance of several selected state-of-the-art segmentation algorithms. The best of current methods is still far away from human-level accuracy, corroborating the necessity of generating such a benchmark. The testing results also demonstrated the strength and weakness of each method and pointed out the directions of further methodological development. The dataset can be downloaded here: https://github.com/yu-lab-vt/NIS3D.

count=1
* SubseasonalClimateUSA: A Dataset for Subseasonal Forecasting and Benchmarking
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/18ef499ee57c4822e1e3ea9b9948af18-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/18ef499ee57c4822e1e3ea9b9948af18-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: SubseasonalClimateUSA: A Dataset for Subseasonal Forecasting and Benchmarking
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Soukayna Mouatadid, Paulo Orenstein, Genevieve Flaspohler, Miruna Oprescu, Judah Cohen, Franklyn Wang, Sean Knight, Maria Geogdzhayeva, Sam Levang, Ernest Fraenkel, Lester Mackey
    * Abstract: Subseasonal forecasting of the weather two to six weeks in advance is critical for resource allocation and climate adaptation but poses many challenges for the forecasting community. At this forecast horizon, physics-based dynamical models have limited skill, and the targets for prediction depend in a complex manner on both local weather variables and global climate variables. Recently, machine learning methods have shown promise in advancing the state of the art but only at the cost of complex data curation, integrating expert knowledge with aggregation across multiple relevant data sources, file formats, and temporal and spatial resolutions.To streamline this process and accelerate future development, we introduce SubseasonalClimateUSA, a curated dataset for training and benchmarking subseasonal forecasting models in the United States. We use this dataset to benchmark a diverse suite of models, including operational dynamical models, classical meteorological baselines, and ten state-of-the-art machine learning and deep learning-based methods from the literature. Overall, our benchmarks suggest simple and effective ways to extend the accuracy of current operational models. SubseasonalClimateUSA is regularly updated and accessible via the https://github.com/microsoft/subseasonal_data/ Python package.

count=1
* Transformer as a hippocampal memory consolidation model based on NMDAR-inspired nonlinearity
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/2f1eb4c897e63870eee9a0a0f7a10332-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/2f1eb4c897e63870eee9a0a0f7a10332-Paper-Conference.pdf)]
    * Title: Transformer as a hippocampal memory consolidation model based on NMDAR-inspired nonlinearity
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Dong Kyum Kim, Jea Kwon, Meeyoung Cha, C. Lee
    * Abstract: The hippocampus plays a critical role in learning, memory, and spatial representation, processes that depend on the NMDA receptor (NMDAR). Inspired by recent findings that compare deep learning models to the hippocampus, we propose a new nonlinear activation function that mimics NMDAR dynamics. NMDAR-like nonlinearity shifts short-term working memory into long-term reference memory in transformers, thus enhancing a process that is similar to memory consolidation in the mammalian brain. We design a navigation task assessing these two memory functions and show that manipulating the activation function (i.e., mimicking the Mg$^{2+}$-gating of NMDAR) disrupts long-term memory processes. Our experiments suggest that place cell-like functions and reference memory reside in the feed-forward network layer of transformers and that nonlinearity drives these processes. We discuss the role of NMDAR-like nonlinearity in establishing this striking resemblance between transformer architecture and hippocampal spatial representation.

count=1
* Conformal Prediction for Time Series with Modern Hopfield Networks
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/aef75887979ae1287b5deb54a1e3cbda-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/aef75887979ae1287b5deb54a1e3cbda-Paper-Conference.pdf)]
    * Title: Conformal Prediction for Time Series with Modern Hopfield Networks
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Andreas Auer, Martin Gauch, Daniel Klotz, Sepp Hochreiter
    * Abstract: To quantify uncertainty, conformal prediction methods are gaining continuously more interest and have already been successfully applied to various domains. However, they are difficult to apply to time series as the autocorrelative structure of time series violates basic assumptions required by conformal prediction. We propose HopCPT, a novel conformal prediction approach for time series that not only copes with temporal structures but leverages them. We show that our approach is theoretically well justified for time series where temporal dependencies are present. In experiments, we demonstrate that our new approach outperforms state-of-the-art conformal prediction methods on multiple real-world time series datasets from four different domains.

