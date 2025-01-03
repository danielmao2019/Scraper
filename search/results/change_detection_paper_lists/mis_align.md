count=26
* Multi-Shot Imaging: Joint Alignment, Deblurring and Resolution-Enhancement
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Zhang_Multi-Shot_Imaging_Joint_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Zhang_Multi-Shot_Imaging_Joint_2014_CVPR_paper.pdf)]
    * Title: Multi-Shot Imaging: Joint Alignment, Deblurring and Resolution-Enhancement
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Haichao Zhang, Lawrence Carin
    * Abstract: The capture of multiple images is a simple way to increase the chance of capturing a good photo with a light-weight hand-held camera, for which the camera-shake blur is typically a nuisance problem. The naive approach of selecting the single best captured photo as output does not take full advantage of all the observations. Conventional multi-image blind deblurring methods can take all observations as input but usually require the multiple images are well aligned. However, the multiple blurry images captured in the presence of camera shake are rarely free from mis-alignment. Registering multiple blurry images is a challenging task due to the presence of blur while deblurring of multiple blurry images requires accurate alignment, leading to an intrinsically coupled problem. In this paper, we propose a blind multi-image restoration method which can achieve joint alignment, non-uniform deblurring, together with resolution enhancement from multiple low-quality images. Experiments on several real-world images with comparison to some previous methods validate the effectiveness of the proposed method.

count=11
* Robust Video Content Alignment and Compensation for Rain Removal in a CNN Framework
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Robust_Video_Content_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Robust_Video_Content_CVPR_2018_paper.pdf)]
    * Title: Robust Video Content Alignment and Compensation for Rain Removal in a CNN Framework
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Jie Chen, Cheen-Hau Tan, Junhui Hou, Lap-Pui Chau, He Li
    * Abstract: Rain removal is important for improving the robustness of outdoor vision based systems. Current rain removal methods show limitations either for complex dynamic scenes shot from fast moving cameras, or under torrential rain fall with opaque occlusions. We propose a novel derain algorithm, which applies superpixel (SP) segmentation to decompose the scene into depth consistent units. Alignment of scene contents are done at the SP level, which proves to be robust towards rain occlusion and fast camera motion. Two alignment output tensors, i.e., optimal temporal match tensor and sorted spatial-temporal match tensor, provide informative clues for rain streak location and occluded background contents to generate an intermediate derain output. These tensors will be subsequently prepared as input features for a convolutional neural network to restore high frequency details to the intermediate output for compensation of misalignment blur. Extensive evaluations show that up to 5dB reconstruction PSNR advantage is achieved over state-of-the-art methods. Visual inspection shows that much cleaner rain removal is achieved especially for highly dynamic scenes with heavy and opaque rainfall from a fast moving camera.

count=11
* Style-Based Global Appearance Flow for Virtual Try-On
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/He_Style-Based_Global_Appearance_Flow_for_Virtual_Try-On_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Style-Based_Global_Appearance_Flow_for_Virtual_Try-On_CVPR_2022_paper.pdf)]
    * Title: Style-Based Global Appearance Flow for Virtual Try-On
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Sen He, Yi-Zhe Song, Tao Xiang
    * Abstract: Image-based virtual try-on aims to fit an in-shop garment into a clothed person image. To achieve this, a key step is garment warping which spatially aligns the target garment with the corresponding body parts in the person image. Prior methods typically adopt a local appearance flow estimation model. They are thus intrinsically susceptible to difficult body poses/occlusions and large mis-alignments between person and garment images. To overcome this limitation, a novel global appearance flow estimation model is proposed in this work. For the first time, a StyleGAN based architecture is adopted for appearance flow estimation. This enables us to take advantage of a global style vector to encode a whole-image context to cope with the aforementioned challenges. To guide the StyleGAN flow generator to pay more attention to local garment deformation, a flow refinement module is introduced to add local context. Experiment results on a popular virtual try-on benchmark show that our method achieves new state-of-the-art performance. It is particularly effective in a 'in-the-wild' application scenario where the reference image is full-body resulting in a large mis-alignment with the garment image.

count=9
* Improving Semantic Segmentation via Video Propagation and Label Relaxation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf)]
    * Title: Improving Semantic Segmentation via Video Propagation and Label Relaxation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yi Zhu,  Karan Sapra,  Fitsum A. Reda,  Kevin J. Shih,  Shawn Newsam,  Andrew Tao,  Bryan Catanzaro
    * Abstract: Semantic segmentation requires large amounts of pixel-wise annotations to learn accurate models. In this paper, we present a video prediction-based methodology to scale up training sets by synthesizing new training samples in order to improve the accuracy of semantic segmentation networks. We exploit video prediction models' ability to predict future frames in order to also predict future labels. A joint propagation strategy is also proposed to alleviate mis-alignments in synthesized samples. We demonstrate that training segmentation models on datasets augmented by the synthesized samples leads to significant improvements in accuracy. Furthermore, we introduce a novel boundary label relaxation technique that makes training robust to annotation noise and propagation artifacts along object boundaries. Our proposed methods achieve state-of-the-art mIoUs of 83.5% on Cityscapes and 82.9% on CamVid. Our single model, without model ensembles, achieves 72.8% mIoU on the KITTI semantic segmentation test set, which surpasses the winning entry of the ROB challenge 2018.

count=9
* Searching Parameterized AP Loss for Object Detection
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/b9009beb804fa097c04d226a8ba5102e-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/b9009beb804fa097c04d226a8ba5102e-Paper.pdf)]
    * Title: Searching Parameterized AP Loss for Object Detection
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Tao Chenxin, Zizhang Li, Xizhou Zhu, Gao Huang, Yong Liu, jifeng dai
    * Abstract: Loss functions play an important role in training deep-network-based object detectors. The most widely used evaluation metric for object detection is Average Precision (AP), which captures the performance of localization and classification sub-tasks simultaneously. However, due to the non-differentiable nature of the AP metric, traditional object detectors adopt separate differentiable losses for the two sub-tasks. Such a mis-alignment issue may well lead to performance degradation. To address this, existing works seek to design surrogate losses for the AP metric manually, which requires expertise and may still be sub-optimal. In this paper, we propose Parameterized AP Loss, where parameterized functions are introduced to substitute the non-differentiable components in the AP calculation. Different AP approximations are thus represented by a family of parameterized functions in a unified formula. Automatic parameter search algorithm is then employed to search for the optimal parameters. Extensive experiments on the COCO benchmark with three different object detectors (i.e., RetinaNet, Faster R-CNN, and Deformable DETR) demonstrate that the proposed Parameterized AP Loss consistently outperforms existing handcrafted losses. Code shall be released.

count=7
* Deep Burst Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.pdf)]
    * Title: Deep Burst Super-Resolution
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Goutam Bhat, Martin Danelljan, Luc Van Gool, Radu Timofte
    * Abstract: While single-image super-resolution (SISR) has attracted substantial interest in recent years, the proposed approaches are limited to learning image priors in order to add high frequency details. In contrast, multi-frame super-resolution (MFSR) offers the possibility of reconstructing rich details by combining signal information from multiple shifted images. This key advantage, along with the increasing popularity of burst photography, have made MFSR an important problem for real-world applications. We propose a novel architecture for the burst super-resolution task. Our network takes multiple noisy RAW images as input, and generates a denoised, super-resolved RGB image as output. This is achieved by explicitly aligning deep embeddings of the input frames using pixel-wise optical flow. The information from all frames are then adaptively merged using an attention-based fusion module. In order to enable training and evaluation on real-world data, we additionally introduce the BurstSR dataset, consisting of smartphone bursts and high-resolution DSLR ground-truth. We perform comprehensive experimental analysis, demonstrating the effectiveness of the proposed architecture.

count=5
* Edit Probability for Scene Text Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Bai_Edit_Probability_for_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bai_Edit_Probability_for_CVPR_2018_paper.pdf)]
    * Title: Edit Probability for Scene Text Recognition
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Fan Bai, Zhanzhan Cheng, Yi Niu, Shiliang Pu, Shuigeng Zhou
    * Abstract: We consider the scene text recognition problem under the attention-based encoder-decoder framework, which is the state of the art. The existing methods usually employ a frame-wise maximal likelihood loss to optimize the models. When we train the model, the misalignment between the ground truth strings and the attention's output sequences of probability distribution, which is caused by missing or superfluous characters, will confuse and mislead the training process, and consequently make the training costly and degrade the recognition accuracy. To handle this problem, we propose a novel method called edit probability (EP) for scene text recognition. EP tries to effectively estimate the probability of generating a string from the output sequence of probability distribution conditioned on the input image, while considering the possible occurrences of missing/superfluous characters. The advantage lies in that the training process can focus on the missing, superfluous and unrecognized characters, and thus the impact of the misalignment problem can be alleviated or even overcome. We conduct extensive experiments on standard benchmarks, including the IIIT-5K, Street View Text and ICDAR datasets. Experimental results show that the EP can substantially boost scene text recognition performance.

count=5
* Multi-Shot Deblurring for 3D Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w10/html/Arun_Multi-Shot_Deblurring_for_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w10/papers/Arun_Multi-Shot_Deblurring_for_ICCV_2015_paper.pdf)]
    * Title: Multi-Shot Deblurring for 3D Scenes
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: M. Arun, A. N. Rajagopalan, Gunasekaran Seetharaman
    * Abstract: The presence of motion blur is unavoidable in hand-held cameras, especially in low-light conditions. In this paper, we address the inverse rendering problem of estimating the latent image, scene depth and camera motion from a set of differently blurred images of the scene. Our framework can account for depth variations, non-uniform motion blur as well as mis-alignments in the captured observations. We initially describe an iterative algorithm to estimate ego motion in 3D scenes by suitably harnessing the point spread functions across the blurred images at different spatial locations. This is followed by recovery of latent image and scene depth by alternate minimization.

count=5
* Burst Reflection Removal Using Reflection Motion Aggregation Cues
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Prasad_Burst_Reflection_Removal_Using_Reflection_Motion_Aggregation_Cues_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Prasad_Burst_Reflection_Removal_Using_Reflection_Motion_Aggregation_Cues_WACV_2023_paper.pdf)]
    * Title: Burst Reflection Removal Using Reflection Motion Aggregation Cues
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: B. H. Pawan Prasad, Green Rosh K. S., Lokesh R. B., Kaushik Mitra
    * Abstract: Single image reflection removal has attracted lot of interest in the recent past with data driven approaches demonstrating significant improvements. However deep learning based approaches for multi-image reflection removal remains relatively less explored. The existing multi-image methods require input images to be captured at sufficiently different view points with wide baselines. This makes it cumbersome for the user who is required to capture the scene by moving the camera in multiple directions. A more convenient way is to capture a burst of images in a short time duration without providing any specific instructions to the user. A burst of images captured on a hand-held device provide crucial cues that rely on the subtle handshakes created during the capture process to separate the reflection and the transmission layers. In this paper, we propose a multi-stage deep learning based approach for burst reflection removal. In the first stage, we perform reflection suppression on the individual images. In the second stage, a novel reflection motion aggregation (RMA) cue is extracted that emphasizes the transmission layer more than the reflection layer to aid better layer separation. In our final stage we use this RMA cue as a guide to remove reflections from the input. We provide the first real world burst images dataset along with ground truth for reflection removal that can enable future benchmarking. We evaluate both qualitatively and quantitatively to demonstrate the superiority of the proposed approach. Our method achieves 2 dB improvement in PSNR over single image based methods and 1 dB over multi-image based methods.

count=4
* A Pose-Sensitive Embedding for Person Re-Identification With Expanded Cross Neighborhood Re-Ranking
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Sarfraz_A_Pose-Sensitive_Embedding_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sarfraz_A_Pose-Sensitive_Embedding_CVPR_2018_paper.pdf)]
    * Title: A Pose-Sensitive Embedding for Person Re-Identification With Expanded Cross Neighborhood Re-Ranking
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: M. Saquib Sarfraz, Arne Schumann, Andreas Eberle, Rainer Stiefelhagen
    * Abstract: Person re-identification is a challenging retrieval task that requires matching a person’s acquired image across non-overlapping camera views. In this paper we propose an effective approach that incorporates both the fine and coarse pose information of the person to learn a discrim- inative embedding. In contrast to the recent direction of explicitly modeling body parts or correcting for misalignment based on these, we show that a rather straightforward inclusion of acquired camera view and/or the detected joint locations into a convolutional neural network helps to learn a very effective representation. To increase retrieval performance, re-ranking techniques based on computed distances have recently gained much attention. We propose a new unsupervised and automatic re-ranking framework that achieves state-of-the-art re-ranking performance. We show that in contrast to the current state-of-the-art re-ranking methods our approach does not require to compute new rank lists for each image pair (e.g., based on reciprocal neighbors) and performs well by using simple direct rank list based comparison or even by just using the already computed euclidean distances between the images. We show that both our learned representation and our re-ranking method achieve state-of-the-art performance on a number of challenging surveillance image and video datasets.

count=4
* Panoramic Image Reflection Removal
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Hong_Panoramic_Image_Reflection_Removal_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_Panoramic_Image_Reflection_Removal_CVPR_2021_paper.pdf)]
    * Title: Panoramic Image Reflection Removal
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yuchen Hong, Qian Zheng, Lingran Zhao, Xudong Jiang, Alex C. Kot, Boxin Shi
    * Abstract: This paper studies the problem of panoramic image reflection removal, aiming at reliving the content ambiguity between reflection and transmission scenes. Although a partial view of the reflection scene is included in the panoramic image, it cannot be utilized directly due to its misalignment with the reflection-contaminated image. We propose a two-step approach to solve this problem, by first accomplishing geometric and photometric alignment for the reflection scene via a coarse-to-fine strategy, and then restoring the transmission scene via a recovery network. The proposed method is trained with a synthetic dataset and verified quantitatively with a real panoramic image dataset. The effectiveness of the proposed method is validated by the significant performance advantage over single image-based reflection removal methods and generalization capacity to limited-FoV scenarios captured by conventional camera or mobile phone users.

count=4
* Partial Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Partial_Person_Re-Identification_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Zheng_Partial_Person_Re-Identification_ICCV_2015_paper.pdf)]
    * Title: Partial Person Re-Identification
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Wei-Shi Zheng, Xiang Li, Tao Xiang, Shengcai Liao, Jianhuang Lai, Shaogang Gong
    * Abstract: We address a new partial person re-identification (re-id) problem, where only a partial observation of a person is available for matching across different non-overlapping camera views. This differs significantly from the conventional person re-id setting where it is assumed that the full body of a person is detected and aligned. To solve this more challenging and realistic re-id problem without the implicit assumption of manual body-parts alignment, we propose a matching framework consisting of 1) a local patch-level matching model based on a novel sparse representation classification formulation with explicit patch ambiguity modelling, and 2) a global part-based matching model providing complementary spatial layout information. Our framework is evaluated on a new partial person re-id dataset as well as two existing datasets modified to include partial person images. The results show that the proposed method outperforms significantly existing re-id methods as well as other partial visual matching methods.

count=4
* Misalignment-Robust Joint Filter for Cross-Modal Image Pairs
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Shibata_Misalignment-Robust_Joint_Filter_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Shibata_Misalignment-Robust_Joint_Filter_ICCV_2017_paper.pdf)]
    * Title: Misalignment-Robust Joint Filter for Cross-Modal Image Pairs
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Takashi Shibata, Masayuki Tanaka, Masatoshi Okutomi
    * Abstract: Although several powerful joint filters for cross-modal image pairs have been proposed, the existing joint filters generate severe artifacts when there are misalignments between a target and a guidance images. Our goal is to generate an artifact-free output image even from the misaligned target and guidance images. We propose a novel misalignment-robust joint filter based on weight-volume-based image composition and joint-filter cost volume. Our proposed method first generates a set of translated guidances. Next, the joint-filter cost volume and a set of filtered images are computed from the target image and the set of the translated guidances. Then, a weight volume is obtained from the joint-filter cost volume while considering a spatial smoothness and a label-sparseness. The final output image is composed by fusing the set of the filtered images with the weight volume for the filtered images. The key is to generate the final output image directly from the set of the filtered images by weighted averaging using the weight volume that is obtained from the joint-filter cost volume. The proposed framework is widely applicable and can involve any kind of joint filter. Experimental results show that the proposed method is effective for various applications including image denosing, image up-sampling, haze removal and depth map interpolation.

count=4
* Improving Generalization of Batch Whitening by Convolutional Unit Optimization
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Cho_Improving_Generalization_of_Batch_Whitening_by_Convolutional_Unit_Optimization_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Cho_Improving_Generalization_of_Batch_Whitening_by_Convolutional_Unit_Optimization_ICCV_2021_paper.pdf)]
    * Title: Improving Generalization of Batch Whitening by Convolutional Unit Optimization
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yooshin Cho, Hanbyel Cho, Youngsoo Kim, Junmo Kim
    * Abstract: Batch Whitening is a technique that accelerates and stabilizes training by transforming input features to have a zero mean (Centering) and a unit variance (Scaling), and by removing linear correlation between channels (Decorrelation). In commonly used structures, which are empirically optimized with Batch Normalization, the normalization layer appears between convolution and activation function. Following Batch Whitening studies have employed the same structure without further analysis; even Batch Whitening was analyzed on the premise that the input of a linear layer is whitened. To bridge the gap, we propose a new Convolutional Unit that in line with the theory, and our method generally improves the performance of Batch Whitening. Moreover, we show the inefficacy of the original Convolutional Unit by investigating rank and correlation of features. As our method is employable off-the-shelf whitening modules, we use Iterative Normalization (IterNorm), the state-of-the-art whitening module, and obtain significantly improved performance on five image classification datasets: CIFAR-10, CIFAR-100, CUB-200-2011, Stanford Dogs, and ImageNet. Notably, we verify that our method improves stability and performance of whitening when using large learning rate, group size, and iteration number.

count=4
* Non-Coaxial Event-Guided Motion Deblurring with Spatial Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Cho_Non-Coaxial_Event-Guided_Motion_Deblurring_with_Spatial_Alignment_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Cho_Non-Coaxial_Event-Guided_Motion_Deblurring_with_Spatial_Alignment_ICCV_2023_paper.pdf)]
    * Title: Non-Coaxial Event-Guided Motion Deblurring with Spatial Alignment
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Hoonhee Cho, Yuhwan Jeong, Taewoo Kim, Kuk-Jin Yoon
    * Abstract: Motion deblurring from a blurred image is a challenging computer vision problem because frame-based cameras lose information during the blurring process. Several attempts have compensated for the loss of motion information by using event cameras, which are bio-inspired sensors with a high temporal resolution. Even though most studies have assumed that image and event data are pixel-wise aligned, this is only possible with low-quality active-pixel sensor (APS) images and synthetic datasets. In real scenarios, obtaining per-pixel aligned event-RGB data is technically challenging since event and frame cameras have different optical axes. For the application of the event camera, we propose the first Non-coaxial Event-guided Image Deblurring (NEID) approach that utilizes the camera setup composed of a standard frame-based camera with a non-coaxial single event camera. To consider the per-pixel alignment between the image and event without additional devices, we propose the first NEID network that spatially aligns events to images while refining the image features from temporally dense event features. For training and evaluation of our network, we also present the first large-scale dataset, consisting of RGB frames with non-aligned events aimed at a breakthrough in motion deblurring with an event camera. Extensive experiments on various datasets demonstrate that the proposed method achieves significantly better results than the prior works in terms of performance and speed, and it can be applied for practical uses of event cameras.

count=4
* Misalign, Contrast then Distill: Rethinking Misalignments in Language-Image Pre-training
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Kim_Misalign_Contrast_then_Distill_Rethinking_Misalignments_in_Language-Image_Pre-training_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Misalign_Contrast_then_Distill_Rethinking_Misalignments_in_Language-Image_Pre-training_ICCV_2023_paper.pdf)]
    * Title: Misalign, Contrast then Distill: Rethinking Misalignments in Language-Image Pre-training
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Bumsoo Kim, Yeonsik Jo, Jinhyung Kim, Seunghwan Kim
    * Abstract: Contrastive Language-Image Pretraining has emerged as a prominent approach for training vision and text encoders with uncurated image-text pairs from the web. To enhance data-efficiency, recent efforts have introduced additional supervision terms that involve random-augmented views of the image. However, since the image augmentation process is unaware of its text counterpart, this procedure could cause various degrees of image-text misalignments during training. Prior methods either disregarded this discrepancy or introduced external models to mitigate the impact of misalignments during training. In contrast, we propose a novel metric learning approach that capitalizes on these misalignments as an additional training source, which we term "Misalign, Contrast then Distill (MCD)". Unlike previous methods that treat augmented images and their text counterparts as simple positive pairs, MCD predicts the continuous scales of misalignment caused by the augmentation. Our extensive experimental results show that our proposed MCD achieves state-of-the-art transferability in multiple classification and retrieval downstream datasets.

count=4
* No Representation Rules Them All in Category Discovery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/3f52ab4322e967efd312c38a68d07f01-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/3f52ab4322e967efd312c38a68d07f01-Paper-Conference.pdf)]
    * Title: No Representation Rules Them All in Category Discovery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Sagar Vaze, Andrea Vedaldi, Andrew Zisserman
    * Abstract: In this paper we tackle the problem of Generalized Category Discovery (GCD). Specifically, given a dataset with labelled and unlabelled images, the task is to cluster all images in the unlabelled subset, whether or not they belong to the labelled categories. Our first contribution is to recognise that most existing GCD benchmarks only contain labels for a single clustering of the data, making it difficult to ascertain whether models are leveraging the available labels to solve the GCD task, or simply solving an unsupervised clustering problem. As such, we present a synthetic dataset, named 'Clevr-4', for category discovery. Clevr-4 contains four equally valid partitions of the data, i.e based on object 'shape', 'texture' or 'color' or 'count'. To solve the task, models are required to extrapolate the taxonomy specified by labelled set, rather than simply latch onto a single natural grouping of the data. We use this dataset to demonstrate the limitations of unsupervised clustering in the GCD setting, showing that even very strong unsupervised models fail on Clevr-4. We further use Clevr-4 to examine the weaknesses of existing GCD algorithms, and propose a new method which addresses these shortcomings, leveraging consistent findings from the representation learning literature to do so. Our simple solution, which is based on `Mean Teachers' and termed $\mu$GCD, substantially outperforms implemented baselines on Clevr-4. Finally, when we transfer these findings to real data on the challenging Semantic Shift Benchmark suite, we find that $\mu$GCD outperforms all prior work, setting a new state-of-the-art.

count=4
* NAVI: Category-Agnostic Image Collections with High-Quality 3D Shape and Pose Annotations
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/efc90033e6e1b05485312dd09fe302b8-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/efc90033e6e1b05485312dd09fe302b8-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: NAVI: Category-Agnostic Image Collections with High-Quality 3D Shape and Pose Annotations
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Varun Jampani, Kevis-kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, Andre Araujo, Ricardo Martin Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, Howard Zhou
    * Abstract: Recent advances in neural reconstruction enable high-quality 3D object reconstruction from casually captured image collections. Current techniques mostly analyze their progress on relatively simple image collections where SfM techniques can provide ground-truth (GT) camera poses. We note that SfM techniques tend to fail on in-the-wild image collections such as image search results with varying backgrounds and illuminations. To enable systematic research progress on 3D reconstruction from casual image captures, we propose `NAVI': a new dataset of category-agnostic image collections of objects with high-quality 3D scans along with per-image 2D-3D alignments providing near-perfect GT camera parameters. These 2D-3D alignments allow us to extract accurate derivative annotations such as dense pixel correspondences, depth and segmentation maps. We demonstrate the use of NAVI image collections on different problem settings and show that NAVI enables more thorough evaluations that were not possible with existing datasets. We believe NAVI is beneficial for systematic research progress on 3D reconstruction and correspondence estimation.

count=3
* Panoramic Stereo Videos With a Single Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Aggarwal_Panoramic_Stereo_Videos_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Aggarwal_Panoramic_Stereo_Videos_CVPR_2016_paper.pdf)]
    * Title: Panoramic Stereo Videos With a Single Camera
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Rajat Aggarwal, Amrisha Vohra, Anoop M. Namboodiri
    * Abstract: We present a practical solution for generating 360 degree stereo panoramic videos using a single camera. Current approaches either use a moving camera that captures multiple images of a scene, which are then stitched together to form the final panorama, or use multiple cameras that are synchronized. A moving camera limits the solution to static scenes, while multi-camera solutions require dedicated calibrated setups. Our approach improves upon the existing solutions in two significant ways: It solves the problem using a single camera, thus minimizing the calibration problem and providing us the ability to convert any digital camera into a panoramic stereo capture device. It captures all the light rays required for stereo panoramas in a single frame using a compact custom designed mirror, thus making the design practical to manufacture and easier to use. We analyze several properties of the design as well as present panoramic stereo and depth estimation results.

count=3
* Spindle Net: Person Re-Identification With Human Body Region Guided Feature Decomposition and Fusion
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Spindle_Net_Person_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Spindle_Net_Person_CVPR_2017_paper.pdf)]
    * Title: Spindle Net: Person Re-Identification With Human Body Region Guided Feature Decomposition and Fusion
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Haiyu Zhao, Maoqing Tian, Shuyang Sun, Jing Shao, Junjie Yan, Shuai Yi, Xiaogang Wang, Xiaoou Tang
    * Abstract: Person re-identification (ReID) is an important task in video surveillance and has various applications. It is non-trivial due to complex background clutters, varying illumination conditions, and uncontrollable camera settings. Moreover, the person body misalignment caused by detectors or pose variations is sometimes too severe for feature matching across images. In this study, we propose a novel Convolutional Neural Network (CNN), called Spindle Net, based on human body region guided multi-stage feature decomposition and tree-structured competitive feature fusion. It is the first time human body structure information is considered in a CNN framework to facilitate feature learning. The proposed Spindle Net brings unique advantages: 1) it separately captures semantic features from different body regions thus the macro- and micro-body features can be well aligned across images, 2) the learned region features from different semantic regions are merged with a competitive scheme and discriminative features can be well preserved. State of the art performance can be achieved on multiple datasets by large margins. We further demonstrate the robustness and effectiveness of the proposed Spindle Net on our proposed dataset SenseReID without fine-tuning.

count=3
* An End-to-End TextSpotter With Explicit Alignment and Attention
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/He_An_End-to-End_TextSpotter_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/He_An_End-to-End_TextSpotter_CVPR_2018_paper.pdf)]
    * Title: An End-to-End TextSpotter With Explicit Alignment and Attention
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Tong He, Zhi Tian, Weilin Huang, Chunhua Shen, Yu Qiao, Changming Sun
    * Abstract: Text detection and recognition in natural images have long been considered as two separate tasks that are processed sequentially. Jointly training two tasks is non-trivial due to significant differences in learning difficulties and convergence rates. In this work, we present a conceptually simple yet efficient framework that simultaneously processes the two tasks in a united framework. Our main contributions are three-fold: (1) we propose a novel textalignment layer that allows it to precisely compute convolutional features of a text instance in arbitrary orientation, which is the key to boost the performance; (2) a character attention mechanism is introduced by using character spatial information as explicit supervision, leading to large improvements in recognition; (3) two technologies, together with a new RNN branch for word recognition, are integrated seamlessly into a single model which is end-to-end trainable. This allows the two tasks to work collaboratively by sharing convolutional features, which is critical to identify challenging text instances. Our model obtains impressive results in end-to-end recognition on the ICDAR 2015, significantly advancing the most recent results, with improvements of F-measure from (0.54, 0.51, 0.47) to (0.82, 0.77, 0.63), by using a strong, weak and generic lexicon respectively. Thanks to joint training, our method can also serve as a good detector by achieving a new state-of-the-art detection performance on related benchmarks. Code is available at https://github. com/tonghe90/textspotter.

count=3
* Perceive Where to Focus: Learning Visibility-Aware Part-Level Features for Partial Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Perceive_Where_to_Focus_Learning_Visibility-Aware_Part-Level_Features_for_Partial_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Perceive_Where_to_Focus_Learning_Visibility-Aware_Part-Level_Features_for_Partial_CVPR_2019_paper.pdf)]
    * Title: Perceive Where to Focus: Learning Visibility-Aware Part-Level Features for Partial Person Re-Identification
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Yifan Sun,  Qin Xu,  Yali Li,  Chi Zhang,  Yikang Li,  Shengjin Wang,  Jian Sun
    * Abstract: This paper considers a realistic problem in person re-identification (re-ID) task, i.e., partial re-ID. Under partial re-ID scenario, the images may contain a partial observation of a pedestrian. If we directly compare a partial pedestrian image with a holistic one, the extreme spatial misalignment significantly compromises the discriminative ability of the learned representation. We propose a Visibility-aware Part Model (VPM) for partial re-ID, which learns to perceive the visibility of regions through self-supervision. The visibility awareness allows VPM to extract region-level features and compare two images with focus on their shared regions (which are visible on both images). VPM gains two-fold benefit toward higher accuracy for partial re-ID. On the one hand, compared with learning a global feature, VPM learns region-level features and thus benefits from fine-grained information. On the other hand, with visibility awareness, VPM is capable to estimate the shared regions between two images and thus suppresses the spatial misalignment. Experimental results confirm that our method significantly improves the learned feature representation and the achieved accuracy is on par with the state of the art.

count=3
* Adversarial Texture Optimization From RGB-D Scans
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_Adversarial_Texture_Optimization_From_RGB-D_Scans_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Adversarial_Texture_Optimization_From_RGB-D_Scans_CVPR_2020_paper.pdf)]
    * Title: Adversarial Texture Optimization From RGB-D Scans
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Jingwei Huang,  Justus Thies,  Angela Dai,  Abhijit Kundu,  Chiyu "Max" Jiang,  Leonidas J. Guibas,  Matthias Niessner,  Thomas Funkhouser
    * Abstract: Realistic color texture generation is an important step in RGB-D surface reconstruction, but remains challenging in practice due to inaccuracies in reconstructed geometry, misaligned camera poses, and view-dependent imaging artifacts. In this work, we present a novel approach for color texture generation using a conditional adversarial loss obtained from weakly-supervised views. Specifically, we propose an approach to produce photorealistic textures for approximate surfaces, even from misaligned images, by learning an objective function that is robust to these errors. The key idea of our approach is to learn a patch-based conditional discriminator which guides the texture optimization to be tolerant to misalignments. Our discriminator takes a synthesized view and a real image, and evaluates whether the synthesized one is realistic, under a broadened definition of realism. We train the discriminator by providing as `real' examples pairs of input views and their misaligned versions -- so that the learned adversarial loss will tolerate errors from the scans. Experiments on synthetic and real data under quantitative or qualitative evaluation demonstrate the advantage of our approach in comparison to state of the art.

count=3
* Robust Partial Matching for Person Search in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhong_Robust_Partial_Matching_for_Person_Search_in_the_Wild_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhong_Robust_Partial_Matching_for_Person_Search_in_the_Wild_CVPR_2020_paper.pdf)]
    * Title: Robust Partial Matching for Person Search in the Wild
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Yingji Zhong,  Xiaoyu Wang,  Shiliang Zhang
    * Abstract: Various factors like occlusions, backgrounds, etc., would lead to misaligned detected bounding boxes , e.g., ones covering only portions of human body. This issue is common but overlooked by previous person search works. To alleviate this issue, this paper proposes an Align-to-Part Network (APNet) for person detection and re-Identification (reID). APNet refines detected bounding boxes to cover the estimated holistic body regions, from which discriminative part features can be extracted and aligned. Aligned part features naturally formulate reID as a partial feature matching procedure, where valid part features are selected for similarity computation, while part features on occluded or noisy regions are discarded. This design enhances the robustness of person search to real-world challenges with marginal computation overhead. This paper also contributes a Large-Scale dataset for Person Search in the wild (LSPS), which is by far the largest and the most challenging dataset for person search. Experiments show that APNet brings considerable performance improvement on LSPS. Meanwhile, it achieves competitive performance on existing person search benchmarks like CUHK-SYSU and PRW.

count=3
* Spatially-Invariant Style-Codes Controlled Makeup Transfer
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_Spatially-Invariant_Style-Codes_Controlled_Makeup_Transfer_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Spatially-Invariant_Style-Codes_Controlled_Makeup_Transfer_CVPR_2021_paper.pdf)]
    * Title: Spatially-Invariant Style-Codes Controlled Makeup Transfer
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Han Deng, Chu Han, Hongmin Cai, Guoqiang Han, Shengfeng He
    * Abstract: Transferring makeup from the misaligned reference image is challenging. Previous methods overcome this barrier by computing pixel-wise correspondences between two images, which is inaccurate and computational-expensive. In this paper, we take a different perspective to break down the makeup transfer problem into a two-step extraction-assignment process. To this end, we propose a Style-based Controllable GAN model that consists of three components, each of which corresponds to target style-code encoding, face identity features extraction, and makeup fusion, respectively. In particular, a Part-specific Style Encoder encodes the component-wise makeup style of the reference image into a style-code in an intermediate latent space W. The style-code discards spatial information and therefore is invariant to spatial misalignment. On the other hand, the style-code embeds component-wise information, enabling flexible partial makeup editing from multiple references. This style-code, together with source identity features, are integrated to a Makeup Fusion Decoder equipped with multiple AdaIN layers to generate the final result. Our proposed method demonstrates great flexibility on makeup transfer by supporting makeup removal, shade-controllable makeup transfer, and part-specific makeup transfer, even with large spatial misalignment. Extensive experiments demonstrate the superiority of our approach over state-of-the-art methods. Code is available at https://github.com/makeuptransfer/SCGAN.

count=3
* SIPSA-Net: Shift-Invariant Pan Sharpening With Moving Object Alignment for Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_SIPSA-Net_Shift-Invariant_Pan_Sharpening_With_Moving_Object_Alignment_for_Satellite_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_SIPSA-Net_Shift-Invariant_Pan_Sharpening_With_Moving_Object_Alignment_for_Satellite_CVPR_2021_paper.pdf)]
    * Title: SIPSA-Net: Shift-Invariant Pan Sharpening With Moving Object Alignment for Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Jaehyup Lee, Soomin Seo, Munchurl Kim
    * Abstract: Pan-sharpening is a process of merging a high-resolution (HR) panchromatic (PAN) image and its corresponding low-resolution (LR) multi-spectral (MS) image to create an HR-MS and pan-sharpened image. However, due to the different sensors' locations, characteristics and acquisition time, PAN and MS image pairs often tend to have various amounts of misalignment. Conventional deep-learning-based methods that were trained with such misaligned PAN-MS image pairs suffer from diverse artifacts such as double-edge and blur artifacts in the resultant PAN-sharpened images. In this paper, we propose a novel framework called shift-invariant pan-sharpening with moving object alignment (SIPSA-Net) which is the first method to take into account such large misalignment of moving object regions for PAN sharpening. The SISPA-Net has a feature alignment module (FAM) that can adjust one feature to be aligned to another feature, even between the two different PAN and MS domains. For better alignment in pan-sharpened images, a shift-invariant spectral loss is newly designed, which ignores the inherent misalignment in the original MS input, thereby having the same effect as optimizing the spectral loss with a well-aligned MS image. Extensive experimental results show that our SIPSA-Net can generate pan-sharpened images with remarkable improvements in terms of visual quality and alignment, compared to the state-of-the-art methods.

count=3
* NTIRE 2021 Challenge on Burst Super-Resolution: Methods and Results
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Bhat_NTIRE_2021_Challenge_on_Burst_Super-Resolution_Methods_and_Results_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Bhat_NTIRE_2021_Challenge_on_Burst_Super-Resolution_Methods_and_Results_CVPRW_2021_paper.pdf)]
    * Title: NTIRE 2021 Challenge on Burst Super-Resolution: Methods and Results
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Goutam Bhat, Martin Danelljan, Radu Timofte
    * Abstract: This paper reviews the NTIRE2021 challenge on burst super-resolution. Given a RAW noisy burst as input, the task in the challenge was to generate a clean RGB image with 4 times higher resolution. The challenge contained two tracks; Track 1 evaluating on synthetically generated data, and Track 2 using real-world bursts from mobile camera. In the final testing phase, 6 teams submitted results using a diverse set of solutions. The top-performing methods set a new state-of-the-art for the burst super-resolution task.

count=3
* Federated Learning With Position-Aware Neurons
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Federated_Learning_With_Position-Aware_Neurons_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Federated_Learning_With_Position-Aware_Neurons_CVPR_2022_paper.pdf)]
    * Title: Federated Learning With Position-Aware Neurons
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Xin-Chun Li, Yi-Chu Xu, Shaoming Song, Bingshuai Li, Yinchuan Li, Yunfeng Shao, De-Chuan Zhan
    * Abstract: Federated Learning (FL) fuses collaborative models from local nodes without centralizing users' data. The permutation invariance property of neural networks and the non-i.i.d. data across clients make the locally updated parameters imprecisely aligned, disabling the coordinate-based parameter averaging. Traditional neurons do not explicitly consider position information. Hence, we propose Position-Aware Neurons (PANs) as an alternative, fusing position-related values (i.e., position encodings) into neuron outputs. PANs couple themselves to their positions and minimize the possibility of dislocation, even updating on heterogeneous data. We turn on/off PANs to disable/enable the permutation invariance property of neural networks. PANs are tightly coupled with positions when applied to FL, making parameters across clients pre-aligned and facilitating coordinate-based parameter averaging. PANs are algorithm-agnostic and could universally improve existing FL algorithms. Furthermore, "FL with PANs" is simple to implement and computationally friendly.

count=3
* AligNeRF: High-Fidelity Neural Radiance Fields via Alignment-Aware Training
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Jiang_AligNeRF_High-Fidelity_Neural_Radiance_Fields_via_Alignment-Aware_Training_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_AligNeRF_High-Fidelity_Neural_Radiance_Fields_via_Alignment-Aware_Training_CVPR_2023_paper.pdf)]
    * Title: AligNeRF: High-Fidelity Neural Radiance Fields via Alignment-Aware Training
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yifan Jiang, Peter Hedman, Ben Mildenhall, Dejia Xu, Jonathan T. Barron, Zhangyang Wang, Tianfan Xue
    * Abstract: Neural Radiance Fields (NeRFs) are a powerful representation for modeling a 3D scene as a continuous function. Though NeRF is able to render complex 3D scenes with view-dependent effects, few efforts have been devoted to exploring its limits in a high-resolution setting. Specifically, existing NeRF-based methods face several limitations when reconstructing high-resolution real scenes, including a very large number of parameters, misaligned input data, and overly smooth details. In this work, we conduct the first pilot study on training NeRF with high-resolution data and propose the corresponding solutions: 1) marrying the multilayer perceptron (MLP) with convolutional layers which can encode more neighborhood information while reducing the total number of parameters; 2) a novel training strategy to address misalignment caused by moving objects or small camera calibration errors; and 3) a high-frequency aware loss. Our approach is nearly free without introducing obvious training/testing costs, while experiments on different datasets demonstrate that it can recover more high-frequency details compared with the current state-of-the-art NeRF models. Project page: https://yifanjiang19.github.io/alignerf.

count=3
* Misalignment-Robust Frequency Distribution Loss for Image Transformation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ni_Misalignment-Robust_Frequency_Distribution_Loss_for_Image_Transformation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ni_Misalignment-Robust_Frequency_Distribution_Loss_for_Image_Transformation_CVPR_2024_paper.pdf)]
    * Title: Misalignment-Robust Frequency Distribution Loss for Image Transformation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zhangkai Ni, Juncheng Wu, Zian Wang, Wenhan Yang, Hanli Wang, Lin Ma
    * Abstract: This paper aims to address a common challenge in deep learning-based image transformation methods such as image enhancement and super-resolution which heavily rely on precisely aligned paired datasets with pixel-level alignments. However creating precisely aligned paired images presents significant challenges and hinders the advancement of methods trained on such data. To overcome this challenge this paper introduces a novel and simple Frequency Distribution Loss (FDL) for computing distribution distance within the frequency domain. Specifically we transform image features into the frequency domain using Discrete Fourier Transformation (DFT). Subsequently frequency components (amplitude and phase) are processed separately to form the FDL loss function. Our method is empirically proven effective as a training constraint due to the thoughtful utilization of global information in the frequency domain. Extensive experimental evaluations focusing on image enhancement and super-resolution tasks demonstrate that FDL outperforms existing misalignment-robust loss functions. Furthermore we explore the potential of our FDL for image style transfer that relies solely on completely misaligned data. Our code is available at: https://github.com/eezkni/FDL

count=3
* Revisiting Single Image Reflection Removal In the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Revisiting_Single_Image_Reflection_Removal_In_the_Wild_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Revisiting_Single_Image_Reflection_Removal_In_the_Wild_CVPR_2024_paper.pdf)]
    * Title: Revisiting Single Image Reflection Removal In the Wild
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yurui Zhu, Xueyang Fu, Peng-Tao Jiang, Hao Zhang, Qibin Sun, Jinwei Chen, Zheng-Jun Zha, Bo Li
    * Abstract: This research focuses on the issue of single-image reflection removal (SIRR) in real-world conditions examining it from two angles: the collection pipeline of real reflection pairs and the perception of real reflection locations. We devise an advanced reflection collection pipeline that is highly adaptable to a wide range of real-world reflection scenarios and incurs reduced costs in collecting large-scale aligned reflection pairs. In the process we develop a large-scale high-quality reflection dataset named Reflection Removal in the Wild (RRW). RRW contains over 14950 high-resolution real-world reflection pairs a dataset forty-five times larger than its predecessors. Regarding perception of reflection locations we identify that numerous virtual reflection objects visible in reflection images are not present in the corresponding ground-truth images. This observation drawn from the aligned pairs leads us to conceive the Maximum Reflection Filter (MaxRF). The MaxRF could accurately and explicitly characterize reflection locations from pairs of images. Building upon this we design a reflection location-aware cascaded framework specifically tailored for SIRR. Powered by these innovative techniques our solution achieves superior performance than current leading methods across multiple real-world benchmarks. Codes and datasets are available at \href https://github.com/zhuyr97/Reflection_RemoVal_CVPR2024 \color blue here .

count=3
* Deep Spatial-Semantic Attention for Fine-Grained Sketch-Based Image Retrieval
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Song_Deep_Spatial-Semantic_Attention_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Song_Deep_Spatial-Semantic_Attention_ICCV_2017_paper.pdf)]
    * Title: Deep Spatial-Semantic Attention for Fine-Grained Sketch-Based Image Retrieval
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Jifei Song, Qian Yu, Yi-Zhe Song, Tao Xiang, Timothy M. Hospedales
    * Abstract: Human sketches are unique in being able to capture both the spatial topology of a visual object, as well as its subtle appearance details. Fine-grained sketch-based image retrieval (FG-SBIR) importantly leverages on such fine-grained characteristics of sketches to conduct instance-level retrieval of photos. Nevertheless, human sketches are often highly abstract and iconic, resulting in severe misalignments with candidate photos which in turn make subtle visual detail matching difficult. Existing FG-SBIR approaches focus only on coarse holistic matching via deep cross-domain representation learning, yet ignore explicitly accounting for fine-grained details and their spatial context. In this paper, a novel deep FG-SBIR model is proposed which differs significantly from the existing models in that: (1) It is spatially aware, achieved by introducing an attention module that is sensitive to the spatial position of visual details; (2) It combines coarse and fine semantic information via a shortcut connection fusion block; and (3) It models feature correlation and is robust to misalignments between the extracted features across the two domains by introducing a novel higher order learnable energy function (HOLEF) based loss. Extensive experiments show that the proposed deep spatial-semantic attention model significantly outperforms the state-of-the-art.

count=3
* Bilinear Attention Networks for Person Retrieval
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Fang_Bilinear_Attention_Networks_for_Person_Retrieval_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fang_Bilinear_Attention_Networks_for_Person_Retrieval_ICCV_2019_paper.pdf)]
    * Title: Bilinear Attention Networks for Person Retrieval
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Pengfei Fang,  Jieming Zhou,  Soumava Kumar Roy,  Lars Petersson,  Mehrtash Harandi
    * Abstract: This paper investigates a novel Bilinear attention (Bi-attention) block, which discovers and uses second order statistical information in an input feature map, for the purpose of person retrieval. The Bi-attention block uses bilinear pooling to model the local pairwise feature interactions along each channel, while preserving the spatial structural information. We propose an Attention in Attention (AiA) mechanism to build inter-dependency among the second order local and global features with the intent to make better use of, or pay more attention to, such higher order statistical relationships. The proposed network, equipped with the proposed Bi-attention is referred to as Bilinear ATtention network (BAT-net). Our approach outperforms current state-of-the-art by a considerable margin across the standard benchmark datasets (e.g., CUHK03, Market-1501, DukeMTMC-reID and MSMT17).

count=3
* Deep Reparametrization of Multi-Frame Super-Resolution and Denoising
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Bhat_Deep_Reparametrization_of_Multi-Frame_Super-Resolution_and_Denoising_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Bhat_Deep_Reparametrization_of_Multi-Frame_Super-Resolution_and_Denoising_ICCV_2021_paper.pdf)]
    * Title: Deep Reparametrization of Multi-Frame Super-Resolution and Denoising
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Goutam Bhat, Martin Danelljan, Fisher Yu, Luc Van Gool, Radu Timofte
    * Abstract: We propose a deep reparametrization of the maximum a posteriori formulation commonly employed in multi-frame image restoration tasks. Our approach is derived by introducing a learned error metric and a latent representation of the target image, which transforms the MAP objective to a deep feature space. The deep reparametrization allows us to directly model the image formation process in the latent space, and to integrate learned image priors into the prediction. Our approach thereby leverages the advantages of deep learning, while also benefiting from the principled multi-frame fusion provided by the classical MAP formulation. We validate our approach through comprehensive experiments on burst denoising and burst super-resolution datasets. Our approach sets a new state-of-the-art for both tasks, demonstrating the generality and effectiveness of the proposed formulation.

count=2
* Fusing Robust Face Region Descriptors via Multiple Metric Learning for Face Recognition in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Cui_Fusing_Robust_Face_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Cui_Fusing_Robust_Face_2013_CVPR_paper.pdf)]
    * Title: Fusing Robust Face Region Descriptors via Multiple Metric Learning for Face Recognition in the Wild
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Zhen Cui, Wen Li, Dong Xu, Shiguang Shan, Xilin Chen
    * Abstract: In many real-world face recognition scenarios, face images can hardly be aligned accurately due to complex appearance variations or low-quality images. To address this issue, we propose a new approach to extract robust face region descriptors. Specifically, we divide each image (resp. video) into several spatial blocks (resp. spatial-temporal volumes) and then represent each block (resp. volume) by sum-pooling the nonnegative sparse codes of position-free patches sampled within the block (resp. volume). Whitened Principal Component Analysis (WPCA) is further utilized to reduce the feature dimension, which leads to our Spatial Face Region Descriptor (SFRD) (resp. Spatial-Temporal Face Region Descriptor, STFRD) for images (resp. videos). Moreover, we develop a new distance metric learning method for face verification called Pairwise-constrained Multiple Metric Learning (PMML) to effectively integrate the face region descriptors of all blocks (resp. volumes) from an image (resp. a video). Our work achieves the stateof-the-art performances on two real-world datasets LFW and YouTube Faces (YTF) according to the restricted protocol.

count=2
* Principal Observation Ray Calibration for Tiled-Lens-Array Integral Imaging Display
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Li_Principal_Observation_Ray_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Li_Principal_Observation_Ray_2013_CVPR_paper.pdf)]
    * Title: Principal Observation Ray Calibration for Tiled-Lens-Array Integral Imaging Display
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Weiming Li, Haitao Wang, Mingcai Zhou, Shandong Wang, Shaohui Jiao, Xing Mei, Tao Hong, Hoyoung Lee, Jiyeun Kim
    * Abstract: Integral imaging display (IID) is a promising technology to provide realistic 3D image without glasses. To achieve a large screen IID with a reasonable fabrication cost, a potential solution is a tiled-lens-array IID (TLA-IID). However, TLA-IIDs are subject to 3D image artifacts when there are even slight misalignments between the lens arrays. This work aims at compensating these artifacts by calibrating the lens array poses with a camera and including them in a ray model used for rendering the 3D image. Since the lens arrays are transparent, this task is challenging for traditional calibration methods. In this paper, we propose a novel calibration method based on defining a set of principle observation rays that pass lens centers of the TLA and the camera's optical center. The method is able to determine the lens array poses with only one camera at an arbitrary unknown position without using any additional markers. The principle observation rays are automatically extracted using a structured light based method from a dense correspondence map between the displayed and captured pixels. Experiments show that lens array misalignments can be estimated with a standard deviation smaller than 0.4 pixels. Based on this, 3D image artifacts are shown to be effectively removed in a test TLA-IID with challenging misalignments.

count=2
* Improving an Object Detector and Extracting Regions Using Superpixels
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Shu_Improving_an_Object_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Shu_Improving_an_Object_2013_CVPR_paper.pdf)]
    * Title: Improving an Object Detector and Extracting Regions Using Superpixels
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Guang Shu, Afshin Dehghan, Mubarak Shah
    * Abstract: We propose an approach to improve the detection performance of a generic detector when it is applied to a particular video. The performance of offline-trained objects detectors are usually degraded in unconstrained video environments due to variant illuminations, backgrounds and camera viewpoints. Moreover, most object detectors are trained using Haar-like features or gradient features but ignore video specific features like consistent color patterns. In our approach, we apply a Superpixel-based Bag-of-Words (BoW) model to iteratively refine the output of a generic detector. Compared to other related work, our method builds a video-specific detector using superpixels, hence it can handle the problem of appearance variation. Most importantly, using Conditional Random Field (CRF) along with our super pixel-based BoW model, we develop and algorithm to segment the object from the background . Therefore our method generates an output of the exact object regions instead of the bounding boxes generated by most detectors. In general, our method takes detection bounding boxes of a generic detector as input and generates the detection output with higher average precision and precise object regions. The experiments on four recent datasets demonstrate the effectiveness of our approach and significantly improves the state-of-art detector by 5-16% in average precision.

count=2
* As-Projective-As-Possible Image Stitching with Moving DLT
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Zaragoza_As-Projective-As-Possible_Image_Stitching_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Zaragoza_As-Projective-As-Possible_Image_Stitching_2013_CVPR_paper.pdf)]
    * Title: As-Projective-As-Possible Image Stitching with Moving DLT
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Julio Zaragoza, Tat-Jun Chin, Michael S. Brown, David Suter
    * Abstract: We investigate projective estimation under model inadequacies, i.e., when the underpinning assumptions of the projective model are not fully satisfied by the data. We focus on the task of image stitching which is customarily solved by estimating a projective warp -a model that is justified when the scene is planar or when the views differ purely by rotation. Such conditions are easily violated in practice, and this yields stitching results with ghosting artefacts that necessitate the usage of deghosting algorithms. To this end we propose as-projective-as-possible warps, i.e., warps that aim to be globally projective, yet allow local non-projective deviations to account for violations to the assumed imaging conditions. Based on a novel estimation technique called Moving Direct Linear Transformation (Moving DLT), our method seamlessly bridges image regions that are inconsistent with the projective model. The result is highly accurate image stitching, with significantly reduced ghosting effects, thus lowering the dependency on post hoc deghosting.

count=2
* DeepReID: Deep Filter Pairing Neural Network for Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Li_DeepReID_Deep_Filter_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf)]
    * Title: DeepReID: Deep Filter Pairing Neural Network for Person Re-Identification
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Wei Li, Rui Zhao, Tong Xiao, Xiaogang Wang
    * Abstract: Person re-identification is to match pedestrian images from disjoint camera views detected by pedestrian detectors. Challenges are presented in the form of complex variations of lightings, poses, viewpoints, blurring effects, image resolutions, camera settings, occlusions and background clutter across camera views. In addition, misalignment introduced by the pedestrian detector will affect most existing person re-identification methods that use manually cropped pedestrian images and assume perfect detection. In this paper, we propose a novel filter pairing neural network (FPNN) to jointly handle misalignment, photometric and geometric transforms, occlusions and background clutter. All the key components are jointly optimized to maximize the strength of each component when cooperating with others. In contrast to existing works that use handcrafted features, our method automatically learns features optimal for the re-identification task from data. The learned filter pairs encode photometric transforms. Its deep architecture makes it possible to model a mixture of complex photometric and geometric transforms. We build the largest benchmark re-id dataset with 13,164 images of 1,360 pedestrians. Unlike existing datasets, which only provide manually cropped pedestrian images, our dataset provides automatically detected bounding boxes for evaluation close to practical applications. Our neural network significantly outperforms state-of-the-art methods on this dataset.

count=2
* Robust Manhattan Frame Estimation From a Single RGB-D Image
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Ghanem_Robust_Manhattan_Frame_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Ghanem_Robust_Manhattan_Frame_2015_CVPR_paper.pdf)]
    * Title: Robust Manhattan Frame Estimation From a Single RGB-D Image
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Bernard Ghanem, Ali Thabet, Juan Carlos Niebles, Fabian Caba Heilbron
    * Abstract: This paper proposes a new framework for estimating the Manhattan Frame (MF) of an indoor scene from a single RGB-D image. Our technique formulates this problem as the estimation of a rotation matrix that best aligns the normals of the captured scene to a canonical world axes. By introducing sparsity constraints, our method can simultaneously estimate the scene MF, the surfaces in the scene that are best aligned to one of three coordinate axes, and the outlier surfaces that do not align with any of the axes. To test our approach, we contribute a new set of annotations to determine ground truth MFs in each image of the popular NYUv2 dataset. We use this new benchmark to experimentally demonstrate that our method is more accurate, faster, more reliable and more robust than the methods used in the literature. We further motivate our technique by showing how it can be used to address the RGB-D SLAM problem in indoor scenes by incorporating it into and improving the performance of a popular RGB-D SLAM method.

count=2
* Large-Pose Face Alignment via CNN-Based Dense 3D Model Fitting
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Jourabloo_Large-Pose_Face_Alignment_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Jourabloo_Large-Pose_Face_Alignment_CVPR_2016_paper.pdf)]
    * Title: Large-Pose Face Alignment via CNN-Based Dense 3D Model Fitting
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Amin Jourabloo, Xiaoming Liu
    * Abstract: Large-pose face alignment is a very challenging problem in computer vision, which is used as a prerequisite for many important vision tasks, e.g, face recognition and 3D face reconstruction. Recently, there have been a few attempts to solve this problem, but still more research is needed to achieve highly accurate results. In this paper, we propose a face alignment method for large-pose face images, by combining the powerful cascaded CNN regressor method and 3DMM. We formulate the face alignment as a 3DMM fitting problem, where the camera projection matrix and 3D shape parameters are estimated by a cascade of CNN-based regressors. The dense 3D shape allows us to design pose-invariant appearance features for effective CNN learning. Extensive experiments are conducted on the challenging databases (AFLW and AFW), with comparison to the state of the art.

count=2
* Hierarchical Gaussian Descriptor for Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Matsukawa_Hierarchical_Gaussian_Descriptor_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Matsukawa_Hierarchical_Gaussian_Descriptor_CVPR_2016_paper.pdf)]
    * Title: Hierarchical Gaussian Descriptor for Person Re-Identification
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Tetsu Matsukawa, Takahiro Okabe, Einoshin Suzuki, Yoichi Sato
    * Abstract: Describing the color and textural information of a person image is one of the most crucial aspects of person re-identification. In this paper, we present a novel descriptor based on a hierarchical distribution of pixel features. A hierarchical covariance descriptor has been successfully applied for image classification. However, the mean information of pixel features, which is absent in covariance, tends to be major discriminative information of person images. To solve this problem, we describe a local region in an image via hierarchical Gaussian distribution in which both means and covariances are included in their parameters. More specifically, we model the region as a set of multiple Gaussian distributions in which each Gaussian represents the appearance of a local patch. The characteristics of the set of Gaussians are again described by another Gaussian distribution. In both steps, unlike the hierarchical covariance descriptor, the proposed descriptor can model both the mean and the covariance information of pixel features properly. The results of experiments conducted on five databases indicate that the proposed descriptor exhibits remarkably high performance which outperforms the state-of-the-art descriptors for person re-identification.

count=2
* Sketch Me That Shoe
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Yu_Sketch_Me_That_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yu_Sketch_Me_That_CVPR_2016_paper.pdf)]
    * Title: Sketch Me That Shoe
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Qian Yu, Feng Liu, Yi-Zhe Song, Tao Xiang, Timothy M. Hospedales, Chen-Change Loy
    * Abstract: We investigate the problem of fine-grained sketch-based image retrieval (SBIR), where free-hand human sketches are used as queries to perform instance-level retrieval of images. This is an extremely challenging task because (i) visual comparisons not only need to be fine-grained but also executed cross-domain, (ii) free-hand (finger) sketches are highly abstract, making fine-grained matching harder, and most importantly (iii) annotated cross-domain sketch-photo datasets required for training are scarce, challenging many state-of-the-art machine learning techniques. In this paper, for the first time, we address all these challenges, providing a step towards the capabilities that would underpin a commercial sketch-based image retrieval application. We introduce a new database of 1,432 sketch-photo pairs from two categories with 32,000 fine-grained triplet ranking annotations. We then develop a deep triplet-ranking model for instance-level SBIR with a novel data augmentation and staged pre-training strategy to alleviate the issue of insufficient fine-grained training data. Extensive experiments are carried out to contribute a variety of insights into the challenges of data sufficiency and over-fitting avoidance when training deep networks for fine-grained cross-domain ranking tasks.

count=2
* Avoiding the Deconvolution: Framework Oriented Color Transfer for Enhancing Low-Light Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w16/html/Florea_Avoiding_the_Deconvolution_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w16/papers/Florea_Avoiding_the_Deconvolution_CVPR_2016_paper.pdf)]
    * Title: Avoiding the Deconvolution: Framework Oriented Color Transfer for Enhancing Low-Light Images
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Laura Florea, Corneliu Florea, Ciprian Ionascu
    * Abstract: In this paper we introduce a novel color transfer method to address the underexposed image amplification problem. Targeted scenario implies a dual acquisition, containing a normally exposed, possibly blurred, image and an underexposed/low-light but sharp one. The problem of enhancing the low-light image is addressed as a color transfer problem. To properly solve the color transfer, the scene is split into perceptual frameworks and we propose a novel piece-wise approximation. The proposed method is shown to lead to robust results from both an objective and a subjective point of view.

count=2
* A Comprehensive Analysis of Deep Learning Based Representation for Face Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w4/html/Ghazi_A_Comprehensive_Analysis_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w4/papers/Ghazi_A_Comprehensive_Analysis_CVPR_2016_paper.pdf)]
    * Title: A Comprehensive Analysis of Deep Learning Based Representation for Face Recognition
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Mostafa Mehdipour Ghazi, Hazim Kemal Ekenel
    * Abstract: Deep learning based approaches have been dominating the face recognition field due to the significant performance improvement they have provided on the challenging wild datasets. These approaches have been extensively tested on such unconstrained datasets, on the Labeled Faces in the Wild and YouTube Faces, to name a few. However, their capability to handle individual appearance variations caused by factors such as head pose, illumination, occlusion, and misalignment has not been thoroughly assessed till now. In this paper, we present a comprehensive study to evaluate the performance of deep learning based face representation under several conditions including the varying head pose angles, upper and lower face occlusion, changing illumination of different strengths, and misalignment due to erroneous facial feature localization. Two successful and publicly available deep learning models, namely VGG-Face and Lightened CNN have been utilized to extract face representations. The obtained results show that although deep learning provides a powerful representation for face recognition, it can still benefit from preprocessing, for example, for pose and illumination normalization in order to achieve better performance under various conditions. Particularly, if these variations are not included in the dataset used to train the deep learning model, the role of preprocessing becomes more crucial. Experimental results also show that deep learning based representation is robust to misalignment and can tolerate facial feature localization errors up to 10% of the interocular distance.

count=2
* Hallucinating Very Low-Resolution Unaligned and Noisy Face Images by Transformative Discriminative Autoencoders
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Yu_Hallucinating_Very_Low-Resolution_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Hallucinating_Very_Low-Resolution_CVPR_2017_paper.pdf)]
    * Title: Hallucinating Very Low-Resolution Unaligned and Noisy Face Images by Transformative Discriminative Autoencoders
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Xin Yu, Fatih Porikli
    * Abstract: Most of the conventional face hallucination methods assume the input image is sufficiently large and aligned, and all require the input image to be noise-free. Their performance degrades drastically if the input image is tiny, unaligned, and contaminated by noise. In this paper, we introduce a novel transformative discriminative autoencoder to 8X super-resolve unaligned noisy and tiny (16X16) low-resolution face images. In contrast to encoder-decoder based autoencoders, our method uses decoder-encoder-decoder networks. We first employ a transformative discriminative decoder network to upsample and denoise simultaneously. Then we use a transformative encoder network to project the intermediate HR faces to aligned and noise-free LR faces. Finally, we use the second decoder to generate hallucinated HR images. Our extensive evaluations on a very large face dataset show that our method achieves superior hallucination results and outperforms the state-of-the-art by a large margin of 1.82dB PSNR.

count=2
* Texture Mapping for 3D Reconstruction With RGB-D Sensor
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Fu_Texture_Mapping_for_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Fu_Texture_Mapping_for_CVPR_2018_paper.pdf)]
    * Title: Texture Mapping for 3D Reconstruction With RGB-D Sensor
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Yanping Fu, Qingan Yan, Long Yang, Jie Liao, Chunxia Xiao
    * Abstract: Acquiring realistic texture details for 3D models is important in 3D reconstruction. However, the existence of geometric errors, caused by noisy RGB-D sensor data, always makes the color images cannot be accurately aligned onto reconstructed 3D models. In this paper, we propose a global-to-local correction strategy to obtain more desired texture mapping results. Our algorithm first adaptively selects an optimal image for each face of the 3D model, which can effectively remove blurring and ghost artifacts produced by multiple image blending. We then adopt a non-rigid global-to-local correction step to reduce the seaming effect between textures. This can effectively compensate for the texture and the geometric misalignment caused by camera pose drift and geometric errors. We evaluate the proposed algorithm in a range of complex scenes and demonstrate its effective performance in generating seamless high fidelity textures for 3D models.

count=2
* Selective Sensor Fusion for Neural Visual-Inertial Odometry
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.pdf)]
    * Title: Selective Sensor Fusion for Neural Visual-Inertial Odometry
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Changhao Chen,  Stefano Rosa,  Yishu Miao,  Chris Xiaoxuan Lu,  Wei Wu,  Andrew Markham,  Niki Trigoni
    * Abstract: Deep learning approaches for Visual-Inertial Odometry (VIO) have proven successful, but they rarely focus on incorporating robust fusion strategies for dealing with imperfect input sensory data. We propose a novel end-to-end selective sensor fusion framework for monocular VIO, which fuses monocular images and inertial measurements in order to estimate the trajectory whilst improving robustness to real-life issues, such as missing and corrupted data or bad sensor synchronization. In particular, we propose two fusion modalities based on different masking strategies: deterministic soft fusion and stochastic hard fusion, and we compare with previously proposed direct fusion baselines. During testing, the network is able to selectively process the features of the available sensor modalities and produce a trajectory at scale. We present a thorough investigation on the performances on three public autonomous driving, Micro Aerial Vehicle (MAV) and hand-held VIO datasets. The results demonstrate the effectiveness of the fusion strategies, which offer better performances compared to direct fusion, particularly in presence of corrupted data. In addition, we study the interpretability of the fusion networks by visualising the masking layers in different scenarios and with varying data corruption, revealing interesting correlations between the fusion networks and imperfect sensory input data.

count=2
* DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Ding_DeepMapping_Unsupervised_Map_Estimation_From_Multiple_Point_Clouds_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_DeepMapping_Unsupervised_Map_Estimation_From_Multiple_Point_Clouds_CVPR_2019_paper.pdf)]
    * Title: DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Li Ding,  Chen Feng
    * Abstract: We propose DeepMapping, a novel registration framework using deep neural networks (DNNs) as auxiliary functions to align multiple point clouds from scratch to a globally consistent frame. We use DNNs to model the highly non-convex mapping process that traditionally involves hand-crafted data association, sensor pose initialization, and global refinement. Our key novelty is that "training" these DNNs with properly defined unsupervised losses is equivalent to solving the underlying registration problem, but less sensitive to good initialization than ICP. Our framework contains two DNNs: a localization network that estimates the poses for input point clouds, and a map network that models the scene structure by estimating the occupancy status of global coordinates. This allows us to convert the registration problem to a binary occupancy classification, which can be solved efficiently using gradient-based optimization. We further show that DeepMapping can be readily extended to address the problem of Lidar SLAM by imposing geometric constraints between consecutive point clouds. Experiments are conducted on both simulated and real datasets. Qualitative and quantitative comparisons demonstrate that DeepMapping often enables more robust and accurate global registration of multiple point clouds than existing techniques. Our code is available at https://ai4ce.github.io/DeepMapping/.

count=2
* Densely Semantically Aligned Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Densely_Semantically_Aligned_Person_Re-Identification_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Densely_Semantically_Aligned_Person_Re-Identification_CVPR_2019_paper.pdf)]
    * Title: Densely Semantically Aligned Person Re-Identification
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Zhizheng Zhang,  Cuiling Lan,  Wenjun Zeng,  Zhibo Chen
    * Abstract: We propose a densely semantically aligned person re-identification (re-ID) framework. It fundamentally addresses the body misalignment problem caused by pose/viewpoint variations, imperfect person detection, occlusion, etc.. By leveraging the estimation of the dense semantics of a person image, we construct a set of densely semantically aligned part images (DSAP-images), where the same spatial positions have the same semantics across different person images. We design a two-stream network that consists of a main full image stream (MF-Stream) and a densely semantically-aligned guiding stream (DSAG-Stream). The DSAG-Stream, with the DSAP-images as input, acts as a regulator to guide the MF-Stream to learn densely semantically aligned features from the original image. In the inference, the DSAG-Stream is discarded and only the MF-Stream is needed, which makes the inference system computationally efficient and robust. To our best knowledge, we are the first to make use of fine grained semantics for addressing misalignment problems for re-ID. Our method achieves rank-1 accuracy of 78.9% (new protocol) on the CUHK03 dataset, 90.4% on the CUHK01 dataset, and 95.7% on the Market1501 dataset, outperforming state-of-the-art methods.

count=2
* MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_MAN_Moment_Alignment_Network_for_Natural_Language_Moment_Retrieval_via_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_MAN_Moment_Alignment_Network_for_Natural_Language_Moment_Retrieval_via_CVPR_2019_paper.pdf)]
    * Title: MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Da Zhang,  Xiyang Dai,  Xin Wang,  Yuan-Fang Wang,  Larry S. Davis
    * Abstract: This research strives for natural language moment retrieval in long, untrimmed video streams. The problem is not trivial especially when a video contains multiple moments of interests and the language describes complex temporal dependencies, which often happens in real scenarios. We identify two crucial challenges: semantic misalignment and structural misalignment. However, existing approaches treat different moments separately and do not explicitly model complex moment-wise temporal relations. In this paper, we present Moment Alignment Network (MAN), a novel framework that unifies the candidate moment encoding and temporal structural reasoning in a single-shot feed-forward network. MAN naturally assigns candidate moment representations aligned with language semantics over different temporal locations and scales. Most importantly, we propose to explicitly model moment-wise temporal relations as a structured graph and devise an iterative graph adjustment network to jointly learn the best structure in an end-to-end manner. We evaluate the proposed approach on two challenging public benchmarks DiDeMo and Charades-STA, where our MAN significantly outperforms the state-of-the-art by a large margin.

count=2
* RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf)]
    * Title: RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Jiankang Deng,  Jia Guo,  Evangelos Ververas,  Irene Kotsia,  Stefanos Zafeiriou
    * Abstract: Though tremendous strides have been made in uncontrolled face detection, accurate and efficient 2D face alignment and 3D face reconstruction in-the-wild remain an open challenge. In this paper, we present a novel single-shot, multi-level face localisation method, named RetinaFace, which unifies face box prediction, 2D facial landmark localisation and 3D vertices regression under one common target: point regression on the image plane. To fill the data gap, we manually annotated five facial landmarks on the WIDER FACE dataset and employed a semi-automatic annotation pipeline to generate 3D vertices for face images from the WIDER FACE, AFLW and FDDB datasets. Based on extra annotations, we propose a mutually beneficial regression target for 3D face reconstruction, that is predicting 3D vertices projected on the image plane constrained by a common 3D topology. The proposed 3D face reconstruction branch can be easily incorporated, without any optimisation difficulty, in parallel with the existing box and 2D landmark regression branches during joint training. Extensive experimental results show that RetinaFace can simultaneously achieve stable face detection, accurate 2D face alignment and robust 3D face reconstruction while being efficient through single-shot inference.

count=2
* Attack to Explain Deep Representation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Jalwana_Attack_to_Explain_Deep_Representation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jalwana_Attack_to_Explain_Deep_Representation_CVPR_2020_paper.pdf)]
    * Title: Attack to Explain Deep Representation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Mohammad A. A. K. Jalwana,  Naveed Akhtar,  Mohammed Bennamoun,  Ajmal Mian
    * Abstract: Deep visual models are susceptible to extremely low magnitude perturbations to input images. Though carefully crafted, the perturbation patterns generally appear noisy, yet they are able to perform controlled manipulation of model predictions. This observation is used to argue that deep representation is misaligned with human perception. This paper counter-argues and proposes the first attack on deep learning that aims at explaining the learned representation instead of fooling it. By extending the input domain of the manipulative signal and employing a model faithful channelling, we iteratively accumulate adversarial perturbations for a deep model. The accumulated signal gradually manifests itself as a collection of visually salient features of the target label (in model fooling), casting adversarial perturbations as primitive features of the target label. Our attack provides the first demonstration of systematically computing perturbations for adversarially non-robust classifiers that comprise salient visual features of objects. We leverage the model explaining character of our algorithm to perform image generation, inpainting and interactive image manipulation by attacking adversarially robust classifiers. The visually appealing results across these applications demonstrate the utility of our attack (and perturbations in general) beyond model fooling.

count=2
* Warping Residual Based Image Stitching for Large Parallax
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Lee_Warping_Residual_Based_Image_Stitching_for_Large_Parallax_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Warping_Residual_Based_Image_Stitching_for_Large_Parallax_CVPR_2020_paper.pdf)]
    * Title: Warping Residual Based Image Stitching for Large Parallax
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Kyu-Yul Lee,  Jae-Young Sim
    * Abstract: Image stitching techniques align two images captured at different viewing positions onto a single wider image. When the captured 3D scene is not planar and the camera baseline is large, two images exhibit parallax where the relative positions of scene structures are quite different from each view. The existing image stitching methods often fail to work on the images with large parallax. In this paper, we propose an image stitching algorithm robust to large parallax based on the novel concept of warping residuals. We first estimate multiple homographies and find their inlier feature matches between two images. Then we evaluate warping residual for each feature match with respect to the multiple homographies. To alleviate the parallax artifacts, we partition input images into superpixels and warp each superpixel adaptively according to an optimal homography which is computed by minimizing the error of feature matches weighted by the warping residuals. Experimental results demonstrate that the proposed algorithm provides accurate stitching results for images with large parallax, and outperforms the existing methods qualitatively and quantitatively.

count=2
* Robust 3D Self-Portraits in Seconds
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Robust_3D_Self-Portraits_in_Seconds_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Robust_3D_Self-Portraits_in_Seconds_CVPR_2020_paper.pdf)]
    * Title: Robust 3D Self-Portraits in Seconds
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zhe Li,  Tao Yu,  Chuanyu Pan,  Zerong Zheng,  Yebin Liu
    * Abstract: In this paper, we propose an efficient method for robust 3D self-portraits using a single RGBD camera. Benefiting from the proposed PIFusion and lightweight bundle adjustment algorithm, our method can generate detailed 3D self-portraits in seconds and shows the ability to handle subjects wearing extremely loose clothes. To achieve highly efficient and robust reconstruction, we propose PIFusion, which combines learning-based 3D recovery with volumetric non-rigid fusion to generate accurate sparse partial scans of the subject. Moreover, a non-rigid volumetric deformation method is proposed to continuously refine the learned shape prior. Finally, a lightweight bundle adjustment algorithm is proposed to guarantee that all the partial scans can not only "loop" with each other but also remain consistent with the selected live key observations. The results and experiments show that the proposed method achieves more robust and efficient 3D self-portraits compared with state-of-the-art methods.

count=2
* End-to-End Learning of Visual Representations From Uncurated Instructional Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Miech_End-to-End_Learning_of_Visual_Representations_From_Uncurated_Instructional_Videos_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Miech_End-to-End_Learning_of_Visual_Representations_From_Uncurated_Instructional_Videos_CVPR_2020_paper.pdf)]
    * Title: End-to-End Learning of Visual Representations From Uncurated Instructional Videos
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Antoine Miech,  Jean-Baptiste Alayrac,  Lucas Smaira,  Ivan Laptev,  Josef Sivic,  Andrew Zisserman
    * Abstract: Annotating videos is cumbersome, expensive and not scalable. Yet, many strong video models still rely on manually annotated data. With the recent introduction of the HowTo100M dataset, narrated videos now offer the possibility of learning video representations without manual supervision. In this work we propose a new learning approach, MIL-NCE, capable of addressing mis- alignments inherent in narrated videos. With this approach we are able to learn strong video representations from scratch, without the need for any manual annotation. We evaluate our representations on a wide range of four downstream tasks over eight datasets: action recognition (HMDB-51, UCF-101, Kinetics-700), text-to- video retrieval (YouCook2, MSR-VTT), action localization (YouTube-8M Segments, CrossTask) and action segmentation (COIN). Our method outperforms all published self-supervised approaches for these tasks as well as several fully supervised baselines.

count=2
* Sequence-to-Sequence Contrastive Learning for Text Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Aberdam_Sequence-to-Sequence_Contrastive_Learning_for_Text_Recognition_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Aberdam_Sequence-to-Sequence_Contrastive_Learning_for_Text_Recognition_CVPR_2021_paper.pdf)]
    * Title: Sequence-to-Sequence Contrastive Learning for Text Recognition
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Aviad Aberdam, Ron Litman, Shahar Tsiper, Oron Anschel, Ron Slossberg, Shai Mazor, R. Manmatha, Pietro Perona
    * Abstract: We propose a framework for sequence-to-sequence contrastive learning (SeqCLR) of visual representations, which we apply to text recognition. To account for the sequence-to-sequence structure, each feature map is divided into different instances over which the contrastive loss is computed. This operation enables us to contrast in a sub-word level, where from each image we extract several positive pairs and multiple negative examples. To yield effective visual representations for text recognition, we further suggest novel augmentation heuristics, different encoder architectures and custom projection heads. Experiments on handwritten text and on scene text show that when a text decoder is trained on the learned representations, our method outperforms non-sequential contrastive methods. In addition, when the amount of supervision is reduced, SeqCLR significantly improves performance compared with supervised training, and when fine-tuned with 100% of the labels, our method achieves state-of-the-art results on standard handwritten text recognition benchmarks.

count=2
* Collaborative Spatial-Temporal Modeling for Language-Queried Video Actor Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Hui_Collaborative_Spatial-Temporal_Modeling_for_Language-Queried_Video_Actor_Segmentation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Hui_Collaborative_Spatial-Temporal_Modeling_for_Language-Queried_Video_Actor_Segmentation_CVPR_2021_paper.pdf)]
    * Title: Collaborative Spatial-Temporal Modeling for Language-Queried Video Actor Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Tianrui Hui, Shaofei Huang, Si Liu, Zihan Ding, Guanbin Li, Wenguan Wang, Jizhong Han, Fei Wang
    * Abstract: Language-queried video actor segmentation aims to predict the pixel-level mask of the actor which performs the actions described by a natural language query in the target frames. Existing methods adopt 3D CNNs over the video clip as a general encoder to extract a mixed spatio-temporal feature for the target frame. Though 3D convolutions are amenable to recognizing which actor is performing the queried actions, it also inevitably introduces misaligned spatial information from adjacent frames, which confuses features of the target frame and yields inaccurate segmentation. Therefore, we propose a collaborative spatial-temporal encoder-decoder framework which contains a 3D temporal encoder over the video clip to recognize the queried actions, and a 2D spatial encoder over the target frame to accurately segment the queried actors. In the decoder, a Language-Guided Feature Selection (LGFS) module is proposed to flexibly integrate spatial and temporal features from the two encoders. We also propose a Cross-Modal Adaptive Modulation (CMAM) module to dynamically recombine spatial- and temporal-relevant linguistic features for multimodal feature interaction in each stage of the two encoders. Our method achieves new state-of-the-art performance on two popular benchmarks with less computational overhead than previous approaches.

count=2
* Prototype-Guided Saliency Feature Learning for Person Search
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Kim_Prototype-Guided_Saliency_Feature_Learning_for_Person_Search_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_Prototype-Guided_Saliency_Feature_Learning_for_Person_Search_CVPR_2021_paper.pdf)]
    * Title: Prototype-Guided Saliency Feature Learning for Person Search
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Hanjae Kim, Sunghun Joung, Ig-Jae Kim, Kwanghoon Sohn
    * Abstract: Existing person search methods integrate person detection and re-identification (re-ID) module into a unified system. Though promising results have been achieved, the misalignment problem, which commonly occurs in person search, limits the discriminative feature representation for re-ID. To overcome this limitation, we introduce a novel framework to learn the discriminative representation by utilizing prototype in OIM loss. Unlike conventional methods using prototype as a representation of person identity, we utilize it as guidance to allow the attention network to consistently highlight multiple instances across different poses. Moreover, we propose a new prototype update scheme with adaptive momentum to increase the discriminative ability across different instances. Extensive ablation experiments demonstrate that our method can significantly enhance the feature discriminative power, outperforming the state-of-the-art results on two person search benchmarks including CUHK-SYSU and PRW.

count=2
* LOHO: Latent Optimization of Hairstyles via Orthogonalization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Saha_LOHO_Latent_Optimization_of_Hairstyles_via_Orthogonalization_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Saha_LOHO_Latent_Optimization_of_Hairstyles_via_Orthogonalization_CVPR_2021_paper.pdf)]
    * Title: LOHO: Latent Optimization of Hairstyles via Orthogonalization
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Rohit Saha, Brendan Duke, Florian Shkurti, Graham W. Taylor, Parham Aarabi
    * Abstract: Hairstyle transfer is challenging due to hair structure differences in the source and target hair. Therefore, we propose Latent Optimization of Hairstyles via Orthogonalization (LOHO), an optimization-based approach using GAN inversion to infill missing hair structure details in latent space during hairstyle transfer. Our approach decomposes hair into three attributes: perceptual structure, appearance, and style, and includes tailored losses to model each of these attributes independently. Furthermore, we propose two-stage optimization and gradient orthogonalization to enable disentangled latent space optimization of our hair attributes. Using LOHO for latent space manipulation, users can synthesize novel photorealistic images by manipulating hair attributes either individually or jointly, transferring the desired attributes from reference hairstyles. LOHO achieves a superior FID compared with the current state-of-the-art (SOTA) for hairstyle transfer. Additionally, LOHO preserves the subject's identity comparably well according to PSNR and SSIM when compared to SOTA image embedding pipelines.

count=2
* Anchor-Free Person Search
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Anchor-Free_Person_Search_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Anchor-Free_Person_Search_CVPR_2021_paper.pdf)]
    * Title: Anchor-Free Person Search
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yichao Yan, Jinpeng Li, Jie Qin, Song Bai, Shengcai Liao, Li Liu, Fan Zhu, Ling Shao
    * Abstract: Person search aims to simultaneously localize and identify a query person from realistic, uncropped images, which can be regarded as the unified task of pedestrian detection and person re-identification (re-id). Most existing works employ two-stage detectors like Faster-RCNN, yielding encouraging accuracy but with high computational overhead. In this work, we present the Feature-Aligned Person Search Network (AlignPS), the first anchor-free framework to efficiently tackle this challenging task. AlignPS explicitly addresses the major challenges, which we summarize as the misalignment issues in different levels (i.e., scale, region, and task), when accommodating an anchor-free detector for this task. More specifically, we propose an aligned feature aggregation module to generate more discriminative and robust feature embeddings by following a "re-id first" principle. Such a simple design directly improves the baseline anchor-free model on CUHK-SYSU by more than 20% in mAP. Moreover, AlignPS outperforms state-of-the-art two-stage methods, with a higher speed. The code is available at https://github.com/daodaofr/AlignPS.

count=2
* Continual Learning in Cross-Modal Retrieval
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/CLVision/html/Wang_Continual_Learning_in_Cross-Modal_Retrieval_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/CLVision/papers/Wang_Continual_Learning_in_Cross-Modal_Retrieval_CVPRW_2021_paper.pdf)]
    * Title: Continual Learning in Cross-Modal Retrieval
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Kai Wang, Luis Herranz, Joost van de Weijer
    * Abstract: Multimodal representations and continual learning are two areas closely related to human intelligence. The former considers the learning of shared representation spaces where information from different modalities can be compared and integrated (we focus on cross-modal retrieval between language and visual representations). The latter studies how to prevent forgetting a previously learned task when learning a new one. While humans excel in these two aspects, deep neural networks are still quite limited. In this paper, we propose a combination of both problems into a continual cross-modal retrieval setting, where we study how the catastrophic interference caused by new tasks impacts the embedding spaces and their cross-modal alignment required for effective retrieval. We propose a general framework that decouples the training, indexing and querying stages. We also identify and study different factors that may lead to forgetting, and propose tools to alleviate it. We found that the indexing stage pays an important role and that simply avoiding reindexing the database with updated embedding networks can lead to significant gains. We evaluated our methods in two image-text retrieval datasets, obtaining significant gains with respect to the fine tuning baseline.

count=2
* Robust Image-to-Image Color Transfer Using Optimal Inlier Maximization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Oskarsson_Robust_Image-to-Image_Color_Transfer_Using_Optimal_Inlier_Maximization_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Oskarsson_Robust_Image-to-Image_Color_Transfer_Using_Optimal_Inlier_Maximization_CVPRW_2021_paper.pdf)]
    * Title: Robust Image-to-Image Color Transfer Using Optimal Inlier Maximization
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Magnus Oskarsson
    * Abstract: In this paper we target the color transfer estimation problem, when we have pixel-to-pixel correspondences. We present a feature-based method, that robustly fits color transforms to data containing gross outliers. Our solution is based on an optimal inlier maximization algorithm that maximizes the number of inliers in polynomial time. We introduce a simple feature detector and descriptor based on the structure tensor that gives the means for reliable matching of the color distributions in two images. Using combinatorial methods from optimization theory and a number of new minimal solvers, we can enumerate all possible stationary points to the inlier maximization problem. In order for our method to be tractable we use a decoupling of the intensity and color direction for a given RGB-vector. This enables the intensity transformation and the color direction transformation to be handled separately. Our method gives results comparable to state-of-the-art methods in the presence of little outliers, and large improvement for moderate or large amounts of outliers in the data. The proposed method has been tested in a number of imaging applications.

count=2
* Temporal Alignment Networks for Long-Term Video
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Han_Temporal_Alignment_Networks_for_Long-Term_Video_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Han_Temporal_Alignment_Networks_for_Long-Term_Video_CVPR_2022_paper.pdf)]
    * Title: Temporal Alignment Networks for Long-Term Video
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Tengda Han, Weidi Xie, Andrew Zisserman
    * Abstract: The objective of this paper is a temporal alignment network that ingests long term video sequences, and associated text sentences, in order to: (1) determine if a sentence is alignable with the video; and (2) if it is alignable, then determine its alignment. The challenge is to train such networks from large-scale datasets, such as HowTo100M, where the associated text sentences have significant noise, and are only weakly aligned when relevant. Apart from proposing the alignment network, we also make four contributions: (i) we describe a novel co-training method that enables to denoise and train on raw instructional videos without using manual annotation, despite the considerable noise; (ii) to benchmark the alignment performance, we manually curate a 10-hour subset of HowTo100M, totalling 80 videos, with sparse temporal descriptions. Our proposed model, trained on HowTo100M, outperforms strong baselines (CLIP, MIL-NCE) on this alignment dataset by a significant margin; (iii) we apply the trained model in the zero-shot settings to multiple downstream video understanding tasks and achieve state-of-the-art results, including text-video retrieval on YouCook2, and weakly supervised video action segmentation on Breakfast-Action. (iv) we use the automatically-aligned HowTo100M annotations for end-to-end finetuning of the backbone model, and obtain improved performance on downstream action recognition tasks.

count=2
* AutoLoss-Zero: Searching Loss Functions From Scratch for Generic Tasks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Li_AutoLoss-Zero_Searching_Loss_Functions_From_Scratch_for_Generic_Tasks_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_AutoLoss-Zero_Searching_Loss_Functions_From_Scratch_for_Generic_Tasks_CVPR_2022_paper.pdf)]
    * Title: AutoLoss-Zero: Searching Loss Functions From Scratch for Generic Tasks
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Hao Li, Tianwen Fu, Jifeng Dai, Hongsheng Li, Gao Huang, Xizhou Zhu
    * Abstract: Significant progress has been achieved in automating the design of various components in deep networks. However, the automatic design of loss functions for generic tasks with various evaluation metrics remains under-investigated. Previous works on handcrafting loss functions heavily rely on human expertise, which limits their extensibility. Meanwhile, searching for loss functions is nontrivial due to the vast search space. Existing efforts mainly tackle the issue by employing task-specific heuristics on specific tasks and particular metrics. Such work cannot be extended to other tasks without arduous human effort. In this paper, we propose AutoLoss-Zero, which is a general framework for searching loss functions from scratch for generic tasks. Specifically, we design an elementary search space composed only of primitive mathematical operators to accommodate the heterogeneous tasks and evaluation metrics. A variant of the evolutionary algorithm is employed to discover loss functions in the elementary search space. A loss-rejection protocol and a gradient-equivalence-check strategy are developed so as to improve the search efficiency, which are applicable to generic tasks. Extensive experiments on various computer vision tasks demonstrate that our searched loss functions are on par with or superior to existing loss functions, which generalize well to different datasets and networks. Code shall be released.

count=2
* Affine Medical Image Registration With Coarse-To-Fine Vision Transformer
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Mok_Affine_Medical_Image_Registration_With_Coarse-To-Fine_Vision_Transformer_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Mok_Affine_Medical_Image_Registration_With_Coarse-To-Fine_Vision_Transformer_CVPR_2022_paper.pdf)]
    * Title: Affine Medical Image Registration With Coarse-To-Fine Vision Transformer
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Tony C. W. Mok, Albert C. S. Chung
    * Abstract: Affine registration is indispensable in a comprehensive medical image registration pipeline. However, only a few studies focus on fast and robust affine registration algorithms. Most of these studies utilize convolutional neural networks (CNNs) to learn joint affine and non-parametric registration, while the standalone performance of the affine subnetwork is less explored. Moreover, existing CNN-based affine registration approaches focus either on the local misalignment or the global orientation and position of the input to predict the affine transformation matrix, which are sensitive to spatial initialization and exhibit limited generalizability apart from the training dataset. In this paper, we present a fast and robust learning-based algorithm, Coarse-to-Fine Vision Transformer (C2FViT), for 3D affine medical image registration. Our method naturally leverages the global connectivity and locality of the convolutional vision transformer and the multi-resolution strategy to learn the global affine registration. We evaluate our method on 3D brain atlas registration and template-matching normalization. Comprehensive results demonstrate that our method is superior to the existing CNNs-based affine registration methods in terms of registration accuracy, robustness and generalizability while preserving the runtime advantage of the learning-based methods. The source code is available at https://github.com/cwmok/C2FViT.

count=2
* Efficient Progressive High Dynamic Range Image Restoration via Attention and Alignment Network
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Yu_Efficient_Progressive_High_Dynamic_Range_Image_Restoration_via_Attention_and_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Yu_Efficient_Progressive_High_Dynamic_Range_Image_Restoration_via_Attention_and_CVPRW_2022_paper.pdf)]
    * Title: Efficient Progressive High Dynamic Range Image Restoration via Attention and Alignment Network
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Gaocheng Yu, Jin Zhang, Zhe Ma, Hongbin Wang
    * Abstract: HDR is an important part of computational photography technology. In this paper, we propose a lightweight neural network called Efficient Attention-and-alignment-guided Progressive Network (EAPNet) for the challenge NTIRE 2022 HDR Track 1 and Track 2. We introduce a multi-dimensional lightweight encoding module to extract features. Besides, we propose Progressive Dilated U-shape Block (PDUB) that can be a progressive plug-and-play module for dynamically tuning MAccs and PSNR. Finally, we use fast and low-power feature-align module to deal with misalignment problem in place of the time-consuming Deformable Convolutional Network (DCN). The experiments show that our method achieves about 20 times compression on MAccs with better mu-PSNR and PSNR compared to the state-of-the-art method. We got the second place of both two tracks during the testing phase. Figure1. shows the visualized result of NTIRE 2022 HDR challenge.

count=2
* CIPPSRNet: A Camera Internal Parameters Perception Network Based Contrastive Learning for Thermal Image Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Wang_CIPPSRNet_A_Camera_Internal_Parameters_Perception_Network_Based_Contrastive_Learning_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/papers/Wang_CIPPSRNet_A_Camera_Internal_Parameters_Perception_Network_Based_Contrastive_Learning_CVPRW_2022_paper.pdf)]
    * Title: CIPPSRNet: A Camera Internal Parameters Perception Network Based Contrastive Learning for Thermal Image Super-Resolution
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Kai Wang, Qigong Sun, Yicheng Wang, Huiyuan Wei, Chonghua Lv, Xiaolin Tian, Xu Liu
    * Abstract: Thermal Image Super-Resolution (TISR) is a technique for converting Low-Resolution (LR) thermal images to High-Resolution (HR) thermal images. This technique has recently become a research hotspot due to its ability to reduce sensor costs and improve visual perception. However, current research does not provide an effective solution for multi-sensor data training, possibly driven by pixel mismatch and simple degradation setting issues. In this paper, we proposed a Camera Internal Parameters Perception Network (CIPPSRNet) for LR thermal image enhancement. The camera internal parameters (CIP) were explicitly modeled as a feature representation, the LR features were transformed into the intermediate domain containing the internal parameters information by perceiving CIP representation. The mapping between the intermediate domain and the spatial domain of the HR features was learned by CIPPSRNet. In addition, we introduced contrastive learning to optimize the pretrained Camera Internal Parameters Representation Network and the feature encoders. Our proposed network is capable of achieving a more efficient transformation from the LR to the HR domains. Additionally, the use of contrastive learning can improve the network's adaptability to misalignment data with insufficient pixel matching and its robustness. Experiments on PBVS2022 TISR Dataset show that our network has achieved state-of-the-art performance for the Thermal SR task.

count=2
* SmartBrush: Text and Shape Guided Object Inpainting With Diffusion Model
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_SmartBrush_Text_and_Shape_Guided_Object_Inpainting_With_Diffusion_Model_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_SmartBrush_Text_and_Shape_Guided_Object_Inpainting_With_Diffusion_Model_CVPR_2023_paper.pdf)]
    * Title: SmartBrush: Text and Shape Guided Object Inpainting With Diffusion Model
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, Kun Zhang
    * Abstract: Generic image inpainting aims to complete a corrupted image by borrowing surrounding information, which barely generates novel content. By contrast, multi-modal inpainting provides more flexible and useful controls on the inpainted content, e.g., a text prompt can be used to describe an object with richer attributes, and a mask can be used to constrain the shape of the inpainted object rather than being only considered as a missing area. We propose a new diffusion-based model named SmartBrush for completing a missing region with an object using both text and shape-guidance. While previous work such as DALLE-2 and Stable Diffusion can do text-guided inapinting they do not support shape guidance and tend to modify background texture surrounding the generated object. Our model incorporates both text and shape guidance with precision control. To preserve the background better, we propose a novel training and sampling strategy by augmenting the diffusion U-net with object-mask prediction. Lastly, we introduce a multi-task training strategy by jointly training inpainting with text-to-image generation to leverage more training data. We conduct extensive experiments showing that our model outperforms all baselines in terms of visual quality, mask controllability, and background preservation.

count=2
* Improved Distribution Matching for Dataset Condensation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Improved_Distribution_Matching_for_Dataset_Condensation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Improved_Distribution_Matching_for_Dataset_Condensation_CVPR_2023_paper.pdf)]
    * Title: Improved Distribution Matching for Dataset Condensation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Ganlong Zhao, Guanbin Li, Yipeng Qin, Yizhou Yu
    * Abstract: Dataset Condensation aims to condense a large dataset into a smaller one while maintaining its ability to train a well-performing model, thus reducing the storage cost and training effort in deep learning applications. However, conventional dataset condensation methods are optimization-oriented and condense the dataset by performing gradient or parameter matching during model optimization, which is computationally intensive even on small datasets and models. In this paper, we propose a novel dataset condensation method based on distribution matching, which is more efficient and promising. Specifically, we identify two important shortcomings of naive distribution matching (i.e., imbalanced feature numbers and unvalidated embeddings for distance computation) and address them with three novel techniques (i.e., partitioning and expansion augmentation, efficient and enriched model sampling, and class-aware distribution regularization). Our simple yet effective method outperforms most previous optimization-oriented methods with much fewer computational resources, thereby scaling data condensation to larger datasets and models. Extensive experiments demonstrate the effectiveness of our method. Codes are available at https://github.com/uitrbn/IDM

count=2
* Robust Partial Fingerprint Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/Biometrics/html/Zhang_Robust_Partial_Fingerprint_Recognition_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/Biometrics/papers/Zhang_Robust_Partial_Fingerprint_Recognition_CVPRW_2023_paper.pdf)]
    * Title: Robust Partial Fingerprint Recognition
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yufei Zhang, Rui Zhao, Ziyi Zhao, Naveen Ramakrishnan, Manoj Aggarwal, Gerard Medioni, Qiang Ji
    * Abstract: Low quality capture and obstruction on fingers often result in partially visible fingerprint images, which imposes challenge for fingerprint recognition. In this work, motivated from the practical use cases, we first systematically studied different types of partial occlusion. Specifically, two major types of partial occlusion, including six granular types, and the corresponding methods to simulate each type for model evaluation and improvement were introduced. Second, we proposed a novel Robust Partial Fingerprint (RPF) recognition framework to mitigate the performance degradation due to occlusion. RPF effectively encodes the knowledge about partial fingerprints through occlusion-enhanced data augmentation, and explicitly captures the missing regions for robust feature extraction through occlusion-aware modeling. Finally, we demonstrated the effectiveness of RPF through extensive experiments. Particularly, baseline fingerprint recognition models can degrade the recognition accuracy measured in FRR @ FAR=0.1% from 14.67% to 17.57% at 10% occlusion ratio on the challenging NIST dataset, while RPF instead improves the recognition performance to 9.99% under the same occlusion ratio. Meanwhile, we presented a set of empirical analysis through visual explanation, matching score analysis, and uncertainty modeling, providing insights into the recognition model's behavior and potential directions of enhancement.

count=2
* VideoCon: Robust Video-Language Alignment via Contrast Captions
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Bansal_VideoCon_Robust_Video-Language_Alignment_via_Contrast_Captions_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Bansal_VideoCon_Robust_Video-Language_Alignment_via_Contrast_Captions_CVPR_2024_paper.pdf)]
    * Title: VideoCon: Robust Video-Language Alignment via Contrast Captions
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Hritik Bansal, Yonatan Bitton, Idan Szpektor, Kai-Wei Chang, Aditya Grover
    * Abstract: Despite being (pre)trained on a massive amount of data state-of-the-art video-language alignment models are not robust to semantically-plausible contrastive changes in the video captions. Our work addresses this by identifying a broad spectrum of contrast misalignments such as replacing entities actions and flipping event order which alignment models should be robust against. To this end we introduce the VideoCon a video-language alignment dataset constructed by a large language model that generates plausible contrast video captions and explanations for differences between original and contrast video captions. Then a generative video-language model is finetuned with VideoCon to assess video-language entailment and generate explanations. Our VideoCon-based alignment model significantly outperforms current models. It exhibits a 12-point increase in AUC for the video-language alignment task on human-generated contrast captions. Finally our model sets new state of the art zero-shot performance in temporally-extensive video-language tasks such as text-to-video retrieval (SSv2-Temporal) and video question answering (ATP-Hard). Moreover our model shows superior performance on novel videos and human-crafted captions and explanations.

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
* RAM-Avatar: Real-time Photo-Realistic Avatar from Monocular Videos with Full-body Control
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Deng_RAM-Avatar_Real-time_Photo-Realistic_Avatar_from_Monocular_Videos_with_Full-body_Control_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_RAM-Avatar_Real-time_Photo-Realistic_Avatar_from_Monocular_Videos_with_Full-body_Control_CVPR_2024_paper.pdf)]
    * Title: RAM-Avatar: Real-time Photo-Realistic Avatar from Monocular Videos with Full-body Control
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xiang Deng, Zerong Zheng, Yuxiang Zhang, Jingxiang Sun, Chao Xu, Xiaodong Yang, Lizhen Wang, Yebin Liu
    * Abstract: This paper focuses on advancing the applicability of human avatar learning methods by proposing RAM-Avatar which learns a Real-time photo-realistic Avatar that supports full-body control from Monocular videos. To achieve this goal RAM-Avatar leverages two statistical templates responsible for modeling the facial expression and hand gesture variations while a sparsely computed dual attention module is introduced upon another body template to facilitate high-fidelity texture rendering for the torsos and limbs. Building on this foundation we deploy a lightweight yet powerful StyleUnet along with a temporal-aware discriminator to achieve real-time realistic rendering. To enable robust animation for out-of-distribution poses we propose a Motion Distribution Align module to compensate for the discrepancies between the training and testing motion distribution. Results and extensive experiments conducted in various experimental settings demonstrate the superiority of our proposed method and a real-time live system is proposed to further push research into applications. The training and testing code will be released for research purposes.

count=2
* C2KD: Bridging the Modality Gap for Cross-Modal Knowledge Distillation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.pdf)]
    * Title: C2KD: Bridging the Modality Gap for Cross-Modal Knowledge Distillation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Fushuo Huo, Wenchao Xu, Jingcai Guo, Haozhao Wang, Song Guo
    * Abstract: Existing Knowledge Distillation (KD) methods typically focus on transferring knowledge from a large-capacity teacher to a low-capacity student model achieving substantial success in unimodal knowledge transfer. However existing methods can hardly be extended to Cross-Modal Knowledge Distillation (CMKD) where the knowledge is transferred from a teacher modality to a different student modality with inference only on the distilled student modality. We empirically reveal that the modality gap i.e. modality imbalance and soft label misalignment incurs the ineffectiveness of traditional KD in CMKD. As a solution we propose a novel \underline C ustomized \underline C rossmodal \underline K nowledge \underline D istillation (C^2KD). Specifically to alleviate the modality gap the pre-trained teacher performs bidirectional distillation with the student to provide customized knowledge. The On-the-Fly Selection Distillation(OFSD) strategy is applied to selectively filter out the samples with misaligned soft labels where we distill cross-modal knowledge from non-target classes to avoid the modality imbalance issue. To further provide receptive cross-modal knowledge proxy student and teacher inheriting unimodal and cross-modal knowledge is formulated to progressively transfer cross-modal knowledge through bidirectional distillation. Experimental results on audio-visual image-text and RGB-depth datasets demonstrate that our method can effectively transfer knowledge across modalities achieving superior performance against traditional KD by a large margin.

count=2
* Rich Human Feedback for Text-to-Image Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.pdf)]
    * Title: Rich Human Feedback for Text-to-Image Generation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, Junjie Ke, Krishnamurthy Dj Dvijotham, Katherine M. Collins, Yiwen Luo, Yang Li, Kai J Kohlhoff, Deepak Ramachandran, Vidhya Navalpakkam
    * Abstract: Recent Text-to-Image (T2I) generation models such as Stable Diffusion and Imagen have made significant progress in generating high-resolution images based on text descriptions. However many generated images still suffer from issues such as artifacts/implausibility misalignment with text descriptions and low aesthetic quality. Inspired by the success of Reinforcement Learning with Human Feedback (RLHF) for large language models prior works collected human-provided scores as feedback on generated images and trained a reward model to improve the T2I generation. In this paper we enrich the feedback signal by (i) marking image regions that are implausible or misaligned with the text and (ii) annotating which words in the text prompt are misrepresented or missing on the image. We collect such rich human feedback on 18K generated images (RichHF-18K) and train a multimodal transformer to predict the rich feedback automatically. We show that the predicted rich human feedback can be leveraged to improve image generation for example by selecting high-quality training data to finetune and improve the generative models or by creating masks with predicted heatmaps to inpaint the problematic regions. Notably the improvements generalize to models (Muse) beyond those used to generate the images on which human feedback data were collected (Stable Diffusion variants). The RichHF-18K data set will be released in our GitHub repository: https://github.com/google-research/google-research/tree/master/richhf_18k.

count=2
* Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Towards_Robust_Event-guided_Low-Light_Image_Enhancement_A_Large-Scale_Real-World_Event-Image_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_Towards_Robust_Event-guided_Low-Light_Image_Enhancement_A_Large-Scale_Real-World_Event-Image_CVPR_2024_paper.pdf)]
    * Title: Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Guoqiang Liang, Kanghao Chen, Hangyu Li, Yunfan Lu, Lin Wang
    * Abstract: Event camera has recently received much attention for low-light image enhancement (LIE) thanks to their distinct advantages such as high dynamic range. However current research is prohibitively restricted by the lack of large-scale real-world and spatial-temporally aligned event-image datasets. To this end we propose a real-world (indoor and outdoor) dataset comprising over 30K pairs of images and events under both low and normal illumination conditions. To achieve this we utilize a robotic arm that traces a consistent non-linear trajectory to curate the dataset with spatial alignment precision under 0.03mm. We then introduce a matching alignment strategy rendering 90% of our dataset with errors less than 0.01s. Based on the dataset we propose a novel event-guided LIE approach called EvLight towards robust performance in real-world low-light scenes. Specifically we first design the multi-scale holistic fusion branch to extract holistic structural and textural information from both events and images. To ensure robustness against variations in the regional illumination and noise we then introduce a Signal-to-Noise-Ratio (SNR)-guided regional feature selection to selectively fuse features of images from regions with high SNR and enhance those with low SNR by extracting regional structural information from events. our EvLight significantly surpasses the frame-based methods e.g. Retinexformer by 1.14 dB and 2.62 dB respectively. Code and datasets are available at https://vlislab22.github.io/eg-lowlight/.

count=2
* MANUS: Markerless Grasp Capture using Articulated 3D Gaussians
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Pokhariya_MANUS_Markerless_Grasp_Capture_using_Articulated_3D_Gaussians_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Pokhariya_MANUS_Markerless_Grasp_Capture_using_Articulated_3D_Gaussians_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Pokhariya_MANUS_Markerless_Grasp_Capture_using_Articulated_3D_Gaussians_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Pokhariya_MANUS_Markerless_Grasp_Capture_using_Articulated_3D_Gaussians_CVPR_2024_paper.pdf)]
    * Title: MANUS: Markerless Grasp Capture using Articulated 3D Gaussians
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chandradeep Pokhariya, Ishaan Nikhil Shah, Angela Xing, Zekun Li, Kefan Chen, Avinash Sharma, Srinath Sridhar
    * Abstract: Understanding how we grasp objects with our hands has important applications in areas like robotics and mixed reality. However this challenging problem requires accurate modeling of the contact between hands and objects.To capture grasps existing methods use skeletons meshes or parametric models that does not represent hand shape accurately resulting in inaccurate contacts. We present MANUS a method for Markerless Hand-Object Grasp Capture using Articulated 3D Gaussians. We build a novel articulated 3D Gaussians representation that extends 3D Gaussian splatting for high-fidelity representation of articulating hands. Since our representation uses Gaussian primitives optimized from the multi-view pixel-aligned losses it enables us to efficiently and accurately estimate contacts between the hand and the object. For the most accurate results our method requires tens of camera views that current datasets do not provide. We therefore build MANUS-Grasps a new dataset that contains hand-object grasps viewed from 50+ cameras across 30+ scenes 3 subjects and comprising over 7M frames. In addition to extensive qualitative results we also show that our method outperforms others on a quantitative contact evaluation method that uses paint transfer from the object to the hand.

count=2
* ProxyCap: Real-time Monocular Full-body Capture in World Space via Human-Centric Proxy-to-Motion Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_ProxyCap_Real-time_Monocular_Full-body_Capture_in_World_Space_via_Human-Centric_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_ProxyCap_Real-time_Monocular_Full-body_Capture_in_World_Space_via_Human-Centric_CVPR_2024_paper.pdf)]
    * Title: ProxyCap: Real-time Monocular Full-body Capture in World Space via Human-Centric Proxy-to-Motion Learning
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yuxiang Zhang, Hongwen Zhang, Liangxiao Hu, Jiajun Zhang, Hongwei Yi, Shengping Zhang, Yebin Liu
    * Abstract: Learning-based approaches to monocular motion capture have recently shown promising results by learning to regress in a data-driven manner. However due to the challenges in data collection and network designs it remains challenging to achieve real-time full-body capture while being accurate in world space. In this work we introduce ProxyCap a human-centric proxy-to-motion learning scheme to learn world-space motions from a proxy dataset of 2D skeleton sequences and 3D rotational motions. Such proxy data enables us to build a learning-based network with accurate world-space supervision while also mitigating the generalization issues. For more accurate and physically plausible predictions in world space our network is designed to learn human motions from a human-centric perspective which enables the understanding of the same motion captured with different camera trajectories. Moreover a contact-aware neural motion descent module is proposed to improve foot-ground contact and motion misalignment with the proxy observations. With the proposed learning-based solution we demonstrate the first real-time monocular full-body capture system with plausible foot-ground contact in world space even using hand-held cameras.

count=2
* Spike-guided Motion Deblurring with Unknown Modal Spatiotemporal Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Spike-guided_Motion_Deblurring_with_Unknown_Modal_Spatiotemporal_Alignment_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Spike-guided_Motion_Deblurring_with_Unknown_Modal_Spatiotemporal_Alignment_CVPR_2024_paper.pdf)]
    * Title: Spike-guided Motion Deblurring with Unknown Modal Spatiotemporal Alignment
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jiyuan Zhang, Shiyan Chen, Yajing Zheng, Zhaofei Yu, Tiejun Huang
    * Abstract: The traditional frame-based cameras that rely on exposure windows for imaging experience motion blur in high-speed scenarios. Frame-based deblurring methods lack reliable motion cues to restore sharp images under extreme blur conditions. The spike camera is a novel neuromorphic visual sensor that outputs spike streams with ultra-high temporal resolution. It can supplement the temporal information lost in traditional cameras and guide motion deblurring. However in real-world scenarios aligning discrete RGB images and continuous spike streams along both temporal and spatial axes is challenging due to the complexity of calibrating their coordinates device displacements in vibrations and time deviations. Misalignment of pixels leads to severe degradation of deblurring. We introduce the first framework for spike-guided motion deblurring without knowing the spatiotemporal alignment between spikes and images. To address the problem we first propose a novel three-stage network containing a basic deblurring net a carefully designed bi-directional deformable aligning module and a flow-based multi-scale fusion net. Experimental results demonstrate that our approach can effectively guide the image deblurring with unknown alignment surpassing the performance of other methods. Public project page: https://github.com/Leozhangjiyuan/UaSDN.

count=2
* Learning Maximum Margin Temporal Warping for Action Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Wang_Learning_Maximum_Margin_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Wang_Learning_Maximum_Margin_2013_ICCV_paper.pdf)]
    * Title: Learning Maximum Margin Temporal Warping for Action Recognition
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Jiang Wang, Ying Wu
    * Abstract: Temporal misalignment and duration variation in video actions largely influence the performance of action recognition, but it is very difficult to specify effective temporal alignment on action sequences. To address this challenge, this paper proposes a novel discriminative learning-based temporal alignment method, called maximum margin temporal warping (MMTW), to align two action sequences and measure their matching score. Based on the latent structure SVM formulation, the proposed MMTW method is able to learn a phantom action template to represent an action class for maximum discrimination against other classes. The recognition of this action class is based on the associated learned alignment of the input action. Extensive experiments on five benchmark datasets have demonstrated that this MMTW model is able to significantly promote the accuracy and robustness of action recognition under temporal misalignment and variations.

count=2
* Person Re-Identification With Correspondence Structure Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Shen_Person_Re-Identification_With_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Shen_Person_Re-Identification_With_ICCV_2015_paper.pdf)]
    * Title: Person Re-Identification With Correspondence Structure Learning
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Yang Shen, Weiyao Lin, Junchi Yan, Mingliang Xu, Jianxin Wu, Jingdong Wang
    * Abstract: This paper addresses the problem of handling spatial misalignments due to camera-view changes or human-pose variations in person re-identification. We first introduce a boosting-based approach to learn a correspondence structure which indicates the patch-wise matching probabilities between images from a target camera pair. The learned correspondence structure can not only capture the spatial correspondence pattern between cameras but also handle the viewpoint or human-pose variation in individual images. We further introduce a global-based matching process. It integrates a global matching constraint over the learned correspondence structure to exclude cross-view misalignments during the image patch matching process, hence achieving a more reliable matching score between images. Experimental results on various datasets demonstrate the effectiveness of our approach.

count=2
* Multi-View Non-Rigid Refinement and Normal Selection for High Quality 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Haque_Multi-View_Non-Rigid_Refinement_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Haque_Multi-View_Non-Rigid_Refinement_ICCV_2017_paper.pdf)]
    * Title: Multi-View Non-Rigid Refinement and Normal Selection for High Quality 3D Reconstruction
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Sk. Mohammadul Haque, Venu Madhav Govindu
    * Abstract: In recent years, there have been a variety of proposals for high quality 3D reconstruction by fusion of depth and normal maps that contain good low and high frequency information respectively. Typically, these methods create an initial mesh representation of the complete object or scene being scanned. Subsequently, normal estimates are assigned to each mesh vertex and a mesh-normal fusion step is carried out. In this paper, we present a complete pipeline for such depth-normal fusion. The key innovations in our pipeline are twofold. Firstly, we introduce a global multi-view non-rigid refinement step that corrects for the non-rigid misalignment present in the depth and normal maps. We demonstrate that such a correction is crucial for preserving fine-scale 3D features in the final reconstruction. Secondly, despite adequate care, the averaging of multiple normals invariably results in blurring of 3D detail. To mitigate this problem, we propose an approach that selects one out of many available normals. Our global cost for normal selection incorporates a variety of desirable properties and can be efficiently solved using graph cuts. We demonstrate the efficacy of our approach in generating high quality 3D reconstructions of both synthetic and real 3D models and compare with existing methods in the literature.

count=2
* Jointly Aligning Millions of Images With Deep Penalised Reconstruction Congealing
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Annunziata_Jointly_Aligning_Millions_of_Images_With_Deep_Penalised_Reconstruction_Congealing_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Annunziata_Jointly_Aligning_Millions_of_Images_With_Deep_Penalised_Reconstruction_Congealing_ICCV_2019_paper.pdf)]
    * Title: Jointly Aligning Millions of Images With Deep Penalised Reconstruction Congealing
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Roberto Annunziata,  Christos Sagonas,  Jacques Cali
    * Abstract: Extrapolating fine-grained pixel-level correspondences in a fully unsupervised manner from a large set of misaligned images can benefit several computer vision and graphics problems, e.g. co-segmentation, super-resolution, image edit propagation, structure-from-motion, and 3D reconstruction. Several joint image alignment and congealing techniques have been proposed to tackle this problem, but robustness to initialisation, ability to scale to large datasets, and alignment accuracy seem to hamper their wide applicability. To overcome these limitations, we propose an unsupervised joint alignment method leveraging a densely fused spatial transformer network to estimate the warping parameters for each image and a low-capacity auto-encoder whose reconstruction error is used as an auxiliary measure of joint alignment. Experimental results on digits from multiple versions of MNIST (i.e., original, perturbed, affNIST and infiMNIST) and faces from LFW, show that our approach is capable of aligning millions of images with high accuracy and robustness to different levels and types of perturbation. Moreover, qualitative and quantitative results suggest that the proposed method outperforms state-of-the-art approaches both in terms of alignment quality and robustness to initialisation.

count=2
* Beyond Human Parts: Dual Part-Aligned Representations for Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Guo_Beyond_Human_Parts_Dual_Part-Aligned_Representations_for_Person_Re-Identification_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Guo_Beyond_Human_Parts_Dual_Part-Aligned_Representations_for_Person_Re-Identification_ICCV_2019_paper.pdf)]
    * Title: Beyond Human Parts: Dual Part-Aligned Representations for Person Re-Identification
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Jianyuan Guo,  Yuhui Yuan,  Lang Huang,  Chao Zhang,  Jin-Ge Yao,  Kai Han
    * Abstract: Person re-identification is a challenging task due to various complex factors. Recent studies have attempted to integrate human parsing results or externally defined attributes to help capture human parts or important object regions. On the other hand, there still exist many useful contextual cues that do not fall into the scope of predefined human parts or attributes. In this paper, we address the missed contextual cues by exploiting both the accurate human parts and the coarse non-human parts. In our implementation, we apply a human parsing model to extract the binary human part masks and a self-attention mechanism to capture the soft latent (non-human) part masks. We verify the effectiveness of our approach with new state-of-the-art performance on three challenging benchmarks: Market-1501, DukeMTMC-reID and CUHK03. Our implementation is available at https://github.com/ggjy/P2Net.pytorch.

count=2
* Fast Object Detection in Compressed Video
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Fast_Object_Detection_in_Compressed_Video_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Fast_Object_Detection_in_Compressed_Video_ICCV_2019_paper.pdf)]
    * Title: Fast Object Detection in Compressed Video
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Shiyao Wang,  Hongchao Lu,  Zhidong Deng
    * Abstract: Object detection in videos has drawn increasing attention since it is more practical in real scenarios. Most of the deep learning methods use CNNs to process each decoded frame in a video stream individually. However, the free of charge yet valuable motion information already embedded in the video compression format is usually overlooked. In this paper, we propose a fast object detection method by taking advantage of this with a novel Motion aided Memory Network (MMNet). The MMNet has two major advantages: 1) It significantly accelerates the procedure of feature extraction for compressed videos. It only need to run a complete recognition network for I-frames, i.e. a few reference frames in a video, and it produces the features for the following P frames (predictive frames) with a light weight memory network, which runs fast; 2) Unlike existing methods that establish an additional network to model motion of frames, we take full advantage of both motion vectors and residual errors that are freely available in video streams. To our best knowledge, the MMNet is the first work that investigates a deep convolutional detector on compressed videos. Our method is evaluated on the large-scale ImageNet VID dataset, and the results show that it is 3x times faster than single image detector R-FCN and 10x times faster than high-performance detector MANet at a minor accuracy loss.

count=2
* FaPN: Feature-Aligned Pyramid Network for Dense Image Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_FaPN_Feature-Aligned_Pyramid_Network_for_Dense_Image_Prediction_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_FaPN_Feature-Aligned_Pyramid_Network_for_Dense_Image_Prediction_ICCV_2021_paper.pdf)]
    * Title: FaPN: Feature-Aligned Pyramid Network for Dense Image Prediction
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Shihua Huang, Zhichao Lu, Ran Cheng, Cheng He
    * Abstract: Recent advancements in deep neural networks have made remarkable leap-forwards in dense image prediction. However, the issue of feature alignment remains as neglected by most existing approaches for simplicity. Direct pixel addition between upsampled and local features leads to feature maps with misaligned contexts that, in turn, translate to mis-classifications in prediction, especially on object boundaries. In this paper, we propose a feature alignment module that learns transformation offsets of pixels to contextually align upsampled higher-level features; and another feature selection module to emphasize the lower-level features with rich spatial details. We then integrate these two modules in a top-down pyramidal architecture and present the Feature-aligned Pyramid Network (FaPN). Extensive experimental evaluations on four dense prediction tasks and four datasets have demonstrated the efficacy of FaPN, yielding an overall improvement of 1.2 - 2.6 points in AP / mIoU over FPN when paired with Faster / Mask R-CNN. In particular, our FaPN achieves the state-of-the-art of 56.7% mIoU on ADE20K when integrated within Mask-Former. The code is available from https://github.com/EMI-Group/FaPN.

count=2
* Dual-Camera Super-Resolution With Aligned Attention Modules
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Dual-Camera_Super-Resolution_With_Aligned_Attention_Modules_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Dual-Camera_Super-Resolution_With_Aligned_Attention_Modules_ICCV_2021_paper.pdf)]
    * Title: Dual-Camera Super-Resolution With Aligned Attention Modules
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Tengfei Wang, Jiaxin Xie, Wenxiu Sun, Qiong Yan, Qifeng Chen
    * Abstract: We present a novel approach to reference-based super-resolution (RefSR) with the focus on dual-camera super-resolution (DCSR), which utilizes reference images for high-quality and high-fidelity results. Our proposed method generalizes the standard patch-based feature matching with spatial alignment operations. We further explore the dual-camera super-resolution that is one promising application of RefSR, and build a dataset that consists of 146 image pairs from the main and telephoto cameras in a smartphone. To bridge the domain gaps between real-world images and the training images, we propose a self-supervised domain adaptation strategy for real-world images. Extensive experiments on our dataset and a public benchmark demonstrate clear improvement achieved by our method over state of the art in both quantitative evaluation and visual comparisons.

count=2
* Learning RAW-to-sRGB Mappings With Inaccurately Aligned Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Learning_RAW-to-sRGB_Mappings_With_Inaccurately_Aligned_Supervision_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learning_RAW-to-sRGB_Mappings_With_Inaccurately_Aligned_Supervision_ICCV_2021_paper.pdf)]
    * Title: Learning RAW-to-sRGB Mappings With Inaccurately Aligned Supervision
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zhilu Zhang, Haolin Wang, Ming Liu, Ruohao Wang, Jiawei Zhang, Wangmeng Zuo
    * Abstract: Learning RAW-to-sRGB mapping has drawn increasing attention in recent years, wherein an input raw image is trained to imitate the target sRGB image captured by another camera. However, the severe color inconsistency makes it very challenging to generate well-aligned training pairs of input raw and target sRGB images. While learning with inaccurately aligned supervision is prone to causing pixel shift and producing blurry results. In this paper, we circumvent such issue by presenting a joint learning model for image alignment and RAW-to-sRGB mapping. To diminish the effect of color inconsistency in image alignment, we introduce to use a global color mapping (GCM) module to generate an initial sRGB image given the input raw image, which can keep the spatial location of the pixels unchanged, and the target sRGB image is utilized to guide GCM for converting the color towards it. Then a pre-trained optical flow estimation network (e.g., PWC-Net) is deployed to warp the target sRGB image to align with the GCM output. To alleviate the effect of inaccurately aligned supervision, the warped target sRGB image is leveraged to learn RAW-to-sRGB mapping. When training is done, the GCM module and optical flow network can be detached, thereby bringing no extra computation cost for inference. Experiments show that our method performs favorably against state-of-the-arts on ZRR and SR-RAW datasets. With our joint learning model, a light-weight backbone can achieve better quantitative and qualitative performance on ZRR dataset. Codes are available at https://github.com/cszhilu1998/RAW-to-sRGB.

count=2
* Learning Specialized Activation Functions With the Piecewise Linear Unit
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_Learning_Specialized_Activation_Functions_With_the_Piecewise_Linear_Unit_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Learning_Specialized_Activation_Functions_With_the_Piecewise_Linear_Unit_ICCV_2021_paper.pdf)]
    * Title: Learning Specialized Activation Functions With the Piecewise Linear Unit
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yucong Zhou, Zezhou Zhu, Zhao Zhong
    * Abstract: The choice of activation functions is crucial for modern deep neural networks. Popular hand-designed activation functions like Rectified Linear Unit(ReLU) and its variants show promising performance in various tasks and models. Swish, the automatically discovered activation function, outperforms ReLU on many challenging datasets. However, it has two main drawbacks. First, the tree-based search space is highly discrete and restricted, making it difficult to searching. Second, the sample-based searching method is inefficient, making it infeasible to find specialized activation functions for each dataset or neural architecture. To tackle these drawbacks, we propose a new activation function called Piecewise Linear Unit(PWLU), which incorporates a carefully designed formulation and learning method. It can learn specialized activation functions and achieves SOTA performance on large-scale datasets like ImageNet and COCO. For example, on ImageNet classification dataset, PWLU improves 0.9%/0.53%/1.0%/1.7%/1.0% top-1 accuracy over Swish for ResNet-18/ResNet-50/MobileNet-V2/MobileNet-V3/EfficientNet-B0. PWLU is also easy to implement and efficient at inference, which can be widely applied in real-world applications.

count=2
* Cross-Modal Matching CNN for Autonomous Driving Sensor Data Monitoring
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/html/Chen_Cross-Modal_Matching_CNN_for_Autonomous_Driving_Sensor_Data_Monitoring_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Chen_Cross-Modal_Matching_CNN_for_Autonomous_Driving_Sensor_Data_Monitoring_ICCVW_2021_paper.pdf)]
    * Title: Cross-Modal Matching CNN for Autonomous Driving Sensor Data Monitoring
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yiqiang Chen, Feng Liu, Ke Pei
    * Abstract: Multiple sensor types have been increasingly used in modern autonomous driving systems (ADS) to ensure safer perception. Through applications of multiple modalities of perception sensors that differ in their physical properties, obtained data complement to each other and provide a more robust view of surroundings. On the other hand, however, sensor data fault is inevitable thus lead to wrong perception results and consequently endangers the overall safety of the vehicle. In this paper, we present a cross-modal Convolutional Neural Networks (CNN) for autonomous driving sensor data monitoring functions, such as fault detection and online data quality assessment. Assuming the overlapping view of different sensors should be consistent under normal circumstances, we detect anomalies such as mis-synchronisation through matching camera image and LIDAR point cloud. A masked pixel-wise metric learning loss is proposed to improve exploration of the local structures and to build an alignment-sensitive pixel embedding. In our experiments with a selected KITTI dataset and specially tailored fault data generation methods, the approach shows a promising success for sensor fault detection and point cloud quality assessment (PCQA) results.

count=2
* Size Does Matter: Size-aware Virtual Try-on via Clothing-oriented Transformation Try-on Network
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Size_Does_Matter_Size-aware_Virtual_Try-on_via_Clothing-oriented_Transformation_Try-on_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Size_Does_Matter_Size-aware_Virtual_Try-on_via_Clothing-oriented_Transformation_Try-on_ICCV_2023_paper.pdf)]
    * Title: Size Does Matter: Size-aware Virtual Try-on via Clothing-oriented Transformation Try-on Network
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Chieh-Yun Chen, Yi-Chung Chen, Hong-Han Shuai, Wen-Huang Cheng
    * Abstract: Virtual try-on tasks aim at synthesizing realistic try-on results by trying target clothes on humans. Most previous works relied on the Thin Plate Spline or appearance flows to warp clothes to fit human body shapes. However, both approaches cannot handle complex warping, leading to over distortion or misalignment. Furthermore, there is a critical unaddressed challenge of adjusting clothing sizes for try-on. To tackle these issues, we propose a Clothing-Oriented Transformation Try-On Network (COTTON). COTTON leverages clothing structure with landmarks and segmentation to design a novel landmark-guided transformation for precisely deforming clothes, allowing for size adjustment during try-on. Additionally, to properly remove the clothing region from the human image without losing significant human characteristics, we propose a clothing elimination policy based on both transformed clothes and human segmentation. This method enables users to try on clothes tucked-in or untucked while retaining more human characteristics. Both qualitative and quantitative results show that COTTON outperforms the state-of-the-art high-resolution virtual try-on approaches. All the code is available at https://github.com/cotton6/COTTON-size-does-matter.

count=2
* Pseudo-label Alignment for Semi-supervised Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Pseudo-label_Alignment_for_Semi-supervised_Instance_Segmentation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Pseudo-label_Alignment_for_Semi-supervised_Instance_Segmentation_ICCV_2023_paper.pdf)]
    * Title: Pseudo-label Alignment for Semi-supervised Instance Segmentation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Jie Hu, Chen Chen, Liujuan Cao, Shengchuan Zhang, Annan Shu, Guannan Jiang, Rongrong Ji
    * Abstract: Pseudo-labeling is significant for semi-supervised instance segmentation, which generates instance masks and classes from unannotated images for subsequent training. However, in existing pipelines, pseudo-labels that contain valuable information may be directly filtered out due to mismatches in class and mask quality. To address this issue, we propose a novel framework, called pseudo-label aligning instance segmentation (PAIS), in this paper. In PAIS, we devise a dynamic aligning loss (DALoss) that adjusts the weights of semi-supervised loss terms with varying class and mask score pairs. Through extensive experiments conducted on the COCO and Cityscapes datasets, we demonstrate that PAIS is a promising framework for semi-supervised instance segmentation, particularly in cases where labeled data is severely limited. Notably, with just 1% labeled data, PAIS achieves 21.2 mAP (based on Mask-RCNN) and 19.9 mAP (based on K-Net) on the COCO dataset, outperforming the current state-of-the-art model, i.e., NoisyBoundary with 7.7 mAP, by a margin of over 12 points. Code is available at: https://github.com/hujiecpp/PAIS.

count=2
* Domain Generalization via Balancing Training Difficulty and Model Capability
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Domain_Generalization_via_Balancing_Training_Difficulty_and_Model_Capability_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Domain_Generalization_via_Balancing_Training_Difficulty_and_Model_Capability_ICCV_2023_paper.pdf)]
    * Title: Domain Generalization via Balancing Training Difficulty and Model Capability
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Xueying Jiang, Jiaxing Huang, Sheng Jin, Shijian Lu
    * Abstract: Domain generalization (DG) aims to learn domaingeneralizable models from one or multiple source domains that can perform well in unseen target domains. Despite its recent progress, most existing work suffers from the misalignment between the difficulty level of training samples and the capability of contemporarily trained models, leading to over-fitting or under-fitting in the trained generalization model. We design MoDify, a Momentum Difficulty framework that tackles the misalignment by balancing the seesaw between the model's capability and the samples' difficulties along the training process. MoDify consists of two novel designs that collaborate to fight against the misalignment while learning domain-generalizable models. The first is MoDify-based Data Augmentation which exploits an RGB Shuffle technique to generate difficulty-aware training samples on the fly. The second is MoDify-based Network Optimization which dynamically schedules the training samples for balanced and smooth learning with appropriate difficulty. Without bells and whistles, a simple implementation of MoDify achieves superior performance across multiple benchmarks. In addition, MoDify can complement existing methods as a plug-in, and it is generic and can work for different visual recognition tasks.

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
* GraphAlign: Enhancing Accurate Feature Alignment by Graph matching for Multi-Modal 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Song_GraphAlign_Enhancing_Accurate_Feature_Alignment_by_Graph_matching_for_Multi-Modal_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Song_GraphAlign_Enhancing_Accurate_Feature_Alignment_by_Graph_matching_for_Multi-Modal_ICCV_2023_paper.pdf)]
    * Title: GraphAlign: Enhancing Accurate Feature Alignment by Graph matching for Multi-Modal 3D Object Detection
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ziying Song, Haiyue Wei, Lin Bai, Lei Yang, Caiyan Jia
    * Abstract: LiDAR and cameras are complementary sensors for 3D object detection in autonomous driving. However, it is challenging to explore the unnatural interaction between point clouds and images, and the critical factor is how to conduct feature alignment of heterogeneous modalities. Currently, many methods achieve feature alignment by projection calibration only, without considering the problem of coordinate conversion accuracy errors between sensors, leading to sub-optimal performance. In this paper, we present GraphAlign, a more accurate feature alignment strategy for 3D object detection by graph matching. Specifically, we fuse image features from a semantic segmentation encoder in the image branch and point cloud features from a 3D Sparse CNN in the LiDAR branch. To save computation, we construct the nearest neighbor relationship by calculating Euclidean distance within the subspaces that are divided into the point cloud features. Through the projection calibration between the image and point cloud, we project the nearest neighbors of point cloud features onto the image features. Then by matching the nearest neighbors with a single point cloud to multiple images, we search for a more appropriate feature alignment. In addition, we provide a self-attention module to enhance the weights of significant relations to fine-tune the feature alignment between heterogeneous modalities. Extensive experiments on nuScenes benchmark demonstrate the effectiveness and efficiency of our GraphAlign.

count=2
* A Graph Based Unsupervised Feature Aggregation for Face Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/LSR/Cheng_A_Graph_Based_Unsupervised_Feature_Aggregation_for_Face_Recognition_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Cheng_A_Graph_Based_Unsupervised_Feature_Aggregation_for_Face_Recognition_ICCVW_2019_paper.pdf)]
    * Title: A Graph Based Unsupervised Feature Aggregation for Face Recognition
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Yu Cheng, Yanfeng Li, Qiankun Liu, Yuan Yao, Venkata Sai Vijay Kumar Pedapudi, Xiaotian Fan, Chi Su, Shengmei Shen
    * Abstract: In most of the testing dataset, the images are collected from video clips or different environment conditions, which implies that the mutual information between pairs are significantly important. To address this problem and utilize this information, in this paper, we propose a graph-based unsupervised feature aggregation method for face recognition. Our method uses the inter-connection between pairs with a directed graph approach thus refine the pair-wise scores. First, based on the assumption that all features follow Gaussian distribution, we derive a iterative updating formula of features. Second, in discrete conditions, we build a directed graph where the affinity matrix is obtained from pair-wise similarities, and filtered by a pre-defined threshold along with K-nearest neighbor. Third, the affinity matrix is used to obtain a pseudo center matrix for the iterative update process. Besides evaluation on face recognition testing dataset, our proposed method can further be applied to semi-supervised learning to handle the unlabelled data for improving the performance of the deep models. We verified the effectiveness on 5 different datasets: IJB-C, CFP, YTF, TrillionPair and IQiYi Video dataset.

count=2
* Robust Template-Based Non-Rigid Motion Tracking Using Local Coordinate Regularization
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Li_Robust_Template-Based_Non-Rigid_Motion_Tracking_Using_Local_Coordinate_Regularization_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Li_Robust_Template-Based_Non-Rigid_Motion_Tracking_Using_Local_Coordinate_Regularization_WACV_2020_paper.pdf)]
    * Title: Robust Template-Based Non-Rigid Motion Tracking Using Local Coordinate Regularization
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Wei Li,  Shang Zhao,  Xiao Xiao,  James Hahn
    * Abstract: In this paper, we propose our template-based non-rigid registration algorithm to address the misalignments in the frame-to-frame motion tracking with single or multiple commodity depth cameras. We analyze the deformation in the local coordinates of neighboring nodes and use this differential representation to formulate the regularization term for the deformation field in our non-rigid registration. The local coordinate regularizations vary for each pair of neighboring nodes based on the tracking status of the surface regions. We propose our tracking strategies for different surface regions to minimize misalignments and reduce error accumulation. This method can thus preserve local geometric features and prevent undesirable distortions. Moreover, we introduce a geodesic-based correspondence estimation algorithm to align surfaces with large displacements. Finally, we demonstrate the effectiveness of our proposed method with detailed experiments.

count=2
* Multi-Level Attentive Adversarial Learning With Temporal Dilation for Unsupervised Video Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Chen_Multi-Level_Attentive_Adversarial_Learning_With_Temporal_Dilation_for_Unsupervised_Video_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Chen_Multi-Level_Attentive_Adversarial_Learning_With_Temporal_Dilation_for_Unsupervised_Video_WACV_2022_paper.pdf)]
    * Title: Multi-Level Attentive Adversarial Learning With Temporal Dilation for Unsupervised Video Domain Adaptation
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Peipeng Chen, Yuan Gao, Andy J. Ma
    * Abstract: Most existing works on unsupervised video domain adaptation attempt to mitigate the distribution gap across domains in frame and video levels. Such two-level distribution alignment approach may suffer from the problems of insufficient alignment for complex video data and misalignment along the temporal dimension. To address these issues, we develop a novel framework of Multi-level Attentive Adversarial Learning with Temporal Dilation (MA2L-TD). Given frame-level features as input, multi-level temporal features are generated and multiple domain discriminators are individually trained by adversarial learning for them. For better distribution alignment, level-wise attention weights are calculated by the degree of domain confusion in each level. To mitigate the negative effect of misalignment, features are aggregated with the attention mechanism determined by individual domain discriminators. Moreover, temporal dilation is designed for sequential non-repeatability to balance the computational efficiency and the possible number of levels. Extensive experimental results show that our proposed method outperforms the state of the arts on four benchmark datasets.

count=2
* Efficient Flow-Guided Multi-Frame De-Fencing
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Tsogkas_Efficient_Flow-Guided_Multi-Frame_De-Fencing_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Tsogkas_Efficient_Flow-Guided_Multi-Frame_De-Fencing_WACV_2023_paper.pdf)]
    * Title: Efficient Flow-Guided Multi-Frame De-Fencing
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Stavros Tsogkas, Fengjia Zhang, Allan Jepson, Alex Levinshtein
    * Abstract: Taking photographs "in-the-wild" is often hindered by fence obstructions that stand between the camera user and the scene of interest, and which are hard or impossible to avoid. De-fencing is the algorithmic process of automatically removing such obstructions from images, revealing the invisible parts of the scene. While this problem can be formulated as a combination of fence segmentation and image inpainting, this often leads to implausible hallucinations of the occluded regions. Existing multi-frame approaches rely on propagating information to a selected keyframe from its temporal neighbors, but they are often inefficient and struggle with alignment of severely obstructed images. In this work we draw inspiration from the video completion literature and develop a simplified framework for multi-frame de-fencing that computes high quality flow maps directly from obstructed frames and uses them to accurately align frames. Our primary focus is efficiency and practicality in a real-world setting: the input to our algorithm is a short image burst (5 frames) -- a data modality commonly available in modern smartphones-- and the output is a single reconstructed keyframe, with the fence removed. Our approach leverages simple yet effective CNN modules, trained on carefully generated synthetic data, and outperforms more complicated alternatives real bursts, both quantitatively and qualitatively, while running real-time.

count=2
* Human Motion Aware Text-to-Video Generation With Explicit Camera Control
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Human_Motion_Aware_Text-to-Video_Generation_With_Explicit_Camera_Control_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Human_Motion_Aware_Text-to-Video_Generation_With_Explicit_Camera_Control_WACV_2024_paper.pdf)]
    * Title: Human Motion Aware Text-to-Video Generation With Explicit Camera Control
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Taehoon Kim, ChanHee Kang, JaeHyuk Park, Daun Jeong, ChangHee Yang, Suk-Ju Kang, Kyeongbo Kong
    * Abstract: With the rise in expectations related to generative models, text-to-video (T2V) models are being actively studied. Existing text-to-video models have limitations such as in generating complex movements replicating human motions. These model often generate unintended human motions, and the scale of the subject is incorrect. To overcome these limitations and generate high-quality videos that depict human motion under plausible viewing angles, we propose a two stage framework in this study. In the first stage a text-driven human motion generation network generates three-dimensional (3D) human motion from input text prompts and then motion-to-skeleton projection module projects generated motions onto a two-dimensional (2D) skeleton. In the second stage, the projected skeletons are used to generate a video in which the movements of a subject are well-represented. We demonstrated that the proposed framework quantitatively and qualitatively outperforms the existing T2V models. Previously reported human motion generation models use texts only or texts and human skeletons. However, our framework only uses texts and outputs a video related to human motion. Moreover, our framework benefits from using skeleton as an additional condition in the text-to-human motion generation networks. To the best of our knowledge, our framework is the first of its kind that uses text-driven human motion generation networks to generate high-quality videos related to human motions. The corresponding codes are available at https://github.com/CSJasper/HMTV.

count=2
* Learning Residual Elastic Warps for Image Stitching Under Dirichlet Boundary Condition
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Learning_Residual_Elastic_Warps_for_Image_Stitching_Under_Dirichlet_Boundary_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Learning_Residual_Elastic_Warps_for_Image_Stitching_Under_Dirichlet_Boundary_WACV_2024_paper.pdf)]
    * Title: Learning Residual Elastic Warps for Image Stitching Under Dirichlet Boundary Condition
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Minsu Kim, Yongjun Lee, Woo Kyoung Han, Kyong Hwan Jin
    * Abstract: Trendy suggestions for learning-based elastic warps enable the deep image stitchings to align images exposed to large parallax errors. Despite the remarkable alignments, the methods struggle with occasional holes or discontinuity between overlapping and non-overlapping regions of a target image as the applied training strategy mostly focuses on overlap region alignment. As a result, they require additional modules such as seam finder and image inpainting for hiding discontinuity and filling holes, respectively. In this work, we suggest Recurrent Elastic Warps (REwarp) that address the problem with Dirichlet boundary condition and boost performances by residual learning for recurrent misalign correction. Specifically, REwarp predicts a homography and a Thin-plate Spline (TPS) under the boundary constraint for discontinuity and hole-free image stitching. Our experiments show the favorable aligns and the competitive computational costs of REwarp compared to the existing stitching methods. Our source code is available at https://github.com/minshu-kim/REwarp.

count=2
* DPPMask: Masked Image Modeling With Determinantal Point Processes
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Xu_DPPMask_Masked_Image_Modeling_With_Determinantal_Point_Processes_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Xu_DPPMask_Masked_Image_Modeling_With_Determinantal_Point_Processes_WACV_2024_paper.pdf)]
    * Title: DPPMask: Masked Image Modeling With Determinantal Point Processes
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Junde Xu, Zikai Lin, Donghao Zhou, Yaodong Yang, Xiangyun Liao, Qiong Wang, Bian Wu, Guangyong Chen, Pheng-Ann Heng
    * Abstract: Masked Image Modeling (MIM) has achieved impressive representative performance with the aim of reconstructing randomly masked images. Despite the empirical success, most previous works have neglected the important fact that it is unreasonable to force the model to reconstruct something beyond recovery, such as those masked objects. In this work, we show that uniformly random masking widely used in previous works unavoidably loses some key objects and changes original semantic information, resulting in a misalignment problem and hurting the representative learning eventually. To address this issue, we augment MIM with a new masking strategy namely the DPPMask by substituting the random process with Determinantal Point Process (DPPs) to reduce the semantic change of the image after masking. Our method is simple yet effective and requires no extra learnable parameters when implemented within various frameworks. In particular, we evaluate our method on two representative MIM frameworks, MAE and iBOT. We show that DPPMask surpassed random sampling under both lower and higher masking ratios, indicating that DPPMask makes the reconstruction task more reasonable. We further test our method on the background challenge and multi-class classification tasks, showing that our method is more robust at various tasks.

count=2
* Multi-source Domain Adaptation for Semantic Segmentation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/db9ad56c71619aeed9723314d1456037-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/db9ad56c71619aeed9723314d1456037-Paper.pdf)]
    * Title: Multi-source Domain Adaptation for Semantic Segmentation
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Sicheng Zhao, Bo Li, Xiangyu Yue, Yang Gu, Pengfei Xu, Runbo Hu, Hua Chai, Kurt Keutzer
    * Abstract: Simulation-to-real domain adaptation for semantic segmentation has been actively studied for various applications such as autonomous driving. Existing methods mainly focus on a single-source setting, which cannot easily handle a more practical scenario of multiple sources with different distributions. In this paper, we propose to investigate multi-source domain adaptation for semantic segmentation. Specifically, we design a novel framework, termed Multi-source Adversarial Domain Aggregation Network (MADAN), which can be trained in an end-to-end manner. First, we generate an adapted domain for each source with dynamic semantic consistency while aligning at the pixel-level cycle-consistently towards the target. Second, we propose sub-domain aggregation discriminator and cross-domain cycle discriminator to make different adapted domains more closely aggregated. Finally, feature-level alignment is performed between the aggregated domain and target domain while training the segmentation network. Extensive experiments from synthetic GTA and SYNTHIA to real Cityscapes and BDDS datasets demonstrate that the proposed MADAN model outperforms state-of-the-art approaches. Our source code is released at: https://github.com/Luodian/MADAN.

count=1
* Enhancing Anchor-based Weakly Supervised Referring Expression Comprehension with Cross-Modality Attention
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Chu_Enhancing_Anchor-based_Weakly_Supervised_Referring_Expression_Comprehension_with_Cross-Modality_Attention_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Chu_Enhancing_Anchor-based_Weakly_Supervised_Referring_Expression_Comprehension_with_Cross-Modality_Attention_ACCV_2024_paper.pdf)]
    * Title: Enhancing Anchor-based Weakly Supervised Referring Expression Comprehension with Cross-Modality Attention
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Ting-Yu Chu, Yong-Xiang Lin, Ching-Chun Huang, Kai-Lung Hua
    * Abstract: Weakly supervised Referring Expression Comprehension (REC) tackles the challenge of identifying specific regions in an image based on textual descriptions without predefined mappings between the text and target objects during training. The primary obstacle lies in the misalignment between visual and textual features, often resulting in inaccurate bounding box predictions. To address this, we propose a novel cross-modality attention module (CMA) module that enhances the discriminative power of grid features and improves localization accuracy by harmonizing textual and visual features. To handle the noise from incorrect labels common in weak supervision, we also introduce a false negative suppression mechanism that uses intra-modal similarities as soft supervision signals. Extensive experiments conducted on four REC benchmark datasets: RefCOCO, RefCOCO+, RefCOCOg, and ReferItGame. Our results show that our model consistently outperforms state-of-the-art methods in accuracy and generalizability.

count=1
* Intrinsic Scene Properties from a Single RGB-D Image
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Barron_Intrinsic_Scene_Properties_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Barron_Intrinsic_Scene_Properties_2013_CVPR_paper.pdf)]
    * Title: Intrinsic Scene Properties from a Single RGB-D Image
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Jonathan T. Barron, Jitendra Malik
    * Abstract: In this paper we extend the "shape, illumination and reflectance from shading" (SIRFS) model [3, 4], which recovers intrinsic scene properties from a single image. Though SIRFS performs well on images of segmented objects, it performs poorly on images of natural scenes, which contain occlusion and spatially-varying illumination. We therefore present Scene-SIRFS, a generalization of SIRFS in which we have a mixture of shapes and a mixture of illuminations, and those mixture components are embedded in a "soft" segmentation of the input image. We additionally use the noisy depth maps provided by RGB-D sensors (in this case, the Kinect) to improve shape estimation. Our model takes as input a single RGB-D image and produces as output an improved depth map, a set of surface normals, a reflectance image, a shading image, and a spatially varying model of illumination. The output of our model can be used for graphics applications, or for any application involving RGB-D images.

count=1
* In Defense of Sparsity Based Face Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Deng_In_Defense_of_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Deng_In_Defense_of_2013_CVPR_paper.pdf)]
    * Title: In Defense of Sparsity Based Face Recognition
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Weihong Deng, Jiani Hu, Jun Guo
    * Abstract: The success of sparse representation based classification (SRC) has largely boosted the research of sparsity based face recognition in recent years. A prevailing view is that the sparsity based face recognition performs well only when the training images have been carefully controlled and the number of samples per class is sufficiently large. This paper challenges the prevailing view by proposing a "prototype plus variation" representation model for sparsity based face recognition. Based on the new model, a Superposed SRC (SSRC), in which the dictionary is assembled by the class centroids and the sample-to-centroid differences, leads to a substantial improvement on SRC. The experiments results on AR, FERET and FRGC databases validate that, if the proposed prototype plus variation representation model is applied, sparse coding plays a crucial role in face recognition, and performs well even when the dictionary bases are collected under uncontrolled conditions and only a single sample per classes is available.

count=1
* Pose from Flow and Flow from Pose
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Fragkiadaki_Pose_from_Flow_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Fragkiadaki_Pose_from_Flow_2013_CVPR_paper.pdf)]
    * Title: Pose from Flow and Flow from Pose
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Katerina Fragkiadaki, Han Hu, Jianbo Shi
    * Abstract: Human pose detectors, although successful in localising faces and torsos of people, often fail with lower arms. Motion estimation is often inaccurate under fast movements of body parts. We build a segmentation-detection algorithm that mediates the information between body parts recognition, and multi-frame motion grouping to improve both pose detection and tracking. Motion of body parts, though not accurate, is often sufficient to segment them from their backgrounds. Such segmentations are crucial for extracting hard to detect body parts out of their interior body clutter. By matching these segments to exemplars we obtain pose labeled body segments. The pose labeled segments and corresponding articulated joints are used to improve the motion flow fields by proposing kinematically constrained affine displacements on body parts. The pose-based articulated motion model is shown to handle large limb rotations and displacements. Our algorithm can detect people under rare poses, frequently missed by pose detectors, showing the benefits of jointly reasoning about pose, segmentation and motion in videos.

count=1
* HDR Deghosting: How to Deal with Saturation?
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Hu_HDR_Deghosting_How_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Hu_HDR_Deghosting_How_2013_CVPR_paper.pdf)]
    * Title: HDR Deghosting: How to Deal with Saturation?
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Jun Hu, Orazio Gallo, Kari Pulli, Xiaobai Sun
    * Abstract: We present a novel method for aligning images in an HDR (high-dynamic-range) image stack to produce a new exposure stack where all the images are aligned and appear as if they were taken simultaneously, even in the case of highly dynamic scenes. Our method produces plausible results even where the image used as a reference is either too dark or bright to allow for an accurate registration.

count=1
* Cloud Motion as a Calibration Cue
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Jacobs_Cloud_Motion_as_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Jacobs_Cloud_Motion_as_2013_CVPR_paper.pdf)]
    * Title: Cloud Motion as a Calibration Cue
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Nathan Jacobs, Mohammad T. Islam, Scott Workman
    * Abstract: We propose cloud motion as a natural scene cue that enables geometric calibration of static outdoor cameras. This work introduces several new methods that use observations of an outdoor scene over days and weeks to estimate radial distortion, focal length and geo-orientation. Cloud-based cues provide strong constraints and are an important alternative to methods that require specific forms of static scene geometry or clear sky conditions. Our method makes simple assumptions about cloud motion and builds upon previous work on motion-based and line-based calibration. We show results on real scenes that highlight the effectiveness of our proposed methods.

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
* Online Object Tracking: A Benchmark
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Wu_Online_Object_Tracking_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Wu_Online_Object_Tracking_2013_CVPR_paper.pdf)]
    * Title: Online Object Tracking: A Benchmark
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Yi Wu, Jongwoo Lim, Ming-Hsuan Yang
    * Abstract: Object tracking is one of the most important components in numerous applications of computer vision. While much progress has been made in recent years with efforts on sharing code and datasets, it is of great importance to develop a library and benchmark to gauge the state of the art. After briefly reviewing recent advances of online object tracking, we carry out large scale experiments with various evaluation criteria to understand how these algorithms perform. The test image sequences are annotated with different attributes for performance evaluation and analysis. By analyzing quantitative results, we identify effective approaches for robust tracking and provide potential future research directions in this field.

count=1
* Towards Pose Robust Face Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Yi_Towards_Pose_Robust_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Yi_Towards_Pose_Robust_2013_CVPR_paper.pdf)]
    * Title: Towards Pose Robust Face Recognition
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Dong Yi, Zhen Lei, Stan Z. Li
    * Abstract: Most existing pose robust methods are too computational complex to meet practical applications and their performance under unconstrained environments are rarely evaluated. In this paper, we propose a novel method for pose robust face recognition towards practical applications, which is fast, pose robust and can work well under unconstrained environments. Firstly, a 3D deformable model is built and a fast 3D model fitting algorithm is proposed to estimate the pose of face image. Secondly, a group of Gabor filters are transformed according to the pose and shape of face image for feature extraction. Finally, PCA is applied on the pose adaptive Gabor features to remove the redundances and Cosine metric is used to evaluate the similarity. The proposed method has three advantages: (1) The pose correction is applied in the filter space rather than image space, which makes our method less affected by the precision of the 3D model; (2) By combining the holistic pose transformation and local Gabor filtering, the final feature is robust to pose and other negative factors in face recognition; (3) The 3D structure and facial symmetry are successfully used to deal with self-occlusion. Extensive experiments on FERET and PIE show the proposed method outperforms state-ofthe-art methods significantly, meanwhile, the method works well on LFW.

count=1
* Robust Separation of Reflection from Multiple Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Guo_Robust_Separation_of_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Guo_Robust_Separation_of_2014_CVPR_paper.pdf)]
    * Title: Robust Separation of Reflection from Multiple Images
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Xiaojie Guo, Xiaochun Cao, Yi Ma
    * Abstract: When one records a video/image sequence through a transparent medium (e.g. glass), the image is often a superposition of a transmitted layer (scene behind the medium) and a reflected layer. Recovering the two layers from such images seems to be a highly ill-posed problem since the number of unknowns to recover is twice as many as the given measurements. In this paper, we propose a robust method to separate these two layers from multiple images, which exploits the correlation of the transmitted layer across multiple images, and the sparsity and independence of the gradient fields of the two layers. A novel Augmented Lagrangian Multiplier based algorithm is designed to efficiently and effectively solve the decomposition problem. The experimental results on both simulated and real data demonstrate the superior performance of the proposed method over the state of the arts, in terms of accuracy and simplicity.

count=1
* Color Transfer Using Probabilistic Moving Least Squares
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Hwang_Color_Transfer_Using_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Hwang_Color_Transfer_Using_2014_CVPR_paper.pdf)]
    * Title: Color Transfer Using Probabilistic Moving Least Squares
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Youngbae Hwang, Joon-Young Lee, In So Kweon, Seon Joo Kim
    * Abstract: This paper introduces a new color transfer method which is a process of transferring color of an image to match the color of another image of the same scene. The color of a scene may vary from image to image because the photographs are taken at different times, with different cameras, and under different camera settings. To solve for a full nonlinear and nonparametric color mapping in the 3D RGB color space, we propose a scattered point interpolation scheme using moving least squares and strengthen it with a probabilistic modeling of the color transfer in the 3D color space to deal with mis-alignments and noise. Experiments show the effectiveness of our method over previous color transfer methods both quantitatively and qualitatively. In addition, our framework can be applied for various instances of color transfer such as transferring color between different camera models, camera settings, and illumination conditions, as well as for video color transfers.

count=1
* Learning-Based Atlas Selection for Multiple-Atlas Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Sanroma_Learning-Based_Atlas_Selection_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Sanroma_Learning-Based_Atlas_Selection_2014_CVPR_paper.pdf)]
    * Title: Learning-Based Atlas Selection for Multiple-Atlas Segmentation
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Gerard Sanroma, Guorong Wu, Yaozong Gao, Dinggang Shen
    * Abstract: Recently, multi-atlas segmentation (MAS) has achieved a great success in the medical imaging area. The key assumption of MAS is that multiple atlases encompass richer anatomical variability than a single atlas. Therefore, we can label the target image more accurately by mapping the label information from the appropriate atlas images that have the most similar structures. The problem of atlas selection, however, still remains unexplored. Current state-of-the-art MAS methods rely on image similarity to select a set of atlases. Unfortunately, this heuristic criterion is not necessarily related to segmentation performance and, thus may undermine segmentation results. To solve this simple but critical problem, we propose a learning-based atlas selection method to pick up the best atlases that would eventually lead to more accurate image segmentation. Our idea is to learn the relationship between the pairwise appearance of observed instances (a pair of atlas and target images) and their final labeling performance (in terms of Dice ratio). In this way, we can select the best atlases according to their expected labeling accuracy. It is worth noting that our atlas selection method is general enough to be integrated with existing MAS methods. As is shown in the experiments, we achieve significant improvement after we integrate our method with 3 widely used MAS methods on ADNI and LONI LPBA40 datasets.

count=1
* High Resolution 3D Shape Texture from Multiple Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Tsiminaki_High_Resolution_3D_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Tsiminaki_High_Resolution_3D_2014_CVPR_paper.pdf)]
    * Title: High Resolution 3D Shape Texture from Multiple Videos
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Vagia Tsiminaki, Jean-Sebastien Franco, Edmond Boyer
    * Abstract: We examine the problem of retrieving high resolution textures of objects observed in multiple videos under small object deformations. In the monocular case, the data redundancy necessary to reconstruct a high-resolution image stems from temporal accumulation. This has been vastly explored and is known as image super-resolution. On the other hand, a handful of methods have considered the texture of a static 3D object observed from several cameras, where the data redundancy is obtained through the different viewpoints. We introduce a unified framework to leverage both possibilities for the estimation of an object's high resolution texture. This framework uniformly deals with any related geometric variability introduced by the acquisition chain or by the evolution over time. To this goal we use 2D warps for all viewpoints and all temporal frames and a linear image formation model from texture to image space. Despite its simplicity, the method is able to successfully handle different views over space and time. As shown experimentally, it demonstrates the interest of temporal information to improve the texture quality. Additionally, we also show that our method outperforms state of the art multi-view super-resolution methods existing for the static case.

count=1
* Human Action Recognition by Representing 3D Skeletons as Points in a Lie Group
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Vemulapalli_Human_Action_Recognition_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Vemulapalli_Human_Action_Recognition_2014_CVPR_paper.pdf)]
    * Title: Human Action Recognition by Representing 3D Skeletons as Points in a Lie Group
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Raviteja Vemulapalli, Felipe Arrate, Rama Chellappa
    * Abstract: Recently introduced cost-effective depth sensors coupled with the real-time skeleton estimation algorithm of Shotton et al. [16] have generated a renewed interest in skeleton-based human action recognition. Most of the existing skeleton-based approaches use either the joint locations or the joint angles to represent a human skeleton. In this paper, we propose a new skeletal representation that explicitly models the 3D geometric relationships between various body parts using rotations and translations in 3D space. Since 3D rigid body motions are members of the special Euclidean group SE(3), the proposed skeletal representation lies in the Lie group SE(3)×. . .×SE(3), which is a curved manifold. Using the proposed representation, human actions can be modeled as curves in this Lie group. Since classification of curves in this Lie group is not an easy task, we map the action curves from the Lie group to its Lie algebra, which is a vector space. We then perform classification using a combination of dynamic time warping, Fourier temporal pyramid representation and linear SVM. Experimental results on three action datasets show that the proposed representation performs better than many existing skeletal representations. The proposed approach also outperforms various state-of-the-art skeleton-based human action recognition approaches.

count=1
* Parallax-tolerant Image Stitching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Zhang_Parallax-tolerant_Image_Stitching_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Zhang_Parallax-tolerant_Image_Stitching_2014_CVPR_paper.pdf)]
    * Title: Parallax-tolerant Image Stitching
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Fan Zhang, Feng Liu
    * Abstract: Parallax handling is a challenging task for image stitching. This paper presents a local stitching method to handle parallax based on the observation that input images do not need to be perfectly aligned over the whole overlapping region for stitching. Instead, they only need to be aligned in a way that there exists a local region where they can be seamlessly blended together. We adopt a hybrid alignment model that combines homography and content-preserving warping to provide flexibility for handling parallax and avoiding objectionable local distortion. We then develop an efficient randomized algorithm to search for a homography, which, combined with content-preserving warping, allows for optimal stitching. We predict how well a homography enables plausible stitching by finding a plausible seam and using the seam cost as the quality metric. We develop a seam finding method that estimates a plausible seam from only roughly aligned images by considering both geometric alignment and image content. We then pre-align input images using the optimal homography and further use content-preserving warping to locally refine the alignment. We finally compose aligned images together using a standard seam-cutting algorithm and a multi-band blending algorithm. Our experiments show that our method can effectively stitch images with large parallax that are difficult for existing methods.

count=1
* Elastic Functional Coding of Human Actions: From Vector-Fields to Latent Variables
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Anirudh_Elastic_Functional_Coding_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Anirudh_Elastic_Functional_Coding_2015_CVPR_paper.pdf)]
    * Title: Elastic Functional Coding of Human Actions: From Vector-Fields to Latent Variables
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Rushil Anirudh, Pavan Turaga, Jingyong Su, Anuj Srivastava
    * Abstract: Human activities observed from visual sensors often give rise to a sequence of smoothly varying features. In many cases, the space of features can be formally defined as a manifold, where the action becomes a trajectory on the manifold. Such trajectories are high dimensional in addition to being non-linear, which can severely limit computations on them. We also argue that by their nature, human actions themselves lie on a much lower dimensional manifold compared to the high dimensional feature space. Learning an accurate low dimensional embedding for actions could have a huge impact in the areas of efficient search and retrieval, visualization, learning, and recognition. Traditional manifold learning addresses this problem for static points in R^n, but its extension to trajectories on Riemannian manifolds is non-trivial and has remained unexplored. The challenge arises due to the inherent non-linearity, and temporal variability that can significantly distort the distance metric between trajectories. To address these issues we use the transport square-root velocity function (TSRVF) space, a recently proposed representation that provides a metric which has favorable theoretical properties such as invariance to group action. We propose to learn the low dimensional embedding with a manifold functional variant of principal component analysis (mfPCA). We show that mfPCA effectively models the manifold trajectories in several applications such as action recognition, clustering and diverse sequence sampling while reducing the dimensionality by a factor of ~250x. The mfPCA features can also be reconstructed back to the original manifold to allow for easy visualization of the latent variable space.

count=1
* Fast Bilateral-Space Stereo for Synthetic Defocus
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Barron_Fast_Bilateral-Space_Stereo_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Barron_Fast_Bilateral-Space_Stereo_2015_CVPR_paper.pdf)]
    * Title: Fast Bilateral-Space Stereo for Synthetic Defocus
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Jonathan T. Barron, Andrew Adams, YiChang Shih, Carlos Hernandez
    * Abstract: Given a stereo pair it is possible to recover a depth map and use that depth to render a synthetically defocused image. Though stereo algorithms are well-studied, rarely are those algorithms considered solely in the context of producing these defocused renderings. In this paper we present a technique for efficiently producing disparity maps using a novel optimization framework in which inference is performed in "bilateral-space". Our approach produces higher-quality "defocus" results than other stereo algorithms while also being 10-100 times faster than comparable techniques.

count=1
* Robust Reconstruction of Indoor Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Choi_Robust_Reconstruction_of_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Choi_Robust_Reconstruction_of_2015_CVPR_paper.pdf)]
    * Title: Robust Reconstruction of Indoor Scenes
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Sungjoon Choi, Qian-Yi Zhou, Vladlen Koltun
    * Abstract: We present an approach to indoor scene reconstruction from RGB-D video. The key idea is to combine geometric registration of scene fragments with robust global optimization based on line processes. Geometric registration is error-prone due to sensor noise, which leads to aliasing of geometric detail and inability to disambiguate different surfaces in the scene. The presented optimization approach disables erroneous geometric alignments even when they significantly outnumber correct ones. Experimental results demonstrate that the presented approach substantially increases the accuracy of reconstructed scene models.

count=1
* Burst Deblurring: Removing Camera Shake Through Fourier Burst Accumulation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Delbracio_Burst_Deblurring_Removing_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Delbracio_Burst_Deblurring_Removing_2015_CVPR_paper.pdf)]
    * Title: Burst Deblurring: Removing Camera Shake Through Fourier Burst Accumulation
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Mauricio Delbracio, Guillermo Sapiro
    * Abstract: Numerous recent approaches attempt to remove image blur due to camera shake, either with one or multiple input images, by explicitly solving an inverse and inherently ill-posed deconvolution problem. If the photographer takes a burst of images, a modality available in virtually all modern digital cameras, we show that it is possible to combine them to get a clean sharp version. This is done without explicitly solving any blur estimation and subsequent inverse problem. The proposed algorithm is strikingly simple: it performs a weighted average in the Fourier domain, with weights depending on the Fourier spectrum magnitude. The method's rationale is that camera shake has a random nature and therefore each image in the burst is generally blurred differently. Experiments with real camera data show that the proposed Fourier Burst Accumulation algorithm achieves state-of-the-art results an order of magnitude faster, with simplicity for on-board implementation on camera phones.

count=1
* Learning to Segment Moving Objects in Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Fragkiadaki_Learning_to_Segment_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Fragkiadaki_Learning_to_Segment_2015_CVPR_paper.pdf)]
    * Title: Learning to Segment Moving Objects in Videos
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Katerina Fragkiadaki, Pablo Arbelaez, Panna Felsen, Jitendra Malik
    * Abstract: We segment moving objects in videos by ranking spatio-temporal segment proposals according to ``moving objectness''; how likely they are to contain a moving object. In each video frame, we compute segment proposals using multiple figure-ground segmentations on per frame motion boundaries. We rank them with a Moving Objectness Detector trained on image and motion fields to detect moving objects and discard over/under segmentations or background parts of the scene. We extend the top ranked segments into spatio-temporal tubes using random walkers on motion affinities of dense point trajectories. Our final tube ranking consistently outperforms previous segmentation methods in the two largest video segmentation benchmarks currently available, for any number of proposals. Further, our per frame moving object proposals increase the detection rate up to 7\% over previous state-of-the-art static proposal methods.

count=1
* A MRF Shape Prior for Facade Parsing With Occlusions
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Kozinski_A_MRF_Shape_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Kozinski_A_MRF_Shape_2015_CVPR_paper.pdf)]
    * Title: A MRF Shape Prior for Facade Parsing With Occlusions
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Mateusz Kozinski, Raghudeep Gadde, Sergey Zagoruyko, Guillaume Obozinski, Renaud Marlet
    * Abstract: We present a new shape prior formalism for segmentation of rectified facade images. It combines the simplicity of split grammars with unprecedented expressive power: the capability of encoding simultaneous alignment in two dimensions, facade occlusions and irregular boundaries between facade elements. Our method simultaneously segments the visible and occluding objects and recovers the structure of the occluded facade. We formulate the task of finding the most likely image segmentation conforming to a prior of the proposed form as a MAP-MRF problem over the standard 4-connected pixel grid with hard constraints on the classes of neighboring pixels, and propose an efficient optimization algorithm for solving it. We demonstrate state of the art results on a number of facade segmentation datasets.

count=1
* Beyond Spatial Pooling: Fine-Grained Representation Learning in Multiple Domains
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Li_Beyond_Spatial_Pooling_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_Beyond_Spatial_Pooling_2015_CVPR_paper.pdf)]
    * Title: Beyond Spatial Pooling: Fine-Grained Representation Learning in Multiple Domains
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Chi Li, Austin Reiter, Gregory D. Hager
    * Abstract: Object recognition systems have shown great progress over recent years. However, creating object representations that are robust to changes in viewpoint while capturing local visual details continues to be a challenge. In particular, recent convolutional architectures employ spatial pooling to achieve scale and shift invariances, but they are still sensitive to out-of-plane rotations. In this paper, we formulate a probabilistic framework for analyzing the performance of pooling. This framework suggests two directions for improvement. First, we apply multiple scales of filters coupled with different pooling granularities, and second we make use of color as an additional pooling domain, thereby reducing the sensitivity to spatial deformations. We evaluate our algorithm on the object instance recognition task using two independent publicly available RGB-D datasets, and demonstrate significant improvements over the current state-of-the-art. In addition, we present a new dataset for industrial objects to further validate the effectiveness of our approach versus other state-of-the-art approaches for object recognition using RGB-D data.

count=1
* Depth Image Enhancement Using Local Tangent Plane Approximations
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Matsuo_Depth_Image_Enhancement_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Matsuo_Depth_Image_Enhancement_2015_CVPR_paper.pdf)]
    * Title: Depth Image Enhancement Using Local Tangent Plane Approximations
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Kiyoshi Matsuo, Yoshimitsu Aoki
    * Abstract: This paper describes a depth image enhancement method for consumer RGB-D cameras. Most existing methods use the pixel-coordinates of the aligned color image. Because the image plane generally has no relationship to the measured surfaces, the global coordinate system is not suitable to handle their local geometries. To improve enhancement accuracy, we use local tangent planes as local coordinates for the measured surfaces. Our method is composed of two steps, a calculation of the local tangents and surface reconstruction. To accurately estimate the local tangents, we propose a color heuristic calculation and an orientation correction using their positional relationships. Additionally, we propose a surface reconstruction method by ray-tracing to local tangents. In our method, accurate depth image enhancement is achieved by using the local geometries approximated by the local tangents. We demonstrate the effectiveness of our method using synthetic and real sensor data. Our method has a high completion rate and achieves the lowest errors in noisy cases when compared with existing techniques.

count=1
* From Single Image Query to Detailed 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Schonberger_From_Single_Image_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Schonberger_From_Single_Image_2015_CVPR_paper.pdf)]
    * Title: From Single Image Query to Detailed 3D Reconstruction
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Johannes L. Schonberger, Filip Radenovic, Ondrej Chum, Jan-Michael Frahm
    * Abstract: Structure-from-Motion for unordered image collections has significantly advanced in scale over the last decade. This impressive progress can be in part attributed to the introduction of efficient retrieval methods for those systems. While this boosts scalability, it also limits the amount of detail that the large-scale reconstruction systems are able to produce. In this paper, we propose a joint reconstruction and retrieval system that maintains the scalability of large-scale Structure-from-Motion systems while also recovering the often lost ability of reconstructing fine details of the scene. We demonstrate our proposed method on a large-scale dataset of 7.4 million images downloaded from the Internet.

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
* Semantic Alignment of LiDAR Data at City Scale
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Yu_Semantic_Alignment_of_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Yu_Semantic_Alignment_of_2015_CVPR_paper.pdf)]
    * Title: Semantic Alignment of LiDAR Data at City Scale
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Fisher Yu, Jianxiong Xiao, Thomas Funkhouser
    * Abstract: This paper describes an automatic algorithm for global alignment of LiDAR data collected with Google Street View cars in urban environments. The problem is challenging because global pose estimation techniques (GPS) do not work well in city environments with tall buildings, and local tracking techniques (integration of inertial sensors, structure-from-motion, etc.) provide solutions that drift over long ranges, leading to solutions where data collected over wide ranges is warped and misaligned by many meters. Our approach to address this problem is to extract ``semantic features'' with object detectors (e.g., for facades, poles, cars, etc.) that can be matched robustly at different scales, and thus are selected for different iterations of an ICP algorithm. We have implemented an all-to-all, non-rigid, global alignment based on this idea that provides better results than alternatives during experiments with data from large regions of New York, San Francisco, Paris, and Rome.

count=1
* Intra-Frame Deblurring by Leveraging Inter-Frame Camera Motion
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Zhang_Intra-Frame_Deblurring_by_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Zhang_Intra-Frame_Deblurring_by_2015_CVPR_paper.pdf)]
    * Title: Intra-Frame Deblurring by Leveraging Inter-Frame Camera Motion
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Haichao Zhang, Jianchao Yang
    * Abstract: Camera motion introduces motion blur, degrading the quality of video. A video deblurring method is proposed based on two observations: (i) camera motion within capture of each individual frame leads to motion blur; (ii) camera motion between frames yields inter-frame mis-alignment that can be exploited for blur removal. The proposed method effectively leverages the information distributed across multiple video frames due to camera motion, jointly estimating the motion between consecutive frames and blur within each frame. This joint analysis is crucial for achieving effective restoration by leveraging temporal information. Extensive experiments are carried out on synthetic data as well as real-world blurry videos. Comparisons with several state-of-the-art methods verify the effectiveness of the proposed method.

count=1
* Layered Scene Decomposition via the Occlusion-CRF
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_Layered_Scene_Decomposition_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Layered_Scene_Decomposition_CVPR_2016_paper.pdf)]
    * Title: Layered Scene Decomposition via the Occlusion-CRF
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Chen Liu, Pushmeet Kohli, Yasutaka Furukawa
    * Abstract: This paper addresses the challenging problem of perceiving the hidden or occluded geometry of the scene depicted in any given RGBD image. Unlike other image labeling problems such as image segmentation where each pixel needs to be assigned a single label, layered decomposition requires us to assign multiple labels to pixels. We propose a novel "Occlusion-CRF" model that allows for the integration of sophisticated priors to regularize the solution space and enables the automatic inference of the layer decomposition. We use a generalization of the Fusion Move algorithm to perform Maximum a Posterior (MAP) inference on the model that can handle the large label sets needed to represent multiple surface assignments to each pixel. We have evaluated the proposed model and the inference algorithm on many RGBD images of cluttered indoor scenes. Our experiments show that not only is our model able to explain occlusions but it also enables automatic inpainting of occluded/invisible surfaces.

count=1
* Needle-Match: Reliable Patch Matching Under High Uncertainty
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Lotan_Needle-Match_Reliable_Patch_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lotan_Needle-Match_Reliable_Patch_CVPR_2016_paper.pdf)]
    * Title: Needle-Match: Reliable Patch Matching Under High Uncertainty
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Or Lotan, Michal Irani
    * Abstract: Reliable patch-matching forms the basis for many algorithms (super-resolution, denoising, inpainting, etc.) However, when the image quality deteriorates (by noise, blur or geometric distortions), the reliability of patch-matching deteriorates as well. Matched patches in the degraded image, do not necessarily imply similarity of the underlying patches in the (unknown) high-quality image. This restricts the applicability of patch-based methods. In this paper we present a patch representation called "Needle", which consists of small multi-scale versions of the patch and its immediate surrounding region. While the patch at the finest image scale is severely degraded, the degradation decreases dramatically in coarser needle scales, revealing reliable information for matching. We show that the Needle is robust to many types of image degradations, leads to matches faithful to the underlying high-quality patches, and to improvement in existing patch-based methods.

count=1
* D3: Deep Dual-Domain Based Fast Restoration of JPEG-Compressed Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Wang_D3_Deep_Dual-Domain_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Wang_D3_Deep_Dual-Domain_CVPR_2016_paper.pdf)]
    * Title: D3: Deep Dual-Domain Based Fast Restoration of JPEG-Compressed Images
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Zhangyang Wang, Ding Liu, Shiyu Chang, Qing Ling, Yingzhen Yang, Thomas S. Huang
    * Abstract: In this paper, we design a Deep Dual-Domain (D3) based fast restoration model to remove artifacts of JPEG compressed images. It leverages the large learning capacity of deep networks, as well as the problem-specific expertise that was hardly incorporated in the past design of deep architectures. For the latter, we take into consideration both the prior knowledge of the JPEG compression scheme, and the successful practice of the sparsity-based dual-domain approach. We further design the One-Step Sparse Inference (1-SI) module, as an efficient and light-weighted feed-forward approximation of sparse coding. Extensive experiments verify the superiority of the proposed D3 model over several state-of-the-art methods. Specifically, our best model is capable of outperforming the latest deep model for around 1 dB in PSNR, and is 30 times faster.

count=1
* ReD-SFA: Relation Discovery Based Slow Feature Analysis for Trajectory Clustering
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhang_ReD-SFA_Relation_Discovery_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_ReD-SFA_Relation_Discovery_CVPR_2016_paper.pdf)]
    * Title: ReD-SFA: Relation Discovery Based Slow Feature Analysis for Trajectory Clustering
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Zhang Zhang, Kaiqi Huang, Tieniu Tan, Peipei Yang, Jun Li
    * Abstract: For spectral embedding/clustering, it is still an open problem on how to construct an relation graph to reflect the intrinsic structures in data. In this paper, we proposed an approach, named Relation Discovery based Slow Feature Analysis (ReD-SFA), for feature learning and graph construction simultaneously. Given an initial graph with only a few nearest but most reliable pairwise relations, new reliable relations are discovered by an assumption of reliability preservation, i.e., the reliable relations will preserve their reliabilities in the learnt projection subspace. We formulate the idea as a cross entropy (CE) minimization problem to reduce the discrepancy between two Bernoulli distributions parameterized by the updated distances and the existing relation graph respectively. Furthermore, to overcome the imbalanced distribution of samples, a Boosting-like strategy is proposed to balance the discovered relations over all clusters. To evaluate the proposed method, extensive experiments are performed with various trajectory clustering tasks, including motion segmentation, time series clustering and crowd detection. The results demonstrate that ReD-SFA can discover reliable intra-cluster relations with high precision, and competitive clustering performance can be achieved in comparison with state-of-the-art.

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
* Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Teutsch_Robust_Detection_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Teutsch_Robust_Detection_of_CVPR_2016_paper.pdf)]
    * Title: Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Michael Teutsch, Michael Grinberg
    * Abstract: Multiple object tracking in Wide Area Motion Imagery (WAMI) data is usually based on initial detections coming from background subtraction or frame differencing. However, these methods are prone to produce split and merged detections. Appearance based vehicle detection can be an alternative but is not well-suited for WAMI data since classifier models are of weak discriminative power for vehicles in top view at low resolution. We introduce a moving vehicle detection algorithm that combines 2-frame differencing with a vehicle appearance model to improve object detection. Our main contributions are (1) integration of robust vehicle detection with split/merge handling and (2) estimation of assignment likelihoods between object hypotheses in consecutive frames using an appearance based similarity measure. Without using any prior knowledge, we achieve state-of-the-art detection rates and produce tracklets that considerably simplify the data association problem for multiple object tracking.

count=1
* Grouper: Optimizing Crowdsourced Face Annotations
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w4/html/Adams_Grouper_Optimizing_Crowdsourced_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w4/papers/Adams_Grouper_Optimizing_Crowdsourced_CVPR_2016_paper.pdf)]
    * Title: Grouper: Optimizing Crowdsourced Face Annotations
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Jocelyn C. Adams, Kristen C. Allen, Timothy Miller, Nathan D. Kalka, Anil K. Jain
    * Abstract: This study focuses on the problem of extracting consistent and accurate face bounding box annotations from crowdsourced workers. Aiming to provide benchmark datasets for facial recognition training and testing, we create a `gold standard' set against which consolidated face bounding box annotations can be evaluated. An evaluation methodology based on scores for several features of bounding box annotations is presented and is shown to predict consolidation performance using information gathered from crowdsourced annotations. Based on this foundation, we present "Grouper," a method leveraging density-based clustering to consolidate annotations by crowd workers. We demonstrate that the proposed consolidation scheme, which should be extensible to any number of region annotation consolidations, improves upon metadata released with the IARPA Janus Benchmark-A. Finally, we compare FR performance using the originally provided IJB-A annotations and Grouper and determine that similarity to the gold standard as measured by our evaluation metric does predict recognition performance.

count=1
* Dynamic FAUST: Registering Human Bodies in Motion
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Bogo_Dynamic_FAUST_Registering_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Bogo_Dynamic_FAUST_Registering_CVPR_2017_paper.pdf)]
    * Title: Dynamic FAUST: Registering Human Bodies in Motion
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Federica Bogo, Javier Romero, Gerard Pons-Moll, Michael J. Black
    * Abstract: While the ready availability of 3D scan data has influenced research throughout computer vision, less attention has focused on 4D data; that is 3D scans of moving non-rigid objects, captured over time. To be useful for vision research, such 4D scans need to be registered, or aligned, to a common topology. Consequently, extending mesh registration methods to 4D is important. Unfortunately, no ground-truth datasets are available for quantitative evaluation and comparison of 4D registration methods. To address this we create a novel dataset of high-resolution 4D scans of human subjects in motion, captured at 60 fps. We propose a new mesh registration method that uses both 3D geometry and texture information to register all scans in a sequence to a common reference topology. The approach exploits consistency in texture over both short and long time intervals and deals with temporal offsets between shape and texture capture. We show how using geometry alone results in significant errors in alignment when the motions are fast and non-rigid. We evaluate the accuracy of our registration and provide a dataset of 40,000 raw and aligned meshes. Dynamic FAUST extends the popular FAUST dataset to dynamic 4D data, and is available for research purposes at http://dfaust.is.tue.mpg.de.

count=1
* Deep Video Deblurring for Hand-Held Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Su_Deep_Video_Deblurring_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Su_Deep_Video_Deblurring_CVPR_2017_paper.pdf)]
    * Title: Deep Video Deblurring for Hand-Held Cameras
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Shuochen Su, Mauricio Delbracio, Jue Wang, Guillermo Sapiro, Wolfgang Heidrich, Oliver Wang
    * Abstract: Motion blur from camera shake is a major problem in videos captured by hand-held devices. Unlike single-image deblurring, video-based approaches can take advantage of the abundant information that exists across neighboring frames. As a result the best performing methods rely on the alignment of nearby frames. However, aligning images is a computationally expensive and fragile procedure, and methods that aggregate information must therefore be able to identify which regions have been accurately aligned and which have not, a task that requires high level scene understanding. In this work, we introduce a deep learning solution to video deblurring, where a CNN is trained end-to-end to learn how to accumulate information across frames. To train this network, we collected a dataset of real videos recorded with a high frame rate camera, which we use to generate synthetic motion blur for supervision. We show that the features learned from this dataset extend to deblurring motion blur that arises due to camera shake in a wide range of videos, and compare the quality of results to a number of other baselines.

count=1
* Joint Detection and Identification Feature Learning for Person Search
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Xiao_Joint_Detection_and_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xiao_Joint_Detection_and_CVPR_2017_paper.pdf)]
    * Title: Joint Detection and Identification Feature Learning for Person Search
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Tong Xiao, Shuang Li, Bochao Wang, Liang Lin, Xiaogang Wang
    * Abstract: Existing person re-identification benchmarks and methods mainly focus on matching cropped pedestrian images between queries and candidates. However, it is different from real-world scenarios where the annotations of pedestrian bounding boxes are unavailable and the target person needs to be searched from a gallery of whole scene images. To close the gap, we propose a new deep learning framework for person search. Instead of breaking it down into two separate tasks---pedestrian detection and person re-identification, we jointly handle both aspects in a single convolutional neural network. An Online Instance Matching (OIM) loss function is proposed to train the network effectively, which is scalable to datasets with numerous identities. To validate our approach, we collect and annotate a large-scale benchmark dataset for person search. It contains 18,184 images, 8,432 identities, and 96,143 pedestrian bounding boxes. Experiments show that our framework outperforms other separate approaches, and the proposed OIM loss function converges much faster and better than the conventional Softmax loss.

count=1
* Semantic Image Inpainting With Deep Generative Models
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf)]
    * Title: Semantic Image Inpainting With Deep Generative Models
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Raymond A. Yeh, Chen Chen, Teck Yian Lim, Alexander G. Schwing, Mark Hasegawa-Johnson, Minh N. Do
    * Abstract: Semantic image inpainting is a challenging task where large missing regions have to be filled based on the available visual data. Existing methods which extract information from only a single image generally produce unsatisfactory results due to the lack of high level context. In this paper, we propose a novel method for semantic image inpainting, which generates the missing content by conditioning on the available data. Given a trained generative model, we search for the closest encoding of the corrupted image in the latent image manifold using our context and prior losses. This encoding is then passed through the generative model to infer the missing content. In our method, inference is possible irrespective of how the missing content is structured, while the state-of-the-art learning based method requires specific information about the holes in the training phase. Experiments on three datasets show that our method successfully predicts information in large missing regions and achieves pixel-level photorealism, significantly outperforming the state-of-the-art methods.

count=1
* Super-FAN: Integrated Facial Landmark Localization and Super-Resolution of Real-World Low Resolution Faces in Arbitrary Poses With GANs
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Bulat_Super-FAN_Integrated_Facial_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bulat_Super-FAN_Integrated_Facial_CVPR_2018_paper.pdf)]
    * Title: Super-FAN: Integrated Facial Landmark Localization and Super-Resolution of Real-World Low Resolution Faces in Arbitrary Poses With GANs
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Adrian Bulat, Georgios Tzimiropoulos
    * Abstract: This paper addresses 2 challenging tasks: improving the quality of low resolution facial images and accurately locating the facial landmarks on such poor resolution images. To this end, we make the following 5 contributions: (a) we propose Super-FAN: the very first end-to-end system that addresses both tasks simultaneously, i.e. both improves face resolution and detects the facial landmarks. The novelty or Super-FAN lies in incorporating structural information in a GAN-based super-resolution algorithm via integrating a sub-network for face alignment through heatmap regression and optimizing a novel heatmap loss. (b) We illustrate the benefit of training the two networks jointly by reporting good results not only on frontal images (as in prior work) but on the whole spectrum of facial poses, and not only on synthetic low resolution images (as in prior work) but also on real-world images. (c) We improve upon the state-of-the-art in face super-resolution by proposing a new residual-based architecture. (d) Quantitatively, we show large improvement over the state-of-the-art for both face super-resolution and alignment. (e) Qualitatively, we show for the first time good results on real-world low resolution images.

count=1
* Re-Weighted Adversarial Adaptation Network for Unsupervised Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Re-Weighted_Adversarial_Adaptation_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Re-Weighted_Adversarial_Adaptation_CVPR_2018_paper.pdf)]
    * Title: Re-Weighted Adversarial Adaptation Network for Unsupervised Domain Adaptation
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Qingchao Chen, Yang Liu, Zhaowen Wang, Ian Wassell, Kevin Chetty
    * Abstract: Unsupervised Domain Adaptation (UDA) aims to transfer domain knowledge from existing well-defined tasks to new ones where labels are unavailable. In the real-world applications, as the domain (task) discrepancies are usually uncontrollable, it is significantly motivated to match the feature distributions even if the domain discrepancies are disparate. Additionally, as no label is available in the target domain, how to successfully adapt the classifier from the source to the target domain still remains an open question. In this paper, we propose the Re-weighted Adversarial Adaptation Network (RAAN) to reduce the feature distribution divergence and adapt the classifier when domain discrepancies are disparate. Specifically, to alleviate the need of common supports in matching the feature distribution, we choose to minimize optimal transport (OT) based Earth-Mover (EM) distance and reformulate it to a minimax objective function. Utilizing this, RAAN can be trained in an end-to-end and adversarial manner. To further adapt the classifier, we propose to match the label distribution and embed it into the adversarial training. Finally, after extensive evaluation of our method using UDA datasets of varying difficulty, RAAN achieved the state-of-the-art results and outperformed other methods by a large margin when the domain shifts are disparate.

count=1
* Burst Denoising With Kernel Prediction Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Mildenhall_Burst_Denoising_With_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mildenhall_Burst_Denoising_With_CVPR_2018_paper.pdf)]
    * Title: Burst Denoising With Kernel Prediction Networks
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Ben Mildenhall, Jonathan T. Barron, Jiawen Chen, Dillon Sharlet, Ren Ng, Robert Carroll
    * Abstract: We present a technique for jointly denoising bursts of images taken from a handheld camera. In particular, we propose a convolutional neural network architecture for predicting spatially varying kernels that can both align and denoise frames, a synthetic data generation approach based on a realistic noise formation model, and an optimization guided by an annealed loss function to avoid undesirable local minima. Our model matches or outperforms the state-of-the-art across a wide range of noise levels on both real and synthetic data.

count=1
* Learning Deep Sketch Abstraction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Muhammad_Learning_Deep_Sketch_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Muhammad_Learning_Deep_Sketch_CVPR_2018_paper.pdf)]
    * Title: Learning Deep Sketch Abstraction
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Umar Riaz Muhammad, Yongxin Yang, Yi-Zhe Song, Tao Xiang, Timothy M. Hospedales
    * Abstract: Human free-hand sketches have been studied in various contexts including sketch recognition, synthesis and fine-grained sketch-based image retrieval (FG-SBIR). A fundamental challenge for sketch analysis is to deal with drastically different human drawing styles, particularly in terms of abstraction level. In this work, we propose the first stroke-level sketch abstraction model based on the insight of sketch abstraction as a process of trading off between the recognizability of a sketch and the number of strokes used to draw it. Concretely, we train a model for abstract sketch generation through reinforcement learning of a stroke removal policy that learns to predict which strokes can be safely removed without affecting recognizability. We show that our abstraction model can be used for various sketch analysis tasks including: (1) modeling stroke saliency and understanding the decision of sketch recognition models, (2) synthesizing sketches of variable abstraction for a given category, or reference object instance in a photo, and (3) training a FG-SBIR model with photos only, bypassing the expensive photo-sketch pair collection step.

count=1
* Attentive Generative Adversarial Network for Raindrop Removal From a Single Image
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.pdf)]
    * Title: Attentive Generative Adversarial Network for Raindrop Removal From a Single Image
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Rui Qian, Robby T. Tan, Wenhan Yang, Jiajun Su, Jiaying Liu
    * Abstract: Raindrops adhered to a glass window or camera lens can severely hamper the visibility of a background scene and degrade an image considerably. In this paper, we address the problem by visually removing raindrops, and thus transforming a raindrop degraded image into a clean one. The problem is intractable, since first the regions occluded by raindrops are not given. Second, the information about the background scene of the occluded regions is completely lost for most part. To resolve the problem, we apply an attentive generative network using adversarial training. Our main idea is to inject visual attention into both the generative and discriminative networks. During the training, our visual attention learns about raindrop regions and their surroundings. Hence, by injecting this information, the generative network will pay more attention to the raindrop regions and the surrounding structures, and the discriminative network will be able to assess the local consistency of the restored regions. This injection of visual attention to both generative and discriminative networks is the main contribution of this paper. Our experiments show the effectiveness of our approach, which outperforms the state of the art methods quantitatively and qualitatively.

count=1
* Super-Resolving Very Low-Resolution Face Images With Supplementary Attributes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Super-Resolving_Very_Low-Resolution_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Super-Resolving_Very_Low-Resolution_CVPR_2018_paper.pdf)]
    * Title: Super-Resolving Very Low-Resolution Face Images With Supplementary Attributes
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Xin Yu, Basura Fernando, Richard Hartley, Fatih Porikli
    * Abstract: Given a tiny face image, conventional face hallucination methods aim to super-resolve its high-resolution (HR) counterpart by learning a mapping from an exemplar dataset. Since a low-resolution (LR) input patch may correspond to many HR candidate patches, this ambiguity may lead to erroneous HR facial details and thus distorts final results, such as gender reversal. An LR input contains low-frequency facial components of its HR version while its residual face image defined as the difference between the HR ground-truth and interpolated LR images contains the missing high-frequency facial details. We demonstrate that supplementing residual images or feature maps with facial attribute information can significantly reduce the ambiguity in face super-resolution. To explore this idea, we develop an attribute-embedded upsampling network, which consists of an upsampling network and a discriminative network. The upsampling network is composed of an autoencoder with skip-connections, which incorporates facial attribute vectors into the residual features of LR inputs at the bottleneck of the autoencoder and deconvolutional layers used for upsampling. The discriminative network is designed to examine whether super-resolved faces contain the desired attributes or not and then its loss is used for updating the upsampling network. In this manner, we can super-resolve tiny unaligned (16$	imes$16 pixels) face images with a large upscaling factor of 8$	imes$ while reducing the uncertainty of one-to-many mappings significantly. By conducting extensive evaluations on a large-scale dataset, we demonstrate that our method achieves superior face hallucination results and outperforms the state-of-the-art.

count=1
* Deep Mutual Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf)]
    * Title: Deep Mutual Learning
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu
    * Abstract: Model distillation is an effective and widely used technique to transfer knowledge from a teacher to a student network. The typical application is to transfer from a powerful large network or ensemble to a small network, in order to meet the low-memory or fast execution requirements. In this paper, we present a deep mutual learning (DML) strategy. Different from the one-way transfer between a static pre-defined teacher and a student in model distillation, with DML, an ensemble of students learn collaboratively and teach each other throughout the training process. Our experiments show that a variety of network architectures benefit from mutual learning and achieve compelling results on both category and instance recognition tasks. Surprisingly, it is revealed that no prior powerful teacher network is necessary -- mutual learning of a collection of simple student networks works, and moreover outperforms distillation from a more powerful yet static teacher.

count=1
* Three Dimensional Fluorescence Microscopy Image Synthesis and Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w44/html/Fu_Three_Dimensional_Fluorescence_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w44/Fu_Three_Dimensional_Fluorescence_CVPR_2018_paper.pdf)]
    * Title: Three Dimensional Fluorescence Microscopy Image Synthesis and Segmentation
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Chichen Fu, Soonam Lee, David Joon Ho, Shuo Han, Paul Salama, Kenneth W. Dunn, Edward J. Delp
    * Abstract: Advances in fluorescence microscopy enable acquisition of 3D image volumes with better image quality and deeper penetration into tissue. Segmentation is a required step to characterize and analyze biological structures in the images and recent 3D segmentation using deep learning has achieved promising results. One issue is that deep learning techniques require a large set of groundtruth data which is impractical to annotate manually for large 3D microscopy volumes. This paper describes a 3D deep learning nuclei segmentation method using synthetic 3D volumes for training. A set of synthetic volumes and the corresponding groundtruth are generated using spatially constrained cycle-consistent adversarial networks. Segmentation results demonstrate that our proposed method is capable of segmenting nuclei successfully for various data sets.

count=1
* Noise-Aware Unsupervised Deep Lidar-Stereo Fusion
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Cheng_Noise-Aware_Unsupervised_Deep_Lidar-Stereo_Fusion_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_Noise-Aware_Unsupervised_Deep_Lidar-Stereo_Fusion_CVPR_2019_paper.pdf)]
    * Title: Noise-Aware Unsupervised Deep Lidar-Stereo Fusion
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Xuelian Cheng,  Yiran Zhong,  Yuchao Dai,  Pan Ji,  Hongdong Li
    * Abstract: In this paper, we present LidarStereoNet, the first unsupervised Lidar-stereo fusion network, which can be trained in an end-to-end manner without the need of ground truth depth maps. By introducing a novel "Feedback Loop" to connect the network input with output, LidarStereoNet could tackle both noisy Lidar points and misalignment between sensors that have been ignored in existing Lidar-stereo fusion work. Besides, we propose to incorporate the piecewise planar model into the network learning to further constrain depths to conform to the underlying 3D geometry. Extensive quantitative and qualitative evaluations on both real and synthetic datasets demonstrate the superiority of our method, which outperforms state-of-the-art stereo matching, depth completion and Lidar-Stereo fusion approaches significantly.

count=1
* Modularized Textual Grounding for Counterfactual Resilience
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Fang_Modularized_Textual_Grounding_for_Counterfactual_Resilience_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fang_Modularized_Textual_Grounding_for_Counterfactual_Resilience_CVPR_2019_paper.pdf)]
    * Title: Modularized Textual Grounding for Counterfactual Resilience
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Zhiyuan Fang,  Shu Kong,  Charless Fowlkes,  Yezhou Yang
    * Abstract: Computer Vision applications often require a textual grounding module with precision, interpretability, and resilience to counterfactual inputs/queries. To achieve high grounding precision, current textual grounding methods heavily rely on large-scale training data with manual annotations at the pixel level. Such annotations are expensive to obtain and thus severely narrow the model's scope of real-world applications. Moreover, most of these methods sacrifice interpretability, generalizability, and they neglect the importance of being resilient to counterfactual inputs. To address these issues, we propose a visual grounding system which is 1) end-to-end trainable in a weakly supervised fashion with only image-level annotations, and 2) counterfactually resilient owing to the modular design. Specifically, we decompose textual descriptions into three levels: entity, semantic attribute, color information, and perform compositional grounding progressively. We validate our model through a series of experiments and demonstrate its improvement over the state-of-the-art methods. In particular, our model's performance not only surpasses other weakly/un-supervised methods and even approaches the strongly supervised ones, but also is interpretable for decision making and performs much better in face of counterfactual classes than all the others.

count=1
* Iterative Residual CNNs for Burst Photography Applications
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Kokkinos_Iterative_Residual_CNNs_for_Burst_Photography_Applications_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kokkinos_Iterative_Residual_CNNs_for_Burst_Photography_Applications_CVPR_2019_paper.pdf)]
    * Title: Iterative Residual CNNs for Burst Photography Applications
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Filippos Kokkinos,  Stamatis Lefkimmiatis
    * Abstract: Modern inexpensive imaging sensors suffer from inherent hardware constraints which often result in captured images of poor quality. Among the most common ways to deal with such limitations is to rely on burst photography, which nowadays acts as the backbone of all modern smartphone imaging applications. In this work, we focus on the fact that every frame of a burst sequence can be accurately described by a forward (physical) model. This, in turn, allows us to restore a single image of higher quality from a sequence of low-quality images as the solution of an optimization problem. Inspired by an extension of the gradient descent method that can handle non-smooth functions, namely the proximal gradient descent, and modern deep learning techniques, we propose a convolutional iterative network with a transparent architecture. Our network uses a burst of low-quality image frames and is able to produce an output of higher image quality recovering fine details which are not distinguishable in any of the original burst frames. We focus both on the burst photography pipeline as a whole, i.e., burst demosaicking and denoising, as well as on the traditional Gaussian denoising task. The developed method demonstrates consistent state-of-the art performance across the two tasks and as opposed to other recent deep learning approaches does not have any inherent restrictions either to the number of frames or their ordering.

count=1
* Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Ku_Monocular_3D_Object_Detection_Leveraging_Accurate_Proposals_and_Shape_Reconstruction_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ku_Monocular_3D_Object_Detection_Leveraging_Accurate_Proposals_and_Shape_Reconstruction_CVPR_2019_paper.pdf)]
    * Title: Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jason Ku,  Alex D. Pon,  Steven L. Waslander
    * Abstract: We present MonoPSR, a monocular 3D object detection method that leverages proposals and shape reconstruction. First, using the fundamental relations of a pinhole camera model, detections from a mature 2D object detector are used to generate a 3D proposal per object in a scene. The 3D location of these proposals prove to be quite accurate, which greatly reduces the difficulty of regressing the final 3D bounding box detection. Simultaneously, a point cloud is predicted in an object centered coordinate system to learn local scale and shape information. However, the key challenge is how to exploit shape information to guide 3D localization. As such, we devise aggregate losses, including a novel projection alignment loss, to jointly optimize these tasks in the neural network to improve 3D localization accuracy. We validate our method on the KITTI benchmark where we set new state-of-the-art results among published monocular methods, including the harder pedestrian and cyclist classes, while maintaining efficient run-time.

count=1
* Separate to Adapt: Open Set Domain Adaptation via Progressive Separation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Separate_to_Adapt_Open_Set_Domain_Adaptation_via_Progressive_Separation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Separate_to_Adapt_Open_Set_Domain_Adaptation_via_Progressive_Separation_CVPR_2019_paper.pdf)]
    * Title: Separate to Adapt: Open Set Domain Adaptation via Progressive Separation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Hong Liu,  Zhangjie Cao,  Mingsheng Long,  Jianmin Wang,  Qiang Yang
    * Abstract: Domain adaptation has become a resounding success in leveraging labeled data from a source domain to learn an accurate classifier for an unlabeled target domain. When deployed in the wild, the target domain usually contains unknown classes that are not observed in the source domain. Such setting is termed Open Set Domain Adaptation (OSDA). While several methods have been proposed to address OSDA, none of them takes into account the openness of the target domain, which is measured by the proportion of unknown classes in all target classes. Openness is a critical point in open set domain adaptation and exerts a significant impact on performance. In addition, current work aligns the entire target domain with the source domain without excluding unknown samples, which may give rise to negative transfer due to the mismatch between unknown and known classes. To this end, this paper presents Separate to Adapt (STA), an end-to-end approach to open set domain adaptation. The approach adopts a coarse-to-fine weighting mechanism to progressively separate the samples of unknown and known classes, and simultaneously weigh their importance on feature distribution alignment. Our approach allows openness-robust open set domain adaptation, which can be adaptive to a variety of openness in the target domain. We evaluate STA on several benchmark datasets of various openness levels. Results verify that STA significantly outperforms previous methods.

count=1
* Deep Rigid Instance Scene Flow
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Ma_Deep_Rigid_Instance_Scene_Flow_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ma_Deep_Rigid_Instance_Scene_Flow_CVPR_2019_paper.pdf)]
    * Title: Deep Rigid Instance Scene Flow
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Wei-Chiu Ma,  Shenlong Wang,  Rui Hu,  Yuwen Xiong,  Raquel Urtasun
    * Abstract: In this paper we tackle the problem of scene flow estimation in the context of self-driving. We leverage deep learning techniques as well as strong priors as in our application domain the motion of the scene can be composed by the motion of the robot and the 3D motion of the actors in the scene. We formulate the problem as energy minimization in a deep structured model, which can be solved efficiently in the GPU by unrolling a Gaussian-Newton solver. Our experiments in the challenging KITTI scene flow dataset show that we outperform the state-of-the-art by a very large margin, while being 800 times faster.

count=1
* Explicit Spatial Encoding for Deep Local Descriptors
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Mukundan_Explicit_Spatial_Encoding_for_Deep_Local_Descriptors_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mukundan_Explicit_Spatial_Encoding_for_Deep_Local_Descriptors_CVPR_2019_paper.pdf)]
    * Title: Explicit Spatial Encoding for Deep Local Descriptors
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Arun Mukundan,  Giorgos Tolias,  Ondrej Chum
    * Abstract: We propose a kernelized deep local-patch descriptor based on efficient match kernels of neural network activations. Response of each receptive field is encoded together with its spatial location using explicit feature maps. Two location parametrizations, Cartesian and polar, are used to provide robustness to a different types of canonical patch misalignment. Additionally, we analyze how the conventional architecture, i.e. a fully connected layer attached after the convolutional part, encodes responses in a spatially variant way. In contrary, explicit spatial encoding is used in our descriptor, whose potential applications are not limited to local-patches. We evaluate the descriptor on standard benchmarks. Both versions, encoding 32x32 or 64x64 patches, consistently outperform all other methods on all benchmarks. The number of parameters of the model is independent of the input patch resolution.

count=1
* Attention-Guided Network for Ghost-Free High Dynamic Range Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Yan_Attention-Guided_Network_for_Ghost-Free_High_Dynamic_Range_Imaging_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yan_Attention-Guided_Network_for_Ghost-Free_High_Dynamic_Range_Imaging_CVPR_2019_paper.pdf)]
    * Title: Attention-Guided Network for Ghost-Free High Dynamic Range Imaging
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Qingsen Yan,  Dong Gong,  Qinfeng Shi,  Anton van den Hengel,  Chunhua Shen,  Ian Reid,  Yanning Zhang
    * Abstract: Ghosting artifacts caused by moving objects or misalignments is a key challenge in high dynamic range (HDR) imaging for dynamic scenes. Previous methods first register the input low dynamic range (LDR) images using optical flow before merging them, which are error-prone and cause ghosts in results. A very recent work tries to bypass optical flows via a deep network with skip-connections, however, which still suffers from ghosting artifacts for severe movement. To avoid the ghosting from the source, we propose a novel attention-guided end-to-end deep neural network (AHDRNet) to produce high-quality ghost-free HDR images. Unlike previous methods directly stacking the LDR images or features for merging, we use attention modules to guide the merging according to the reference image. The attention modules automatically suppress undesired components caused by misalignments and saturation and enhance desirable fine details in the non-reference images. In addition to the attention model, we use dilated residual dense block (DRDB) to make full use of the hierarchical features and increase the receptive field for hallucinating the missing details. The proposed AHDRNet is a non-flow-based method, which can also avoid the artifacts generated by optical-flow estimation error. Experiments on different datasets show that the proposed AHDRNet can achieve state-of-the-art quantitative and qualitative results.

count=1
* Patch-Based Discriminative Feature Learning for Unsupervised Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Patch-Based_Discriminative_Feature_Learning_for_Unsupervised_Person_Re-Identification_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Patch-Based_Discriminative_Feature_Learning_for_Unsupervised_Person_Re-Identification_CVPR_2019_paper.pdf)]
    * Title: Patch-Based Discriminative Feature Learning for Unsupervised Person Re-Identification
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Qize Yang,  Hong-Xing Yu,  Ancong Wu,  Wei-Shi Zheng
    * Abstract: While discriminative local features have been shown effective in solving the person re-identification problem, they are limited to be trained on fully pairwise labelled data which is expensive to obtain. In this work, we overcome this problem by proposing a patch-based unsupervised learning framework in order to learn discriminative feature from patches instead of the whole images. The patch-based learning leverages similarity between patches to learn a discriminative model. Specifically, we develop a PatchNet to select patches from the feature map and learn discriminative features for these patches. To provide effective guidance for the PatchNet to learn discriminative patch feature on unlabeled datasets, we propose an unsupervised patch-based discriminative feature learning loss. In addition, we design an image-level feature learning loss to leverage all the patch features of the same image to serve as an image-level guidance for the PatchNet. Extensive experiments validate the superiority of our method for unsupervised person re-id. Our code is available at https://github.com/QizeYang/PAUL.

count=1
* Zoom to Learn, Learn to Zoom
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Zoom_to_Learn_Learn_to_Zoom_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Zoom_to_Learn_Learn_to_Zoom_CVPR_2019_paper.pdf)]
    * Title: Zoom to Learn, Learn to Zoom
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Xuaner Zhang,  Qifeng Chen,  Ren Ng,  Vladlen Koltun
    * Abstract: This paper shows that when applying machine learning to digital zoom, it is beneficial to operate on real, RAW sensor data. Existing learning-based super-resolution methods do not use real sensor data, instead operating on processed RGB images. We show that these approaches forfeit detail and accuracy that can be gained by operating on raw data, particularly when zooming in on distant objects. The key barrier to using real sensor data for training is that ground-truth high-resolution imagery is missing. We show how to obtain such ground-truth data via optical zoom and contribute a dataset, SR-RAW, for real-world computational zoom. We use SR-RAW to train a deep network with a novel contextual bilateral loss that is robust to mild misalignment between input and outputs images. The trained network achieves state-of-the-art performance in 4X and 8X computational zoom. We also show that synthesizing sensor data by resampling high-resolution RGB images is an oversimplified approximation of real sensor data and noise, resulting in worse image quality.

count=1
* Re-Identification With Consistent Attentive Siamese Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Re-Identification_With_Consistent_Attentive_Siamese_Networks_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Re-Identification_With_Consistent_Attentive_Siamese_Networks_CVPR_2019_paper.pdf)]
    * Title: Re-Identification With Consistent Attentive Siamese Networks
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Meng Zheng,  Srikrishna Karanam,  Ziyan Wu,  Richard J. Radke
    * Abstract: We propose a new deep architecture for person re-identification (re-id). While re-id has seen much recent progress, spatial localization and view-invariant representation learning for robust cross-view matching remain key, unsolved problems. We address these questions by means of a new attention-driven Siamese learning architecture, called the Consistent Attentive Siamese Network. Our key innovations compared to existing, competing methods include (a) a flexible framework design that produces attention with only identity labels as supervision, (b) explicit mechanisms to enforce attention consistency among images of the same person, and (c) a new Siamese framework that integrates attention and attention consistency, producing principled supervisory signals as well as the first mechanism that can explain the reasoning behind the Siamese framework's predictions. We conduct extensive evaluations on the CUHK03-NP, DukeMTMC-ReID, and Market-1501 datasets and report competitive performance.

count=1
* Two-Shot Spatially-Varying BRDF and Shape Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Boss_Two-Shot_Spatially-Varying_BRDF_and_Shape_Estimation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Boss_Two-Shot_Spatially-Varying_BRDF_and_Shape_Estimation_CVPR_2020_paper.pdf)]
    * Title: Two-Shot Spatially-Varying BRDF and Shape Estimation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Mark Boss,  Varun Jampani,  Kihwan Kim,  Hendrik P.A. Lensch,  Jan Kautz
    * Abstract: Capturing the shape and spatially-varying appearance (SVBRDF) of an object from images is a challenging task that has applications in both computer vision and graphics. Traditional optimization-based approaches often need a large number of images taken from multiple views in a controlled environment. Newer deep learning-based approaches require only a few input images, but the reconstruction quality is not on par with optimization techniques. We propose a novel deep learning architecture with a stage-wise estimation of shape and SVBRDF. The previous predictions guide each estimation, and a joint refinement network later refines both SVBRDF and shape. We follow a practical mobile image capture setting and use unaligned two-shot flash and no-flash images as input. Both our two-shot image capture and network inference can run on mobile hardware. We also create a large-scale synthetic training dataset with domain-randomized geometry and realistic materials. Extensive experiments on both synthetic and real-world datasets show that our network trained on a synthetic dataset can generalize well to real-world images. Comparisons with recent approaches demonstrate the superior performance of the proposed approach.

count=1
* Norm-Aware Embedding for Efficient Person Search
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Norm-Aware_Embedding_for_Efficient_Person_Search_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Norm-Aware_Embedding_for_Efficient_Person_Search_CVPR_2020_paper.pdf)]
    * Title: Norm-Aware Embedding for Efficient Person Search
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Di Chen,  Shanshan Zhang,  Jian Yang,  Bernt Schiele
    * Abstract: Person Search is a practically relevant task that aims to jointly solve Person Detection and Person Re-identification (re-ID). Specifically, it requires to find and locate all instances with the same identity as the query person in a set of panoramic gallery images. One major challenge comes from the contradictory goals of the two sub-tasks, i.e., person detection focuses on finding the commonness of all persons while person re-ID handles the differences among multiple identities. Therefore, it is crucial to reconcile the relationship between the two sub-tasks in a joint person search model. To this end, We present a novel approach called Norm-Aware Embedding to disentangle the person embedding into norm and angle for detection and re-ID respectively, allowing for both effective and efficient multi-task training. We further extend the proposal-level person embedding to pixel-level, whose discrimination ability is less affected by mis-alignment. We outperform other one-step methods by a large margin and achieve comparable performance to two-step methods on both CUHK-SYSU and PRW. Also, Our method is easy to train and resource-friendly, running at 12 fps on a single GPU.

count=1
* Say As You Wish: Fine-Grained Control of Image Caption Generation With Abstract Scene Graphs
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Say_As_You_Wish_Fine-Grained_Control_of_Image_Caption_Generation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Say_As_You_Wish_Fine-Grained_Control_of_Image_Caption_Generation_CVPR_2020_paper.pdf)]
    * Title: Say As You Wish: Fine-Grained Control of Image Caption Generation With Abstract Scene Graphs
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Shizhe Chen,  Qin Jin,  Peng Wang,  Qi Wu
    * Abstract: Humans are able to describe image contents with coarse to fine details as they wish. However, most image captioning models are intention-agnostic which cannot generate diverse descriptions according to different user intentions initiatively. In this work, we propose the Abstract Scene Graph (ASG) structure to represent user intention in fine-grained level and control what and how detailed the generated description should be. The ASG is a directed graph consisting of three types of abstract nodes (object, attribute, relationship) grounded in the image without any concrete semantic labels. Thus it is easy to obtain either manually or automatically. From the ASG, we propose a novel ASG2Caption model, which is able to recognise user intentions and semantics in the graph, and therefore generate desired captions following the graph structure. Our model achieves better controllability conditioning on ASGs than carefully designed baselines on both VisualGenome and MSCOCO datasets. It also significantly improves the caption diversity via automatically sampling diverse ASGs as control signals. Code will be released at https://github.com/cshizhe/asg2cap.

count=1
* Panoptic-Based Image Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Dundar_Panoptic-Based_Image_Synthesis_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dundar_Panoptic-Based_Image_Synthesis_CVPR_2020_paper.pdf)]
    * Title: Panoptic-Based Image Synthesis
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Aysegul Dundar,  Karan Sapra,  Guilin Liu,  Andrew Tao,  Bryan Catanzaro
    * Abstract: Conditional image synthesis for generating photorealistic images serves various applications for content editing to content generation. Previous conditional image synthesis algorithms mostly rely on semantic maps, and often fail in complex environments where multiple instances occlude each other. We propose a panoptic aware image synthesis network to generate high fidelity and photorealistic images conditioned on panoptic maps which unify semantic and instance information. To achieve this, we efficiently use panoptic maps in convolution and upsampling layers. We show that with the proposed changes to the generator, we can improve on the previous state-of-the-art methods by generating images in complex instance interaction environments in higher fidelity and tiny objects in more details. Furthermore, our proposed method also outperforms the previous state-of-the-art methods in metrics of mean IoU (Intersection over Union), and detAP (Detection Average Precision).

count=1
* A Quantum Computational Approach to Correspondence Problems on Point Sets
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Golyanik_A_Quantum_Computational_Approach_to_Correspondence_Problems_on_Point_Sets_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Golyanik_A_Quantum_Computational_Approach_to_Correspondence_Problems_on_Point_Sets_CVPR_2020_paper.pdf)]
    * Title: A Quantum Computational Approach to Correspondence Problems on Point Sets
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Vladislav Golyanik,  Christian Theobalt
    * Abstract: Modern adiabatic quantum computers (AQC) are already used to solve difficult combinatorial optimisation problems in various domains of science. Currently, only a few applications of AQC in computer vision have been demonstrated. We review AQC and derive a new algorithm for correspondence problems on point sets suitable for execution on AQC. Our algorithm has a subquadratic computational complexity of the state preparation. Examples of successful transformation estimation and point set alignment by simulated sampling are shown in the systematic experimental evaluation. Finally, we analyse the differences in the solutions and the corresponding energy values.

count=1
* Single Image Reflection Removal With Physically-Based Training Images
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Single_Image_Reflection_Removal_With_Physically-Based_Training_Images_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Single_Image_Reflection_Removal_With_Physically-Based_Training_Images_CVPR_2020_paper.pdf)]
    * Title: Single Image Reflection Removal With Physically-Based Training Images
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Soomin Kim,  Yuchi Huo,  Sung-Eui Yoon
    * Abstract: Recently, deep learning-based single image reflection separation methods have been exploited widely. To benefit the learning approach, a large number of training image pairs (i.e., with and without reflections) were synthesized in various ways, yet they are away from a physically-based direction. In this paper, physically based rendering is used for faithfully synthesizing the required training images, and a corresponding network structure and loss term are proposed. We utilize existing RGBD/RGB images to estimate meshes, then physically simulate the light transportation between meshes, glass, and lens with path tracing to synthesize training data, which successfully reproduce the spatially variant anisotropic visual effect of glass reflection. For guiding the separation better, we additionally consider a module, backtrack network (BT-net) for backtracking the reflections, which removes complicated ghosting, attenuation, blurred and defocused effect of glass/lens. This enables obtaining a priori information before having the distortion. The proposed method considering additional a priori information with physically simulated training data is validated with various real reflection images and shows visually pleasant and numerical advantages compared with state-of-the-art techniques.

count=1
* Blur Aware Calibration of Multi-Focus Plenoptic Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Labussiere_Blur_Aware_Calibration_of_Multi-Focus_Plenoptic_Camera_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Labussiere_Blur_Aware_Calibration_of_Multi-Focus_Plenoptic_Camera_CVPR_2020_paper.pdf)]
    * Title: Blur Aware Calibration of Multi-Focus Plenoptic Camera
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Mathieu Labussiere,  Celine Teuliere,  Frederic Bernardin,  Omar Ait-Aider
    * Abstract: This paper presents a novel calibration algorithm for Multi-Focus Plenoptic Cameras (MFPCs) using raw images only. The design of such cameras is usually complex and relies on precise placement of optic elements. Several calibration procedures have been proposed to retrieve the camera parameters but relying on simplified models, reconstructed images to extract features, or multiple calibrations when several types of micro-lens are used. Considering blur information, we propose a new Blur Aware Plenoptic (BAP) feature. It is first exploited in a pre-calibration step that retrieves initial camera parameters, and secondly to express a new cost function for our single optimization process. The effectiveness of our calibration method is validated by quantitative and qualitative experiments.

count=1
* Polarized Reflection Removal With Perfect Alignment in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Lei_Polarized_Reflection_Removal_With_Perfect_Alignment_in_the_Wild_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lei_Polarized_Reflection_Removal_With_Perfect_Alignment_in_the_Wild_CVPR_2020_paper.pdf)]
    * Title: Polarized Reflection Removal With Perfect Alignment in the Wild
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Chenyang Lei,  Xuhua Huang,  Mengdi Zhang,  Qiong Yan,  Wenxiu Sun,  Qifeng Chen
    * Abstract: We present a novel formulation to removing reflection from polarized images in the wild. We first identify the misalignment issues of existing reflection removal datasets where the collected reflection-free images are not perfectly aligned with input mixed images due to glass refraction. Then we build a new dataset with more than 100 types of glass in which obtained transmission images are perfectly aligned with input mixed images. Second, capitalizing on the special relationship between reflection and polarized light, we propose a polarized reflection removal model with a two-stage architecture. In addition, we design a novel perceptual NCC loss that can improve the performance of reflection removal and general image decomposition tasks. We conduct extensive experiments, and results suggest that our model outperforms state-of-the-art methods on reflection removal.

count=1
* Enhanced Blind Face Restoration With Multi-Exemplar Images and Adaptive Spatial Feature Fusion
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Enhanced_Blind_Face_Restoration_With_Multi-Exemplar_Images_and_Adaptive_Spatial_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Enhanced_Blind_Face_Restoration_With_Multi-Exemplar_Images_and_Adaptive_Spatial_CVPR_2020_paper.pdf)]
    * Title: Enhanced Blind Face Restoration With Multi-Exemplar Images and Adaptive Spatial Feature Fusion
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Xiaoming Li,  Wenyu Li,  Dongwei Ren,  Hongzhi Zhang,  Meng Wang,  Wangmeng Zuo
    * Abstract: In many real-world face restoration applications, e.g., smartphone photo albums and old films, multiple high-quality (HQ) images of the same person usually are available for a given degraded low-quality (LQ) observation. However, most existing guided face restoration methods are based on single HQ exemplar image, and are limited in properly exploiting guidance for improving the generalization ability to unknown degradation process. To address these issues, this paper suggests to enhance blind face restoration performance by utilizing multi-exemplar images and adaptive fusion of features from guidance and degraded images. First, given a degraded observation, we select the optimal guidance based on the weighted affine distance on landmark sets, where the landmark weights are learned to make the guidance image optimized to HQ image reconstruction. Second, moving least-square and adaptive instance normalization are leveraged for spatial alignment and illumination translation of guidance image in the feature space. Finally, for better feature fusion, multiple adaptive spatial feature fusion (ASFF) layers are introduced to incorporate guidance features in an adaptive and progressive manner, resulting in our ASFFNet. Experiments show that our ASFFNet performs favorably in terms of quantitative and qualitative evaluation, and is effective in generating photo-realistic results on real-world LQ images. The source code and models are available at https://github.com/csxmli2016/ASFFNet.

count=1
* Learning From Noisy Anchors for One-Stage Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Learning_From_Noisy_Anchors_for_One-Stage_Object_Detection_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Learning_From_Noisy_Anchors_for_One-Stage_Object_Detection_CVPR_2020_paper.pdf)]
    * Title: Learning From Noisy Anchors for One-Stage Object Detection
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Hengduo Li,  Zuxuan Wu,  Chen Zhu,  Caiming Xiong,  Richard Socher,  Larry S. Davis
    * Abstract: State-of-the-art object detectors rely on regressing and classifying an extensive list of possible anchors, which are divided into positive and negative samples based on their intersection-over-union (IoU) with corresponding ground-truth objects. Such a harsh split conditioned on IoU results in binary labels that are potentially noisy and challenging for training. In this paper, we propose to mitigate noise incurred by imperfect label assignment such that the contributions of anchors are dynamically determined by a carefully constructed cleanliness score associated with each anchor. Exploring outputs from both regression and classification branches, the cleanliness scores, estimated without incurring any additional computational overhead, are used not only as soft labels to supervise the training of the classification branch but also sample re-weighting factors for improved localization and classification accuracy. We conduct extensive experiments on COCO, and demonstrate, among other things, the proposed approach steadily improves RetinaNet by 2% with various backbones.

count=1
* Single Image Reflection Removal Through Cascaded Refinement
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Single_Image_Reflection_Removal_Through_Cascaded_Refinement_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Single_Image_Reflection_Removal_Through_Cascaded_Refinement_CVPR_2020_paper.pdf)]
    * Title: Single Image Reflection Removal Through Cascaded Refinement
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Chao Li,  Yixiao Yang,  Kun He,  Stephen Lin,  John E. Hopcroft
    * Abstract: We address the problem of removing undesirable reflections from a single image captured through a glass surface, which is an ill-posed, challenging but practically important problem for photo enhancement. Inspired by iterative structure reduction for hidden community detection in social networks, we propose an Iterative Boost Convolutional LSTM Network (IBCLN) that enables cascaded prediction for reflection removal. IBCLN is a cascaded network that iteratively refines the estimates of transmission and reflection layers in a manner that they can boost the prediction quality to each other, and information across steps of the cascade is transferred using an LSTM. The intuition is that the transmission is the strong, dominant structure while the reflection is the weak, hidden structure. They are complementary to each other in a single image and thus a better estimate and reduction on one side from the original image leads to a more accurate estimate on the other side. To facilitate training over multiple cascade steps, we employ LSTM to address the vanishing gradient problem, and propose residual reconstruction loss as further training guidance. Besides, we create a dataset of real-world images with reflection and ground-truth transmission layers to mitigate the problem of insufficient data. Comprehensive experiments demonstrate that the proposed method can effectively remove reflections in real and synthetic images compared with state-of-the-art reflection removal methods.

count=1
* Learning to See Through Obstructions
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Learning_to_See_Through_Obstructions_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Learning_to_See_Through_Obstructions_CVPR_2020_paper.pdf)]
    * Title: Learning to See Through Obstructions
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Yu-Lun Liu,  Wei-Sheng Lai,  Ming-Hsuan Yang,  Yung-Yu Chuang,  Jia-Bin Huang
    * Abstract: We present a learning-based approach for removing unwanted obstructions, such as window reflections, fence occlusions or raindrops, from a short sequence of images captured by a moving camera. Our method leverages the motion differences between the background and the obstructing elements to recover both layers. Specifically, we alternate between estimating dense optical flow fields of the two layers and reconstructing each layer from the flow-warped images via a deep convolutional neural network. The learning-based layer reconstruction allows us to accommodate potential errors in the flow estimation and brittle assumptions such as brightness consistency. We show that training on synthetically generated data transfers well to real images. Our results on numerous challenging scenarios of reflection and fence removal demonstrate the effectiveness of the proposed method.

count=1
* Don't Hit Me! Glass Detection in Real-World Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Mei_Dont_Hit_Me_Glass_Detection_in_Real-World_Scenes_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mei_Dont_Hit_Me_Glass_Detection_in_Real-World_Scenes_CVPR_2020_paper.pdf)]
    * Title: Don't Hit Me! Glass Detection in Real-World Scenes
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Haiyang Mei,  Xin Yang,  Yang Wang,  Yuanyuan Liu,  Shengfeng He,  Qiang Zhang,  Xiaopeng Wei,  Rynson W.H. Lau
    * Abstract: Glass is very common in our daily life. Existing computer vision systems neglect it and thus may have severe consequences, e.g., a robot may crash into a glass wall. However, sensing the presence of glass is not straightforward. The key challenge is that arbitrary objects/scenes can appear behind the glass, and the content within the glass region is typically similar to those behind it. In this paper, we propose an important problem of detecting glass from a single RGB image. To address this problem, we construct a large-scale glass detection dataset (GDD) and design a glass detection network, called GDNet, which explores abundant contextual cues for robust glass detection with a novel large-field contextual feature integration (LCFI) module. Extensive experiments demonstrate that the proposed method achieves more superior glass detection results on our GDD test set than state-of-the-art methods fine-tuned for glass detection.

count=1
* 3DRegNet: A Deep Neural Network for 3D Point Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Pais_3DRegNet_A_Deep_Neural_Network_for_3D_Point_Registration_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pais_3DRegNet_A_Deep_Neural_Network_for_3D_Point_Registration_CVPR_2020_paper.pdf)]
    * Title: 3DRegNet: A Deep Neural Network for 3D Point Registration
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: G. Dias Pais,  Srikumar Ramalingam,  Venu Madhav Govindu,  Jacinto C. Nascimento,  Rama Chellappa,  Pedro Miraldo
    * Abstract: We present 3DRegNet, a novel deep learning architecture for the registration of 3D scans. Given a set of 3D point correspondences, we build a deep neural network to address the following two challenges: (i) classification of the point correspondences into inliers/outliers, and (ii) regression of the motion parameters that align the scans into a common reference frame. With regard to regression, we present two alternative approaches: (i) a Deep Neural Network (DNN) registration and (ii) a Procrustes approach using SVD to estimate the transformation. Our correspondence-based approach achieves a higher speedup compared to competing baselines. We further propose the use of a refinement network, which consists of a smaller 3DRegNet as a refinement to improve the accuracy of the registration. Extensive experiments on two challenging datasets demonstrate that we outperform other methods and achieve state-of-the-art results. The code is available.

count=1
* Seeing the World in a Bag of Chips
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Park_Seeing_the_World_in_a_Bag_of_Chips_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Seeing_the_World_in_a_Bag_of_Chips_CVPR_2020_paper.pdf)]
    * Title: Seeing the World in a Bag of Chips
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Jeong Joon Park,  Aleksander Holynski,  Steven M. Seitz
    * Abstract: We address the dual problems of novel view synthesis and environment reconstruction from hand-held RGBD sensors. Our contributions include 1) modeling highly specular objects, 2) modeling inter-reflections and Fresnel effects, and 3) enabling surface light field reconstruction with the same input needed to reconstruct shape alone. In cases where scene surface has a strong mirror-like material component, we generate highly detailed environment images, revealing room composition, objects, people, buildings, and trees visible through windows. Our approach yields state of the art view synthesis techniques, operates on low dynamic range imagery, and is robust to geometric and calibration errors.

count=1
* Towards Backward-Compatible Representation Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.pdf)]
    * Title: Towards Backward-Compatible Representation Learning
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Yantao Shen,  Yuanjun Xiong,  Wei Xia,  Stefano Soatto
    * Abstract: We propose a way to learn visual features that are compatible with previously computed ones even when they have different dimensions and are learned via different neural network architectures and loss functions. Compatible means that, if such features are used to compare images, then "new" features can be compared directly to "old" features, so they can be used interchangeably. This enables visual search systems to bypass computing new features for all previously seen images when updating the embedding models, a process known as backfilling. Backward compatibility is critical to quickly deploy new embedding models that leverage ever-growing large-scale training datasets and improvements in deep learning architectures and training methods. We propose a framework to train embedding models, called backward-compatible training (BCT), as a first step towards backward compatible representation learning. In experiments on learning embeddings for face recognition, models trained with BCT successfully achieve backward compatibility without sacrificing accuracy, thus enabling backfill-free model updates of visual embeddings.

count=1
* Deep Parametric Shape Predictions Using Distance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Smirnov_Deep_Parametric_Shape_Predictions_Using_Distance_Fields_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Smirnov_Deep_Parametric_Shape_Predictions_Using_Distance_Fields_CVPR_2020_paper.pdf)]
    * Title: Deep Parametric Shape Predictions Using Distance Fields
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Dmitriy Smirnov,  Matthew Fisher,  Vladimir G. Kim,  Richard Zhang,  Justin Solomon
    * Abstract: Many tasks in graphics and vision demand machinery for converting shapes into consistent representations with sparse sets of parameters; these representations facilitate rendering, editing, and storage. When the source data is noisy or ambiguous, however, artists and engineers often manually construct such representations, a tedious and potentially time-consuming process. While advances in deep learning have been successfully applied to noisy geometric data, the task of generating parametric shapes has so far been difficult for these methods. Hence, we propose a new framework for predicting parametric shape primitives using deep learning. We use distance fields to transition between shape parameters like control points and input data on a pixel grid. We demonstrate efficacy on 2D and 3D tasks, including font vectorization and surface abstraction.

count=1
* VecRoad: Point-Based Iterative Graph Exploration for Road Graphs Extraction
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_VecRoad_Point-Based_Iterative_Graph_Exploration_for_Road_Graphs_Extraction_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_VecRoad_Point-Based_Iterative_Graph_Exploration_for_Road_Graphs_Extraction_CVPR_2020_paper.pdf)]
    * Title: VecRoad: Point-Based Iterative Graph Exploration for Road Graphs Extraction
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Yong-Qiang Tan,  Shang-Hua Gao,  Xuan-Yi Li,  Ming-Ming Cheng,  Bo Ren
    * Abstract: Extracting road graphs from aerial images automatically is more efficient and costs less than from field acquisition. This can be done by a post-processing step that vectorizes road segmentation predicted by CNN, but imperfect predictions will result in road graphs with low connectivity. On the other hand, iterative next move exploration could construct road graphs with better road connectivity, but often focuses on local information and does not provide precise alignment with the real road. To enhance the road connectivity while maintaining the precise alignment between the graph and real road, we propose a point-based iterative graph exploration scheme with segmentation-cues guidance and flexible steps. In our approach, we represent the location of the next move as a 'point' that unifies the representation of multiple constraints such as the direction and step size in each moving step. Information cues such as road segmentation and road junctions are jointly detected and utilized to guide the next move and achieve better alignment of roads. We demonstrate that our proposed method has a considerable improvement over state-of-the-art road graph extraction methods in terms of F-measure and road connectivity metrics on common datasets.

count=1
* High-Order Information Matters: Learning Relation and Topology for Occluded Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Order_Information_Matters_Learning_Relation_and_Topology_for_Occluded_Person_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_High-Order_Information_Matters_Learning_Relation_and_Topology_for_Occluded_Person_CVPR_2020_paper.pdf)]
    * Title: High-Order Information Matters: Learning Relation and Topology for Occluded Person Re-Identification
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Guan'an Wang,  Shuo Yang,  Huanyu Liu,  Zhicheng Wang,  Yang Yang,  Shuliang Wang,  Gang Yu,  Erjin Zhou,  Jian Sun
    * Abstract: Occluded person re-identification (ReID) aims to match occluded person images to holistic ones across dis-joint cameras. In this paper, we propose a novel framework by learning high-order relation and topology information for discriminative features and robust alignment. At first, we use a CNN backbone to learn feature maps and key-points estimation model to extract semantic local features. Even so, occluded images still suffer from occlusion and outliers. Then, we view the extracted local features of an image as nodes of a graph and propose an adaptive direction graph convolutional (ADGC) layer to pass relation information between nodes. The proposed ADGC layer can automatically suppress the message passing of meaningless features by dynamically learning direction and degree of linkage. When aligning two groups of local features, we view it as a graph matching problem and propose a cross-graph embedded-alignment (CGEA) layer to joint learn and embed topology information to local features, and straightly predict similarity score. The proposed CGEA layer can both take full use of alignment learned by graph matching and replace sensitive one-to-one alignment with a robust soft one. Finally, extensive experiments on occluded, partial, and holistic ReID tasks show the effectiveness of our proposed method. Specifically, our framework significantly outperforms state-of-the-art by 6.5% mAP scores on Occluded-Duke dataset.

count=1
* Joint Filtering of Intensity Images and Neuromorphic Events for High-Resolution Noise-Robust Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Joint_Filtering_of_Intensity_Images_and_Neuromorphic_Events_for_High-Resolution_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Joint_Filtering_of_Intensity_Images_and_Neuromorphic_Events_for_High-Resolution_CVPR_2020_paper.pdf)]
    * Title: Joint Filtering of Intensity Images and Neuromorphic Events for High-Resolution Noise-Robust Imaging
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zihao W. Wang,  Peiqi Duan,  Oliver Cossairt,  Aggelos Katsaggelos,  Tiejun Huang,  Boxin Shi
    * Abstract: We present a novel computational imaging system with high resolution and low noise. Our system consists of a traditional video camera which captures high-resolution intensity images, and an event camera which encodes high-speed motion as a stream of asynchronous binary events. To process the hybrid input, we propose a unifying framework that first bridges the two sensing modalities via a noise-robust motion compensation model, and then performs joint image filtering. The filtered output represents the temporal gradient of the captured space-time volume, which can be viewed as motion-compensated event frames with high resolution and low noise. Therefore, the output can be widely applied to many existing event-based algorithms that are highly dependent on spatial resolution and noise robustness. In experimental results performed on both publicly available datasets as well as our contributing RGB-DAVIS dataset, we show systematic performance improvement in applications such as high frame-rate video synthesis, feature/corner detection and tracking, as well as high dynamic range image reconstruction.

count=1
* TCTS: A Task-Consistent Two-Stage Framework for Person Search
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_TCTS_A_Task-Consistent_Two-Stage_Framework_for_Person_Search_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_TCTS_A_Task-Consistent_Two-Stage_Framework_for_Person_Search_CVPR_2020_paper.pdf)]
    * Title: TCTS: A Task-Consistent Two-Stage Framework for Person Search
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Cheng Wang,  Bingpeng Ma,  Hong Chang,  Shiguang Shan,  Xilin Chen
    * Abstract: The state of the art person search methods separate person search into detection and re-ID stages, but ignore the consistency between these two stages. The general person detector has no special attention on the query target; The re-ID model is trained on hand-drawn bounding boxes which are not available in person search. To address the consistency problem, we introduce a Task-Consist Two-Stage (TCTS) person search framework, includes an identity-guided query (IDGQ) detector and a Detection Results Adapted (DRA) re-ID model. In the detection stage, the IDGQ detector learns an auxiliary identity branch to compute query similarity scores for proposals. With consideration of the query similarity scores and foreground score, IDGQ produces query-like bounding boxes for the re-ID stage. In the re-ID stage, we predict identity labels of detected bounding boxes, and use these examples to construct a more practical mixed train set for the DRA model. Training on the mixed train set improves the robustness of the re-ID stage to inaccurate detection. We evaluate our method on two benchmark datasets, CUHK-SYSU and PRW. Our framework achieves 93.9% of mAP and 95.1% of rank1 accuracy on CUHK-SYSU, outperforming the previous state of the art methods.

count=1
* Rethinking Classification and Localization for Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_Rethinking_Classification_and_Localization_for_Object_Detection_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Rethinking_Classification_and_Localization_for_Object_Detection_CVPR_2020_paper.pdf)]
    * Title: Rethinking Classification and Localization for Object Detection
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Yue Wu,  Yinpeng Chen,  Lu Yuan,  Zicheng Liu,  Lijuan Wang,  Hongzhi Li,  Yun Fu
    * Abstract: Two head structures (i.e. fully connected head and convolution head) have been widely used in R-CNN based detectors for classification and localization tasks. However, there is a lack of understanding of how does these two head structures work for these two tasks. To address this issue, we perform a thorough analysis and find an interesting fact that the two head structures have opposite preferences towards the two tasks. Specifically, the fully connected head (fc-head) is more suitable for the classification task, while the convolution head (conv-head) is more suitable for the localization task. Furthermore, we examine the output feature maps of both heads and find that fc-head has more spatial sensitivity than conv-head. Thus, fc-head has more capability to distinguish a complete object from part of an object, but is not robust to regress the whole object. Based upon these findings, we propose a Double-Head method, which has a fully connected head focusing on classification and a convolution head for bounding box regression. Without bells and whistles, our method gains +3.5 and +2.8 AP on MS COCO dataset from Feature Pyramid Network (FPN) baselines with ResNet-50 and ResNet-101 backbones, respectively.

count=1
* Temporal-Context Enhanced Detection of Heavily Occluded Pedestrians
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_Temporal-Context_Enhanced_Detection_of_Heavily_Occluded_Pedestrians_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Temporal-Context_Enhanced_Detection_of_Heavily_Occluded_Pedestrians_CVPR_2020_paper.pdf)]
    * Title: Temporal-Context Enhanced Detection of Heavily Occluded Pedestrians
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Jialian Wu,  Chunluan Zhou,  Ming Yang,  Qian Zhang,  Yuan Li,  Junsong Yuan
    * Abstract: State-of-the-art pedestrian detectors have performed promisingly on non-occluded pedestrians, yet they are still confronted by heavy occlusions. Although many previous works have attempted to alleviate the pedestrian occlusion issue, most of them rest on still images. In this paper, we exploit the local temporal context of pedestrians in videos and propose a tube feature aggregation network (TFAN) aiming at enhancing pedestrian detectors against severe occlusions. Specifically, for an occluded pedestrian in the current frame, we iteratively search for its relevant counterparts along temporal axis to form a tube. Then, features from the tube are aggregated according to an adaptive weight to enhance the feature representations of the occluded pedestrian. Furthermore, we devise a temporally discriminative embedding module (TDEM) and a part-based relation module (PRM), respectively, which adapts our approach to better handle tube drifting and heavy occlusions. Extensive experiments are conducted on three datasets, Caltech, NightOwls and KAIST, showing that our proposed method is significantly effective for heavily occluded pedestrian detection. Moreover, we achieve the state-of-the-art performance on the Caltech and NightOwls datasets.

count=1
* Structure Preserving Generative Cross-Domain Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Xia_Structure_Preserving_Generative_Cross-Domain_Learning_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xia_Structure_Preserving_Generative_Cross-Domain_Learning_CVPR_2020_paper.pdf)]
    * Title: Structure Preserving Generative Cross-Domain Learning
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Haifeng Xia,  Zhengming Ding
    * Abstract: Unsupervised domain adaptation (UDA) casts a light when dealing with insufficient or no labeled data in the target domain by exploring the well-annotated source knowledge in different distributions. Most research efforts on UDA explore to seek a domain-invariant classifier over source supervision. However, due to the scarcity of label information in the target domain, such a classifier has a lack of ground-truth target supervision, which dramatically obstructs the robustness and discrimination of the classifier. To this end, we develop a novel Generative cross-domain learning via Structure-Preserving (GSP), which attempts to transform target data into the source domain in order to take advantage of source supervision. Specifically, a novel cross-domain graph alignment is developed to capture the intrinsic relationship across two domains during target-source translation. Simultaneously, two distinct classifiers are trained to trigger the domain-invariant feature learning both guided with source supervision, one is a traditional source classifier and the other is a source-supervised target classifier. Extensive experimental results on several cross-domain visual benchmarks have demonstrated the effectiveness of our model by comparing with other state-of-the-art UDA algorithms.

count=1
* RPM-Net: Robust Point Matching Using Learned Features
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Yew_RPM-Net_Robust_Point_Matching_Using_Learned_Features_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yew_RPM-Net_Robust_Point_Matching_Using_Learned_Features_CVPR_2020_paper.pdf)]
    * Title: RPM-Net: Robust Point Matching Using Learned Features
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zi Jian Yew,  Gim Hee Lee
    * Abstract: Iterative Closest Point (ICP) solves the rigid point cloud registration problem iteratively in two steps: (1) make hard assignments of spatially closest point correspondences, and then (2) find the least-squares rigid transformation. The hard assignments of closest point correspondences based on spatial distances are sensitive to the initial rigid transformation and noisy/outlier points, which often cause ICP to converge to wrong local minima. In this paper, we propose the RPM-Net -- a less sensitive to initialization and more robust deep learning-based approach for rigid point cloud registration. To this end, our network uses the differentiable Sinkhorn layer and annealing to get soft assignments of point correspondences from hybrid features learned from both spatial coordinates and local geometry. To further improve registration performance, we introduce a secondary network to predict optimal annealing parameters. Unlike some existing methods, our RPM-Net handles missing correspondences and point clouds with partial visibility. Experimental results show that our RPM-Net achieves state-of-the-art performance compared to existing non-deep learning and recent deep learning methods. Our source code is available at the project website (https://github.com/yewzijian/RPMNet).

count=1
* What Does Plate Glass Reveal About Camera Calibration?
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_What_Does_Plate_Glass_Reveal_About_Camera_Calibration_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_What_Does_Plate_Glass_Reveal_About_Camera_Calibration_CVPR_2020_paper.pdf)]
    * Title: What Does Plate Glass Reveal About Camera Calibration?
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Qian Zheng,  Jinnan Chen,  Zhan Lu,  Boxin Shi,  Xudong Jiang,  Kim-Hui Yap,  Ling-Yu Duan,  Alex C. Kot
    * Abstract: This paper aims to calibrate the orientation of glass and the field of view of the camera from a single reflection-contaminated image. We show how a reflective amplitude coefficient map can be used as a calibration cue. Different from existing methods, the proposed solution is free from image contents. To reduce the impact of a noisy calibration cue estimated from a reflection-contaminated image, we propose two strategies: an optimization-based method that imposes part of though reliable entries on the map and a learning-based method that fully exploits all entries. We collect a dataset containing 320 samples as well as their camera parameters for evaluation. We demonstrate that our method not only facilitates a general single image camera calibration method that leverages image contents but also contributes to improving the performance of single image reflection removal. Furthermore, we show our byproduct output helps alleviate the ill-posed problem of estimating the panorama from a single image.

count=1
* SPSG: Self-Supervised Photometric Scene Generation From RGB-D Scans
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_SPSG_Self-Supervised_Photometric_Scene_Generation_From_RGB-D_Scans_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_SPSG_Self-Supervised_Photometric_Scene_Generation_From_RGB-D_Scans_CVPR_2021_paper.pdf)]
    * Title: SPSG: Self-Supervised Photometric Scene Generation From RGB-D Scans
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Angela Dai, Yawar Siddiqui, Justus Thies, Julien Valentin, Matthias Niessner
    * Abstract: We present SPSG, a novel approach to generate high-quality, colored 3D models of scenes from RGB-D scan observations by learning to infer unobserved scene geometry and color in a self-supervised fashion. Our self-supervised approach learns to jointly inpaint geometry and color by correlating an incomplete RGB-D scan with a more complete version of that scan. Notably, rather than relying on 3D reconstruction losses to inform our 3D geometry and color reconstruction, we propose adversarial and perceptual losses operating on 2D renderings in order to achieve high-resolution, high-quality colored reconstructions of scenes. This exploits the high-resolution, self-consistent signal from individual raw RGB-D frames, in contrast to fused 3D reconstructions of the frames which exhibit inconsistencies from view-dependent effects, such as color balancing or pose inconsistencies. Thus, by informing our 3D scene generation directly through 2D signal, we produce high-quality colored reconstructions of 3D scenes, outperforming state of the art on both synthetic and real data.

count=1
* Compatibility-Aware Heterogeneous Visual Search
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Duggal_Compatibility-Aware_Heterogeneous_Visual_Search_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Duggal_Compatibility-Aware_Heterogeneous_Visual_Search_CVPR_2021_paper.pdf)]
    * Title: Compatibility-Aware Heterogeneous Visual Search
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Rahul Duggal, Hao Zhou, Shuo Yang, Yuanjun Xiong, Wei Xia, Zhuowen Tu, Stefano Soatto
    * Abstract: We tackle the problem of visual search under resource constraints. Existing systems use the same embedding model to compute representations (embeddings) for the query and gallery images. Such systems inherently face a hard accuracy-efficiency trade-off: the embedding model needs to be large enough to ensure high accuracy, yet small enough to enable query-embedding computation on resource-constrained platforms. This trade-off could be mitigated if gallery embeddings are generated from a large model and query embeddings are extracted using a compact model. The key to building such a system is to ensure representation compatibility between the query and gallery models. In this paper, we address two forms of compatibility: One enforced by modifying the parameters of each model that computes the embeddings. The other by modifying the architectures that compute the embeddings, leading to compatibility-aware neural architecture search (CMP-NAS). We test CMP-NAS on challenging retrieval tasks for fashion images (DeepFashion2), and face images (IJB-C). Compared to ordinary (homogeneous) visual search using the largest embedding model (paragon), CMP-NAS achieves 80-fold and 23-fold cost reduction while maintaining accuracy within 0.3% and 1.6% of the paragon on DeepFashion2 and IJB-C respectively.

count=1
* Cross-Domain Similarity Learning for Face Recognition in Unseen Domains
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Faraki_Cross-Domain_Similarity_Learning_for_Face_Recognition_in_Unseen_Domains_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Faraki_Cross-Domain_Similarity_Learning_for_Face_Recognition_in_Unseen_Domains_CVPR_2021_paper.pdf)]
    * Title: Cross-Domain Similarity Learning for Face Recognition in Unseen Domains
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Masoud Faraki, Xiang Yu, Yi-Hsuan Tsai, Yumin Suh, Manmohan Chandraker
    * Abstract: Face recognition models trained under the assumption of identical training and test distributions often suffer from poor generalization when faced with unknown variations, such as a novel ethnicity or unpredictable individual make-ups during test time. In this paper, we introduce a novel cross-domain metric learning loss, which we dub Cross-Domain Triplet (CDT) loss, to improve face recognition in unseen domains. The CDT loss encourages learning semantically meaningful features by enforcing compact feature clusters of identities from one domain, where the compactness is measured by underlying similarity metrics that belong to another training domain with different statistics. Intuitively, it discriminatively correlates explicit metrics derived from one domain, with triplet samples from another domain in a unified loss function to be minimized within a network, which leads to better alignment of the training domains. The network parameters are further enforced to learn generalized features under domain shift, in a model-agnostic learning pipeline. Unlike the recent work of Meta Face Recognition, our method does not require careful hard-pair sample mining and filtering strategy during training. Extensive experiments on various face recognition benchmarks show the superiority of our method in handling variations, compared to baseline methods and the state-of-the-arts.

count=1
* Geo-FARM: Geodesic Factor Regression Model for Misaligned Pre-Shape Responses in Statistical Shape Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Geo-FARM_Geodesic_Factor_Regression_Model_for_Misaligned_Pre-Shape_Responses_in_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Geo-FARM_Geodesic_Factor_Regression_Model_for_Misaligned_Pre-Shape_Responses_in_CVPR_2021_paper.pdf)]
    * Title: Geo-FARM: Geodesic Factor Regression Model for Misaligned Pre-Shape Responses in Statistical Shape Analysis
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Chao Huang, Anuj Srivastava, Rongjie Liu
    * Abstract: The problem of using covariates to predict shapes of objects in a regression setting is important in many fields. A formal statistical approach, termed geodesic regression model, is commonly used for modeling and analyzing relationships between Euclidean predictors and shape responses. Despite its popularity, this model faces several key challenges, including (i) misalignment of shapes due to pre-processing steps, (ii) difficulties in shape alignment due to imaging heterogeneity, and (iii) lack of spatial correlation in shape structures. This paper proposes a comprehensive geodesic factor regression model that addresses all these challenges. Instead of using shapes as extracted from pre-registered data, it takes a more fundamental approach, incorporating alignment step within the proposed regression model and learns them using both pre-shape and covariate data. Additionally, it specifies spatial correlation structures using low-dimensional representations, including latent factors on the tangent space and isotropic error terms. The proposed framework results in substantial improvements in regression performance, as demonstrated through simulation studies and a real data analysis on Corpus Callosum contour data obtained from the ADNI study.

count=1
* Robust Consistent Video Depth Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Kopf_Robust_Consistent_Video_Depth_Estimation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Kopf_Robust_Consistent_Video_Depth_Estimation_CVPR_2021_paper.pdf)]
    * Title: Robust Consistent Video Depth Estimation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Johannes Kopf, Xuejian Rong, Jia-Bin Huang
    * Abstract: We present an algorithm for estimating consistent dense depth maps and camera poses from a monocular video. We integrate a learning-based depth prior, in the form of a convolutional neural network trained for single-image depth estimation, with geometric optimization, to estimate a smooth camera trajectory as well as detailed and stable depth reconstruction. Our algorithm combines two complementary techniques: (1) flexible deformation-splines for low-frequency large-scale alignment and (2) geometry-aware depth filtering for high-frequency alignment of fine depth details. In contrast to prior approaches, our method does not require camera poses as input and achieves robust reconstruction for challenging hand-held cell phone captures that contain a significant amount of noise, shake, motion blur, and rolling shutter deformations. Our method quantitatively outperforms state-of-the-arts on the Sintel benchmark for both depth and pose estimations, and attains favorable qualitative results across diverse wild datasets.

count=1
* Robust Reflection Removal With Reflection-Free Flash-Only Cues
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lei_Robust_Reflection_Removal_With_Reflection-Free_Flash-Only_Cues_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lei_Robust_Reflection_Removal_With_Reflection-Free_Flash-Only_Cues_CVPR_2021_paper.pdf)]
    * Title: Robust Reflection Removal With Reflection-Free Flash-Only Cues
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Chenyang Lei, Qifeng Chen
    * Abstract: We propose a simple yet effective reflection-free cue for robust reflection removal from a pair of flash and ambient (no-flash) images. The reflection-free cue exploits a flash-only image obtained by subtracting the ambient image from the corresponding flash image in raw data space. The flash-only image is equivalent to an image taken in a dark environment with only a flash on. We observe that this flash-only image is visually reflection-free, and thus it can provide robust cues to infer the reflection in the ambient image. Since the flash-only image usually has artifacts, we further propose a dedicated model that not only utilizes the reflection-free cue but also avoids introducing artifacts, which helps accurately estimate reflection and transmission. Our experiments on real-world images with various types of reflection demonstrate the effectiveness of our model with reflection-free flash-only cues: our model outperforms state-of-the-art reflection removal approaches by more than 5.23dB in PSNR, 0.04 in SSIM, and 0.068 in LPIPS. Our source code and dataset are publicly available at github.com/ChenyangLEI/flash-reflection-removal.

count=1
* Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.pdf)]
    * Title: Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Jichang Li, Guanbin Li, Yemin Shi, Yizhou Yu
    * Abstract: In semi-supervised domain adaptation, a few labeled samples per class in the target domain guide features of the remaining target samples to aggregate around them. However, the trained model cannot produce a highly discriminative feature representation for the target domain because the training data is dominated by labeled samples from the source domain. This could lead to disconnection between the labeled and unlabeled target samples as well as misalignment between unlabeled target samples and the source domain. In this paper, we propose a novel approach called Cross-domain Adaptive Clustering to address this problem. To achieve both inter-domain and intra-domain adaptation, we first introduce an adversarial adaptive clustering loss to group features of unlabeled target data into clusters and perform cluster-wise feature alignment across the source and target domains. We further apply pseudo labeling to unlabeled samples in the target domain and retain pseudo-labels with high confidence. Pseudo labeling expands the number of "labeled" samples in each class in the target domain, and thus produces a more robust and powerful cluster core for each class to facilitate adversarial learning. Extensive experiments on benchmark datasets, including DomainNet, Office-Home and Office, demonstrate that our proposed approach achieves the state-of-the-art performance in semi-supervised domain adaptation.

count=1
* Spatial Feature Calibration and Temporal Fusion for Effective One-Stage Video Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Spatial_Feature_Calibration_and_Temporal_Fusion_for_Effective_One-Stage_Video_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spatial_Feature_Calibration_and_Temporal_Fusion_for_Effective_One-Stage_Video_CVPR_2021_paper.pdf)]
    * Title: Spatial Feature Calibration and Temporal Fusion for Effective One-Stage Video Instance Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Minghan Li, Shuai Li, Lida Li, Lei Zhang
    * Abstract: Modern one-stage video instance segmentation networks suffer from two limitations. First, convolutional features are neither aligned with anchor boxes nor with ground-truth bounding boxes, reducing the mask sensitivity to spatial location. Second, a video is directly divided into individual frames for frame-level instance segmentation, ignoring the temporal correlation between adjacent frames. To address these issues, we propose a simple yet effective one-stage video instance segmentation framework by spatial calibration and temporal fusion, namely STMask. To ensure spatial feature calibration with ground-truth bounding boxes, we first predict regressed bounding boxes around ground-truth bounding boxes, and extract features from them for frame-level instance segmentation. To further explore temporal correlation among video frames, we aggregate a temporal fusion module to infer instance masks from each frame to its adjacent frames, which helps our framework to handle challenging videos such as motion blur, partial occlusion and unusual object-to-camera poses. Experiments on the YouTube-VIS valid set show that the proposed STMask with ResNet-50/-101 backbone obtains 33.5 % / 36.8 % mask AP, while achieving 28.6 / 23.4 FPS on video instance segmentation. The code is released online https://github.com/MinghanLi/STMask.

count=1
* M3DSSD: Monocular 3D Single Stage Object Detector
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Luo_M3DSSD_Monocular_3D_Single_Stage_Object_Detector_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_M3DSSD_Monocular_3D_Single_Stage_Object_Detector_CVPR_2021_paper.pdf)]
    * Title: M3DSSD: Monocular 3D Single Stage Object Detector
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Shujie Luo, Hang Dai, Ling Shao, Yong Ding
    * Abstract: In this paper, we propose a Monocular 3D Single Stage object Detector (M3DSSD) with feature alignment and asymmetric non-local attention. Current anchor-based monocular 3D object detection methods suffer from feature mismatching. To overcome this, we propose a two-step feature alignment approach. In the first step, the shape alignment is performed to enable the receptive field of the feature map to focus on the pre-defined anchors with high confidence scores. In the second step, the center alignment is used to align the features at 2D/3D centers. Further, it is often difficult to learn global information and capture long-range relationships, which are important for the depth prediction of objects. Therefore, we propose a novel asymmetric non-local attention block with multi-scale sampling to extract depth-wise features. The proposed M3DSSD achieves significantly better performance than the monocular 3D object detection methods on the KITTI dataset, in both 3D object detection and bird's eye view tasks. The code is released at https://github.com/mumianyuxin/M3DSSD.

count=1
* Learning Semantic Person Image Generation by Region-Adaptive Normalization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lv_Learning_Semantic_Person_Image_Generation_by_Region-Adaptive_Normalization_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lv_Learning_Semantic_Person_Image_Generation_by_Region-Adaptive_Normalization_CVPR_2021_paper.pdf)]
    * Title: Learning Semantic Person Image Generation by Region-Adaptive Normalization
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Zhengyao Lv, Xiaoming Li, Xin Li, Fu Li, Tianwei Lin, Dongliang He, Wangmeng Zuo
    * Abstract: Human pose transfer has received great attention due to its wide applications, yet is still a challenging task that is not well solved. Recent works have achieved great success to transfer the person image from the source to the target pose. However, most of them cannot well capture the semantic appearance, resulting in inconsistent and less realistic textures on the reconstructed results. To address this issue, we propose a new two-stage framework to handle the pose and appearance translation. In the first stage, we predict the target semantic parsing maps to eliminate the difficulties of pose transfer and further benefit the latter translation of per-region appearance style. In the second one, with the predicted target semantic maps, we suggest a new person image generation method by incorporating the region-adaptive normalization, in which it takes the per-region styles to guide the target appearance generation. Extensive experiments show that our proposed SPGNet can generate more semantic, consistent, and photo-realistic results and perform favorably against the state of the art methods in terms of quantitative and qualitative evaluation. The source code and model are available at https://github.com/cszy98/SPGNet.git.

count=1
* Pixel Codec Avatars
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Pixel_Codec_Avatars_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_Pixel_Codec_Avatars_CVPR_2021_paper.pdf)]
    * Title: Pixel Codec Avatars
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Shugao Ma, Tomas Simon, Jason Saragih, Dawei Wang, Yuecheng Li, Fernando De la Torre, Yaser Sheikh
    * Abstract: Telecommunication with photorealistic avatars in virtual or augmented reality is a promising path for achieving authentic face-to-face communication in 3D over remote physical distances. In this work, we present the Pixel Codec Avatars (PiCA): a deep generative model of 3D human faces that achieves state of the art reconstruction performance while being computationally efficient and adaptive to the rendering conditions during execution. Our model combines two core ideas: (1) a fully convolutional architecture for decoding spatially varying features, and (2) a rendering-adaptive per-pixel decoder. Both techniques are integrated via a dense surface representation that is learned in a weakly-supervised manner from low-topology mesh tracking over training images. We demonstrate that PiCA improves reconstruction over existing techniques across testing expressions and views on persons of different gender and skin tone. Importantly, we show that the PiCA model is much smaller than the state-of-art baseline model, and makes multi-person telecommunicaiton possible: on a single Oculus Quest 2 mobile VR headset, 5 avatars are rendered in realtime in the same scene.

count=1
* DeepSurfels: Learning Online Appearance Fusion
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Mihajlovic_DeepSurfels_Learning_Online_Appearance_Fusion_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Mihajlovic_DeepSurfels_Learning_Online_Appearance_Fusion_CVPR_2021_paper.pdf)]
    * Title: DeepSurfels: Learning Online Appearance Fusion
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Marko Mihajlovic, Silvan Weder, Marc Pollefeys, Martin R. Oswald
    * Abstract: We present DeepSurfels, a novel hybrid scene representation for geometry and appearance information. DeepSurfels combines explicit and neural building blocks to jointly encode geometry and appearance information. In contrast to established representations, DeepSurfels better represents high-frequency textures, is well-suited for online updates of appearance information, and can be easily combined with machine learning methods. We further present an end-to-end trainable online appearance fusion pipeline that fuses information from RGB images into the proposed scene representation and is trained using self-supervision imposed by the reprojection error with respect to the input images. Our method compares favorably to classical texture mapping approaches as well as recent learning-based techniques. Moreover, we demonstrate lower runtime, improved generalization capabilities, and better scalability to larger scenes compared to existing methods.

count=1
* Neural Surface Maps
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Morreale_Neural_Surface_Maps_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Morreale_Neural_Surface_Maps_CVPR_2021_paper.pdf)]
    * Title: Neural Surface Maps
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Luca Morreale, Noam Aigerman, Vladimir G. Kim, Niloy J. Mitra
    * Abstract: Maps are arguably one of the most fundamental concepts used to define and operate on manifold surfaces in differentiable geometry. Accordingly, in geometry processing, maps are ubiquitous and are used in many core applications, such as paramterization, shape analysis, remeshing, and deformation. Unfortunately, most computational representations of surface maps do not lend themselves to manipulation and optimization, usually entailing hard, discrete problems. While algorithms exist to solve these problems, they are problem-specific, and a general framework for surface maps is still in need. In this paper, we advocate to consider neural networks as encoding surface maps. Since neural networks can be composed on one another and are differentiable, we show it is easy to use them to define surfaces via atlases, compose them for surface-to-surface mappings, and optimize differentiable objectives relating to them, such as any notion of distortion, in a trivial manner. In our experiments, we represent surfaces by generating a neural map that approximates a UV parameterization of a 3D model. Then, we compose this map with other neural maps which we optimize with respect to distortion measures. We show that our formulation enables trivial optimization of rather elusive mapping tasks, such as maps between a collection of surfaces.

count=1
* Improving Unsupervised Image Clustering With Robust Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Park_Improving_Unsupervised_Image_Clustering_With_Robust_Learning_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Park_Improving_Unsupervised_Image_Clustering_With_Robust_Learning_CVPR_2021_paper.pdf)]
    * Title: Improving Unsupervised Image Clustering With Robust Learning
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Sungwon Park, Sungwon Han, Sundong Kim, Danu Kim, Sungkyu Park, Seunghoon Hong, Meeyoung Cha
    * Abstract: Unsupervised image clustering methods often introduce alternative objectives to indirectly train the model and are subject to faulty predictions and overconfident results. To overcome these challenges, the current research proposes an innovative model RUC that is inspired by robust learning. RUC's novelty is at utilizing pseudo-labels of existing image clustering models as a noisy dataset that may include misclassified samples. Its retraining process can revise misaligned knowledge and alleviate the overconfidence problem in predictions. The model's flexible structure makes it possible to be used as an add-on module to other clustering methods and helps them achieve better performance on multiple datasets. Extensive experiments show that the proposed model can adjust the model confidence with better calibration and gain additional robustness against adversarial noise.

count=1
* Reconsidering Representation Alignment for Multi-View Clustering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Trosten_Reconsidering_Representation_Alignment_for_Multi-View_Clustering_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Trosten_Reconsidering_Representation_Alignment_for_Multi-View_Clustering_CVPR_2021_paper.pdf)]
    * Title: Reconsidering Representation Alignment for Multi-View Clustering
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Daniel J. Trosten, Sigurd Lokse, Robert Jenssen, Michael Kampffmeyer
    * Abstract: Aligning distributions of view representations is a core component of today's state of the art models for deep multi-view clustering. However, we identify several drawbacks with naively aligning representation distributions. We demonstrate that these drawbacks both lead to less separable clusters in the representation space, and inhibit the model's ability to prioritize views. Based on these observations, we develop a simple baseline model for deep multi-view clustering. Our baseline model avoids representation alignment altogether, while performing similar to, or better than, the current state of the art. We also expand our baseline model by adding a contrastive learning component. This introduces a selective alignment procedure that preserves the model's ability to prioritize views. Our experiments show that the contrastive learning component enhances the baseline model, improving on the current state of the art by a large margin on several datasets.

count=1
* PointAugmenting: Cross-Modal Augmentation for 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_PointAugmenting_Cross-Modal_Augmentation_for_3D_Object_Detection_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_PointAugmenting_Cross-Modal_Augmentation_for_3D_Object_Detection_CVPR_2021_paper.pdf)]
    * Title: PointAugmenting: Cross-Modal Augmentation for 3D Object Detection
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Chunwei Wang, Chao Ma, Ming Zhu, Xiaokang Yang
    * Abstract: Camera and LiDAR are two complementary sensors for 3D object detection in the autonomous driving context. Camera provides rich texture and color cues while LiDAR specializes in relative distance sensing. The challenge of 3D object detection lies in effectively fusing 2D camera images with 3D LiDAR points. In this paper, we present a novel cross-modal 3D object detection algorithm, named PointAugmenting. On one hand, PointAugmenting decorates point clouds with corresponding point-wise CNN features extracted by pretrained 2D detection models, and then performs 3D object detection over the decorated point clouds. In comparison with highly abstract semantic segmentation scores to decorate point clouds, CNN features from detection networks adapt to object appearance variations, achieving significant improvement. On the other hand, PointAugmenting benefits from a novel cross-modal data augmentation algorithm, which consistently pastes virtual objects into images and point clouds during network training. Extensive experiments on the large-scale nuScenes and Waymo datasets demonstrate the effectiveness and efficiency of our PointAugmenting. Notably, PointAugmenting outperforms the LiDAR-only baseline detector by +6.5% mAP and achieves the new state-of-the-art results on the nuScenes leaderboard to date.

count=1
* DANNet: A One-Stage Domain Adaptation Network for Unsupervised Nighttime Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_DANNet_A_One-Stage_Domain_Adaptation_Network_for_Unsupervised_Nighttime_Semantic_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_DANNet_A_One-Stage_Domain_Adaptation_Network_for_Unsupervised_Nighttime_Semantic_CVPR_2021_paper.pdf)]
    * Title: DANNet: A One-Stage Domain Adaptation Network for Unsupervised Nighttime Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Xinyi Wu, Zhenyao Wu, Hao Guo, Lili Ju, Song Wang
    * Abstract: Semantic segmentation of nighttime images plays an equally important role as that of daytime images in autonomous driving, but the former is much more challenging due to poor illuminations and arduous human annotations. In this paper, we propose a novel domain adaptation network (DANNet) for nighttime semantic segmentation without using labeled nighttime image data. It employs an adversarial training with a labeled daytime dataset and an unlabeled dataset that contains coarsely aligned day-night image pairs. Specifically, for the unlabeled day-night image pairs, we use the pixel-level predictions of static object categories on a daytime image as a pseudo supervision to segment its counterpart nighttime image. We further design a re-weighting strategy to handle the inaccuracy caused by misalignment between day-night image pairs and wrong predictions of daytime images, as well as boost the prediction accuracy of small objects. The proposed DANNet is the first one stage adaptation framework for nighttime semantic segmentation, which does not train additional day-night image transfer models as a separate pre-processing stage. Extensive experiments on Dark Zurich and Nighttime Driving datasets show that our method achieves state-of-the-art performance for nighttime semantic segmentation.

count=1
* Deep Denoising of Flash and No-Flash Pairs for Photography in Low-Light Environments
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Xia_Deep_Denoising_of_Flash_and_No-Flash_Pairs_for_Photography_in_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Xia_Deep_Denoising_of_Flash_and_No-Flash_Pairs_for_Photography_in_CVPR_2021_paper.pdf)]
    * Title: Deep Denoising of Flash and No-Flash Pairs for Photography in Low-Light Environments
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Zhihao Xia, Michael Gharbi, Federico Perazzi, Kalyan Sunkavalli, Ayan Chakrabarti
    * Abstract: We introduce a neural network-based method to denoise pairs of images taken in quick succession in low-light environments, with and without a flash. Our goal is to produce a high-quality rendering of the scene that preserves the color and mood from the ambient illumination of the noisy no-flash image, while recovering surface texture and detail revealed by the flash. Our network outputs a gain map and a field of kernels, the latter obtained by linearly mixing elements of a per-image low-rank kernel basis. We first apply the kernel field to the no-flash image, and then multiply the result with the gain map to create the final output. We show our network effectively learns to produce high-quality images by combining a smoothed out estimate of the scene's ambient appearance from the no-flash image, with high-frequency albedo details extracted from the flash input. Our experiments show significant improvements over alternative captures without a flash, and baseline denoisers that use flash no-flash pairs. In particular, our method produces images that are both noise-free and contain accurate ambient colors without the sharp shadows or strong specular highlights visible in the flash image.

count=1
* Dynamic Weighted Learning for Unsupervised Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Xiao_Dynamic_Weighted_Learning_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_Dynamic_Weighted_Learning_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf)]
    * Title: Dynamic Weighted Learning for Unsupervised Domain Adaptation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Ni Xiao, Lei Zhang
    * Abstract: Unsupervised domain adaptation (UDA) aims to improve the classification performance on an unlabeled target domain by leveraging information from a fully labeled source domain. Recent approaches explore domain-invariant and class-discriminant representations to tackle this task. These methods, however, ignore the interaction between domain alignment learning and class discrimination learning. As a result, the missing or inadequate tradeoff between domain alignment and class discrimination are prone to the problem of negative transfer. In this paper, we propose Dynamic Weighted Learning (DWL) to avoid the discriminability vanishing problem caused by excessive alignment learning and domain misalignment problem caused by excessive discriminant learning. Technically, DWL dynamically weights the learning losses of alignment and discriminability by introducing the degree of alignment and discriminability. Besides, the problem of sample imbalance across domains is first considered in our work, and we solve the problem by weighing the samples to guarantee information balance across domains. Extensive experiments demonstrate that DWL has an excellent performance in several benchmark datasets.

count=1
* CT-Net: Complementary Transfering Network for Garment Transfer With Arbitrary Geometric Changes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_CT-Net_Complementary_Transfering_Network_for_Garment_Transfer_With_Arbitrary_Geometric_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_CT-Net_Complementary_Transfering_Network_for_Garment_Transfer_With_Arbitrary_Geometric_CVPR_2021_paper.pdf)]
    * Title: CT-Net: Complementary Transfering Network for Garment Transfer With Arbitrary Geometric Changes
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Fan Yang, Guosheng Lin
    * Abstract: Garment transfer shows great potential in realistic applications with the goal of transfering outfits across different people images. However, garment transfer between images with heavy misalignments or severe occlusions still remains as a challenge. In this work, we propose Complementary Transfering Network (CT-Net) to adaptively model different levels of geometric changes and transfer outfits between different people. In specific, CT-Net consists of three modules: i) A complementary warping module first estimates two complementary warpings to transfer the desired clothes in different granularities. ii) A layout prediction module is proposed to predict the target layout, which guides the preservation or generation of the body parts in the synthesized images. iii) A dynamic fusion module adaptively combines the advantages of the complementary warpings to render the garment transfer results. Extensive experiments conducted on DeepFashion dataset demonstrate that our network synthesizes high-quality garment transfer images and significantly outperforms the state-of-art methods both qualitatively and quantitatively. Our source code will be available online.

count=1
* Deep Optimized Priors for 3D Shape Modeling and Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_Deep_Optimized_Priors_for_3D_Shape_Modeling_and_Reconstruction_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Deep_Optimized_Priors_for_3D_Shape_Modeling_and_Reconstruction_CVPR_2021_paper.pdf)]
    * Title: Deep Optimized Priors for 3D Shape Modeling and Reconstruction
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Mingyue Yang, Yuxin Wen, Weikai Chen, Yongwei Chen, Kui Jia
    * Abstract: Many learning-based approaches have difficulty scaling to unseen data, as the generality of its learned prior is limited to the scale and variations of the training samples. This holds particularly true with 3D learning tasks, given the sparsity of 3D datasets available. We introduce a new learning framework for 3D modeling and reconstruction that greatly improves the generalization ability of a deep generator. Our approach strives to connect the good ends of both learning-based and optimization-based methods. In particular, unlike the common practice that fixes the pre-trained priors at test time, we propose to further optimize the learned prior and latent code according to the input physical measurements after the training. We show that the proposed strategy effectively breaks the barriers constrained by the pre-trained priors and could lead to high-quality adaptation to unseen data. We realize our framework using the implicit surface representation and validate the efficacy of our approach in a variety of challenging tasks that take highly sparse or collapsed observations as input. Experimental results show that our approach compares favorably with the state-of-the-art methods in terms of both generality and accuracy.

count=1
* PISE: Person Image Synthesis and Editing With Decoupled GAN
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_PISE_Person_Image_Synthesis_and_Editing_With_Decoupled_GAN_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_PISE_Person_Image_Synthesis_and_Editing_With_Decoupled_GAN_CVPR_2021_paper.pdf)]
    * Title: PISE: Person Image Synthesis and Editing With Decoupled GAN
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Jinsong Zhang, Kun Li, Yu-Kun Lai, Jingyu Yang
    * Abstract: Person image synthesis, e.g., pose transfer, is a challenging problem due to large variation and occlusion. Existing methods have difficulties predicting reasonable invisible regions and fail to decouple the shape and style of clothing, which limits their applications on person image editing. In this paper, we propose PISE, a novel two-stage generative model for person image synthesis and editing, which can generate realistic person images with desired poses, textures, and semantic layouts. To better predict the invisible region, we first synthesize a human parsing map aligned with the target pose to represent the shape of clothing by a parsing generator, and then generate the final image by an image generator. To decouple the shape and style of clothing, we propose joint global and local per-region encoding and normalization to predict the reasonable style of clothing for invisible regions. We also propose spatial-aware normalization to retain the spatial context relationship in the source image. The results of qualitative and quantitative experiments demonstrate the superiority of our model. Besides, the results of texture transfer and parsing editing show that our model can be applied to person image editing.

count=1
* DAP: Detection-Aware Pre-Training With Weak Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhong_DAP_Detection-Aware_Pre-Training_With_Weak_Supervision_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_DAP_Detection-Aware_Pre-Training_With_Weak_Supervision_CVPR_2021_paper.pdf)]
    * Title: DAP: Detection-Aware Pre-Training With Weak Supervision
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yuanyi Zhong, Jianfeng Wang, Lijuan Wang, Jian Peng, Yu-Xiong Wang, Lei Zhang
    * Abstract: This paper presents a detection-aware pre-training (DAP) approach, which leverages only weakly-labeled classification-style datasets (e.g., ImageNet) for pre-training, but is specifically tailored to benefit object detection tasks. In contrast to the widely used image classification-based pre-training (e.g., on ImageNet), which does not include any location-related training tasks, we transform a classification dataset into a detection dataset through a weakly supervised object localization method based on Class Activation Maps to directly pre-train a detector, making the pre-trained model location-aware and capable of predicting bounding boxes. We show that DAP can outperform the traditional classification pre-training in terms of both sample efficiency and convergence speed in downstream detection tasks including VOC and COCO. In particular, DAP boosts the detection accuracy by a large margin when the number of examples in the downstream task is small.

count=1
* Where and What? Examining Interpretable Disentangled Representations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Where_and_What_Examining_Interpretable_Disentangled_Representations_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Where_and_What_Examining_Interpretable_Disentangled_Representations_CVPR_2021_paper.pdf)]
    * Title: Where and What? Examining Interpretable Disentangled Representations
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Xinqi Zhu, Chang Xu, Dacheng Tao
    * Abstract: Capturing interpretable variations has long been one of the goals in disentanglement learning. However, unlike the independence assumption, interpretability has rarely been exploited to encourage disentanglement in the unsupervised setting. In this paper, we examine the interpretability of disentangled representations by investigating two questions: where to be interpreted and what to be interpreted? A latent code is easily to be interpreted if it would consistently impact a certain subarea of the resulting generated image. We thus propose to learn a spatial mask to localize the effect of each individual latent dimension. On the other hand, interpretability usually comes from latent dimensions that capture simple and basic variations in data. We thus impose a perturbation on a certain dimension of the latent code, and expect to identify the perturbation along this dimension from the generated images so that the encoding of simple variations can be enforced. Additionally, we develop an unsupervised model selection method, which accumulates perceptual distance scores along axes in the latent space. On various datasets, our models can learn high-quality disentangled representations without supervision, showing the proposed modeling of interpretability is an effective proxy for achieving unsupervised disentanglement.

count=1
* Progressive Temporal Feature Alignment Network for Video Inpainting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zou_Progressive_Temporal_Feature_Alignment_Network_for_Video_Inpainting_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zou_Progressive_Temporal_Feature_Alignment_Network_for_Video_Inpainting_CVPR_2021_paper.pdf)]
    * Title: Progressive Temporal Feature Alignment Network for Video Inpainting
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Xueyan Zou, Linjie Yang, Ding Liu, Yong Jae Lee
    * Abstract: Video inpainting aims to fill spatio-temporal "corrupted" regions with plausible content. To achieve this goal, it is necessary to find correspondences from neighbouring frames to faithfully hallucinate the unknown content. Current methods achieve this goal through attention, flow-based warping, or 3D temporal convolution. However, flow-based warping can create artifacts when optical flow is not accurate, while temporal convolution may suffer from spatial misalignment. We propose `Progressive Temporal Feature Alignment Network', which progressively enriches features extracted from the current frame with the feature warped from neighbouring frames using optical flow. Our approach corrects the spatial misalignment in the temporal feature propagation stage, greatly improving visual quality and temporal consistency of the inpainted videos. Using the proposed architecture, we achieve state-of-the-art performance on the DAVIS and FVI datasets compared to existing deep learning approaches. Code is available at https://github.com/MaureenZOU/TSAM.

count=1
* On-Orbit Inspection of an Unknown, Tumbling Target Using NASA's Astrobee Robotic Free-Flyers
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Oestreich_On-Orbit_Inspection_of_an_Unknown_Tumbling_Target_Using_NASAs_Astrobee_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Oestreich_On-Orbit_Inspection_of_an_Unknown_Tumbling_Target_Using_NASAs_Astrobee_CVPRW_2021_paper.pdf)]
    * Title: On-Orbit Inspection of an Unknown, Tumbling Target Using NASA's Astrobee Robotic Free-Flyers
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Charles Oestreich, Antonio Teran Espinoza, Jessica Todd, Keenan Albee, Richard Linares
    * Abstract: Autonomous spacecraft critically depend on on-orbit inspection (i.e., relative navigation and inertial properties estimation) to intercept tumbling debris objects or defunct satellites. This work presents a practical method for on-orbit inspection and demonstrates its performance in simulation using NASA's Astrobee robotic free-flyers. The problem is formulated as a simultaneous localization and mapping task, utilizing IMU data from an observing "chaser" spacecraft and point clouds of the observed "target" spacecraft obtained via a 3D time-of-flight camera. The relative navigation between the chaser and target is solved via a factor graph-based approach. The target's principal axes of inertia are then estimated via a conic fit optimization procedure using a polhode analysis. Simulation results indicate the accuracy of the proposed method in preparation for hardware experiments on the International Space Station.

count=1
* Region-Adaptive Deformable Network for Image Quality Assessment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Shi_Region-Adaptive_Deformable_Network_for_Image_Quality_Assessment_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Shi_Region-Adaptive_Deformable_Network_for_Image_Quality_Assessment_CVPRW_2021_paper.pdf)]
    * Title: Region-Adaptive Deformable Network for Image Quality Assessment
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Shuwei Shi, Qingyan Bai, Mingdeng Cao, Weihao Xia, Jiahao Wang, Yifan Chen, Yujiu Yang
    * Abstract: Image quality assessment (IQA) aims to assess the perceptual quality of images. The outputs of the IQA algorithms are expected to be consistent with human subjective perception. In image restoration and enhancement tasks, images generated by generative adversarial networks (GAN) can achieve better visual performance than traditional CNN-generated images, although they have spatial shift and texture noise. Unfortunately, the existing IQA methods have unsatisfactory performance on the GAN-based distortion partially because of their low tolerance to spatial misalignment. To this end, we propose the reference-oriented deformable convolution, which can improve the performance of an IQA network on GAN-based distortion by adaptively considering this misalignment. We further propose a patch-level attention module to enhance the interaction among different patch regions, which are processed independently in previous patch-based methods. The modified residual block is also proposed by applying modifications to the classic residual block to construct a patch-region-based baseline called WResNet. Equipping this baseline with the two proposed modules, we further propose Region-Adaptive Deformable Network (RADN). The experiment results on the NTIRE 2021 Perceptual Image Quality Assessment Challenge dataset show the superior performance of RADN, and the ensemble approach won fourth place in the final testing phase of the challenge.

count=1
* Dissecting the High-Frequency Bias in Convolutional Neural Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/UG2/html/Abello_Dissecting_the_High-Frequency_Bias_in_Convolutional_Neural_Networks_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/UG2/papers/Abello_Dissecting_the_High-Frequency_Bias_in_Convolutional_Neural_Networks_CVPRW_2021_paper.pdf)]
    * Title: Dissecting the High-Frequency Bias in Convolutional Neural Networks
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Antonio A. Abello, Roberto Hirata, Zhangyang Wang
    * Abstract: For convolutional neural networks (CNNs), a common hypothesis that explains both their capability of generalization and their characteristic brittleness is that these models are implicitly regularized to rely on imperceptible high-frequency patterns, more than humans would do. This hypothesis has seen some empirical validation, but most works do not rigorously divide the image frequency spectrum. We present a model to divide the spectrum in disjointed discs based on the distribution of energy and apply simple feature importance procedures to test whether high-frequencies are more important than lower ones. We find evidence that mid or high-level frequencies are disproportionately important for CNNs. The evidence is robust across different datasets and networks. Moreover, we find the diverse effects of the network's attributes, such as architecture and depth, on frequency bias and robustness in general. Code for reproducing our experiments is available at: https://github.com/Abello966/FrequencyBiasExperiments

count=1
* Neural RGB-D Surface Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Azinovic_Neural_RGB-D_Surface_Reconstruction_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Azinovic_Neural_RGB-D_Surface_Reconstruction_CVPR_2022_paper.pdf)]
    * Title: Neural RGB-D Surface Reconstruction
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Dejan Azinović, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, Justus Thies
    * Abstract: Obtaining high-quality 3D reconstructions of room-scale scenes is of paramount importance for upcoming applications in AR or VR. These range from mixed reality applications for teleconferencing, virtual measuring, virtual room planing, to robotic applications. While current volume-based view synthesis methods that use neural radiance fields (NeRFs) show promising results in reproducing the appearance of an object or scene, they do not reconstruct an actual surface. The volumetric representation of the surface based on densities leads to artifacts when a surface is extracted using Marching Cubes, since during optimization, densities are accumulated along the ray and are not used at a single sample point in isolation. Instead of this volumetric representation of the surface, we propose to represent the surface using an implicit function (truncated signed distance function). We show how to incorporate this representation in the NeRF framework, and extend it to use depth measurements from a commodity RGB-D sensor, such as a Kinect. In addition, we propose a pose and camera refinement technique which improves the overall reconstruction quality. In contrast to concurrent work on integrating depth priors in NeRF which concentrates on novel view synthesis, our approach is able to reconstruct high-quality, metrical 3D reconstructions.

count=1
* Incorporating Semi-Supervised and Positive-Unlabeled Learning for Boosting Full Reference Image Quality Assessment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Cao_Incorporating_Semi-Supervised_and_Positive-Unlabeled_Learning_for_Boosting_Full_Reference_Image_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_Incorporating_Semi-Supervised_and_Positive-Unlabeled_Learning_for_Boosting_Full_Reference_Image_CVPR_2022_paper.pdf)]
    * Title: Incorporating Semi-Supervised and Positive-Unlabeled Learning for Boosting Full Reference Image Quality Assessment
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yue Cao, Zhaolin Wan, Dongwei Ren, Zifei Yan, Wangmeng Zuo
    * Abstract: Full-reference (FR) image quality assessment (IQA) evaluates the visual quality of a distorted image by measuring its perceptual difference with pristine-quality reference, and has been widely used in low level vision tasks. Pairwise labeled data with mean opinion score (MOS) are required in training FR-IQA model, but is time-consuming and cumbersome to collect. In contrast, unlabeled data can be easily collected from an image degradation or restoration process, making it encouraging to exploit unlabeled training data to boost FR-IQA performance. Moreover, due to the distribution inconsistency between labeled and unlabeled data, outliers may occur in unlabeled data, further increasing the training difficulty. In this paper, we suggest to incorporate semi-supervised and positive-unlabeled (PU) learning for exploiting unlabeled data while mitigating the adverse effect of outliers. Particularly, by treating all labeled data as positive samples, PU learning is leveraged to identify negative samples (i.e., outliers) from unlabeled data. Semi-supervised learning (SSL) is further deployed to exploit positive unlabeled data by dynamically generating pseudo-MOS. We adopt a dual-branch network including reference and distortion branches. Furthermore, spatial attention is introduced in the reference branch to concentrate more on the informative regions, and sliced Wasserstein distance is used for robust difference map computation to address the misalignment issues caused by images recovered by GAN models. Extensive experiments show that our method performs favorably against state-of-the-arts on the benchmark datasets PIPAL, KADID-10k, TID2013, LIVE and CSIQ.

count=1
* BasicVSR++: Improving Video Super-Resolution With Enhanced Propagation and Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.pdf)]
    * Title: BasicVSR++: Improving Video Super-Resolution With Enhanced Propagation and Alignment
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, Chen Change Loy
    * Abstract: A recurrent structure is a popular framework choice for the task of video super-resolution. The state-of-the-art method BasicVSR adopts bidirectional propagation with feature alignment to effectively exploit information from the entire input video. In this study, we redesign BasicVSR by proposing second-order grid propagation and flow-guided deformable alignment. We show that by empowering the recurrent framework with enhanced propagation and alignment, one can exploit spatiotemporal information across misaligned video frames more effectively. The new components lead to an improved performance under a similar computational constraint. In particular, our model BasicVSR++ surpasses BasicVSR by a significant 0.82 dB in PSNR with similar number of parameters. BasicVSR++ is generalizable to other video restoration tasks, and obtains three champions and one first runner-up in NTIRE 2021 video restoration challenge.

count=1
* Focal Sparse Convolutional Networks for 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Focal_Sparse_Convolutional_Networks_for_3D_Object_Detection_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Focal_Sparse_Convolutional_Networks_for_3D_Object_Detection_CVPR_2022_paper.pdf)]
    * Title: Focal Sparse Convolutional Networks for 3D Object Detection
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yukang Chen, Yanwei Li, Xiangyu Zhang, Jian Sun, Jiaya Jia
    * Abstract: Non-uniformed 3D sparse data, e.g., point clouds or voxels in different spatial positions, make contribution to the task of 3D object detection in different ways. Existing basic components in sparse convolutional networks (Sparse CNNs) process all sparse data, regardless of regular or submanifold sparse convolution. In this paper, we introduce two new modules to enhance the capability of Sparse CNNs, both are based on making feature sparsity learnable with position-wise importance prediction. They are focal sparse convolution (Focals Conv) and its multi-modal variant of focal sparse convolution with fusion, or Focals Conv-F for short. The new modules can readily substitute their plain counterparts in existing Sparse CNNs and be jointly trained in an end-to-end fashion. For the first time, we show that spatially learnable sparsity in sparse convolution is essential for sophisticated 3D object detection. Extensive experiments on the KITTI, nuScenes and Waymo benchmarks validate the effectiveness of our approach. Without bells and whistles, our results outperform all existing single-model entries on the nuScenes test benchmark.

count=1
* Geometry-Aware Guided Loss for Deep Crack Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Geometry-Aware_Guided_Loss_for_Deep_Crack_Recognition_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Geometry-Aware_Guided_Loss_for_Deep_Crack_Recognition_CVPR_2022_paper.pdf)]
    * Title: Geometry-Aware Guided Loss for Deep Crack Recognition
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Zhuangzhuang Chen, Jin Zhang, Zhuonan Lai, Jie Chen, Zun Liu, Jianqiang Li
    * Abstract: Despite the substantial progress of deep models for crack recognition, due to the inconsistent cracks in varying sizes, shapes, and noisy background textures, there still lacks the discriminative power of the deeply learned features when supervised by the cross-entropy loss. In this paper, we propose the geometry-aware guided loss (GAGL) that enhances the discrimination ability and is only applied in the training stage without extra computation and memory during inference. The GAGL consists of the feature-based geometry-aware projected gradient descent method (FGA-PGD) that approximates the geometric distances of the features to the class boundaries, and the geometry-aware update rule that learns an anchor of each class as the approximation of the feature expected to have the largest geometric distance to the corresponding class boundary. Then the discriminative power can be enhanced by minimizing the distances between the features and their corresponding class anchors in the feature space. To address the limited availability of related benchmarks, we collect a fully annotated dataset, namely, NPP2021, which involves inconsistent cracks and noisy backgrounds in real-world nuclear power plants. Our proposed GAGL outperforms the state of the arts on various benchmark datasets including CRACK2019, SDNET2018, and our NPP2021.

count=1
* The Implicit Values of a Good Hand Shake: Handheld Multi-Frame Neural Depth Refinement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Chugunov_The_Implicit_Values_of_a_Good_Hand_Shake_Handheld_Multi-Frame_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Chugunov_The_Implicit_Values_of_a_Good_Hand_Shake_Handheld_Multi-Frame_CVPR_2022_paper.pdf)]
    * Title: The Implicit Values of a Good Hand Shake: Handheld Multi-Frame Neural Depth Refinement
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Ilya Chugunov, Yuxuan Zhang, Zhihao Xia, Xuaner Zhang, Jiawen Chen, Felix Heide
    * Abstract: Modern smartphones can continuously stream multi-megapixel RGB images at 60Hz, synchronized with high-quality 3D pose information and low-resolution LiDAR-driven depth estimates. During a snapshot photograph, the natural unsteadiness of the photographer's hands offers millimeter-scale variation in camera pose, which we can capture along with RGB and depth in a circular buffer. In this work we explore how, from a bundle of these measurements acquired during viewfinding, we can combine dense micro-baseline parallax cues with kilopixel LiDAR depth to distill a high-fidelity depth map. We take a test-time optimization approach and train a coordinate MLP to output photometrically and geometrically consistent depth estimates at the continuous coordinates along the path traced by the photographer's natural hand shake. With no additional hardware, artificial hand motion, or user interaction beyond the press of a button, our proposed method brings high-resolution depth estimates to point-and-shoot "tabletop" photography -- textured objects at close range.

count=1
* Dressing in the Wild by Watching Dance Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Dressing_in_the_Wild_by_Watching_Dance_Videos_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Dressing_in_the_Wild_by_Watching_Dance_Videos_CVPR_2022_paper.pdf)]
    * Title: Dressing in the Wild by Watching Dance Videos
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Xin Dong, Fuwei Zhao, Zhenyu Xie, Xijin Zhang, Daniel K. Du, Min Zheng, Xiang Long, Xiaodan Liang, Jianchao Yang
    * Abstract: While significant progress has been made in garment transfer, one of the most applicable directions of human-centric image generation, existing works overlook the in-the-wild imagery, presenting severe garment-person misalignment as well as noticeable degradation in fine texture details. This paper, therefore, attends to virtual try-on in real-world scenes and brings essential improvements in authenticity and naturalness especially for loose garment (e.g., skirts, formal dresses), challenging poses (e.g., cross arms, bent legs), and cluttered backgrounds. Specifically, we find that the pixel flow excels at handling loose garments whereas the vertex flow is preferred for hard poses, and by combining their advantages we propose a novel generative network called wFlow that can effectively push up garment transfer to in-the-wild context. Moreover, former approaches require paired images for training. Instead, we cut down the laboriousness by working on a newly constructed large-scale video dataset named Dance50k with self-supervised cross-frame training and an online cycle optimization. The proposed Dance50k can boost real-world virtual dressing by covering a wide variety of garments under dancing poses. Extensive experiments demonstrate the superiority of our wFlow in generating realistic garment transfer results for in-the-wild images without resorting to expensive paired datasets.

count=1
* UCC: Uncertainty Guided Cross-Head Co-Training for Semi-Supervised Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Fan_UCC_Uncertainty_Guided_Cross-Head_Co-Training_for_Semi-Supervised_Semantic_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_UCC_Uncertainty_Guided_Cross-Head_Co-Training_for_Semi-Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf)]
    * Title: UCC: Uncertainty Guided Cross-Head Co-Training for Semi-Supervised Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jiashuo Fan, Bin Gao, Huan Jin, Lihui Jiang
    * Abstract: Deep neural networks (DNNs) have witnessed great successes in semantic segmentation, which requires a large number of labeled data for training. We present a novel learning framework called Uncertainty guided Cross-head Co-training (UCC) for semi-supervised semantic segmentation. Our framework introduces weak and strong augmentations within a shared encoder to achieve co-training, which naturally combines the benefits of consistency and self-training. Every segmentation head interacts with its peers and, the weak augmentation result is used for supervising the strong. The consistency training samples' diversity can be boosted by Dynamic Cross-Set Copy-Paste (DCSCP), which also alleviates the distribution mismatch and class imbalance problems. Moreover, our proposed Uncertainty Guided Re-weight Module (UGRM) enhances the self-training pseudo labels by suppressing the effect of the low-quality pseudo labels from its peer via modeling uncertainty. Extensive experiments on Cityscapes and PASCAL VOC 2012 demonstrate the effectiveness of our UCC, our approach significantly outperforms other state-of-the-art semi-supervised semantic segmentation methods. It achieves 77.17%, 76.49% mIoU on Cityscapes and PASCAL VOC 2012 datasets respectively under 1/16 protocols, which are +10.1%, +7.91% better than the supervised baseline.

count=1
* Learning Where To Learn in Cross-View Self-Supervised Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Learning_Where_To_Learn_in_Cross-View_Self-Supervised_Learning_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learning_Where_To_Learn_in_Cross-View_Self-Supervised_Learning_CVPR_2022_paper.pdf)]
    * Title: Learning Where To Learn in Cross-View Self-Supervised Learning
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Lang Huang, Shan You, Mingkai Zheng, Fei Wang, Chen Qian, Toshihiko Yamasaki
    * Abstract: Self-supervised learning (SSL) has made enormous progress and largely narrowed the gap with the supervised ones, where the representation learning is mainly guided by a projection into an embedding space. During the projection, current methods simply adopt uniform aggregation of pixels for embedding; however, this risks involving object-irrelevant nuisances and spatial misalignment for different augmentations. In this paper, we present a new approach, Learning Where to Learn (LEWEL), to adaptively aggregate spatial information of features, so that the projected embeddings could be exactly aligned and thus guide the feature learning better. Concretely, we reinterpret the projection head in SSL as a per-pixel projection and predict a set of spatial alignment maps from the original features by this weight-sharing projection head. A spectrum of aligned embeddings is thus obtained by aggregating the features with spatial weighting according to these alignment maps. As a result of this adaptive alignment, we observe substantial improvements on both image-level prediction and dense prediction at the same time: LEWEL improves MoCov2 by 1.6%/1.3%/0.5%/0.4% points, improves BYOL by 1.3%/1.3%/0.7%/0.6% points, on ImageNet linear/semi-supervised classification, Pascal VOC semantic segmentation, and object detection, respectively.

count=1
* Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Jin_Deformation_and_Correspondence_Aware_Unsupervised_Synthetic-to-Real_Scene_Flow_Estimation_for_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Jin_Deformation_and_Correspondence_Aware_Unsupervised_Synthetic-to-Real_Scene_Flow_Estimation_for_CVPR_2022_paper.pdf)]
    * Title: Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Zhao Jin, Yinjie Lei, Naveed Akhtar, Haifeng Li, Munawar Hayat
    * Abstract: Point cloud scene flow estimation is of practical importance for dynamic scene navigation in autonomous driving. Since scene flow labels are hard to obtain, current methods train their models on synthetic data and transfer them to real scenes. However, large disparities between existing synthetic datasets and real scenes lead to poor model transfer. We make two major contributions to address that. First, we develop a point cloud collector and scene flow annotator for GTA-V engine to automatically obtain diverse realistic training samples without human intervention. With that, we develop a large-scale synthetic scene flow dataset GTA-SF. Second, we propose a mean-teacher-based domain adaptation framework that leverages self-generated pseudo-labels of the target domain. It also explicitly incorporates shape deformation regularization and surface correspondence refinement to address distortions and misalignments in domain transfer. Through extensive experiments, we show that our GTA-SF dataset leads to a consistent boost in model generalization to three real datasets (i.e., Waymo, Lyft and KITTI) as compared to the most widely used FT3D dataset. Moreover, our framework achieves superior adaptation performance on six source-target dataset pairs, remarkably closing the average domain gap by 60%. Data and codes are available at https://github.com/leolyj/DCA-SRSFE

count=1
* Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Liao_Graph_Sampling_Based_Deep_Metric_Learning_for_Generalizable_Person_Re-Identification_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liao_Graph_Sampling_Based_Deep_Metric_Learning_for_Generalizable_Person_Re-Identification_CVPR_2022_paper.pdf)]
    * Title: Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Shengcai Liao, Ling Shao
    * Abstract: Recent studies show that, both explicit deep feature matching as well as large-scale and diverse training data can significantly improve the generalization of person re-identification. However, the efficiency of learning deep matchers on large-scale data has not yet been adequately studied. Though learning with classification parameters or class memory is a popular way, it incurs large memory and computational costs. In contrast, pairwise deep metric learning within mini batches would be a better choice. However, the most popular random sampling method, the well-known PK sampler, is not informative and efficient for deep metric learning. Though online hard example mining has improved the learning efficiency to some extent, the mining in mini batches after random sampling is still limited. This inspires us to explore the use of hard example mining earlier, in the data sampling stage. To do so, in this paper, we propose an efficient mini-batch sampling method, called graph sampling (GS), for large-scale deep metric learning. The basic idea is to build a nearest neighbor relationship graph for all classes at the beginning of each epoch. Then, each mini batch is composed of a randomly selected class and its nearest neighboring classes so as to provide informative and challenging examples for learning. Together with an adapted competitive baseline, we improve the state of the art in generalizable person re-identification significantly, by 25.1% in Rank-1 on MSMT17 when trained on RandPerson. Besides, the proposed method also outperforms the competitive baseline, by 6.8% in Rank-1 on CUHK03-NP when trained on MSMT17. Meanwhile, the training time is significantly reduced, from 25.4 hours to 2 hours when trained on RandPerson with 8,000 identities. Code is available at https://github.com/ShengcaiLiao/QAConv.

count=1
* BCOT: A Markerless High-Precision 3D Object Tracking Benchmark
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Li_BCOT_A_Markerless_High-Precision_3D_Object_Tracking_Benchmark_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_BCOT_A_Markerless_High-Precision_3D_Object_Tracking_Benchmark_CVPR_2022_paper.pdf)]
    * Title: BCOT: A Markerless High-Precision 3D Object Tracking Benchmark
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jiachen Li, Bin Wang, Shiqiang Zhu, Xin Cao, Fan Zhong, Wenxuan Chen, Te Li, Jason Gu, Xueying Qin
    * Abstract: Template-based 3D object tracking still lacks a high-precision benchmark of real scenes due to the difficulty of annotating the accurate 3D poses of real moving video objects without using markers. In this paper, we present a multi-view approach to estimate the accurate 3D poses of real moving objects, and then use binocular data to construct a new benchmark for monocular textureless 3D object tracking. The proposed method requires no markers, and the cameras only need to be synchronous, relatively fixed as cross-view and calibrated. Based on our object-centered model, we jointly optimize the object pose by minimizing shape re-projection constraints in all views, which greatly improves the accuracy compared with the single-view approach, and is even more accurate than the depth-based method. Our new benchmark dataset contains 20 textureless objects, 22 scenes, 404 video sequences and 126K images captured in real scenes. The annotation error is guaranteed to be less than 2mm, according to both theoretical analysis and validation experiments. We re-evaluate the state-of-the-art 3D object tracking methods with our dataset, reporting their performance ranking in real scenes. Our BCOT benchmark and code can be found at https://ar3dv.github.io/BCOT-Benchmark/.

count=1
* Reduce Information Loss in Transformers for Pluralistic Image Inpainting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Reduce_Information_Loss_in_Transformers_for_Pluralistic_Image_Inpainting_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Reduce_Information_Loss_in_Transformers_for_Pluralistic_Image_Inpainting_CVPR_2022_paper.pdf)]
    * Title: Reduce Information Loss in Transformers for Pluralistic Image Inpainting
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Qiankun Liu, Zhentao Tan, Dongdong Chen, Qi Chu, Xiyang Dai, Yinpeng Chen, Mengchen Liu, Lu Yuan, Nenghai Yu
    * Abstract: Transformers have achieved great success in pluralistic image inpainting recently. However, we find existing transformer based solutions regard each pixel as a token, thus suffer from information loss issue from two aspects: 1) They downsample the input image into much lower resolutions for efficiency consideration, incurring information loss and extra misalignment for the boundaries of masked regions. 2) They quantize 2563 RGB pixels to a small number (such as 512) of quantized pixels. The indices of quantized pixels are used as tokens for the inputs and prediction targets of transformer. Although an extra CNN network is used to upsample and refine the low-resolution results, it is difficult to retrieve the lost information back. To keep input information as much as possible, we propose a new transformer based framework "PUT". Specifically, to avoid input downsampling while maintaining the computation efficiency, we design a patch-based auto-encoder PVQVAE, where the encoder converts the masked image into non-overlapped patch tokens and the decoder recovers the masked regions from the inpainted tokens while keeping the unmasked regions unchanged. To eliminate the information loss caused by quantization, an Un-Quantized Transformer (UQ-Transformer) is applied, which directly takes the features from P-VQVAE encoder as input without quantization and regards the quantized tokens only as prediction targets. Extensive experiments show that PUT greatly outperforms state-of-the-art methods on image fidelity, especially for large masked regions and complex large-scale datasets.

count=1
* Temporal Feature Alignment and Mutual Information Maximization for Video-Based Human Pose Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Temporal_Feature_Alignment_and_Mutual_Information_Maximization_for_Video-Based_Human_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Temporal_Feature_Alignment_and_Mutual_Information_Maximization_for_Video-Based_Human_CVPR_2022_paper.pdf)]
    * Title: Temporal Feature Alignment and Mutual Information Maximization for Video-Based Human Pose Estimation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Zhenguang Liu, Runyang Feng, Haoming Chen, Shuang Wu, Yixing Gao, Yunjun Gao, Xiang Wang
    * Abstract: Multi-frame human pose estimation has long been a compelling and fundamental problem in computer vision. This task is challenging due to fast motion and pose occlusion that frequently occur in videos. State-of-the-art methods strive to incorporate additional visual evidences from neighboring frames (supporting frames) to facilitate the pose estimation of the current frame (key frame). One aspect that has been obviated so far, is the fact that current methods directly aggregate unaligned contexts across frames. The spatial-misalignment between pose features of the current frame and neighboring frames might lead to unsatisfactory results. More importantly, existing approaches build upon the straightforward pose estimation loss, which unfortunately cannot constrain the network to fully leverage useful information from neighboring frames. To tackle these problems, we present a novel hierarchical alignment framework, which leverages coarse-to-fine deformations to progressively update a neighboring frame to align with the current frame at the feature level. We further propose to explicitly supervise the knowledge extraction from neighboring frames, guaranteeing that useful complementary cues are extracted. To achieve this goal, we theoretically analyzed the mutual information between the frames and arrived at a loss that maximizes the taskrelevant mutual information. These allow us to rank No.1 in the Multi-frame Person Pose Estimation Challenge on benchmark dataset PoseTrack2017, and obtain state-of-the-art performance on benchmarks Sub-JHMDB and PoseTrack2018. Our code is released at https://github.com/Pose-Group/FAMI-Pose, hoping that it will be useful to the community.

count=1
* Unsupervised Vision-Language Parsing: Seamlessly Bridging Visual Scene Graphs With Language Structures via Dependency Relationships
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Lou_Unsupervised_Vision-Language_Parsing_Seamlessly_Bridging_Visual_Scene_Graphs_With_Language_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Lou_Unsupervised_Vision-Language_Parsing_Seamlessly_Bridging_Visual_Scene_Graphs_With_Language_CVPR_2022_paper.pdf)]
    * Title: Unsupervised Vision-Language Parsing: Seamlessly Bridging Visual Scene Graphs With Language Structures via Dependency Relationships
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Chao Lou, Wenjuan Han, Yuhuan Lin, Zilong Zheng
    * Abstract: Understanding realistic visual scene images together with language descriptions is a fundamental task towards generic visual understanding. Previous works have shown compelling comprehensive results by building hierarchical structures for visual scenes (e.g., scene graphs) and natural languages (e.g., dependency trees), individually. However, how to construct a joint vision-language (VL) structure has barely been investigated. More challenging but worthwhile, we introduce a new task that targets on inducing such a joint VL structure in an unsupervised manner. Our goal is to bridge the visual scene graphs and linguistic dependency trees seamlessly. Due to the lack of VL structural data, we start by building a new dataset VLParse. Rather than using labor-intensive labeling from scratch, we propose an automatic alignment procedure to produce coarse structures followed by human refinement to produce high-quality ones. Moreover, we benchmark our dataset by proposing a contrastive learning (CL)-based framework VLGAE, short for Vision-Language Graph Autoencoder. Our model obtains superior performance on two derived tasks, i.e., language grammar induction and VL phrase grounding. Ablations show the effectiveness of both visual cues and dependency relationships on fine-grained VL structure construction.

count=1
* A Unified Model for Line Projections in Catadioptric Cameras With Rotationally Symmetric Mirrors
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Miraldo_A_Unified_Model_for_Line_Projections_in_Catadioptric_Cameras_With_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Miraldo_A_Unified_Model_for_Line_Projections_in_Catadioptric_Cameras_With_CVPR_2022_paper.pdf)]
    * Title: A Unified Model for Line Projections in Catadioptric Cameras With Rotationally Symmetric Mirrors
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Pedro Miraldo, José Pedro Iglesias
    * Abstract: Lines are among the most used computer vision features, in applications such as camera calibration to object detection. Catadioptric cameras with rotationally symmetric mirrors are omnidirectional imaging devices, capturing up to a 360 degrees field of view. These are used in many applications ranging from robotics to panoramic vision. Although known for some specific configurations, the modeling of line projection was never fully solved for general central and non-central catadioptric cameras. We start by taking some general point reflection assumptions and derive a line reflection constraint. This constraint is then used to define a line projection into the image. Next, we compare our model with previous methods, showing that our general approach outputs the same polynomial degrees as previous configuration-specific systems. We run several experiments using synthetic and real-world data, validating our line projection model. Lastly, we show an application of our methods to an absolute camera pose problem.

count=1
* Neural Convolutional Surfaces
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Morreale_Neural_Convolutional_Surfaces_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Morreale_Neural_Convolutional_Surfaces_CVPR_2022_paper.pdf)]
    * Title: Neural Convolutional Surfaces
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Luca Morreale, Noam Aigerman, Paul Guerrero, Vladimir G. Kim, Niloy J. Mitra
    * Abstract: This work is concerned with representation of shapes while disentangling fine, local and possibly repeating geometry, from global, coarse structures. Achieving such disentanglement leads to two unrelated advantages: i) a significant compression in the number of parameters required to represent a given geometry; ii) the ability to manipulate either global geometry, or local details, without harming the other. At the core of our approach lies a novel pipeline and neural architecture, which are optimized to represent one specific atlas, representing one 3D surface. Our pipeline and architecture are designed so that disentanglement of global geometry from local details is accomplished through optimization, in a completely unsupervised manner. We show that this approach achieves better neural shape compression than the state of the art, as well as enabling manipulation and transfer of shape details.

count=1
* NAN
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Pearl_NAN_Noise-Aware_NeRFs_for_Burst-Denoising_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Pearl_NAN_Noise-Aware_NeRFs_for_Burst-Denoising_CVPR_2022_paper.pdf)]
    * Title: NAN: Noise-Aware NeRFs for Burst-Denoising
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Naama Pearl, Tali Treibitz, Simon Korman
    * Abstract: Burst denoising is now more relevant than ever, as computational photography helps overcome sensitivity issues inherent in mobile phones and small cameras. A major challenge in burst-denoising is in coping with pixel misalignment, which was so far handled with rather simplistic assumptions of simple motion, or the ability to align in pre-processing. Such assumptions are not realistic in the presence of large motion and high levels of noise. We show that Neural Radiance Fields (NeRFs), originally suggested for physics-based novel-view rendering, can serve as a powerful framework for burst denoising. NeRFs have an inherent capability of handling noise as they integrate information from multiple images, but they are limited in doing so, mainly since they build on pixel-wise operations which are suitable to ideal imaging conditions. Our approach, termed NAN, leverages inter-view and spatial information in NeRFs to better deal with noise. It achieves state-of-the-art results in burst denoising and is especially successful in coping with large movement and occlusions, under very high levels of noise. With the rapid advances in accelerating NeRFs, it could provide a powerful platform for denoising in challenging environments.

count=1
* Mining Multi-View Information: A Strong Self-Supervised Framework for Depth-Based 3D Hand Pose and Mesh Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Mining_Multi-View_Information_A_Strong_Self-Supervised_Framework_for_Depth-Based_3D_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_Mining_Multi-View_Information_A_Strong_Self-Supervised_Framework_for_Depth-Based_3D_CVPR_2022_paper.pdf)]
    * Title: Mining Multi-View Information: A Strong Self-Supervised Framework for Depth-Based 3D Hand Pose and Mesh Estimation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Pengfei Ren, Haifeng Sun, Jiachang Hao, Jingyu Wang, Qi Qi, Jianxin Liao
    * Abstract: In this work, we study the cross-view information fusion problem in the task of self-supervised 3D hand pose estimation from the depth image. Previous methods usually adopt a hand-crafted rule to generate pseudo labels from multi-view estimations in order to supervise the network training in each view. However, these methods ignore the rich semantic information in each view and ignore the complex dependencies between different regions of different views. To solve these problems, we propose a cross-view fusion network to fully exploit and adaptively aggregate multi-view information. We encode diverse semantic information in each view into multiple compact nodes. Then, we introduce the graph convolution to model the complex dependencies between nodes and perform cross-view information interaction. Based on the cross-view fusion network, we propose a strong self-supervised framework for 3D hand pose and hand mesh estimation. Furthermore, we propose a pseudo multi-view training strategy to extend our framework to a more general scenario in which only single-view training data is used. Results on NYU dataset demonstrate that our method outperforms the previous self-supervised methods by 17.5% and 30.3% in multi-view and single-view scenarios. Meanwhile, our framework achieves comparable results to several strongly supervised methods.

count=1
* Self-Supervised Bulk Motion Artifact Removal in Optical Coherence Tomography Angiography
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Self-Supervised_Bulk_Motion_Artifact_Removal_in_Optical_Coherence_Tomography_Angiography_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_Self-Supervised_Bulk_Motion_Artifact_Removal_in_Optical_Coherence_Tomography_Angiography_CVPR_2022_paper.pdf)]
    * Title: Self-Supervised Bulk Motion Artifact Removal in Optical Coherence Tomography Angiography
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jiaxiang Ren, Kicheon Park, Yingtian Pan, Haibin Ling
    * Abstract: Optical coherence tomography angiography (OCTA) is an important imaging modality in many bioengineering tasks. The image quality of OCTA, however, is often degraded by Bulk Motion Artifacts (BMA), which are due to micromotion of subjects and typically appear as bright stripes surrounded by blurred areas. State-of-the-art methods usually treat BMA removal as a learning-based image inpainting problem, but require numerous training samples with nontrivial annotation. In addition, these methods discard the rich structural and appearance information carried in the BMA stripe region. To address these issues, in this paper we propose a self-supervised content-aware BMA removal model. First, the gradient-based structural information and appearance feature are extracted from the BMA area and injected into the model to capture more connectivity. Second, with easily collected defective masks, the model is trained in a self-supervised manner, in which only the clear areas are used for training while the BMA areas for inference. With the structural information and appearance feature from noisy image as references, our model can remove larger BMA and produce better visualizing result. In addition, only 2D images with defective masks are involved, hence improving the efficiency of our method. Experiments on OCTA of mouse cortex demonstrate that our model can remove most BMA with extremely large sizes and inconsistent intensities while previous methods fail.

count=1
* Structure-Aware Flow Generation for Human Body Reshaping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Structure-Aware_Flow_Generation_for_Human_Body_Reshaping_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_Structure-Aware_Flow_Generation_for_Human_Body_Reshaping_CVPR_2022_paper.pdf)]
    * Title: Structure-Aware Flow Generation for Human Body Reshaping
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jianqiang Ren, Yuan Yao, Biwen Lei, Miaomiao Cui, Xuansong Xie
    * Abstract: Body reshaping is an important procedure in portrait photo retouching. Due to the complicated structure and multifarious appearance of human bodies, existing methods either fall back on the 3D domain via body morphable model or resort to keypoint-based image deformation, leading to inefficiency and unsatisfied visual quality. In this paper, we address these limitations by formulating an end-to-end flow generation architecture under the guidance of body structural priors, including skeletons and Part Affinity Fields, and achieve unprecedentedly controllable performance under arbitrary poses and garments. A compositional attention mechanism is introduced for capturing both visual perceptual correlations and structural associations of the human body to reinforce the manipulation consistency among related parts. For a comprehensive evaluation, we construct the first large-scale body reshaping dataset, namely BR-5K, which contains 5,000 portrait photos as well as professionally retouched targets. Extensive experiments demonstrate that our approach significantly outperforms existing state-of-the-art methods in terms of visual performance, controllability, and efficiency. The dataset is available at our website: https://github.com/JianqiangRen/FlowBasedBodyReshaping.

count=1
* End-to-End Multi-Person Pose Estimation With Transformers
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Shi_End-to-End_Multi-Person_Pose_Estimation_With_Transformers_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Shi_End-to-End_Multi-Person_Pose_Estimation_With_Transformers_CVPR_2022_paper.pdf)]
    * Title: End-to-End Multi-Person Pose Estimation With Transformers
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Dahu Shi, Xing Wei, Liangqi Li, Ye Ren, Wenming Tan
    * Abstract: Current methods of multi-person pose estimation typically treat the localization and association of body joints separately. In this paper, we propose the first fully end-to-end multi-person Pose Estimation framework with TRansformers, termed PETR. Our method views pose estimation as a hierarchical set prediction problem and effectively removes the need for many hand-crafted modules like RoI cropping, NMS and grouping post-processing. In PETR, multiple pose queries are learned to directly reason a set of full-body poses. Then a joint decoder is utilized to further refine the poses by exploring the kinematic relations between body joints. With the attention mechanism, the proposed method is able to adaptively attend to the features most relevant to target keypoints, which largely overcomes the feature misalignment difficulty in pose estimation and improves the performance considerably. Extensive experiments on the MS COCO and CrowdPose benchmarks show that PETR plays favorably against state-of-the-art approaches in terms of both accuracy and efficiency. The code and models are available at https://github.com/hikvision-research/opera.

count=1
* MAD: A Scalable Dataset for Language Grounding in Videos From Movie Audio Descriptions
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Soldan_MAD_A_Scalable_Dataset_for_Language_Grounding_in_Videos_From_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Soldan_MAD_A_Scalable_Dataset_for_Language_Grounding_in_Videos_From_CVPR_2022_paper.pdf)]
    * Title: MAD: A Scalable Dataset for Language Grounding in Videos From Movie Audio Descriptions
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Mattia Soldan, Alejandro Pardo, Juan León Alcázar, Fabian Caba, Chen Zhao, Silvio Giancola, Bernard Ghanem
    * Abstract: The recent and increasing interest in video-language research has driven the development of large-scale datasets that enable data-intensive machine learning techniques. In comparison, limited effort has been made at assessing the fitness of these datasets for the video-language grounding task. Recent works have begun to discover significant limitations in these datasets, suggesting that state-of-the-art techniques commonly overfit to hidden dataset biases. In this work, we present MAD (Movie Audio Descriptions), a novel benchmark that departs from the paradigm of augmenting existing video datasets with text annotations and focuses on crawling and aligning available audio descriptions of mainstream movies. MAD contains over 384,000 natural language sentences grounded in over 1,200 hours of videos and exhibits a significant reduction in the currently diagnosed biases for video-language grounding datasets. MAD's collection strategy enables a novel and more challenging version of video-language grounding, where short temporal moments (typically seconds long) must be accurately grounded in diverse long-form videos that can last up to three hours. We have released MAD's data and baselines code at https://github.com/Soldelli/MAD.

count=1
* Globetrotter: Connecting Languages by Connecting Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Suris_Globetrotter_Connecting_Languages_by_Connecting_Images_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Suris_Globetrotter_Connecting_Languages_by_Connecting_Images_CVPR_2022_paper.pdf)]
    * Title: Globetrotter: Connecting Languages by Connecting Images
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Dídac Surís, Dave Epstein, Carl Vondrick
    * Abstract: Machine translation between many languages at once is highly challenging, since training with ground truth requires supervision between all language pairs, which is difficult to obtain. Our key insight is that, while languages may vary drastically, the underlying visual appearance of the world remains consistent. We introduce a method that uses visual observations to bridge the gap between languages, rather than relying on parallel corpora or topological properties of the representations. We train a model that aligns segments of text from different languages if and only if the images associated with them are similar and each image in turn is well-aligned with its textual description. We train our model from scratch on a new dataset of text in over fifty languages with accompanying images. Experiments show that our method outperforms previous work on unsupervised word and sentence translation using retrieval.

count=1
* Sound and Visual Representation Learning With Multiple Pretraining Tasks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Vasudevan_Sound_and_Visual_Representation_Learning_With_Multiple_Pretraining_Tasks_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Vasudevan_Sound_and_Visual_Representation_Learning_With_Multiple_Pretraining_Tasks_CVPR_2022_paper.pdf)]
    * Title: Sound and Visual Representation Learning With Multiple Pretraining Tasks
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Arun Balajee Vasudevan, Dengxin Dai, Luc Van Gool
    * Abstract: Different self-supervised tasks (SSL) reveal different features from the data. The learned feature representations can exhibit different performance for each downstream task. In this light, this work aims to combine Multiple SSL tasks (Multi-SSL) that generalizes well for all downstream tasks. For this study, we investigate binaural sounds and image data. For binaural sounds, we propose three SSL tasks namely, spatial alignment, temporal synchronization of foreground objects and binaural audio and temporal gap prediction. We investigate several approaches of Multi-SSL and give insights into the downstream task performance on video retrieval, spatial sound super resolution, and semantic prediction on the OmniAudio dataset. Our experiments on binaural sound representations demonstrate that Multi-SSL via incremental learning (IL) of SSL tasks outperforms single SSL task models and fully supervised models in the downstream task performance. As a check of applicability on other modality, we also formulate our Multi-SSL models for image representation learning and we use the recently proposed SSL tasks, MoCov2 and DenseCL. Here, Multi-SSL surpasses recent methods such as MoCov2, DenseCL and DetCo by 2.06%, 3.27% and 1.19% on VOC07 classification and +2.83, +1.56 and +1.61 AP on COCO detection.

count=1
* 3D Moments From Near-Duplicate Photos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_3D_Moments_From_Near-Duplicate_Photos_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_3D_Moments_From_Near-Duplicate_Photos_CVPR_2022_paper.pdf)]
    * Title: 3D Moments From Near-Duplicate Photos
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Qianqian Wang, Zhengqi Li, David Salesin, Noah Snavely, Brian Curless, Janne Kontkanen
    * Abstract: We introduce 3D Moments, a new computational photography effect. As input we take a pair of near-duplicate photos, i.e., photos of moving subjects from similar viewpoints, common in people's photo collections. As output, we produce a video that smoothly interpolates the scene motion from the first photo to the second, while also producing camera motion with parallax that gives a heightened sense of 3D. To achieve this effect, we represent the scene as a pair of feature-based layered depth images augmented with scene flow. This representation enables motion interpolation along with independent control of the camera viewpoint. Our system produces photorealistic space-time videos with motion parallax and scene dynamics, while plausibly recovering regions occluded in the original views. We conduct extensive experiments demonstrating superior performance over baselines on public datasets and in-the-wild photos. Project page: https://3d-moments.github.io/.

count=1
* Feature Erasing and Diffusion Network for Occluded Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Feature_Erasing_and_Diffusion_Network_for_Occluded_Person_Re-Identification_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Feature_Erasing_and_Diffusion_Network_for_Occluded_Person_Re-Identification_CVPR_2022_paper.pdf)]
    * Title: Feature Erasing and Diffusion Network for Occluded Person Re-Identification
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Zhikang Wang, Feng Zhu, Shixiang Tang, Rui Zhao, Lihuo He, Jiangning Song
    * Abstract: Occluded person re-identification (ReID) aims at matching occluded person images to holistic ones across different camera views. Target Pedestrians (TP) are often disturbed by Non-Pedestrian Occlusions (NPO) and Non-Target Pedestrians (NTP). Previous methods mainly focus on increasing the model's robustness against NPO while ignoring feature contamination from NTP. In this paper, we propose a novel Feature Erasing and Diffusion Network (FED) to simultaneously handle challenges from NPO and NTP. Specifically, aided by the NPO augmentation strategy that simulates NPO on holistic pedestrian images and generates precise occlusion masks, NPO features are explicitly eliminated by our proposed Occlusion Erasing Module (OEM). Subsequently, we diffuse the pedestrian representations with other memorized features to synthesize the NTP characteristics in the feature space through the novel Feature Diffusion Module (FDM). With the guidance of the occlusion scores from OEM, the feature diffusion process is conducted on visible body parts, thereby improving the quality of the synthesized NTP characteristics. We can greatly improve the model's perception ability towards TP and alleviate the influence of NPO and NTP by jointly optimizing OEM and FDM. Furthermore, the proposed FDM works as an auxiliary module for training and will not be engaged in the inference phase, thus with high flexibility. Experiments on occluded and holistic person ReID benchmarks demonstrate the superiority of FED over state-of-the-art methods.

count=1
* HairCLIP: Design Your Hair by Text and Reference Image
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wei_HairCLIP_Design_Your_Hair_by_Text_and_Reference_Image_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_HairCLIP_Design_Your_Hair_by_Text_and_Reference_Image_CVPR_2022_paper.pdf)]
    * Title: HairCLIP: Design Your Hair by Text and Reference Image
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Tianyi Wei, Dongdong Chen, Wenbo Zhou, Jing Liao, Zhentao Tan, Lu Yuan, Weiming Zhang, Nenghai Yu
    * Abstract: Hair editing is an interesting and challenging problem in computer vision and graphics. Many existing methods require well-drawn sketches or masks as conditional inputs for editing, however these interactions are neither straightforward nor efficient. In order to free users from the tedious interaction process, this paper proposes a new hair editing interaction mode, which enables manipulating hair attributes individually or jointly based on the texts or reference images provided by users. For this purpose, we encode the image and text conditions in a shared embedding space and propose a unified hair editing framework by leveraging the powerful image text representation capability of the Contrastive Language-Image Pre-Training (CLIP) model. With the carefully designed network structures and loss functions, our framework can perform high-quality hair editing in a disentangled manner. Extensive experiments demonstrate the superiority of our approach in terms of manipulation accuracy, visual realism of editing results, and irrelevant attribute preservation.

count=1
* H2FA R-CNN: Holistic and Hierarchical Feature Alignment for Cross-Domain Weakly Supervised Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_H2FA_R-CNN_Holistic_and_Hierarchical_Feature_Alignment_for_Cross-Domain_Weakly_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_H2FA_R-CNN_Holistic_and_Hierarchical_Feature_Alignment_for_Cross-Domain_Weakly_CVPR_2022_paper.pdf)]
    * Title: H2FA R-CNN: Holistic and Hierarchical Feature Alignment for Cross-Domain Weakly Supervised Object Detection
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yunqiu Xu, Yifan Sun, Zongxin Yang, Jiaxu Miao, Yi Yang
    * Abstract: Cross-domain weakly supervised object detection (CDWSOD) aims to adapt the detection model to a novel target domain with easily acquired image-level annotations. How to align the source and target domains is critical to the CDWSOD accuracy. Existing methods usually focus on partial detection components for domain alignment. In contrast, this paper considers that all the detection components are important and proposes a Holistic and Hierarchical Feature Alignment (H^2FA) R-CNN. H^2FA R-CNN enforces two image-level alignments for the backbone features, as well as two instance-level alignments for the RPN and detection head. This coarse-to-fine aligning hierarchy is in pace with the detection pipeline, i.e., processing the image-level feature and the instance-level features from bottom to top. Importantly, we devise a novel hybrid supervision method for learning two instance-level alignments. It enables the RPN and detection head to simultaneously receive weak/full supervision from the target/source domains. Combining all these feature alignments, H^2FA R-CNN effectively mitigates the gap between the source and target domains. Experimental results show that H^2FA R-CNN significantly improves cross-domain object detection accuracy and sets new state of the art on popular benchmarks. Code and pre-trained models are available at https://github.com/XuYunqiu/H2FA_R-CNN.

count=1
* RFNet: Unsupervised Network for Mutually Reinforcing Multi-Modal Image Registration and Fusion
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_RFNet_Unsupervised_Network_for_Mutually_Reinforcing_Multi-Modal_Image_Registration_and_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_RFNet_Unsupervised_Network_for_Mutually_Reinforcing_Multi-Modal_Image_Registration_and_CVPR_2022_paper.pdf)]
    * Title: RFNet: Unsupervised Network for Mutually Reinforcing Multi-Modal Image Registration and Fusion
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Han Xu, Jiayi Ma, Jiteng Yuan, Zhuliang Le, Wei Liu
    * Abstract: In this paper, we propose a novel method to realize multi-modal image registration and fusion in a mutually reinforcing framework, termed as RFNet. We handle the registration in a coarse-to-fine fashion. For the first time, we exploit the feedback of image fusion to promote the registration accuracy rather than treating them as two separate issues. The fine-registered results also improve the fusion performance. Specifically, for image registration, we solve the bottlenecks of defining registration metrics applicable for multi-modal images and facilitating the network convergence. The metrics are defined based on image translation and image fusion respectively in the coarse and fine stages. The convergence is facilitated by the designed metrics and a deformable convolution-based network. For image fusion, we focus on texture preservation, which not only increases the information amount and quality of fusion results but also improves the feedback of fusion results. The proposed method is evaluated on multi-modal images with large global parallaxes, images with local misalignments and aligned images to validate the performances of registration and fusion. The results in these cases demonstrate the effectiveness of our method.

count=1
* PhotoScene: Photorealistic Material and Lighting Transfer for Indoor Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yeh_PhotoScene_Photorealistic_Material_and_Lighting_Transfer_for_Indoor_Scenes_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yeh_PhotoScene_Photorealistic_Material_and_Lighting_Transfer_for_Indoor_Scenes_CVPR_2022_paper.pdf)]
    * Title: PhotoScene: Photorealistic Material and Lighting Transfer for Indoor Scenes
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yu-Ying Yeh, Zhengqin Li, Yannick Hold-Geoffroy, Rui Zhu, Zexiang Xu, Miloš Hašan, Kalyan Sunkavalli, Manmohan Chandraker
    * Abstract: Most indoor 3D scene reconstruction methods focus on recovering 3D geometry and scene layout. In this work, we go beyond this to propose PhotoScene, a framework that takes input image(s) of a scene along with approximately aligned CAD geometry (either reconstructed automatically or manually specified) and builds a photorealistic digital twin with high-quality materials and similar lighting. We model scene materials using procedural material graphs; such graphs represent photorealistic and resolution-independent materials. We optimize the parameters of these graphs and their texture scale and rotation, as well as the scene lighting to best match the input image via a differentiable rendering layer. We evaluate our technique on objects and layout reconstructions from ScanNet, SUN RGB-D and stock photographs, and demonstrate that our method reconstructs high-quality, fully relightable 3D scenes that can be re-rendered under arbitrary viewpoints, zooms and lighting.

count=1
* Memory-Augmented Non-Local Attention for Video Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Memory-Augmented_Non-Local_Attention_for_Video_Super-Resolution_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Memory-Augmented_Non-Local_Attention_for_Video_Super-Resolution_CVPR_2022_paper.pdf)]
    * Title: Memory-Augmented Non-Local Attention for Video Super-Resolution
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jiyang Yu, Jingen Liu, Liefeng Bo, Tao Mei
    * Abstract: In this paper, we propose a simple yet effective video super-resolution method that aims at generating high-fidelity high-resolution (HR) videos from low-resolution (LR) ones. Previous methods predominantly leverage temporal neighbor frames to assist the super-resolution of the current frame. Those methods achieve limited performance as they suffer from the challenges in spatial frame alignment and the lack of useful information from similar LR neighbor frames. In contrast, we devise a cross-frame non-local attention mechanism that allows video super-resolution without frame alignment, leading to being more robust to large motions in the video. In addition, to acquire general video prior information beyond neighbor frames, and to compensate for the information loss caused by large motions, we design a novel memory-augmented attention module to memorize general video details during the super-resolution training. We have thoroughly evaluated our work on various challenging datasets. Compared to other recent video super-resolution approaches, our method not only achieves significant performance gains on large motion videos but also shows better generalization. Our source code and the new Parkour benchmark dataset will be released.

count=1
* CAT-Det: Contrastively Augmented Transformer for Multi-Modal 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_CAT-Det_Contrastively_Augmented_Transformer_for_Multi-Modal_3D_Object_Detection_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_CAT-Det_Contrastively_Augmented_Transformer_for_Multi-Modal_3D_Object_Detection_CVPR_2022_paper.pdf)]
    * Title: CAT-Det: Contrastively Augmented Transformer for Multi-Modal 3D Object Detection
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yanan Zhang, Jiaxin Chen, Di Huang
    * Abstract: In autonomous driving, LiDAR point-clouds and RGB images are two major data modalities with complementary cues for 3D object detection. However, it is quite difficult to sufficiently use them, due to large inter-modal discrepancies. To address this issue, we propose a novel framework, namely Contrastively Augmented Transformer for multi-modal 3D object Detection (CAT-Det). Specifically, CAT-Det adopts a two-stream structure consisting of a Pointformer (PT) branch, an Imageformer (IT) branch along with a Cross-Modal Transformer (CMT) module. PT, IT and CMT jointly encode intra-modal and inter-modal long-range contexts for representing an object, thus fully exploring multi-modal information for detection. Furthermore, we propose an effective One-way Multi-modal Data Augmentation (OMDA) approach via hierarchical contrastive learning at both the point and object levels, significantly improving the accuracy only by augmenting point-clouds, which is free from complex generation of paired samples of the two modalities. Extensive experiments on the KITTI benchmark show that CAT-Det achieves a new state-of-the-art, highlighting its effectiveness.

count=1
* Efficient Two-Stage Detection of Human-Object Interactions With a Novel Unary-Pairwise Transformer
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Efficient_Two-Stage_Detection_of_Human-Object_Interactions_With_a_Novel_Unary-Pairwise_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Efficient_Two-Stage_Detection_of_Human-Object_Interactions_With_a_Novel_Unary-Pairwise_CVPR_2022_paper.pdf)]
    * Title: Efficient Two-Stage Detection of Human-Object Interactions With a Novel Unary-Pairwise Transformer
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Frederic Z. Zhang, Dylan Campbell, Stephen Gould
    * Abstract: Recent developments in transformer models for visual data have led to significant improvements in recognition and detection tasks. In particular, using learnable queries in place of region proposals has given rise to a new class of one-stage detection models, spearheaded by the Detection Transformer (DETR). Variations on this one-stage approach have since dominated human-object interaction (HOI) detection. However, the success of such one-stage HOI detectors can largely be attributed to the representation power of transformers. We discovered that when equipped with the same transformer, their two-stage counterparts can be more performant and memory-efficient, while taking a fraction of the time to train. In this work, we propose the Unary-Pairwise Transformer, a two-stage detector that exploits unary and pairwise representations for HOIs. We observe that the unary and pairwise parts of our transformer network specialise, with the former preferentially increasing the scores of positive examples and the latter decreasing the scores of negative examples. We evaluate our method on the HICO-DET and V-COCO datasets, and significantly outperform state-of-the-art approaches. At inference time, our model with ResNet50 approaches real-time performance on a single GPU.

count=1
* High-Fidelity Human Avatars From a Single RGB Camera
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_High-Fidelity_Human_Avatars_From_a_Single_RGB_Camera_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_High-Fidelity_Human_Avatars_From_a_Single_RGB_Camera_CVPR_2022_paper.pdf)]
    * Title: High-Fidelity Human Avatars From a Single RGB Camera
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Hao Zhao, Jinsong Zhang, Yu-Kun Lai, Zerong Zheng, Yingdi Xie, Yebin Liu, Kun Li
    * Abstract: In this paper, we propose a coarse-to-fine framework to reconstruct a personalized high-fidelity human avatar from a monocular video. To deal with the misalignment problem caused by the changed poses and shapes in different frames, we design a dynamic surface network to recover pose-dependent surface deformations, which help to decouple the shape and texture of the person. To cope with the complexity of textures and generate photo-realistic results, we propose a reference-based neural rendering network and exploit a bottom-up sharpening-guided fine-tuning strategy to obtain detailed textures. Our framework also enables photo-realistic novel view/pose synthesis and shape editing applications. Experimental results on both the public dataset and our collected dataset demonstrate that our method outperforms the state-of-the-art methods. The code and dataset will be available at http://cic.tju.edu.cn/faculty/likun/projects/HF-Avatar.

count=1
* Revisiting Temporal Alignment for Video Restoration
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Revisiting_Temporal_Alignment_for_Video_Restoration_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Revisiting_Temporal_Alignment_for_Video_Restoration_CVPR_2022_paper.pdf)]
    * Title: Revisiting Temporal Alignment for Video Restoration
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Kun Zhou, Wenbo Li, Liying Lu, Xiaoguang Han, Jiangbo Lu
    * Abstract: Long-range temporal alignment is critical yet challenging for video restoration tasks. Recently, some works attempt to divide the long-range alignment into several sub-alignments and handle them progressively. Although this operation is helpful in modeling distant correspondences, error accumulation is inevitable due to the propagation mechanism. In this work, we present a novel, generic iterative alignment module which employs a gradual refinement scheme for sub-alignments, yielding more accurate motion compensation. To further enhance the alignment accuracy and temporal consistency, we develop a non-parametric re-weighting method, where the importance of each neighboring frame is adaptively evaluated in a spatial-wise way for aggregation. By virtue of the proposed strategies, our model achieves state-of-the-art performance on multiple benchmarks across a range of video restoration tasks including video super-resolution, denoising and deblurring.

count=1
* Gamma-Enhanced Spatial Attention Network for Efficient High Dynamic Range Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Li_Gamma-Enhanced_Spatial_Attention_Network_for_Efficient_High_Dynamic_Range_Imaging_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Li_Gamma-Enhanced_Spatial_Attention_Network_for_Efficient_High_Dynamic_Range_Imaging_CVPRW_2022_paper.pdf)]
    * Title: Gamma-Enhanced Spatial Attention Network for Efficient High Dynamic Range Imaging
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Fangya Li, Ruipeng Gang, Chenghua Li, Jinjing Li, Sai Ma, Chenming Liu, Yizhen Cao
    * Abstract: High dynamic range(HDR) imaging is the task of recovering HDR image from one or multiple input Low Dynamic Range (LDR) images. In this paper, we present Gamma-enhanced Spatial Attention Network(GSANet), a novel framework for reconstructing HDR images. This problem comprises two intractable challenges of how to tackle overexposed and underexposed regions and how to overcome the paradox of performance and complexity trade-off. To address the former, after applying gamma correction on the LDR images, we adopt a spatial attention module to adaptively select the most appropriate regions of various exposure low dynamic range images for fusion. For the latter one, we propose an efficient channel attention module, which only involves a handful of parameters while bringing clear performance gain. Experimental results show that the proposed method achieves better visual quality on the HDR dataset. The code will be available at:https://github.com/fancyicookie/GSANet.

count=1
* Adaptive Feature Consolidation Network for Burst Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Mehta_Adaptive_Feature_Consolidation_Network_for_Burst_Super-Resolution_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Mehta_Adaptive_Feature_Consolidation_Network_for_Burst_Super-Resolution_CVPRW_2022_paper.pdf)]
    * Title: Adaptive Feature Consolidation Network for Burst Super-Resolution
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Nancy Mehta, Akshay Dudhane, Subrahmanyam Murala, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan
    * Abstract: Modern digital cameras generally count on image signal processing (ISP) pipelines for producing naturalistic RGB images. Nevertheless, in comparison to DSLR cameras, low-quality images are generally output from portable mobile devices due to their physical limitations. The synthesized low-quality images usually have multiple degradations - low-resolution owing to small camera sensors, mosaic patterns on account of camera filter array and sub-pixel shifts due to camera motion. Such degradation usually restrain the performance of single image super-resolution methodologies for retrieving high-resolution (HR) image from a single low-resolution (LR) image. Burst image super-resolution aims at restoring a photo-realistic HR image by capturing the abundant information from multiple LR images. Lately, the soaring popularity of burst photography has made multi-frame processing an attractive solution for overcoming the limitations of single image processing. In our work, we thus aim to propose a generic architecture, adaptive feature consolidation network (AFCNet) for multi-frame processing. To alleviate the challenge of effectively modelling the long-range dependency problem, that multi-frame approaches struggle to solve, we utilize encoder-decoder based transformer backbone which learns multi-scale local-global representations. We propose feature alignment module to align LR burst frame features. Further, the aligned features are fused and reconstructed by abridged pseudo-burst fusion module and adaptive group upsampling modules, respectively. Our proposed approach clearly outperforms the other existing state-of-the-art techniques on benchmark datasets. The experimental results illustrate the effectiveness and generality of our proposed framework in upgrading the visual quality of HR images.

count=1
* A Lightweight Network for High Dynamic Range Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Yan_A_Lightweight_Network_for_High_Dynamic_Range_Imaging_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Yan_A_Lightweight_Network_for_High_Dynamic_Range_Imaging_CVPRW_2022_paper.pdf)]
    * Title: A Lightweight Network for High Dynamic Range Imaging
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Qingsen Yan, Song Zhang, Weiye Chen, Yuhang Liu, Zhen Zhang, Yanning Zhang, Javen Qinfeng Shi, Dong Gong
    * Abstract: Multi-frame high dynamic range (HDR) reconstruction methods try to expand the range of illuminance with differently exposed images. They suffer from ghost artifacts when camera jittering or object moving. Several methods can generate high-quality HDR images with high computational complexity, but the inference process is too slow. However, the network with small parameters will produce unsatisfactory results. To balance the quality and computational complexity, we propose a lightweight network for HDR imaging that has small parameters and fast speed. Specifically, following AHDRNet, we employ a spatial attention module to detect the misaligned regions to avoid ghost artifacts. Considering the missing details in over-/under- exposure regions, we propose a dual attention module for selectively retaining information to force the fusion network to learn more details for degenerated regions. Furthermore, we employ an encoder-decoder structure with a lightweight block to achieve the fusion process. As a result, the high-quality content and features can be reconstructed after the attention module. Finally, we fuse high-resolution features and the encoder-decoder features into the HDR imaging results. Experimental results demonstrate that the proposed method performs favorably against the state-of-the-art methods, achieving a PSNR of 39.05 and a PSNR-mu of 37.27 with 156.12 GMAcs in NTIRE 2022 HDR Challenge (Track 2 Fidelity).

count=1
* Progressive Training of a Two-Stage Framework for Video Restoration
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Zheng_Progressive_Training_of_a_Two-Stage_Framework_for_Video_Restoration_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Zheng_Progressive_Training_of_a_Two-Stage_Framework_for_Video_Restoration_CVPRW_2022_paper.pdf)]
    * Title: Progressive Training of a Two-Stage Framework for Video Restoration
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Meisong Zheng, Qunliang Xing, Minglang Qiao, Mai Xu, Lai Jiang, Huaida Liu, Ying Chen
    * Abstract: As a widely studied task, video restoration aims to enhance the quality of the videos with multiple potential degradations, such as noises, blurs and compression artifacts. Among video restorations, compressed video quality enhancement and video super-resolution are two of the main tacks with significant values in practical scenarios. Recently, recurrent neural networks and transformers attract increasing research interests in this field, due to their impressive capability in sequence-to-sequence modeling. However, the training of these models is not only costly but also relatively hard to converge, with gradient exploding and vanishing problems. To cope with these problems, we proposed a two-stage framework including a multi-frame recurrent network and a single-frame transformer. Besides, multiple training strategies, such as transfer learning and progressive training, are developed to shorten the training time and improve the model performance. Benefiting from the above technical contributions, our solution wins two champions and a runner-up in the NTIRE 2022 super-resolution and quality enhancement of compressed video challenges.

count=1
* DC2: Dual-Camera Defocus Control by Learning To Refocus
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Alzayer_DC2_Dual-Camera_Defocus_Control_by_Learning_To_Refocus_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Alzayer_DC2_Dual-Camera_Defocus_Control_by_Learning_To_Refocus_CVPR_2023_paper.pdf)]
    * Title: DC2: Dual-Camera Defocus Control by Learning To Refocus
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Hadi Alzayer, Abdullah Abuolaim, Leung Chun Chan, Yang Yang, Ying Chen Lou, Jia-Bin Huang, Abhishek Kar
    * Abstract: Smartphone cameras today are increasingly approaching the versatility and quality of professional cameras through a combination of hardware and software advancements. However, fixed aperture remains a key limitation, preventing users from controlling the depth of field (DoF) of captured images. At the same time, many smartphones now have multiple cameras with different fixed apertures - specifically, an ultra-wide camera with wider field of view and deeper DoF and a higher resolution primary camera with shallower DoF. In this work, we propose DC^2, a system for defocus control for synthetically varying camera aperture, focus distance and arbitrary defocus effects by fusing information from such a dual-camera system. Our key insight is to leverage real-world smartphone camera dataset by using image refocus as a proxy task for learning to control defocus. Quantitative and qualitative evaluations on real-world data demonstrate our system's efficacy where we outperform state-of-the-art on defocus deblurring, bokeh rendering, and image refocus. Finally, we demonstrate creative post-capture defocus control enabled by our method, including tilt-shift and content-based defocus effects.

count=1
* TMO: Textured Mesh Acquisition of Objects With a Mobile Device by Using Differentiable Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Choi_TMO_Textured_Mesh_Acquisition_of_Objects_With_a_Mobile_Device_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Choi_TMO_Textured_Mesh_Acquisition_of_Objects_With_a_Mobile_Device_CVPR_2023_paper.pdf)]
    * Title: TMO: Textured Mesh Acquisition of Objects With a Mobile Device by Using Differentiable Rendering
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jaehoon Choi, Dongki Jung, Taejae Lee, Sangwook Kim, Youngdong Jung, Dinesh Manocha, Donghwan Lee
    * Abstract: We present a new pipeline for acquiring a textured mesh in the wild with a single smartphone which offers access to images, depth maps, and valid poses. Our method first introduces an RGBD-aided structure from motion, which can yield filtered depth maps and refines camera poses guided by corresponding depth. Then, we adopt the neural implicit surface reconstruction method, which allows for high quality mesh and develops a new training process for applying a regularization provided by classical multi-view stereo methods. Moreover, we apply a differentiable rendering to fine-tune incomplete texture maps and generate textures which are perceptually closer to the original scene. Our pipeline can be applied to any common objects in the real world without the need for either in-the-lab environments or accurate mask images. We demonstrate results of captured objects with complex shapes and validate our method numerically against existing 3D reconstruction and texture mapping methods.

count=1
* STDLens: Model Hijacking-Resilient Federated Learning for Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Chow_STDLens_Model_Hijacking-Resilient_Federated_Learning_for_Object_Detection_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Chow_STDLens_Model_Hijacking-Resilient_Federated_Learning_for_Object_Detection_CVPR_2023_paper.pdf)]
    * Title: STDLens: Model Hijacking-Resilient Federated Learning for Object Detection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Ka-Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu
    * Abstract: Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates. The source code is available at https://github.com/git-disl/STDLens.

count=1
* Generating Aligned Pseudo-Supervision From Non-Aligned Data for Image Restoration in Under-Display Camera
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Feng_Generating_Aligned_Pseudo-Supervision_From_Non-Aligned_Data_for_Image_Restoration_in_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_Generating_Aligned_Pseudo-Supervision_From_Non-Aligned_Data_for_Image_Restoration_in_CVPR_2023_paper.pdf)]
    * Title: Generating Aligned Pseudo-Supervision From Non-Aligned Data for Image Restoration in Under-Display Camera
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Ruicheng Feng, Chongyi Li, Huaijin Chen, Shuai Li, Jinwei Gu, Chen Change Loy
    * Abstract: Due to the difficulty in collecting large-scale and perfectly aligned paired training data for Under-Display Camera (UDC) image restoration, previous methods resort to monitor-based image systems or simulation-based methods, sacrificing the realness of the data and introducing domain gaps. In this work, we revisit the classic stereo setup for training data collection -- capturing two images of the same scene with one UDC and one standard camera. The key idea is to "copy" details from a high-quality reference image and "paste" them on the UDC image. While being able to generate real training pairs, this setting is susceptible to spatial misalignment due to perspective and depth of field changes. The problem is further compounded by the large domain discrepancy between the UDC and normal images, which is unique to UDC restoration. In this paper, we mitigate the non-trivial domain discrepancy and spatial misalignment through a novel Transformer-based framework that generates well-aligned yet high-quality target data for the corresponding UDC input. This is made possible through two carefully designed components, namely, the Domain Alignment Module (DAM) and Geometric Alignment Module (GAM), which encourage robust and accurate discovery of correspondence between the UDC and normal views. Extensive experiments show that high-quality and well-aligned pseudo UDC training pairs are beneficial for training a robust restoration network. Code and the dataset are available at https://github.com/jnjaby/AlignFormer.

count=1
* Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Geng_Dense-Localizing_Audio-Visual_Events_in_Untrimmed_Videos_A_Large-Scale_Benchmark_and_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Geng_Dense-Localizing_Audio-Visual_Events_in_Untrimmed_Videos_A_Large-Scale_Benchmark_and_CVPR_2023_paper.pdf)]
    * Title: Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Tiantian Geng, Teng Wang, Jinming Duan, Runmin Cong, Feng Zheng
    * Abstract: Existing audio-visual event localization (AVE) handles manually trimmed videos with only a single instance in each of them. However, this setting is unrealistic as natural videos often contain numerous audio-visual events with different categories. To better adapt to real-life applications, in this paper we focus on the task of dense-localizing audio-visual events, which aims to jointly localize and recognize all audio-visual events occurring in an untrimmed video. The problem is challenging as it requires fine-grained audio-visual scene and context understanding. To tackle this problem, we introduce the first Untrimmed Audio-Visual (UnAV-100) dataset, which contains 10K untrimmed videos with over 30K audio-visual events. Each video has 2.8 audio-visual events on average, and the events are usually related to each other and might co-occur as in real-life scenes. Next, we formulate the task using a new learning-based framework, which is capable of fully integrating audio and visual modalities to localize audio-visual events with various lengths and capture dependencies between them in a single pass. Extensive experiments demonstrate the effectiveness of our method as well as the significance of multi-scale cross-modal perception and dependency modeling for this task.

count=1
* Hierarchical Neural Memory Network for Low Latency Event Processing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Hamaguchi_Hierarchical_Neural_Memory_Network_for_Low_Latency_Event_Processing_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Hamaguchi_Hierarchical_Neural_Memory_Network_for_Low_Latency_Event_Processing_CVPR_2023_paper.pdf)]
    * Title: Hierarchical Neural Memory Network for Low Latency Event Processing
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Ryuhei Hamaguchi, Yasutaka Furukawa, Masaki Onishi, Ken Sakurada
    * Abstract: This paper proposes a low latency neural network architecture for event-based dense prediction tasks. Conventional architectures encode entire scene contents at a fixed rate regardless of their temporal characteristics. Instead, the proposed network encodes contents at a proper temporal scale depending on its movement speed. We achieve this by constructing temporal hierarchy using stacked latent memories that operate at different rates. Given low latency event steams, the multi-level memories gradually extract dynamic to static scene contents by propagating information from the fast to the slow memory modules. The architecture not only reduces the redundancy of conventional architectures but also exploits long-term dependencies. Furthermore, an attention-based event representation efficiently encodes sparse event streams into the memory cells. We conduct extensive evaluations on three event-based dense prediction tasks, where the proposed approach outperforms the existing methods on accuracy and latency, while demonstrating effective event and image fusion capabilities. The code is available at https://hamarh.github.io/hmnet/

count=1
* Dual Alignment Unsupervised Domain Adaptation for Video-Text Retrieval
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Hao_Dual_Alignment_Unsupervised_Domain_Adaptation_for_Video-Text_Retrieval_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Hao_Dual_Alignment_Unsupervised_Domain_Adaptation_for_Video-Text_Retrieval_CVPR_2023_paper.pdf)]
    * Title: Dual Alignment Unsupervised Domain Adaptation for Video-Text Retrieval
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Xiaoshuai Hao, Wanqian Zhang, Dayan Wu, Fei Zhu, Bo Li
    * Abstract: Video-text retrieval is an emerging stream in both computer vision and natural language processing communities, which aims to find relevant videos given text queries. In this paper, we study the notoriously challenging task, i.e., Unsupervised Domain Adaptation Video-text Retrieval (UDAVR), wherein training and testing data come from different distributions. Previous works merely alleviate the domain shift, which however overlook the pairwise misalignment issue in target domain, i.e., there exist no semantic relationships between target videos and texts. To tackle this, we propose a novel method named Dual Alignment Domain Adaptation (DADA). Specifically, we first introduce the cross-modal semantic embedding to generate discriminative source features in a joint embedding space. Besides, we utilize the video and text domain adaptations to smoothly balance the minimization of the domain shifts. To tackle the pairwise misalignment in target domain, we introduce the Dual Alignment Consistency (DAC) to fully exploit the semantic information of both modalities in target domain. The proposed DAC adaptively aligns the video-text pairs which are more likely to be relevant in target domain, enabling that positive pairs are increasing progressively and the noisy ones will potentially be aligned in the later stages. To that end, our method can generate more truly aligned target pairs and ensure the discriminality of target features.Compared with the state-of-the-art methods, DADA achieves 20.18% and 18.61% relative improvements on R@1 under the setting of TGIF->MSRVTT and TGIF->MSVD respectively, demonstrating the superiority of our method.

count=1
* Continuous Sign Language Recognition With Correlation Network
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Continuous_Sign_Language_Recognition_With_Correlation_Network_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Continuous_Sign_Language_Recognition_With_Correlation_Network_CVPR_2023_paper.pdf)]
    * Title: Continuous Sign Language Recognition With Correlation Network
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Lianyu Hu, Liqing Gao, Zekang Liu, Wei Feng
    * Abstract: Human body trajectories are a salient cue to identify actions in video. Such body trajectories are mainly conveyed by hands and face across consecutive frames in sign language. However, current methods in continuous sign language recognition(CSLR) usually process frames independently to capture frame-wise features, thus failing to capture cross-frame trajectories to effectively identify a sign. To handle this limitation, we propose correlation network (CorrNet) to explicitly leverage body trajectories across frames to identify signs. In specific, an identification module is first presented to emphasize informative regions in each frame that are beneficial in expressing a sign. A correlation module is then proposed to dynamically compute correlation maps between current frame and adjacent neighbors to capture cross-frame trajectories. As a result, the generated features are able to gain an overview of local temporal movements to identify a sign. Thanks to its special attention on body trajectories, CorrNet achieves new state-of-the-art accuracy on four large-scale datasets, PHOENIX14, PHOENIX14-T, CSL-Daily, and CSL. A comprehensive comparison between CorrNet and previous spatial-temporal reasoning methods verifies its effectiveness. Visualizations are given to demonstrate the effects of CorrNet on emphasizing human body trajectories across adjacent frames.

count=1
* DNF: Decouple and Feedback Network for Seeing in the Dark
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.pdf)]
    * Title: DNF: Decouple and Feedback Network for Seeing in the Dark
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Xin Jin, Ling-Hao Han, Zhen Li, Chun-Le Guo, Zhi Chai, Chongyi Li
    * Abstract: The exclusive properties of RAW data have shown great potential for low-light image enhancement. Nevertheless, the performance is bottlenecked by the inherent limitations of existing architectures in both single-stage and multi-stage methods. Mixed mapping across two different domains, noise-to-clean and RAW-to-sRGB, misleads the single-stage methods due to the domain ambiguity. The multi-stage methods propagate the information merely through the resulting image of each stage, neglecting the abundant features in the lossy image-level dataflow. In this paper, we probe a generalized solution to these bottlenecks and propose a Decouple aNd Feedback framework, abbreviated as DNF. To mitigate the domain ambiguity, domainspecific subtasks are decoupled, along with fully utilizing the unique properties in RAW and sRGB domains. The feature propagation across stages with a feedback mechanism avoids the information loss caused by image-level dataflow. The two key insights of our method resolve the inherent limitations of RAW data-based low-light image enhancement satisfactorily, empowering our method to outperform the previous state-of-the-art method by a large margin with only 19% parameters, achieving 0.97dB and 1.30dB PSNR improvements on the Sony and Fuji subsets of SID.

count=1
* Human Pose Estimation in Extremely Low-Light Conditions
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Human_Pose_Estimation_in_Extremely_Low-Light_Conditions_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Lee_Human_Pose_Estimation_in_Extremely_Low-Light_Conditions_CVPR_2023_paper.pdf)]
    * Title: Human Pose Estimation in Extremely Low-Light Conditions
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Sohyun Lee, Jaesung Rim, Boseung Jeong, Geonu Kim, Byungju Woo, Haechan Lee, Sunghyun Cho, Suha Kwak
    * Abstract: We study human pose estimation in extremely low-light images. This task is challenging due to the difficulty of collecting real low-light images with accurate labels, and severely corrupted inputs that degrade prediction quality significantly. To address the first issue, we develop a dedicated camera system and build a new dataset of real low-light images with accurate pose labels. Thanks to our camera system, each low-light image in our dataset is coupled with an aligned well-lit image, which enables accurate pose labeling and is used as privileged information during training. We also propose a new model and a new training strategy that fully exploit the privileged information to learn representation insensitive to lighting conditions. Our method demonstrates outstanding performance on real extremely low-light images, and extensive analyses validate that both of our model and dataset contribute to the success.

count=1
* DropKey for Vision Transformer
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Li_DropKey_for_Vision_Transformer_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_DropKey_for_Vision_Transformer_CVPR_2023_paper.pdf)]
    * Title: DropKey for Vision Transformer
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Bonan Li, Yinhan Hu, Xuecheng Nie, Congying Han, Xiangjian Jiang, Tiande Guo, Luoqi Liu
    * Abstract: In this paper, we focus on analyzing and improving the dropout technique for self-attention layers of Vision Transformer, which is important while surprisingly ignored by prior works. In particular, we conduct researches on three core questions: First, what to drop in self-attention layers? Different from dropping attention weights in literature, we propose to move dropout operations forward ahead of attention matrix calculation and set the Key as the dropout unit, yielding a novel dropout-before-softmax scheme. We theoretically verify that this scheme helps keep both regularization and probability features of attention weights, alleviating the overfittings problem to specific patterns and enhancing the model to globally capture vital information; Second, how to schedule the drop ratio in consecutive layers? In contrast to exploit a constant drop ratio for all layers, we present a new decreasing schedule that gradually decreases the drop ratio along the stack of self-attention layers. We experimentally validate the proposed schedule can avoid overfittings in low-level features and missing in high-level semantics, thus improving the robustness and stableness of model training; Third, whether need to perform structured dropout operation as CNN? We attempt patch-based block-version of dropout operation and find that this useful trick for CNN is not essential for ViT. Given exploration on the above three questions, we present the novel DropKey method that regards Key as the drop unit and exploits decreasing schedule for drop ratio, improving ViTs in a general way. Comprehensive experiments demonstrate the effectiveness of DropKey for various ViT architectures, e.g. T2T, VOLO, CeiT and DeiT, as well as for various vision tasks, e.g., image classification, object detection, human-object interaction detection and human body shape recovery.

count=1
* MobileBrick: Building LEGO for 3D Reconstruction on Mobile Devices
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Li_MobileBrick_Building_LEGO_for_3D_Reconstruction_on_Mobile_Devices_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_MobileBrick_Building_LEGO_for_3D_Reconstruction_on_Mobile_Devices_CVPR_2023_paper.pdf)]
    * Title: MobileBrick: Building LEGO for 3D Reconstruction on Mobile Devices
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Kejie Li, Jia-Wang Bian, Robert Castle, Philip H.S. Torr, Victor Adrian Prisacariu
    * Abstract: High-quality 3D ground-truth shapes are critical for 3D object reconstruction evaluation. However, it is difficult to create a replica of an object in reality, and even 3D reconstructions generated by 3D scanners have artefacts that cause biases in evaluation. To address this issue, we introduce a novel multi-view RGBD dataset captured using a mobile device, which includes highly precise 3D ground-truth annotations for 153 object models featuring a diverse set of 3D structures. We obtain precise 3D ground-truth shape without relying on high-end 3D scanners by utilising LEGO models with known geometry as the 3D structures for image capture. The distinct data modality offered by high- resolution RGB images and low-resolution depth maps captured on a mobile device, when combined with precise 3D geometry annotations, presents a unique opportunity for future research on high-fidelity 3D reconstruction. Furthermore, we evaluate a range of 3D reconstruction algorithms on the proposed dataset.

count=1
* Depth Estimation From Camera Image and mmWave Radar Point Cloud
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Singh_Depth_Estimation_From_Camera_Image_and_mmWave_Radar_Point_Cloud_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Singh_Depth_Estimation_From_Camera_Image_and_mmWave_Radar_Point_Cloud_CVPR_2023_paper.pdf)]
    * Title: Depth Estimation From Camera Image and mmWave Radar Point Cloud
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Akash Deep Singh, Yunhao Ba, Ankur Sarker, Howard Zhang, Achuta Kadambi, Stefano Soatto, Mani Srivastava, Alex Wong
    * Abstract: We present a method for inferring dense depth from a camera image and a sparse noisy radar point cloud. We first describe the mechanics behind mmWave radar point cloud formation and the challenges that it poses, i.e. ambiguous elevation and noisy depth and azimuth components that yields incorrect positions when projected onto the image, and how existing works have overlooked these nuances in camera-radar fusion. Our approach is motivated by these mechanics, leading to the design of a network that maps each radar point to the possible surfaces that it may project onto in the image plane. Unlike existing works, we do not process the raw radar point cloud as an erroneous depth map, but query each raw point independently to associate it with likely pixels in the image -- yielding a semi-dense radar depth map. To fuse radar depth with an image, we propose a gated fusion scheme that accounts for the confidence scores of the correspondence so that we selectively combine radar and camera embeddings to yield a dense depth map. We test our method on the NuScenes benchmark and show a 10.3% improvement in mean absolute error and a 9.1% improvement in root-mean-square error over the best method.

count=1
* Robust Single Image Reflection Removal Against Adversarial Attacks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Song_Robust_Single_Image_Reflection_Removal_Against_Adversarial_Attacks_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Robust_Single_Image_Reflection_Removal_Against_Adversarial_Attacks_CVPR_2023_paper.pdf)]
    * Title: Robust Single Image Reflection Removal Against Adversarial Attacks
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zhenbo Song, Zhenyuan Zhang, Kaihao Zhang, Wenhan Luo, Zhaoxin Fan, Wenqi Ren, Jianfeng Lu
    * Abstract: This paper addresses the problem of robust deep single-image reflection removal (SIRR) against adversarial attacks. Current deep learning based SIRR methods have shown significant performance degradation due to unnoticeable distortions and perturbations on input images. For a comprehensive robustness study, we first conduct diverse adversarial attacks specifically for the SIRR problem, i.e. towards different attacking targets and regions. Then we propose a robust SIRR model, which integrates the cross-scale attention module, the multi-scale fusion module, and the adversarial image discriminator. By exploiting the multi-scale mechanism, the model narrows the gap between features from clean and adversarial images. The image discriminator adaptively distinguishes clean or noisy inputs, and thus further gains reliable robustness. Extensive experiments on Nature, SIR^2, and Real datasets demonstrate that our model remarkably improves the robustness of SIRR across disparate scenes.

count=1
* Graph Transformer GANs for Graph-Constrained House Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tang_Graph_Transformer_GANs_for_Graph-Constrained_House_Generation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Graph_Transformer_GANs_for_Graph-Constrained_House_Generation_CVPR_2023_paper.pdf)]
    * Title: Graph Transformer GANs for Graph-Constrained House Generation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Hao Tang, Zhenyu Zhang, Humphrey Shi, Bo Li, Ling Shao, Nicu Sebe, Radu Timofte, Luc Van Gool
    * Abstract: We present a novel graph Transformer generative adversarial network (GTGAN) to learn effective graph node relations in an end-to-end fashion for the challenging graph-constrained house generation task. The proposed graph-Transformer-based generator includes a novel graph Transformer encoder that combines graph convolutions and self-attentions in a Transformer to model both local and global interactions across connected and non-connected graph nodes. Specifically, the proposed connected node attention (CNA) and non-connected node attention (NNA) aim to capture the global relations across connected nodes and non-connected nodes in the input graph, respectively. The proposed graph modeling block (GMB) aims to exploit local vertex interactions based on a house layout topology. Moreover, we propose a new node classification-based discriminator to preserve the high-level semantic and discriminative node features for different house components. Finally, we propose a novel graph-based cycle-consistency loss that aims at maintaining the relative spatial relationships between ground truth and predicted graphs. Experiments on two challenging graph-constrained house generation tasks (i.e., house layout and roof generation) with two public datasets demonstrate the effectiveness of GTGAN in terms of objective quantitative scores and subjective visual realism. New state-of-the-art results are established by large margins on both tasks.

count=1
* Stare at What You See: Masked Image Modeling Without Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Xue_Stare_at_What_You_See_Masked_Image_Modeling_Without_Reconstruction_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Xue_Stare_at_What_You_See_Masked_Image_Modeling_Without_Reconstruction_CVPR_2023_paper.pdf)]
    * Title: Stare at What You See: Masked Image Modeling Without Reconstruction
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Hongwei Xue, Peng Gao, Hongyang Li, Yu Qiao, Hao Sun, Houqiang Li, Jiebo Luo
    * Abstract: Masked Autoencoders (MAE) have been prevailing paradigms for large-scale vision representation pre-training. By reconstructing masked image patches from a small portion of visible image regions, MAE forces the model to infer semantic correlation within an image. Recently, some approaches apply semantic-rich teacher models to extract image features as the reconstruction target, leading to better performance. However, unlike the low-level features such as pixel values, we argue the features extracted by powerful teacher models already encode rich semantic correlation across regions in an intact image. This raises one question: is reconstruction necessary in Masked Image Modeling (MIM) with a teacher model? In this paper, we propose an efficient MIM paradigm named MaskAlign. MaskAlign simply learns the consistency of visible patch feature extracted by the student model and intact image features extracted by the teacher model. To further advance the performance and tackle the problem of input inconsistency between the student and teacher model, we propose a Dynamic Alignment (DA) module to apply learnable alignment. Our experimental results demonstrate that masked modeling does not lose effectiveness even without reconstruction on masked regions. Combined with Dynamic Alignment, MaskAlign can achieve state-of-the-art performance with much higher efficiency.

count=1
* Learning Event Guided High Dynamic Range Video Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Learning_Event_Guided_High_Dynamic_Range_Video_Reconstruction_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Learning_Event_Guided_High_Dynamic_Range_Video_Reconstruction_CVPR_2023_paper.pdf)]
    * Title: Learning Event Guided High Dynamic Range Video Reconstruction
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yixin Yang, Jin Han, Jinxiu Liang, Imari Sato, Boxin Shi
    * Abstract: Limited by the trade-off between frame rate and exposure time when capturing moving scenes with conventional cameras, frame based HDR video reconstruction suffers from scene-dependent exposure ratio balancing and ghosting artifacts. Event cameras provide an alternative visual representation with a much higher dynamic range and temporal resolution free from the above issues, which could be an effective guidance for HDR imaging from LDR videos. In this paper, we propose a multimodal learning framework for event guided HDR video reconstruction. In order to better leverage the knowledge of the same scene from the two modalities of visual signals, a multimodal representation alignment strategy to learn a shared latent space and a fusion module tailored to complementing two types of signals for different dynamic ranges in different regions are proposed. Temporal correlations are utilized recurrently to suppress the flickering effects in the reconstructed HDR video. The proposed HDRev-Net demonstrates state-of-the-art performance quantitatively and qualitatively for both synthetic and real-world data.

count=1
* Modeling Entities As Semantic Points for Visual Information Extraction in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Modeling_Entities_As_Semantic_Points_for_Visual_Information_Extraction_in_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Modeling_Entities_As_Semantic_Points_for_Visual_Information_Extraction_in_CVPR_2023_paper.pdf)]
    * Title: Modeling Entities As Semantic Points for Visual Information Extraction in the Wild
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zhibo Yang, Rujiao Long, Pengfei Wang, Sibo Song, Humen Zhong, Wenqing Cheng, Xiang Bai, Cong Yao
    * Abstract: Recently, Visual Information Extraction (VIE) has been becoming increasingly important in both academia and industry, due to the wide range of real-world applications. Previously, numerous works have been proposed to tackle this problem. However, the benchmarks used to assess these methods are relatively plain, i.e., scenarios with real-world complexity are not fully represented in these benchmarks. As the first contribution of this work, we curate and release a new dataset for VIE, in which the document images are much more challenging in that they are taken from real applications, and difficulties such as blur, partial occlusion, and printing shift are quite common. All these factors may lead to failures in information extraction. Therefore, as the second contribution, we explore an alternative approach to precisely and robustly extract key information from document images under such tough conditions. Specifically, in contrast to previous methods, which usually either incorporate visual information into a multi-modal architecture or train text spotting and information extraction in an end-to-end fashion, we explicitly model entities as semantic points, i.e., center points of entities are enriched with semantic information describing the attributes and relationships of different entities, which could largely benefit entity labeling and linking. Extensive experiments on standard benchmarks in this field as well as the proposed dataset demonstrate that the proposed method can achieve significantly enhanced performance on entity labeling and linking, compared with previous state-of-the-art models.

count=1
* Linking Garment With Person via Semantically Associated Landmarks for Virtual Try-On
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yan_Linking_Garment_With_Person_via_Semantically_Associated_Landmarks_for_Virtual_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_Linking_Garment_With_Person_via_Semantically_Associated_Landmarks_for_Virtual_CVPR_2023_paper.pdf)]
    * Title: Linking Garment With Person via Semantically Associated Landmarks for Virtual Try-On
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Keyu Yan, Tingwei Gao, Hui Zhang, Chengjun Xie
    * Abstract: In this paper, a novel virtual try-on algorithm, dubbed SAL-VTON, is proposed, which links the garment with the person via semantically associated landmarks to alleviate misalignment. The semantically associated landmarks are a series of landmark pairs with the same local semantics on the in-shop garment image and the try-on image. Based on the semantically associated landmarks, SAL-VTON effectively models the local semantic association between garment and person, making up for the misalignment in the overall deformation of the garment. The outcome is achieved with a three-stage framework: 1) the semantically associated landmarks are estimated using the landmark localization model; 2) taking the landmarks as input, the warping model explicitly associates the corresponding parts of the garment and person for obtaining the local flow, thus refining the alignment in the global flow; 3) finally, a generator consumes the landmarks to better capture local semantics and control the try-on results.Moreover, we propose a new landmark dataset with a unified labelling rule of landmarks for diverse styles of garments. Extensive experimental results on popular datasets demonstrate that SAL-VTON can handle misalignment and outperform state-of-the-art methods both qualitatively and quantitatively. The dataset is available on https://modelscope.cn/datasets/damo/SAL-HG/summary.

count=1
* Delivering Arbitrary-Modal Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Delivering_Arbitrary-Modal_Semantic_Segmentation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Delivering_Arbitrary-Modal_Semantic_Segmentation_CVPR_2023_paper.pdf)]
    * Title: Delivering Arbitrary-Modal Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jiaming Zhang, Ruiping Liu, Hao Shi, Kailun Yang, Simon Reiß, Kunyu Peng, Haodong Fu, Kaiwei Wang, Rainer Stiefelhagen
    * Abstract: Multimodal fusion can make semantic segmentation more robust. However, fusing an arbitrary number of modalities remains underexplored. To delve into this problem, we create the DeLiVER arbitrary-modal segmentation benchmark, covering Depth, LiDAR, multiple Views, Events, and RGB. Aside from this, we provide this dataset in four severe weather conditions as well as five sensor failure cases to exploit modal complementarity and resolve partial outages. To facilitate this data, we present the arbitrary cross-modal segmentation model CMNeXt. It encompasses a Self-Query Hub (SQ-Hub) designed to extract effective information from any modality for subsequent fusion with the RGB representation and adds only negligible amounts of parameters ( 0.01M) per additional modality. On top, to efficiently and flexibly harvest discriminative cues from the auxiliary modalities, we introduce the simple Parallel Pooling Mixer (PPX). With extensive experiments on a total of six benchmarks, our CMNeXt achieves state-of-the-art performance, allowing to scale from 1 to 81 modalities on the DeLiVER, KITTI-360, MFNet, NYU Depth V2, UrbanLF, and MCubeS datasets. On the freshly collected DeLiVER, the quad-modal CMNeXt reaches up to 66.30% in mIoU with a +9.10% gain as compared to the mono-modal baseline.

count=1
* Frame-Event Alignment and Fusion Network for High Frame Rate Tracking
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Frame-Event_Alignment_and_Fusion_Network_for_High_Frame_Rate_Tracking_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Frame-Event_Alignment_and_Fusion_Network_for_High_Frame_Rate_Tracking_CVPR_2023_paper.pdf)]
    * Title: Frame-Event Alignment and Fusion Network for High Frame Rate Tracking
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jiqing Zhang, Yuanchen Wang, Wenxi Liu, Meng Li, Jinpeng Bai, Baocai Yin, Xin Yang
    * Abstract: Most existing RGB-based trackers target low frame rate benchmarks of around 30 frames per second. This setting restricts the tracker's functionality in the real world, especially for fast motion. Event-based cameras as bioinspired sensors provide considerable potential for high frame rate tracking due to their high temporal resolution. However, event-based cameras cannot offer fine-grained texture information like conventional cameras. This unique complementarity motivates us to combine conventional frames and events for high frame rate object tracking under various challenging conditions. In this paper, we propose an end-to-end network consisting of multi-modality alignment and fusion modules to effectively combine meaningful information from both modalities at different measurement rates. The alignment module is responsible for cross-modality and cross-frame-rate alignment between frame and event modalities under the guidance of the moving cues furnished by events. While the fusion module is accountable for emphasizing valuable features and suppressing noise information by the mutual complement between the two modalities. Extensive experiments show that the proposed approach outperforms state-of-the-art trackers by a significant margin in high frame rate tracking. With the FE240hz dataset, our approach achieves high frame rate tracking up to 240Hz.

count=1
* Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Gradient_Norm_Aware_Minimization_Seeks_First-Order_Flatness_and_Improves_Generalization_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Gradient_Norm_Aware_Minimization_Seeks_First-Order_Flatness_and_Improves_Generalization_CVPR_2023_paper.pdf)]
    * Title: Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Xingxuan Zhang, Renzhe Xu, Han Yu, Hao Zou, Peng Cui
    * Abstract: Recently, flat minima are proven to be effective for improving generalization and sharpness-aware minimization (SAM) achieves state-of-the-art performance. Yet the current definition of flatness discussed in SAM and its follow-ups are limited to the zeroth-order flatness (i.e., the worst-case loss within a perturbation radius). We show that the zeroth-order flatness can be insufficient to discriminate minima with low generalization error from those with high generalization error both when there is a single minimum or multiple minima within the given perturbation radius. Thus we present first-order flatness, a stronger measure of flatness focusing on the maximal gradient norm within a perturbation radius which bounds both the maximal eigenvalue of Hessian at local minima and the regularization function of SAM. We also present a novel training procedure named Gradient norm Aware Minimization (GAM) to seek minima with uniformly small curvature across all directions. Experimental results show that GAM improves the generalization of models trained with current optimizers such as SGD and AdamW on various datasets and networks. Furthermore, we show that GAM can help SAM find flatter minima and achieve better generalization.

count=1
* Regularized Vector Quantization for Tokenized Image Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Regularized_Vector_Quantization_for_Tokenized_Image_Synthesis_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Regularized_Vector_Quantization_for_Tokenized_Image_Synthesis_CVPR_2023_paper.pdf)]
    * Title: Regularized Vector Quantization for Tokenized Image Synthesis
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jiahui Zhang, Fangneng Zhan, Christian Theobalt, Shijian Lu
    * Abstract: Quantizing images into discrete representations has been a fundamental problem in unified generative modeling. Predominant approaches learn the discrete representation either in a deterministic manner by selecting the best-matching token or in a stochastic manner by sampling from a predicted distribution. However, deterministic quantization suffers from severe codebook collapse and misaligned inference stage while stochastic quantization suffers from low codebook utilization and perturbed reconstruction objective. This paper presents a regularized vector quantization framework that allows to mitigate above issues effectively by applying regularization from two perspectives. The first is a prior distribution regularization which measures the discrepancy between a prior token distribution and predicted token distribution to avoid codebook collapse and low codebook utilization. The second is a stochastic mask regularization that introduces stochasticity during quantization to strike a good balance between inference stage misalignment and unperturbed reconstruction objective. In addition, we design a probabilistic contrastive loss which serves as a calibrated metric to further mitigate the perturbed reconstruction objective. Extensive experiments show that the proposed quantization framework outperforms prevailing vector quantizers consistently across different generative models including auto-regressive models and diffusion models.

count=1
* FreqHPT: Frequency-Aware Attention and Flow Fusion for Human Pose Transfer
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVFAD/html/Ma_FreqHPT_Frequency-Aware_Attention_and_Flow_Fusion_for_Human_Pose_Transfer_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVFAD/papers/Ma_FreqHPT_Frequency-Aware_Attention_and_Flow_Fusion_for_Human_Pose_Transfer_CVPRW_2023_paper.pdf)]
    * Title: FreqHPT: Frequency-Aware Attention and Flow Fusion for Human Pose Transfer
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Liyuan Ma, Tingwei Gao, Haibin Shen, Kejie Huang
    * Abstract: Human pose transfer is a challenging task that synthesizes images in various target poses while preserving the original appearance. This is typically achieved through aligning the source texture and supplementing it to the target pose. However, most of previous alignment methods only rely on either attention or flow, thereby failing to fully leverage distinctive strengths of these two methods. Moreover, the receptive field of these methods is generally limited in supplementation, resulting in the lack of global texture consistency. To address this issue, observing that attention and flow exhibit distinct characteristics in terms of their frequency distribution, Frequency-aware Human Pose Transfer (FreqHPT) is proposed in this paper. FreqHPT investigates the complementarity between attention and flow from the frequency perspective for improving texture-preserving pose transfer. To this end, FreqHPT first transforms the features from attention and flow into the wavelet domain and then fuses them over multi-frequency bands in an adaptive manner. Subsequently, FreqHPT globally refines the fused features in the Fourier space for texture supplement, enhancing the overall semantic consistency. Extensive experiments on the DeepFashion dataset demonstrate the superiority of FreqHPT in generating texture-preserving and realistic pose transfer images.

count=1
* LSTC-rPPG: Long Short-Term Convolutional Network for Remote Photoplethysmography
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVPM/html/Lee_LSTC-rPPG_Long_Short-Term_Convolutional_Network_for_Remote_Photoplethysmography_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVPM/papers/Lee_LSTC-rPPG_Long_Short-Term_Convolutional_Network_for_Remote_Photoplethysmography_CVPRW_2023_paper.pdf)]
    * Title: LSTC-rPPG: Long Short-Term Convolutional Network for Remote Photoplethysmography
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jun Seong Lee, Gyutae Hwang, Moonwook Ryu, Sang Jun Lee
    * Abstract: Remote photoplethysmography (rPPG) is a non-contact technique for measuring blood pulse signals associated with cardiac activity. Although rPPG is considered an alternative to traditional contact-based photoplethysmography (PPG) because of its non-contact nature, obtaining reliable measurements remains a challenge owing to the sensitiveness of rPPG. In recent years, deep learning-based methods have improved the reliability of rPPG, but they suffer from certain limitations in utilizing long-term features such as periodic tendencies over long durations. In this paper, we propose a deep learning-based method that models long short-term spatio-temporal features and optimizes the long short-term features, ensuring reliable rPPG. The proposed method is composed of three key components: i) a deep learning architecture, denoted by LSTC-rPPG, which models long short-term spatio-temporal features and combines the features for reliable rPPG, ii) a temporal attention refinement module that mitigates temporal mismatches between the long-term and short-term features, and iii) a frequency scale invariant hybrid loss to guide long-short term features. In experiments on the UBFC-rPPG database, the proposed method demonstrated a mean absolute error of 0.7, root mean square error of 1.0, and Pearson correlation coefficient of 0.99 for heart rate estimation accuracy, outperforming contemporary state-of-the-art methods.

count=1
* Benchmarking the Robustness of LiDAR-Camera Fusion for 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/html/Yu_Benchmarking_the_Robustness_of_LiDAR-Camera_Fusion_for_3D_Object_Detection_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Yu_Benchmarking_the_Robustness_of_LiDAR-Camera_Fusion_for_3D_Object_Detection_CVPRW_2023_paper.pdf)]
    * Title: Benchmarking the Robustness of LiDAR-Camera Fusion for 3D Object Detection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Kaicheng Yu, Tang Tao, Hongwei Xie, Zhiwei Lin, Tingting Liang, Bing Wang, Peng Chen, Dayang Hao, Yongtao Wang, Xiaodan Liang
    * Abstract: To achieve autonomous driving, developing 3D detection fusion methods, which aim to fuse the camera and LiDAR information, has draw great research interest in recent years. As a common practice, people rely on large-scale datasets to fairly compare the performance of different methods. While these datasets have been carefully cleaned to ideally minimize any potential noise, we observe that they cannot truly reflect the data seen on a real autonomous vehicle, whose data tends to be noisy due to various reasons. This hinders the ability to simply estimate the robust performance under realistic noisy settings. To this end, we collect a series of real-world cases with noisy data distribution, and systematically formulate a robustness benchmark toolkit. It that can simulate these cases on any clean dataset, which has the camera and LiDAR input modality. We showcase the effectiveness of our toolkit by establishing two novel robustness benchmarks on widely-adopted datasets, nuScenes and Waymo, then holistically evaluate the state-of-the-art fusion methods. We discover that: i) most fusion methods, when solely developed on these data, tend to fail inevitably when there is a disruption to the LiDAR input; ii) the improvement of the camera input is significantly inferior to the LiDAR one. We publish the robust fusion dataset, benchmark, detailed documents and instructions on https://anonymous-benchmark.github.io/robust-benchmark-website2/.

count=1
* Handheld Burst Super-Resolution Meets Multi-Exposure Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Lafenetre_Handheld_Burst_Super-Resolution_Meets_Multi-Exposure_Satellite_Imagery_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Lafenetre_Handheld_Burst_Super-Resolution_Meets_Multi-Exposure_Satellite_Imagery_CVPRW_2023_paper.pdf)]
    * Title: Handheld Burst Super-Resolution Meets Multi-Exposure Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jamy Lafenetre, Ngoc Long Nguyen, Gabriele Facciolo, Thomas Eboli
    * Abstract: Image resolution is an important criterion for many applications based on satellite imagery. In this work, we adapt a state-of-the-art kernel regression technique for smartphone camera burst super-resolution to satellites. This technique leverages the local structure of the image to optimally steer the fusion kernels, limiting blur in the final high-resolution prediction, denoising the image, and recovering details up to a zoom factor of 2. We extend this approach to the multi-exposure case to predict from a sequence of multi-exposure low-resolution frames a high-resolution and noise-free one. Experiments on both single and multi-exposure scenarios show the merits of the approach. Since the fusion is learning-free, the proposed method is ensured to not hallucinate details, which is crucial for many remote sensing applications.

count=1
* Asynchronous Federated Continual Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/FedVision/html/Shenaj_Asynchronous_Federated_Continual_Learning_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/FedVision/papers/Shenaj_Asynchronous_Federated_Continual_Learning_CVPRW_2023_paper.pdf)]
    * Title: Asynchronous Federated Continual Learning
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Donald Shenaj, Marco Toldo, Alberto Rigon, Pietro Zanuttigh
    * Abstract: The standard class-incremental continual learning setting assumes a set of tasks seen one after the other in a fixed and predefined order. This is not very realistic in federated learning environments where each client works independently in an asynchronous manner getting data for the different tasks in time-frames and orders totally uncorrelated with the other ones. We introduce a novel federated learning setting (AFCL) where the continual learning of multiple tasks happens at each client with different orderings and in asynchronous time slots. We tackle this novel task using prototype-based learning, a representation loss, fractal pre-training, and a modified aggregation policy. Our approach, called FedSpace, effectively tackles this task as shown by the results on the CIFAR-100 dataset using 3 different federated splits with 50, 100, and 500 clients, respectively. The code and federated splits are available at https://github.com/LTTM/FedSpace.

count=1
* Visual Gyroscope: Combination of Deep Learning Features and Direct Alignment for Panoramic Stabilization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/OmniCV/html/Berenguel-Baeta_Visual_Gyroscope_Combination_of_Deep_Learning_Features_and_Direct_Alignment_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/OmniCV/papers/Berenguel-Baeta_Visual_Gyroscope_Combination_of_Deep_Learning_Features_and_Direct_Alignment_CVPRW_2023_paper.pdf)]
    * Title: Visual Gyroscope: Combination of Deep Learning Features and Direct Alignment for Panoramic Stabilization
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Bruno Berenguel-Baeta, Antoine N. André, Guillaume Caron, Jesus Bermudez-Cameo, Jose J. Guerrero
    * Abstract: In this article we present a visual gyroscope based on equirectangular panoramas. We propose a new pipeline where we leverage the advantages of the combination of three different methods to obtain a fast and accurate estimation of the attitude of the camera. We quantitatively and qualitatively validate our method on two image sequences taken with a 360 dual-fisheye camera mounted on different aerial vehicles.

count=1
* Thermal Image Super-Resolution Challenge Results - PBVS 2023
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PBVS/html/Rivadeneira_Thermal_Image_Super-Resolution_Challenge_Results_-_PBVS_2023_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PBVS/papers/Rivadeneira_Thermal_Image_Super-Resolution_Challenge_Results_-_PBVS_2023_CVPRW_2023_paper.pdf)]
    * Title: Thermal Image Super-Resolution Challenge Results - PBVS 2023
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Rafael E. Rivadeneira, Angel D. Sappa, Boris X. Vintimilla, Dai Bin, Li Ruodi, Li Shengye, Zhiwei Zhong, Xianming Liu, Junjun Jiang, Chenyang Wang
    * Abstract: This paper presents the results of two tracks from the fourth Thermal Image Super-Resolution (TISR) challenge, held at the Perception Beyond the Visible Spectrum (PBVS) 2023 workshop. Track-1 uses the same thermal image dataset as previous challenges, with 951 training images and 50 validation images at each resolution. In this track, two evaluations were conducted: the first consists of generating a SR image from a HR thermal noisy image downsampled by four, and the second consists of generating a SR image from a mid-resolution image and compare it with its semi-registered HR image (acquired with another camera). The results of Track-1 outperformed those from last year's challenge. On the other hand, Track-2 uses a new acquired dataset consisting of 160 registered visible and thermal images of the same scenario for training and 30 validation images. This year, more than 150 teams participated in the challenge tracks, demonstrating the community's ongoing interest in this topic.

count=1
* FLIGHT Mode On: A Feather-Light Network for Low-Light Image Enhancement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/UG2/html/Ozcan_FLIGHT_Mode_On_A_Feather-Light_Network_for_Low-Light_Image_Enhancement_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/UG2/papers/Ozcan_FLIGHT_Mode_On_A_Feather-Light_Network_for_Low-Light_Image_Enhancement_CVPRW_2023_paper.pdf)]
    * Title: FLIGHT Mode On: A Feather-Light Network for Low-Light Image Enhancement
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Mustafa Ozcan, Hamza Ergezer, Mustafa Ayazoğlu
    * Abstract: Low-light image enhancement (LLIE) is an ill-posed inverse problem due to the lack of knowledge of the desired image which is obtained under ideal illumination conditions. Low-light conditions give rise to two main issues: a suppressed image histogram and inconsistent relative color distributions with low signal-to-noise ratio. In order to address these problems, we propose a novel approach named FLIGHT-Net using a sequence of neural architecture blocks. The first block regulates illumination conditions through pixel-wise scene dependent illumination adjustment. The output image is produced in the output of the second block, which includes channel attention and denoising sub-blocks. Our highly efficient neural network architecture delivers state-of-the-art performance with only 25K parameters. The method's code, pretrained models and resulting images will be publicly available

count=1
* N-Pad: Neighboring Pixel-Based Industrial Anomaly Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Jang_N-Pad_Neighboring_Pixel-Based_Industrial_Anomaly_Detection_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/VISION/papers/Jang_N-Pad_Neighboring_Pixel-Based_Industrial_Anomaly_Detection_CVPRW_2023_paper.pdf)]
    * Title: N-Pad: Neighboring Pixel-Based Industrial Anomaly Detection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: JunKyu Jang, Eugene Hwang, Sung-Hyuk Park
    * Abstract: Identifying defects in the images of industrial products has been an important task to enhance quality control and reduce maintenance costs. In recent studies, industrial anomaly detection models were developed using pre-trained networks to learn nominal representations. To employ the relative positional information of each pixel, we present N-pad, a novel method for anomaly detection and segmentation in a one-class learning setting that includes the neighborhood of the target pixel for model training and evaluation. Within the model architecture, pixel-wise nominal distributions are estimated by using the features of neighboring pixels with the target pixel to allow possible marginal misalignment. Moreover, the centroids from clusters of nominal features are identified as a representative nominal set. Accordingly, anomaly scores are inferred based on the Mahalanobis distances and Euclidean distances between the target pixel and the estimated distributions or the centroid set, respectively. Thus, we have achieved state-of-the-art performance in MVTec-AD with AUROC of 99.37 for anomaly detection and 98.75 for anomaly segmentation, reducing the error by 34% compared to the next best performing model. Experiments in various settings further validate our model.

count=1
* OrCo: Towards Better Generalization via Orthogonality and Contrast for Few-Shot Class-Incremental Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ahmed_OrCo_Towards_Better_Generalization_via_Orthogonality_and_Contrast_for_Few-Shot_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ahmed_OrCo_Towards_Better_Generalization_via_Orthogonality_and_Contrast_for_Few-Shot_CVPR_2024_paper.pdf)]
    * Title: OrCo: Towards Better Generalization via Orthogonality and Contrast for Few-Shot Class-Incremental Learning
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Noor Ahmed, Anna Kukleva, Bernt Schiele
    * Abstract: Few-Shot Class-Incremental Learning (FSCIL) introduces a paradigm in which the problem space expands with limited data. FSCIL methods inherently face the challenge of catastrophic forgetting as data arrives incrementally making models susceptible to overwriting previously acquired knowledge. Moreover given the scarcity of labeled samples available at any given time models may be prone to overfitting and find it challenging to strike a balance between extensive pretraining and the limited incremental data. To address these challenges we propose the OrCo framework built on two core principles: features' orthogonality in the representation space and contrastive learning. In particular we improve the generalization of the embedding space by employing a combination of supervised and self-supervised contrastive losses during the pretraining phase. Additionally we introduce OrCo loss to address challenges arising from data limitations during incremental sessions. Through feature space perturbations and orthogonality between classes the OrCo loss maximizes margins and reserves space for the following incremental data. This in turn ensures the accommodation of incoming classes in the feature space without compromising previously acquired knowledge. Our experimental results showcase state-of-the-art performance across three benchmark datasets including mini-ImageNet CIFAR100 and CUB datasets. Code is available at https://github.com/noorahmedds/OrCo

count=1
* Looking 3D: Anomaly Detection with 2D-3D Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Bhunia_Looking_3D_Anomaly_Detection_with_2D-3D_Alignment_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Bhunia_Looking_3D_Anomaly_Detection_with_2D-3D_Alignment_CVPR_2024_paper.pdf)]
    * Title: Looking 3D: Anomaly Detection with 2D-3D Alignment
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ankan Bhunia, Changjian Li, Hakan Bilen
    * Abstract: Automatic anomaly detection based on visual cues holds practical significance in various domains such as manufacturing and product quality assessment. This paper introduces a new conditional anomaly detection problem which involves identifying anomalies in a query image by comparing it to a reference shape. To address this challenge we have created a large dataset BrokenChairs-180K consisting of around 180K images with diverse anomalies geometries and textures paired with 8143 reference 3D shapes. To tackle this task we have proposed a novel transformer-based approach that explicitly learns the correspondence between the query image and reference 3D shape via feature alignment and leverages a customized attention mechanism for anomaly detection. Our approach has been rigorously evaluated through comprehensive experiments serving as a benchmark for future research in this domain.

count=1
* On the Robustness of Language Guidance for Low-Level Vision Tasks: Findings from Depth Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Chatterjee_On_the_Robustness_of_Language_Guidance_for_Low-Level_Vision_Tasks_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Chatterjee_On_the_Robustness_of_Language_Guidance_for_Low-Level_Vision_Tasks_CVPR_2024_paper.pdf)]
    * Title: On the Robustness of Language Guidance for Low-Level Vision Tasks: Findings from Depth Estimation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Agneet Chatterjee, Tejas Gokhale, Chitta Baral, Yezhou Yang
    * Abstract: Recent advances in monocular depth estimation have been made by incorporating natural language as additional guidance. Although yielding impressive results the impact of the language prior particularly in terms of generalization and robustness remains unexplored. In this paper we address this gap by quantifying the impact of this prior and introduce methods to benchmark its effectiveness across various settings. We generate "low-level" sentences that convey object-centric three-dimensional spatial relationships incorporate them as additional language priors and evaluate their downstream impact on depth estimation. Our key finding is that current language-guided depth estimators perform optimally only with scene-level descriptions and counter-intuitively fare worse with low level descriptions. Despite leveraging additional data these methods are not robust to directed adversarial attacks and decline in performance with an increase in distribution shift. Finally to provide a foundation for future research we identify points of failures and offer insights to better understand these shortcomings. With an increasing number of methods using language for depth estimation our findings highlight the opportunities and pitfalls that require careful consideration for effective deployment in real-world settings.

count=1
* ConsistDreamer: 3D-Consistent 2D Diffusion for High-Fidelity Scene Editing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_ConsistDreamer_3D-Consistent_2D_Diffusion_for_High-Fidelity_Scene_Editing_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_ConsistDreamer_3D-Consistent_2D_Diffusion_for_High-Fidelity_Scene_Editing_CVPR_2024_paper.pdf)]
    * Title: ConsistDreamer: 3D-Consistent 2D Diffusion for High-Fidelity Scene Editing
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jun-Kun Chen, Samuel Rota Bulò, Norman Müller, Lorenzo Porzi, Peter Kontschieder, Yu-Xiong Wang
    * Abstract: This paper proposes ConsistDreamer - a novel framework that lifts 2D diffusion models with 3D awareness and 3D consistency thus enabling high-fidelity instruction-guided scene editing. To overcome the fundamental limitation of missing 3D consistency in 2D diffusion models our key insight is to introduce three synergetic strategies that augment the input of the 2D diffusion model to become 3D-aware and to explicitly enforce 3D consistency during the training process. Specifically we design surrounding views as context-rich input for the 2D diffusion model and generate 3D-consistent structured noise instead of image-independent noise. Moreover we introduce self-supervised consistency-enforcing training within the per-scene editing procedure. Extensive evaluation shows that our ConsistDreamer achieves state-of-the-art performance for instruction-guided scene editing across various scenes and editing instructions particularly in complicated large-scale indoor scenes from ScanNet++ with significantly improved sharpness and fine-grained textures. Notably ConsistDreamer stands as the first work capable of successfully editing complex (e.g. plaid/checkered) patterns.

count=1
* Driving-Video Dehazing with Non-Aligned Regularization for Safety Assistance
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Driving-Video_Dehazing_with_Non-Aligned_Regularization_for_Safety_Assistance_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Driving-Video_Dehazing_with_Non-Aligned_Regularization_for_Safety_Assistance_CVPR_2024_paper.pdf)]
    * Title: Driving-Video Dehazing with Non-Aligned Regularization for Safety Assistance
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Junkai Fan, Jiangwei Weng, Kun Wang, Yijun Yang, Jianjun Qian, Jun Li, Jian Yang
    * Abstract: Real driving-video dehazing poses a significant challenge due to the inherent difficulty in acquiring precisely aligned hazy/clear video pairs for effective model training especially in dynamic driving scenarios with unpredictable weather conditions. In this paper we propose a pioneering approach that addresses this challenge through a nonaligned regularization strategy. Our core concept involves identifying clear frames that closely match hazy frames serving as references to supervise a video dehazing network. Our approach comprises two key components: reference matching and video dehazing. Firstly we introduce a non-aligned reference frame matching module leveraging an adaptive sliding window to match high-quality reference frames from clear videos. Video dehazing incorporates flow-guided cosine attention sampler and deformable cosine attention fusion modules to enhance spatial multiframe alignment and fuse their improved information. To validate our approach we collect a GoProHazy dataset captured effortlessly with GoPro cameras in diverse rural and urban road environments. Extensive experiments demonstrate the superiority of the proposed method over current state-of-the-art methods in the challenging task of real driving-video dehazing. Project page.

count=1
* ConTex-Human: Free-View Rendering of Human from a Single Image with Texture-Consistent Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_ConTex-Human_Free-View_Rendering_of_Human_from_a_Single_Image_with_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_ConTex-Human_Free-View_Rendering_of_Human_from_a_Single_Image_with_CVPR_2024_paper.pdf)]
    * Title: ConTex-Human: Free-View Rendering of Human from a Single Image with Texture-Consistent Synthesis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xiangjun Gao, Xiaoyu Li, Chaopeng Zhang, Qi Zhang, Yanpei Cao, Ying Shan, Long Quan
    * Abstract: In this work we propose a method to address the challenge of rendering a 3D human from a single image in a free-view manner. Some existing approaches could achieve this by using generalizable pixel-aligned implicit fields to reconstruct a textured mesh of a human or by employing a 2D diffusion model as guidance with the Score Distillation Sampling (SDS) method to lift the 2D image into 3D space. However a generalizable implicit field often results in an over-smooth texture field while the SDS method tends to lead to a texture-inconsistent novel view with the input image. In this paper we introduce a texture-consistent back view synthesis method that could transfer the reference image content to the back view through depth-guided mutual self-attention. With this method we could achieve high-fidelity and texture-consistent human rendering from a single image. Moreover to alleviate the color distortion that occurs in the side region we propose a visibility-aware patch consistency regularization combined with the synthesized back view texture. Experiments conducted on both real and synthetic data demonstrate the effectiveness of our method and show that our approach outperforms previous baseline methods.

count=1
* Can Biases in ImageNet Models Explain Generalization?
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Gavrikov_Can_Biases_in_ImageNet_Models_Explain_Generalization_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Gavrikov_Can_Biases_in_ImageNet_Models_Explain_Generalization_CVPR_2024_paper.pdf)]
    * Title: Can Biases in ImageNet Models Explain Generalization?
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Paul Gavrikov, Janis Keuper
    * Abstract: The robust generalization of models to rare in-distribution (ID) samples drawn from the long tail of the training distribution and to out-of-training-distribution (OOD) samples is one of the major challenges of current deep learning methods. For image classification this manifests in the existence of adversarial attacks the performance drops on distorted images and a lack of generalization to concepts such as sketches. The current understanding of generalization in neural networks is very limited but some biases that differentiate models from human vision have been identified and might be causing these limitations. Consequently several attempts with varying success have been made to reduce these biases during training to improve generalization. We take a step back and sanity-check these attempts. Fixing the architecture to the well-established ResNet-50 we perform a large-scale study on 48 ImageNet models obtained via different training methods to understand how and if these biases - including shape bias spectral biases and critical bands - interact with generalization. Our extensive study results reveal that contrary to previous findings these biases are insufficient to accurately predict the generalization of a model holistically. We provide access to all checkpoints and evaluation code at https://github.com/paulgavrikov/biases_vs_generalization/

count=1
* Multi-agent Collaborative Perception via Motion-aware Robust Communication Network
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Hong_Multi-agent_Collaborative_Perception_via_Motion-aware_Robust_Communication_Network_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_Multi-agent_Collaborative_Perception_via_Motion-aware_Robust_Communication_Network_CVPR_2024_paper.pdf)]
    * Title: Multi-agent Collaborative Perception via Motion-aware Robust Communication Network
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Shixin Hong, Yu Liu, Zhi Li, Shaohui Li, You He
    * Abstract: Collaborative perception allows for information sharing between multiple agents such as vehicles and infrastructure to obtain a comprehensive view of the environment through communication and fusion. Current research on multi-agent collaborative perception systems often assumes ideal communication and perception environments and neglects the effect of real-world noise such as pose noise motion blur and perception noise. To address this gap in this paper we propose a novel motion-aware robust communication network (MRCNet) that mitigates noise interference and achieves accurate and robust collaborative perception. MRCNet consists of two main components: multi-scale robust fusion (MRF) addresses pose noise by developing cross-semantic multi-scale enhanced aggregation to fuse features of different scales while motion enhanced mechanism (MEM) captures motion context to compensate for information blurring caused by moving objects. Experimental results on popular collaborative 3D object detection datasets demonstrate that MRCNet outperforms competing methods in noisy scenarios with improved perception performance using less bandwidth.

count=1
* Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.pdf)]
    * Title: Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xiuquan Hou, Meiqin Liu, Senlin Zhang, Ping Wei, Badong Chen
    * Abstract: DETR-like methods have significantly increased detection performance in an end-to-end manner. The mainstream two-stage frameworks of them perform dense self-attention and select a fraction of queries for sparse cross-attention which is proven effective for improving performance but also introduces a heavy computational burden and high dependence on stable query selection. This paper demonstrates that suboptimal two-stage selection strategies result in scale bias and redundancy due to the mismatch between selected queries and objects in two-stage initialization. To address these issues we propose hierarchical salience filtering refinement which performs transformer encoding only on filtered discriminative queries for a better trade-off between computational efficiency and precision. The filtering process overcomes scale bias through a novel scale-independent salience supervision. To compensate for the semantic misalignment among queries we introduce elaborate query refinement modules for stable two-stage initialization. Based on above improvements the proposed Salience DETR achieves significant improvements of +4.0% AP +0.2% AP +4.4% AP on three challenging task-specific detection datasets as well as 49.2% AP on COCO 2017 with less FLOPs. The code is available at https://github.com/xiuqhou/Salience-DETR.

count=1
* SurMo: Surface-based 4D Motion Modeling for Dynamic Human Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_SurMo_Surface-based_4D_Motion_Modeling_for_Dynamic_Human_Rendering_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_SurMo_Surface-based_4D_Motion_Modeling_for_Dynamic_Human_Rendering_CVPR_2024_paper.pdf)]
    * Title: SurMo: Surface-based 4D Motion Modeling for Dynamic Human Rendering
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tao Hu, Fangzhou Hong, Ziwei Liu
    * Abstract: Dynamic human rendering from video sequences has achieved remarkable progress by formulating the rendering as a mapping from static poses to human images. However existing methods focus on the human appearance reconstruction of every single frame while the temporal motion relations are not fully explored. In this paper we propose a new 4D motion modeling paradigm SurMo that jointly models the temporal dynamics and human appearances in a unified framework with three key designs: 1) Surface-based motion encoding that models 4D human motions with an efficient compact surface-based triplane. It encodes both spatial and temporal motion relations on the dense surface manifold of a statistical body template which inherits body topology priors for generalizable novel view synthesis with sparse training observations. 2) Physical motion decoding that is designed to encourage physical motion learning by decoding the motion triplane features at timestep t to predict both spatial derivatives and temporal derivatives at the next timestep t+1 in the training stage. 3) 4D appearance decoding that renders the motion triplanes into images by an efficient volumetric surface-conditioned renderer that focuses on the rendering of body surfaces with motion learning conditioning. Extensive experiments validate the state-of-the-art performance of our new paradigm and illustrate the expressiveness of surface-based motion triplanes for rendering high-fidelity view-consistent humans with fast motions and even motion-dependent shadows. Our project page is at: https://taohuumd.github.io/projects/SurMo.

count=1
* Anomaly Score: Evaluating Generative Models and Individual Generated Images based on Complexity and Vulnerability
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Hwang_Anomaly_Score_Evaluating_Generative_Models_and_Individual_Generated_Images_based_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Hwang_Anomaly_Score_Evaluating_Generative_Models_and_Individual_Generated_Images_based_CVPR_2024_paper.pdf)]
    * Title: Anomaly Score: Evaluating Generative Models and Individual Generated Images based on Complexity and Vulnerability
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jaehui Hwang, Junghyuk Lee, Jong-Seok Lee
    * Abstract: With the advancement of generative models the assessment of generated images becomes increasingly more important. Previous methods measure distances between features of reference and generated images from trained vision models. In this paper we conduct an extensive investigation into the relationship between the representation space and input space around generated images. We first propose two measures related to the presence of unnatural elements within images: complexity which indicates how non-linear the representation space is and vulnerability which is related to how easily the extracted feature changes by adversarial input changes. Based on these we introduce a new metric to evaluating image-generative models called anomaly score (AS). Moreover we propose AS-i (anomaly score for individual images) that can effectively evaluate generated images individually. Experimental results demonstrate the validity of the proposed approach.

count=1
* NViST: In the Wild New View Synthesis from a Single Image with Transformers
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Jang_NViST_In_the_Wild_New_View_Synthesis_from_a_Single_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Jang_NViST_In_the_Wild_New_View_Synthesis_from_a_Single_CVPR_2024_paper.pdf)]
    * Title: NViST: In the Wild New View Synthesis from a Single Image with Transformers
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Wonbong Jang, Lourdes Agapito
    * Abstract: We propose NViST a transformer-based model for efficient and generalizable novel-view synthesis from a single image for real-world scenes. In contrast to many methods that are trained on synthetic data object-centred scenarios or in a category-specific manner NViST is trained on MVImgNet a large-scale dataset of casually-captured real-world videos of hundreds of object categories with diverse backgrounds. NViST transforms image inputs directly into a radiance field conditioned on camera parameters via adaptive layer normalisation. In practice NViST exploits fine-tuned masked autoencoder (MAE) features and translates them to 3D output tokens via cross-attention while addressing occlusions with self-attention. To move away from object-centred datasets and enable full scene synthesis NViST adopts a 6-DOF camera pose model and only requires relative pose dropping the need for canonicalization of the training data which removes a substantial barrier to it being used on casually captured datasets. We show results on unseen objects and categories from MVImgNet and even generalization to casual phone captures. We conduct qualitative and quantitative evaluations on MVImgNet and ShapeNet to show that our model represents a step forward towards enabling true in-the-wild generalizable novel-view synthesis from a single image. Project webpage: https://wbjang.github.io/nvist_webpage.

count=1
* KeyPoint Relative Position Encoding for Face Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.pdf)]
    * Title: KeyPoint Relative Position Encoding for Face Recognition
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Minchul Kim, Yiyang Su, Feng Liu, Anil Jain, Xiaoming Liu
    * Abstract: In this paper we address the challenge of making ViT models more robust to unseen affine transformations. Such robustness becomes useful in various recognition tasks such as face recognition when image alignment failures occur. We propose a novel method called KP-RPE which leverages key points (e.g.facial landmarks) to make ViT more resilient to scale translation and pose variations. We begin with the observation that Relative Position Encoding (RPE) is a good way to bring affine transform generalization to ViTs. RPE however can only inject the model with prior knowledge that nearby pixels are more important than far pixels. Keypoint RPE (KP-RPE) is an extension of this principle where the significance of pixels is not solely dictated by their proximity but also by their relative positions to specific keypoints within the image. By anchoring the significance of pixels around keypoints the model can more effectively retain spatial relationships even when those relationships are disrupted by affine transformations. We show the merit of KP-RPE in face and gait recognition. The experimental results demonstrate the effectiveness in improving face recognition performance from low-quality images particularly where alignment is prone to failure. Code and pre-trained models are available.

count=1
* Flow-Guided Online Stereo Rectification for Wide Baseline Stereo
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kumar_Flow-Guided_Online_Stereo_Rectification_for_Wide_Baseline_Stereo_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Kumar_Flow-Guided_Online_Stereo_Rectification_for_Wide_Baseline_Stereo_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xing_Enhancing_Quality_of_Compressed_Images_by_Mitigating_Enhancement_Bias_Towards_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xing_Enhancing_Quality_of_Compressed_Images_by_Mitigating_Enhancement_Bias_Towards_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kumar_Flow-Guided_Online_Stereo_Rectification_for_Wide_Baseline_Stereo_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Kumar_Flow-Guided_Online_Stereo_Rectification_for_Wide_Baseline_Stereo_CVPR_2024_paper.pdf)]
    * Title: Flow-Guided Online Stereo Rectification for Wide Baseline Stereo
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Anush Kumar, Fahim Mannan, Omid Hosseini Jafari, Shile Li, Felix Heide
    * Abstract: Stereo rectification is widely considered "solved" due to the abundance of traditional approaches to perform rectification. However autonomous vehicles and robots in-the-wild require constant re-calibration due to exposure to various environmental factors including vibration and structural stress when cameras are arranged in a wide-baseline configuration. Conventional rectification methods fail in these challenging scenarios: especially for larger vehicles such as autonomous freight trucks and semi-trucks the resulting incorrect rectification severely affects the quality of downstream tasks that use stereo/multi-view data. To tackle these challenges we propose an online rectification approach that operates at real-time rates while achieving high accuracy. We propose a novel learning-based online calibration approach that utilizes stereo correlation volumes built from a feature representation obtained from cross-image attention. Our model is trained to minimize vertical optical flow as proxy rectification constraint and predicts the relative rotation between the stereo pair. The method is real-time and even outperforms conventional methods used for offline calibration and substantially improves downstream stereo depth post-rectification. We release two public datasets (https://light.princeton.edu/online-stereo-recification/) a synthetic and experimental wide baseline dataset to foster further research.

count=1
* HardMo: A Large-Scale Hardcase Dataset for Motion Capture
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liao_HardMo_A_Large-Scale_Hardcase_Dataset_for_Motion_Capture_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_HardMo_A_Large-Scale_Hardcase_Dataset_for_Motion_Capture_CVPR_2024_paper.pdf)]
    * Title: HardMo: A Large-Scale Hardcase Dataset for Motion Capture
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jiaqi Liao, Chuanchen Luo, Yinuo Du, Yuxi Wang, Xucheng Yin, Man Zhang, Zhaoxiang Zhang, Junran Peng
    * Abstract: Recent years have witnessed rapid progress in monocular human mesh recovery. Despite their impressive performance on public benchmarks existing methods are vulnerable to unusual poses which prevents them from deploying to challenging scenarios such as dance and martial arts. This issue is mainly attributed to the domain gap induced by the data scarcity in relevant cases. Most existing datasets are captured in constrained scenarios and lack samples of such complex movements. For this reason we propose a data collection pipeline comprising automatic crawling precise annotation and hardcase mining. Based on this pipeline we establish a large dataset in a short time. The dataset named HardMo contains 7M images along with precise annotations covering 15 categories of dance and 14 categories of martial arts. Empirically we find that the prediction failure in dance and martial arts is mainly characterized by the misalignment of hand-wrist and foot-ankle. To dig deeper into the two hardcases we leverage the proposed automatic pipeline to filter collected data and construct two subsets named HardMo-Hand and HardMo-Foot. Extensive experiments demonstrate the effectiveness of the annotation pipeline and the data-driven solution to failure cases. Specifically after being trained on HardMo HMR an early pioneering method can even outperform the current state of the art 4DHumans on our benchmarks.

count=1
* FaceCom: Towards High-fidelity 3D Facial Shape Completion via Optimization and Inpainting Guidance
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_FaceCom_Towards_High-fidelity_3D_Facial_Shape_Completion_via_Optimization_and_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_FaceCom_Towards_High-fidelity_3D_Facial_Shape_Completion_via_Optimization_and_CVPR_2024_paper.pdf)]
    * Title: FaceCom: Towards High-fidelity 3D Facial Shape Completion via Optimization and Inpainting Guidance
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yinglong Li, Hongyu Wu, Xiaogang Wang, Qingzhao Qin, Yijiao Zhao, Yong Wang, Aimin Hao
    * Abstract: We propose FaceCom a method for 3D facial shape completion which delivers high-fidelity results for incomplete facial inputs of arbitrary forms. Unlike end-to-end shape completion methods based on point clouds or voxels our approach relies on a mesh-based generative network that is easy to optimize enabling it to handle shape completion for irregular facial scans. We first train a shape generator on a mixed 3D facial dataset containing 2405 identities. Based on the incomplete facial input we fit complete faces using an optimization approach under image inpainting guidance. The completion results are refined through a post-processing step. FaceCom demonstrates the ability to effectively and naturally complete facial scan data with varying missing regions and degrees of missing areas. Our method can be used in medical prosthetic fabrication and the registration of deficient scanning data. Our experimental results demonstrate that FaceCom achieves exceptional performance in fitting and shape completion tasks.

count=1
* Split to Merge: Unifying Separated Modalities for Unsupervised Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Split_to_Merge_Unifying_Separated_Modalities_for_Unsupervised_Domain_Adaptation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Split_to_Merge_Unifying_Separated_Modalities_for_Unsupervised_Domain_Adaptation_CVPR_2024_paper.pdf)]
    * Title: Split to Merge: Unifying Separated Modalities for Unsupervised Domain Adaptation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xinyao Li, Yuke Li, Zhekai Du, Fengling Li, Ke Lu, Jingjing Li
    * Abstract: Large vision-language models (VLMs) like CLIP have demonstrated good zero-shot learning performance in the unsupervised domain adaptation task. Yet most transfer approaches for VLMs focus on either the language or visual branches overlooking the nuanced interplay between both modalities. In this work we introduce a Unified Modality Separation (UniMoS) framework for unsupervised domain adaptation. Leveraging insights from modality gap studies we craft a nimble modality separation network that distinctly disentangles CLIP's features into language-associated and vision-associated components. Our proposed Modality-Ensemble Training (MET) method fosters the exchange of modality-agnostic information while maintaining modality-specific nuances. We align features across domains using a modality discriminator. Comprehensive evaluations on three benchmarks reveal our approach sets a new state-of-the-art with minimal computational costs. Code: https://github.com/TL-UESTC/UniMoS.

count=1
* Self-Calibrating Vicinal Risk Minimisation for Model Calibration
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Self-Calibrating_Vicinal_Risk_Minimisation_for_Model_Calibration_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Self-Calibrating_Vicinal_Risk_Minimisation_for_Model_Calibration_CVPR_2024_paper.pdf)]
    * Title: Self-Calibrating Vicinal Risk Minimisation for Model Calibration
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jiawei Liu, Changkun Ye, Ruikai Cui, Nick Barnes
    * Abstract: Model calibration measuring the alignment between the prediction accuracy and model confidence is an important metric reflecting model trustworthiness. Existing dense binary classification methods without proper regularisation of model confidence are prone to being over-confident. To calibrate Deep Neural Networks (DNNs) we propose a Self-Calibrating Vicinal Risk Minimisation (SCVRM) that explores the vicinity space of labeled data where vicinal images that are farther away from labeled images adopt the groundtruth label with decreasing label confidence. We prove that in the logistic regression problem SCVRM can be seen as a Vicinal Risk Minimisation plus a regularisation term that penalises the over-confident predictions. In practical implementation SCVRM is approximated using Monte Carlo sampling that samples additional augmented training images and labels from the vicinal distributions. Experimental results demonstrate that SCVRM can significantly enhance model calibration for different dense classification tasks on both in-distribution and out-of-distribution data. Code is available at https://github.com/Carlisle-Liu/SCVRM.

count=1
* RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Qiu_RichDreamer_A_Generalizable_Normal-Depth_Diffusion_Model_for_Detail_Richness_in_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Qiu_RichDreamer_A_Generalizable_Normal-Depth_Diffusion_Model_for_Detail_Richness_in_CVPR_2024_paper.pdf)]
    * Title: RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, Xiaoguang Han
    * Abstract: Lifting 2D diffusion for 3D generation is a challenging problem due to the lack of geometric prior and the complex entanglement of materials and lighting in natural images. Existing methods have shown promise by first creating the geometry through score-distillation sampling (SDS) applied to rendered surface normals followed by appearance modeling. However relying on a 2D RGB diffusion model to optimize surface normals is suboptimal due to the distribution discrepancy between natural images and normals maps leading to instability in optimization. In this paper recognizing that the normal and depth information effectively describe scene geometry and be automatically estimated from images we propose to learn a generalizable Normal-Depth diffusion model for 3D generation. We achieve this by training on the large-scale LAION dataset together with the generalizable image-to-depth and normal prior models. In an attempt to alleviate the mixed illumination effects in the generated materials we introduce an albedo diffusion model to impose data-driven constraints on the albedo component. Our experiments show that when integrated into existing text-to-3D pipelines our models significantly enhance the detail richness achieving state-of-the-art results. Our project page is at https://aigc3d.github.io/richdreamer/.

count=1
* Towards Modern Image Manipulation Localization: A Large-Scale Dataset and Novel Methods
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Qu_Towards_Modern_Image_Manipulation_Localization_A_Large-Scale_Dataset_and_Novel_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Qu_Towards_Modern_Image_Manipulation_Localization_A_Large-Scale_Dataset_and_Novel_CVPR_2024_paper.pdf)]
    * Title: Towards Modern Image Manipulation Localization: A Large-Scale Dataset and Novel Methods
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chenfan Qu, Yiwu Zhong, Chongyu Liu, Guitao Xu, Dezhi Peng, Fengjun Guo, Lianwen Jin
    * Abstract: In recent years image manipulation localization has attracted increasing attention due to its pivotal role in ensuring social media security. However effectively identifying forged regions remains an open challenge. The high acquisition cost and the severe scarcity of high-quality data are major factors hindering the performance improvement of modern image manipulation localization systems. To address this issue we propose a novel paradigm termed as CAAA to automatically and accurately annotate the manually forged images from the web at the pixel-level. We further propose a novel metric termed as QES to assist in filtering out unreliable annotations. With CAAA and QES we construct a large-scale diverse and high-quality dataset comprising 123150 manually forged images with mask annotations. Furthermore we develop a new model termed as APSC-Net for accurate image manipulation localization. According to extensive experiments our methods outperforms previous state-of-the-art methods our dataset significantly improves the performance of various models on the widely-used benchmarks. The dataset and codes are publicly available at https://github.com/qcf-568/MIML.

count=1
* UnScene3D: Unsupervised 3D Instance Segmentation for Indoor Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Rozenberszki_UnScene3D_Unsupervised_3D_Instance_Segmentation_for_Indoor_Scenes_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Rozenberszki_UnScene3D_Unsupervised_3D_Instance_Segmentation_for_Indoor_Scenes_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Rozenberszki_UnScene3D_Unsupervised_3D_Instance_Segmentation_for_Indoor_Scenes_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Rozenberszki_UnScene3D_Unsupervised_3D_Instance_Segmentation_for_Indoor_Scenes_CVPR_2024_paper.pdf)]
    * Title: UnScene3D: Unsupervised 3D Instance Segmentation for Indoor Scenes
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: David Rozenberszki, Or Litany, Angela Dai
    * Abstract: 3D instance segmentation is fundamental to geometric understanding of the world around us. Existing methods for instance segmentation of 3D scenes rely on supervision from expensive manual 3D annotations. We propose UnScene3D the first fully unsupervised 3D learning approach for class-agnostic 3D instance segmentation of indoor scans. UnScene3D first generates pseudo masks by leveraging self-supervised color and geometry features to find potential object regions. We operate on a basis of 3D segment primitives enabling efficient representation and learning on high-resolution 3D data. The coarse proposals are then refined through self-training our model on its predictions. Our approach improves over state-of-the-art unsupervised 3D instance segmentation methods by more than 300% Average Precision score demonstrating effective instance segmentation even in challenging cluttered 3D scenes.

count=1
* AlignMiF: Geometry-Aligned Multimodal Implicit Field for LiDAR-Camera Joint Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Tao_AlignMiF_Geometry-Aligned_Multimodal_Implicit_Field_for_LiDAR-Camera_Joint_Synthesis_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tao_AlignMiF_Geometry-Aligned_Multimodal_Implicit_Field_for_LiDAR-Camera_Joint_Synthesis_CVPR_2024_paper.pdf)]
    * Title: AlignMiF: Geometry-Aligned Multimodal Implicit Field for LiDAR-Camera Joint Synthesis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tang Tao, Guangrun Wang, Yixing Lao, Peng Chen, Jie Liu, Liang Lin, Kaicheng Yu, Xiaodan Liang
    * Abstract: Neural implicit fields have been a de facto standard in novel view synthesis. Recently there exist some methods exploring fusing multiple modalities within a single field aiming to share implicit features from different modalities to enhance reconstruction performance. However these modalities often exhibit misaligned behaviors: optimizing for one modality such as LiDAR can adversely affect another like camera performance and vice versa. In this work we conduct comprehensive analyses on the multimodal implicit field of LiDAR-camera joint synthesis revealing the underlying issue lies in the misalignment of different sensors. Furthermore we introduce AlignMiF a geometrically aligned multimodal implicit field with two proposed modules: Geometry-Aware Alignment (GAA) and Shared Geometry Initialization (SGI). These modules effectively align the coarse geometry across different modalities significantly enhancing the fusion process between LiDAR and camera data. Through extensive experiments across various datasets and scenes we demonstrate the effectiveness of our approach in facilitating better interaction between LiDAR and camera modalities within a unified neural field. Specifically our proposed AlignMiF achieves remarkable improvement over recent implicit fusion methods (+2.01 and +3.11 image PSNR on the KITTI-360 and Waymo datasets) and consistently surpasses single modality performance (13.8% and 14.2% reduction in LiDAR Chamfer Distance on the respective datasets).

count=1
* Image Processing GNN: Breaking Rigidity in Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Tian_Image_Processing_GNN_Breaking_Rigidity_in_Super-Resolution_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tian_Image_Processing_GNN_Breaking_Rigidity_in_Super-Resolution_CVPR_2024_paper.pdf)]
    * Title: Image Processing GNN: Breaking Rigidity in Super-Resolution
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yuchuan Tian, Hanting Chen, Chao Xu, Yunhe Wang
    * Abstract: Super-Resolution (SR) reconstructs high-resolution images from low-resolution ones. CNNs and window-attention methods are two major categories of canonical SR models. However these measures are rigid: in both operations each pixel gathers the same number of neighboring pixels hindering their effectiveness in SR tasks. Alternatively we leverage the flexibility of graphs and propose the Image Processing GNN (IPG) model to break the rigidity that dominates previous SR methods. Firstly SR is unbalanced in that most reconstruction efforts are concentrated to a small proportion of detail-rich image parts. Hence we leverage degree flexibility by assigning higher node degrees to detail-rich image nodes. Then in order to construct graphs for SR-effective aggregation we treat images as pixel node sets rather than patch nodes. Lastly we hold that both local and global information are crucial for SR performance. In the hope of gathering pixel information from both local and global scales efficiently via flexible graphs we search node connections within nearby regions to construct local graphs; and find connections within a strided sampling space of the whole image for global graphs. The flexibility of graphs boosts the SR performance of the IPG model. Experiment results on various datasets demonstrates that the proposed IPG outperforms State-of-the-Art baselines. Codes are available at https://github.com/huawei-noah/Efficient-Computing/tree/master/LowLevel/IPG.

count=1
* MotionEditor: Editing Video Motion via Content-Aware Diffusion
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Tu_MotionEditor_Editing_Video_Motion_via_Content-Aware_Diffusion_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tu_MotionEditor_Editing_Video_Motion_via_Content-Aware_Diffusion_CVPR_2024_paper.pdf)]
    * Title: MotionEditor: Editing Video Motion via Content-Aware Diffusion
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Shuyuan Tu, Qi Dai, Zhi-Qi Cheng, Han Hu, Xintong Han, Zuxuan Wu, Yu-Gang Jiang
    * Abstract: Existing diffusion-based video editing models have made gorgeous advances for editing attributes of a source video over time but struggle to manipulate the motion information while preserving the original protagonist's appearance and background. To address this we propose MotionEditor the first diffusion model for video motion editing. MotionEditor incorporates a novel content-aware motion adapter into ControlNet to capture temporal motion correspondence. While ControlNet enables direct generation based on skeleton poses it encounters challenges when modifying the source motion in the inverted noise due to contradictory signals between the noise (source) and the condition (reference). Our adapter complements ControlNet by involving source content to transfer adapted control signals seamlessly. Further we build up a two-branch architecture (a reconstruction branch and an editing branch) with a high-fidelity attention injection mechanism facilitating branch interaction. This mechanism enables the editing branch to query the key and value from the reconstruction branch in a decoupled manner making the editing branch retain the original background and protagonist appearance. We also propose a skeleton alignment algorithm to address the discrepancies in pose size and position. Experiments demonstrate the promising motion editing ability of MotionEditor both qualitatively and quantitatively. To the best of our knowledge MotionEditor is the first to use diffusion models specifically for video motion editing considering the origin dynamic background and camera movement.

count=1
* MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.pdf)]
    * Title: MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel
    * Abstract: Contrastive pre-training of image-text foundation models such as CLIP demonstrated excellent zero-shot performance and improved robustness on a wide range of downstream tasks. However these models utilize large transformer-based encoders with significant memory and latency overhead which pose challenges for deployment on mobile devices. In this work we introduce MobileCLIP - a new family of efficient image-text models optimized for runtime performance along with a novel and efficient training approach namely multi-modal reinforced training. The proposed training approach leverages knowledge transfer from an image captioning model and an ensemble of strong CLIP encoders to improve the accuracy of efficient models. Our approach avoids train-time compute overhead by storing the additional knowledge in a reinforced dataset. MobileCLIP sets a new state-of-the-art latency-accuracy tradeoff for zero-shot classification and retrieval tasks on several datasets. Our MobileCLIP-S2 variant is 2.3x faster while more accurate compared to previous best CLIP model based on ViT-B/16. We further demonstrate the effectiveness of our multi-modal reinforced training by training a CLIP model based on ViT-B/16 image backbone and achieving +2.9% average performance improvement on 38 evaluation benchmarks compared to the previous best. Moreover we show that the proposed approach achieves 10x-1000x improved learning efficiency when compared with non- reinforced CLIP training. Code and models are available at https://github.com/apple/ml-mobileclip

count=1
* DiffPerformer: Iterative Learning of Consistent Latent Guidance for Diffusion-based Human Video Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DiffPerformer_Iterative_Learning_of_Consistent_Latent_Guidance_for_Diffusion-based_Human_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DiffPerformer_Iterative_Learning_of_Consistent_Latent_Guidance_for_Diffusion-based_Human_CVPR_2024_paper.pdf)]
    * Title: DiffPerformer: Iterative Learning of Consistent Latent Guidance for Diffusion-based Human Video Generation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chenyang Wang, Zerong Zheng, Tao Yu, Xiaoqian Lv, Bineng Zhong, Shengping Zhang, Liqiang Nie
    * Abstract: Existing diffusion models for pose-guided human video generation mostly suffer from temporal inconsistency in the generated appearance and poses due to the inherent randomization nature of the generation process. In this paper we propose a novel framework DiffPerformer to synthesize high-fidelity and temporally consistent human video. Without complex architecture modification or costly training DiffPerformer finetunes a pretrained diffusion model on a single video of the target character and introduces an implicit video representation as a proxy to learn temporally consistent guidance for the diffusion model. The guidance is encoded into VAE latent space and an iterative optimization loop is constructed between the implicit video representation and the diffusion model allowing to harness the smooth property of the implicit video representation and the generative capabilities of the diffusion model in a mutually beneficial way. Moreover we propose 3D-aware human flow as a temporal constraint during the optimization to explicitly model the correspondence between driving poses and human appearance. This alleviates the misalignment between guided poses and target performer and therefore maintains the appearance coherence under various motions. Extensive experiments demonstrate that our method outperforms the state-of-the-art methods.

count=1
* PARA-Drive: Parallelized Architecture for Real-time Autonomous Driving
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Weng_PARA-Drive_Parallelized_Architecture_for_Real-time_Autonomous_Driving_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_PARA-Drive_Parallelized_Architecture_for_Real-time_Autonomous_Driving_CVPR_2024_paper.pdf)]
    * Title: PARA-Drive: Parallelized Architecture for Real-time Autonomous Driving
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, Marco Pavone
    * Abstract: Recent works have proposed end-to-end autonomous vehicle (AV) architectures comprised of differentiable modules achieving state-of-the-art driving performance. While they provide advantages over the traditional perception-prediction-planning pipeline (e.g. removing information bottlenecks between components and alleviating integration challenges) they do so using a diverse combination of tasks modules and their interconnectivity. As of yet however there has been no systematic analysis of the necessity of these modules or the impact of their connectivity placement and internal representations on overall driving performance. Addressing this gap our work conducts a comprehensive exploration of the design space of end-to-end modular AV stacks. Our findings culminate in the development of PARA-Drive: a fully parallel end-to-end AV architecture. PARA-Drive not only achieves state-of-the-art performance in perception prediction and planning but also significantly enhances runtime speed by nearly 3x without compromising on interpretability or safety.

count=1
* Cache Me if You Can: Accelerating Diffusion Models through Block Caching
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wimbauer_Cache_Me_if_You_Can_Accelerating_Diffusion_Models_through_Block_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wimbauer_Cache_Me_if_You_Can_Accelerating_Diffusion_Models_through_Block_CVPR_2024_paper.pdf)]
    * Title: Cache Me if You Can: Accelerating Diffusion Models through Block Caching
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, Christian Rupprecht, Daniel Cremers, Peter Vajda, Jialiang Wang
    * Abstract: Diffusion models have recently revolutionized the field of image synthesis due to their ability to generate photorealistic images. However one of the major drawbacks of diffusion models is that the image generation process is costly. A large image-to-image network has to be applied many times to iteratively refine an image from random noise. While many recent works propose techniques to reduce the number of required steps they generally treat the underlying denoising network as a black box. In this work we investigate the behavior of the layers within the network and find that 1) the layers' output changes smoothly over time 2) the layers show distinct patterns of change and 3) the change from step to step is often very small. We hypothesize that many layer computations in the denoising network are redundant. Leveraging this we introduce Block Caching in which we reuse outputs from layer blocks of previous steps to speed up inference. Furthermore we propose a technique to automatically determine caching schedules based on each block's changes over timesteps. In our experiments we show through FID human evaluation and qualitative analysis that Block Caching allows to generate images with higher visual quality at the same computational cost. We demonstrate this for different state-of-the-art models (LDM and EMU) and solvers (DDIM and DPM).

count=1
* DIBS: Enhancing Dense Video Captioning with Unlabeled Videos via Pseudo Boundary Enrichment and Online Refinement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.pdf)]
    * Title: DIBS: Enhancing Dense Video Captioning with Unlabeled Videos via Pseudo Boundary Enrichment and Online Refinement
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Hao Wu, Huabin Liu, Yu Qiao, Xiao Sun
    * Abstract: We present Dive Into the Boundaries (DIBS) a novel pretraining framework for dense video captioning (DVC) that elaborates on improving the quality of the generated event captions and their associated pseudo event boundaries from unlabeled videos. By leveraging the capabilities of diverse large language models (LLMs) we generate rich DVC-oriented caption candidates and optimize the corresponding pseudo boundaries under several meticulously designed objectives considering diversity event-centricity temporal ordering and coherence. Moreover we further introduce a novel online boundary refinement strategy that iteratively improves the quality of pseudo boundaries during training. Comprehensive experiments have been conducted to examine the effectiveness of the proposed technique components. By leveraging a substantial amount of unlabeled video data such as HowTo100M we achieve a remarkable advancement on standard DVC datasets like YouCook2 and ActivityNet. We outperform the previous state-of-the-art Vid2Seq across a majority of metrics achieving this with just 0.4% of the unlabeled video data used for pre-training by Vid2Seq.

count=1
* Perception-Oriented Video Frame Interpolation via Asymmetric Blending
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Perception-Oriented_Video_Frame_Interpolation_via_Asymmetric_Blending_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Perception-Oriented_Video_Frame_Interpolation_via_Asymmetric_Blending_CVPR_2024_paper.pdf)]
    * Title: Perception-Oriented Video Frame Interpolation via Asymmetric Blending
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Guangyang Wu, Xin Tao, Changlin Li, Wenyi Wang, Xiaohong Liu, Qingqing Zheng
    * Abstract: Previous methods for Video Frame Interpolation (VFI) have encountered challenges notably the manifestation of blur and ghosting effects. These issues can be traced back to two pivotal factors: unavoidable motion errors and misalignment in supervision. In practice motion estimates often prove to be error-prone resulting in misaligned features. Furthermore the reconstruction loss tends to bring blurry results particularly in misaligned regions. To mitigate these challenges we propose a new paradigm called PerVFI (Perception-oriented Video Frame Interpolation). Our approach incorporates an Asymmetric Synergistic Blending module (ASB) that utilizes features from both sides to synergistically blend intermediate features. One reference frame emphasizes primary content while the other contributes complementary information. To impose a stringent constraint on the blending process we introduce a self-learned sparse quasi-binary mask which effectively mitigates ghosting and blur artifacts in the output. Additionally we employ a normalizing flow-based generator and utilize the negative log-likelihood loss to learn the conditional distribution of the output which further facilitates the generation of clear and fine details. Experimental results validate the superiority of PerVFI demonstrating significant improvements in perceptual quality compared to existing methods. Codes are available at https://github.com/mulns/PerVFI

count=1
* Self-correcting LLM-controlled Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Self-correcting_LLM-controlled_Diffusion_Models_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Self-correcting_LLM-controlled_Diffusion_Models_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Self-correcting_LLM-controlled_Diffusion_Models_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Self-correcting_LLM-controlled_Diffusion_Models_CVPR_2024_paper.pdf)]
    * Title: Self-correcting LLM-controlled Diffusion Models
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tsung-Han Wu, Long Lian, Joseph E. Gonzalez, Boyi Li, Trevor Darrell
    * Abstract: Text-to-image generation has witnessed significant progress with the advent of diffusion models. Despite the ability to generate photorealistic images current text-to-image diffusion models still often struggle to accurately interpret and follow complex input text prompts. In contrast to existing models that aim to generate images only with their best effort we introduce Self-correcting LLM-controlled Diffusion (SLD). SLD is a framework that generates an image from the input prompt assesses its alignment with the prompt and performs self-corrections on the inaccuracies in the generated image. Steered by an LLM controller SLD turns text-to-image generation into an iterative closed-loop process ensuring correctness in the resulting image. SLD is not only training-free but can also be seamlessly integrated with diffusion models behind API access such as DALL-E 3 to further boost the performance of state-of-the-art diffusion models. Experimental results show that our approach can rectify a majority of incorrect generations particularly in generative numeracy attribute binding and spatial relationships. Furthermore by simply adjusting the instructions to the LLM SLD can perform image editing tasks bridging the gap between text-to-image generation and image editing pipelines. Our code is available at: https://self-correcting-llm-diffusion.github.io.

count=1
* Text-Guided 3D Face Synthesis - From Generation to Editing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Text-Guided_3D_Face_Synthesis_-_From_Generation_to_Editing_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Text-Guided_3D_Face_Synthesis_-_From_Generation_to_Editing_CVPR_2024_paper.pdf)]
    * Title: Text-Guided 3D Face Synthesis - From Generation to Editing
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yunjie Wu, Yapeng Meng, Zhipeng Hu, Lincheng Li, Haoqian Wu, Kun Zhou, Weiwei Xu, Xin Yu
    * Abstract: Text-guided 3D face synthesis has achieved remarkable results by leveraging text-to-image (T2I) diffusion models. However most existing works focus solely on the direct generation ignoring the editing restricting them from synthesizing customized 3D faces through iterative adjustments. In this paper we propose a unified text-guided framework from face generation to editing. In the generation stage we propose a geometry-texture decoupled generation to mitigate the loss of geometric details caused by coupling. Besides decoupling enables us to utilize the generated geometry as a condition for texture generation yielding highly geometry-texture aligned results. We further employ a fine-tuned texture diffusion model to enhance texture quality in both RGB and YUV space. In the editing stage we first employ a pre-trained diffusion model to update facial geometry or texture based on the texts. To enable sequential editing we introduce a UV domain consistency preservation regularization preventing unintentional changes to irrelevant facial attributes. Besides we propose a self-guided consistency weight strategy to improve editing efficacy while preserving consistency. Through comprehensive experiments we showcase our method's superiority in face synthesis.

count=1
* Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Adaptive_Multi-Modal_Cross-Entropy_Loss_for_Stereo_Matching_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Adaptive_Multi-Modal_Cross-Entropy_Loss_for_Stereo_Matching_CVPR_2024_paper.pdf)]
    * Title: Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Peng Xu, Zhiyu Xiang, Chengyu Qiao, Jingyun Fu, Tianyu Pu
    * Abstract: Despite the great success of deep learning in stereo matching recovering accurate disparity maps is still challenging. Currently L1 and cross-entropy are the two most widely used losses for stereo network training. Compared with the former the latter usually performs better thanks to its probability modeling and direct supervision to the cost volume. However how to accurately model the stereo ground-truth for cross-entropy loss remains largely under-explored. Existing works simply assume that the ground-truth distributions are uni-modal which ignores the fact that most of the edge pixels can be multi-modal. In this paper a novel adaptive multi-modal cross-entropy loss (ADL) is proposed to guide the networks to learn different distribution patterns for each pixel. Moreover we optimize the disparity estimator to further alleviate the bleeding or misalignment artifacts in inference. Extensive experimental results show that our method is generic and can help classic stereo networks regain state-of-the-art performance. In particular GANet with our method ranks 1st on both the KITTI 2015 and 2012 benchmarks among the published methods. Meanwhile excellent synthetic-to-realistic generalization performance can be achieved by simply replacing the traditional loss with ours. Code is available at https://github.com/xxxupeng/ADL.

count=1
* Multi-Attribute Interactions Matter for 3D Visual Grounding
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Multi-Attribute_Interactions_Matter_for_3D_Visual_Grounding_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Multi-Attribute_Interactions_Matter_for_3D_Visual_Grounding_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Multi-Attribute_Interactions_Matter_for_3D_Visual_Grounding_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Multi-Attribute_Interactions_Matter_for_3D_Visual_Grounding_CVPR_2024_paper.pdf)]
    * Title: Multi-Attribute Interactions Matter for 3D Visual Grounding
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Can Xu, Yuehui Han, Rui Xu, Le Hui, Jin Xie, Jian Yang
    * Abstract: 3D visual grounding aims to localize 3D objects described by free-form language sentences. Following the detection-then-matching paradigm existing methods mainly focus on embedding object attributes in unimodal feature extraction and multimodal feature fusion to enhance the discriminability of the proposal feature for accurate grounding. However most of them ignore the explicit interaction of multiple attributes causing a bias in unimodal representation and misalignment in multimodal fusion. In this paper we propose a multi-attribute aware Transformer for 3D visual grounding learning the multi-attribute interactions to refine the intra-modal and inter-modal grounding cues. Specifically we first develop an attribute causal analysis module to quantify the causal effect of different attributes for the final prediction which provides powerful supervision to correct the misleading attributes and adaptively capture other discriminative features. Then we design an exchanging-based multimodal fusion module which dynamically replaces tokens with low attribute attention between modalities before directly integrating low-dimensional global features. This ensures an attribute-level multimodal information fusion and helps align the language and vision details more efficiently for fine-grained multimodal features. Extensive experiments show that our method can achieve state-of-the-art performance on ScanRefer and Sr3D/Nr3D datasets.

count=1
* Forecasting of 3D Whole-body Human Poses with Grasping Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Forecasting_of_3D_Whole-body_Human_Poses_with_Grasping_Objects_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Forecasting_of_3D_Whole-body_Human_Poses_with_Grasping_Objects_CVPR_2024_paper.pdf)]
    * Title: Forecasting of 3D Whole-body Human Poses with Grasping Objects
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Haitao Yan, Qiongjie Cui, Jiexin Xie, Shijie Guo
    * Abstract: In the context of computer vision and human-robot interaction forecasting 3D human poses is crucial for understanding human behavior and enhancing the predictive capabilities of intelligent systems. While existing methods have made significant progress they often focus on predicting major body joints overlooking fine-grained gestures and their interaction with objects. Human hand movements particularly during object interactions play a pivotal role and provide more precise expressions of human poses. This work fills this gap and introduces a novel paradigm: forecasting 3D whole-body human poses with a focus on grasping objects. This task involves predicting activities across all joints in the body and hands encompassing the complexities of internal heterogeneity and external interactivity. To tackle these challenges we also propose a novel approach: C^3HOST cross-context cross-modal consolidation for 3D whole-body pose forecasting effectively handles the complexities of internal heterogeneity and external interactivity. C^3HOST involves distinct steps including the heterogeneous content encoding and alignment and cross-modal feature learning and interaction. These enable us to predict activities across all body and hand joints ensuring high-precision whole-body human pose prediction even during object grasping. Extensive experiments on two benchmarks demonstrate that our model significantly enhances the accuracy of whole-body human motion prediction. The project page is available at https://sites.google.com/view/c3host.

count=1
* TextureDreamer: Image-Guided Texture Synthesis Through Geometry-Aware Diffusion
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yeh_TextureDreamer_Image-Guided_Texture_Synthesis_Through_Geometry-Aware_Diffusion_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yeh_TextureDreamer_Image-Guided_Texture_Synthesis_Through_Geometry-Aware_Diffusion_CVPR_2024_paper.pdf)]
    * Title: TextureDreamer: Image-Guided Texture Synthesis Through Geometry-Aware Diffusion
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yu-Ying Yeh, Jia-Bin Huang, Changil Kim, Lei Xiao, Thu Nguyen-Phuoc, Numair Khan, Cheng Zhang, Manmohan Chandraker, Carl S Marshall, Zhao Dong, Zhengqin Li
    * Abstract: We present TextureDreamer a novel image-guided texture synthesis method to transfer relightable textures from a small number of input images (3 to 5) to target 3D shapes across arbitrary categories. Texture creation is a pivotal challenge in vision and graphics. Industrial companies hire experienced artists to manually craft textures for 3D assets. Classical methods require densely sampled views and accurately aligned geometry while learning-based methods are confined to category-specific shapes within the dataset. In contrast TextureDreamer can transfer highly detailed intricate textures from real-world environments to arbitrary objects with only a few casually captured images potentially significantly democratizing texture creation. Our core idea personalized geometry-aware score distillation (PGSD) draws inspiration from recent advancements in diffuse models including personalized modeling for texture information extraction score distillation for detailed appearance synthesis and explicit geometry guidance with ControlNet. Our integration and several essential modifications substantially improve the texture quality. Experiments on real images spanning different categories show that TextureDreamer can successfully transfer highly realistic semantic meaningful texture to arbitrary objects surpassing the visual quality of previous state-of-the-art. Project page: https://texturedreamer.github.io

count=1
* Boosting Adversarial Training via Fisher-Rao Norm-based Regularization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_Boosting_Adversarial_Training_via_Fisher-Rao_Norm-based_Regularization_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_Boosting_Adversarial_Training_via_Fisher-Rao_Norm-based_Regularization_CVPR_2024_paper.pdf)]
    * Title: Boosting Adversarial Training via Fisher-Rao Norm-based Regularization
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xiangyu Yin, Wenjie Ruan
    * Abstract: Adversarial training is extensively utilized to improve the adversarial robustness of deep neural networks. Yet mitigating the degradation of standard generalization performance in adversarial-trained models remains an open problem. This paper attempts to resolve this issue through the lens of model complexity. First We leverage the Fisher-Rao norm a geometrically invariant metric for model complexity to establish the non-trivial bounds of the Cross-Entropy Loss-based Rademacher complexity for a ReLU-activated Multi-Layer Perceptron. Building upon this observation we propose a novel regularization framework called Logit-Oriented Adversarial Training (LOAT) which can mitigate the trade-off between robustness and accuracy while imposing only a negligible increase in computational overhead. Our extensive experiments demonstrate that the proposed regularization strategy can boost the performance of the prevalent adversarial training algorithms including PGD-AT TRADES TRADES (LSE) MART and DM-AT across various network architectures. Our code will be available at https://github.com/TrustAI/LOAT.

count=1
* FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Youk_FMA-Net_Flow-Guided_Dynamic_Filtering_and_Iterative_Feature_Refinement_with_Multi-Attention_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Youk_FMA-Net_Flow-Guided_Dynamic_Filtering_and_Iterative_Feature_Refinement_with_Multi-Attention_CVPR_2024_paper.pdf)]
    * Title: FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Geunhyuk Youk, Jihyong Oh, Munchurl Kim
    * Abstract: We present a joint learning scheme of video super-resolution and deblurring called VSRDB to restore clean high-resolution (HR) videos from blurry low-resolution (LR) ones. This joint restoration problem has drawn much less attention compared to single restoration problems. In this paper we propose a novel flow-guided dynamic filtering (FGDF) and iterative feature refinement with multi-attention (FRMA) which constitutes our VSRDB framework denoted as FMA-Net. Specifically our proposed FGDF enables precise estimation of both spatio-temporally-variant degradation and restoration kernels that are aware of motion trajectories through sophisticated motion representation learning. Compared to conventional dynamic filtering the FGDF enables the FMA-Net to effectively handle large motions into the VSRDB. Additionally the stacked FRMA blocks trained with our novel temporal anchor (TA) loss which temporally anchors and sharpens features refine features in a coarse-to-fine manner through iterative updates. Extensive experiments demonstrate the superiority of the proposed FMA-Net over state-of-the-art methods in terms of both quantitative and qualitative quality. Codes and pre-trained models are available at: https://kaist-viclab.github.io/fmanet-site.

count=1
* LoSh: Long-Short Text Joint Prediction Network for Referring Video Object Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_LoSh_Long-Short_Text_Joint_Prediction_Network_for_Referring_Video_Object_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_LoSh_Long-Short_Text_Joint_Prediction_Network_for_Referring_Video_Object_CVPR_2024_paper.pdf)]
    * Title: LoSh: Long-Short Text Joint Prediction Network for Referring Video Object Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Linfeng Yuan, Miaojing Shi, Zijie Yue, Qijun Chen
    * Abstract: Referring video object segmentation (RVOS) aims to segment the target instance referred by a given text expression in a video clip. The text expression normally contains sophisticated description of the instance's appearance action and relation with others. It is therefore rather difficult for a RVOS model to capture all these attributes correspondingly in the video; in fact the model often favours more on the action- and relation-related visual attributes of the instance. This can end up with partial or even incorrect mask prediction of the target instance. We tackle this problem by taking a subject-centric short text expression from the original long text expression. The short one retains only the appearance-related information of the target instance so that we can use it to focus the model's attention on the instance's appearance. We let the model make joint predictions using both long and short text expressions; and insert a long-short cross-attention module to interact the joint features and a long-short predictions intersection loss to regulate the joint predictions. Besides the improvement on the linguistic part we also introduce a forward-backward visual consistency loss which utilizes optical flows to warp visual features between the annotated frames and their temporal neighbors for consistency. We build our method on top of two state of the art pipelines. Extensive experiments on A2D-Sentences Refer-YouTube-VOS JHMDB-Sentences and Refer-DAVIS17 show impressive improvements of our method. Code is available here.

count=1
* Binarized Low-light Raw Video Enhancement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Binarized_Low-light_Raw_Video_Enhancement_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Binarized_Low-light_Raw_Video_Enhancement_CVPR_2024_paper.pdf)]
    * Title: Binarized Low-light Raw Video Enhancement
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Gengchen Zhang, Yulun Zhang, Xin Yuan, Ying Fu
    * Abstract: Recently deep neural networks have achieved excellent performance on low-light raw video enhancement. However they often come with high computational complexity and large memory costs which hinder their applications on resource-limited devices. In this paper we explore the feasibility of applying the extremely compact binary neural network (BNN) to low-light raw video enhancement. Nevertheless there are two main issues with binarizing video enhancement models. One is how to fuse the temporal information to improve low-light denoising without complex modules. The other is how to narrow the performance gap between binary convolutions with the full precision ones. To address the first issue we introduce a spatial-temporal shift operation which is easy-to-binarize and effective. The temporal shift efficiently aggregates the features of neighbor frames and the spatial shift handles the misalignment caused by the large motion in videos. For the second issue we present a distribution-aware binary convolution which captures the distribution characteristics of real-valued input and incorporates them into plain binary convolutions to alleviate the degradation in performance. Extensive quantitative and qualitative experiments have shown our high-efficiency binarized low-light raw video enhancement method can attain a promising performance. The code is available at https://github.com/ying-fu/BRVE.

count=1
* Diffusion-based Blind Text Image Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Diffusion-based_Blind_Text_Image_Super-Resolution_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Diffusion-based_Blind_Text_Image_Super-Resolution_CVPR_2024_paper.pdf)]
    * Title: Diffusion-based Blind Text Image Super-Resolution
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yuzhe Zhang, Jiawei Zhang, Hao Li, Zhouxia Wang, Luwei Hou, Dongqing Zou, Liheng Bian
    * Abstract: Recovering degraded low-resolution text images is challenging especially for Chinese text images with complex strokes and severe degradation in real-world scenarios. Ensuring both text fidelity and style realness is crucial for high-quality text image super-resolution. Recently diffusion models have achieved great success in natural image synthesis and restoration due to their powerful data distribution modeling abilities and data generation capabilities. In this work we propose an Image Diffusion Model (IDM) to restore text images with realistic styles. For diffusion models they are not only suitable for modeling realistic image distribution but also appropriate for learning text distribution. Since text prior is important to guarantee the correctness of the restored text structure according to existing arts we also propose a Text Diffusion Model (TDM) for text recognition which can guide IDM to generate text images with correct structures. We further propose a Mixture of Multi-modality module (MoM) to make these two diffusion models cooperate with each other in all the diffusion steps. Extensive experiments on synthetic and real-world datasets demonstrate that our Diffusion-based Blind Text Image Super-Resolution (DiffTSR) can restore text images with more accurate text structures as well as more realistic appearances simultaneously. Code is available at https://github.com/YuzheZhang-1999/DiffTSR.

count=1
* SeNM-VAE: Semi-Supervised Noise Modeling with Hierarchical Variational Autoencoder
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_SeNM-VAE_Semi-Supervised_Noise_Modeling_with_Hierarchical_Variational_Autoencoder_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zheng_SeNM-VAE_Semi-Supervised_Noise_Modeling_with_Hierarchical_Variational_Autoencoder_CVPR_2024_paper.pdf)]
    * Title: SeNM-VAE: Semi-Supervised Noise Modeling with Hierarchical Variational Autoencoder
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Dihan Zheng, Yihang Zou, Xiaowen Zhang, Chenglong Bao
    * Abstract: The data bottleneck has emerged as a fundamental challenge in learning based image restoration methods. Researchers have attempted to generate synthesized training data using paired or unpaired samples to address this challenge. This study proposes SeNM-VAE a semi-supervised noise modeling method that leverages both paired and unpaired datasets to generate realistic degraded data. Our approach is based on modeling the conditional distribution of degraded and clean images with a specially designed graphical model. Under the variational inference framework we develop an objective function for handling both paired and unpaired data. We employ our method to generate paired training samples for real-world image denoising and super-resolution tasks. Our approach excels in the quality of synthetic degraded images compared to other unpaired and paired noise modeling methods. Furthermore our approach demonstrates remarkable performance in downstream image restoration tasks even with limited paired data. With more paired data our method achieves the best performance on the SIDD dataset.

count=1
* Language-guided Image Reflection Separation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhong_Language-guided_Image_Reflection_Separation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhong_Language-guided_Image_Reflection_Separation_CVPR_2024_paper.pdf)]
    * Title: Language-guided Image Reflection Separation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Haofeng Zhong, Yuchen Hong, Shuchen Weng, Jinxiu Liang, Boxin Shi
    * Abstract: This paper studies the problem of language-guided reflection separation which aims at addressing the ill-posed reflection separation problem by introducing language descriptions to provide layer content. We propose a unified framework to solve this problem which leverages the cross-attention mechanism with contrastive learning strategies to construct the correspondence between language descriptions and image layers. A gated network design and a randomized training strategy are employed to tackle the recognizable layer ambiguity. The effectiveness of the proposed method is validated by the significant performance advantage over existing reflection separation methods on both quantitative and qualitative comparisons.

count=1
* Addressing Background Context Bias in Few-Shot Segmentation through Iterative Modulation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Addressing_Background_Context_Bias_in_Few-Shot_Segmentation_through_Iterative_Modulation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Addressing_Background_Context_Bias_in_Few-Shot_Segmentation_through_Iterative_Modulation_CVPR_2024_paper.pdf)]
    * Title: Addressing Background Context Bias in Few-Shot Segmentation through Iterative Modulation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Lanyun Zhu, Tianrun Chen, Jianxiong Yin, Simon See, Jun Liu
    * Abstract: Existing few-shot segmentation methods usually extract foreground prototypes from support images to guide query image segmentation. However different background contexts of support and query images can cause their foreground features to be misaligned. This phenomenon known as background context bias can hinder the effectiveness of support prototypes in guiding query image segmentation. In this work we propose a novel framework with an iterative structure to address this problem. In each iteration of the framework we first generate a query prediction based on a support foreground feature. Next we extract background context from the query image to modulate the support foreground feature thus eliminating the foreground feature misalignment caused by the different backgrounds. After that we design a confidence-biased attention to eliminate noise and cleanse information. By integrating these components through an iterative structure we create a novel network that can leverage the synergies between different modules to improve their performance in a mutually reinforcing manner. Through these carefully designed components and structures our network can effectively eliminate background context bias in few-shot segmentation thus achieving outstanding performance. We conduct extensive experiments on the PASCAL-5^ i and COCO-20^ i datasets and achieve state-of-the-art (SOTA) results which demonstrate the effectiveness of our approach.

count=1
* M&M VTO: Multi-Garment Virtual Try-On and Editing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_MM_VTO_Multi-Garment_Virtual_Try-On_and_Editing_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_MM_VTO_Multi-Garment_Virtual_Try-On_and_Editing_CVPR_2024_paper.pdf)]
    * Title: M&M VTO: Multi-Garment Virtual Try-On and Editing
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Luyang Zhu, Yingwei Li, Nan Liu, Hao Peng, Dawei Yang, Ira Kemelmacher-Shlizerman
    * Abstract: We present M&M VTO-a mix and match virtual try-on method that takes as input multiple garment images text description for garment layout and an image of a person. An example input includes: an image of a shirt an image of a pair of pants "rolled sleeves shirt tucked in" and an image of a person. The output is a visualization of how those garments (in the desired layout) would look like on the given person. Key contributions of our method are: 1) a single stage diffusion based model with no super resolution cascading that allows to mix and match multiple garments at 1024x512 resolution preserving and warping intricate garment details 2) architecture design (VTO UNet Diffusion Transformer) to disentangle denoising from person specific features allowing for a highly effective finetuning strategy for identity preservation (6MB model per individual vs 4GB achieved with e.g. dreambooth finetuning); solving a common identity loss problem in current virtual try-on methods 3) layout control for multiple garments via text inputs finetuned over PaLI-3 for virtual try-on task. Experimental results indicate that M&M VTO achieves state-of-the-art performance both qualitatively and quantitatively as well as opens up new opportunities for virtual try-on via language-guided and multi-garment try-on.

count=1
* Low Latency Point Cloud Rendering with Learned Splatting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Hu_Low_Latency_Point_Cloud_Rendering_with_Learned_Splatting_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Hu_Low_Latency_Point_Cloud_Rendering_with_Learned_Splatting_CVPRW_2024_paper.pdf)]
    * Title: Low Latency Point Cloud Rendering with Learned Splatting
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yueyu Hu, Ran Gong, Qi Sun, Yao Wang
    * Abstract: Point cloud is a critical 3D representation with many emerging applications. Because of the point sparsity and irregularity high-quality rendering of point clouds is challenging and often requires complex computations to recover the continuous surface representation. On the other hand to avoid visual discomfort the motion-to-photon latency has to be very short under 10 ms. Existing rendering solutions lack in either quality or speed. To tackle these challenges we present a framework that unlocks interactive free-viewing and high-fidelity point cloud rendering. We pre-train a neural network to estimate 3D elliptical Gaussians from arbitrary point clouds and use differentiable surface splatting to render smooth texture and surface normal for arbitrary views. Our approach does not require per-scene optimization and enable real-time rendering of dynamic point cloud. Experimental results demonstrate the proposed solution enjoys superior visual quality and speed as well as generalizability to different scene content and robustness to compression artifacts.

count=1
* MA-AVT: Modality Alignment for Parameter-Efficient Audio-Visual Transformers
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/ECV24/html/Mahmud_MA-AVT_Modality_Alignment_for_Parameter-Efficient_Audio-Visual_Transformers_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/ECV24/papers/Mahmud_MA-AVT_Modality_Alignment_for_Parameter-Efficient_Audio-Visual_Transformers_CVPRW_2024_paper.pdf)]
    * Title: MA-AVT: Modality Alignment for Parameter-Efficient Audio-Visual Transformers
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tanvir Mahmud,Shentong Mo,Yapeng Tian,Diana Marculescu
    * Abstract: Recent advances in pre-trained vision transformers have shown promise in parameter-efficient audio-visual learning without audio pre-training. However few studies have investigated effective methods for aligning multimodal features in parameter-efficient audio-visual transformers. In this paper we propose MA-AVT a new parameter-efficient audio-visual transformer employing deep modality alignment for corresponding multimodal semantic features. Specifically we introduce joint unimodal and multimodal token learning for aligning the two modalities with a frozen modality-shared transformer. This allows the model to learn separate representations for each modality while also attending to the cross-modal relationships between them. In addition unlike prior work that only aligns coarse features from the output of unimodal encoders we introduce blockwise contrastive learning to align coarse-to-fine-grained hierarchical features throughout the encoding phase. Furthermore to suppress the background features in each modality from foreground matched audio-visual features we introduce a robust discriminative foreground mining scheme. Through extensive experiments on benchmark AVE VGGSound and CREMA-D datasets we achieve considerable performance improvements over SOTA methods.

count=1
* Image-caption Difficulty for Efficient Weakly-supervised Object Detection from In-the-wild Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/html/Nebbia_Image-caption_Difficulty_for_Efficient_Weakly-supervised_Object_Detection_from_In-the-wild_Data_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Nebbia_Image-caption_Difficulty_for_Efficient_Weakly-supervised_Object_Detection_from_In-the-wild_Data_CVPRW_2024_paper.pdf)]
    * Title: Image-caption Difficulty for Efficient Weakly-supervised Object Detection from In-the-wild Data
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Giacomo Nebbia, Adriana Kovashka
    * Abstract: In recent years we have witnessed the collection of larger and larger multi-modal image-caption datasets: from hundreds of thousands such pairs to hundreds of millions. Such datasets allow researchers to build powerful deep learning models at the cost of requiring intensive computational resources. In this work we ask: can we use such datasets efficiently without sacrificing performance? We tackle this problem by extracting difficulty scores from each image-caption sample and by using such scores to make training more effective and efficient. We compare two ways to use difficulty scores to influence training: filtering a representative subset of each dataset and ordering samples through curriculum learning. We analyze and compare difficulty scores extracted from a single modality---captions (i.e. caption length and number of object mentions) or images (i.e. region proposals' size and number)---or based on alignment of image-caption pairs (i.e. CLIP and concreteness). We focus on Weakly-Supervised Object Detection where image-level labels are extracted from captions. We discover that (1) combining filtering and curriculum learning can achieve large gains in performance but not all methods are stable across experimental settings (2) single-modality scores often outperform alignment-based ones (3) alignment scores show the largest gains when training time is limited.

count=1
* A Method of Moments Embedding Constraint and its Application to Semi-Supervised Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/LIMIT/html/Majurski_A_Method_of_Moments_Embedding_Constraint_and_its_Application_to_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/LIMIT/papers/Majurski_A_Method_of_Moments_Embedding_Constraint_and_its_Application_to_CVPRW_2024_paper.pdf)]
    * Title: A Method of Moments Embedding Constraint and its Application to Semi-Supervised Learning
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Michael Majurski, Sumeet Menon, Parniyan Favardin, David Chapman
    * Abstract: Discriminative deep learning models with a linear+softmax final layer have a problem: the latent space only predicts the conditional probabilities p(Y|X) but not the full joint distribution p(YX) which necessitates a generative approach. The conditional probability cannot detect outliers causing outlier sensitivity in softmax networks. This exacerbates model over-confidence impacting many problems such as hallucinations confounding biases and dependence on large datasets. To address this we introduce a novel embedding constraint based on the Method of Moments (MoM). We investigate the use of polynomial moments ranging from 1st through 4th order hyper-covariance matrices. Furthermore we use this embedding constraint to train an Axis-Aligned Gaussian Mixture Model (AAGMM) final layer which learns not only the conditional but also the joint distribution of the latent space. We apply this method to the domain of semi-supervised image classification by extending FlexMatch with our technique. We find our MoM constraint with the AAGMM layer is able to match the reported FlexMatch accuracy while also modeling the joint distribution thereby reducing outlier sensitivity. We also present a preliminary outlier detection strategy based on Mahalanobis distance and discuss future improvements to this strategy.

count=1
* End-to-End Neural Network Compression via l1/l2 Regularized Latency Surrogates
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/MAI/html/Nasery_End-to-End_Neural_Network_Compression_via_l1l2_Regularized_Latency_Surrogates_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/MAI/papers/Nasery_End-to-End_Neural_Network_Compression_via_l1l2_Regularized_Latency_Surrogates_CVPRW_2024_paper.pdf)]
    * Title: End-to-End Neural Network Compression via l1/l2 Regularized Latency Surrogates
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Anshul Nasery, Hardik Shah, Arun Sai Suggala, Prateek Jain
    * Abstract: Neural network (NN) compression via techniques such as pruning quantization requires setting compression hyperparameters (e.g. number of channels to be pruned bitwidths for quantization) for each layer either manually or via neural architecture search (NAS) which can be computationally expensive. We address this problem by providing an end-to-end technique that optimizes for model's Floating Point Operations (FLOPs) via a novel l_1 / l_2 latency surrogate. Our algorithm is versatile and can be used with many popular compression methods including pruning low-rank factorization and quantization and can optimize for on-device latency. Crucially it is fast and runs in almost the same amount of time as a single model training run ; which is a significant training speed-up over standard NAS methods. For BERT compression on GLUE fine-tuning tasks we achieve 50% reduction in FLOPs with only 1% drop in performance. For compressing MobileNetV3 on ImageNet-1K we achieve 15% reduction in FLOPs without drop in accuracy while still requiring 3 times less training compute than SOTA NAS techniques. Finally for transfer learning on smaller datasets our technique identifies 1.2 times-1.4 times cheaper architectures than standard MobileNetV3 EfficientNet suite of architectures at almost the same training cost and accuracy.

count=1
* S3R-Net: A Single-Stage Approach to Self-Supervised Shadow Removal
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Kubiak_S3R-Net_A_Single-Stage_Approach_to_Self-Supervised_Shadow_Removal_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Kubiak_S3R-Net_A_Single-Stage_Approach_to_Self-Supervised_Shadow_Removal_CVPRW_2024_paper.pdf)]
    * Title: S3R-Net: A Single-Stage Approach to Self-Supervised Shadow Removal
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Nikolina Kubiak, Armin Mustafa, Graeme Phillipson, Stephen Jolly, Simon Hadfield
    * Abstract: In this paper we present S3R-Net the Self-Supervised Shadow Removal Network. The two-branch WGAN model achieves self-supervision relying on the unify-and-adapt phenomenon - it unifies the style of the output data and infers its characteristics from a database of unaligned shadow-free reference images. This approach stands in contrast to the large body of supervised frameworks. S3R-Net also differentiates itself from the few existing self-supervised models operating in a cycle-consistent manner as it is a non-cyclic unidirectional solution. The proposed framework achieves comparable numerical scores to recent self-supervised shadow removal models while exhibiting superior qualitative performance and keeping the computational cost low.

count=1
* Exploring AIGC Video Quality: A Focus on Visual Harmony Video-Text Consistency and Domain Distribution Gap
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Qu_Exploring_AIGC_Video_Quality_A_Focus_on_Visual_Harmony_Video-Text_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Qu_Exploring_AIGC_Video_Quality_A_Focus_on_Visual_Harmony_Video-Text_CVPRW_2024_paper.pdf)]
    * Title: Exploring AIGC Video Quality: A Focus on Visual Harmony Video-Text Consistency and Domain Distribution Gap
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Bowen Qu, Xiaoyu Liang, Shangkun Sun, Wei Gao
    * Abstract: The recent advancements in Text-to-Video Artificial Intelligence Generated Content (AIGC) have been remarkable. Compared with traditional videos the assessment of AIGC videos encounters various challenges: visual inconsistency that defy common sense discrepancies between content and the textual prompt and distribution gap between various generative models etc. Target at these challenges in this work we categorize the assessment of AIGC video quality into three dimensions: visual harmony video-text consistency and domain distribution gap. For each dimension we design specific modules to provide a comprehensive quality assessment of AIGC videos. Furthermore our research identifies significant variations in visual quality fluidity and style among videos generated by different text-to-video models. Predicting the source generative model can make the AIGC video features more discriminative which enhances the quality assessment performance. The proposed method was used in the third-place winner of the NTIRE 2024 Quality Assessment for AI-Generated Content - Track 2 Video demonstrating its effectiveness.

count=1
* NTIRE 2024 Image Shadow Removal Challenge Report
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Vasluianu_NTIRE_2024_Image_Shadow_Removal_Challenge_Report_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Vasluianu_NTIRE_2024_Image_Shadow_Removal_Challenge_Report_CVPRW_2024_paper.pdf)]
    * Title: NTIRE 2024 Image Shadow Removal Challenge Report
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Florin-Alexandru Vasluianu, Tim Seizinger, Zhuyun Zhou, Zongwei Wu, Cailian Chen, Radu Timofte, Wei Dong, Han Zhou, Yuqiong Tian, Jun Chen, Xueyang Fu, Xin Lu, Yurui Zhu, Xi Wang, Dong Li, Jie Xiao, Yunpeng Zhang, Zheng-Jun Zha, Zhao Zhang, Suiyi Zhao, Bo Wang, Yan Luo, Yanyan Wei, Zhihao Zhao, Long Sun, Tingting Yang, Jinshan Pan, Jiangxin Dong, Jinhui Tang, Bilel Benjdira, Mohammed Nassif, Anis Koubaa, Ahmed Elhayek, Anas M. Ali, Kyotaro Tokoro, Kento Kawai, Kaname Yokoyama, Takuya Seno, Yuki Kondo, Norimichi Ukita, Chenghua Li, Bo Yang, Zhiqi Wu, Gao Chen, Yihan Yu, Sixiang Chen, Kai Zhang, Tian Ye, Wenbin Zou, Yunlong Lin, Zhaohu Xing, Jinbin Bai, Wenhao Chai, Lei Zhu, Ritik Maheshwari, Rakshank Verma, Rahul Tekchandani, Praful Hambarde, Satya Narayan Tazi, Santosh Kumar Vipparthi, Subrahmanyam Murala, Jaeho Lee, Seongwan Kim, Sharif S M A, Nodirkhuja Khujaev, Roman Tsoy, Fan Gao, Weidan Yan, Wenze Shao, Dengyin Zhang, Bin Chen, Siqi Zhang, Yanxin Qian, Yuanbin Chen, Yuanbo Zhou, Tong Tong, Rongfeng Wei, Ruiqi Sun, Yue Liu, Nikhil Akalwadi, Amogh Joshi, Sampada Malagi, Chaitra Desai, Ramesh Ashok Tabib, Uma Mudenagudi, Ali Murtaza, Uswah Khairuddin, Ahmad Athif Mohd Faudzi, Adinath Dukre, Vivek Deshmukh, Shruti S. Phutke, Ashutosh Kulkarni, Santosh Kumar Vipparthi, Anil Gonde, Subrahmanyam Murala, Arun Karthik K, Manasa N, Shri Hari Priya, Wei Hao, Xingzhuo Yan, Minghan Fu
    * Abstract: This work reviews the results of the NTIRE 2024 Challenge on Shadow Removal. Building on the last year edition the current challenge was organized in two tracks with a track focused on increased fidelity reconstruction and a separate ranking for high performing perceptual quality solutions. Track 1 (fidelity) had 214 registered participants with 17 teams submitting in the final phase while Track 2 (perceptual) registered 185 participants resulting in 18 final phase submissions. Both tracks were based on data from the WSRD dataset simulating interactions between self-shadows and cast shadows with a large variety of represented objects textures and materials. Improved image alignment enabled increased fidelity reconstruction with restored frames mostly indistinguishable from the references images for top performing solutions.

count=1
* High Quality Reference Feature for Two Stage Bracketing Image Restoration and Enhancement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Xing_High_Quality_Reference_Feature_for_Two_Stage_Bracketing_Image_Restoration_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Xing_High_Quality_Reference_Feature_for_Two_Stage_Bracketing_Image_Restoration_CVPRW_2024_paper.pdf)]
    * Title: High Quality Reference Feature for Two Stage Bracketing Image Restoration and Enhancement
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xiaoxia Xing, Hyunhee Park, Fan Wang, Ying Zhang, Sejun Song, Changho Kim, Xiangyu Kong
    * Abstract: In a low-light environment it is difficult to obtain high-quality or high-resolution images with sharp details and high dynamic range (HDR) without noise or blur. To solve this problem the Bracketing Image Restoration and Enhancement integrates Dnoise Deblur HDR Reconstruction and Super Resolution techniques into a unified framework. However we find that most methods select the image that aligns with GT as the reference image. Since the details of the reference image are not good enough seriously affects the feature fusion which finally leads to details being blurred. To generate a high dynamic range and a high-quality image we propose a two-stage Bracketing method named RT-IRE. In the first stage we generate the high-quality reference feature to guide feature fusion remove the degradation and reconstruct HDR to get coarse results. The second stage learns the residuals between the coarse result and the GT which further enhances and generates details. Extensive experiments show the effectiveness of the proposed module. In particular RT-IRE won two champions in the NTIRE 2024 Bracketing Image Restoration and Enhancement Challenge.

count=1
* Partition and Reunion: A Two-Branch Neural Network for Vehicle Re-identification
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/AI_City/Chen_Partition_and_Reunion_A_Two-Branch_Neural_Network_for_Vehicle_Re-identification_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI City/Chen_Partition_and_Reunion_A_Two-Branch_Neural_Network_for_Vehicle_Re-identification_CVPRW_2019_paper.pdf)]
    * Title: Partition and Reunion: A Two-Branch Neural Network for Vehicle Re-identification
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Hao Chen,  Benoit Lagadec,  Francois Bremond
    * Abstract: The smart city vision raises the prospect that cities will become more intelligent in various fields, such as more sustainable environment and a better quality of life for residents. As a key component of smart cities, intelligent transportation system highlights the importance of vehicle re-identification (Re-ID). However, as compared to the rapid progress on person Re-ID, vehicle Re-ID advances at a relatively slow pace. Some previous state-of-the-art approaches strongly rely on extra annotation, like attributes (e.g., vehicle color and type) and key-points (e.g., wheels and lamps). Recent work on person Re-ID shows that extracting more local features can achieve a better performance without considering extra annotation. In this paper, we propose an end-to-end trainable two-branch Partition and Reunion Network (PRN) for the challenging vehicle ReID task. Utilizing only identity labels, our proposed method outperforms existing state-of-the-art methods on four vehicle Re-ID benchmark datasets, including VeRi-776, VehicleID, VRIC and CityFlow-ReID by a large margin.

count=1
* Segmentation-Less and Non-Holistic Deep-Learning Frameworks for Iris Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/Biometrics/Proenca_Segmentation-Less_and_Non-Holistic_Deep-Learning_Frameworks_for_Iris_Recognition_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Biometrics/Proenca_Segmentation-Less_and_Non-Holistic_Deep-Learning_Frameworks_for_Iris_Recognition_CVPRW_2019_paper.pdf)]
    * Title: Segmentation-Less and Non-Holistic Deep-Learning Frameworks for Iris Recognition
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Hugo Proenca,  Joao C. Neves
    * Abstract: Driven by the pioneer iris biometrics approach, the most relevant recognition methods published over the years are "phase-based", and segment/normalize the iris to obtain dimensionless representations of the data that attenuate the differences in scale, translation, rotation and pupillary dilation. In this paper we present a recognition method that dispenses the iris segmentation, noise detection and normalization phases, and is agnostic to the levels of pupillary dilation, while maintaining state-of-the-art performance. Based on deep-learning classification models, we analyze the displacements between biologically corresponding patches in pairs of iris images, to discriminate between genuine and impostor comparisons. Such corresponding patches are firstly learned in the normalized representations of the irises - the domain where they are optimally distinguishable - but are remapped into a segmentation-less polar coordinate system that uniquely requires iris detection. In recognition time, samples are only converted into this segmentation-less coordinate system, where matching is performed. In the experiments, we considered the challenging open-world setting, and used three well known data sets (CASIA-4-Lamp, CASIA-4-Thousand and WVU), concluding positively about the effectiveness of the proposed algorithm, particularly in cases where accurately segmenting the iris is a challenge.

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
* Learning Minutiae Neighborhoods : A New Binary Representation for Matching Fingerprints
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W01/html/Vij_Learning_Minutiae_Neighborhoods_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W01/papers/Vij_Learning_Minutiae_Neighborhoods_2014_CVPR_paper.pdf)]
    * Title: Learning Minutiae Neighborhoods : A New Binary Representation for Matching Fingerprints
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Akhil Vij, Anoop Namboodiri
    * Abstract: Representation of fingerprints is one of the key factors that limits the accuracy and efficiency of matching algorithms. Most popular methods represent each fingerprint as an unordered set of minutiae with variable cardinality and the matching algorithms are left with the task of finding the best correspondence between the two sets of minutiae. While this makes the representation more flexible and matching more accurate, the task becomes computationally intensive. Fixed length representations with aligned features are highly efficient to match. However, creating an aligned representation without the knowledge of the sample to which it is to be matched, makes the problem of representation more complex. Some of the fixed-length representations only provide partial alignment, leaving the rest to the matching stage. In this paper, we propose a fixed length representation for fingerprints that provides exact alignment between the features, thus enabling high-speed matching with minimal computational effort. The representation extends the idea of object representation using bag of words into a bag of minutiae neighborhoods. The representation is provably invariant to affine transformations (rotation, translation and uniform scaling), and is shown to be highly discriminative for the task of verification. Experimental results on FVC 2002 and 2004 datasets clearly show the superiority of the representation with competing methods. As the proposed representation can be computed from the standard minutiae templates, the method is applicable to existing datasets, where the original fingerprint images are not available.

count=1
* Projection Center Calibration for a Co-located Projector Camera System
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W13/html/Amano_Projection_Center_Calibration_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W13/papers/Amano_Projection_Center_Calibration_2014_CVPR_paper.pdf)]
    * Title: Projection Center Calibration for a Co-located Projector Camera System
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Toshiyuki Amano
    * Abstract: A co-located projector camera system where the projector and camera are positioned in the same optical position by a plate beam splitter enables various spatial augmented reality applications for dynamic three dimensional scenes. The extremely precise alignment of the projection centers of the camera and projector is necessary for these applications. However, the conventional calibration procedure for a camera and projector cannot achieve high accuracy because an iterative verification process for the alignment is not included. This paper proposes a novel interactive alignment approach that displays a capture of the projected grid pattern on the calibration screen. Additionally, a misalignment display technique that employs projector camera feedback is proposed for fine adjustment.

count=1
* Video Stitching With Spatial-Temporal Content-Preserving Warping
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W10/html/Jiang_Video_Stitching_With_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W10/papers/Jiang_Video_Stitching_With_2015_CVPR_paper.pdf)]
    * Title: Video Stitching With Spatial-Temporal Content-Preserving Warping
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Wei Jiang, Jinwei Gu
    * Abstract: We propose a novel algorithm for stitching multiple synchronized video streams into a single panoramic video with spatial-temporal content-preserving warping. Compared to image stitching, video stitching faces several new challenges including temporal coherence, dominate foreground objects moving across views, and camera jittering. To overcome these issues, the proposed algorithm draws upon ideas from recent local warping methods in image stitching and video stabilization. For video frame alignment, we propose spatial-temporal local warping, which locally aligns frames from different videos while maintaining the temporal consistency. For aligned video frame composition, we find stitching seams with 3D graphcut on overlapped spatial-temporal volumes, where the 3D graph is weighted with object and motion saliency to reduce stitching artifacts. Experimental results show the advantages of the proposed algorithm over several state-of-the-art alternatives, especially in challenging conditions.

count=1
* Affine-Constrained Group Sparse Coding and Its Application to Image-Based Classifications
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Chi_Affine-Constrained_Group_Sparse_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Chi_Affine-Constrained_Group_Sparse_2013_ICCV_paper.pdf)]
    * Title: Affine-Constrained Group Sparse Coding and Its Application to Image-Based Classifications
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Yu-Tseh Chi, Mohsen Ali, Muhammad Rushdi, Jeffrey Ho
    * Abstract: This paper proposes a novel approach for sparse coding that further improves upon the sparse representation-based classification (SRC) framework. The proposed framework, Affine-Constrained Group Sparse Coding (ACGSC), extends the current SRC framework to classification problems with multiple input samples. Geometrically, the affineconstrained group sparse coding essentially searches for the vector in the convex hull spanned by the input vectors that can best be sparse coded using the given dictionary. The resulting objective function is still convex and can be efficiently optimized using iterative block-coordinate descent scheme that is guaranteed to converge. Furthermore, we provide a form of sparse recovery result that guarantees, at least theoretically, that the classification performance of the constrained group sparse coding should be at least as good as the group sparse coding. We have evaluated the proposed approach using three different recognition experiments that involve illumination variation of faces and textures, and face recognition under occlusions. Preliminary experiments have demonstrated the effectiveness of the proposed approach, and in particular, the results from the recognition/occlusion experiment are surprisingly accurate and robust.

count=1
* Fine-Grained Categorization by Alignments
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Gavves_Fine-Grained_Categorization_by_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Gavves_Fine-Grained_Categorization_by_2013_ICCV_paper.pdf)]
    * Title: Fine-Grained Categorization by Alignments
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: E. Gavves, B. Fernando, C.G.M. Snoek, A.W.M. Smeulders, T. Tuytelaars
    * Abstract: The aim of this paper is fine-grained categorization without human interaction. Different from prior work, which relies on detectors for specific object parts, we propose to localize distinctive details by roughly aligning the objects using just the overall shape, since implicit to fine-grained categorization is the existence of a super-class shape shared among all classes. The alignments are then used to transfer part annotations from training images to test images (supervised alignment), or to blindly yet consistently segment the object in a number of regions (unsupervised alignment). We furthermore argue that in the distinction of finegrained sub-categories, classification-oriented encodings like Fisher vectors are better suited for describing localized information than popular matching oriented features like HOG. We evaluate the method on the CU-2011 Birds and Stanford Dogs fine-grained datasets, outperforming the state-of-the-art.

count=1
* Beyond Hard Negative Mining: Efficient Detector Learning via Block-Circulant Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Henriques_Beyond_Hard_Negative_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Henriques_Beyond_Hard_Negative_2013_ICCV_paper.pdf)]
    * Title: Beyond Hard Negative Mining: Efficient Detector Learning via Block-Circulant Decomposition
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Joao F. Henriques, Joao Carreira, Rui Caseiro, Jorge Batista
    * Abstract: Competitive sliding window detectors require vast training sets. Since a pool of natural images provides a nearly endless supply of negative samples, in the form of patches at different scales and locations, training with all the available data is considered impractical. A staple of current approaches is hard negative mining, a method of selecting relevant samples, which is nevertheless expensive. Given that samples at slightly different locations have overlapping support, there seems to be an enormous amount of duplicated work. It is natural, then, to ask whether these redundancies can be eliminated. In this paper, we show that the Gram matrix describing such data is block-circulant. We derive a transformation based on the Fourier transform that block-diagonalizes the Gram matrix, at once eliminating redundancies and partitioning the learning problem. This decomposition is valid for any dense features and several learning algorithms, and takes full advantage of modern parallel architectures. Surprisingly, it allows training with all the potential samples in sets of thousands of images. By considering the full set, we generate in a single shot the optimal solution, which is usually obtained only after several rounds of hard negative mining. We report speed gains on Caltech Pedestrians and INRIA Pedestrians of over an order of magnitude, allowing training on a desktop computer in a couple of minutes.

count=1
* Exploiting Reflection Change for Automatic Reflection Removal
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Li_Exploiting_Reflection_Change_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Li_Exploiting_Reflection_Change_2013_ICCV_paper.pdf)]
    * Title: Exploiting Reflection Change for Automatic Reflection Removal
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Yu Li, Michael S. Brown
    * Abstract: This paper introduces an automatic method for removing reflection interference when imaging a scene behind a glass surface. Our approach exploits the subtle changes in the reflection with respect to the background in a small set of images taken at slightly different view points. Key to this idea is the use of SIFT-flow to align the images such that a pixel-wise comparison can be made across the input set. Gradients with variation across the image set are assumed to belong to the reflected scenes while constant gradients are assumed to belong to the desired background scene. By correctly labelling gradients belonging to reflection or background, the background scene can be separated from the reflection interference. Unlike previous approaches that exploit motion, our approach does not make any assumptions regarding the background or reflected scenes' geometry, nor requires the reflection to be static. This makes our approach practical for use in casual imaging scenarios. Our approach is straight forward and produces good results compared with existing methods.

count=1
* Pose-Configurable Generic Tracking of Elongated Objects
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Wesierski_Pose-Configurable_Generic_Tracking_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Wesierski_Pose-Configurable_Generic_Tracking_2013_ICCV_paper.pdf)]
    * Title: Pose-Configurable Generic Tracking of Elongated Objects
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Daniel Wesierski, Patrick Horain
    * Abstract: Elongated objects have various shapes and can shift, rotate, change scale, and be rigid or deform by flexing, articulating, and vibrating, with examples as varied as a glass bottle, a robotic arm, a surgical suture, a finger pair, a tram, and a guitar string. This generally makes tracking of poses of elongated objects very challenging. We describe a unified, configurable framework for tracking the pose of elongated objects, which move in the image plane and extend over the image region. Our method strives for simplicity, versatility, and efficiency. The object is decomposed into a chained assembly of segments of multiple parts that are arranged under a hierarchy of tailored spatio-temporal constraints. In this hierarchy, segments can rescale independently while their elasticity is controlled with global orientations and local distances. While the trend in tracking is to design complex, structure-free algorithms that update object appearance online, we show that our tracker, with the novel but remarkably simple, structured organization of parts with constant appearance, reaches or improves state-of-the-art performance. Most importantly, our model can be easily configured to track exact pose of arbitrary, elongated objects in the image plane. The tracker can run up to 100 fps on a desktop PC, yet the computation time scales linearly with the number of object parts. To our knowledge, this is the first approach to generic tracking of elongated objects.

count=1
* Merging the Unmatchable: Stitching Visually Disconnected SfM Models
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Cohen_Merging_the_Unmatchable_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Cohen_Merging_the_Unmatchable_ICCV_2015_paper.pdf)]
    * Title: Merging the Unmatchable: Stitching Visually Disconnected SfM Models
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Andrea Cohen, Torsten Sattler, Marc Pollefeys
    * Abstract: Recent advances in Structure-from-Motion not only enable the reconstruction of large scale scenes, but are also able to detect ambiguous structures caused by repeating elements that might result in incorrect reconstructions. Yet, it is not always possible to fully reconstruct a scene. The images required to merge different sub-models might be missing or it might be impossible to acquire such images in the first place due to occlusions or the structure of the scene. The problem of aligning multiple reconstructions that do not have visual overlap is impossible to solve in general. An important variant of this problem is the case in which individual sides of a building can be reconstructed but not joined due to the missing visual overlap. In this paper, we present a combinatorial approach for solving this variant by automatically stitching multiple sides of a building together. Our approach exploits symmetries and semantic information to reason about the possible geometric relations between the individual models. We show that our approach is able to reconstruct complete building models where traditional SfM ends up with disconnected building sides.

count=1
* Cutting Edge: Soft Correspondences in Multimodal Scene Parsing
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Namin_Cutting_Edge_Soft_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Namin_Cutting_Edge_Soft_ICCV_2015_paper.pdf)]
    * Title: Cutting Edge: Soft Correspondences in Multimodal Scene Parsing
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Sarah Taghavi Namin, Mohammad Najafi, Mathieu Salzmann, Lars Petersson
    * Abstract: Exploiting multiple modalities for semantic scene parsing has been shown to improve accuracy over the single modality scenario. Existing methods, however, assume that corresponding regions in two modalities have the same label. In this paper, we address the problem of data misalignment and label inconsistencies, e.g., due to moving objects, in semantic labeling, which violate the assumption of existing techniques. To this end, we formulate multimodal semantic labeling as inference in a CRF, and introduce latent nodes to explicitly model inconsistencies between two domains. These latent nodes allow us not only to leverage information from both domains to improve their labeling, but also to cut the edges between inconsistent regions. To eliminate the need for hand tuning the parameters of our model, we propose to learn intra-domain and inter-domain potential functions from training data. We demonstrate the benefits of our approach on two publicly available datasets containing 2D imagery and 3D point clouds. Thanks to our latent nodes and our learning strategy, our method outperforms the state-of-the-art in both cases.

count=1
* PIEFA: Personalized Incremental and Ensemble Face Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Peng_PIEFA_Personalized_Incremental_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Peng_PIEFA_Personalized_Incremental_ICCV_2015_paper.pdf)]
    * Title: PIEFA: Personalized Incremental and Ensemble Face Alignment
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Xi Peng, Shaoting Zhang, Yu Yang, Dimitris N. Metaxas
    * Abstract: Face alignment, especially on real-time or large-scale sequential images, is a challenging task with broad applications. Both generic and joint alignment approaches have been proposed with varying degrees of success. However, many generic methods are heavily sensitive to initializations and usually rely on offline-trained static models, which limit their performance on sequential images with extensive variations. On the other hand, joint methods are restricted to offline applications, since they require all frames to conduct batch alignment. To address these limitations, we propose to exploit incremental learning for personalized ensemble alignment. We sample multiple initial shapes to achieve image congealing within one frame, which enables us to incrementally conduct ensemble alignment by group-sparse regularized rank minimization. At the same time, personalized modeling is obtained by subspace adaptation under the same incremental framework, while correction strategy is used to alleviate model drifting. Experimental results on multiple controlled and in-the-wild databases demonstrate the superior performance of our approach compared with state-of-the-arts in terms of fitting accuracy and efficiency.

count=1
* AgeNet: Deeply Learned Regressor and Classifier for Robust Apparent Age Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w11/html/Liu_AgeNet_Deeply_Learned_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w11/papers/Liu_AgeNet_Deeply_Learned_ICCV_2015_paper.pdf)]
    * Title: AgeNet: Deeply Learned Regressor and Classifier for Robust Apparent Age Estimation
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Xin Liu, Shaoxin Li, Meina Kan, Jie Zhang, Shuzhe Wu, Wenxian Liu, Hu Han, Shiguang Shan, Xilin Chen
    * Abstract: Apparent age estimation from face image has attracted more and more attentions as it is favorable in some real-world applications. In this work, we propose an end-to-end learning approach for robust apparent age estimation, named by us AgeNet. Specifically, we address the apparent age estimation problem by fusing two kinds of models, i.e., real-value based regression models and Gaussian label distribution based classification models. For both kind of models, large-scale deep convolutional neural network is adopted to learn informative age representations. Another key feature of the proposed AgeNet is that, to avoid the problem of over-fitting on small apparent age training set, we exploit a general-to-specific transfer learning scheme. Technically, the AgeNet is first pre-trained on a large-scale web-collected face dataset with identity label, and then it is fine-tuned on a large-scale real age dataset with noisy age label. Finally, it is fine-tuned on a small training set with apparent age label. The experimental results on the ChaLearn 2015 Apparent Age Competition demonstrate that our AgeNet achieves the state-of-the-art performance in apparent age estimation.

count=1
* Predicting Ball Ownership in Basketball From a Monocular View Using Only Player Trajectories
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w21/html/Wei_Predicting_Ball_Ownership_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w21/papers/Wei_Predicting_Ball_Ownership_ICCV_2015_paper.pdf)]
    * Title: Predicting Ball Ownership in Basketball From a Monocular View Using Only Player Trajectories
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Xinyu Wei, Long Sha, Patrick Lucey, Peter Carr, Sridha Sridharan, Iain Matthews
    * Abstract: Tracking objects like a basketball from a monocular view is challenging due to its small size, potential to move at high velocities as well as the high frequency of occlusion. However, humans with a deep knowledge of a game like basketball can predict with high accuracy the location of the ball even without seeing it due to the location and motion of nearby objects, as well as information of where it was last seen. Learning from tracking data is problematic however, due to the high variance in player locations. In this paper, we show that by simply ``permuting'' the multi-agent data we obtain a compact role-ordered feature which accurately predict the ball owner. We also show that our formulation can incorporate other information sources such as a vision-based ball detector to improve prediction accuracy.

count=1
* Surface Normals in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Chen_Surface_Normals_in_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Surface_Normals_in_ICCV_2017_paper.pdf)]
    * Title: Surface Normals in the Wild
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Weifeng Chen, Donglai Xiang, Jia Deng
    * Abstract: We study the problem of single-image depth estimation for images in the wild. We collect human annotated surface normals and use them to help train a neural network that directly predicts pixel-wise depth. We propose two novel loss functions for training with surface normal annotations. Experiments on NYU Depth, KITTI, and our own dataset demonstrate that our approach can significantly improve the quality of depth estimation in the wild.

count=1
* Weakly- and Self-Supervised Learning for Content-Aware Deep Image Retargeting
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Cho_Weakly-_and_Self-Supervised_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Cho_Weakly-_and_Self-Supervised_ICCV_2017_paper.pdf)]
    * Title: Weakly- and Self-Supervised Learning for Content-Aware Deep Image Retargeting
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Donghyeon Cho, Jinsun Park, Tae-Hyun Oh, Yu-Wing Tai, In So Kweon
    * Abstract: This paper proposes a weakly- and self-supervised deep convolutional neural network (WSSDCNN) for content-aware image retargeting. Our network takes a source image and a target aspect ratio, and then directly outputs a retargeted image. Retargeting is performed through a shift map, which is a pixel-wise mapping from the source to the target grid. Our method implicitly learns an attention map, which leads to a content-aware shift map for image retargeting. As a result, discriminative parts in an image are preserved, while background regions are adjusted seamlessly. In the training phase, pairs of an image and its image level annotation are used to compute content and structure losses. We demonstrate the effectiveness of our proposed method for a retargeting application with insightful analyses.

count=1
* HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_HydraPlus-Net_Attentive_Deep_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_HydraPlus-Net_Attentive_Deep_ICCV_2017_paper.pdf)]
    * Title: HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Xihui Liu, Haiyu Zhao, Maoqing Tian, Lu Sheng, Jing Shao, Shuai Yi, Junjie Yan, Xiaogang Wang
    * Abstract: Pedestrian analysis plays a vital role in intelligent video surveillance and is a key component for security-centric computer vision systems. Despite that the convolutional neural networks are remarkable in learning discriminative features from images, the learning of comprehensive features of pedestrians for fine-grained tasks remains an open problem. In this study, we propose a new attention-based deep neural network, named as HydraPlus-Net (HP-net), that multi-directionally feeds the multi-level attention maps to different feature layers. The attentive deep features learned from the proposed HP-net bring unique advantages: (1) the model is capable of capturing multiple attentions from low-level to semantic-level, and (2) it explores the multi-scale selectiveness of attentive features to enrich the final feature representations for a pedestrian image. We demonstrate the effectiveness and generality of the proposed HP-net for pedestrian analysis on two tasks, i.e. pedestrian attribute recognition and person re-identification. Intensive experimental results have been provided to prove that the HP-net outperforms the state-of-the-art methods on various datasets.

count=1
* Unsupervised Domain Adaptation for Face Recognition in Unlabeled Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Sohn_Unsupervised_Domain_Adaptation_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Sohn_Unsupervised_Domain_Adaptation_ICCV_2017_paper.pdf)]
    * Title: Unsupervised Domain Adaptation for Face Recognition in Unlabeled Videos
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Kihyuk Sohn, Sifei Liu, Guangyu Zhong, Xiang Yu, Ming-Hsuan Yang, Manmohan Chandraker
    * Abstract: Despite rapid advances in face recognition, there remains a clear gap between the performance of still image-based face recognition and video-based face recognition, due to the vast difference in visual quality between the domains and the difficulty of curating diverse large-scale video datasets. This paper addresses both of those challenges, through an image to video feature-level domain adaptation approach, to learn discriminative video frame representations. The framework utilizes large-scale unlabeled video data to reduce the gap between different domains while transferring discriminative knowledge from large-scale labeled still images. Given a face recognition network that is pretrained in the image domain, the adaptation is achieved by (i) distilling knowledge from the network to a video adaptation network through feature matching, (ii) performing feature restoration through synthetic data augmentation and (iii) learning a domain-invariant feature through a domain adversarial discriminator. We further improve performance through a discriminator-guided feature fusion that boosts high-quality frames while eliminating those degraded by video domain-specific factors. Experiments on the YouTube Faces and IJB-A datasets demonstrate that each module contributes to our feature-level domain adaptation framework and substantially improves video face recognition performance to achieve state-of-the-art accuracy. We demonstrate qualitatively that the network learns to suppress diverse artifacts in videos such as pose, illumination or occlusion without being explicitly trained for them.

count=1
* Deep Free-Form Deformation Network for Object-Mask Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_Deep_Free-Form_Deformation_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Deep_Free-Form_Deformation_ICCV_2017_paper.pdf)]
    * Title: Deep Free-Form Deformation Network for Object-Mask Registration
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Haoyang Zhang, Xuming He
    * Abstract: This paper addresses the problem of object-mask registration, which aligns a shape mask to a target object instance. Prior work typically formulate the problem as an object segmentation task with mask prior, which is challenging to solve. In this work, we take a transformation based approach that predicts a 2D non-rigid spatial transform and warps the shape mask onto the target object. In particular, we propose a deep spatial transformer network that learns free-form deformations (FFDs) to non-rigidly warp the shape mask based on a multi-level dual mask feature pooling strategy. The FFD transforms are based on B-splines and parameterized by the offsets of predefined control points, which are differentiable. Therefore, we are able to train the entire network in an end-to-end manner based on L2 matching loss. We evaluate our FFD network on a challenging object-mask alignment task, which aims to refine a set of object segment proposals, and our approach achieves the state-of-the-art performance on the Cityscapes, the PASCAL VOC and the MSCOCO datasets.

count=1
* Person Re-Identification by Deep Learning Multi-Scale Representations
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w37/html/Chen_Person_Re-Identification_by_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w37/Chen_Person_Re-Identification_by_ICCV_2017_paper.pdf)]
    * Title: Person Re-Identification by Deep Learning Multi-Scale Representations
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Yanbei Chen, Xiatian Zhu, Shaogang Gong
    * Abstract: Existing person re-identification (re-id) methods depend mostly on single-scale appearance information. This not only ignores the potentially useful explicit information of other different scales, but also loses the chance of mining the implicit correlated complementary advantages across scales. In this work, we formulate a novel Deep Pyramid Feature Learning (DPFL) CNN architecture for multi-scale appearance feature fusion optimised simultaneously by concurrent per-scale re-id losses and interactive cross-scale consensus regularisation in a closed-loop design. Extensive comparative evaluations demonstrate the re-id advantages of the proposed DPFL model over a wide range of state-of-the-art re-id methods on three benchmarks Market-1501, CUHK03, and DukeMTMC-reID.

count=1
* Learning to Detect Fine-Grained Change Under Variant Imaging Conditions
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w42/html/Huang_Learning_to_Detect_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w42/Huang_Learning_to_Detect_ICCV_2017_paper.pdf)]
    * Title: Learning to Detect Fine-Grained Change Under Variant Imaging Conditions
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Rui Huang, Wei Feng, Zezheng Wang, Mingyuan Fan, Liang Wan, Jizhou Sun
    * Abstract: Fine-grained change detection under variant imaging conditions is an important and challenging task for high-value scene monitoring in culture heritage. In this paper, we show that after a simple coarse alignment of lighting and camera differences, fine-grained change detection can be reliably solved by a deep network model, which is specifically composed of three functional parts, i.e., camera pose correction network (PCN), fine-grained change detection network (FCDN), and detection confidence boosting. Since our model is properly pre-trained and fine-tuned on both general and specialized data, it exhibits very good generalization capability to produce high-quality minute change detection on real-world scenes under varied imaging conditions. Extensive experiments validate the superior effectiveness and reliability over state-of-the-art methods. We have achieved 67.41% relative F1-measure improvement over the best competitor on real-world benchmark dataset.

count=1
* ABD-Net: Attentive but Diverse Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_ABD-Net_Attentive_but_Diverse_Person_Re-Identification_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_ABD-Net_Attentive_but_Diverse_Person_Re-Identification_ICCV_2019_paper.pdf)]
    * Title: ABD-Net: Attentive but Diverse Person Re-Identification
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Tianlong Chen,  Shaojin Ding,  Jingyi Xie,  Ye Yuan,  Wuyang Chen,  Yang Yang,  Zhou Ren,  Zhangyang Wang
    * Abstract: Attention mechanisms have been found effective for person re-identification (Re-ID). However, the learned "attentive" features are often not naturally uncorrelated or "diverse", which compromises the retrieval performance based on the Euclidean distance. We advocate the complementary powers of attention and diversity for Re-ID, by proposing an Attentive but Diverse Network (ABD-Net). ABD-Net seamlessly integrates attention modules and diversity regularizations throughout the entire network to learn features that are representative, robust, and more discriminative. Specifically, we introduce a pair of complementary attention modules, focusing on channel aggregation and position awareness, respectively. Then, we plug in a novel orthogonality constraint that efficiently enforces diversity on both hidden activations and weights. Through an extensive set of ablation study, we verify that the attentive and diverse terms each contributes to the performance boosts of ABD-Net. It consistently outperforms existing state-of-the-art methods on there popular person Re-ID benchmarks.

count=1
* IMP: Instance Mask Projection for High Accuracy Semantic Segmentation of Things
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Fu_IMP_Instance_Mask_Projection_for_High_Accuracy_Semantic_Segmentation_of_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_IMP_Instance_Mask_Projection_for_High_Accuracy_Semantic_Segmentation_of_ICCV_2019_paper.pdf)]
    * Title: IMP: Instance Mask Projection for High Accuracy Semantic Segmentation of Things
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Cheng-Yang Fu,  Tamara L. Berg,  Alexander C. Berg
    * Abstract: In this work, we present a new operator, called Instance Mask Projection (IMP), which projects a predicted instance segmentation as a new feature for semantic segmentation. It also supports back propagation and is trainable end-to end. By adding this operator, we introduce a new way to combine top-down and bottom-up information in semantic segmentation. Our experiments show the effectiveness of IMP on both clothing parsing (with complex layering, large deformations, and non-convex objects), and on street scene segmentation (with many overlapping instances and small objects). On the Varied Clothing Parsing dataset (VCP), we show instance mask projection can improve mIOU by 3 points over a state-of-the-art Panoptic FPN segmentation approach. On the ModaNet clothing parsing dataset, we show a dramatic improvement of 20.4% compared to existing baseline semantic segmentation results. In addition, the Instance Mask Projection operator works well on other (non-clothing) datasets, providing an improvement in mIOU of 3 points on "thing" classes of Cityscapes, a self-driving dataset, over a state-of-the-art approach.

count=1
* Learning Local RGB-to-CAD Correspondences for Object Pose Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Georgakis_Learning_Local_RGB-to-CAD_Correspondences_for_Object_Pose_Estimation_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Georgakis_Learning_Local_RGB-to-CAD_Correspondences_for_Object_Pose_Estimation_ICCV_2019_paper.pdf)]
    * Title: Learning Local RGB-to-CAD Correspondences for Object Pose Estimation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Georgios Georgakis,  Srikrishna Karanam,  Ziyan Wu,  Jana Kosecka
    * Abstract: We consider the problem of 3D object pose estimation. While much recent work has focused on the RGB domain, the reliance on accurately annotated images limits generalizability and scalability. On the other hand, the easily available object CAD models are rich sources of data, providing a large number of synthetically rendered images. In this paper, we solve this key problem of existing methods requiring expensive 3D pose annotations by proposing a new method that matches RGB images to CAD models for object pose estimation. Our key innovations compared to existing work include removing the need for either real-world textures for CAD models or explicit 3D pose annotations for RGB images. We achieve this through a series of objectives that learn how to select keypoints and enforce viewpoint and modality invariance across RGB images and CAD model renderings. Our experiments demonstrate that the proposed method can reliably estimate object pose in RGB images and generalize to object instances not seen during training.

count=1
* Video Object Segmentation Using Space-Time Memory Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.pdf)]
    * Title: Video Object Segmentation Using Space-Time Memory Networks
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Seoung Wug Oh,  Joon-Young Lee,  Ning Xu,  Seon Joo Kim
    * Abstract: We propose a novel solution for semi-supervised video object segmentation. By the nature of the problem, available cues (e.g. video frame(s) with object masks) become richer with the intermediate predictions. However, the existing methods are unable to fully exploit this rich source of information. We resolve the issue by leveraging memory networks and learn to read relevant information from all available sources. In our framework, the past frames with object masks form an external memory, and the current frame as the query is segmented using the mask information in the memory. Specifically, the query and the memory are densely matched in the feature space, covering all the space-time pixel locations in a feed-forward fashion. Contrast to the previous approaches, the abundant use of the guidance information allows us to better handle the challenges such as appearance changes and occlussions. We validate our method on the latest benchmark sets and achieved the state-of-the-art performance (overall score of 79.4 on Youtube-VOS val set, J of 88.7 and 79.2 on DAVIS 2016/2017 val set respectively) while having a fast runtime (0.16 second/frame on DAVIS 2016 val set).

count=1
* Robust Change Captioning
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Robust_Change_Captioning_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Robust_Change_Captioning_ICCV_2019_paper.pdf)]
    * Title: Robust Change Captioning
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Dong Huk Park,  Trevor Darrell,  Anna Rohrbach
    * Abstract: Describing what has changed in a scene can be useful to a user, but only if generated text focuses on what is semantically relevant. It is thus important to distinguish distractors (e.g. a viewpoint change) from relevant changes (e.g. an object has moved). We present a novel Dual Dynamic Attention Model (DUDA) to perform robust Change Captioning. Our model learns to distinguish distractors from semantic changes, localize the changes via Dual Attention over "before" and "after" images, and accurately describe them in natural language via Dynamic Speaker, by adaptively focusing on the necessary visual inputs (e.g. "before" or "after" image). To study the problem in depth, we collect a CLEVR-Change dataset, built off the CLEVR engine, with 5 types of scene changes. We benchmark a number of baselines on our dataset, and systematically study different change types and robustness to distractors. We show the superiority of our DUDA model in terms of both change captioning and localization. We also show that our approach is general, obtaining state-of-the-art results on the recent realistic Spot-the-Diff dataset which has no distractors.

count=1
* CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_CAMP_Cross-Modal_Adaptive_Message_Passing_for_Text-Image_Retrieval_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_CAMP_Cross-Modal_Adaptive_Message_Passing_for_Text-Image_Retrieval_ICCV_2019_paper.pdf)]
    * Title: CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Zihao Wang,  Xihui Liu,  Hongsheng Li,  Lu Sheng,  Junjie Yan,  Xiaogang Wang,  Jing Shao
    * Abstract: Text-image cross-modal retrieval is a challenging task in the field of language and vision. Most previous approaches independently embed images and sentences into a joint embedding space and compare their similarities. However, previous approaches rarely explore the interactions between images and sentences before calculating similarities in the joint space. Intuitively, when matching between images and sentences, human beings would alternatively attend to regions in images and words in sentences, and select the most salient information considering the interaction between both modalities. In this paper, we propose Cross-modal Adaptive Message Passing (CAMP), which adaptively controls the information flow for message passing across modalities. Our approach not only takes comprehensive and fine-grained cross-modal interactions into account, but also properly handles negative pairs and irrelevant information with an adaptive gating scheme. Moreover, instead of conventional joint embedding approaches for text-image matching, we infer the matching score based on the fused features, and propose a hardest negative binary cross-entropy loss for training. Results on COCO and Flickr30k significantly surpass state-of-the-art methods, demonstrating the effectiveness of our approach.

count=1
* Second-Order Non-Local Attention Networks for Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Xia_Second-Order_Non-Local_Attention_Networks_for_Person_Re-Identification_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xia_Second-Order_Non-Local_Attention_Networks_for_Person_Re-Identification_ICCV_2019_paper.pdf)]
    * Title: Second-Order Non-Local Attention Networks for Person Re-Identification
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Bryan (Ning) Xia,  Yuan Gong,  Yizhe Zhang,  Christian Poellabauer
    * Abstract: Recent efforts have shown promising results for person re-identification by designing part-based architectures to allow a neural network to learn discriminative representations from semantically coherent parts. Some efforts use soft attention to reallocate distant outliers to their most similar parts, while others adjust part granularity to incorporate more distant positions for learning the relationships. Others seek to generalize part-based methods by introducing a dropout mechanism on consecutive regions of the feature map to enhance distant region relationships. However, only few prior efforts model the distant or non-local positions of the feature map directly for the person re-ID task. In this paper, we propose a novel attention mechanism to directly model long-range relationships via second-order feature statistics. When combined with a generalized DropBlock module, our method performs equally to or better than state-of-the-art results for mainstream person re-identification datasets, including Market1501, CUHK03, and DukeMTMC-reID.

count=1
* End-to-End Hand Mesh Recovery From a Monocular RGB Image
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_End-to-End_Hand_Mesh_Recovery_From_a_Monocular_RGB_Image_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_End-to-End_Hand_Mesh_Recovery_From_a_Monocular_RGB_Image_ICCV_2019_paper.pdf)]
    * Title: End-to-End Hand Mesh Recovery From a Monocular RGB Image
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Xiong Zhang,  Qiang Li,  Hong Mo,  Wenbo Zhang,  Wen Zheng
    * Abstract: In this paper, we present a HAnd Mesh Recovery (HAMR) framework to tackle the problem of reconstructing the full 3D mesh of a human hand from a single RGB image. In contrast to existing research on 2D or 3D hand pose estimation from RGB or/and depth image data, HAMR can provide a more expressive and useful mesh representation for monocular hand image understanding. In particular, the mesh representation is achieved by parameterizing a generic 3D hand model with shape and relative 3D joint angles. By utilizing this mesh representation, we can easily compute the 3D joint locations via linear interpolations between the vertexes of the mesh, while obtain the 2D joint locations with a projection of the 3D joints. To this end, a differentiable re-projection loss can be defined in terms of the derived representations and the ground-truth labels, thus making our framework end-to-end trainable. Qualitative experiments show that our framework is capable of recovering appealing 3D hand mesh even in the presence of severe occlusions. Quantitatively, our approach also outperforms the state-of-the-art methods for both 2D and 3D hand pose estimation from a monocular RGB image on several benchmark datasets.

count=1
* Towards Adversarially Robust Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Towards_Adversarially_Robust_Object_Detection_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Towards_Adversarially_Robust_Object_Detection_ICCV_2019_paper.pdf)]
    * Title: Towards Adversarially Robust Object Detection
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Haichao Zhang,  Jianyu Wang
    * Abstract: Object detection is an important vision task and has emerged as an indispensable component in many vision system, rendering its robustness as an increasingly important performance factor for practical applications. While object detection models have been demonstrated to be vulnerable against adversarial attacks by many recent works, very few efforts have been devoted to improving their robustness. In this work, we take an initial attempt towards this direction. We first revisit and systematically analyze object detectors and many recently developed attacks from the perspective of model robustness. We then present a multi-task learning perspective of object detection and identify an asymmetric role of task losses. We further develop an adversarial training approach which can leverage the multiple sources of attacks for improving the robustness of detection models. Extensive experiments on PASCAL-VOC and MS-COCO verified the effectiveness of the proposed approach.

count=1
* FashionMirror: Co-Attention Feature-Remapping Virtual Try-On With Sequential Template Poses
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_FashionMirror_Co-Attention_Feature-Remapping_Virtual_Try-On_With_Sequential_Template_Poses_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_FashionMirror_Co-Attention_Feature-Remapping_Virtual_Try-On_With_Sequential_Template_Poses_ICCV_2021_paper.pdf)]
    * Title: FashionMirror: Co-Attention Feature-Remapping Virtual Try-On With Sequential Template Poses
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Chieh-Yun Chen, Ling Lo, Pin-Jui Huang, Hong-Han Shuai, Wen-Huang Cheng
    * Abstract: Virtual try-on tasks have drawn increased attention. Prior arts focus on tackling this task via warping clothes and fusing the information at the pixel level with the help of semantic segmentation. However, conducting semantic segmentation is time-consuming and easily causes error accumulation over time. Besides, warping the information at the pixel level instead of the feature level limits the performance (e.g., unable to generate different views) and is unstable since it directly demonstrates the results even with a misalignment. In contrast, fusing information at the feature level can be further refined by the convolution to obtain the final results. Based on these assumptions, we propose a co-attention feature-remapping framework, namely FashionMirror, that generates the try-on results according to the driven-pose sequence in two stages. In the first stage, we consider the source human image and the target try-on clothes to predict the removed mask and the try-on clothing mask, which replaces the pre-processed semantic segmentation and reduces the inference time. In the second stage, we first remove the clothes on the source human via the removed mask and warp the clothing features conditioning on the try-on clothing mask to fit the next frame human. Meanwhile, we predict the optical flows from the consecutive 2D poses and warp the source human to the next frame at the feature level. Then, we enhance the clothing features and source human features in every frame to generate realistic try-on results with spatio-temporal smoothness. Both qualitative and quantitative results show that FashionMirror outperforms the state-of-the-art virtual try-on approaches.

count=1
* Generative Adversarial Registration for Improved Conditional Deformable Templates
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Dey_Generative_Adversarial_Registration_for_Improved_Conditional_Deformable_Templates_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Dey_Generative_Adversarial_Registration_for_Improved_Conditional_Deformable_Templates_ICCV_2021_paper.pdf)]
    * Title: Generative Adversarial Registration for Improved Conditional Deformable Templates
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Neel Dey, Mengwei Ren, Adrian V. Dalca, Guido Gerig
    * Abstract: Deformable templates are essential to large-scale medical image registration, segmentation, and population analysis. Current conventional and deep network-based methods for template construction use only regularized registration objectives and often yield templates with blurry and/or anatomically implausible appearance, confounding downstream biomedical interpretation. We reformulate deformable registration and conditional template estimation as an adversarial game wherein we encourage realism in the moved templates with a generative adversarial registration framework conditioned on flexible image covariates. The resulting templates exhibit significant gain in specificity to attributes such as age and disease, better fit underlying group-wise spatiotemporal trends, and achieve improved sharpness and centrality. These improvements enable more accurate population modeling with diverse covariates for standardized downstream analyses and easier anatomical delineation for structures of interest.

count=1
* Location-Aware Single Image Reflection Removal
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Dong_Location-Aware_Single_Image_Reflection_Removal_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Dong_Location-Aware_Single_Image_Reflection_Removal_ICCV_2021_paper.pdf)]
    * Title: Location-Aware Single Image Reflection Removal
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zheng Dong, Ke Xu, Yin Yang, Hujun Bao, Weiwei Xu, Rynson W.H. Lau
    * Abstract: This paper proposes a novel location-aware deep-learning-based single image reflection removal method. Our network has a reflection detection module to regress a probabilistic reflection confidence map, taking multi-scale Laplacian features as inputs. This probabilistic map tells if a region is reflection-dominated or transmission-dominated, and it is used as a cue for the network to control the feature flow when predicting the reflection and transmission layers. We design our network as a recurrent network to progressively refine reflection removal results at each iteration. The novelty is that we leverage Laplacian kernel parameters to emphasize the boundaries of strong reflections. It is beneficial to strong reflection detection and substantially improves the quality of reflection removal results. Extensive experiments verify the superior performance of the proposed method over state-of-the-art approaches. Our code and the pre-trained model can be found at https://github.com/zdlarr/Location-aware-SIRR.

count=1
* TOOD: Task-Aligned One-Stage Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.pdf)]
    * Title: TOOD: Task-Aligned One-Stage Object Detection
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R. Scott, Weilin Huang
    * Abstract: One-stage object detection is commonly implemented by optimizing two sub-tasks: object classification and localization, using heads with two parallel branches, which might lead to a certain level of spatial misalignment in predictions between the two tasks. In this work, we propose a Task-aligned One-stage Object Detection (TOOD) that explicitly aligns the two tasks in a learning-based manner. First, we design a novel Task-aligned Head (T-Head) which offers a better balance between learning task-interactive and task-specific features, as well as a greater flexibility to learn the alignment via a task-aligned predictor. Second, we propose Task Alignment Learning (TAL) to explicitly pull closer (or even unify) the optimal anchors for the two tasks during training via a designed sample assignment scheme and a task-aligned loss. Extensive experiments are conducted on MS-COCO, where TOOD achieves a 51.1 AP at single-model single-scale testing. This surpasses the recent one-stage detectors by a large margin, such as ATSS (47.7 AP), GFL (48.2 AP), and PAA (49.0 AP), with fewer parameters and FLOPs. Qualitative results also demonstrate the effectiveness of TOOD for better aligning the tasks of object classification and localization. Code is available at https://github.com/fcjian/TOOD.

count=1
* STRIVE: Scene Text Replacement in Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/G_STRIVE_Scene_Text_Replacement_in_Videos_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/G_STRIVE_Scene_Text_Replacement_in_Videos_ICCV_2021_paper.pdf)]
    * Title: STRIVE: Scene Text Replacement in Videos
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Vijay Kumar B G, Jeyasri Subramanian, Varnith Chordia, Eugene Bart, Shaobo Fang, Kelly Guan, Raja Bala
    * Abstract: We propose replacing scene text in videos using deep style transfer and learned photometric transformations. Building on recent progress on still image text replacement, we present extensions that alter text while preserving the appearance and motion characteristics of the original video. Compared to the problem of still image text replacement, our method addresses additional challenges introduced by video, namely effects induced by changing lighting, motion blur, diverse variations in camera-object pose over time, and preservation of temporal consistency. We parse the problem into three steps. First, the text in all frames is normalized to a frontal pose using a spatio-temporal transformer network. Second, the text is replaced in a single reference frame using a state-of-art still-image text replacement method. Finally, the new text is transferred from the reference to remaining frames using a novel learned image transformation network that captures lighting and blur effects in a temporally consistent manner. Results on synthetic and challenging real videos show realistic text transfer, competitive quantitative and qualitative performance, and superior inference speed relative to alternatives. We introduce new synthetic and real-world datasets with paired text objects. To the best of our knowledge this is the first attempt at deep video text replacement.

count=1
* Video Object Segmentation With Dynamic Memory Networks and Adaptive Object Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Liang_Video_Object_Segmentation_With_Dynamic_Memory_Networks_and_Adaptive_Object_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Video_Object_Segmentation_With_Dynamic_Memory_Networks_and_Adaptive_Object_ICCV_2021_paper.pdf)]
    * Title: Video Object Segmentation With Dynamic Memory Networks and Adaptive Object Alignment
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Shuxian Liang, Xu Shen, Jianqiang Huang, Xian-Sheng Hua
    * Abstract: In this paper, we propose a novel solution for object-matching based semi-supervised video object segmentation, where the target object masks in the first frame are provided. Existing object-matching based methods focus on the matching between the raw object features of the current frame and the first/previous frames. However, two issues are still not solved by these object-matching based methods. As the appearance of the video object changes drastically over time, 1) unseen parts/details of the object present in the current frame, resulting in incomplete annotation in the first annotated frame (e.g., view/scale changes). 2) even for the seen parts/details of the object in the current frame, their positions change relatively (e.g., pose changes/camera motion), leading to a misalignment for the object matching. To obtain the complete information of the target object, we propose a novel object-based dynamic memory network that exploits visual contents of all the past frames. To solve the misalignment problem caused by position changes of visual contents, we propose an adaptive object alignment module by incorporating a region translation function that aligns object proposals towards templates in the feature space. Our method achieves state-of-the-art results on latest benchmark datasets DAVIS 2017 (J of 81.4% and F of 87.5% on the validation set) and YouTube-VOS (the overall score of 82.7% on the validation set) with a very efficient inference time (0.16 second/frame on DAVIS 2017 validation set). Code is available at: https://github.com/liang4sx/DMN-AOA.

count=1
* COMISR: Compression-Informed Video Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Li_COMISR_Compression-Informed_Video_Super-Resolution_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_COMISR_Compression-Informed_Video_Super-Resolution_ICCV_2021_paper.pdf)]
    * Title: COMISR: Compression-Informed Video Super-Resolution
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yinxiao Li, Pengchong Jin, Feng Yang, Ce Liu, Ming-Hsuan Yang, Peyman Milanfar
    * Abstract: Most video super-resolution methods focus on restoring high-resolution video frames from low-resolution videos without taking into account compression. However, most videos on the web or mobile devices are compressed, and the compression can be severe when the bandwidth is limited. In this paper, we propose a new compression-informed video super-resolution model to restore high-resolution content without introducing artifacts caused by compression. The proposed model consists of three modules for video super-resolution: bi-directional recurrent warping, detail-preserving flow estimation, and Laplacian enhancement. All these three modules are used to deal with compression properties such as the location of the intra-frames in the input and smoothness in the output frames. For thorough performance evaluation, we conducted extensive experiments on standard datasets with a wide range of compression rates, covering many real video use cases. We showed that our method not only recovers high-resolution content on uncompressed frames from the widely-used benchmark datasets, but also achieves state-of-the-art performance in super-resolving compressed videos based on numerous quantitative metrics. We also evaluated the proposed method by simulating streaming from YouTube to demonstrate its effectiveness and robustness. The source codes and trained models are available at https://github.com/google-research/google-research/tree/master/comisr.

count=1
* Semantic Concentration for Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Semantic_Concentration_for_Domain_Adaptation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Semantic_Concentration_for_Domain_Adaptation_ICCV_2021_paper.pdf)]
    * Title: Semantic Concentration for Domain Adaptation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Shuang Li, Mixue Xie, Fangrui Lv, Chi Harold Liu, Jian Liang, Chen Qin, Wei Li
    * Abstract: Domain adaptation (DA) paves the way for label annotation and dataset bias issues by the knowledge transfer from a label-rich source domain to a related but unlabeled target domain. A mainstream of DA methods is to align the feature distributions of the two domains. However, the majority of them focus on the entire image features where irrelevant semantic information, e.g., the messy background, is inevitably embedded. Enforcing feature alignments in such case will negatively influence the correct matching of objects and consequently lead to the semantically negative transfer due to the confusion of irrelevant semantics. To tackle this issue, we propose Semantic Concentration for Domain Adaptation (SCDA), which encourages the model to concentrate on the most principal features via the pair-wise adversarial alignment of prediction distributions. Specifically, we train the classifier to class-wisely maximize the prediction distribution divergence of each sample pair, which enables the model to find the region with large differences among the same class of samples. Meanwhile, the feature extractor attempts to minimize that discrepancy, which suppresses the features of dissimilar regions among the same class of samples and accentuates the features of principal parts. As a general method, SCDA can be easily integrated into various DA methods as a regularizer to further boost their performance. Extensive experiments on the cross-domain benchmarks show the efficacy of SCDA.

count=1
* BabelCalib: A Universal Approach to Calibrating Central Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Lochman_BabelCalib_A_Universal_Approach_to_Calibrating_Central_Cameras_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Lochman_BabelCalib_A_Universal_Approach_to_Calibrating_Central_Cameras_ICCV_2021_paper.pdf)]
    * Title: BabelCalib: A Universal Approach to Calibrating Central Cameras
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yaroslava Lochman, Kostiantyn Liepieshov, Jianhui Chen, Michal Perdoch, Christopher Zach, James Pritts
    * Abstract: Existing calibration methods occasionally fail for large field-of-view cameras due to the non-linearity of the underlying problem and the lack of good initial values for all parameters of the used camera model. This might occur because a simpler projection model is assumed in an initial step, or a poor initial guess for the internal parameters is pre-defined. A lot of the difficulties of general camera calibration lie in the use of a forward projection model. We side-step these challenges by first proposing a solver to calibrate the parameters in terms of a back-projection model and then regress the parameters for a target forward model. These steps are incorporated in a robust estimation framework to cope with outlying detections. Extensive experiments demonstrate that our approach is very reliable and returns the most accurate calibration parameters as measured on the downstream task of absolute pose estimation on test sets. The code is released at https://github.com/ylochman/babelcalib

count=1
* Unsupervised Domain Adaptive 3D Detection With Multi-Level Consistency
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Luo_Unsupervised_Domain_Adaptive_3D_Detection_With_Multi-Level_Consistency_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Luo_Unsupervised_Domain_Adaptive_3D_Detection_With_Multi-Level_Consistency_ICCV_2021_paper.pdf)]
    * Title: Unsupervised Domain Adaptive 3D Detection With Multi-Level Consistency
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zhipeng Luo, Zhongang Cai, Changqing Zhou, Gongjie Zhang, Haiyu Zhao, Shuai Yi, Shijian Lu, Hongsheng Li, Shanghang Zhang, Ziwei Liu
    * Abstract: Deep learning-based 3D object detection has achieved unprecedented success with the advent of large-scale autonomous driving datasets. However, drastic performance degradation remains a critical challenge for cross-domain deployment. In addition, existing 3D domain adaptive detection methods often assume prior access to the target domain annotations, which is rarely feasible in the real world. To address this challenge, we study a more realistic setting, unsupervised 3D domain adaptive detection, which only utilizes source domain annotations. 1) We first comprehensively investigate the major underlying factors of the domain gap in 3D detection. Our key insight is that geometric mismatch is the key factor of domain shift. 2) Then, we propose a novel and unified framework, Multi-Level Consistency Network (MLC-Net), which employs a teacher-student paradigm to generate adaptive and reliable pseudo-targets. MLC-Net exploits point-, instance- and neural statistics-level consistency to facilitate cross-domain transfer. Extensive experiments demonstrate that MLC-Net outperforms existing state-of-the-art methods (including those using additional target domain information) on standard benchmarks. Notably, our approach is detector-agnostic, which achieves consistent gains on both single- and two-stage 3D detectors. Code will be released.

count=1
* SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Prabhu_SENTRY_Selective_Entropy_Optimization_via_Committee_Consistency_for_Unsupervised_Domain_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Prabhu_SENTRY_Selective_Entropy_Optimization_via_Committee_Consistency_for_Unsupervised_Domain_ICCV_2021_paper.pdf)]
    * Title: SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Viraj Prabhu, Shivam Khare, Deeksha Kartik, Judy Hoffman
    * Abstract: Many existing approaches for unsupervised domain adaptation (UDA) focus on adapting under only data distribution shift and offer limited success under additional cross-domain label distribution shift. Recent work based on self-training using target pseudolabels has shown promise, but on challenging shifts pseudolabels may be highly unreliable and using them for self-training may lead to error accumulation and domain misalignment. We propose Selective Entropy Optimization via Committee Consistency (SENTRY), a UDA algorithm that judges the reliability of a target instance based on its predictive consistency under a committee of random image transformations. Our algorithm then selectively minimizes predictive entropy to increase confidence on highly consistent target instances, while maximizing predictive entropy to reduce confidence on highly inconsistent ones. In combination with pseudolabel-based approximate target class balancing, our approach leads to significant improvements over the state-of-the-art on 27/31 domain shifts from standard UDA benchmarks as well as benchmarks designed to stress-test adaptation under label distribution shift.

count=1
* V-DESIRR: Very Fast Deep Embedded Single Image Reflection Removal
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Prasad_V-DESIRR_Very_Fast_Deep_Embedded_Single_Image_Reflection_Removal_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Prasad_V-DESIRR_Very_Fast_Deep_Embedded_Single_Image_Reflection_Removal_ICCV_2021_paper.pdf)]
    * Title: V-DESIRR: Very Fast Deep Embedded Single Image Reflection Removal
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: B H Pawan Prasad, Green Rosh K S, Lokesh R. Boregowda, Kaushik Mitra, Sanjoy Chowdhury
    * Abstract: Real world images often gets corrupted due to unwanted reflections and their removal is highly desirable. A major share of such images originate from smart phone cameras capable of very high resolution captures. Most of the existing methods either focus on restoration quality by compromising on processing speed and memory requirements or, focus on removing reflections at very low resolutions, there by limiting their practical deploy-ability. We propose a light weight deep learning model for reflection removal using a novel scale space architecture. Our method processes the corrupted image in two stages, a Low Scale Sub-network (LSSNet) to process the lowest scale and a Progressive Inference (PI) stage to process all the higher scales. In order to reduce the computational complexity, the sub-networks in PI stage are designed to be much shallower than LSSNet. Moreover, we employ weight sharing between various scales within the PI stage to limit the model size. This also allows our method to generalize to very high resolutions without explicit retraining. Our method is superior both qualitatively and quantitatively compared to the state of the art methods and at the same time 20x faster with 50x less number of parameters compared to the most recent state-of-the-art algorithm RAGNet. We implemented our method on an android smart phone, where a high resolution 12 MP image is restored in under 5 seconds.

count=1
* Partial Video Domain Adaptation With Partial Adversarial Temporal Attentive Network
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Partial_Video_Domain_Adaptation_With_Partial_Adversarial_Temporal_Attentive_Network_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Partial_Video_Domain_Adaptation_With_Partial_Adversarial_Temporal_Attentive_Network_ICCV_2021_paper.pdf)]
    * Title: Partial Video Domain Adaptation With Partial Adversarial Temporal Attentive Network
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yuecong Xu, Jianfei Yang, Haozhi Cao, Zhenghua Chen, Qi Li, Kezhi Mao
    * Abstract: Partial Domain Adaptation (PDA) is a practical and general domain adaptation scenario, which relaxes the fully shared label space assumption such that the source label space subsumes the target one. The key challenge of PDA is the issue of negative transfer caused by source-only classes. For videos, such negative transfer could be triggered by both spatial and temporal features, which leads to a more challenging Partial Video Domain Adaptation (PVDA) problem. In this paper, we propose a novel Partial Adversarial Temporal Attentive Network (PATAN) to address the PVDA problem by utilizing both spatial and temporal features for filtering source-only classes. Besides, PATAN constructs effective overall temporal features by attending to local temporal features that contribute more toward the class filtration process. We further introduce new benchmarks to facilitate research on PVDA problems, covering a wide range of PVDA scenarios. Empirical results demonstrate the state-of-the-art performance of our proposed PATAN across the multiple PVDA benchmarks.

count=1
* Structure-Transformed Texture-Enhanced Network for Person Image Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Structure-Transformed_Texture-Enhanced_Network_for_Person_Image_Synthesis_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Structure-Transformed_Texture-Enhanced_Network_for_Person_Image_Synthesis_ICCV_2021_paper.pdf)]
    * Title: Structure-Transformed Texture-Enhanced Network for Person Image Synthesis
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Munan Xu, Yuanqi Chen, Shan Liu, Thomas H. Li, Ge Li
    * Abstract: Pose-guided virtual try-on task aims to modify the fashion item based on pose transfer task. These two tasks that belong to person image synthesis have strong correlations and similarities. However, existing methods treat them as two individual tasks and do not explore correlations between them. Moreover, these two tasks are challenging due to large misalignment and occlusions, thus most of these methods are prone to generate unclear human body structure and blurry fine-grained textures. In this paper, we devise a structure-transformed texture-enhanced network to generate high-quality person images and construct the relationships between two tasks. It consists of two modules: structure-transformed renderer and texture-enhanced stylizer. The structure-transformed renderer is introduced to transform the source person structure to the target one, while the texture-enhanced stylizer is served to enhance detailed textures and controllably inject the fashion style founded on the structural transformation. With the two modules, our model can generate photorealistic person images in diverse poses and even with various fashion styles. Extensive experiments demonstrate that our approach achieves state-of-the-art results on two tasks.

count=1
* Real-World Video Super-Resolution: A Benchmark Dataset and a Decomposition Based Learning Scheme
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Real-World_Video_Super-Resolution_A_Benchmark_Dataset_and_a_Decomposition_Based_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Real-World_Video_Super-Resolution_A_Benchmark_Dataset_and_a_Decomposition_Based_ICCV_2021_paper.pdf)]
    * Title: Real-World Video Super-Resolution: A Benchmark Dataset and a Decomposition Based Learning Scheme
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Xi Yang, Wangmeng Xiang, Hui Zeng, Lei Zhang
    * Abstract: Video super-resolution (VSR) aims to improve the spatial resolution of low-resolution (LR) videos. Existing VSR methods are mostly trained and evaluated on synthetic datasets, where the LR videos are uniformly downsampled from their high-resolution (HR) counterparts by some simple operators (e.g., bicubic downsampling). Such simple synthetic degradation models, however, cannot well describe the complex degradation processes in real-world videos, and thus the trained VSR models become ineffective in real-world applications. As an attempt to bridge the gap, we build a real-world video super-resolution (RealVSR) dataset by capturing paired LR-HR video sequences using the multi-camera system of iPhone 11 Pro Max. Since the LR-HR video pairs are captured by two separate cameras, there are inevitably certain misalignment and luminance/color differences between them. To more robustly train the VSR model and recover more details from the LR inputs, we convert the LR-HR videos into YCbCr space and decompose the luminance channel into a Laplacian pyramid, and then apply different loss functions to different components. Experiments validate that VSR models trained on our RealVSR dataset demonstrate better visual quality than those trained on synthetic datasets under real-world settings. They also exhibit good generalization capability in cross-camera tests. The dataset and code can be found at https://github.com/IanYeung/RealVSR.

count=1
* Auxiliary Tasks and Exploration Enable ObjectGoal Navigation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Ye_Auxiliary_Tasks_and_Exploration_Enable_ObjectGoal_Navigation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Auxiliary_Tasks_and_Exploration_Enable_ObjectGoal_Navigation_ICCV_2021_paper.pdf)]
    * Title: Auxiliary Tasks and Exploration Enable ObjectGoal Navigation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Joel Ye, Dhruv Batra, Abhishek Das, Erik Wijmans
    * Abstract: ObjectGoal Navigation (ObjectNav) is an embodied task wherein agents are to navigate to an object instance in an unseen environment. Prior works have shown that end-to-end ObjectNav agents that use vanilla visual and recurrent modules, e.g. a CNN+RNN, perform poorly due to overfitting and sample inefficiency. This has motivated current state-of-the-art methods to mix analytic and learned components and operate on explicit spatial maps of the environment. We instead re-enable a generic learned agent by adding auxiliary learning tasks and an exploration reward. Our agents achieve 24.5% success and 8.1% SPL, a 37% and 8% relative improvement over prior state-of-the-art, respectively, on the Habitat ObjectNav Challenge. From our analysis, we propose that agents will act to simplify their visual inputs so as to smooth their RNN dynamics, and that auxiliary tasks reduce overfitting by minimizing effective RNN dimensionality; i.e. a performant ObjectNav agent that must maintain coherent plans over long horizons does so by learning smooth, low-dimensional recurrent dynamics.

count=1
* Training Multi-Object Detector by Estimating Bounding Box Distribution for Input Image
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Yoo_Training_Multi-Object_Detector_by_Estimating_Bounding_Box_Distribution_for_Input_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Yoo_Training_Multi-Object_Detector_by_Estimating_Bounding_Box_Distribution_for_Input_ICCV_2021_paper.pdf)]
    * Title: Training Multi-Object Detector by Estimating Bounding Box Distribution for Input Image
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jaeyoung Yoo, Hojun Lee, Inseop Chung, Geonseok Seo, Nojun Kwak
    * Abstract: In multi-object detection using neural networks, the fundamental problem is, "How should the network learn a variable number of bounding boxes in different input images?". Previous methods train a multi-object detection network through a procedure that directly assigns the ground truth bounding boxes to the specific locations of the network's output. However, this procedure makes the training of a multi-object detection network too heuristic and complicated. In this paper, we reformulate the multi-object detection task as a problem of density estimation of bounding boxes. Instead of assigning each ground truth to specific locations of network's output, we train a network by estimating the probability density of bounding boxes in an input image using a mixture model. For this purpose, we propose a novel network for object detection called Mixture Density Object Detector (MDOD), and the corresponding objective function for the density-estimation-based training. We applied MDOD to MS COCO dataset. Our proposed method not only deals with multi-object detection problems in a new approach, but also improves detection performances through MDOD. The code is available: https://github.com/yoojy31/MDOD.

count=1
* WaveFill: A Wavelet-Based Generation Network for Image Inpainting
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_WaveFill_A_Wavelet-Based_Generation_Network_for_Image_Inpainting_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_WaveFill_A_Wavelet-Based_Generation_Network_for_Image_Inpainting_ICCV_2021_paper.pdf)]
    * Title: WaveFill: A Wavelet-Based Generation Network for Image Inpainting
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yingchen Yu, Fangneng Zhan, Shijian Lu, Jianxiong Pan, Feiying Ma, Xuansong Xie, Chunyan Miao
    * Abstract: Image inpainting aims to complete the missing or corrupted regions of images with realistic contents. The prevalent approaches adopt a hybrid objective of reconstruction and perceptual quality by using generative adversarial networks. However, the reconstruction loss and adversarial loss focus on synthesizing contents of different frequencies and simply applying them together often leads to inter-frequency conflicts and compromised inpainting. This paper presents WaveFill, a wavelet-based inpainting network that decomposes images into multiple frequency bands and fills the missing regions in each frequency band separately and explicitly. WaveFill decomposes images by using discrete wavelet transform (DWT) that preserves spatial information naturally. It applies L1 reconstruction loss to the decomposed low-frequency bands and adversarial loss to high-frequency bands, hence effectively mitigate inter-frequency conflicts while completing images in spatial domain. To address the inpainting inconsistency in different frequency bands and fuse features with distinct statistics, we design a novel normalization scheme that aligns and fuses the multi-frequency features effectively. Extensive experiments over multiple datasets show that WaveFill achieves superior image inpainting qualitatively and quantitatively.

count=1
* Hand Image Understanding via Deep Multi-Task Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Hand_Image_Understanding_via_Deep_Multi-Task_Learning_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Hand_Image_Understanding_via_Deep_Multi-Task_Learning_ICCV_2021_paper.pdf)]
    * Title: Hand Image Understanding via Deep Multi-Task Learning
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Xiong Zhang, Hongsheng Huang, Jianchao Tan, Hongmin Xu, Cheng Yang, Guozhu Peng, Lei Wang, Ji Liu
    * Abstract: Analyzing and understanding hand information from multimedia materials like images or videos is important for many real world applications and remains to be very active in research community. There are various works focusing on recovering hand information from single image, however, they usually solve a single task, for example, hand mask segmentation, 2D/3D hand pose estimation, or hand mesh reconstruction and perform not well in challenging scenarios. To further improve the performance of these tasks, we propose a novel Hand Image Understanding (HIU) framework (HIU-DMTL) to extract comprehensive information of the hand object from a single RGB image, by jointly considering the relationships between these tasks. To achieve this goal, a cascaded multi-task learning (MTL) backbone is designed to estimate the 2D heat maps, to learn the segmentation mask, and to generate the intermediate 3D information encoding, followed by a coarse-to-fine learning paradigm and a self-supervised learning strategy. Qualitative experiments demonstrate that our approach is capable of recovering reasonable mesh representations even in challenging situations. Quantitatively, our method significantly outperforms the state-of-the-art approaches on various widely-used datasets, in terms of diverse evaluation metrics.

count=1
* PyMAF: 3D Human Pose and Shape Regression With Pyramidal Mesh Alignment Feedback Loop
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_PyMAF_3D_Human_Pose_and_Shape_Regression_With_Pyramidal_Mesh_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_PyMAF_3D_Human_Pose_and_Shape_Regression_With_Pyramidal_Mesh_ICCV_2021_paper.pdf)]
    * Title: PyMAF: 3D Human Pose and Shape Regression With Pyramidal Mesh Alignment Feedback Loop
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, Zhenan Sun
    * Abstract: Regression-based methods have recently shown promising results in reconstructing human meshes from monocular images. By directly mapping raw pixels to model parameters, these methods can produce parametric models in a feed-forward manner via neural networks. However, minor deviation in parameters may lead to noticeable misalignment between the estimated meshes and image evidences. To address this issue, we propose a Pyramidal Mesh Alignment Feedback (PyMAF) loop to leverage a feature pyramid and rectify the predicted parameters explicitly based on the mesh-image alignment status in our deep regressor. In PyMAF, given the currently predicted parameters, mesh-aligned evidences will be extracted from finer-resolution features accordingly and fed back for parameter rectification. To reduce noise and enhance the reliability of these evidences, an auxiliary pixel-wise supervision is imposed on the feature encoder, which provides mesh-image correspondence guidance for our network to preserve the most related information in spatial features. The efficacy of our approach is validated on several benchmarks, including Human3.6M, 3DPW, LSP, and COCO, where experimental results show that our approach consistently improves the mesh-image alignment of the reconstruction. The project page with code and video results can be found at https://hongwenzhang.github.io/pymaf.

count=1
* Heterogeneous Relational Complement for Vehicle Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Heterogeneous_Relational_Complement_for_Vehicle_Re-Identification_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Heterogeneous_Relational_Complement_for_Vehicle_Re-Identification_ICCV_2021_paper.pdf)]
    * Title: Heterogeneous Relational Complement for Vehicle Re-Identification
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jiajian Zhao, Yifan Zhao, Jia Li, Ke Yan, Yonghong Tian
    * Abstract: The crucial problem in vehicle re-identification is to find the same vehicle identity when reviewing this object from cross-view cameras, which sets a higher demand for learning viewpoint-invariant representations. In this paper, we propose to solve this problem from two aspects: constructing robust feature representations and proposing camera-sensitive evaluations. We first propose a novel Heterogeneous Relational Complement Network (HRCN) by incorporating region-specific features and cross-level features as complements for the original high-level output. Considering the distributional differences and semantic misalignment, we propose graph-based relation modules to embed these heterogeneous features into one unified high-dimensional space. On the other hand, considering the deficiencies of cross-camera evaluations in existing measures (i.e., CMC and AP), we then propose a Cross-camera Generalization Measure (CGM) to improve the evaluations by introducing position-sensitivity and cross-camera generalization penalties. We further construct a new benchmark of existing models with our proposed CGM and experimental results reveal that our proposed HRCN model achieves new state-of-the-art in VeRi-776, VehicleID, and VERI-Wild.

count=1
* C3-SemiSeg: Contrastive Semi-Supervised Segmentation via Cross-Set Learning and Dynamic Class-Balancing
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_C3-SemiSeg_Contrastive_Semi-Supervised_Segmentation_via_Cross-Set_Learning_and_Dynamic_Class-Balancing_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_C3-SemiSeg_Contrastive_Semi-Supervised_Segmentation_via_Cross-Set_Learning_and_Dynamic_Class-Balancing_ICCV_2021_paper.pdf)]
    * Title: C3-SemiSeg: Contrastive Semi-Supervised Segmentation via Cross-Set Learning and Dynamic Class-Balancing
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yanning Zhou, Hang Xu, Wei Zhang, Bin Gao, Pheng-Ann Heng
    * Abstract: The semi-supervised semantic segmentation methods utilize the unlabeled data to increase the feature discriminative ability to alleviate the burden of the annotated data. However, the dominant consistency learning diagram is limited by a) the misalignment between features from labeled and unlabeled data; b) treating each image and region separately without considering crucial semantic dependencies among classes. In this work, we introduce a novel C^3-SemiSeg to improve consistency-based semi-supervised learning by exploiting better feature alignment under perturbations and enhancing discriminative of the inter-class features cross images. Specifically, we first introduce a cross-set region-level data augmentation strategy to reduce the feature discrepancy between labeled data and unlabeled data. Cross-set pixel-wise contrastive learning is further integrated into the pipeline to facilitate discriminative and consistent intra-class features in a `compared to learn' way. To stabilize training from the noisy label, we propose a dynamic confidence region selection strategy to focus on the high confidence region for loss calculation. We validate the proposed approach on Cityscapes and BDD100K dataset, which significantly outperforms other state-of-the-art semi-supervised semantic segmentation methods.

count=1
* Distilling Reflection Dynamics for Single-Image Reflection Removal
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Zheng_Distilling_Reflection_Dynamics_for_Single-Image_Reflection_Removal_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Zheng_Distilling_Reflection_Dynamics_for_Single-Image_Reflection_Removal_ICCVW_2021_paper.pdf)]
    * Title: Distilling Reflection Dynamics for Single-Image Reflection Removal
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Quanlong Zheng, Xiaotian Qiao, Ying Cao, Shi Guo, Lei Zhang, Rynson W.H. Lau
    * Abstract: Single-image reflection removal (SIRR) aims to restore the transmitted image given a single image shot through glass or window. Existing methods rely mainly on information extracted from a single image along with some pre-defined priors, and fail to give satisfying results on real-world images, due to inherent ambiguity and lack of large and diverse real-world training data. In this paper, instead of reasoning about a single image only, we propose to distill a representation of reflection dynamics from multi-view images (i.e., the motions of reflection and transmission layers over time), and transfer the learned knowledge for the SIRR problem. In particular, we propose a teacher-student framework where the teacher network learns a representation of reflection dynamics by watching a sequence of multi-view images of a scene captured by a moving camera and teaches a student network to remove reflection from a single input image. In addition, we collect a large real-world multi-view reflection image dataset for reflection dynamics knowledge distillation. Extensive experiments show that our model yields state-of-the-art performances.

count=1
* Self-Supervised Burst Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Bhat_Self-Supervised_Burst_Super-Resolution_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Bhat_Self-Supervised_Burst_Super-Resolution_ICCV_2023_paper.pdf)]
    * Title: Self-Supervised Burst Super-Resolution
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Goutam Bhat, Michaël Gharbi, Jiawen Chen, Luc Van Gool, Zhihao Xia
    * Abstract: We introduce a self-supervised training strategy for burst super-resolution that only uses noisy low-resolution bursts during training. Our approach eliminates the need to carefully tune synthetic data simulation pipelines, which often do not match real-world image statistics. Compared to weakly-paired training strategies, which require noisy smartphone burst photos of static scenes, paired with a clean reference obtained from a tripod-mounted DSLR camera, our approach is more scalable, and avoids the color mismatch between the smartphone and DSLR. To achieve this, we propose a new self-supervised objective that uses a forward imaging model to recover a high-resolution image from aliased high frequencies in the burst. Our approach does not require any manual tuning of the forward model's parameters; we learn them from data. Furthermore, we show our training strategy is robust to dynamic scene motion in the burst, which enables training burst super-resolution models using in-the-wild data. Extensive experiments on real and synthetic data show that, despite only using noisy bursts during training, models trained with our self-supervised strategy match, and sometimes surpass, the quality of fully-supervised baselines trained with synthetic data or weakly-paired ground-truth. Finally, we show our training strategy is general using four different burst super-resolution architectures.

count=1
* Activate and Reject: Towards Safe Domain Generalization under Category Shift
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Activate_and_Reject_Towards_Safe_Domain_Generalization_under_Category_Shift_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Activate_and_Reject_Towards_Safe_Domain_Generalization_under_Category_Shift_ICCV_2023_paper.pdf)]
    * Title: Activate and Reject: Towards Safe Domain Generalization under Category Shift
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Chaoqi Chen, Luyao Tang, Leitian Tao, Hong-Yu Zhou, Yue Huang, Xiaoguang Han, Yizhou Yu
    * Abstract: Albeit the notable performance on in-domain test points, it is non-trivial for deep neural networks to attain satisfactory accuracy when deploying in the open world, where novel domains and object classes often occur. In this paper, we study a practical problem of Domain Generalization under Category Shift (DGCS), which aims to simultaneously detect unknown-class samples and classify known-class samples in the target domains. Compared to prior DG works, we face two new challenges: 1) how to learn the concept of "unknown" during training with only source known-class samples, and 2) how to adapt the source-trained model to unseen environments for safe model deployment. To this end, we propose a novel Activate and Reject (ART) framework to reshape the model's decision boundary to accommodate unknown classes and conduct post hoc modification to further discriminate known and unknown classes using unlabeled test data. Specifically, during training, we promote the response to the unknown by optimizing the unknown probability and then smoothing the overall output to mitigate the overconfidence issue. At test time, we introduce a step-wise online adaptation method that predicts the label by virtue of the cross-domain nearest neighbor and class prototype information without updating the network's parameters or using threshold-based mechanisms. Experiments reveal that ART consistently improves the generalization capability of deep networks on different vision tasks. For image classification, ART improves the H-score by 6.1% on average compared to the previous best method. For object detection and semantic segmentation, we establish new benchmarks and achieve competitive performance.

count=1
* Learning Continuous Exposure Value Representations for Single-Image HDR Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Learning_Continuous_Exposure_Value_Representations_for_Single-Image_HDR_Reconstruction_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Learning_Continuous_Exposure_Value_Representations_for_Single-Image_HDR_Reconstruction_ICCV_2023_paper.pdf)]
    * Title: Learning Continuous Exposure Value Representations for Single-Image HDR Reconstruction
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Su-Kai Chen, Hung-Lin Yen, Yu-Lun Liu, Min-Hung Chen, Hou-Ning Hu, Wen-Hsiao Peng, Yen-Yu Lin
    * Abstract: Deep learning is commonly used to produce impressive results in reconstructing HDR images from LDR images. LDR stack-based methods are used for single-image HDR reconstruction, generating an HDR image from a deep learning generated LDR stack. However, current methods generate the LDR stack with predetermined exposure values (EVs), which may limit the quality of HDR reconstruction. To address this, we propose the continuous exposure value representation (CEVR) model, which uses an implicit function to generate LDR images with arbitrary EVs, including those unseen during training. Our flexible approach generates a continuous stack with more images containing diverse EVs, significantly improving HDR reconstruction. We use a cycle training strategy to supervise the model in generating continuous EV LDR images without corresponding ground truths. Our CEVR model outperforms existing methods, as demonstrated by experimental results.

count=1
* LAN-HDR: Luminance-based Alignment Network for High Dynamic Range Video Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chung_LAN-HDR_Luminance-based_Alignment_Network_for_High_Dynamic_Range_Video_Reconstruction_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chung_LAN-HDR_Luminance-based_Alignment_Network_for_High_Dynamic_Range_Video_Reconstruction_ICCV_2023_paper.pdf)]
    * Title: LAN-HDR: Luminance-based Alignment Network for High Dynamic Range Video Reconstruction
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Haesoo Chung, Nam Ik Cho
    * Abstract: As demands for high-quality videos continue to rise, high-resolution and high-dynamic range (HDR) imaging techniques are drawing attention. To generate an HDR video from low dynamic range (LDR) images, one of the critical steps is the motion compensation between LDR frames, for which most existing works employed the optical flow algorithm. However, these methods suffer from flow estimation errors when saturation or complicated motions exist. In this paper, we propose an end-to-end HDR video composition framework, which aligns LDR frames in the feature space and then merges aligned features into an HDR frame, without relying on pixel-domain optical flow. Specifically, we propose a luminance-based alignment network for HDR (LAN-HDR) consisting of an alignment module and a hallucination module. The alignment module aligns a frame to the adjacent reference by evaluating luminance-based attention, excluding color information. The hallucination module generates sharp details, especially for washed-out areas due to saturation. The aligned and hallucinated features are then blended adaptively to complement each other. Finally, we merge the features to generate a final HDR frame. In training, we adopt a temporal loss, in addition to frame reconstruction losses, to enhance temporal consistency and thus reduce flickering. Extensive experiments demonstrate that our method performs better or comparable to state-of-the-art methods on several benchmarks. Codes are available at https://github.com/haesoochung/LAN-HDR.

count=1
* Single Image Reflection Separation via Component Synergy
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Single_Image_Reflection_Separation_via_Component_Synergy_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Single_Image_Reflection_Separation_via_Component_Synergy_ICCV_2023_paper.pdf)]
    * Title: Single Image Reflection Separation via Component Synergy
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Qiming Hu, Xiaojie Guo
    * Abstract: The reflection superposition phenomenon is complex and widely distributed in the real world, which derives various simplified linear and nonlinear formulations of the problem. In this paper, based on the investigation of the weaknesses of existing models, we propose a more general form of the superposition model by introducing a learnable residue term, which can effectively capture residual information during decomposition, guiding the separated layers to be complete. In order to fully capitalize on its advantages, we further design the network structure elaborately, including a novel dual-stream interaction mechanism and a powerful decomposition network with a semantic pyramid encoder. Extensive experiments and ablation studies are conducted to verify our superiority over state-of-the-art approaches on multiple real-world benchmark datasets.

count=1
* PG-RCNN: Semantic Surface Point Generation for 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Koo_PG-RCNN_Semantic_Surface_Point_Generation_for_3D_Object_Detection_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Koo_PG-RCNN_Semantic_Surface_Point_Generation_for_3D_Object_Detection_ICCV_2023_paper.pdf)]
    * Title: PG-RCNN: Semantic Surface Point Generation for 3D Object Detection
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Inyong Koo, Inyoung Lee, Se-Ho Kim, Hee-Seon Kim, Woo-jin Jeon, Changick Kim
    * Abstract: One of the main challenges in LiDAR-based 3D object detection is that the sensors often fail to capture the complete spatial information about the objects due to long distance and occlusion. Two-stage detectors with point cloud completion approaches tackle this problem by adding more points to the regions of interest (RoIs) with a pre-trained network. However, these methods generate dense point clouds of objects for all region proposals, assuming that objects always exist in the RoIs. This leads to the indiscriminate point generation for incorrect proposals as well. Motivated by this, we propose Point Generation R-CNN (PG-RCNN), a novel end-to-end detector that generates semantic surface points of foreground objects for accurate detection. Our method uses a jointly trained RoI point generation module to process the contextual information of RoIs and estimate the complete shape and displacement of foreground objects. For every generated point, PG-RCNN assigns a semantic feature that indicates the estimated foreground probability. Extensive experiments show that the point clouds generated by our method provide geometrically and semantically rich information for refining false positive and misaligned proposals. PG-RCNN achieves competitive performance on the KITTI benchmark, with significantly fewer parameters than state-of-the-art models. The code is available at https://github.com/quotation2520/PG-RCNN.

count=1
* BEV-DG: Cross-Modal Learning under Bird's-Eye View for Domain Generalization of 3D Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_BEV-DG_Cross-Modal_Learning_under_Birds-Eye_View_for_Domain_Generalization_of_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_BEV-DG_Cross-Modal_Learning_under_Birds-Eye_View_for_Domain_Generalization_of_ICCV_2023_paper.pdf)]
    * Title: BEV-DG: Cross-Modal Learning under Bird's-Eye View for Domain Generalization of 3D Semantic Segmentation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Miaoyu Li, Yachao Zhang, Xu Ma, Yanyun Qu, Yun Fu
    * Abstract: Cross-modal Unsupervised Domain Adaptation (UDA) aims to exploit the complementarity of 2D-3D data to overcome the lack of annotation in a new domain. However, UDA methods rely on access to the target domain during training, meaning the trained model only works in a specific target domain. In light of this, we propose cross-modal learning under bird's-eye view for Domain Generalization (DG) of 3D semantic segmentation, called BEV-DG. DG is more challenging because the model cannot access the target domain during training, meaning it needs to rely on cross-modal learning to alleviate the domain gap. Since 3D semantic segmentation requires the classification of each point, existing cross-modal learning is directly conducted point-to-point, which is sensitive to the misalignment in projections between pixels and points. To this end, our approach aims to optimize domain-irrelevant representation modeling with the aid of cross-modal learning under bird's-eye view. We propose BEV-based Area-to-area Fusion (BAF) to conduct cross-modal learning under bird's-eye view, which has a higher fault tolerance for point-level misalignment. Furthermore, to model domain-irrelevant representations, we propose BEV-driven Domain Contrastive Learning (BDCL) with the help of cross-modal learning under bird's-eye view. We design three domain generalization settings based on three 3D datasets, and BEV-DG significantly outperforms state-of-the-art competitors with tremendous margins in all settings.

count=1
* Neural Characteristic Function Learning for Conditional Image Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Neural_Characteristic_Function_Learning_for_Conditional_Image_Generation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Neural_Characteristic_Function_Learning_for_Conditional_Image_Generation_ICCV_2023_paper.pdf)]
    * Title: Neural Characteristic Function Learning for Conditional Image Generation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Shengxi Li, Jialu Zhang, Yifei Li, Mai Xu, Xin Deng, Li Li
    * Abstract: The emergence of conditional generative adversarial networks (cGANs) has revolutionised the way we approach and control the generation, by means of adversarially learning joint distributions of data and auxiliary information. Despite the success, cGANs have been consistently put under scrutiny due to their ill-posed discrepancy measure between distributions, leading to mode collapse and instability problems in training. To address this issue, we propose a novel conditional characteristic function generative adversarial network (CCF-GAN) to reduce the discrepancy by the characteristic functions (CFs), which is able to learn accurate distance measure of joint distributions under theoretical soundness. More specifically, the difference between CFs is first proved to be complete and optimisation-friendly, for measuring the discrepancy of two joint distributions. To relieve the problem of curse of dimensionality in calculating CF difference, we propose to employ the neural network, namely neural CF (NCF), to efficiently minimise an upper bound of the difference. Based on the NCF, we establish the CCF-GAN framework to explicitly decompose CFs of joint distributions, which allows for learning the data distribution and auxiliary information with classified importance. The experimental results on synthetic and real-world datasets verify the superior performances of our CCF-GAN, on both the generation quality and stability.

count=1
* Reference-guided Controllable Inpainting of Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Mirzaei_Reference-guided_Controllable_Inpainting_of_Neural_Radiance_Fields_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Mirzaei_Reference-guided_Controllable_Inpainting_of_Neural_Radiance_Fields_ICCV_2023_paper.pdf)]
    * Title: Reference-guided Controllable Inpainting of Neural Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, Igor Gilitschenski
    * Abstract: The popularity of Neural Radiance Fields (NeRFs) for view synthesis has led to a desire for NeRF editing tools. Here, we focus on inpainting regions in a view-consistent and controllable manner. In addition to the typical NeRF inputs and masks delineating the unwanted region in each view, we require only a single inpainted view of the scene, i.e., a reference view. We use monocular depth estimators to back-project the inpainted view to the correct 3D positions. Then, via a novel rendering technique, a bilateral solver can construct view-dependent effects in non-reference views, making the inpainted region appear consistent from any view. For non-reference disoccluded regions, which cannot be supervised by the single reference view, we devise a method based on image inpainters to guide both the geometry and appearance. Our approach shows superior performance to NeRF inpainting baselines, with the additional advantage that a user can control the generated scene via a single inpainted image.

count=1
* Mining bias-target Alignment from Voronoi Cells
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Nahon_Mining_bias-target_Alignment_from_Voronoi_Cells_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Nahon_Mining_bias-target_Alignment_from_Voronoi_Cells_ICCV_2023_paper.pdf)]
    * Title: Mining bias-target Alignment from Voronoi Cells
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Rémi Nahon, Van-Tam Nguyen, Enzo Tartaglione
    * Abstract: Despite significant research efforts, deep neural networks remain vulnerable to biases: this raises concerns about their fairness and limits their generalization. In this paper, we propose a bias-agnostic approach to mitigate the impact of biases in deep neural networks. Unlike traditional debiasing approaches, we rely on a metric to quantify "bias alignment/misalignment" on target classes and use this information to discourage the propagation of bias-target alignment information through the network. We conduct experiments on several commonly used datasets for debiasing and compare our method with supervised and bias-specific approaches. Our results indicate that the proposed method achieves comparable performance to state-of-the-art supervised approaches, despite being bias-agnostic, even in the presence of multiple biases in the same sample.

count=1
* Conceptual and Hierarchical Latent Space Decomposition for Face Editing
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Ozkan_Conceptual_and_Hierarchical_Latent_Space_Decomposition_for_Face_Editing_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Ozkan_Conceptual_and_Hierarchical_Latent_Space_Decomposition_for_Face_Editing_ICCV_2023_paper.pdf)]
    * Title: Conceptual and Hierarchical Latent Space Decomposition for Face Editing
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Savas Ozkan, Mete Ozay, Tom Robinson
    * Abstract: Generative Adversarial Networks (GANs) can produce photo-realistic results using an unconditional image-generation pipeline. However, the images generated by GANs (e.g., StyleGAN) are entangled in feature spaces, which makes it difficult to interpret and control the contents of images. In this paper, we present an encoder-decoder model that decomposes the entangled GAN space into a conceptual and hierarchical latent space in a self-supervised manner. The outputs of 3D morphable face models are leveraged to independently control image synthesis parameters like pose, expression, and illumination. For this purpose, a novel latent space decomposition pipeline is introduced using transformer networks and generative models. Later, this new space is used to optimize a transformer-based GAN space controller for face editing. In this work, a StyleGAN2 model for faces is utilized. Since our method manipulates only GAN features, the photo-realism of StyleGAN2 is fully preserved. The results demonstrate that our method qualitatively and quantitatively outperforms baselines in terms of identity preservation and editing precision.

count=1
* GlueGen: Plug and Play Multi-modal Encoders for X-to-image Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Qin_GlueGen_Plug_and_Play_Multi-modal_Encoders_for_X-to-image_Generation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Qin_GlueGen_Plug_and_Play_Multi-modal_Encoders_for_X-to-image_Generation_ICCV_2023_paper.pdf)]
    * Title: GlueGen: Plug and Play Multi-modal Encoders for X-to-image Generation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Can Qin, Ning Yu, Chen Xing, Shu Zhang, Zeyuan Chen, Stefano Ermon, Yun Fu, Caiming Xiong, Ran Xu
    * Abstract: Text-to-image (T2I) models based on diffusion processes have achieved remarkable success in controllable image generation using user-provided captions. However, the tight coupling between the current text encoder and image decoder in T2I models makes it challenging to replace or upgrade. Such changes often require massive fine-tuning or even training from scratch with the prohibitive expense. To address this problem, we propose GlueGen, which applies a newly proposed GlueNet model to align features from single-modal or multi-modal encoders with the latent space of an existing T2I model. The approach introduces a new training objective that leverages parallel corpora to align the representation spaces of different encoders. Empirical results show that GlueNet can be trained efficiently and enables various capabilities beyond previous state-of-the-art models: 1) multilingual language models such as XLM-Roberta can be aligned with existing T2I models, allowing for the generation of high-quality images from captions beyond English; 2) GlueNet can align multi-modal encoders such as AudioCLIP with the Stable Diffusion model, enabling sound-to-image generation; 3) it can also upgrade the current text encoder of the latent diffusion model for challenging case generation. By the alignment of various feature representations, the GlueNet allows for flexible and efficient integration of new functionality into existing T2I models and sheds light on X-to-image (X2I) generation.

count=1
* Decoupled Iterative Refinement Framework for Interacting Hands Reconstruction from a Single RGB Image
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Ren_Decoupled_Iterative_Refinement_Framework_for_Interacting_Hands_Reconstruction_from_a_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Ren_Decoupled_Iterative_Refinement_Framework_for_Interacting_Hands_Reconstruction_from_a_ICCV_2023_paper.pdf)]
    * Title: Decoupled Iterative Refinement Framework for Interacting Hands Reconstruction from a Single RGB Image
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Pengfei Ren, Chao Wen, Xiaozheng Zheng, Zhou Xue, Haifeng Sun, Qi Qi, Jingyu Wang, Jianxin Liao
    * Abstract: Reconstructing interacting hands from a single RGB image is a very challenging task. On the one hand, severe mutual occlusion and similar local appearance between two hands confuse the extraction of visual features, resulting in the misalignment of estimated hand meshes and the image. On the other hand, there are complex spatial relationship between interacting hands, which significantly increases the solution space of hand poses and increases the difficulty of network learning. In this paper, we propose a decoupled iterative refinement framework to achieve pixel-alignment hand reconstruction while efficiently modeling the spatial relationship between hands. Specifically, we define two feature spaces with different characteristics, namely 2D visual feature space and 3D joint feature space. First, we obtain joint-wise features from the visual feature map and utilize a graph convolution network and a transformer to perform intra- and inter-hand information interaction in the 3D joint feature space, respectively. Then, we project the joint features with global information back into the 2D visual feature space in an obfuscation-free manner and utilize the 2D convolution for pixel-wise enhancement. By performing multiple alternate enhancements in the two feature spaces, our method can achieve an accurate and robust reconstruction of interacting hands. Our method outperforms all existing two-hand reconstruction methods by a large margin on the InterHand2.6M dataset.

count=1
* Unified Pre-Training with Pseudo Texts for Text-To-Image Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Shao_Unified_Pre-Training_with_Pseudo_Texts_for_Text-To-Image_Person_Re-Identification_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Shao_Unified_Pre-Training_with_Pseudo_Texts_for_Text-To-Image_Person_Re-Identification_ICCV_2023_paper.pdf)]
    * Title: Unified Pre-Training with Pseudo Texts for Text-To-Image Person Re-Identification
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zhiyin Shao, Xinyu Zhang, Changxing Ding, Jian Wang, Jingdong Wang
    * Abstract: The pre-training task is indispensable for the text-to-image person re-identification (T2I-ReID) task. However, there are two underlying inconsistencies between these two tasks that may impact the performance: i) Data inconsistency. A large domain gap exists between the generic images/texts used in public pre-trained models and the specific person data in the T2I-ReID task. This gap is especially severe for texts, as general textual data are usually unable to describe specific people in fine-grained detail. ii) Training inconsistency. The processes of pre-training of images and texts are independent, despite cross-modality learning being critical to T2I-ReID. To address the above issues, we present a new unified pre-training pipeline (UniPT) designed specifically for the T2I-ReID task. We first build a large-scale text-labeled person dataset "LUPerson-T", in which pseudo-textual descriptions of images are automatically generated by the CLIP paradigm using a divide-conquer-combine strategy. Benefiting from this dataset, we then utilize a simple vision-and-language pre-training framework to explicitly align the feature space of the image and text modalities during pre-training. In this way, the pre-training task and the T2I-ReID task are made consistent with each other on both data and training levels. Without the need for any bells and whistles, our UniPT achieves competitive Rank-1 accuracy of, i.e., 68.50%, 60.09%, and 51.85% on CUHK-PEDES, ICFG-PEDES and RSTPReid, respectively. Both the LUPerson-T dataset and code are available at https://github.com/ZhiyinShao-H/UniPT.

count=1
* Exploring the Sim2Real Gap Using Digital Twins
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Sudhakar_Exploring_the_Sim2Real_Gap_Using_Digital_Twins_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Sudhakar_Exploring_the_Sim2Real_Gap_Using_Digital_Twins_ICCV_2023_paper.pdf)]
    * Title: Exploring the Sim2Real Gap Using Digital Twins
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Sruthi Sudhakar, Jon Hanzelka, Josh Bobillot, Tanmay Randhavane, Neel Joshi, Vibhav Vineet
    * Abstract: It is very time consuming to create datasets for training computer vision models. An emerging alternative is to use synthetic data, but if the synthetic data is not similar enough to the real data, the performance is typically below that of training with real data. Thus using synthetic data still requires a large amount of time, money, and skill as one needs to author the data carefully. In this paper, we seek to understand which aspects of this authoring process are most critical. We present an analysis of which factors of variation between simulated and real data are most important. We capture images of YCB objects to create a novel YCB-Real dataset. We then create a novel synthetic "digital twin" dataset, YCB-Synthetic, which matches the YCB-Real dataset and includes variety of artifacts added to the synthetic data. We study the affects of these artifacts on our dataset and two existing published datasets on two different computer vision tasks: object detection and instance segmentation. We provide an analysis of the cost-benefit trade-offs between artist time for fixing artifacts and trained model accuracy. We plan to release this dataset (images and 3D assets) so they can be further used by the community.

count=1
* Too Large; Data Reduction for Vision-Language Pre-Training
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Too_Large_Data_Reduction_for_Vision-Language_Pre-Training_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Too_Large_Data_Reduction_for_Vision-Language_Pre-Training_ICCV_2023_paper.pdf)]
    * Title: Too Large; Data Reduction for Vision-Language Pre-Training
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Alex Jinpeng Wang, Kevin Qinghong Lin, David Junhao Zhang, Stan Weixian Lei, Mike Zheng Shou
    * Abstract: This paper examines the problems of severe image-text misalignment and high redundancy in the widely-used large-scale Vision-Language Pre-Training (VLP) datasets. To address these issues, we propose an efficient and straightforward Vision-Language learning algorithm called TL;DR which aims to compress the existing large VLP data into a small, high-quality set. Our approach consists of two major steps. First, a codebook-based encoder-decoder captioner is developed to select representative samples. Second, a new caption is generated to complement the original captions for selected samples, mitigating the text-image misalignment problem while maintaining uniqueness. As the result, TL;DR enables us to reduce the large dataset into a small set of high-quality data, which can serve as an alternative pre-training dataset. This algorithm significantly speeds up the time-consuming pretraining process. Specifically, TL;DR can compress the mainstream VLP datasets at a high ratio, e.g., reduce well-cleaned CC3M dataset from 2.8M to 0.67M ( 24%) and noisy YFCC15M from 15M to 2.5M ( 16.7%). Extensive experiments with three popular VLP models over seven downstream tasks show that VLP model trained on the compressed dataset provided by TL;DR can perform similar or even better results compared with training on the full-scale dataset.

count=1
* Cross-view Semantic Alignment for Livestreaming Product Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Cross-view_Semantic_Alignment_for_Livestreaming_Product_Recognition_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Cross-view_Semantic_Alignment_for_Livestreaming_Product_Recognition_ICCV_2023_paper.pdf)]
    * Title: Cross-view Semantic Alignment for Livestreaming Product Recognition
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Wenjie Yang, Yiyi Chen, Yan Li, Yanhua Cheng, Xudong Liu, Quan Chen, Han Li
    * Abstract: Live commerce is the act of selling products online through livestreaming. The customer's diverse demands for online products introduces more challenges to Livestreaming Product Recognition. Previous works are either focus on fashion clothing data or subject to single-modal input, thus inconsistent with the real-world scenario where multimodal data from various categories are present. In this paper, we contribute LPR4M, a large-scale multimodal dataset that covers 34 categories, comprises 3 modalities (image, video, and text), and is 50 times larger than the largest publicly available dataset. In addition, LPR4M contains diverse videos and noise modality pair while also having a long-tailed distribution, resembling real-world problems. Moreover, a cRoss-vIew semantiC alignmEnt (RICE) model is proposed to learn discriminative instance features from the two views (image and video) of products via instance-level contrastive learning as well as cross-view patch-level feature propagation. A novel Patch Feature Reconstruction loss is proposed to penalize the semantic misalignment between the cross-view patches. Extensive ablation studies demonstrate the effectiveness of RICE and provide insights into the importance of dataset diversity and expressivity.

count=1
* FedPD: Federated Open Set Recognition with Parameter Disentanglement
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_FedPD_Federated_Open_Set_Recognition_with_Parameter_Disentanglement_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_FedPD_Federated_Open_Set_Recognition_with_Parameter_Disentanglement_ICCV_2023_paper.pdf)]
    * Title: FedPD: Federated Open Set Recognition with Parameter Disentanglement
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Chen Yang, Meilu Zhu, Yifan Liu, Yixuan Yuan
    * Abstract: Existing federated learning (FL) approaches are deployed under the unrealistic closed-set setting, with both training and testing classes belong to the same set, which makes the global model fail to identify the unseen classes as `unknown'. To this end, we aim to study a novel problem of federated open-set recognition (FedOSR), which learns an open-set recognition (OSR) model under federated paradigm such that it classifies seen classes while at the same time detects unknown classes. In this work, we propose a parameter disentanglement guided federated open-set recognition (FedPD) algorithm to address two core challenges of FedOSR: cross-client inter-set interference between learning closed-set and open-set knowledge and cross-client intra-set inconsistency by data heterogeneity. The proposed FedPD framework mainly leverages two modules, i.e., local parameter disentanglement (LPD) and global divide-and-conquer aggregation (GDCA), to first disentangle client OSR model into different subnetworks, then align the corresponding parts cross clients for matched model aggregation. Specifically, on the client side, LPD decouples an OSR model into a closed-set subnetwork and an open-set subnetwork by the task-related importance, thus preventing inter-set interference. On the server side, GDCA first partitions the two subnetworks into specific and shared parts, and subsequently aligns the corresponding parts through optimal transport to eliminate parameter misalignment. Extensive experiments on various datasets demonstrate the superior performance of our proposed method.

count=1
* Adverse Weather Removal with Codebook Priors
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Ye_Adverse_Weather_Removal_with_Codebook_Priors_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Ye_Adverse_Weather_Removal_with_Codebook_Priors_ICCV_2023_paper.pdf)]
    * Title: Adverse Weather Removal with Codebook Priors
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Tian Ye, Sixiang Chen, Jinbin Bai, Jun Shi, Chenghao Xue, Jingxia Jiang, Junjie Yin, Erkang Chen, Yun Liu
    * Abstract: Despite recent advancements in unified adverse weather removal methods, there remains a significant challenge of achieving realistic fine-grained texture and reliable background reconstruction to mitigate serious distortions. Inspired by recent advancements in codebook and vector quantization (VQ) techniques, we present a novel Adverse Weather Removal network with Codebook Priors (AWRCP) to address the problem of unified adverse weather removal. AWRCP leverages high-quality codebook priors derived from undistorted images to recover vivid texture details and faithful background structures. However, simply utilizing high-quality features from the codebook does not guarantee good results in terms of fine-grained details and structural fidelity. Therefore, we develop a deformable cross-attention with sparse sampling mechanism for flexible perform feature interaction between degraded features and high-quality features from codebook priors. In order to effectively incorporate high-quality texture features while maintaining the realism of the details generated by codebook priors, we propose a hierarchical texture warping head that gradually fuses hierarchical codebook prior features into high-resolution features at final restoring stage. With the utilization of the VQ codebook as a feature dictionary of high quality and the proposed designs, AWRCP can largely improve the restored quality of texture details, achieving the state-of-the-art performance across multiple adverse weather removal benchmark.

count=1
* Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Decoupled_DETR_Spatially_Disentangling_Localization_and_Classification_for_Improved_End-to-End_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Decoupled_DETR_Spatially_Disentangling_Localization_and_Classification_for_Improved_End-to-End_ICCV_2023_paper.pdf)]
    * Title: Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Manyuan Zhang, Guanglu Song, Yu Liu, Hongsheng Li
    * Abstract: The introduction of DETR represents a new paradigm for object detection. However, its decoder conducts classification and box localization using shared queries and cross-attention layers, leading to suboptimal results. We observe that different regions of interest in the visual feature map are suitable for performing query classification and box localization tasks, even for the same object. Salient regions provide vital information for classification, while the boundaries around them are more favorable for box regression. Unfortunately, such spatial misalignment between these two tasks greatly hinders DETR's training. Therefore, in this work, we focus on decoupling localization and classification tasks in DETR. To achieve this, we introduce a new design scheme called spatially decoupled DETR (SD-DETR), which includes a task-aware query generation module and a disentangled feature learning process. We elaborately design the task-aware query initialization process and divide the cross-attention block in the decoder to allow the task-aware queries to match different visual regions. Meanwhile, we also observe that the prediction misalignment problem for high classification confidence and precise localization exists, so we propose an alignment loss to further guide the spatially decoupled DETR training. Through extensive experiments, we demonstrate that our approach achieves a significant improvement in MSCOCO datasets compared to previous work. For instance, we improve the performance of Conditional DETR by 4.5%. By spatially disentangling the two tasks, our method overcomes the misalignment problem and greatly improves the performance of DETR for object detection.

count=1
* H3WB: Human3.6M 3D WholeBody Dataset and Benchmark
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_H3WB_Human3.6M_3D_WholeBody_Dataset_and_Benchmark_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_H3WB_Human3.6M_3D_WholeBody_Dataset_and_Benchmark_ICCV_2023_paper.pdf)]
    * Title: H3WB: Human3.6M 3D WholeBody Dataset and Benchmark
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yue Zhu, Nermin Samet, David Picard
    * Abstract: We present a benchmark for 3D human whole-body pose estimation, which involves identifying accurate 3D keypoints on the entire human body, including face, hands, body, and feet. Currently, the lack of a fully annotated and accurate 3D whole-body dataset results in deep networks being trained separately on specific body parts, which are combined during inference. Or they rely on pseudo-groundtruth provided by parametric body models which are not as accurate as detection based methods. To overcome these issues, we introduce the Human3.6M 3D WholeBody (H3WB) dataset, which provides whole-body annotations for the Human3.6M dataset using the COCO Wholebody layout. H3WB comprises 133 whole-body keypoint annotations on 100K images, made possible by our new multi-view pipeline. We also propose three tasks: i) 3D whole-body pose lifting from 2D complete whole-body pose, ii) 3D whole-body pose lifting from 2D incomplete whole-body pose, and iii) 3D whole-body pose estimation from a single RGB image. Additionally, we report several baselines from popular methods for these tasks. Furthermore, we also provide automated 3D whole-body annotations of TotalCapture and experimentally show that when used with H3WB it helps to improve the performance.

count=1
* Learning Interpretable Forensic Representations via Local Window Modulation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/DFAD/html/Das_Learning_Interpretable_Forensic_Representations_via_Local_Window_Modulation_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/DFAD/papers/Das_Learning_Interpretable_Forensic_Representations_via_Local_Window_Modulation_ICCVW_2023_paper.pdf)]
    * Title: Learning Interpretable Forensic Representations via Local Window Modulation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Sowmen Das, Md. Ruhul Amin
    * Abstract: The majority of existing image forgeries involve augmenting a specific region of the source image which leaves detectable artifacts and forensic traces. These distinguishing features are mostly found in and around the local neighborhood of the manipulated pixels. However, patch-based detection approaches quickly become intractable due to inefficient computation and low robustness. In this work, we investigate how to effectively learn these forensic representations using local window-based attention techniques. We propose Forensic Modulation Network (ForMoNet) that uses focal modulation and gated attention layers to automatically identify the long and short-range context for any query pixel. Furthermore, the network is more interpretable and computationally efficient than standard self-attention, which is critical for real-world applications. Our evaluation of various benchmarks shows that ForMoNet outperforms existing transformer-based forensic networks by 6% to 11% on different forgeries.

count=1
* JEDI: Joint Expert Distillation in a Semi-Supervised Multi-Dataset Student-Teacher Scenario for Video Action Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/html/Bicsi_JEDI_Joint_Expert_Distillation_in_a_Semi-Supervised_Multi-Dataset_Student-Teacher_Scenario_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/papers/Bicsi_JEDI_Joint_Expert_Distillation_in_a_Semi-Supervised_Multi-Dataset_Student-Teacher_Scenario_ICCVW_2023_paper.pdf)]
    * Title: JEDI: Joint Expert Distillation in a Semi-Supervised Multi-Dataset Student-Teacher Scenario for Video Action Recognition
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Lucian Bicsi, Bogdan Alexe, Radu Tudor Ionescu, Marius Leordeanu
    * Abstract: We propose JEDI, a multi-dataset semi-supervised learning method, which efficiently combines knowledge from multiple experts, learned on different datasets, to train and improve the performance of individual, per dataset, student models. Our approach achieves this by addressing two important problems in current machine learning research: generalization across datasets and limitations of supervised training due to scarcity of labeled data. We start with an arbitrary number of experts, pretrained on their own specific dataset, which form the initial set of student models. The teachers are immediately derived by concatenating the feature representations from the penultimate layers of the students. We then train all models in a student teacher semi-supervised learning scenario until convergence. In our efficient approach, student-teacher training is carried out jointly and end-to-end, showing that both students and teachers improve their generalization capacity during training. We validate our approach on four video action recognition datasets. By simultaneously considering all datasets within a unified semi-supervised setting, we demonstrate significant improvements over the initial experts.

count=1
* VSCHH 2023: A Benchmark for the View Synthesis Challenge of Human Heads
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/RHWC/html/Jang_VSCHH_2023_A_Benchmark_for_the_View_Synthesis_Challenge_of_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/RHWC/papers/Jang_VSCHH_2023_A_Benchmark_for_the_View_Synthesis_Challenge_of_ICCVW_2023_paper.pdf)]
    * Title: VSCHH 2023: A Benchmark for the View Synthesis Challenge of Human Heads
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Youngkyoon Jang, Jiali Zheng, Jifei Song, Helisa Dhamo, Eduardo Pérez-Pellitero, Thomas Tanay, Matteo Maggioni, Richard Shaw, Sibi Catley-Chandar, Yiren Zhou, Jiankang Deng, Ruijie Zhu, Jiahao Chang, Ziyang Song, Jiahuan Yu, Tianzhu Zhang, Khanh-Binh Nguyen, Joon-Sung Yang, Andreea Dogaru, Bernhard Egger, Heng Yu, Aarush Gupta, Joel Julin, László A. Jeni, Hyeseong Kim, Jungbin Cho, Dosik Hwang, Deukhee Lee, Doyeon Kim, Dongseong Seo, SeungJin Jeon, YoungDon Choi, Jun Seok Kang, Ahmet Cagatay Seker, Sang Chul Ahn, Ales Leonardis, Stefanos Zafeiriou
    * Abstract: This manuscript presents the results of the "A View Synthesis Challenge for Humans Heads (VSCHH)", which was part of the ICCV 2023 workshops. This paper describes the competition setup and provides details on replicating our initial baseline, TensoRF. Additionally, we provide a summary of the participants' methods and their results in our benchmark table. The challenge aimed to synthesize novel camera views of human heads using a given set of sparse training view images. The proposed solutions of the participants were evaluated and ranked based on objective fidelity metrics, such as PSNR and SSIM, computed against unseen validation and test sets. In the supplementary material, we detailed the methods used by all participants in the VSCHH challenge, which opened on May 15th, 2023, and concluded on July 24th, 2023.

count=1
* Single-Stage Joint Face Detection and Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/WIDER/Deng_Single-Stage_Joint_Face_Detection_and_Alignment_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/WIDER/Deng_Single-Stage_Joint_Face_Detection_and_Alignment_ICCVW_2019_paper.pdf)]
    * Title: Single-Stage Joint Face Detection and Alignment
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Jiankang Deng, Jia Guo, Stefanos Zafeiriou
    * Abstract: In practice, there are huge demands to localize faces in images and videos under unconstrained pose variation, illumination change, severe occlusion and low resolution, which pose a great challenge to existing face detectors. This challenge report presents a single-stage joint face detection and alignment method. In detail, we employ feature pyramid network, single-stage detection, context modelling, multi-task learning and cascade regression to construct a practical face detector. On the Wider Face Hard validation subset, our single model achieves state-of-the-art performance (92.0% AP) compared with both academic and commercial face detectors for detecting unconstrained faces in cluttered scenes. In the Wider Face AND PERSON CHALLENGE 2019, our ensemble model achieves 56.66% average AP (runner-up) in the face detection track. To facilitate further research on the topic, the training code and models have been provided publicly available.

count=1
* Attention-based Fusion for Multi-source Human Image Generation
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Lathuiliere_Attention-based_Fusion_for_Multi-source_Human_Image_Generation_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Lathuiliere_Attention-based_Fusion_for_Multi-source_Human_Image_Generation_WACV_2020_paper.pdf)]
    * Title: Attention-based Fusion for Multi-source Human Image Generation
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Stephane Lathuiliere,  Enver Sangineto,  Aliaksandr Siarohin,  Nicu Sebe
    * Abstract: We present a generalization of the person-image generation task, in which a human image is generated conditioned on a target pose and a set X of source appearance images. In this way, we can exploit multiple, possibly complementary images of the same person which are usually available at training and at testing time. The solution we propose is mainly based on a local attention mechanism which selects relevant information from different source image regions, avoiding the necessity to build specific generators for each specific cardinality of X. The empirical evaluation of our method shows the practical interest of addressing the person-image generation problem in a multi-source setting.

count=1
* AlignNet: A Unifying Approach to Audio-Visual Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Wang_AlignNet_A_Unifying_Approach_to_Audio-Visual_Alignment_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_AlignNet_A_Unifying_Approach_to_Audio-Visual_Alignment_WACV_2020_paper.pdf)]
    * Title: AlignNet: A Unifying Approach to Audio-Visual Alignment
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Jianren Wang,  Zhaoyuan Fang,  Hang Zhao
    * Abstract: We present AlignNet, a model that synchronizes videos with reference audios under non-uniform and irregular misalignments. AlignNet learns the end-to-end dense correspondence between each frame of a video and an audio. Our method is designed according to simple and well-established principles: attention, pyramidal processing, warping, and affinity function. Together with the model, we release a dancing dataset Dance50 for training and evaluation. Qualitative, quantitative and subjective evaluation results on dance-music alignment and speech-lip alignment demonstrate that our method far outperforms the state-of-the-art methods. Code, dataset and sample videos are available at our project page.

count=1
* Single Image Reflection Removal With Edge Guidance, Reflection Classifier, and Recurrent Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Chang_Single_Image_Reflection_Removal_With_Edge_Guidance_Reflection_Classifier_and_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Chang_Single_Image_Reflection_Removal_With_Edge_Guidance_Reflection_Classifier_and_WACV_2021_paper.pdf)]
    * Title: Single Image Reflection Removal With Edge Guidance, Reflection Classifier, and Recurrent Decomposition
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Ya-Chu Chang, Chia-Ni Lu, Chia-Chi Cheng, Wei-Chen Chiu
    * Abstract: Removing undesired reflection from an image captured through a glass window is a notable task in computer vision. In this paper, we propose a novel model with auxiliary techniques to tackle the problem of single image reflection removal. Our model takes a reflection contaminated image as input, and decomposes it into the reflection layer and the transmission layer. In order to ensure quality of the transmission layer, we introduce three auxiliary techniques into our architecture, including the edge guidance, a reflection classifier, and the recurrent decomposition. The contributions and the efficacy of these techniques are investigated and verified in the ablation study. Furthermore, in comparison to the state-of-the-art baselines of reflection removal, both quantitative and qualitative results demonstrate that our proposed method is able to deal with different kinds of images, achieving the best results in average.

count=1
* DualSR: Zero-Shot Dual Learning for Real-World Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Emad_DualSR_Zero-Shot_Dual_Learning_for_Real-World_Super-Resolution_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Emad_DualSR_Zero-Shot_Dual_Learning_for_Real-World_Super-Resolution_WACV_2021_paper.pdf)]
    * Title: DualSR: Zero-Shot Dual Learning for Real-World Super-Resolution
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Mohammad Emad, Maurice Peemen, Henk Corporaal
    * Abstract: Advanced methods for single image super-resolution (SISR) based upon Deep learning have demonstrated a remarkable reconstruction performance on downscaled images. However, for real-world low-resolution images (e.g. images captured straight from the camera) they often generate blurry images and highlight unpleasant artifacts. The main reason is the training data that does not reflect the real-world super-resolution problem. They train the network using images downsampled with an ideal (usually bicubic) kernel. However, for real-world images the degradation process is more complex and can vary from image to image. This paper proposes a new dual-path architecture (DualSR) that learns an image-specific low-to-high resolution mapping using only patches of the input test image. For every image, a downsampler learns the degradation process using a generative adversarial network, and an upsampler learns to super-resolve that specific image. In the DualSR architecture, the upsampler and downsampler are trained simultaneously and they improve each other using cycle consistency losses. For better visual quality and eliminating undesired artifacts, the upsampler is constrained by a masked interpolation loss. On standard benchmarks with unknown degradation kernels, DualSR outperforms recent blind and non-blind super-resolution methods in term of SSIM and generates images with higher perceptual quality. On real-world LR images it generates visually pleasing and artifact-free results.

count=1
* The Devil Is in the Boundary: Exploiting Boundary Representation for Basis-Based Instance Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Kim_The_Devil_Is_in_the_Boundary_Exploiting_Boundary_Representation_for_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Kim_The_Devil_Is_in_the_Boundary_Exploiting_Boundary_Representation_for_WACV_2021_paper.pdf)]
    * Title: The Devil Is in the Boundary: Exploiting Boundary Representation for Basis-Based Instance Segmentation
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Myungchul Kim, Sanghyun Woo, Dahun Kim, In So Kweon
    * Abstract: Pursuing a more coherent scene understanding towards real-time vision applications, single-stage instance segmentation has recently gained popularity, achieving a simpler and more efficient design than its two-stage counterparts. Besides, its global mask representation often leads to superior accuracy to the two-stage Mask R-CNN which has been dominant thus far. Despite the promising advances in single-stage methods, finer delineation of instance boundaries still remains unexcavated. Indeed, boundary information provides a strong shape representation that can operate in synergy with the fully-convolutional mask features of the single-stage segmented. In this work, we propose Boundary Basis based Instance Segmentation(B2Inst) to learn a global boundary representation that can complement existing global-mask-based methods that are often lacking high-frequency details. Besides, we devise a unified quality measure of both mask and boundary and introduce a network block that learns to score the per-instance predictions of itself. When applied to the strongest baselines in single-stage instance segmentation, our B2Inst leads to consistent improvements and accurately parse out the instance boundaries in a scene. Regardless of being single-stage or two-stage frameworks, we outperform the existing state-of-the-art methods on the COCO dataset with the same ResNet-50 and ResNet-101 backbones.

count=1
* SMPLpix: Neural Avatars From 3D Human Models
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Prokudin_SMPLpix_Neural_Avatars_From_3D_Human_Models_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Prokudin_SMPLpix_Neural_Avatars_From_3D_Human_Models_WACV_2021_paper.pdf)]
    * Title: SMPLpix: Neural Avatars From 3D Human Models
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Sergey Prokudin, Michael J. Black, Javier Romero
    * Abstract: Recent advances in deep generative models have led to an unprecedented level of realism for synthetically generated images of humans. However, one of the remaining fundamental limitations of these models is the ability to flexibly control the generative process, e.g. change the camera and human pose while retaining the subject identity. At the same time, deformable human body models like SMPL and its successors provide full control over pose and shape but rely on classic computer graphics pipelines for rendering. Such rendering pipelines require explicit mesh rasterization that (a) does not have the potential to fix artifacts or lack of realism in the original 3D geometry and (b) until recently, were not fully incorporated into deep learning frameworks. In this work, we propose to bridge the gap between classic geometry-based rendering and the latest generative networks operating in pixel space. We train a network that directly converts a sparse set of 3D mesh vertices into photorealistic images, alleviating the need for traditional rasterization mechanism. We train our model on a large corpus of human 3D models and corresponding real photos, and show the advantage over conventional differentiable renderers both in terms of the level of photorealism and rendering efficiency.

count=1
* Neural Contrast Enhancement of CT Image
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Seo_Neural_Contrast_Enhancement_of_CT_Image_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Seo_Neural_Contrast_Enhancement_of_CT_Image_WACV_2021_paper.pdf)]
    * Title: Neural Contrast Enhancement of CT Image
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Minkyo Seo, Dongkeun Kim, Kyungmoon Lee, Seunghoon Hong, Jae Seok Bae, Jung Hoon Kim, Suha Kwak
    * Abstract: Contrast materials are often injected into body to contrast specific tissues in Computed Tomography (CT) images. Contrast Enhanced CT (CECT) images obtained in this way are more useful than Non-Enhanced CT (NECT) images for medical diagnosis, but not available for everyone due to side effects of the contrast materials. Motivated by this, we develop a neural network that takes NECT images and generates their CECT counterparts. Learning such a network is extremely challenging since NECT and CECT images for training are not aligned even at the same location of the same patient due to movements of internal organs. We propose a two-stage framework to address this issue. The first stage trains an auxiliary network that removes the effect of contrast enhancement in CECT images to synthesize their NECT counterparts well-aligned with them. In the second stage, the target model is trained to predict the real CECTimages given a synthetic NECT image as input. Experimental results and analysis by physicians on abdomen CT images suggest that our method outperforms existing models for neural image synthesis.

count=1
* SporeAgent: Reinforced Scene-Level Plausibility for Object Pose Refinement
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Bauer_SporeAgent_Reinforced_Scene-Level_Plausibility_for_Object_Pose_Refinement_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Bauer_SporeAgent_Reinforced_Scene-Level_Plausibility_for_Object_Pose_Refinement_WACV_2022_paper.pdf)]
    * Title: SporeAgent: Reinforced Scene-Level Plausibility for Object Pose Refinement
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Dominik Bauer, Timothy Patten, Markus Vincze
    * Abstract: Observational noise, inaccurate segmentation and ambiguity due to symmetry and occlusion lead to inaccurate object pose estimates. While depth- and RGB-based pose refinement approaches increase the accuracy of the resulting pose estimates, they are susceptible to ambiguity in the observation as they consider visual alignment. We propose to leverage the fact that we often observe static, rigid scenes. Thus, the objects therein need to be under physically plausible poses. We show that considering plausibility reduces ambiguity and, in consequence, allows poses to be more accurately predicted in cluttered environments. To this end, we extend a recent RL-based registration approach towards iterative refinement of object poses. Experiments on the LINEMOD and YCB-VIDEO datasets demonstrate the state-of-the-art performance of our depth-based refinement approach.

count=1
* To Miss-Attend Is to Misalign! Residual Self-Attentive Feature Alignment for Adapting Object Detectors
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Khindkar_To_Miss-Attend_Is_to_Misalign_Residual_Self-Attentive_Feature_Alignment_for_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Khindkar_To_Miss-Attend_Is_to_Misalign_Residual_Self-Attentive_Feature_Alignment_for_WACV_2022_paper.pdf)]
    * Title: To Miss-Attend Is to Misalign! Residual Self-Attentive Feature Alignment for Adapting Object Detectors
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Vaishnavi Khindkar, Chetan Arora, Vineeth N Balasubramanian, Anbumani Subramanian, Rohit Saluja, C.V. Jawahar
    * Abstract: Advancements in adaptive object detection can lead to tremendous improvements in applications like autonomous navigation, as they alleviate the distributional shifts along the detection pipeline. Prior works adopt adversarial learning to align image features at global and local levels, yet the instance-specific misalignment persists. Also, adaptive object detection remains challenging due to visual diversity in background scenes and intricate combinations of objects. Motivated by structural importance, we aim to attend prominent instance-specific regions, overcoming the feature misalignment issue. We propose a novel resIduaL seLf-attentive featUre alignMEnt ( ILLUME ) method for adaptive object detection. ILLUME comprises Self-Attention Feature Map (SAFM) module that enhances structural attention to object-related regions and thereby generates domain invariant features. Our approach significantly reduces the domain distance with the improved feature alignment of the instances. Qualitative results demonstrate the ability of ILLUME to attend important object instances required for alignment. Experimental results on several benchmark datasets show that our method outperforms the existing state-of-the-art approaches.

count=1
* Single-Photon Camera Guided Extreme Dynamic Range Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Liu_Single-Photon_Camera_Guided_Extreme_Dynamic_Range_Imaging_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Liu_Single-Photon_Camera_Guided_Extreme_Dynamic_Range_Imaging_WACV_2022_paper.pdf)]
    * Title: Single-Photon Camera Guided Extreme Dynamic Range Imaging
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Yuhao Liu, Felipe Gutierrez-Barragan, Atul Ingle, Mohit Gupta, Andreas Velten
    * Abstract: Reconstruction of high-resolution extreme dynamic range images from a small number of low dynamic range (LDR) images is crucial for many computer vision applications. Current high dynamic range (HDR) cameras based on CMOS image sensor technology rely on multiexposure bracketing which suffers from motion artifacts and signal-to-noise (SNR) dip artifacts in extreme dynamic range scenes. Recently, single-photon cameras (SPCs) have been shown to achieve orders of magnitude higher dynamic range for passive imaging than conventional CMOS sensors. SPCs are becoming increasingly available commercially, even in some consumer devices. Unfortunately, current SPCs suffer from low spatial resolution. To overcome the limitations of CMOS and SPC sensors, we propose a learning-based CMOS-SPC fusion method to recover high-resolution extreme dynamic range images. We compare the performance of our method against various traditional and state-of-the-art baselines using both synthetic and experimental data. Our method outperforms these baselines, both in terms of visual quality and quantitative metrics.

count=1
* Feature Disentanglement Learning With Switching and Aggregation for Video-Based Person Re-Identification
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Kim_Feature_Disentanglement_Learning_With_Switching_and_Aggregation_for_Video-Based_Person_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Kim_Feature_Disentanglement_Learning_With_Switching_and_Aggregation_for_Video-Based_Person_WACV_2023_paper.pdf)]
    * Title: Feature Disentanglement Learning With Switching and Aggregation for Video-Based Person Re-Identification
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Minjung Kim, MyeongAh Cho, Sangyoun Lee
    * Abstract: In video person re-identification (Re-ID), the network must consistently extract features of the target person from successive frames. Existing methods tend to focus only on how to use temporal information, which often leads to networks being fooled by similar appearances and same backgrounds. In this paper, we propose a Disentanglement and Switching and Aggregation Network (DSANet), which segregates the features representing identity and features based on camera characteristics, and pays more attention to ID information. We also introduce an auxiliary task that utilizes a new pair of features created through switching and aggregation to increase the network's capability for various camera scenarios. Furthermore, we devise a Target Localization Module (TLM) that extracts robust features against a change in the position of the target according to the frame flow and a Frame Weight Generation (FWG) that reflects temporal information in the final representation. Various loss functions for disentanglement learning are designed so that each component of the network can cooperate while satisfactorily performing its own role. Quantitative and qualitative results from extensive experiments demonstrate the superiority of DSANet over state-of-the-art methods on three benchmark datasets.

count=1
* Camera Alignment and Weighted Contrastive Learning for Domain Adaptation in Video Person ReID
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Mekhazni_Camera_Alignment_and_Weighted_Contrastive_Learning_for_Domain_Adaptation_in_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Mekhazni_Camera_Alignment_and_Weighted_Contrastive_Learning_for_Domain_Adaptation_in_WACV_2023_paper.pdf)]
    * Title: Camera Alignment and Weighted Contrastive Learning for Domain Adaptation in Video Person ReID
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Djebril Mekhazni, Maximilien Dufau, Christian Desrosiers, Marco Pedersoli, Eric Granger
    * Abstract: Systems for person re-identification (ReID) can achieve a high level of accuracy when trained on large fully-labeled image datasets. However, the domain shift typically associated with diverse operational capture conditions (e.g., camera viewpoints and lighting) may translate to a significant decline in performance. This paper focuses on unsupervised domain adaptation (UDA) for video-based ReID -- a relevant scenario that is less explored in the literature. In this scenario, the ReID model must adapt to a complex target domain defined by a network of diverse video cameras based on tracklet information. State-of-art methods cluster unlabeled target data, yet domain shifts across target cameras (sub-domains) can lead to poor initialization of clustering methods that propagates noise across epochs, and the ReID model cannot accurately associate samples of the same identity. In this paper, an UDA method is introduced for video person ReID that leverages knowledge on video tracklets, and on the distribution of frames captured over target cameras to improve the performance of CNN backbones trained using pseudo-labels. Our method relies on an adversarial approach, where a camera-discriminator network is introduced to extract discriminant camera-independent representations, facilitating the subsequent clustering. In addition, a weighted contrastive loss is proposed to leverage the confidence of clusters, and mitigate the risk of incorrect identity associations. Experimental results obtained on three challenging video-based person ReID datasets -- PRID2011, iLIDS-VID, and MARS -- indicate that our proposed method can outperform related state-of-the-art methods. The code is available at: https://github.com/wacv23775/775.

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
* Closer Look at the Transferability of Adversarial Examples: How They Fool Different Models Differently
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Waseda_Closer_Look_at_the_Transferability_of_Adversarial_Examples_How_They_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Waseda_Closer_Look_at_the_Transferability_of_Adversarial_Examples_How_They_WACV_2023_paper.pdf)]
    * Title: Closer Look at the Transferability of Adversarial Examples: How They Fool Different Models Differently
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Futa Waseda, Sosuke Nishikawa, Trung-Nghia Le, Huy H. Nguyen, Isao Echizen
    * Abstract: Deep neural networks are vulnerable to adversarial examples (AEs), which have adversarial transferability: AEs generated for the source model can mislead another (target) model's predictions. However, the transferability has not been understood in terms of to which class target model's predictions were misled (i.e., class-aware transferability). In this paper, we differentiate the cases in which a target model predicts the same wrong class as the source model ("same mistake") or a different wrong class ("different mistake") to analyze and provide an explanation of the mechanism. We find that (1) AEs tend to cause same mistakes, which correlates with "non-targeted transferability"; however, (2) different mistakes occur even between similar models, regardless of the perturbation size. Furthermore, we present evidence that the difference between same mistakes and different mistakes can be explained by non-robust features, predictive but human-uninterpretable patterns: different mistakes occur when non-robust features in AEs are used differently by models. Non-robust features can thus provide consistent explanations for the class-aware transferability of AEs.

count=1
* From Chaos to Calibration: A Geometric Mutual Information Approach To Target-Free Camera LiDAR Extrinsic Calibration
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Borer_From_Chaos_to_Calibration_A_Geometric_Mutual_Information_Approach_To_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Borer_From_Chaos_to_Calibration_A_Geometric_Mutual_Information_Approach_To_WACV_2024_paper.pdf)]
    * Title: From Chaos to Calibration: A Geometric Mutual Information Approach To Target-Free Camera LiDAR Extrinsic Calibration
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Jack Borer, Jeremy Tschirner, Florian Ölsner, Stefan Milz
    * Abstract: Sensor fusion is vital for the safe and robust operation of autonomous vehicles. Accurate extrinsic sensor to sensor calibration is necessary to accurately fuse multiple sensor's data in a common spatial reference frame. In this paper, we propose a target free extrinsic calibration algorithm that requires no ground truth training data, artificially constrained motion trajectories, hand engineered features or offline optimization and that is accurate, precise and extremely robust to initialization error. Most current research on online camera-LiDAR extrinsic calibration requires ground truth training data which is impossible to capture at scale. We revisit analytical mutual information based methods first proposed in 2012 and demonstrate that geometric features provide a robust information metric for camera-LiDAR extrinsic calibration. We demonstrate our proposed improvement using the KITTI and KITTI-360 fisheye data set.

count=1
* ReCLIP: Refine Contrastive Language Image Pre-Training With Source Free Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Hu_ReCLIP_Refine_Contrastive_Language_Image_Pre-Training_With_Source_Free_Domain_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Hu_ReCLIP_Refine_Contrastive_Language_Image_Pre-Training_With_Source_Free_Domain_WACV_2024_paper.pdf)]
    * Title: ReCLIP: Refine Contrastive Language Image Pre-Training With Source Free Domain Adaptation
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Xuefeng Hu, Ke Zhang, Lu Xia, Albert Chen, Jiajia Luo, Yuyin Sun, Ken Wang, Nan Qiao, Xiao Zeng, Min Sun, Cheng-Hao Kuo, Ram Nevatia
    * Abstract: Large-scale pre-training vision-language models (VLM) such as CLIP has demonstrated outstanding performance in zero-shot classification, e.g. achieving 76.3% top-1 accuracy on ImageNet without seeing any example, which leads to potential benefits to many tasks that have no labeled data. However, while applying CLIP to a downstream target domain, the presence of visual and text domain gaps and cross-modality misalignment can greatly impact the model performance. To address such challenges, we propose ReCLIP, a novel source-free domain adaptation method for vision-language models, which does not require any source data or target labeled data. ReCLIP first learns a projection space to mitigate the misaligned visual-text embeddings and learns pseudo labels, and then deploys cross-modality self-training with the pseudo labels, to update visual and text encoders, refine labels and reduce domain gaps and misalignment iteratively. With extensive experiments, we demonstrate that ReCLIP outperforms all the baselines with significant margin and improves the averaged accuracy of CLIP from 69.83% to 74.94% on 22 image classification benchmarks.

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
* ArcGeo: Localizing Limited Field-of-View Images Using Cross-View Matching
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Shugaev_ArcGeo_Localizing_Limited_Field-of-View_Images_Using_Cross-View_Matching_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Shugaev_ArcGeo_Localizing_Limited_Field-of-View_Images_Using_Cross-View_Matching_WACV_2024_paper.pdf)]
    * Title: ArcGeo: Localizing Limited Field-of-View Images Using Cross-View Matching
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Maxim Shugaev, Ilya Semenov, Kyle Ashley, Michael Klaczynski, Naresh Cuntoor, Mun Wai Lee, Nathan Jacobs
    * Abstract: Cross-view matching techniques for image geolocalization attempt to match features in ground level imagery against a collection of satellite images to determine the position of given query image. We present a novel cross-view image matching approach called ArcGeo which introduces a batch-all angular margin loss and several train-time strategies including large-scale pretraining and FoV-based data augmentation. This allows our model to perform well even in challenging cases with limited field-of-view (FoV). Further, we evaluate multiple model architectures, data augmentation approaches and optimization strategies to train a deep cross-view matching network, specifically optimized for limited FoV cases. In low FoV experiments (FoV = 90deg) our method improves top-1 image recall rate on the CVUSA dataset from 30.12% to 43.08%. We also demonstrate improved performance over the state-of-the-art techniques for panoramic cross-view retrieval, improving top-1 recall from 95.43% to 96.06% on the CVUSA dataset and from 64.52% to 79.88% on the CVACT test dataset. Lastly, we evaluate the role of large-scale pretraining for improved robustness. With appropriate pretraining on external data, our model improves top-1 recall dramatically to 66.83% for FoV = 90deg test case on CVUSA, an increase of over twice what is reported by existing approaches.

count=1
* Learning invariant representations and applications to face verification
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2013/hash/ad3019b856147c17e82a5bead782d2a8-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2013/file/ad3019b856147c17e82a5bead782d2a8-Paper.pdf)]
    * Title: Learning invariant representations and applications to face verification
    * Publisher: NeurIPS
    * Publication Date: `2013`
    * Authors: Qianli Liao, Joel Z. Leibo, Tomaso Poggio
    * Abstract: One approach to computer object recognition and modeling the brain's ventral stream involves unsupervised learning of representations that are invariant to common transformations. However, applications of these ideas have usually been limited to 2D affine transformations, e.g., translation and scaling, since they are easiest to solve via convolution. In accord with a recent theory of transformation-invariance, we propose a model that, while capturing other common convolutional networks as special cases, can also be used with arbitrary identity-preserving transformations. The model's wiring can be learned from videos of transforming objects---or any other grouping of images into sets by their depicted object. Through a series of successively more complex empirical tests, we study the invariance/discriminability properties of this model with respect to different transformations. First, we empirically confirm theoretical predictions for the case of 2D affine transformations. Next, we apply the model to non-affine transformations: as expected, it performs well on face verification tasks requiring invariance to the relatively smooth transformations of 3D rotation-in-depth and changes in illumination direction. Surprisingly, it can also tolerate clutter transformations'' which map an image of a face on one background to an image of the same face on a different background. Motivated by these empirical findings, we tested the same model on face verification benchmark tasks from the computer vision literature: Labeled Faces in the Wild, PubFig and a new dataset we gathered---achieving strong performance in these highly unconstrained cases as well.

count=1
* Scale Adaptive Blind Deblurring
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2014/hash/facf9f743b083008a894eee7baa16469-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2014/file/facf9f743b083008a894eee7baa16469-Paper.pdf)]
    * Title: Scale Adaptive Blind Deblurring
    * Publisher: NeurIPS
    * Publication Date: `2014`
    * Authors: Haichao Zhang, Jianchao Yang
    * Abstract: The presence of noise and small scale structures usually leads to large kernel estimation errors in blind image deblurring empirically, if not a total failure. We present a scale space perspective on blind deblurring algorithms, and introduce a cascaded scale space formulation for blind deblurring. This new formulation suggests a natural approach robust to noise and small scale structures through tying the estimation across multiple scales and balancing the contributions of different scales automatically by learning from data. The proposed formulation also allows to handle non-uniform blur with a straightforward extension. Experiments are conducted on both benchmark dataset and real-world images to validate the effectiveness of the proposed method. One surprising finding based on our approach is that blur kernel estimation is not necessarily best at the finest scale.

count=1
* Your Classifier can Secretly Suffice Multi-Source Domain Adaptation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/3181d59d19e76e902666df5c7821259a-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/3181d59d19e76e902666df5c7821259a-Paper.pdf)]
    * Title: Your Classifier can Secretly Suffice Multi-Source Domain Adaptation
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Naveen Venkat, Jogendra Nath Kundu, Durgesh Singh, Ambareesh Revanur, Venkatesh Babu R
    * Abstract: Multi-Source Domain Adaptation (MSDA) deals with the transfer of task knowledge from multiple labeled source domains to an unlabeled target domain, under a domain-shift. Existing methods aim to minimize this domain-shift using auxiliary distribution alignment objectives. In this work, we present a different perspective to MSDA wherein deep models are observed to implicitly align the domains under label supervision. Thus, we aim to utilize implicit alignment without additional training objectives to perform adaptation. To this end, we use pseudo-labeled target samples and enforce a classifier agreement on the pseudo-labels, a process called Self-supervised Implicit Alignment (SImpAl). We find that SImpAl readily works even under category-shift among the source domains. Further, we propose classifier agreement as a cue to determine the training convergence, resulting in a simple training algorithm. We provide a thorough evaluation of our approach on five benchmarks, along with detailed insights into each component of our approach.

count=1
* Self-Learning Transformations for Improving Gaze and Head Redirection
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/98f2d76d4d9caf408180b5abfa83ae87-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/98f2d76d4d9caf408180b5abfa83ae87-Paper.pdf)]
    * Title: Self-Learning Transformations for Improving Gaze and Head Redirection
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Yufeng Zheng, Seonwook Park, Xucong Zhang, Shalini De Mello, Otmar Hilliges
    * Abstract: Many computer vision tasks rely on labeled data. Rapid progress in generative modeling has led to the ability to synthesize photorealistic images. However, controlling specific aspects of the generation process such that the data can be used for supervision of downstream tasks remains challenging. In this paper we propose a novel generative model for images of faces, that is capable of producing high-quality images under fine-grained control over eye gaze and head orientation angles. This requires the disentangling of many appearance related factors including gaze and head orientation but also lighting, hue etc. We propose a novel architecture which learns to discover, disentangle and encode these extraneous variations in a self-learned manner. We further show that explicitly disentangling task-irrelevant factors results in more accurate modelling of gaze and head orientation. A novel evaluation scheme shows that our method improves upon the state-of-the-art in redirection accuracy and disentanglement between gaze direction and head orientation changes. Furthermore, we show that in the presence of limited amounts of real-world training data, our method allows for improvements in the downstream task of semi-supervised cross-dataset gaze estimation. Please check our project page at: https://ait.ethz.ch/projects/2020/STED-gaze/

count=1
* Consequences of Misaligned AI
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/b607ba543ad05417b8507ee86c54fcb7-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/b607ba543ad05417b8507ee86c54fcb7-Paper.pdf)]
    * Title: Consequences of Misaligned AI
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Simon Zhuang, Dylan Hadfield-Menell
    * Abstract: AI systems often rely on two key components: a specified goal or reward function and an optimization algorithm to compute the optimal behavior for that goal. This approach is intended to provide value for a principal: the user on whose behalf the agent acts. The objectives given to these agents often refer to a partial specification of the principal's goals. We consider the cost of this incompleteness by analyzing a model of a principal and an agent in a resource constrained world where the L features of the state correspond to different sources of utility for the principal. We assume that the reward function given to the agent only has support on J < L features. The contributions of our paper are as follows: 1) we propose a novel model of an incomplete principal—agent problem from artificial intelligence; 2) we provide necessary and sufficient conditions under which indefinitely optimizing for any incomplete proxy objective leads to arbitrarily low overall utility; and 3) we show how modifying the setup to allow reward functions that reference the full state or allowing the principal to update the proxy objective over time can lead to higher utility solutions. The results in this paper argue that we should view the design of reward functions as an interactive and dynamic process and identifies a theoretical scenario where some degree of interactivity is desirable.

count=1
* Non-local Latent Relation Distillation for Self-Adaptive 3D Human Pose Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/018b59ce1fd616d874afad0f44ba338d-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/018b59ce1fd616d874afad0f44ba338d-Paper.pdf)]
    * Title: Non-local Latent Relation Distillation for Self-Adaptive 3D Human Pose Estimation
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Jogendra Nath Kundu, Siddharth Seth, Anirudh Jamkhandi, Pradyumna YM, Varun Jampani, Anirban Chakraborty, Venkatesh Babu R
    * Abstract: Available 3D human pose estimation approaches leverage different forms of strong (2D/3D pose) or weak (multi-view or depth) paired supervision. Barring synthetic or in-studio domains, acquiring such supervision for each new target environment is highly inconvenient. To this end, we cast 3D pose learning as a self-supervised adaptation problem that aims to transfer the task knowledge from a labeled source domain to a completely unpaired target. We propose to infer image-to-pose via two explicit mappings viz. image-to-latent and latent-to-pose where the latter is a pre-learned decoder obtained from a prior-enforcing generative adversarial auto-encoder. Next, we introduce relation distillation as a means to align the unpaired cross-modal samples i.e., the unpaired target videos and unpaired 3D pose sequences. To this end, we propose a new set of non-local relations in order to characterize long-range latent pose interactions, unlike general contrastive relations where positive couplings are limited to a local neighborhood structure. Further, we provide an objective way to quantify non-localness in order to select the most effective relation set. We evaluate different self-adaptation settings and demonstrate state-of-the-art 3D human pose estimation performance on standard benchmarks.

count=1
* Breaking the Dilemma of Medical Image-to-image Translation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/0f2818101a7ac4b96ceeba38de4b934c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/0f2818101a7ac4b96ceeba38de4b934c-Paper.pdf)]
    * Title: Breaking the Dilemma of Medical Image-to-image Translation
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Lingke Kong, Chenyu Lian, Detian Huang, zhenjiang li, Yanle Hu, Qichao Zhou
    * Abstract: Supervised Pix2Pix and unsupervised Cycle-consistency are two modes that dominate the field of medical image-to-image translation. However, neither modes are ideal. The Pix2Pix mode has excellent performance. But it requires paired and well pixel-wise aligned images, which may not always be achievable due to respiratory motion or anatomy change between times that paired images are acquired. The Cycle-consistency mode is less stringent with training data and works well on unpaired or misaligned images. But its performance may not be optimal. In order to break the dilemma of the existing modes, we propose a new unsupervised mode called RegGAN for medical image-to-image translation. It is based on the theory of "loss-correction". In RegGAN, the misaligned target images are considered as noisy labels and the generator is trained with an additional registration network to fit the misaligned noise distribution adaptively. The goal is to search for the common optimal solution to both image-to-image translation and registration tasks. We incorporated RegGAN into a few state-of-the-art image-to-image translation methods and demonstrated that RegGAN could be easily combined with these methods to improve their performances. Such as a simple CycleGAN in our mode surpasses latest NICEGAN even though using less network parameters. Based on our results, RegGAN outperformed both Pix2Pix on aligned data and Cycle-consistency on misaligned or unpaired data. RegGAN is insensitive to noises which makes it a better choice for a wide range of scenarios, especially for medical image-to-image translation tasks in which well pixel-wise aligned data are not available. Code and dataset are available at https://github.com/Kid-Liet/Reg-GAN.

count=1
* TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/0f49c89d1e7298bb9930789c8ed59d48-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/0f49c89d1e7298bb9930789c8ed59d48-Paper.pdf)]
    * Title: TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Shengcai Liao, Ling Shao
    * Abstract: Transformers have recently gained increasing attention in computer vision. However, existing studies mostly use Transformers for feature representation learning, e.g. for image classification and dense predictions, and the generalizability of Transformers is unknown. In this work, we further investigate the possibility of applying Transformers for image matching and metric learning given pairs of images. We find that the Vision Transformer (ViT) and the vanilla Transformer with decoders are not adequate for image matching due to their lack of image-to-image attention. Thus, we further design two naive solutions, i.e. query-gallery concatenation in ViT, and query-gallery cross-attention in the vanilla Transformer. The latter improves the performance, but it is still limited. This implies that the attention mechanism in Transformers is primarily designed for global feature aggregation, which is not naturally suitable for image matching. Accordingly, we propose a new simplified decoder, which drops the full attention implementation with the softmax weighting, keeping only the query-key similarity computation. Additionally, global max pooling and a multilayer perceptron (MLP) head are applied to decode the matching result. This way, the simplified decoder is computationally more efficient, while at the same time more effective for image matching. The proposed method, called TransMatcher, achieves state-of-the-art performance in generalizable person re-identification, with up to 6.1% and 5.7% performance gains in Rank-1 and mAP, respectively, on several popular datasets. Code is available at https://github.com/ShengcaiLiao/QAConv.

count=1
* Federated Graph Classification over Non-IID Graphs
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/9c6947bd95ae487c81d4e19d3ed8cd6f-Paper.pdf)]
    * Title: Federated Graph Classification over Non-IID Graphs
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Han Xie, Jing Ma, Li Xiong, Carl Yang
    * Abstract: Federated learning has emerged as an important paradigm for training machine learning models in different domains. For graph-level tasks such as graph classification, graphs can also be regarded as a special type of data samples, which can be collected and stored in separate local systems. Similar to other domains, multiple local systems, each holding a small set of graphs, may benefit from collaboratively training a powerful graph mining model, such as the popular graph neural networks (GNNs). To provide more motivation towards such endeavors, we analyze real-world graphs from different domains to confirm that they indeed share certain graph properties that are statistically significant compared with random graphs. However, we also find that different sets of graphs, even from the same domain or same dataset, are non-IID regarding both graph structures and node features. To handle this, we propose a graph clustered federated learning (GCFL) framework that dynamically finds clusters of local systems based on the gradients of GNNs, and theoretically justify that such clusters can reduce the structure and feature heterogeneity among graphs owned by the local systems. Moreover, we observe the gradients of GNNs to be rather fluctuating in GCFL which impedes high-quality clustering, and design a gradient sequence-based clustering mechanism based on dynamic time warping (GCFL+). Extensive experimental results and in-depth analysis demonstrate the effectiveness of our proposed frameworks.

count=1
* Environment Diversification with Multi-head Neural Network for Invariant Learning
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/062d711fb777322e2152435459e6e9d9-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/062d711fb777322e2152435459e6e9d9-Paper-Conference.pdf)]
    * Title: Environment Diversification with Multi-head Neural Network for Invariant Learning
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Bo-Wei Huang, Keng-Te Liao, Chang-Sheng Kao, Shou-De Lin
    * Abstract: Neural networks are often trained with empirical risk minimization; however, it has been shown that a shift between training and testing distributions can cause unpredictable performance degradation. On this issue, a research direction, invariant learning, has been proposed to extract causal features insensitive to the distributional changes. This work proposes EDNIL, an invariant learning framework containing a multi-head neural network to absorb data biases. We show that this framework does not require prior knowledge about environments or strong assumptions about the pre-trained model. We also reveal that the proposed algorithm has theoretical connections to recent studies discussing properties of variant and invariant features. Finally, we demonstrate that models trained with EDNIL are empirically more robust against distributional shifts.

count=1
* UniCLIP: Unified Framework for Contrastive Language-Image Pre-training
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/072fd0525592b43da661e254bbaadc27-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/072fd0525592b43da661e254bbaadc27-Paper-Conference.pdf)]
    * Title: UniCLIP: Unified Framework for Contrastive Language-Image Pre-training
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Janghyeon Lee, Jongsuk Kim, Hyounguk Shon, Bumsoo Kim, Seung Hwan Kim, Honglak Lee, Junmo Kim
    * Abstract: Pre-training vision-language models with contrastive objectives has shown promising results that are both scalable to large uncurated datasets and transferable to many downstream applications. Some following works have targeted to improve data efficiency by adding self-supervision terms, but inter-domain (image-text) contrastive loss and intra-domain (image-image) contrastive loss are defined on individual spaces in those works, so many feasible combinations of supervision are overlooked. To overcome this issue, we propose UniCLIP, a Unified framework for Contrastive Language-Image Pre-training. UniCLIP integrates the contrastive loss of both inter-domain pairs and intra-domain pairs into a single universal space. The discrepancies that occur when integrating contrastive loss between different domains are resolved by the three key components of UniCLIP: (1) augmentation-aware feature embedding, (2) MP-NCE loss, and (3) domain dependent similarity measure. UniCLIP outperforms previous vision-language pre-training methods on various single- and multi-modality downstream tasks. In our experiments, we show that each component that comprises UniCLIP contributes well to the final performance.

count=1
* Retaining Knowledge for Learning with Dynamic Definition
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/5fcd540792da599adf1b932624e98f1f-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/5fcd540792da599adf1b932624e98f1f-Paper-Conference.pdf)]
    * Title: Retaining Knowledge for Learning with Dynamic Definition
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Zichang Liu, Benjamin Coleman, Tianyi Zhang, Anshumali Shrivastava
    * Abstract: Machine learning models are often deployed in settings where they must be constantly updated in response to the changes in class definitions while retaining high accuracy on previously learned definitions. A classical use case is fraud detection, where new fraud schemes come one after another. While such an update can be accomplished by re-training on the complete data, the process is inefficient and prevents real-time and on-device learning. On the other hand, efficient methods that incrementally learn from new data often result in the forgetting of previously-learned knowledge. We define this problem as Learning with Dynamic Definition (LDD) and demonstrate that popular models, such as the Vision Transformer and Roberta, exhibit substantial forgetting of past definitions. We present the first practical and provable solution to LDD. Our proposal is a hash-based sparsity model \textit{RIDDLE} that solves evolving definitions by associating samples only to relevant parameters. We prove that our model is a universal function approximator and theoretically bounds the knowledge lost during the update process. On practical tasks with evolving class definition in vision and natural language processing, \textit{RIDDLE} outperforms baselines by up to 30\% on the original dataset while providing competitive accuracy on the update dataset.

count=1
* AnimeRun: 2D Animation Visual Correspondence from Open Source 3D Movies
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/78b23d272f58fe3789ab490ebf080fa5-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/78b23d272f58fe3789ab490ebf080fa5-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: AnimeRun: 2D Animation Visual Correspondence from Open Source 3D Movies
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Li Siyao, Yuhang Li, Bo Li, Chao Dong, Ziwei Liu, Chen Change Loy
    * Abstract: Visual correspondence of 2D animation is the core of many applications and deserves careful study. Existing correspondence datasets for 2D cartoon suffer from simple frame composition and monotonic movements, making them insufficient to simulate real animations. In this work, we present a new 2D animation visual correspondence dataset, AnimeRun, by converting open source 3D movies to full scenes in 2D style, including simultaneous moving background and interactions of multiple subjects. Statistics show that our proposed dataset not only resembles real anime more in image composition, but also possesses richer and more complex motion patterns compared to existing datasets. With this dataset, we establish a comprehensive benchmark by evaluating several existing optical flow and segment matching methods, and analyze shortcomings of these methods on animation data. Data are available at https://lisiyao21.github.io/projects/AnimeRun.

count=1
* Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/aa56c74513a5e35768a11f4e82dd7ffb-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/aa56c74513a5e35768a11f4e82dd7ffb-Paper-Conference.pdf)]
    * Title: Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Randall Balestriero, Yann LeCun
    * Abstract: Self-Supervised Learning (SSL) surmises that inputs and pairwise positive relationships are enough to learn meaningful representations. Although SSL has recently reached a milestone: outperforming supervised methods in many modalities\dots the theoretical foundations are limited, method-specific, and fail to provide principled design guidelines to practitioners. In this paper, we propose a unifying framework under the helm of spectral manifold learning. Through the course of this study, we will demonstrate that VICReg, SimCLR, BarlowTwins et al. correspond to eponymous spectral methods such as Laplacian Eigenmaps, ISOMAP et al.From this unified viewpoint, we obtain (i) the close-form optimal representation, (ii) the close-form optimal network parameters in the linear regime, (iii) the impact of the pairwise relations used during training on each of those quantities and on downstream task performances, and most importantly, (iv) the first theoretical bridge between contrastive and non-contrastive methods to global and local spectral methods respectively hinting at the benefits and limitations of each. For example, if the pairwise relation is aligned with the downstream task, all SSL methods produce optimal representations for that downstream task.

count=1
* Rethinking Alignment in Video Super-Resolution Transformers
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/ea4d65c59073e8faf79222654d25fbe2-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/ea4d65c59073e8faf79222654d25fbe2-Paper-Conference.pdf)]
    * Title: Rethinking Alignment in Video Super-Resolution Transformers
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Shuwei Shi, Jinjin Gu, Liangbin Xie, Xintao Wang, Yujiu Yang, Chao Dong
    * Abstract: The alignment of adjacent frames is considered an essential operation in video super-resolution (VSR). Advanced VSR models, including the latest VSR Transformers, are generally equipped with well-designed alignment modules. However, the progress of the self-attention mechanism may violate this common sense. In this paper, we rethink the role of alignment in VSR Transformers and make several counter-intuitive observations. Our experiments show that: (i) VSR Transformers can directly utilize multi-frame information from unaligned videos, and (ii) existing alignment methods are sometimes harmful to VSR Transformers. These observations indicate that we can further improve the performance of VSR Transformers simply by removing the alignment module and adopting a larger attention window. Nevertheless, such designs will dramatically increase the computational burden, and cannot deal with large motions. Therefore, we propose a new and efficient alignment method called patch alignment, which aligns image patches instead of pixels. VSR Transformers equipped with patch alignment could demonstrate state-of-the-art performance on multiple benchmarks. Our work provides valuable insights on how multi-frame information is used in VSR and how to select alignment methods for different networks/datasets. Codes and models will be released at https://github.com/XPixelGroup/RethinkVSRAlignment.

count=1
* Scaling Multimodal Pre-Training via Cross-Modality Gradient Harmonization
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/eacad5b8e67850f2b8dd33d87691d097-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/eacad5b8e67850f2b8dd33d87691d097-Paper-Conference.pdf)]
    * Title: Scaling Multimodal Pre-Training via Cross-Modality Gradient Harmonization
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Junru Wu, Yi Liang, feng han, Hassan Akbari, Zhangyang Wang, Cong Yu
    * Abstract: Self-supervised pre-training recently demonstrates success on large-scale multimodal data, and state-of-the-art contrastive learning methods often enforce the feature consistency from cross-modality inputs, such as video/audio or video/text pairs. Despite its convenience to formulate and leverage in practice, such cross-modality alignment (CMA) is only a weak and noisy supervision, since two modalities can be semantically misaligned even they are temporally aligned. For example, even in the (often adopted) instructional videos, a speaker can sometimes refer to something that is not visually present in the current frame; and the semantic misalignment would only be more unpredictable for the raw videos collected from unconstrained internet sources. We conjecture that might cause conflicts and biases among modalities, and may hence prohibit CMA from scaling up to training with larger and more heterogeneous data. This paper first verifies our conjecture by observing that, even in the latest VATT pre-training using only narrated videos, there exist strong gradient conflicts between different CMA losses within the same sample triplet (video, audio, text), indicating them as the noisy source of supervision. We then propose to harmonize such gradients during pre-training, via two techniques: (i) cross-modality gradient realignment: modifying different CMA loss gradients for one sample triplet, so that their gradient directions are in more agreement; and (ii) gradient-based curriculum learning: leveraging the gradient conflict information on an indicator of sample noisiness, to develop a curriculum learning strategy to prioritize training with less noisy sample triplets. Applying those gradient harmonization techniques to pre-training VATT on the HowTo100M dataset, we consistently improve its performance on different downstream tasks. Moreover, we are able to scale VATT pre-training to more complicated non-narrative Youtube8M dataset to further improve the state-of-the-arts.

count=1
* Degraded Polygons Raise Fundamental Questions of Neural Network Perception
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/1ec408df112bc9b186d7b8fe0ada902a-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/1ec408df112bc9b186d7b8fe0ada902a-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Degraded Polygons Raise Fundamental Questions of Neural Network Perception
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Leonard Tang, Dan Ley
    * Abstract: It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deeplearning vision models suffer in a variety of settings that humans capably handle. Inlight of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images underdegradation, first introduced over 30 years ago in the Recognition-by-Componentstheory of human vision. Specifically, we study the performance and behavior ofneural networks on the seemingly simple task of classifying regular polygons atvarying orders of degradation along their perimeters. To this end, we implement theAutomated Shape Recoverability Testfor rapidly generating large-scale datasetsof perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity ofneural networks to recognize and recover such degraded shapes when initializedwith different priors. Ultimately, we find that neural networks’ behavior on thissimple task conflicts with human behavior, raising a fundamental question of therobustness and learning capabilities of modern computer vision models

count=1
* Semantic HELM: A Human-Readable Memory for Reinforcement Learning
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/1eeacdf8770e6dd5164cdeec8bcfa8cc-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/1eeacdf8770e6dd5164cdeec8bcfa8cc-Paper-Conference.pdf)]
    * Title: Semantic HELM: A Human-Readable Memory for Reinforcement Learning
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Fabian Paischer, Thomas Adler, Markus Hofmarcher, Sepp Hochreiter
    * Abstract: Reinforcement learning agents deployed in the real world often have to cope with partially observable environments. Therefore, most agents employ memory mechanisms to approximate the state of the environment. Recently, there have been impressive success stories in mastering partially observable environments, mostly in the realm of computer games like Dota 2, StarCraft II, or MineCraft. However, existing methods lack interpretability in the sense that it is not comprehensible for humans what the agent stores in its memory.In this regard, we propose a novel memory mechanism that represents past events in human language.Our method uses CLIP to associate visual inputs with language tokens. Then we feed these tokens to a pretrained language model that serves the agent as memory and provides it with a coherent and human-readable representation of the past.We train our memory mechanism on a set of partially observable environments and find that it excels on tasks that require a memory component, while mostly attaining performance on-par with strong baselines on tasks that do not. On a challenging continuous recognition task, where memorizing the past is crucial, our memory mechanism converges two orders of magnitude faster than prior methods.Since our memory mechanism is human-readable, we can peek at an agent's memory and check whether crucial pieces of information have been stored.This significantly enhances troubleshooting and paves the way toward more interpretable agents.

count=1
* Benchmarking Encoder-Decoder Architectures for Biplanar X-ray to 3D Bone Shape Reconstruction
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/412732f172bdd5ad0efde2fafa110700-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Benchmarking Encoder-Decoder Architectures for Biplanar X-ray to 3D Bone Shape Reconstruction
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Mahesh Shakya, Bishesh Khanal
    * Abstract: Various deep learning models have been proposed for 3D bone shape reconstruction from two orthogonal (biplanar) X-ray images.However, it is unclear how these models compare against each other since they are evaluated on different anatomy, cohort and (often privately held) datasets.Moreover, the impact of the commonly optimized image-based segmentation metrics such as dice score on the estimation of clinical parameters relevant in 2D-3D bone shape reconstruction is not well known.To move closer toward clinical translation, we propose a benchmarking framework that evaluates tasks relevant to real-world clinical scenarios, including reconstruction of fractured bones, bones with implants, robustness to population shift, and error in estimating clinical parameters.Our open-source platform provides reference implementations of 8 models (many of whose implementations were not publicly available), APIs to easily collect and preprocess 6 public datasets, and the implementation of automatic clinical parameter and landmark extraction methods. We present an extensive evaluation of 8 2D-3D models on equal footing using 6 public datasets comprising images for four different anatomies.Our results show that attention-based methods that capture global spatial relationships tend to perform better across all anatomies and datasets; performance on clinically relevant subgroups may be overestimated without disaggregated reporting; ribs are substantially more difficult to reconstruct compared to femur, hip and spine; and the dice score improvement does not always bring corresponding improvement in the automatic estimation of clinically relevant parameters.

count=1
* Beta Diffusion
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/5fe1b43c882d746c187456eb4c8cdf52-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/5fe1b43c882d746c187456eb4c8cdf52-Paper-Conference.pdf)]
    * Title: Beta Diffusion
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Mingyuan Zhou, Tianqi Chen, Zhendong Wang, Huangjie Zheng
    * Abstract: We introduce beta diffusion, a novel generative modeling method that integrates demasking and denoising to generate data within bounded ranges. Using scaled and shifted beta distributions, beta diffusion utilizes multiplicative transitions over time to create both forward and reverse diffusion processes, maintaining beta distributions in both the forward marginals and the reverse conditionals, given the data at any point in time. Unlike traditional diffusion-based generative models relying on additive Gaussian noise and reweighted evidence lower bounds (ELBOs), beta diffusion is multiplicative and optimized with KL-divergence upper bounds (KLUBs) derived from the convexity of the KL divergence. We demonstrate that the proposed KLUBs are more effective for optimizing beta diffusion compared to negative ELBOs, which can also be derived as the KLUBs of the same KL divergence with its two arguments swapped. The loss function of beta diffusion, expressed in terms of Bregman divergence, further supports the efficacy of KLUBs for optimization. Experimental results on both synthetic data and natural images demonstrate the unique capabilities of beta diffusion in generative modeling of range-bounded data and validate the effectiveness of KLUBs in optimizing diffusion models, thereby making them valuable additions to the family of diffusion-based generative models and the optimization techniques used to train them.

count=1
* Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/661caac7729aa7d8c6b8ac0d39ccbc6a-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/661caac7729aa7d8c6b8ac0d39ccbc6a-Paper-Conference.pdf)]
    * Title: Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, Liang-Chieh Chen
    * Abstract: Open-vocabulary segmentation is a challenging task requiring segmenting and recognizing objects from an open set of categories in diverse environments. One way to address this challenge is to leverage multi-modal models, such as CLIP, to provide image and text features in a shared embedding space, which effectively bridges the gap between closed-vocabulary and open-vocabulary recognition.Hence, existing methods often adopt a two-stage framework to tackle the problem, where the inputs first go through a mask generator and then through the CLIP model along with the predicted masks. This process involves extracting features from raw images multiple times, which can be ineffective and inefficient. By contrast, we propose to build everything into a single-stage framework using a shared Frozen Convolutional CLIP backbone, which not only significantly simplifies the current two-stage pipeline, but also remarkably yields a better accuracy-cost trade-off. The resulting single-stage system, called FC-CLIP, benefits from the following observations: the frozen CLIP backbone maintains the ability of open-vocabulary classification and can also serve as a strong mask generator, and the convolutional CLIP generalizes well to a larger input resolution than the one used during contrastive image-text pretraining. Surprisingly, FC-CLIP advances state-of-the-art results on various benchmarks, while running practically fast. Specifically, when training on COCO panoptic data only and testing in a zero-shot manner, FC-CLIP achieve 26.8 PQ, 16.8 AP, and 34.1 mIoU on ADE20K, 18.2 PQ, 27.9 mIoU on Mapillary Vistas, 44.0 PQ, 26.8 AP, 56.2 mIoU on Cityscapes, outperforming the prior art under the same setting by +4.2 PQ, +2.4 AP, +4.2 mIoU on ADE20K, +4.0 PQ on Mapillary Vistas and +20.1 PQ on Cityscapes, respectively. Additionally, the training and testing time of FC-CLIP is 7.5x and 6.6x significantly faster than the same prior art, while using 5.9x fewer total model parameters. Meanwhile, FC-CLIP also sets a new state-of-the-art performance across various open-vocabulary semantic segmentation datasets. Code and models are available at https://github.com/bytedance/fc-clip

count=1
* Nonparametric Identifiability of Causal Representations from Unknown Interventions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/97fe251c25b6f99a2a23b330a75b11d4-Paper-Conference.pdf)]
    * Title: Nonparametric Identifiability of Causal Representations from Unknown Interventions
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Julius von Kügelgen, Michel Besserve, Liang Wendong, Luigi Gresele, Armin Kekić, Elias Bareinboim, David Blei, Bernhard Schölkopf
    * Abstract: We study causal representation learning, the task of inferring latent causal variables and their causal relations from high-dimensional functions (“mixtures”) of the variables. Prior work relies on weak supervision, in the form of counterfactual pre- and post-intervention views or temporal structure; places restrictive assumptions, such as linearity, on the mixing function or latent causal model; or requires partial knowledge of the generative process, such as the causal graph or intervention targets. We instead consider the general setting in which both the causal model and the mixing function are nonparametric. The learning signal takes the form of multiple datasets, or environments, arising from unknown interventions in the underlying causal model. Our goal is to identify both the ground truth latents and their causal graph up to a set of ambiguities which we show to be irresolvable from interventional data. We study the fundamental setting of two causal variables and prove that the observational distribution and one perfect intervention per node suffice for identifiability, subject to a genericity condition. This condition rules out spurious solutions that involve fine-tuning of the intervened and observational distributions, mirroring similar conditions for nonlinear cause-effect inference. For an arbitrary number of variables, we show that at least one pair of distinct perfect interventional domains per node guarantees identifiability. Further, we demonstrate that the strengths of causal influences among the latent variables are preserved by all equivalent solutions, rendering the inferred representation appropriate for drawing causal conclusions from new data. Our study provides the first identifiability results for the general nonparametric setting with unknown interventions, and elucidates what is possible and impossible for causal representation learning without more direct supervision.

count=1
* Django: Detecting Trojans in Object Detection Models via Gaussian Focus Calibration
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/a102d6cb996be3482c059c1e18bbe523-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/a102d6cb996be3482c059c1e18bbe523-Paper-Conference.pdf)]
    * Title: Django: Detecting Trojans in Object Detection Models via Gaussian Focus Calibration
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Guangyu Shen, Siyuan Cheng, Guanhong Tao, Kaiyuan Zhang, Yingqi Liu, Shengwei An, Shiqing Ma, Xiangyu Zhang
    * Abstract: Object detection models are vulnerable to backdoor or trojan attacks, where an attacker can inject malicious triggers into the model, leading to altered behavior during inference. As a defense mechanism, trigger inversion leverages optimization to reverse-engineer triggers and identify compromised models. While existing trigger inversion methods assume that each instance from the support set is equally affected by the injected trigger, we observe that the poison effect can vary significantly across bounding boxes in object detection models due to its dense prediction nature, leading to an undesired optimization objective misalignment issue for existing trigger reverse-engineering methods. To address this challenge, we propose the first object detection backdoor detection framework Django (Detecting Trojans in Object Detection Models via Gaussian Focus Calibration). It leverages a dynamic Gaussian weighting scheme that prioritizes more vulnerable victim boxes and assigns appropriate coefficients to calibrate the optimization objective during trigger inversion. In addition, we combine Django with a novel label proposal pre-processing technique to enhance its efficiency. We evaluate Django on 3 object detection image datasets, 3 model architectures, and 2 types of attacks, with a total of 168 models. Our experimental results show that Django outperforms 6 state-of-the-art baselines, with up to 38% accuracy improvement and 10x reduced overhead. The code is available at https://github.com/PurduePAML/DJGO.

count=1
* Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/cabfaeecaae7d6540ee797a66f0130b0-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/cabfaeecaae7d6540ee797a66f0130b0-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, Hang Zhao
    * Abstract: Robotic perception requires the modeling of both 3D geometry and semantics. Existing methods typically focus on estimating 3D bounding boxes, neglecting finer geometric details and struggling to handle general, out-of-vocabulary objects. 3D occupancy prediction, which estimates the detailed occupancy states and semantics of a scene, is an emerging task to overcome these limitations.To support 3D occupancy prediction, we develop a label generation pipeline that produces dense, visibility-aware labels for any given scene. This pipeline comprises three stages: voxel densification, occlusion reasoning, and image-guided voxel refinement. We establish two benchmarks, derived from the Waymo Open Dataset and the nuScenes Dataset, namely Occ3D-Waymo and Occ3D-nuScenes benchmarks. Furthermore, we provide an extensive analysis of the proposed dataset with various baseline models. Lastly, we propose a new model, dubbed Coarse-to-Fine Occupancy (CTF-Occ) network, which demonstrates superior performance on the Occ3D benchmarks.The code, data, and benchmarks are released at \url{https://tsinghua-mars-lab.github.io/Occ3D/}.

