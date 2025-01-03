count=3
* Manifold SLIC: A Fast Method to Compute Content-Sensitive Superpixels
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_Manifold_SLIC_A_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Manifold_SLIC_A_CVPR_2016_paper.pdf)]
    * Title: Manifold SLIC: A Fast Method to Compute Content-Sensitive Superpixels
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Yong-Jin Liu, Cheng-Chi Yu, Min-Jing Yu, Ying He
    * Abstract: Superpixels are perceptually meaningful atomic regions that can effectively capture image features. Among various methods for computing uniform superpixels, simple linear iterative clustering (SLIC) is popular due to its simplicity and high performance. In this paper, we extend SLIC to compute content-sensitive superpixels, i.e., small superpixels in content-dense regions (e.g., with high intensity or color variation) and large superpixels in content-sparse regions. Rather than the conventional SLIC method that clusters pixels in R5, we map the image I to a 2-dimensional manifold M in R5, whose area elements are a good measure of the content density in I. We propose an efficient method to compute restricted centroidal Voronoi tessellation (RCVT) --- a uniform tessellation --- on M, which induces the content-sensitive superpixels in I. Unlike other algorithms that characterize content-sensitivity by geodesic distances, manifold SLIC tackles the problem by measuring areas of Voronoi cells on M, which can be computed at a very low cost. As a result, it runs 10 times faster than the state-of-the-art content-sensitive superpixels algorithm. We evaluate manifold SLIC and seven representative methods on the BSDS500 benchmark and observe that our method outperforms the existing methods.

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
* SUPRA: Superpixel Guided Loss for Improved Multi-Modal Segmentation in Endoscopy
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/LatinX/html/Martinez-Garcia-Pena_SUPRA_Superpixel_Guided_Loss_for_Improved_Multi-Modal_Segmentation_in_Endoscopy_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/LatinX/papers/Martinez-Garcia-Pena_SUPRA_Superpixel_Guided_Loss_for_Improved_Multi-Modal_Segmentation_in_Endoscopy_CVPRW_2023_paper.pdf)]
    * Title: SUPRA: Superpixel Guided Loss for Improved Multi-Modal Segmentation in Endoscopy
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Rafael Martínez-García-Peña, Mansoor Ali Teevno, Gilberto Ochoa-Ruiz, Sharib Ali
    * Abstract: Domain shift is a well-known problem in the medical imaging community. In particular, for endoscopic image analysis data can have different modalities that cause the performance of deep learning (DL) methods to become adversely affected. Methods developed on one modality cannot be used for a different modality without retraining. However, in real clinical settings, endoscopists switch between modalities depending on the specifics of the condition being explored. In this paper, we explore domain generalisation to enable DL methods to be used in such scenarios. To this extent, we propose to use superpixels generated with Simple Linear Iterative Clustering (SLIC), which we refer to as "SUPRA" for SUPeRpixel Augmented method. SUPRA first generates a preliminary segmentation mask making use of our new loss "SLICLoss" that encourages both an accurate and superpixel-consistent segmentation. We demonstrate that SLICLoss when combined with Binary Cross Entropy loss (BCE) can improve the model's generalisability with data that presents significant domain shift due to a change in lighting modalities. We validate this novel compound loss on a vanilla UNet using the EndoUDA dataset, which contains images for Barret's Esophagus from two modalities. We show that our method yields a relative improvement of more than 20% IoU in the target domain set compared to the baseline.

count=2
* A Video Representation Using Temporal Superpixels
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Chang_A_Video_Representation_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Chang_A_Video_Representation_2013_CVPR_paper.pdf)]
    * Title: A Video Representation Using Temporal Superpixels
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Jason Chang, Donglai Wei, John W. Fisher III
    * Abstract: We develop a generative probabilistic model for temporally consistent superpixels in video sequences. In contrast to supervoxel methods, object parts in different frames are tracked by the same temporal superpixel. We explicitly model flow between frames with a bilateral Gaussian process and use this information to propagate superpixels in an online fashion. We consider four novel metrics to quantify performance of a temporal superpixel representation and demonstrate superior performance when compared to supervoxel methods.

count=2
* A Comparison of Deep Learning Methods for Semantic Segmentation of Coral Reef Survey Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w28/html/King_A_Comparison_of_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w28/King_A_Comparison_of_CVPR_2018_paper.pdf)]
    * Title: A Comparison of Deep Learning Methods for Semantic Segmentation of Coral Reef Survey Images
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Andrew King, Suchendra M. Bhandarkar, Brian M. Hopkinson
    * Abstract: Two major deep learning methods for semantic segmentation, i.e., patch-based convolutional neural network (CNN) approaches and fully convolutional neural network (FCNN) models, are studied in the context of classification of regions in underwater images of coral reef ecosystems into biologically meaningful categories. For the patch-based CNN approaches, we use image data extracted from underwater video accompanied by individual point-wise ground truth annotations. We show that patch-based CNN methods can outperform a previously proposed approach that uses support vector machine (SVM)-based classifiers in conjunction with texture-based features. We compare the results of five different CNN architectures in our formulation of patch-based CNN methods. The Resnet152 CNN architecture is observed to perform the best on our annotated dataset of underwater coral reef images. We also examine and compare the results of four different FCNN models for semantic segmentation of coral reef images. We develop a tool for fast generation of segmentation maps to serve as ground truth segmentations for our FCNN models. The FCNN architecture Deeplab v2 is observed to yield the best results for semantic segmentation of underwater coral reef images.

count=2
* Superpixels and Graph Convolutional Neural Networks for Efficient Detection of Nutrient Deficiency Stress From Aerial Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AgriVision/html/Dadsetan_Superpixels_and_Graph_Convolutional_Neural_Networks_for_Efficient_Detection_of_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AgriVision/papers/Dadsetan_Superpixels_and_Graph_Convolutional_Neural_Networks_for_Efficient_Detection_of_CVPRW_2021_paper.pdf)]
    * Title: Superpixels and Graph Convolutional Neural Networks for Efficient Detection of Nutrient Deficiency Stress From Aerial Imagery
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Saba Dadsetan, David Pichler, David Wilson, Naira Hovakimyan, Jennifer Hobbs
    * Abstract: Advances in remote sensing technology have led to the capture of massive amounts of data. Increased image resolution, more frequent revisit times, and additional spectral channels have created an explosion in the amount of data that is available to provide analyses and intelligence across domains, including agriculture. However, the processing of this data comes with a cost in terms of computation time and money, both of which must be considered when the goal of an algorithm is to provide real-time intelligence to improve efficiencies. Specifically, we seek to identify nutrient deficient areas from remotely sensed data to alert farmers to regions that require attention; detection of nutrient deficient areas is a key task in precision agriculture as farmers must quickly respond to struggling areas to protect their harvests. Past methods have focused on pixel-level classification (i.e. semantic segmentation) of the field to achieve these tasks, often using deep learning models with tens-of-millions of parameters. In contrast, we propose a much lighter graph-based method to perform node-based classification. We first use Simple Linear Iterative Cluster (SLIC) to produce superpixels across the field. Then, to perform segmentation across the non-Euclidean domain of superpixels, we leverage a Graph Convolutional Neural Network (GCN). This model has 4-orders-of-magnitude fewer parameters than a CNN model and trains in a matter of minutes.

count=2
* Deep Learning for Semantic Segmentation of Coral Reef Images Using Multi-View Information
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/AAMVEM/King_Deep_Learning_for_Semantic_Segmentation_of_Coral_Reef_Images_Using_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AAMVEM/King_Deep_Learning_for_Semantic_Segmentation_of_Coral_Reef_Images_Using_CVPRW_2019_paper.pdf)]
    * Title: Deep Learning for Semantic Segmentation of Coral Reef Images Using Multi-View Information
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Andrew King, Suchendra M.Bhandarkar,  Brian M. Hopkinson
    * Abstract: Two major deep learning architectures, i.e., patch-based convolutional neural networks (CNNs) and fully convolutional neural networks (FCNNs), are studied in the context of semantic segmentation of underwater images of coral reef ecosystems. Patch-based CNNs are typically used to enable single-entity classification whereas FCNNs are used to generate a semantically segmented output from an input image. In coral reef mapping tasks, one typically obtains multiple images of a coral reef from varying viewpoints either using stereoscopic image acquisition or while conducting underwater video surveys. We propose and compare patch-based CNN and FCNN architectures capable of exploiting multi-view image information to improve the accuracy of classification and semantic segmentation of the input images. We investigate extensions of the conventional FCNN architecture to incorporate stereoscopic input image data and extensions of patch-based CNN architectures to incorporate multi-view input image data. Experimental results show the proposed TwinNet architecture to be the best performing FCNN architecture, performing comparably with its baseline Dilation8 architecture when using just a left-perspective input image, but markedly improving over Dilation8 when using a stereo pair of input images. Likewise, the proposed nViewNet-8 architecture is shown to be the best performing patch-based CNN architecture, outperforming its single-image ResNet152 baseline architecture in terms of classification accuracy.

count=2
* SWAG: Superpixels Weighted by Average Gradients for Explanations of CNNs
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Hartley_SWAG_Superpixels_Weighted_by_Average_Gradients_for_Explanations_of_CNNs_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Hartley_SWAG_Superpixels_Weighted_by_Average_Gradients_for_Explanations_of_CNNs_WACV_2021_paper.pdf)]
    * Title: SWAG: Superpixels Weighted by Average Gradients for Explanations of CNNs
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Thomas Hartley, Kirill Sidorov, Christopher Willis, David Marshall
    * Abstract: Providing an explanation of the operation of CNNs that is both accurate and interpretable is becoming essential in fields like medical image analysis, surveillance, and autonomous driving. In these areas, it is important to have confidence that the CNN is working as expected and explanations from saliency maps provide an efficient way of doing this. In this paper, we propose a pair of complementary contributions that improve upon the state of the art for region-based explanations in both accuracy and utility. The first is SWAG, a method for generating accurate explanations quickly using superpixels for discriminative regions which is meant to be a more accurate, efficient, and tunable drop in replacement method for Grad-CAM, LIME, or other region-based methods. The second contribution is based on an investigation into how to best generate the superpixels used to represent the features found within the image. Using SWAG, we compare using superpixels created from the image, a combination of the image and backpropagated gradients, and the gradients themselves. To the best of our knowledge, this is the first method proposed to generate explanations using superpixels explicitly created to represent the discriminative features important to the network. To compare we use both ImageNet and challenging fine-grained datasets over a range of metrics. We demonstrate experimentally that our methods provide the best local and global accuracy compared to Grad-CAM, Grad-CAM++, LIME, XRAI, and RISE.

count=1
* Generic Image Segmentation in Fully Convolutional Networks by Superpixel Merging Map
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Huang_Generic_Image_Segmentation_in_Fully_Convolutional_Networks_by_Superpixel_Merging_ACCV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2020/papers/Huang_Generic_Image_Segmentation_in_Fully_Convolutional_Networks_by_Superpixel_Merging_ACCV_2020_paper.pdf)]
    * Title: Generic Image Segmentation in Fully Convolutional Networks by Superpixel Merging Map
    * Publisher: ACCV
    * Publication Date: `2020`
    * Authors: Jin-Yu Huang, Jian-Jiun Ding
    * Abstract: Recently, the Fully Convolutional Network (FCN) has been adopted in image segmentation. However, existing FCN-based segmentation algorithms were designed for semantic segmentation. Before learning-based algorithms were developed, many advanced generic segmentation algorithms are superpixel-based. However, due to the irregular shape and size of superpixels, it is hard to apply deep learning to superpixel-based image segmentation directly. In this paper, we combined the merits of the FCN and superpixels and proposed a highly accurate and extremely fast generic image segmentation algorithm. We treated image segmentation as multiple superpixel merging decision problems and determined whether the boundary between two adjacent superpixels should be kept. In other words, if the boundary of two adjacent superpixels should be deleted, then the two superpixels will be merged. The network applies the colors, the edge map, and the superpixel information to make decision about merging suprepixels. By solving all the superpixel-merging subproblems with just one forward pass, the FCN facilitates the speed of the whole segmentation process by a wide margin meanwhile gaining higher accuracy. Simulations show that the proposed algorithm has favorable runtime, meanwhile achieving highly accurate segmentation results. It outperforms state-of-the-art image segmentation methods, including feature-based and learning-based methods, in all metrics.

count=1
* Road Obstacle Detection Method Based on an Autoencoder with Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Ohgushi_Road_Obstacle_Detection_Method_Based_on_an_Autoencoder_with_Semantic_ACCV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2020/papers/Ohgushi_Road_Obstacle_Detection_Method_Based_on_an_Autoencoder_with_Semantic_ACCV_2020_paper.pdf)]
    * Title: Road Obstacle Detection Method Based on an Autoencoder with Semantic Segmentation
    * Publisher: ACCV
    * Publication Date: `2020`
    * Authors: Toshiaki Ohgushi, Kenji Horiguchi, Masao Yamanaka
    * Abstract: Accurate detection of road obstacles is vital for ensuring safe autonomous driving, particularly on highways.However, existing methods tend to perform poorly when analyzing road scenes with complex backgrounds, because supervised approaches cannot detect unknown objects that are not included in the training dataset.Hence, in this study, we propose a road obstacle detection method using an autoencoder with semantic segmentation that was trained with only data from normal road scenes.The proposed method requires only a color image captured by a common in-vehicle camera as input. It then creates a resynthesized image using an autoencoder consisting of a semantic image generator as the encoder and a photographic image generator as the decoder.Extensive experiments demonstrate that the performance of the proposed method is comparable to that of existing methods, even without postprocessing. The proposed method with postprocessing outperformed state-of-the-art methods on the Lost and Found dataset.Further, in evaluations using our Highway Anomaly Dataset, which includes actual and synthetic road obstacles, the proposed method significantly outperformed a supervised method that explicitly learns road obstacles.Thus, the proposed machine-learning-based road obstacle detection method is a practical solution that will advance the development of autonomous driving systems.

count=1
* Utilizing Transfer Learning and a Customized Loss Function for Optic Disc Segmentation from Retinal Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Sarhan_Utilizing_Transfer_Learning_and_a_Customized_Loss_Function_for_Optic_ACCV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2020/papers/Sarhan_Utilizing_Transfer_Learning_and_a_Customized_Loss_Function_for_Optic_ACCV_2020_paper.pdf)]
    * Title: Utilizing Transfer Learning and a Customized Loss Function for Optic Disc Segmentation from Retinal Images
    * Publisher: ACCV
    * Publication Date: `2020`
    * Authors: Abdullah Sarhan, Ali Al-Khaz'Aly, Adam Gorner, Andrew Swift, Jon Rokne, Reda Alhajj, Andrew Crichton
    * Abstract: Accurate segmentation of the optic disc from a retinal image is vital to extracting retinal features that may be highly correlated with retinal conditions such as glaucoma. In this paper, we propose a deep-learning based approach capable of segmenting the optic disc given a high-precision retinal fundus image. Our approach utilizes a UNET-based model with a VGG16 encoder trained on the ImageNet dataset. This study can be distinguished from other studies in the customization made for the VGG16 model, the diversity of the datasets adopted, the duration of disc segmentation, the loss function utilized, and the number of parameters required to train our model. Our approach was tested on seven publicly available datasets augmented by a dataset from a private clinic that was annotated by two Doctors of Optometry through a web portal built for this purpose. We achieved an accuracy of 99.78% and a Dice coefficient of 94.73% for a disc segmentation from a retinal image in 0.03 seconds. The results obtained from comprehensive experiments demonstrate the robustness of our approach to disc segmentation of retinal images obtained from different sources.

count=1
* Voxel Cloud Connectivity Segmentation - Supervoxels for Point Clouds
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Papon_Voxel_Cloud_Connectivity_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Papon_Voxel_Cloud_Connectivity_2013_CVPR_paper.pdf)]
    * Title: Voxel Cloud Connectivity Segmentation - Supervoxels for Point Clouds
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Jeremie Papon, Alexey Abramov, Markus Schoeler, Florentin Worgotter
    * Abstract: Unsupervised over-segmentation of an image into regions of perceptually similar pixels, known as superpixels, is a widely used preprocessing step in segmentation algorithms. Superpixel methods reduce the number of regions that must be considered later by more computationally expensive algorithms, with a minimal loss of information. Nevertheless, as some information is inevitably lost, it is vital that superpixels not cross object boundaries, as such errors will propagate through later steps. Existing methods make use of projected color or depth information, but do not consider three dimensional geometric relationships between observed data points which can be used to prevent superpixels from crossing regions of empty space. We propose a novel over-segmentation algorithm which uses voxel relationships to produce over-segmentations which are fully consistent with the spatial geometry of the scene in three dimensional, rather than projective, space. Enforcing the constraint that segmented regions must have spatial connectivity prevents label flow across semantic object boundaries which might otherwise be violated. Additionally, as the algorithm works directly in 3D space, observations from several calibrated RGB+D cameras can be segmented jointly. Experiments on a large data set of human annotated RGB+D images demonstrate a significant reduction in occurrence of clusters crossing object boundaries, while maintaining speeds comparable to state-of-the-art 2D methods.

count=1
* Automatic Feature Learning for Robust Shadow Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Khan_Automatic_Feature_Learning_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Khan_Automatic_Feature_Learning_2014_CVPR_paper.pdf)]
    * Title: Automatic Feature Learning for Robust Shadow Detection
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Salman Hameed Khan, Mohammed Bennamoun, Ferdous Sohel, Roberto Togneri
    * Abstract: We present a practical framework to automatically detect shadows in real world scenes from a single photograph. Previous works on shadow detection put a lot of effort in designing shadow variant and invariant hand-crafted features. In contrast, our framework automatically learns the most relevant features in a supervised manner using multiple convolutional deep neural networks (ConvNets). The 7-layer network architecture of each ConvNet consists of alternating convolution and sub-sampling layers. The proposed framework learns features at the super-pixel level and along the object boundaries. In both cases, features are extracted using a context aware window centered at interest points. The predicted posteriors based on the learned features are fed to a conditional random field model to generate smooth shadow contours. Our proposed framework consistently performed better than the state-of-the-art on all major shadow databases collected under a variety of conditions.

count=1
* KL Divergence Based Agglomerative Clustering for Automated Vitiligo Grading
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Gupta_KL_Divergence_Based_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gupta_KL_Divergence_Based_2015_CVPR_paper.pdf)]
    * Title: KL Divergence Based Agglomerative Clustering for Automated Vitiligo Grading
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Mithun Das Gupta, Srinidhi Srinivasa, Madhukara J., Meryl Antony
    * Abstract: In this paper we present a symmetric KL divergence based agglomerative clustering framework to segment multiple levels of depigmentation in Vitiligo images. The proposed framework starts with a simple merge cost based on symmetric KL divergence. We extend the recent body of work related to Bregman divergence based agglomerative clustering and prove that the symmetric KL divergence is an upper-bound for uni-modal Gaussian distributions. This leads to a very simple yet elegant method for bottomup agglomerative clustering. We introduce albedo and reflectance fields as features for the distance computations. We compare against other established methods to bring out possible pros and cons of the proposed method.

count=1
* Unconstrained Realtime Facial Performance Capture
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Hsieh_Unconstrained_Realtime_Facial_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Hsieh_Unconstrained_Realtime_Facial_2015_CVPR_paper.pdf)]
    * Title: Unconstrained Realtime Facial Performance Capture
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Pei-Lun Hsieh, Chongyang Ma, Jihun Yu, Hao Li
    * Abstract: We introduce a realtime facial tracking system specifically designed for performance capture in unconstrained settings using a consumer-level RGB-D sensor. Our framework provides uninterrupted 3D facial tracking, even in the presence of extreme occlusions such as those caused by hair, hand-to-face gestures, and wearable accessories. Anyone's face can be instantly tracked and the users can be switched without an extra calibration step. During tracking, we explicitly segment face regions from any occluding parts by detecting outliers in the shape and appearance input using an exponentially smoothed and user-adaptive tracking model as prior. Our face segmentation combines depth and RGB input data and is also robust against illumination changes. To enable continuous and reliable facial feature tracking in the color channels, we synthesize plausible face textures in the occluded regions. Our tracking model is personalized on-the-fly by progressively refining the user's identity, expressions, and texture with reliable samples and temporal filtering. We demonstrate robust and high-fidelity facial tracking on a wide range of subjects with highly incomplete and largely occluded data. Our system works in everyday environments and is fully unobtrusive to the user, impacting consumer AR applications and surveillance.

count=1
* A Weighted Sparse Coding Framework for Saliency Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Li_A_Weighted_Sparse_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_A_Weighted_Sparse_2015_CVPR_paper.pdf)]
    * Title: A Weighted Sparse Coding Framework for Saliency Detection
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Nianyi Li, Bilin Sun, Jingyi Yu
    * Abstract: There is an emerging interest on using high-dimensional datasets beyond 2D images in saliency detection. Examples include 3D data based on stereo matching and Kinect sensors and more recently 4D light field data. However, these techniques adopt very different solution frameworks, in both type of features and procedures on using them. In this paper, we present a unified saliency detection framework for handling heterogenous types of input data. Our approach builds dictionaries using data-specific features. Specifically, we first select a group of potential background superpixels to build a primitive non-saliency dictionary. We then prune the outliers in the dictionary and test on the remaining superpixels to iteratively refine the dictionary. Comprehensive experiments show that our approach universally outperforms the state-of-the-art solution on all 2D, 3D and 4D data.

count=1
* Robust Saliency Detection via Regularized Random Walks Ranking
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Li_Robust_Saliency_Detection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_Robust_Saliency_Detection_2015_CVPR_paper.pdf)]
    * Title: Robust Saliency Detection via Regularized Random Walks Ranking
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Changyang Li, Yuchen Yuan, Weidong Cai, Yong Xia, David Dagan Feng
    * Abstract: In the field of saliency detection, many graph-based algorithms heavily depend on the accuracy of the pre-processed superpixel segmentation, which leads to significant sacrifice of detail information from the input image. In this paper, we propose a novel bottom-up saliency detection approach that takes advantage of both region-based features and image details. To provide more accurate saliency estimations, we first optimize the image boundary selection by the proposed erroneous boundary removal. By taking the image details and region-based estimations into account, we then propose the regularized random walks ranking to formulate pixel-wised saliency maps from the superpixel-based background and foreground saliency estimations. Experiment results on two public datasets indicate the significantly improved accuracy and robustness of the proposed algorithm in comparison with 12 state-of-the-art saliency detection approaches.

count=1
* Saliency Detection via Cellular Automata
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Qin_Saliency_Detection_via_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Qin_Saliency_Detection_via_2015_CVPR_paper.pdf)]
    * Title: Saliency Detection via Cellular Automata
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Yao Qin, Huchuan Lu, Yiqun Xu, He Wang
    * Abstract: In this paper, we introduce Cellular Automata--a dynamic evolution model to intuitively detect the salient object. First, we construct a background-based map using color and space contrast with the clustered boundary seeds. Then, a novel propagation mechanism dependent on Cellular Automata is proposed to exploit the intrinsic relevance of similar regions through interactions with neighbors. Impact factor matrix and coherence matrix are constructed to balance the influential power towards each cell's next state. The saliency values of all cells will be renovated simultaneously according to the proposed updating rule. It's surprising to find out that parallel evolution can improve all the existing methods to a similar level regardless of their original results. Finally, we present an integration algorithm in the Bayesian framework to take advantage of multiple saliency maps. Extensive experiments on six public datasets demonstrate that the proposed algorithm outperforms state-of-the-art methods.

count=1
* Salient Object Detection via Bootstrap Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Tong_Salient_Object_Detection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Tong_Salient_Object_Detection_2015_CVPR_paper.pdf)]
    * Title: Salient Object Detection via Bootstrap Learning
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Na Tong, Huchuan Lu, Xiang Ruan, Ming-Hsuan Yang
    * Abstract: We propose a bootstrap learning algorithm for salient object detection in which both weak and strong models are exploited. First, a weak saliency map is constructed based on image priors to generate training samples for a strong model. Second, a strong classifier based on samples directly from an input image is learned to detect salient pixels. Results from multiscale saliency maps are integrated to further improve the detection performance. Extensive experiments on five benchmark datasets demonstrate that the proposed bootstrap learning algorithm performs favorably against the state-of-the-art saliency detection methods. Furthermore, we show that the proposed bootstrap learning approach can be easily applied to other bottom-up saliency models for significant improvement.

count=1
* GraB: Visual Saliency via Novel Graph Model and Background Priors
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Wang_GraB_Visual_Saliency_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Wang_GraB_Visual_Saliency_CVPR_2016_paper.pdf)]
    * Title: GraB: Visual Saliency via Novel Graph Model and Background Priors
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Qiaosong Wang, Wen Zheng, Robinson Piramuthu
    * Abstract: We propose an unsupervised bottom-up saliency detection approach by exploiting novel graph structure and background priors. The input image is represented as an undirected graph with superpixels as nodes. Feature vectors are extracted from each node to cover regional color, contrast and texture information. A novel graph model is proposed to effectively capture local and global saliency cues. To obtain more accurate saliency estimations, we optimize the saliency map by using a robust background measure. Comprehensive evaluations on benchmark datasets indicate that our algorithm universally surpasses state-of-the-art unsupervised solutions and performs favorably against supervised approaches.

count=1
* Contour-Constrained Superpixels for Image and Video Processing
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Lee_Contour-Constrained_Superpixels_for_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lee_Contour-Constrained_Superpixels_for_CVPR_2017_paper.pdf)]
    * Title: Contour-Constrained Superpixels for Image and Video Processing
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Se-Ho Lee, Won-Dong Jang, Chang-Su Kim
    * Abstract: A novel contour-constrained superpixel (CCS) algorithm is proposed in this work. We initialize superpixels and regions in a regular grid and then refine the superpixel label of each region hierarchically from block to pixel levels. To make superpixel boundaries compatible with object contours, we propose the notion of contour pattern matching and formulate an objective function including the contour constraint. Furthermore, we extend the CCS algorithm to generate temporal superpixels for video processing. We initialize superpixel labels in each frame by transferring those in the previous frame and refine the labels to make superpixels temporally consistent as well as compatible with object contours. Experimental results demonstrate that the proposed algorithm provides better performance than the state-of-the-art superpixel methods.

count=1
* Topology Reconstruction of Tree-Like Structure in Images via Structural Similarity Measure and Dominant Set Clustering
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Topology_Reconstruction_of_Tree-Like_Structure_in_Images_via_Structural_Similarity_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Topology_Reconstruction_of_Tree-Like_Structure_in_Images_via_Structural_Similarity_CVPR_2019_paper.pdf)]
    * Title: Topology Reconstruction of Tree-Like Structure in Images via Structural Similarity Measure and Dominant Set Clustering
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jianyang Xie,  Yitian Zhao,  Yonghuai Liu,  Pan Su,  Yifan Zhao,  Jun Cheng,  Yalin Zheng,  Jiang Liu
    * Abstract: The reconstruction and analysis of tree-like topological structures in the biomedical images is crucial for biologists and surgeons to understand biomedical conditions and plan surgical procedures. The underlying tree-structure topology reveals how different curvilinear components are anatomically connected to each other. Existing automated topology reconstruction methods have great difficulty in identifying the connectivity when two or more curvilinear components cross or bifurcate, due to their projection ambiguity, imaging noise and low contrast. In this paper, we propose a novel curvilinear structural similarity measure to guide a dominant-set clustering approach to address this indispensable issue. The novel similarity measure takes into account both intensity and geometric properties in representing the curvilinear structure locally and globally, and group curvilinear objects at crossover points into different connected branches by dominant-set clustering. The proposed method is applicable to different imaging modalities, and quantitative and qualitative results on retinal vessel, plant root, and neuronal network datasets show that our methodology is capable of advancing the current state-of-the-art techniques.

count=1
* Learning a Weakly-Supervised Video Actor-Action Segmentation Model With a Wise Selection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Learning_a_Weakly-Supervised_Video_Actor-Action_Segmentation_Model_With_a_Wise_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_a_Weakly-Supervised_Video_Actor-Action_Segmentation_Model_With_a_Wise_CVPR_2020_paper.pdf)]
    * Title: Learning a Weakly-Supervised Video Actor-Action Segmentation Model With a Wise Selection
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Jie Chen,  Zhiheng Li,  Jiebo Luo,  Chenliang Xu
    * Abstract: We address weakly-supervised video actor-action segmentation (VAAS), which extends general video object segmentation (VOS) to additionally consider action labels of the actors. The most successful methods on VOS synthesize a pool of pseudo-annotations (PAs) and then refine them iteratively. However, they face challenges as to how to select from a massive amount of PAs high-quality ones, how to set an appropriate stop condition for weakly-supervised training, and how to initialize PAs pertaining to VAAS. To overcome these challenges, we propose a general Weakly-Supervised framework with a Wise Selection of training samples and model evaluation criterion (WS^2). Instead of blindly trusting quality-inconsistent PAs, WS^2 employs a learning-based selection to select effective PAs and a novel region integrity criterion as a stopping condition for weakly-supervised training. In addition, a 3D-Conv GCAM is devised to adapt to the VAAS task. Extensive experiments show that WS^2 achieves state-of-the-art performance on both weakly-supervised VOS and VAAS tasks and is on par with the best fully-supervised method on VAAS.

count=1
* Transferring and Regularizing Prediction for Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Transferring_and_Regularizing_Prediction_for_Semantic_Segmentation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Transferring_and_Regularizing_Prediction_for_Semantic_Segmentation_CVPR_2020_paper.pdf)]
    * Title: Transferring and Regularizing Prediction for Semantic Segmentation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Yiheng Zhang,  Zhaofan Qiu,  Ting Yao,  Chong-Wah Ngo,  Dong Liu,  Tao Mei
    * Abstract: Semantic segmentation often requires a large set of images with pixel-level annotations. In the view of extremely expensive expert labeling, recent research has shown that the models trained on photo-realistic synthetic data (e.g., computer games) with computer-generated annotations can be adapted to real images. Despite this progress, without constraining the prediction on real images, the models will easily overfit on synthetic data due to severe domain mismatch. In this paper, we novelly exploit the intrinsic properties of semantic segmentation to alleviate such problem for model transfer. Specifically, we present a Regularizer of Prediction Transfer (RPT) that imposes the intrinsic properties as constraints to regularize model transfer in an unsupervised fashion. These constraints include patch-level, cluster-level and context-level semantic prediction consistencies at different levels of image formation. As the transfer is label-free and data-driven, the robustness of prediction is addressed by selectively involving a subset of image regions for model regularization. Extensive experiments are conducted to verify the proposal of RPT on the transfer of models trained on GTA5 and SYNTHIA (synthetic data) to Cityscapes dataset (urban street scenes). RPT shows consistent improvements when injecting the constraints on several neural networks for semantic segmentation. More remarkably, when integrating RPT into the adversarial-based segmentation framework, we report to-date the best results: mIoU of 53.2%/51.7% when transferring from GTA5/SYNTHIA to Cityscapes, respectively.

count=1
* GridShift: A Faster Mode-Seeking Algorithm for Image Segmentation and Object Tracking
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Kumar_GridShift_A_Faster_Mode-Seeking_Algorithm_for_Image_Segmentation_and_Object_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Kumar_GridShift_A_Faster_Mode-Seeking_Algorithm_for_Image_Segmentation_and_Object_CVPR_2022_paper.pdf)]
    * Title: GridShift: A Faster Mode-Seeking Algorithm for Image Segmentation and Object Tracking
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Abhishek Kumar, Oladayo S. Ajani, Swagatam Das, Rammohan Mallipeddi
    * Abstract: In machine learning, MeanShift is one of the popular clustering algorithms. It iteratively moves each data point to the weighted mean of its neighborhood data points. The computational cost required for finding neighborhood data points for each one is quadratic to the number of data points. Therefore, it is very slow for large-scale datasets. To address this issue, we propose a mode-seeking algorithm, GridShift, with faster computing and principally based on MeanShift that uses a grid-based approach. To speed up, GridShift employs a grid-based approach for neighbor search, which is linear to the number of data points. In addition, GridShift moves the active grid cells (grid cells associated with at least one data point) in place of data points towards the higher density, which provides more speed up. The runtime of GridShift is linear to the number of active grid cells and exponential to the number of features. Therefore, it is ideal for large-scale low-dimensional applications such as object tracking and image segmentation. Through extensive experiments, we showcase the superior performance of GridShift compared to other MeanShift-based algorithms and state-of-the-art algorithms in terms of accuracy and runtime on benchmark datasets, image segmentation. Finally, we provide a new object-tracking algorithm based on GridShift and show promising results for object tracking compared to camshift and MeanShift++.

count=1
* CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition With Variational Alignment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.pdf)]
    * Title: CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition With Variational Alignment
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jiangbin Zheng, Yile Wang, Cheng Tan, Siyuan Li, Ge Wang, Jun Xia, Yidong Chen, Stan Z. Li
    * Abstract: Sign language recognition (SLR) is a weakly supervised task that annotates sign videos as textual glosses. Recent studies show that insufficient training caused by the lack of large-scale available sign datasets becomes the main bottleneck for SLR. Most SLR works thereby adopt pretrained visual modules and develop two mainstream solutions. The multi-stream architectures extend multi-cue visual features, yielding the current SOTA performances but requiring complex designs and might introduce potential noise. Alternatively, the advanced single-cue SLR frameworks using explicit cross-modal alignment between visual and textual modalities are simple and effective, potentially competitive with the multi-cue framework. In this work, we propose a novel contrastive visual-textual transformation for SLR, CVT-SLR, to fully explore the pretrained knowledge of both the visual and language modalities. Based on the single-cue cross-modal alignment framework, we propose a variational autoencoder (VAE) for pretrained contextual knowledge while introducing the complete pretrained language module. The VAE implicitly aligns visual and textual modalities while benefiting from pretrained contextual knowledge as the traditional contextual module. Meanwhile, a contrastive cross-modal alignment algorithm is designed to explicitly enhance the consistency constraints. Extensive experiments on public datasets (PHOENIX-2014 and PHOENIX-2014T) demonstrate that our proposed CVT-SLR consistently outperforms existing single-cue methods and even outperforms SOTA multi-cue methods.

count=1
* Understanding Video Transformers via Universal Concept Discovery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kowal_Understanding_Video_Transformers_via_Universal_Concept_Discovery_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Kowal_Understanding_Video_Transformers_via_Universal_Concept_Discovery_CVPR_2024_paper.pdf)]
    * Title: Understanding Video Transformers via Universal Concept Discovery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Matthew Kowal, Achal Dave, Rares Ambrus, Adrien Gaidon, Konstantinos G. Derpanis, Pavel Tokmakov
    * Abstract: This paper studies the problem of concept-based interpretability of transformer representations for videos. Concretely we seek to explain the decision-making process of video transformers based on high-level spatiotemporal concepts that are automatically discovered. Prior research on concept-based interpretability has concentrated solely on image-level tasks. Comparatively video models deal with the added temporal dimension increasing complexity and posing challenges in identifying dynamic concepts over time. In this work we systematically address these challenges by introducing the first Video Transformer Concept Discovery (VTCD) algorithm. To this end we propose an efficient approach for unsupervised identification of units of video transformer representations - concepts and ranking their importance to the output of a model. The resulting concepts are highly interpretable revealing spatio-temporal reasoning mechanisms and object-centric representations in unstructured video models. Performing this analysis jointly over a diverse set of supervised and self-supervised representations we discover that some of these mechanism are universal in video transformers. Finally we show that VTCD can be used for fine-grained action recognition and video object segmentation.

count=1
* Region-Based Representations Revisited
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Shlapentokh-Rothman_Region-Based_Representations_Revisited_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Shlapentokh-Rothman_Region-Based_Representations_Revisited_CVPR_2024_paper.pdf)]
    * Title: Region-Based Representations Revisited
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Michal Shlapentokh-Rothman, Ansel Blume, Yao Xiao, Yuqun Wu, Sethuraman TV, Heyi Tao, Jae Yong Lee, Wilfredo Torres, Yu-Xiong Wang, Derek Hoiem
    * Abstract: We investigate whether region-based representations are effective for recognition. Regions were once a mainstay in recognition approaches but pixel and patch-based features are now used almost exclusively. We show that recent class-agnostic segmenters like SAM can be effectively combined with strong unsupervised representations like DINOv2 and used for a wide variety of tasks including semantic segmentation object-based image retrieval and multi-image analysis. Once the masks and features are extracted these representations even with linear decoders enable competitive performance making them well suited to applications that require custom queries. The compactness of the representation also makes it well-suited to video analysis and other problems requiring inference across many images.

count=1
* Point-Supervised Semantic Segmentation of Natural Scenes via Hyperspectral Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/html/Ren_Point-Supervised_Semantic_Segmentation_of_Natural_Scenes_via_Hyperspectral_Imaging_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/papers/Ren_Point-Supervised_Semantic_Segmentation_of_Natural_Scenes_via_Hyperspectral_Imaging_CVPRW_2024_paper.pdf)]
    * Title: Point-Supervised Semantic Segmentation of Natural Scenes via Hyperspectral Imaging
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tianqi Ren, Qiu Shen, Ying Fu, Shaodi You
    * Abstract: Natural scene semantic segmentation is an important task in computer vision. While training accurate models for semantic segmentation relies heavily on detailed and accurate pixel-level annotations which are hard and time-consuming to be collected especially for complicated natural scenes. Weakly-supervised methods can reduce labeling cost greatly at the expense of significant performance degradation. In this paper we explore the possibility of introducing hyperspectral imaging to improve the performance of weakly-supervised semantic segmentation. Specifically we take two challenging hyperspectral datasets of outdoor natural scenes as example and randomly label dozens of points with semantic categories to conduct a point-supervised semantic segmentation benchmark. Then we propose a spectral and spatial fusion method to generate detailed pixel-level annotations which are used to supervise the semantic segmentation models. With multiple experiments we find that hyperspectral information can be greatly helpful to point-supervised semantic segmentation as it is more distinctive than RGB. As a result our proposed method with only point-supervision can achieve approximate 90% performance of the fully-supervised method in many cases.

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
* Road Segmentation Using Multipass Single-Pol Synthetic Aperture Radar Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/Koch_Road_Segmentation_Using_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/Koch_Road_Segmentation_Using_2015_CVPR_paper.pdf)]
    * Title: Road Segmentation Using Multipass Single-Pol Synthetic Aperture Radar Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Mark W. Koch, Mary M. Moya, James G. Chow, Jeremy Goold, Rebecca Malinas
    * Abstract: Synthetic aperture radar (SAR) is a remote sensing technology that can truly operate 24/7. It's an all-weather system that can operate at any time except in the most extreme conditions. By making multiple passes over a wide area, a SAR can provide surveillance over a long time period. For high level processing it is convenient to segment and classify the SAR images into objects that identify various terrains and man-made structures that we call "static features." In this paper we concentrate on automatic road segmentation. This not only serves as a surrogate for finding other static features, but road detection in of itself is important for aligning SAR images with other data sources. In this paper we introduce a novel SAR image product that captures how different regions decorrelate at different rates. We also show how a modified Kolmogorov-Smirnov test can be used to model the static features even when the independent observation assumption is violated.

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
* On-the-Fly Hand Detection Training With Application in Egocentric Action Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W15/html/Kumar_On-the-Fly_Hand_Detection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W15/papers/Kumar_On-the-Fly_Hand_Detection_2015_CVPR_paper.pdf)]
    * Title: On-the-Fly Hand Detection Training With Application in Egocentric Action Recognition
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Jayant Kumar, Qun Li, Survi Kyal, Edgar A. Bernal, Raja Bala
    * Abstract: We propose a novel approach to segment hand regions in egocentric video that requires no manual labeling of training samples. The user wearing a head-mounted camera is prompted to perform a simple gesture during an initial calibration step. A combination of color and motion analysis that exploits knowledge of the expected gesture is applied on the calibration video frames to automatically label hand pixels in an unsupervised fashion. The hand pixels identified in this manner are used to train a statistical-model based hand detector. Superpixel region growing is used to perform segmentation refinement and improve robustness to noise. Experiments show that our hand detection technique based on the proposed on the-fly training approach significantly outperforms state-of the-art techniques with respect to accuracy and robustness on a variety of challenging videos. This is due primarily to the fact that training samples are personalized to a specific user and environmental conditions. We also demonstrate the utility of our hand detection technique to inform an adaptive video sampling strategy that improves both computational speed and accuracy of egocentric action recognition algorithms. Finally, we offer an egocentric video dataset of an insulin self-injection procedure with action labels and hand masks that can serve towards future research on both hand detection and egocentric action recognition.

count=1
* Saliency Detection via Dense and Sparse Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Li_Saliency_Detection_via_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Li_Saliency_Detection_via_2013_ICCV_paper.pdf)]
    * Title: Saliency Detection via Dense and Sparse Reconstruction
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Xiaohui Li, Huchuan Lu, Lihe Zhang, Xiang Ruan, Ming-Hsuan Yang
    * Abstract: In this paper, we propose a visual saliency detection algorithm from the perspective of reconstruction errors. The image boundaries are first extracted via superpixels as likely cues for background templates, from which dense and sparse appearance models are constructed. For each image region, we first compute dense and sparse reconstruction errors. Second, the reconstruction errors are propagated based on the contexts obtained from K-means clustering. Third, pixel-level saliency is computed by an integration of multi-scale reconstruction errors and refined by an object-biased Gaussian model. We apply the Bayes formula to integrate saliency measures based on dense and sparse reconstruction errors. Experimental results show that the proposed algorithm performs favorably against seventeen state-of-the-art methods in terms of precision and recall. In addition, the proposed algorithm is demonstrated to be more effective in highlighting salient objects uniformly and robust to background noise.

count=1
* Semi-Supervised Normalized Cuts for Image Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Chew_Semi-Supervised_Normalized_Cuts_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Chew_Semi-Supervised_Normalized_Cuts_ICCV_2015_paper.pdf)]
    * Title: Semi-Supervised Normalized Cuts for Image Segmentation
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Selene E. Chew, Nathan D. Cahill
    * Abstract: Since its introduction as a powerful graph-based method for image segmentation, the Normalized Cuts (NCuts) algorithm has been generalized to incorporate expert knowledge about how certain pixels or regions should be grouped, or how the resulting segmentation should be biased to be correlated with priors. Previous approaches incorporate hard must-link constraints on how certain pixels should be grouped as well as hard cannot-link constraints on how other pixels should be separated into different groups. In this paper, we reformulate NCuts to allow both sets of constraints to be handled in a soft manner, enabling the user to tune the degree to which the constraints are satisfied. An approximate spectral solution to the reformulated problem exists without requiring explicit construction of a large, dense matrix; hence, computation time is comparable to that of unconstrained NCuts. Using synthetic data and real imagery, we show that soft handling of constraints yields better results than unconstrained NCuts and enables more robust clustering and segmentation than is possible when the constraints are strictly enforced.

count=1
* Cluster-Based Point Set Saliency
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Tasse_Cluster-Based_Point_Set_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Tasse_Cluster-Based_Point_Set_ICCV_2015_paper.pdf)]
    * Title: Cluster-Based Point Set Saliency
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Flora Ponjou Tasse, Jiri Kosinka, Neil Dodgson
    * Abstract: We propose a cluster-based approach to point set saliency detection, a challenge since point sets lack topological information. A point set is first decomposed into small clusters, using fuzzy clustering. We evaluate cluster uniqueness and spatial distribution of each cluster and combine these values into a cluster saliency function. Finally, the probabilities of points belonging to each cluster are used to assign a saliency to each point. Our approach detects fine-scale salient features and uninteresting regions consistently have lower saliency values. We evaluate the proposed saliency model by testing our saliency-based keypoint detection against a 3D interest point detection benchmark. The evaluation shows that our method achieves a good balance between false positive and false negative error rates, without using any topological information.

count=1
* SymmSLIC: Symmetry Aware Superpixel Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w24/html/Nagar_SymmSLIC_Symmetry_Aware_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w24/Nagar_SymmSLIC_Symmetry_Aware_ICCV_2017_paper.pdf)]
    * Title: SymmSLIC: Symmetry Aware Superpixel Segmentation
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Rajendra Nagar, Shanmuganathan Raman
    * Abstract: Over-segmentation of an image into superpixels has become an useful tool for solving various problems in computer vision. Reflection symmetry is quite prevalent in both natural and man-made objects. Existing algorithms for estimating superpixels do not preserve the reflection symmetry of an object which leads to different sizes and shapes of superpixels across the symmetry axis. In this work, we propose an algorithm to over-segment an image through the propagation of reflection symmetry evident at the pixel level to superpixel boundaries. In order to achieve this goal, we exploit the detection of a set of pairs of pixels which are mirror reflections of each other. We partition the image into superpixels while preserving this reflection symmetry information through an iterative algorithm. We compare the proposed method with state-of-the-art superpixel generation methods and show the effectiveness of the method in preserving the size and shape of superpixel boundaries across the reflection symmetry axes. We also present an application called unsupervised symmetric object segmentation to illustrate the effectiveness of the proposed approach.

count=1
* View-Consistent 4D Light Field Superpixel Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Khan_View-Consistent_4D_Light_Field_Superpixel_Segmentation_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Khan_View-Consistent_4D_Light_Field_Superpixel_Segmentation_ICCV_2019_paper.pdf)]
    * Title: View-Consistent 4D Light Field Superpixel Segmentation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Numair Khan,  Qian Zhang,  Lucas Kasser,  Henry Stone,  Min H. Kim,  James Tompkin
    * Abstract: Many 4D light field processing applications rely on superpixel segmentations, for which occlusion-aware view consistency is important. Yet, existing methods often enforce consistency by propagating clusters from a central view only, which can lead to inconsistent superpixels for non-central views. Our proposed approach combines an occlusion-aware angular segmentation in horizontal and vertical EPI spaces with an occlusion-aware clustering and propagation step across all views. Qualitative video demonstrations show that this helps to remove flickering and inconsistent boundary shapes versus the state-of-the-art approach, and quantitative metrics reflect these findings with improved boundary accuracy and view consistency scores.

count=1
* Deep Edge-Aware Interactive Colorization Against Color-Bleeding Effects
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Deep_Edge-Aware_Interactive_Colorization_Against_Color-Bleeding_Effects_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Deep_Edge-Aware_Interactive_Colorization_Against_Color-Bleeding_Effects_ICCV_2021_paper.pdf)]
    * Title: Deep Edge-Aware Interactive Colorization Against Color-Bleeding Effects
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Eungyeup Kim, Sanghyeon Lee, Jeonghoon Park, Somi Choi, Choonghyun Seo, Jaegul Choo
    * Abstract: Deep neural networks for automatic image colorization often suffer from the color-bleeding artifact, a problematic color spreading near the boundaries between adjacent objects. Such color-bleeding artifacts debase the reality of generated outputs, limiting the applicability of colorization models in practice. Although previous approaches have attempted to address this problem in an automatic manner, they tend to work only in limited cases where a high contrast of gray-scale values are given in an input image. Alternatively, leveraging user interactions would be a promising approach for solving this color-breeding artifacts. In this paper, we propose a novel edge-enhancing network for the regions of interest via simple user scribbles indicating where to enhance. In addition, our method requires a minimal amount of effort from users for their satisfactory enhancement. Experimental results demonstrate that our interactive edge-enhancing approach effectively improves the color-bleeding artifacts compared to the existing baselines across various datasets.

count=1
* WheatNet-Lite: A Novel Light Weight Network for Wheat Head Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/html/Bhagat_WheatNet-Lite_A_Novel_Light_Weight_Network_for_Wheat_Head_Detection_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/papers/Bhagat_WheatNet-Lite_A_Novel_Light_Weight_Network_for_Wheat_Head_Detection_ICCVW_2021_paper.pdf)]
    * Title: WheatNet-Lite: A Novel Light Weight Network for Wheat Head Detection
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Sandesh Bhagat, Manesh Kokare, Vineet Haswani, Praful Hambarde, Ravi Kamble
    * Abstract: Recently, the potential for wheat head detection has been significantly enhanced using deep learning techniques. However, the significant challenges are variation in growth stages of wheat heads, canopy, genotype, and wheat head orientation. Furthermore, the wheat head detection task gets even more complex due to the overlapping density of wheat heads and the blur image due to the wind. For real-time wheat head detection, designing lightweight deep learning models for edge devices is also challenging. This paper proposes a lightweight WheatNet-Lite architecture to enhance the efficiency and accuracy of wheat head detection. The proposed method utilizes Mixed Depthwise Conv (MDWConv) with an inverted residual bottleneck in the backbone. Additionally, the Modified Spatial Pyramidal Polling (MSPP) effectively extracts the multi-scale features. The final wheat head bounding box prediction is achieved using WheatNet-lite Neck by utilizing Depthwise Convolution (DWConv) with a Feature Pyramid structure. It reduces 54.2 M network parameters in comparison to YOLOV3. The proposed approach outperforms the existing state-of-the-art methods with mean average precision (mAP) of 91.32 mAP@0.5 and 86.10 mAP@0.5 on GWHD and SPIKE datasets, respectively, with only 8.2 M parameters. Also, the new ACID dataset is proposed with bounding box annotation with 76.32 mAP@0.5. The experimental results are demonstrated on three different datasets viz. Global Wheat Head Detection (GWHD), SPIKE dataset, and Annotated Crop Image Dataset (ACID) showing a significant improvement in the wheat head detection with speed and accuracy.

count=1
* A Semi-Self-Supervised Learning Approach for Wheat Head Detection Using Extremely Small Number of Labeled Samples
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/html/Najafian_A_Semi-Self-Supervised_Learning_Approach_for_Wheat_Head_Detection_Using_Extremely_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/papers/Najafian_A_Semi-Self-Supervised_Learning_Approach_for_Wheat_Head_Detection_Using_Extremely_ICCVW_2021_paper.pdf)]
    * Title: A Semi-Self-Supervised Learning Approach for Wheat Head Detection Using Extremely Small Number of Labeled Samples
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Keyhan Najafian, Alireza Ghanbari, Ian Stavness, Lingling Jin, Gholam Hassan Shirdel, Farhad Maleki
    * Abstract: Most of the success of deep learning is owed to supervised learning, where a large-scale annotated dataset is used for model training. However, developing such datasets is challenging. In this paper, we develop a semi-self-supervised learning approach for wheat head detection. The proposed method utilized a few short video clips and only one annotated image from each video clip of wheat fields to simulate a large computationally annotated dataset used for model building. Considering the domain gap between the simulated and real images, we applied two domain adaptation steps to alleviate the challenge of distributional shift. The resulting model achieved high performance when applied to real unannotated datasets. When fine-tuned on the dataset from the Global Wheat Head Detection Challenge, the performance was further improved. The model achieved a mean average precision of 0.827, where an overlap of 50% or more between a predicted bounding box and ground truth was considered as a correct prediction. Although the utility of the proposed methodology was shown by applying it to wheat head detection, the proposed method is not limited to this application and could be used for other domains, such as detecting different crop types, alleviating the barrier of lack of large-scale annotated datasets in those domains.

count=1
* Representing Objects in Video as Space-Time Volumes by Combining Top-Down and Bottom-Up Processes
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Ilic_Representing_Objects_in_Video_as_Space-Time_Volumes_by_Combining_Top-Down_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Ilic_Representing_Objects_in_Video_as_Space-Time_Volumes_by_Combining_Top-Down_WACV_2020_paper.pdf)]
    * Title: Representing Objects in Video as Space-Time Volumes by Combining Top-Down and Bottom-Up Processes
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Filip Ilic,  Axel Pinz
    * Abstract: As top-down based approaches of object recognition from video are getting more powerful, a structured way to combine them with bottom-up grouping processes becomes feasible. When done right, the resulting representation is able to describe objects and their decomposition into parts at appropriate spatio-temporal scales.We propose a method that uses a modern object detector to focus on salient structures in video, and a dense optical flow estimator to supplement feature extraction. From these structures we extract space-time volumes of interest (STVIs) by smoothing in spatio-temporal Gaussian Scale Space that guides bottom-up grouping.The resulting novel representation enables us to analyze and visualize the decomposition of an object into meaningful parts while preserving temporal object continuity. Our experimental validation is twofold. First, we achieve competitive results on a common video object segmentation benchmark. Second, we extend this benchmark with high quality object part annotations, DAVIS Parts, on which we establish a strong baseline by showing that our method yields spatio-temporally meaningful object parts. Our new representation will support applications that require high-level space-time reasoning at the parts level.

count=1
* SWAG-V: Explanations for Video Using Superpixels Weighted by Average Gradients
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Hartley_SWAG-V_Explanations_for_Video_Using_Superpixels_Weighted_by_Average_Gradients_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Hartley_SWAG-V_Explanations_for_Video_Using_Superpixels_Weighted_by_Average_Gradients_WACV_2022_paper.pdf)]
    * Title: SWAG-V: Explanations for Video Using Superpixels Weighted by Average Gradients
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Thomas Hartley, Kirill Sidorov, Christopher Willis, David Marshall
    * Abstract: CNN architectures that take videos as an input are often overlooked when it comes to the development of explanation techniques. This is despite their use in critical domains such as surveillance and healthcare. Explanation techniques developed for these networks must take into account the additional temporal domain if they are to be successful. In this paper we introduce SWAG-V, an extension of SWAG for use with networks that take video as an input. By creating superpixels that incorporate individual frames of the input video we are able to create explanations that better locate regions of the input that are important to the networks prediction. We demonstrate using Kinetics-400 with both the C3D and R(2+1)D network architectures that SWAG-V outperforms Grad-CAM, Grad-CAM++ and Saliency Tubes over a range of common metrics such as explanation accuracy and localisation.

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
* Efficient Few-Shot Learning for Pixel-Precise Handwritten Document Layout Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/De_Nardin_Efficient_Few-Shot_Learning_for_Pixel-Precise_Handwritten_Document_Layout_Analysis_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/De_Nardin_Efficient_Few-Shot_Learning_for_Pixel-Precise_Handwritten_Document_Layout_Analysis_WACV_2023_paper.pdf)]
    * Title: Efficient Few-Shot Learning for Pixel-Precise Handwritten Document Layout Analysis
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Axel De Nardin, Silvia Zottin, Matteo Paier, Gian Luca Foresti, Emanuela Colombi, Claudio Piciarelli
    * Abstract: Layout analysis is a task of uttermost importance in ancient handwritten document analysis and represents a fundamental step toward the simplification of subsequent tasks such as optical character recognition and automatic transcription. However, many of the approaches adopted to solve this problem rely on a fully supervised learning paradigm. While these systems achieve very good performance on this task, the drawback is that pixel-precise text labeling of the entire training set is a very time-consuming process, which makes this type of information rarely available in a real-world scenario. In the present paper, we address this problem by proposing an efficient few-shot learning framework that achieves performances comparable to current state-of-the-art fully supervised methods on the publicly available DIVA-HisDB dataset

count=1
* Image Segmentation-Based Unsupervised Multiple Objects Discovery
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Kara_Image_Segmentation-Based_Unsupervised_Multiple_Objects_Discovery_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Kara_Image_Segmentation-Based_Unsupervised_Multiple_Objects_Discovery_WACV_2023_paper.pdf)]
    * Title: Image Segmentation-Based Unsupervised Multiple Objects Discovery
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Sandra Kara, Hejer Ammar, Florian Chabot, Quoc-Cuong Pham
    * Abstract: Unsupervised object discovery aims to localize objects in images, while removing the dependence on annotations required by most deep learning-based methods. To address this problem, we propose a fully unsupervised, bottom-up approach, for multiple objects discovery. The proposed approach is a two-stage framework. First, instances of object parts are segmented by using the intra-image similarity between self-supervised local features. The second step merges and filters the object parts to form complete object instances. The latter is performed by two CNN models that capture semantic information on objects from the entire dataset. We demonstrate that the pseudo-labels generated by our method provide a better precision-recall trade-off than existing single and multiple objects discovery methods. In particular, we provide state-of-the-art results for both unsupervised class-agnostic object detection and unsupervised image segmentation.

count=1
* Unsupervised Video Object Segmentation via Prototype Memory Network
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Lee_Unsupervised_Video_Object_Segmentation_via_Prototype_Memory_Network_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Lee_Unsupervised_Video_Object_Segmentation_via_Prototype_Memory_Network_WACV_2023_paper.pdf)]
    * Title: Unsupervised Video Object Segmentation via Prototype Memory Network
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Minhyeok Lee, Suhwan Cho, Seunghoon Lee, Chaewon Park, Sangyoun Lee
    * Abstract: Unsupervised video object segmentation aims to segment a target object in the video without a ground truth mask in the initial frame. This challenging task requires extracting features for the most salient common objects within a video sequence. This difficulty can be solved by using motion information such as optical flow, but using only the information between adjacent frames results in poor connectivity between distant frames and poor performance. To solve this problem, we propose a novel prototype memory network architecture. The proposed model effectively extracts the RGB and motion information by extracting superpixel-based component prototypes from the input RGB images and optical flow maps. In addition, the model scores the usefulness of the component prototypes in each frame based on a self-learning algorithm and adaptively stores the most useful prototypes in memory and discards obsolete prototypes. We use the prototypes in the memory bank to predict the next query frame's mask, which enhances the association between distant frames to help with accurate mask prediction. Our method is evaluated on three datasets, achieving state-of-the-art performance. We prove the effectiveness of the proposed model with various ablation studies.

count=1
* Segment Any Point Cloud Sequences by Distilling Vision Foundation Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/753d9584b57ba01a10482f1ea7734a89-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/753d9584b57ba01a10482f1ea7734a89-Paper-Conference.pdf)]
    * Title: Segment Any Point Cloud Sequences by Distilling Vision Foundation Models
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Youquan Liu, Lingdong Kong, Jun CEN, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu
    * Abstract: Recent advancements in vision foundation models (VFMs) have opened up new possibilities for versatile and efficient visual perception. In this work, we introduce Seal, a novel framework that harnesses VFMs for segmenting diverse automotive point cloud sequences. Seal exhibits three appealing properties: i) Scalability: VFMs are directly distilled into point clouds, obviating the need for annotations in either 2D or 3D during pretraining. ii) Consistency: Spatial and temporal relationships are enforced at both the camera-to-LiDAR and point-to-segment regularization stages, facilitating cross-modal representation learning. iii) Generalizability: Seal enables knowledge transfer in an off-the-shelf manner to downstream tasks involving diverse point clouds, including those from real/synthetic, low/high-resolution, large/small-scale, and clean/corrupted datasets. Extensive experiments conducted on eleven different point cloud datasets showcase the effectiveness and superiority of Seal. Notably, Seal achieves a remarkable 45.0% mIoU on nuScenes after linear probing, surpassing random initialization by 36.9% mIoU and outperforming prior arts by 6.1% mIoU. Moreover, Seal demonstrates significant performance gains over existing methods across 20 different few-shot fine-tuning tasks on all eleven tested point cloud datasets. The code is available at this link.

count=1
* SmoothHess: ReLU Network Feature Interactions via Stein's Lemma
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/9ef5e965720193681fc8d16372ac4717-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/9ef5e965720193681fc8d16372ac4717-Paper-Conference.pdf)]
    * Title: SmoothHess: ReLU Network Feature Interactions via Stein's Lemma
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Max Torop, Aria Masoomi, Davin Hill, Kivanc Kose, Stratis Ioannidis, Jennifer Dy
    * Abstract: Several recent methods for interpretability model feature interactions by looking at the Hessian of a neural network. This poses a challenge for ReLU networks, which are piecewise-linear and thus have a zero Hessian almost everywhere. We propose SmoothHess, a method of estimating second-order interactions through Stein's Lemma. In particular, we estimate the Hessian of the network convolved with a Gaussian through an efficient sampling algorithm, requiring only network gradient calls. SmoothHess is applied post-hoc, requires no modifications to the ReLU network architecture, and the extent of smoothing can be controlled explicitly. We provide a non-asymptotic bound on the sample complexity of our estimation procedure. We validate the superior ability of SmoothHess to capture interactions on benchmark datasets and a real-world medical spirometry dataset.

