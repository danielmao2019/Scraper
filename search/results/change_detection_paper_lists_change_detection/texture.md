count=96
* Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.pdf)]
    * Title: Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Peri Akiva, Matthew Purri, Matthew Leotta
    * Abstract: Self-supervised learning aims to learn image feature representations without the usage of manually annotated labels. It is often used as a precursor step to obtain useful initial network weights which contribute to faster convergence and superior performance of downstream tasks. While self-supervision allows one to reduce the domain gap between supervised and unsupervised learning without the usage of labels, the self-supervised objective still requires a strong inductive bias to downstream tasks for effective transfer learning. In this work, we present our material and texture based self-supervision method named MATTER (MATerial and TExture Representation Learning), which is inspired by classical material and texture methods. Material and texture can effectively describe any surface, including its tactile properties, color, and specularity. By extension, effective representation of material and texture can describe other semantic classes strongly associated with said material and texture. MATTER leverages multi-temporal, spatially aligned remote sensing imagery over unchanged regions to learn invariance to illumination and viewing angle as a mechanism to achieve consistency of material and texture representation. We show that our self-supervision pre-training method allows for up to 24.22% and 6.33% performance increase in unsupervised and fine-tuned setups, and up to 76% faster convergence on change detection, land cover classification, and semantic segmentation tasks.

count=49
* A Spatiotemporal Oriented Energy Network for Dynamic Texture Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Hadji_A_Spatiotemporal_Oriented_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hadji_A_Spatiotemporal_Oriented_ICCV_2017_paper.pdf)]
    * Title: A Spatiotemporal Oriented Energy Network for Dynamic Texture Recognition
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Isma Hadji, Richard P. Wildes
    * Abstract: This paper presents a novel hierarchical spatiotemporal orientation representation for spacetime image analysis. It is designed to combine the benefits of the multilayer architecture of ConvNets and a more controlled approach to spacetime analysis. A distinguishing aspect of the approach is that unlike most contemporary convolutional networks no learning is involved; rather, all design decisions are specified analytically with theoretical motivations. This approach makes it possible to understand what information is being extracted at each stage and layer of processing as well as to minimize heuristic choices in design. Another key aspect of the network is its recurrent nature, whereby the output of each layer of processing feeds back to the input. To keep the network size manageable across layers, a novel cross-channel feature pooling is proposed. The multilayer architecture that results systematically reveals hierarchical image structure in terms of multiscale, multiorientation properties of visual spacetime. To illustrate its utility, the network has been applied to the task of dynamic texture recognition. Empirical evaluation on multiple standard datasets shows that it sets a new state-of-the-art.

count=33
* Building Bridges across Spatial and Temporal Resolutions: Reference-Based Super-Resolution via Change Priors and Conditional Diffusion Model
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Dong_Building_Bridges_across_Spatial_and_Temporal_Resolutions_Reference-Based_Super-Resolution_via_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Dong_Building_Bridges_across_Spatial_and_Temporal_Resolutions_Reference-Based_Super-Resolution_via_CVPR_2024_paper.pdf)]
    * Title: Building Bridges across Spatial and Temporal Resolutions: Reference-Based Super-Resolution via Change Priors and Conditional Diffusion Model
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Runmin Dong, Shuai Yuan, Bin Luo, Mengxuan Chen, Jinxiao Zhang, Lixian Zhang, Weijia Li, Juepeng Zheng, Haohuan Fu
    * Abstract: Reference-based super-resolution (RefSR) has the potential to build bridges across spatial and temporal resolutions of remote sensing images. However existing RefSR methods are limited by the faithfulness of content reconstruction and the effectiveness of texture transfer in large scaling factors. Conditional diffusion models have opened up new opportunities for generating realistic high-resolution images but effectively utilizing reference images within these models remains an area for further exploration. Furthermore content fidelity is difficult to guarantee in areas without relevant reference information. To solve these issues we propose a change-aware diffusion model named Ref-Diff for RefSR using the land cover change priors to guide the denoising process explicitly. Specifically we inject the priors into the denoising model to improve the utilization of reference information in unchanged areas and regulate the reconstruction of semantically relevant content in changed areas. With this powerful guidance we decouple the semantics-guided denoising and reference texture-guided denoising processes to improve the model performance. Extensive experiments demonstrate the superior effectiveness and robustness of the proposed method compared with state-of-the-art RefSR methods in both quantitative and qualitative evaluations. The code and data are available at https://github.com/dongrunmin/RefDiff.

count=20
* Parcel3D: Shape Reconstruction From Single RGB Images for Applications in Transportation Logistics
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Naumann_Parcel3D_Shape_Reconstruction_From_Single_RGB_Images_for_Applications_in_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/VISION/papers/Naumann_Parcel3D_Shape_Reconstruction_From_Single_RGB_Images_for_Applications_in_CVPRW_2023_paper.pdf)]
    * Title: Parcel3D: Shape Reconstruction From Single RGB Images for Applications in Transportation Logistics
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Alexander Naumann, Felix Hertlein, Laura Dörr, Kai Furmans
    * Abstract: We focus on enabling damage and tampering detection in logistics and tackle the problem of 3D shape reconstruction of potentially damaged parcels. As input we utilize single RGB images, which corresponds to use-cases where only simple handheld devices are available, e.g. for postmen during delivery or clients on delivery. We present a novel synthetic dataset, named Parcel3D, that is based on the Google Scanned Objects (GSO) dataset and consists of more than 13,000 images of parcels with full 3D annotations. The dataset contains intact, i.e. cuboid-shaped, parcels and damaged parcels, which were generated in simulations. We work towards detecting mishandling of parcels by presenting a novel architecture called CubeRefine R-CNN, which combines estimating a 3D bounding box with an iterative mesh refinement. We benchmark our approach on Parcel3D and an existing dataset of cuboid-shaped parcels in real-world scenarios. Our results show, that while training on Parcel3D enables transfer to the real world, enabling reliable deployment in real-world scenarios is still challenging. CubeRefine R-CNN yields competitive performance in terms of Mesh AP and is the only model that directly enables deformation assessment by 3D mesh comparison and tampering detection by comparing viewpoint invariant parcel side surface representations. Dataset and code are available at https://a-nau.github.io/parcel3d.

count=18
* Ensemble Video Object Cut in Highly Dynamic Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Ren_Ensemble_Video_Object_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ren_Ensemble_Video_Object_2013_CVPR_paper.pdf)]
    * Title: Ensemble Video Object Cut in Highly Dynamic Scenes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Xiaobo Ren, Tony X. Han, Zhihai He
    * Abstract: We consider video object cut as an ensemble of framelevel background-foreground object classifiers which fuses information across frames and refine their segmentation results in a collaborative and iterative manner. Our approach addresses the challenging issues of modeling of background with dynamic textures and segmentation of foreground objects from cluttered scenes. We construct patch-level bagof-words background models to effectively capture the background motion and texture dynamics. We propose a foreground salience graph (FSG) to characterize the similarity of an image patch to the bag-of-words background models in the temporal domain and to neighboring image patches in the spatial domain. We incorporate this similarity information into a graph-cut energy minimization framework for foreground object segmentation. The background-foreground classification results at neighboring frames are fused together to construct a foreground probability map to update the graph weights. The resulting object shapes at neighboring frames are also used as constraints to guide the energy minimization process during graph cut. Our extensive experimental results and performance comparisons over a diverse set of challenging videos with dynamic scenes, including the new Change Detection Challenge Dataset, demonstrate that the proposed ensemble video object cut method outperforms various state-ofthe-art algorithms.

count=18
* Fast Image Gradients Using Binary Feature Convolutions
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/St-Charles_Fast_Image_Gradients_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/St-Charles_Fast_Image_Gradients_CVPR_2016_paper.pdf)]
    * Title: Fast Image Gradients Using Binary Feature Convolutions
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: The recent increase in popularity of binary feature descriptors has opened the door to new lightweight computer vision applications. Most research efforts thus far have been dedicated to the introduction of new large-scale binary features, which are primarily used for keypoint description and matching. In this paper, we show that the side products of small-scale binary feature computations can efficiently filter images and estimate image gradients. The improved efficiency of low-level operations can be especially useful in time-constrained applications. Through our experiments, we show that efficient binary feature convolutions can be used to mimic various image processing operations, and even outperform Sobel gradient estimation in the edge detection problem, both in terms of speed and F-Measure.

count=16
* Detecting and Suppressing Marine Snow for Underwater Visual SLAM
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/IMW/html/Hodne_Detecting_and_Suppressing_Marine_Snow_for_Underwater_Visual_SLAM_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/IMW/papers/Hodne_Detecting_and_Suppressing_Marine_Snow_for_Underwater_Visual_SLAM_CVPRW_2022_paper.pdf)]
    * Title: Detecting and Suppressing Marine Snow for Underwater Visual SLAM
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Lars Martin Hodne, Eirik Leikvoll, Mauhing Yip, Andreas Langeland Teigen, Annette Stahl, Rudolf Mester
    * Abstract: Conventional SLAM methods which work very well in typical above-water situations, are based on detecting keypoints that are tracked between images, from which egomotion and the 3D structure of the scene are estimated. However, in underwater environments with marine snow -- small particles of organic matter which are carried by ocean currents throughout the water column -- keypoint detectors are prone to detect the marine snow particles. As the vast majority of SLAM front ends are sensitive against outliers, and the marine snow acts as severe "motion noise", failure of the regular egomotion and 3D structure estimation is expected. For this reason, we investigate the structure and appearance of marine snow and developed two schemes which classify keypoints into "marine snow" or "clean" based on either the image patches obtained from usual keypoint detectors or the descriptors computed from these patches. This way the subsequent SLAM pipeline is protected against 'false' keypoints. We quantitatively evaluate the performance of our marine snow classifier on both real underwater video scenes as well as on simulated underwater footage that contains marine snow. These simulated image sequences have been created by extracting real marine snow elements from real underwater footage, and subsequently overlaying these on "clean" underwater videos. Qualitative evaluation is also done on a nightime road sequence with snowfall to demonstrate applicability in other areas of autonomy. We furthermore evaluate the performance and the effect of marine snow detection & suppression by integrating the snow suppression module in a full SLAM pipeline based on the pySLAM system.

count=15
* Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.pdf)]
    * Title: Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Olaf Wysocki, Yan Xia, Magdalena Wysocki, Eleonora Grilli, Ludwig Hoegner, Daniel Cremers, Uwe Stilla
    * Abstract: Reconstructing semantic 3D building models at the level of detail (LoD) 3 is a long-standing challenge. Unlike mesh-based models, they require watertight geometry and object-wise semantics at the facade level. The principal challenge of such demanding semantic 3D reconstruction is reliable facade-level semantic segmentation of 3D input data. We present a novel method, called Scan2LoD3, that accurately reconstructs semantic LoD3 building models by improving facade-level semantic 3D segmentation. To this end, we leverage laser physics and 3D building model priors to probabilistically identify model conflicts. These probabilistic physical conflicts propose locations of model openings: Their final semantics and shapes are inferred in a Bayesian network fusing multimodal probabilistic maps of conflicts, 3D point clouds, and 2D images. To fulfill demanding LoD3 requirements, we use the estimated shapes to cut openings in 3D building priors and fit semantic 3D objects from a library of facade objects. Extensive experiments on the TUM city campus datasets demonstrate the superior performance of the proposed Scan2LoD3 over the state-of-the-art methods in facade-level detection, semantic segmentation, and LoD3 building model reconstruction. We believe our method can foster the development of probability-driven semantic 3D reconstruction at LoD3 since not only the high-definition reconstruction but also reconstruction confidence becomes pivotal for various applications such as autonomous driving and urban simulations.

count=13
* High-Speed Image Reconstruction Through Short-Term Plasticity for Spiking Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_High-Speed_Image_Reconstruction_Through_Short-Term_Plasticity_for_Spiking_Cameras_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_High-Speed_Image_Reconstruction_Through_Short-Term_Plasticity_for_Spiking_Cameras_CVPR_2021_paper.pdf)]
    * Title: High-Speed Image Reconstruction Through Short-Term Plasticity for Spiking Cameras
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yajing Zheng, Lingxiao Zheng, Zhaofei Yu, Boxin Shi, Yonghong Tian, Tiejun Huang
    * Abstract: Fovea, located in the centre of the retina, is specialized for high-acuity vision. Mimicking the sampling mechanism of the fovea, a retina-inspired camera, named spiking camera, is developed to record the external information with a sampling rate of 40,000 Hz, and outputs asynchronous binary spike streams. Although the temporal resolution of visual information is improved, how to reconstruct the scenes is still a challenging problem. In this paper, we present a novel high-speed image reconstruction model through the short-term plasticity (STP) mechanism of the brain. We derive the relationship between postsynaptic potential regulated by STP and the firing frequency of each pixel. By setting up the STP model at each pixel of the spiking camera, we can infer the scene radiance with the temporal regularity of the spike stream. Moreover, we show that STP can be used to distinguish the static and motion areas and further enhance the reconstruction results. The experimental results show that our methods achieve state-of-the-art performance in both image quality and computing time.

count=13
* Detection and Localization of Facial Expression Manipulations
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Mazaheri_Detection_and_Localization_of_Facial_Expression_Manipulations_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Mazaheri_Detection_and_Localization_of_Facial_Expression_Manipulations_WACV_2022_paper.pdf)]
    * Title: Detection and Localization of Facial Expression Manipulations
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Ghazal Mazaheri, Amit K. Roy-Chowdhury
    * Abstract: Concerns regarding the wide-spread use of forged images and videos in social media necessitate precise detection of such fraud. Facial manipulations can be created by Identity swap (DeepFake) or Expression swap. Contrary to the identity swap, which can easily be detected with novel deepfake detection methods, expression swap detection has not yet been addressed extensively. The importance of facial expressions in inter-person communication is known. Consequently, it is important to develop methods that can detect and localize manipulations in facial expressions. To this end, we present a novel framework to exploit the underlying feature representations of facial expressions learned from expression recognition models to identify the manipulated features. Using discriminative feature maps extracted from a facial expression recognition framework, our manipulation detector is able to localize the manipulated regions of input images and videos. On the Face2Face dataset, (abundant expression manipulation), and NeuralTextures dataset (facial expressions manipulation corresponding to the mouth regions), our method achieves higher accuracy for both classification and localization of manipulations compared to state-of-the-art methods. Furthermore, we demonstrate that our method performs at-par with the state-of-the-art methods in cases where the expression is not manipulated, but rather the identity is changed, leading to a generalized approach for facial manipulation detection.

count=13
* SyntheWorld: A Large-Scale Synthetic Dataset for Land Cover Mapping and Building Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Song_SyntheWorld_A_Large-Scale_Synthetic_Dataset_for_Land_Cover_Mapping_and_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Song_SyntheWorld_A_Large-Scale_Synthetic_Dataset_for_Land_Cover_Mapping_and_WACV_2024_paper.pdf)]
    * Title: SyntheWorld: A Large-Scale Synthetic Dataset for Land Cover Mapping and Building Change Detection
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Jian Song, Hongruixuan Chen, Naoto Yokoya
    * Abstract: Synthetic datasets, recognized for their cost effectiveness, play a pivotal role in advancing computer vision tasks and techniques. However, when it comes to remote sensing image processing, the creation of synthetic datasets becomes challenging due to the demand for larger-scale and more diverse 3D models. This complexity is compounded by the difficulties associated with real remote sensing datasets, including limited data acquisition and high annotation costs, which amplifies the need for high-quality synthetic alternatives. To address this, we present SyntheWorld, a synthetic dataset unparalleled in quality, diversity, and scale. It includes 40,000 images with submeter-level pixels and fine-grained land cover annotations of eight categories, and it also provides 40,000 pairs of bitemporal image pairs with building change annotations for building change detection. We conduct experiments on multiple benchmark remote sensing datasets to verify the effectiveness of SyntheWorld and to investigate the conditions under which our synthetic data yield advantages. The dataset is available at https://github.com/JTRNEO/SyntheWorld.

count=12
* Embedded Motion Detection via Neural Response Mixture Background Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/html/Shafiee_Embedded_Motion_Detection_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/papers/Shafiee_Embedded_Motion_Detection_CVPR_2016_paper.pdf)]
    * Title: Embedded Motion Detection via Neural Response Mixture Background Modeling
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Mohammad Javad Shafiee, Parthipan Siva, Paul Fieguth, Alexander Wong
    * Abstract: Recent studies have shown that deep neural networks (DNNs) can outperform state-of-the-art for a multitude of computer vision tasks. However, the ability to leverage DNNs for near real-time performance on embedded systems have been all but impossible so far without requiring specialized processors or GPUs. In this paper, we present a new motion detection algorithm that leverages the power of DNNs while maintaining low computational complexity needed for near real-time embedded performance without specialized hardware. The proposed Neural Response Mixture (NeRM) model leverages rich deep features extracted from the neural responses of an efficient, stochastically-formed deep neural network for constructing Gaussian mixture models to detect moving objects in a scene. NeRM was implemented on an embedded system on an Axis surveillance camera, and results demonstrated that the proposed NeRM approach can strong motion detection accuracy while operating at near real-time performance.

count=11
* Patches, Planes and Probabilities: A Non-Local Prior for Volumetric 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Ulusoy_Patches_Planes_and_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Ulusoy_Patches_Planes_and_CVPR_2016_paper.pdf)]
    * Title: Patches, Planes and Probabilities: A Non-Local Prior for Volumetric 3D Reconstruction
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Ali Osman Ulusoy, Michael J. Black, Andreas Geiger
    * Abstract: In this paper, we propose a non-local structured prior for volumetric multi-view 3D reconstruction. Towards this goal, we present a novel Markov random field model based on ray potentials in which assumptions about large 3D surface patches such as planarity or Manhattan world constraints can be efficiently encoded as probabilistic priors. We further derive an inference algorithm that reasons jointly about voxels, pixels and image segments, and estimates marginal distributions of appearance, occupancy, depth, normals and planarity. Key to tractable inference is a novel hybrid representation that spans both voxel and pixel space and that integrates non-local information from 2D image segmentations in a principled way. We compare our non-local prior to commonly employed local smoothness assumptions and a variety of state-of-the-art volumetric reconstruction baselines on challenging outdoor scenes with textureless and reflective surfaces. Our experiments indicate that regularizing over larger distances has the potential to resolve ambiguities where local regularizers fail.

count=11
* Background Subtraction Using Local SVD Binary Pattern
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/Guo_Background_Subtraction_Using_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/Guo_Background_Subtraction_Using_CVPR_2016_paper.pdf)]
    * Title: Background Subtraction Using Local SVD Binary Pattern
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Lili Guo, Dan Xu, Zhenping Qiang
    * Abstract: Background subtraction is a basic problem for change detection in videos and also the first step of high-level computer vision applications. Most background subtraction methods rely on color and texture feature. However, due to illuminations changes in different scenes and affections of noise pixels, those methods often resulted in high false positives in a complex environment. To solve this problem, we propose an adaptive background subtraction model which uses a novel Local SVD Binary Pattern (named LSBP) feature instead of simply depending on color intensity. This feature can describe the potential structure of the local regions in a given image, thus, it can enhance the robustness to illumination variation, noise, and shadows. We use a sample consensus model which is well suited for our LSBP feature. Experimental results on CDnet 2012 dataset demonstrate that our background subtraction method using LSBP feature is more effective than many state-of-the-art methods.

count=11
* SPIDeRS: Structured Polarization for Invisible Depth and Reflectance Sensing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ichikawa_SPIDeRS_Structured_Polarization_for_Invisible_Depth_and_Reflectance_Sensing_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ichikawa_SPIDeRS_Structured_Polarization_for_Invisible_Depth_and_Reflectance_Sensing_CVPR_2024_paper.pdf)]
    * Title: SPIDeRS: Structured Polarization for Invisible Depth and Reflectance Sensing
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tomoki Ichikawa, Shohei Nobuhara, Ko Nishino
    * Abstract: Can we capture shape and reflectance in stealth? Such capability would be valuable for many application domains in vision xR robotics and HCI. We introduce structured polarization for invisible depth and reflectance sensing (SPIDeRS) the first depth and reflectance sensing method using patterns of polarized light. The key idea is to modulate the angle of linear polarization (AoLP) of projected light at each pixel. The use of polarization makes it invisible and lets us recover not only depth but also directly surface normals and even reflectance. We implement SPIDeRS with a liquid crystal spatial light modulator (SLM) and a polarimetric camera. We derive a novel method for robustly extracting the projected structured polarization pattern from the polarimetric object appearance. We evaluate the effectiveness of SPIDeRS by applying it to a number of real-world objects. The results show that our method successfully reconstructs object shapes of various materials and is robust to diffuse reflection and ambient light. We also demonstrate relighting using recovered surface normals and reflectance. We believe SPIDeRS opens a new avenue of polarization use in visual sensing.

count=11
* A Generic Deformation Model for Dense Non-rigid Surface Registration: A Higher-Order MRF-Based Approach
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Zeng_A_Generic_Deformation_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Zeng_A_Generic_Deformation_2013_ICCV_paper.pdf)]
    * Title: A Generic Deformation Model for Dense Non-rigid Surface Registration: A Higher-Order MRF-Based Approach
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Yun Zeng, Chaohui Wang, Xianfeng Gu, Dimitris Samaras, Nikos Paragios
    * Abstract: We propose a novel approach for dense non-rigid 3D surface registration, which brings together Riemannian geometry and graphical models. To this end, we first introduce a generic deformation model, called Canonical Distortion Coefficients (CDCs), by characterizing the deformation of every point on a surface using the distortions along its two principle directions. This model subsumes the deformation groups commonly used in surface registration such as isometry and conformality, and is able to handle more complex deformations. We also derive its discrete counterpart which can be computed very efficiently in a closed form. Based on these, we introduce a higher-order Markov Random Field (MRF) model which seamlessly integrates our deformation model and a geometry/texture similarity metric. Then we jointly establish the optimal correspondences for all the points via maximum a posteriori (MAP) inference. Moreover, we develop a parallel optimization algorithm to efficiently perform the inference for the proposed higher-order MRF model. The resulting registration algorithm outperforms state-of-the-art methods in both dense non-rigid 3D surface registration and tracking.

count=11
* Simultaneous Foreground Detection and Classification With Hybrid Features
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Kim_Simultaneous_Foreground_Detection_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Kim_Simultaneous_Foreground_Detection_ICCV_2015_paper.pdf)]
    * Title: Simultaneous Foreground Detection and Classification With Hybrid Features
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Jaemyun Kim, Adin Ramirez Rivera, Byungyong Ryu, Oksam Chae
    * Abstract: In this paper, we propose a hybrid background model that relies on edge and non-edge features of the image to produce the model. We encode these features into a coding scheme, that we called Local Hybrid Pattern (LHP), that selectively models edges and non-edges features of each pixel. Furthermore, we model each pixel with an adaptive code dictionary to represent the background dynamism, and update it by adding stable codes and discarding unstable ones. We weight each code in the dictionary to enhance its description of the pixel it models. The foreground is detected as the incoming codes that deviate from the dictionary. We can detect (as foreground or background) and classify (as edge or inner region) each pixel simultaneously. We tested our proposed method in existing databases with promising results.

count=9
* Large-Scale Localization Datasets in Crowded Indoor Spaces
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Large-Scale_Localization_Datasets_in_Crowded_Indoor_Spaces_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Large-Scale_Localization_Datasets_in_Crowded_Indoor_Spaces_CVPR_2021_paper.pdf)]
    * Title: Large-Scale Localization Datasets in Crowded Indoor Spaces
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Donghwan Lee, Soohyun Ryu, Suyong Yeon, Yonghan Lee, Deokhwa Kim, Cheolho Han, Yohann Cabon, Philippe Weinzaepfel, Nicolas Guerin, Gabriela Csurka, Martin Humenberger
    * Abstract: Estimating the precise location of a camera using visual localization enables interesting applications such as augmented reality or robot navigation. This is particularly useful in indoor environments where other localization technologies, such as GNSS, fail. Indoor spaces impose interesting challenges on visual localization algorithms: occlusions due to people, textureless surfaces, large viewpoint changes, low light, repetitive textures, etc. Existing indoor datasets are either comparably small or do only cover a subset of the mentioned challenges. In this paper, we introduce 5 new indoor datasets for visual localization in challenging real-world environments. They were captured in a large shopping mall and a large metro station in Seoul, South Korea, using a dedicated mapping platform consisting of 10 cameras and 2 laser scanners. In order to obtain accurate ground truth camera poses, we developed a robust LiDAR SLAM which provides initial poses that are then refined using a novel structure-from-motion based optimization. We present a benchmark of modern visual localization algorithms on these challenging datasets showing superior performance of structure-based methods using robust image features. The datasets are available at: https://naverlabs.com/datasets

count=9
* HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Bandara_HyperTransformer_A_Textural_and_Spectral_Feature_Fusion_Transformer_for_Pansharpening_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bandara_HyperTransformer_A_Textural_and_Spectral_Feature_Fusion_Transformer_for_Pansharpening_CVPR_2022_paper.pdf)]
    * Title: HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Wele Gedara Chaminda Bandara, Vishal M. Patel
    * Abstract: Pansharpening aims to fuse a registered high-resolution panchromatic image (PAN) with a low-resolution hyperspectral image (LR-HSI) to generate an enhanced HSI with high spectral and spatial resolution. Existing pansharpening approaches neglect using an attention mechanism to transfer HR texture features from PAN to LR-HSI features, resulting in spatial and spectral distortions. In this paper, we present a novel attention mechanism for pansharpening called HyperTransformer, in which features of LR-HSI and PAN are formulated as queries and keys in a transformer, respectively. HyperTransformer consists of three main modules, namely two separate feature extractors for PAN and HSI, a multi-head feature soft attention module, and a spatial-spectral feature fusion module. Such a network improves both spatial and spectral quality measures of the pansharpened HSI by learning cross-feature space dependencies and long-range details of PAN and LR-HSI. Furthermore, HyperTransformer can be utilized across multiple spatial scales at the backbone for obtaining improved performance. Extensive experiments conducted on three widely used datasets demonstrate that HyperTransformer achieves significant improvement over the state-of-the-art methods on both spatial and spectral quality measures. Implementation code and pre-trained weights can be accessed at https://github.com/wgcban/HyperTransformer.

count=8
* Underwater Moving Object Detection Using an End-to-End Encoder-Decoder Architecture and GraphSage With Aggregator and Refactoring
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/WiCV/html/Kapoor_Underwater_Moving_Object_Detection_Using_an_End-to-End_Encoder-Decoder_Architecture_and_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/WiCV/papers/Kapoor_Underwater_Moving_Object_Detection_Using_an_End-to-End_Encoder-Decoder_Architecture_and_CVPRW_2023_paper.pdf)]
    * Title: Underwater Moving Object Detection Using an End-to-End Encoder-Decoder Architecture and GraphSage With Aggregator and Refactoring
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Meghna Kapoor, Suvam Patra, Badri Narayan Subudhi, Vinit Jakhetiya, Ankur Bansal
    * Abstract: Underwater environments are greatly affected by several factors, including low visibility, high turbidity, back-scattering, dynamic background, etc., and hence pose challenges in object detection. Several algorithms consider convolutional neural networks to extract deep features and then object detection using the same. However, the dependency on the kernel's size and the network's depth results in fading relationships of latent space features and also are unable to characterize the spatial-contextual bonding of the pixels. Hence, they are unable to procure satisfactory results in complex underwater scenarios. To re-establish this relationship, we propose a unique architecture for underwater object detection where U-Net architecture is considered with the ResNet-50 backbone. Further, the latent space features from the encoder are fed to the decoder through a GraphSage model. GraphSage-based model is explored to reweight the node relationship in non-euclidean space using different aggregator functions and hence characterize the spatio-contextual bonding among the pixels. Further, we explored the dependency on different aggregator functions: mean, max, and LSTM, to evaluate the model's performance. We evaluated the proposed model on two underwater benchmark databases: F4Knowledge and underwater change detection. The performance of the proposed model is evaluated against eleven state-of-the-art techniques in terms of both visual and quantitative evaluation measures.

count=7
* Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.pdf)]
    * Title: Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jhony H. Giraldo, Sajid Javed, Naoufel Werghi, Thierry Bouwmans
    * Abstract: Moving Object Detection (MOD) is a fundamental step for many computer vision applications. MOD becomes very challenging when a video sequence captured from a static or moving camera suffers from the challenges: camouflage, shadow, dynamic backgrounds, and lighting variations, to name a few. Deep learning methods have been successfully applied to address MOD with competitive performance. However, in order to handle the overfitting problem, deep learning methods require a large amount of labeled data which is a laborious task as exhaustive annotations are always not available. Moreover, some MOD deep learning methods show performance degradation in the presence of unseen video sequences because the testing and training splits of the same sequences are involved during the network learning process. In this work, we pose the problem of MOD as a node classification problem using Graph Convolutional Neural Networks (GCNNs). Our algorithm, dubbed as GraphMOD-Net, encompasses instance segmentation, background initialization, feature extraction, and graph construction. GraphMOD-Net is tested on unseen videos and outperforms state-of-the-art methods in unsupervised, semi-supervised, and supervised learning in several challenges of the Change Detection 2014 (CDNet2014) and UCSD background subtraction datasets.

count=6
* Online Dominant and Anomalous Behavior Detection in Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Roshtkhari_Online_Dominant_and_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Roshtkhari_Online_Dominant_and_2013_CVPR_paper.pdf)]
    * Title: Online Dominant and Anomalous Behavior Detection in Videos
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Mehrsan Javan Roshtkhari, Martin D. Levine
    * Abstract: We present a novel approach for video parsing and simultaneous online learning of dominant and anomalous behaviors in surveillance videos. Dominant behaviors are those occurring frequently in videos and hence, usually do not attract much attention. They can be characterized by different complexities in space and time, ranging from a scene background to human activities. In contrast, an anomalous behavior is defined as having a low likelihood of occurrence. We do not employ any models of the entities in the scene in order to detect these two kinds of behaviors. In this paper, video events are learnt at each pixel without supervision using densely constructed spatio-temporal video volumes. Furthermore, the volumes are organized into large contextual graphs. These compositions are employed to construct a hierarchical codebook model for the dominant behaviors. By decomposing spatio-temporal contextual information into unique spatial and temporal contexts, the proposed framework learns the models of the dominant spatial and temporal events. Thus, it is ultimately capable of simultaneously modeling high-level behaviors as well as low-level spatial, temporal and spatio-temporal pixel level changes.

count=6
* Human vs. Computer in Scene and Object Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Borji_Human_vs._Computer_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Borji_Human_vs._Computer_2014_CVPR_paper.pdf)]
    * Title: Human vs. Computer in Scene and Object Recognition
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Ali Borji, Laurent Itti
    * Abstract: Several decades of research in computer and primate vision have resulted in many models (some specialized for one problem, others more general) and invaluable experimental data. Here, to help focus research efforts onto the hardest unsolved problems, and bridge computer and human vision, we define a battery of 5 tests that measure the gap between human and machine performances in several dimensions (generalization across scene categories, generalization from images to edge maps and line drawings, invariance to rotation and scaling, local/global information with jumbled images, and object recognition performance). We measure model accuracy and the correlation between model and human error patterns. Experimenting over 7 datasets, where human data is available, and gauging 14 well-established models, we find that none fully resembles humans in all aspects, and we learn from each test which models and features are more promising in approaching humans in the tested dimension. Across all tests, we find that models based on local edge histograms consistently resemble humans more, while several scene statistics or "gist" models do perform well with both scenes and objects. While computer vision has long been inspired by human vision, we believe systematic efforts, such as this, will help better identify shortcomings of models and find new paths forward.

count=6
* Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.pdf)]
    * Title: Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ke Li, Di Wang, Zhangyuan Hu, Wenxuan Zhu, Shaofeng Li, Quan Wang
    * Abstract: Deep Convolutional Neural Networks (DCNNs) have achieved remarkable performance in synthetic aperture radar (SAR) object detection but this comes at the cost of tremendous computational resources partly due to extracting redundant features within a single convolutional layer. Recent works either delve into model compression methods or focus on the carefully-designed lightweight models both of which result in performance degradation. In this paper we propose an efficient convolution module for SAR object detection called SFS-Conv which increases feature diversity within each convolutional layer through a shunt-perceive-select strategy. Specifically we shunt input feature maps into space and frequency aspects. The former perceives the context of various objects by dynamically adjusting receptive field while the latter captures abundant frequency variations and textural features via fractional Gabor transformer. To adaptively fuse features from space and frequency aspects a parameter-free feature selection module is proposed to ensure that the most representative and distinctive information are preserved. With SFS-Conv we build a lightweight SAR object detection network called SFS-CNet. Experimental results show that SFS-CNet outperforms state-of-the-art (SoTA) models on a series of SAR object detection benchmarks while simultaneously reducing both the model size and computational cost.

count=6
* Robust Change Captioning
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Robust_Change_Captioning_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Robust_Change_Captioning_ICCV_2019_paper.pdf)]
    * Title: Robust Change Captioning
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Dong Huk Park,  Trevor Darrell,  Anna Rohrbach
    * Abstract: Describing what has changed in a scene can be useful to a user, but only if generated text focuses on what is semantically relevant. It is thus important to distinguish distractors (e.g. a viewpoint change) from relevant changes (e.g. an object has moved). We present a novel Dual Dynamic Attention Model (DUDA) to perform robust Change Captioning. Our model learns to distinguish distractors from semantic changes, localize the changes via Dual Attention over "before" and "after" images, and accurately describe them in natural language via Dynamic Speaker, by adaptively focusing on the necessary visual inputs (e.g. "before" or "after" image). To study the problem in depth, we collect a CLEVR-Change dataset, built off the CLEVR engine, with 5 types of scene changes. We benchmark a number of baselines on our dataset, and systematically study different change types and robustness to distractors. We show the superiority of our DUDA model in terms of both change captioning and localization. We also show that our approach is general, obtaining state-of-the-art results on the recent realistic Spot-the-Diff dataset which has no distractors.

count=6
* TAMPAR: Visual Tampering Detection for Parcel Logistics in Postal Supply Chains
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Naumann_TAMPAR_Visual_Tampering_Detection_for_Parcel_Logistics_in_Postal_Supply_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Naumann_TAMPAR_Visual_Tampering_Detection_for_Parcel_Logistics_in_Postal_Supply_WACV_2024_paper.pdf)]
    * Title: TAMPAR: Visual Tampering Detection for Parcel Logistics in Postal Supply Chains
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Alexander Naumann, Felix Hertlein, Laura Dörr, Kai Furmans
    * Abstract: Due to the steadily rising amount of valuable goods in supply chains, tampering detection for parcels is becoming increasingly important. In this work, we focus on the use-case last-mile delivery, where only a single RGB image is taken and compared against a reference from an existing database to detect potential appearance changes that indicate tampering. We propose a tampering detection pipeline that utilizes keypoint detection to identify the eight corner points of a parcel. This permits applying a perspective transformation to create normalized fronto-parallel views for each visible parcel side surface. These viewpoint-invariant parcel side surface representations facilitate the identification of signs of tampering on parcels within the supply chain, since they reduce the problem to parcel side surface matching with pair-wise appearance change detection. Experiments with multiple classical and deep learning-based change detection approaches are performed on our newly collected TAMpering detection dataset for PARcels, called TAMPAR. We evaluate keypoint and change detection separately, as well as in a unified system for tampering detection. Our evaluation shows promising results for keypoint (Keypoint AP 75.76) and tampering detection (81% accuracy, F1-Score 0.83) on real images. Furthermore, a sensitivity analysis for tampering types, lens distortion and viewing angles is presented. Code and dataset are available at https://a-nau.github.io/tampar.

count=5
* Towards Progressive Multi-Frequency Representation for Image Warping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Towards_Progressive_Multi-Frequency_Representation_for_Image_Warping_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Towards_Progressive_Multi-Frequency_Representation_for_Image_Warping_CVPR_2024_paper.pdf)]
    * Title: Towards Progressive Multi-Frequency Representation for Image Warping
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jun Xiao, Zihang Lyu, Cong Zhang, Yakun Ju, Changjian Shui, Kin-Man Lam
    * Abstract: Image warping a classic task in computer vision aims to use geometric transformations to change the appearance of images. Recent methods learn the resampling kernels for warping through neural networks to estimate missing values in irregular grids which however fail to capture local variations in deformed content and produce images with distortion and less high-frequency details. To address this issue this paper proposes an effective method namely MFR to learn Multi-Frequency Representations from input images for image warping. Specifically we propose a progressive filtering network to learn image representations from different frequency subbands and generate deformable images in a coarse-to-fine manner. Furthermore we employ learnable Gabor wavelet filters to improve the model's capability to learn local spatial-frequency representations. Comprehensive experiments including homography transformation equirectangular to perspective projection and asymmetric image super-resolution demonstrate that the proposed MFR significantly outperforms state-of-the-art image warping methods. Our method also showcases superior generalization to out-of-distribution domains where the generated images are equipped with rich details and less distortion thereby high visual quality. The source code is available at https://github.com/junxiao01/MFR.

count=5
* Integrating Dashcam Views Through Inter-Video Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Chen_Integrating_Dashcam_Views_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_Integrating_Dashcam_Views_ICCV_2015_paper.pdf)]
    * Title: Integrating Dashcam Views Through Inter-Video Mapping
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Hsin-I Chen, Yi-Ling Chen, Wei-Tse Lee, Fan Wang, Bing-Yu Chen
    * Abstract: In this paper, an inter-video mapping approach is proposed to integrate video footages from two dashcams installed on a preceding and its following vehicle to provide the illusion that the driver of the following vehicle can see-through the preceding one. The key challenge is to adapt the perspectives of the two videos based on a small number of common features since a large portion of the common region in the video captured by the following vehicle is occluded by the preceding one. Inspired by the observation that images with the most similar viewpoints yield dense and high-quality matches, the proposed inter-video mapping estimates spatially-varying motions across two videos utilizing images of very similar contents. Specifically, we estimate frame-to-frame motions of each two consecutive images and incrementally add new views into a merged representation. In this way, long-rang motion estimation is achieved, and the observed perspective discrepancy between the two videos can be well approximated our motion estimation. Once the inter-video mapping is established, the correspondences can be updated incrementally, so the proposed method is suitable for on-line applications. Our experiments demonstrate the effectiveness of our approach on real-world challenging videos.

count=4
* Modeling Actions through State Changes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Fathi_Modeling_Actions_through_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Fathi_Modeling_Actions_through_2013_CVPR_paper.pdf)]
    * Title: Modeling Actions through State Changes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Alireza Fathi, James M. Rehg
    * Abstract: In this paper we present a model of action based on the change in the state of the environment. Many actions involve similar dynamics and hand-object relationships, but differ in their purpose and meaning. The key to differentiating these actions is the ability to identify how they change the state of objects and materials in the environment. We propose a weakly supervised method for learning the object and material states that are necessary for recognizing daily actions. Once these state detectors are learned, we can apply them to input videos and pool their outputs to detect actions. We further demonstrate that our method can be used to segment discrete actions from a continuous video of an activity. Our results outperform state-of-the-art action recognition and activity segmentation results.

count=4
* Solving Temporal Puzzles
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Dicle_Solving_Temporal_Puzzles_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Dicle_Solving_Temporal_Puzzles_CVPR_2016_paper.pdf)]
    * Title: Solving Temporal Puzzles
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Caglayan Dicle, Burak Yilmaz, Octavia Camps, Mario Sznaier
    * Abstract: Many physical phenomena, within short time windows, can be explained by low order differential relations. In a discrete world, these relations can be described using low order difference equations or equivalently low order auto regressive (AR) models. In this paper, based on this intuition, we propose an algorithm for solving time-sort temporal puzzles, defined as scrambled time series that need to be sorted out. We frame this highly combinatorial problem using a mixed-integer semi definite programming formulation and show how to turn it into a mixed-integer linear programming problem by using the recently introduced atomic norm framework. Our experiments show the effectiveness and generality of our approach in different scenarios.

count=4
* The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.pdf)]
    * Title: The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, Antonio M. Lopez
    * Abstract: Vision-based semantic segmentation in urban scenarios is a key functionality for autonomous driving. Recent revolutionary results of deep convolutional neural networks (DCNNs) foreshadow the advent of reliable classifiers to perform such visual tasks. However, DCNNs require learning of many parameters from raw images; thus, having a sufficient amount of diverse images with class annotations is needed. These annotations are obtained via cumbersome, human labour which is particularly challenging for semantic segmentation since pixel-level annotations are required. In this paper, we propose to use a virtual world to automatically generate realistic synthetic images with pixel-level annotations. Then, we address the question of how useful such data can be for semantic segmentation -- in particular, when using a DCNN paradigm. In order to answer this question we have generated a synthetic collection of diverse urban images, named SYNTHIA, with automatically generated class annotations. We use SYNTHIA in combination with publicly available real-world urban images with manually provided annotations. Then, we conduct experiments with DCNNs that show how the inclusion of SYNTHIA in the training stage significantly improves performance on the semantic segmentation task.

count=4
* Semantic Multi-View Stereo: Jointly Estimating Objects and Voxels
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Ulusoy_Semantic_Multi-View_Stereo_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ulusoy_Semantic_Multi-View_Stereo_CVPR_2017_paper.pdf)]
    * Title: Semantic Multi-View Stereo: Jointly Estimating Objects and Voxels
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Ali Osman Ulusoy, Michael J. Black, Andreas Geiger
    * Abstract: Dense 3D reconstruction from RGB images is a highly ill-posed problem due to occlusions, textureless or reflective surfaces, as well as other challenges. We propose object-level shape priors to address these ambiguities. Towards this goal, we formulate a probabilistic model that integrates multi-view image evidence with 3D shape information from multiple objects. Inference in this model yields a dense 3D reconstruction of the scene as well as the existence and precise 3D pose of the objects in it. Our approach is able to recover fine details not captured in the input shapes while defaulting to the input models in occluded regions where image evidence is weak. Due to its probabilistic nature, the approach is able to cope with the approximate geometry of the 3D models as well as input shapes that are not present in the scene. We evaluate the approach quantitatively on several challenging indoor and outdoor datasets.

count=4
* Image Change Captioning by Learning From an Auxiliary Task
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Hosseinzadeh_Image_Change_Captioning_by_Learning_From_an_Auxiliary_Task_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Hosseinzadeh_Image_Change_Captioning_by_Learning_From_an_Auxiliary_Task_CVPR_2021_paper.pdf)]
    * Title: Image Change Captioning by Learning From an Auxiliary Task
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Mehrdad Hosseinzadeh, Yang Wang
    * Abstract: We tackle the challenging task of image change captioning. The goal is to describe the subtle difference between two very similar images by generating a sentence caption. While the recent methods mainly focus on proposing new model architectures for this problem, we instead focus on an alternative training scheme. Inspired by the success of multi-task learning, we formulate a training scheme that uses an auxiliary task to improve the training of the change captioning network. We argue that the task of composed query image retrieval is a natural choice as the auxiliary task. Given two almost similar images as the input, the primary network generates a caption describing the fine change between those two images. Next, the auxiliary network is provided with the generated caption and one of those two images. It then tries to pick the second image among a set of candidates. This forces the primary network to generate detailed and precise captions via having an extra supervision loss by the auxiliary network. Furthermore, we propose a new scheme for selecting a negative set of candidates for the retrieval task that can effectively improve the performance. We show that the proposed training strategy performs well on the task of change captioning on benchmark datasets.

count=4
* Integrating LIDAR Range Scans and Photographs with Temporal Changes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W19/html/Morago_Integrating_LIDAR_Range_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W19/papers/Morago_Integrating_LIDAR_Range_2014_CVPR_paper.pdf)]
    * Title: Integrating LIDAR Range Scans and Photographs with Temporal Changes
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Brittany Morago, Giang Bui, Ye Duan
    * Abstract: Registering 2D and 3D data is a rapidly growing research area. Motivating much of this work is the fact that 3D range scans and 2D imagery provide different, but complementing information about the same subject. Combining these two perspectives leads to the creation of accurate 3D models that are texture mapped with high resolution color information. Imagery can even be obtained on different days and in different seasons and registered together to show how a scene has changed with time. Finding correspondences among data captured with different cameras and containing content and temporal changes can be a challenging task. We address these difficulties by presenting a contextual approach for finding 2D matches, performing 2D-3D fusion by solving the projection matrix of a camera directly from its relationship to highly accurate range scan points, and minimizing an energy function based on gradient information in a 3D depth image.

count=4
* A Comparison of Stereo and Multiview 3-D Reconstruction Using Cross-Sensor Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/Ozcanli_A_Comparison_of_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/Ozcanli_A_Comparison_of_2015_CVPR_paper.pdf)]
    * Title: A Comparison of Stereo and Multiview 3-D Reconstruction Using Cross-Sensor Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Ozge C. Ozcanli, Yi Dong, Joseph L. Mundy, Helen Webb, Riad Hammoud, Victor Tom
    * Abstract: High-resolution and accurate Digital Elevation Model (DEM) generation from satellite imagery is a challenging problem. In this work, a stereo 3-D reconstruction framework is outlined that is applicable to nonstereoscopic satellite image pairs that may be captured by different satellites. The orthographic height maps given by stereo reconstruction are compared to height maps given by a multiview approach based on Probabilistic Volumetric Representation (PVR). Height map qualities are measured in comparison to manually prepared ground-truth height maps in three sites from different parts of the world with urban, semi-urban and rural features. The results along with strengths and weaknesses of the two techniques are summarized.

count=4
* Effective Semantic Pixel Labelling With Convolutional Networks and Conditional Random Fields
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Paisitkriangkrai_Effective_Semantic_Pixel_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Paisitkriangkrai_Effective_Semantic_Pixel_2015_CVPR_paper.pdf)]
    * Title: Effective Semantic Pixel Labelling With Convolutional Networks and Conditional Random Fields
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Sakrapee Paisitkriangkrai, Jamie Sherrah, Pranam Janney, Anton Van-Den Hengel
    * Abstract: Large amounts of available training data and increasing computing power have led to the recent success of deep convolutional neural networks (CNN) on a large number of applications. In this paper, we propose an effective semantic pixel labelling using CNN features, hand-crafted features and Conditional Random Fields (CRFs). Both CNN and hand-crafted features are applied to dense image patches to produce per-pixel class probabilities. The CRF infers a labelling that smooths regions while respecting the edges present in the imagery. The method is applied to the ISPRS 2D semantic labelling challenge dataset with competitive classification accuracy.

count=4
* Dynamic Probabilistic Volumetric Models
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Ulusoy_Dynamic_Probabilistic_Volumetric_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Ulusoy_Dynamic_Probabilistic_Volumetric_2013_ICCV_paper.pdf)]
    * Title: Dynamic Probabilistic Volumetric Models
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Ali Osman Ulusoy, Octavian Biris, Joseph L. Mundy
    * Abstract: This paper presents a probabilistic volumetric framework for image based modeling of general dynamic 3-d scenes. The framework is targeted towards high quality modeling of complex scenes evolving over thousands of frames. Extensive storage and computational resources are required in processing large scale space-time (4-d) data. Existing methods typically store separate 3-d models at each time step and do not address such limitations. A novel 4-d representation is proposed that adaptively subdivides in space and time to explain the appearance of 3-d dynamic surfaces. This representation is shown to achieve compression of 4-d data and provide efficient spatio-temporal processing. The advances of the proposed framework is demonstrated on standard datasets using free-viewpoint video and 3-d tracking applications.

count=4
* Robust and Optimal Sum-of-Squares-Based Point-to-Plane Registration of Image Sets and Structured Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Paudel_Robust_and_Optimal_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Paudel_Robust_and_Optimal_ICCV_2015_paper.pdf)]
    * Title: Robust and Optimal Sum-of-Squares-Based Point-to-Plane Registration of Image Sets and Structured Scenes
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Danda Pani Paudel, Adlane Habed, Cedric Demonceaux, Pascal Vasseur
    * Abstract: This paper deals with the problem of registering a known structured 3D scene and its metric Structure-from-Motion (SfM) counterpart. The proposed work relies on a prior plane segmentation of the 3D scene and aligns the data obtained from both modalities by solving the point-to-plane assignment problem. An inliers-maximization approach within a Branch-and-Bound (BnB) search scheme is adopted. For the first time in this paper, a Sum-of-Squares optimization theory framework is employed for identifying point-to-plane mismatches (i.e. outliers) with certainty. This allows us to iteratively build potential inliers sets and converge to the solution satisfied by the largest number of point-to-plane assignments. Furthermore, our approach is boosted by new plane visibility conditions which are also introduced in this paper. Using this framework, we solve the registration problem in two cases: (i) a set of putative point-to-plane correspondences (with possibly overwhelmingly many outliers) is given as input and (ii) no initial correspondences are given. In both cases, our approach yields outstanding results in terms of robustness and optimality.

count=4
* 3D Semantic Label Transfer in Human-Robot Collaboration
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/html/Rozenberszki_3D_Semantic_Label_Transfer_in_Human-Robot_Collaboration_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/papers/Rozenberszki_3D_Semantic_Label_Transfer_in_Human-Robot_Collaboration_ICCVW_2021_paper.pdf)]
    * Title: 3D Semantic Label Transfer in Human-Robot Collaboration
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Dávid Rozenberszki, Gábor Sörös, Szilvia Szeier, András Lőrincz
    * Abstract: We tackle two practical problems in robotic scene understanding. First, the computational requirements of current semantic segmentation algorithms are prohibitive for typical robots. Second, the viewpoints of ground robots are quite different from typical human viewpoints of training datasets which may lead to misclassified objects from robot viewpoints. We present a system for sharing and reusing 3D semantic information between multiple agents with different viewpoints. We first co-localize all agents in the same coordinate system. Next, we create a 3D dense semantic model of the space from human viewpoints close to real time. Finally, by re-rendering the model's semantic labels (and/or depth maps) from the ground robots' own estimated viewpoints and sharing them over the network, we can give 3D semantic understanding to simpler agents. We evaluate the reconstruction quality and show how tiny robots can reuse knowledge about the space collected by more capable peers.

count=4
* Self-Supervised Object Detection from Egocentric Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.pdf)]
    * Title: Self-Supervised Object Detection from Egocentric Videos
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Peri Akiva, Jing Huang, Kevin J Liang, Rama Kovvuri, Xingyu Chen, Matt Feiszli, Kristin Dana, Tal Hassner
    * Abstract: Understanding the visual world from the perspective of humans (egocentric) has been a long-standing challenge in computer vision. Egocentric videos exhibit high scene complexity and irregular motion flows compared to typical video understanding tasks. With the egocentric domain in mind, we address the problem of self-supervised, class-agnostic object detection, which aims to locate all objects in a given view, regardless of category, without any annotations or pre-training weights. Our method, self-supervised object Detection from Egocentric VIdeos (DEVI), generalizes appearance-based methods to learn features that are category-specific and invariant to viewing angles and illumination conditions from highly ambiguous environments in an end-to-end manner. Our approach leverages typical human behavior and its egocentric perception to sample diverse views of the same objects for our multi-view and scale-regression loss functions. With our learned cluster residual module, we are able to effectively describe multi-category patches for better complex scene understanding. DEVI provides a boost in performance on recent egocentric datasets, with performance gains up to 4.11% AP50, 0.11% AR1, 1.32% AR10, and 5.03% AR100, while significantly reducing model complexity. We also demonstrate competitive performance on out-of-domain datasets without additional training or fine-tuning.

count=4
* An Interactive Method for Adaptive Acquisition in Reflectance Transformation Imaging for Cultural Heritage
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/e-Heritage/html/Khawaja_An_Interactive_Method_for_Adaptive_Acquisition_in_Reflectance_Transformation_Imaging_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/e-Heritage/papers/Khawaja_An_Interactive_Method_for_Adaptive_Acquisition_in_Reflectance_Transformation_Imaging_ICCVW_2023_paper.pdf)]
    * Title: An Interactive Method for Adaptive Acquisition in Reflectance Transformation Imaging for Cultural Heritage
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Muhammad Arsalan Khawaja, Sony George, Franck Marzani, Jon Yngve Hardeberg, Alamin Mansouri
    * Abstract: This paper investigates the optimization of acquisition in Reflectance Transformation Imaging (RTI). Current methods for RTI acquisition are either computationally expensive or impractical, which leads to continued reliance on conventional classical methods like homogenous equally spaced methods in museums. We propose a methodology that is aimed at dynamic collaboration between automated analysis and cultural heritage expert knowledge to obtain optimized light positions. Our approach is cost-effective and adaptive to both linear and non-linear reflectance profile scenarios. The practical contribution of research in this field has a considerable impact on the cultural heritage context and beyond.

count=4
* Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/html/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/papers/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.pdf)]
    * Title: Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Csaba Beleznai, Peter Gemeiner, Christian Zinner
    * Abstract: Reliable and timely detection of abandoned items in public places still represents an unsolved problem for automated visual surveillance. Typical surveilled scenarios are associated with high visual ambiguity such as shadows, occlusions, illumination changes and substantial clutter consisting of a mixture of dynamic and stationary objects. Motivated by these challenges we propose a reliable left item detection approach based on the combination of intensity and depth data from a passive stereo setup. The employed in-house developed stereo system consists of low-cost sensors and it is capable to perform detection in environments of up to 10m x 10m in size. The proposed algorithm is tested on a set of indoor sequences and compared to manually annotated ground truth data. Obtained results show that many failure modes of intensity-based approaches are absent and even small-sized objects such as a handbag can be reliably detected when left behind in a scene. The presented results display a very promising approach, which can robustly detect left luggage in dynamic environments at a close to real-time computational speed.

count=4
* Multi-View 3D Object Reconstruction and Uncertainty Modelling With Neural Shape Prior
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Liao_Multi-View_3D_Object_Reconstruction_and_Uncertainty_Modelling_With_Neural_Shape_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Liao_Multi-View_3D_Object_Reconstruction_and_Uncertainty_Modelling_With_Neural_Shape_WACV_2024_paper.pdf)]
    * Title: Multi-View 3D Object Reconstruction and Uncertainty Modelling With Neural Shape Prior
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Ziwei Liao, Steven L. Waslander
    * Abstract: 3D object reconstruction is important for semantic scene understanding. It is challenging to reconstruct detailed 3D shapes from monocular images directly due to a lack of depth information, occlusion and noise. Most current methods generate deterministic object models without any awareness of the uncertainty of the reconstruction. We tackle this problem by leveraging a neural object representation which learns an object shape distribution from large dataset of 3d object models and maps it into a latent space. We propose a method to model uncertainty as part of the representation and define an uncertainty-aware encoder which generates latent codes with uncertainty directly from individual input images. Further, we propose a method to propagate the uncertainty in the latent code to SDF values and generate a 3d object mesh with local uncertainty for each mesh component. Finally, we propose an incremental fusion method under a Bayesian framework to fuse the latent codes from multi-view observations. We evaluate the system in both synthetic and real datasets to demonstrate the effectiveness of uncertainty-based fusion to improve 3D object reconstruction accuracy.

count=4
* FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/353ca88f722cdd0c481b999428ae113a-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/353ca88f722cdd0c481b999428ae113a-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Anatol Garioud, Nicolas Gonthier, Loic Landrieu, Apolline De Wit, Marion Valette, Marc Poupée, Sebastien Giordano, boris Wattrelos
    * Abstract: We introduce the French Land cover from Aerospace ImageRy (FLAIR), an extensive dataset from the French National Institute of Geographical and Forest Information (IGN) that provides a unique and rich resource for large-scale geospatial analysis. FLAIR contains high-resolution aerial imagery with a ground sample distance of 20 cm and over 20 billion individually labeled pixels for precise land-cover classification. The dataset also integrates temporal and spectral data from optical satellite time series. FLAIR thus combines data with varying spatial, spectral, and temporal resolutions across over 817 km² of acquisitions representing the full landscape diversity of France. This diversity makes FLAIR a valuable resource for the development and evaluation of novel methods for large-scale land-cover semantic segmentation and raises significant challenges in terms of computer vision, data fusion, and geospatial analysis. We also provide powerful uni- and multi-sensor baseline models that can be employed to assess algorithm's performance and for downstream applications.

count=3
* Moving Object Segmentation: All You Need Is SAM (and Flow)
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Xie_Moving_Object_Segmentation_All_You_Need_Is_SAM_and_Flow_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Xie_Moving_Object_Segmentation_All_You_Need_Is_SAM_and_Flow_ACCV_2024_paper.pdf)]
    * Title: Moving Object Segmentation: All You Need Is SAM (and Flow)
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Junyu Xie, Charig Yang, Weidi Xie, Andrew Zisserman
    * Abstract: The objective of this paper is motion segmentation -- discovering and segmenting the moving objects in a video. This is a much studied area with numerous careful, and sometimes complex, approaches and training schemes including: self-supervised learning, learning from synthetic datasets, object-centric representations, amodal representations, and many more. Our interest in this paper is to determine if the Segment Anything model (SAM) can contribute to this task. We investigate two models for combining SAM with optical flow that harness the segmentation power of SAM with the ability of flow to discover and group moving objects. In the first model, we adapt SAM to take optical flow, rather than RGB, as an input. In the second, SAM takes RGB as an input, and flow is used as a segmentation prompt. These surprisingly simple methods, without any further modifications, outperform all previous approaches by a considerable margin in both single and multi-object benchmarks. We also extend these frame-level segmentations to sequence-level segmentations that maintain object identity. Again, this simple model achieves outstanding performance across multiple moving object segmentation benchmarks.

count=3
* Detecting Changes in 3D Structure of a Scene from Multi-view Images Captured by a Vehicle-Mounted Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Sakurada_Detecting_Changes_in_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Sakurada_Detecting_Changes_in_2013_CVPR_paper.pdf)]
    * Title: Detecting Changes in 3D Structure of a Scene from Multi-view Images Captured by a Vehicle-Mounted Camera
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Ken Sakurada, Takayuki Okatani, Koichiro Deguchi
    * Abstract: This paper proposes a method for detecting temporal changes of the three-dimensional structure of an outdoor scene from its multi-view images captured at two separate times. For the images, we consider those captured by a camera mounted on a vehicle running in a city street. The method estimates scene structures probabilistically, not deterministically, and based on their estimates, it evaluates the probability of structural changes in the scene, where the inputs are the similarity of the local image patches among the multi-view images. The aim of the probabilistic treatment is to maximize the accuracy of change detection, behind which there is our conjecture that although it is difficult to estimate the scene structures deterministically, it should be easier to detect their changes. The proposed method is compared with the methods that use multi-view stereo (MVS) to reconstruct the scene structures of the two time points and then differentiate them to detect changes. The experimental results show that the proposed method outperforms such MVS-based methods.

count=3
* Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Teutsch_Robust_Detection_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Teutsch_Robust_Detection_of_CVPR_2016_paper.pdf)]
    * Title: Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Michael Teutsch, Michael Grinberg
    * Abstract: Multiple object tracking in Wide Area Motion Imagery (WAMI) data is usually based on initial detections coming from background subtraction or frame differencing. However, these methods are prone to produce split and merged detections. Appearance based vehicle detection can be an alternative but is not well-suited for WAMI data since classifier models are of weak discriminative power for vehicles in top view at low resolution. We introduce a moving vehicle detection algorithm that combines 2-frame differencing with a vehicle appearance model to improve object detection. Our main contributions are (1) integration of robust vehicle detection with split/merge handling and (2) estimation of assignment likelihoods between object hypotheses in consecutive frames using an appearance based similarity measure. Without using any prior knowledge, we achieve state-of-the-art detection rates and produce tracklets that considerably simplify the data association problem for multiple object tracking.

count=3
* Turning an Urban Scene Video Into a Cinemagraph
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Yan_Turning_an_Urban_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Turning_an_Urban_CVPR_2017_paper.pdf)]
    * Title: Turning an Urban Scene Video Into a Cinemagraph
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Hang Yan, Yebin Liu, Yasutaka Furukawa
    * Abstract: This paper proposes an algorithm that turns a regular video capturing urban scenes into a high-quality endless animation, known as a Cinemagraph. The creation of a Cinemagraph usually requires a static camera in a carefully configured scene. The task becomes challenging for a regular video with a moving camera and objects. Our approach first warps an input video into the viewpoint of a reference camera. Based on the warped video, we propose effective temporal analysis algorithms to detect regions with static geometry and dynamic appearance, where geometric modeling is reliable and visually attractive animations can be created. Lastly, the algorithm applies a sequence of video processing techniques to produce a Cinemagraph movie. We have tested the proposed approach on numerous challenging real scenes. To our knowledge, this work is the first to automatically generate Cinemagraph animations from regular movies in the wild.

count=3
* Object State Recognition for Automatic AR-Based Maintenance Guidance
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w14/html/Dvorak_Object_State_Recognition_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w14/papers/Dvorak_Object_State_Recognition_CVPR_2017_paper.pdf)]
    * Title: Object State Recognition for Automatic AR-Based Maintenance Guidance
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Pavel Dvorak, Radovan Josth, Elisabetta Delponte
    * Abstract: This paper describes a component of an Augmented Reality (AR) based system focused on supporting workers in manufacturing and maintenance industry. Particularly, it describes a component responsible for verification of performed steps. Correct handling is crucial in both manufacturing and maintenance industries and deviations may cause problems in later stages of the production and assembly. The primary aim of such support systems is making the training of new employees faster and more efficient and reducing the error rate. We present a method for automatically recognizing an object's state with the objective of verifying a set of tasks performed by a user. The novelty of our approach is that the system can automatically recognize the state of the object and provide immediate feedback to the operator using an AR visualization enabling fully automatic step-by-step instructions.

count=3
* Omnimatte: Associating Objects and Their Effects in Video
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lu_Omnimatte_Associating_Objects_and_Their_Effects_in_Video_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lu_Omnimatte_Associating_Objects_and_Their_Effects_in_Video_CVPR_2021_paper.pdf)]
    * Title: Omnimatte: Associating Objects and Their Effects in Video
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Erika Lu, Forrester Cole, Tali Dekel, Andrew Zisserman, William T. Freeman, Michael Rubinstein
    * Abstract: Computer vision has become increasingly better at segmenting objects in images and videos; however, scene effects related to the objects -- shadows, reflections, generated smoke, etc. -- are typically overlooked. Identifying such scene effects and associating them with the objects producing them is important for improving our fundamental understanding of visual scenes, and applications such as removing, duplicating, or enhancing objects in video. We take a step towards solving this novel problem of automatically associating objects with their effects in video. Given an ordinary video and a rough segmentation mask over time of one or more subjects of interest, we estimate an omnimatte for each subject -- an alpha matte and color image that includes the subject along with all its related time-varying scene elements. Our model is trained only on the input video in a self-supervised manner, without any manual labels, and is generic -- it produces omnimattes automatically for arbitrary objects and a variety of effects. We show results on real-world videos containing interactions between different types of subjects (cars, animals, people) and complex effects, ranging from semi-transparent smoke and reflections to fully opaque objects attached to the subject.

count=3
* (ASNA) An Attention-Based Siamese-Difference Neural Network With Surrogate Ranking Loss Function for Perceptual Image Quality Assessment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Ayyoubzadeh_ASNA_An_Attention-Based_Siamese-Difference_Neural_Network_With_Surrogate_Ranking_Loss_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Ayyoubzadeh_ASNA_An_Attention-Based_Siamese-Difference_Neural_Network_With_Surrogate_Ranking_Loss_CVPRW_2021_paper.pdf)]
    * Title: (ASNA) An Attention-Based Siamese-Difference Neural Network With Surrogate Ranking Loss Function for Perceptual Image Quality Assessment
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Seyed Mehdi Ayyoubzadeh, Ali Royat
    * Abstract: Recently, deep convolutional neural networks (DCNN) that leverage the adversarial training framework for image restoration and enhancement have significantly improved the processed images' sharpness. Surprisingly, although these DCNNs produced crispier images than other methods visually, they may get a lower quality score when popular measures are employed for evaluating them. Therefore it is necessary to develop a quantitative metric to reflect their performances, which is well-aligned with the perceived quality of an image. Famous quantitative metrics such as Peak signal-to-noise ratio (PSNR), The structural similarity index measure (SSIM), and Perceptual Index (PI) are not well-correlated with the mean opinion score (MOS) for an image, especially for the neural networks trained with adversarial loss functions. This paper has proposed a convolutional neural network using an extension architecture of the traditional Siamese network so-called Siamese-Difference neural network. We have equipped this architecture with the spatial and channel-wise attention mechanism to increase our method's performance. Finally, we employed an auxiliary loss function to train our model. The suggested additional cost function surrogates ranking loss to increase Spearman's rank correlation coefficient while it is differentiable concerning the neural network parameters. Our method achieved superior performance in NTIRE 2021 Perceptual Image Quality Assessment Challenge. The implementations of our proposed method are publicly available.

count=3
* Semantic-Aware Domain Generalized Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.pdf)]
    * Title: Semantic-Aware Domain Generalized Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Duo Peng, Yinjie Lei, Munawar Hayat, Yulan Guo, Wen Li
    * Abstract: Deep models trained on source domain lack generalization when evaluated on unseen target domains with different data distributions. The problem becomes even more pronounced when we have no access to target domain samples for adaptation. In this paper, we address domain generalized semantic segmentation, where a segmentation model is trained to be domain-invariant without using any target domain data. Existing approaches to tackle this problem standardize data into a unified distribution. We argue that while such a standardization promotes global normalization, the resulting features are not discriminative enough to get clear segmentation boundaries. To enhance separation between categories while simultaneously promoting domain invariance, we propose a framework including two novel modules: Semantic-Aware Normalization (SAN) and Semantic-Aware Whitening (SAW). Specifically, SAN focuses on category-level center alignment between features from different image styles, while SAW enforces distributed alignment for the already center-aligned features. With the help of SAN and SAW, we encourage both intraclass compactness and inter-class separability. We validate our approach through extensive experiments on widely-used datasets (i.e. GTAV, SYNTHIA, Cityscapes, Mapillary and BDDS). Our approach shows significant improvements over existing state-of-the-art on various backbone networks. Code is available at https://github.com/leolyj/SAN-SAW

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
* ZBS: Zero-Shot Background Subtraction via Instance-Level Background Modeling and Foreground Selection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/An_ZBS_Zero-Shot_Background_Subtraction_via_Instance-Level_Background_Modeling_and_Foreground_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/An_ZBS_Zero-Shot_Background_Subtraction_via_Instance-Level_Background_Modeling_and_Foreground_CVPR_2023_paper.pdf)]
    * Title: ZBS: Zero-Shot Background Subtraction via Instance-Level Background Modeling and Foreground Selection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yongqi An, Xu Zhao, Tao Yu, Haiyun Guo, Chaoyang Zhao, Ming Tang, Jinqiao Wang
    * Abstract: Background subtraction (BGS) aims to extract all moving objects in the video frames to obtain binary foreground segmentation masks. Deep learning has been widely used in this field. Compared with supervised-based BGS methods, unsupervised methods have better generalization. However, previous unsupervised deep learning BGS algorithms perform poorly in sophisticated scenarios such as shadows or night lights, and they cannot detect objects outside the pre-defined categories. In this work, we propose an unsupervised BGS algorithm based on zero-shot object detection called Zero-shot Background Subtraction ZBS. The proposed method fully utilizes the advantages of zero-shot object detection to build the open-vocabulary instance-level background model. Based on it, the foreground can be effectively extracted by comparing the detection results of new frames with the background model. ZBS performs well for sophisticated scenarios, and it has rich and extensible categories. Furthermore, our method can easily generalize to other tasks, such as abandoned object detection in unseen environments. We experimentally show that ZBS surpasses state-of-the-art unsupervised BGS methods by 4.70% F-Measure on the CDnet 2014 dataset. The code is released at https://github.com/CASIA-IVA-Lab/ZBS.

count=3
* SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.pdf)]
    * Title: SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xin Guo, Jiangwei Lao, Bo Dang, Yingying Zhang, Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang, Kang Wu, Dingxiang Hu, Huimei He, Jian Wang, Jingdong Chen, Ming Yang, Yongjun Zhang, Yansheng Li
    * Abstract: Prior studies on Remote Sensing Foundation Model (RSFM) reveal immense potential towards a generic model for Earth Observation. Nevertheless these works primarily focus on a single modality without temporal and geo-context modeling hampering their capabilities for diverse tasks. In this study we present SkySense a generic billion-scale model pre-trained on a curated multi-modal Remote Sensing Imagery (RSI) dataset with 21.5 million temporal sequences. SkySense incorporates a factorized multi-modal spatiotemporal encoder taking temporal sequences of optical and Synthetic Aperture Radar (SAR) data as input. This encoder is pre-trained by our proposed Multi-Granularity Contrastive Learning to learn representations across different modal and spatial granularities. To further enhance the RSI representations by the geo-context clue we introduce Geo-Context Prototype Learning to learn region-aware prototypes upon RSI's multi-modal spatiotemporal features. To our best knowledge SkySense is the largest Multi-Modal RSFM to date whose modules can be flexibly combined or used individually to accommodate various tasks. It demonstrates remarkable generalization capabilities on a thorough evaluation encompassing 16 datasets over 7 tasks from single- to multi-modal static to temporal and classification to localization. SkySense surpasses 18 recent RSFMs in all test scenarios. Specifically it outperforms the latest models such as GFM SatLas and Scale-MAE by a large margin i.e. 2.76% 3.67% and 3.61% on average respectively. We will release the pre-trained weights to facilitate future research and Earth Observation applications.

count=3
* Should We Encode Rain Streaks in Video as Deterministic or Stochastic?
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Wei_Should_We_Encode_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wei_Should_We_Encode_ICCV_2017_paper.pdf)]
    * Title: Should We Encode Rain Streaks in Video as Deterministic or Stochastic?
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Wei Wei, Lixuan Yi, Qi Xie, Qian Zhao, Deyu Meng, Zongben Xu
    * Abstract: Videos taken in the wild sometimes contain unexpected rain streaks, which brings difficulty in subsequent video processing tasks. Rain streak removal in a video (RSRV) is thus an important issue and has been attracting much attention in computer vision. Different from previous RSRV methods formulating rain streaks as a deterministic message, this work first encodes the rains in a stochastic manner, i.e., a patch-based mixture of Gaussians. Such modification makes the proposed model capable of finely adapting a wider range of rain variations instead of certain types of rain configurations as traditional. By integrating with the spatiotemporal smoothness configuration of moving objects and low-rank structure of background scene, we propose a concise model for RSRV, containing one likelihood term imposed on the rain streak layer and two prior terms on the moving object and background scene layers of the video. Experiments implemented on videos with synthetic and real rains verify the superiority of the proposed method, as com- pared with the state-of-the-art methods, both visually and quantitatively in various performance metrics.

count=3
* The Visual Object Tracking VOT2017 Challenge Results
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w28/html/Kristan_The_Visual_Object_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Kristan_The_Visual_Object_ICCV_2017_paper.pdf)]
    * Title: The Visual Object Tracking VOT2017 Challenge Results
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pflugfelder, Luka Cehovin Zajc, Tomas Vojir, Gustav Hager, Alan Lukezic, Abdelrahman Eldesokey, Gustavo Fernandez
    * Abstract: The Visual Object Tracking challenge VOT2017 is the fifth annual tracker benchmarking activity organized by the VOT initiative. Results of 51 trackers are presented; many are state-of-the-art published at major computer vision conferences or journals in recent years. The evaluation included the standard VOT and other popular methodologies and a new "real-time" experiment simulating a situation where a tracker processes images as if provided by a continuously running sensor. Performance of the tested trackers typically by far exceeds standard baselines. The source code for most of the trackers is publicly available from the VOT page. The VOT2017 goes beyond its predecessors by (i) improving the VOT public dataset and introducing a separate VOT2017 sequestered dataset, (ii) introducing a real-time tracking experiment and (iii) releasing a redesigned toolkit that supports complex experiments. The dataset, the evaluation kit and the results are publicly available at the challenge w ....

count=3
* RIO: 3D Object Instance Re-Localization in Changing Indoor Environments
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.pdf)]
    * Title: RIO: 3D Object Instance Re-Localization in Changing Indoor Environments
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Johanna Wald,  Armen Avetisyan,  Nassir Navab,  Federico Tombari,  Matthias Niessner
    * Abstract: In this work, we introduce the task of 3D object instance re-localization (RIO): given one or multiple objects in an RGB-D scan, we want to estimate their corresponding 6DoF poses in another 3D scan of the same environment taken at a later point in time. We consider RIO a particularly important task in 3D vision since it enables a wide range of practical applications, including AI-assistants or robots that are asked to find a specific object in a 3D scene. To address this problem, we first introduce 3RScan, a novel dataset and benchmark, which features 1482 RGB-D scans of 478 environments across multiple time steps. Each scene includes several objects whose positions change over time, together with ground truth annotations of object instances and their respective 6DoF mappings among re-scans. Automatically finding 6DoF object poses leads to a particular challenging feature matching task due to varying partial observations and changes in the surrounding context. To this end, we introduce a new data-driven approach that efficiently finds matching features using a fully-convolutional 3D correspondence network operating on multiple spatial scales. Combined with a 6DoF pose optimization, our method outperforms state-of-the-art baselines on our newly-established benchmark, achieving an accuracy of 30.58%.

count=3
* Viewpoint-Agnostic Change Captioning With Cycle Consistency
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Viewpoint-Agnostic_Change_Captioning_With_Cycle_Consistency_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Viewpoint-Agnostic_Change_Captioning_With_Cycle_Consistency_ICCV_2021_paper.pdf)]
    * Title: Viewpoint-Agnostic Change Captioning With Cycle Consistency
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Hoeseong Kim, Jongseok Kim, Hyungseok Lee, Hyunsung Park, Gunhee Kim
    * Abstract: Change captioning is the task of identifying the change and describing it with a concise caption. Despite recent advancements, filtering out insignificant changes still remains as a challenge. Namely, images from different camera perspectives can cause issues; a mere change in viewpoint should be disregarded while still capturing the actual changes. In order to tackle this problem, we present a new Viewpoint-Agnostic change captioning network with Cycle Consistency (VACC) that requires only one image each for the before and after scene, without depending on any other information. We achieve this by devising a new difference encoder module which can encode viewpoint information and model the difference more effectively. In addition, we propose a cycle consistency module that can potentially improve the performance of any change captioning networks in general by matching the composite feature of the generated caption and before image with the after image feature. We evaluate the performance of our proposed model across three datasets for change captioning, including a novel dataset we introduce here that contains images with changes under extreme viewpoint shifts. Through our experiments, we show the excellence of our method with respect to the CIDEr, BLEU-4, METEOR and SPICE scores. Moreover, we demonstrate that attaching our proposed cycle consistency module yields a performance boost for existing change captioning networks, even with varying image encoding mechanisms.

count=3
* PanFlowNet: A Flow-Based Deep Network for Pan-Sharpening
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_PanFlowNet_A_Flow-Based_Deep_Network_for_Pan-Sharpening_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_PanFlowNet_A_Flow-Based_Deep_Network_for_Pan-Sharpening_ICCV_2023_paper.pdf)]
    * Title: PanFlowNet: A Flow-Based Deep Network for Pan-Sharpening
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Gang Yang, Xiangyong Cao, Wenzhe Xiao, Man Zhou, Aiping Liu, Xun Chen, Deyu Meng
    * Abstract: Pan-sharpening aims to generate a high-resolution multispectral (HRMS) image by integrating the spectral information of a low-resolution multispectral (LRMS) image with the texture details of a high-resolution panchromatic (PAN) image. It essentially inherits the ill-posed nature of the super-resolution (SR) task that diverse HRMS images can degrade into an LRMS image. However, existing deep learning-based methods recover only one HRMS image from the LRMS image and PAN image using a deterministic mapping, thus ignoring the diversity of the HRMS image. In this paper, to alleviate this ill-posed issue, we propose a flow-based pan-sharpening network (PanFlowNet) to directly learn the conditional distribution of HRMS image given LRMS image and PAN image instead of learning a deterministic mapping. Specifically, we first transform this unknown conditional distribution into a given Gaussian distribution by an invertible network, and the conditional distribution can thus be explicitly defined. Then, we design an invertible Conditional Affine Coupling Block (CACB) and further build the architecture of PanFlowNet by stacking a series of CACBs. Finally, the PanFlowNet is trained by maximizing the log-likelihood of the conditional distribution given a training set and can then be used to predict diverse HRMS images. The experimental results verify that the proposed PanFlowNet can generate various HRMS images given an LRMS image and a PAN image. Additionally, the experimental results on different kinds of satellite datasets also demonstrate the superiority of our PanFlowNet compared with other state-of-the-art methods both visually and quantitatively. Code is available at Github.

count=3
* RethNet: Object-by-Object Learning for Detecting Facial Skin Problems
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/VRMI/Bekmirzaev_RethNet_Object-by-Object_Learning_for_Detecting_Facial_Skin_Problems_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Bekmirzaev_RethNet_Object-by-Object_Learning_for_Detecting_Facial_Skin_Problems_ICCVW_2019_paper.pdf)]
    * Title: RethNet: Object-by-Object Learning for Detecting Facial Skin Problems
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Shohrukh Bekmirzaev, Seoyoung Oh, Sangwook Yo
    * Abstract: Semantic segmentation is a hot topic in computer vision where the most challenging tasks of object detection and recognition have been handling by the success of semantic segmentation approaches. We propose a concept of objectby-object learning technique to detect 11 types of facial skin lesions using semantic segmentation methods. Detecting individual skin lesion in a dense group is a challenging task, because of ambiguities in the appearance of the visual data. We observe that there exist co-occurrent visual relations between object classes (e.g., wrinkle and age spot, or papule and whitehead, etc.). In fact, rich contextual information significantly helps to handle the issue. Therefore, we propose REthinker blocks that are composed of the locally constructed convLSTM/Conv3D layers and SE module as a one-shot attention mechanism whose responsibility is to increase network's sensitivity in the local and global contextual representation that supports to capture ambiguously appeared objects and co-occurrence interactions between object classes. Experiments show that our proposed model reached MIoU of 79.46% on the test of a prepared dataset, representing a 15.34% improvement over Deeplab v3+ (MIoU of 64.12%).

count=3
* A Convex Relaxation Approach to Space Time Multi-view 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W09/html/Oswald_A_Convex_Relaxation_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W09/papers/Oswald_A_Convex_Relaxation_2013_ICCV_paper.pdf)]
    * Title: A Convex Relaxation Approach to Space Time Multi-view 3D Reconstruction
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Martin R. Oswald, Daniel Cremers
    * Abstract: We propose a convex relaxation approach to space-time 3D reconstruction from multiple videos. Generalizing the works [16], [8] to the 4D setting, we cast the problem of reconstruction over time as a binary labeling problem in a 4D space. We propose a variational formulation which combines a photoconsistency based data term with a spatiotemporal total variation regularization. In particular, we propose a novel data term that is both faster to compute and better suited for wide-baseline camera setups when photoconsistency measures are unreliable or missing. The proposed functional can be globally minimized using convex relaxation techniques. Numerous experiments on a variety of publically available data sets demonstrate that we can compute detailed and temporally consistent reconstructions. In particular, the temporal regularization allows to reduce jittering of voxels over time.

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
* A Novel Inspection System For Variable Data Printing Using Deep Learning
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Haik_A_Novel_Inspection_System_For_Variable_Data_Printing_Using_Deep_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Haik_A_Novel_Inspection_System_For_Variable_Data_Printing_Using_Deep_WACV_2020_paper.pdf)]
    * Title: A Novel Inspection System For Variable Data Printing Using Deep Learning
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Oren Haik,  Oded Perry,  Eli Chen,  Peter Klammer
    * Abstract: We present a novel approach for inspecting variable data prints (VDP) with an ultra-low false alarm rate (0.005%) and potential applicability to other real-world problems. The system is based on a comparison between two images: a reference image and an image captured by low-cost scanners. The comparison task is challenging as low-cost imaging systems create artifacts that may erroneously be classified as true (genuine) defects. To address this challenge we introduce two new fusion methods, for change detection applications, which are both fast and efficient. The first is an early fusion method that combines the two input images into a single pseudo-color image. The second, called Change-Detection Single Shot Detector (CD-SSD) leverages the SSD by fusing features in the middle of the network. We demonstrate the effectiveness of the proposed deep learning-based approach with a large dataset from real-world printing scenarios. Finally, we evaluate our models on a different domain of aerial imagery change detection (AICD). Our best method clearly outperforms the state-of-the-art baseline on this dataset.

count=3
* Self-Pair: Synthesizing Changes From Single Source for Object Change Detection in Remote Sensing Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Seo_Self-Pair_Synthesizing_Changes_From_Single_Source_for_Object_Change_Detection_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Seo_Self-Pair_Synthesizing_Changes_From_Single_Source_for_Object_Change_Detection_WACV_2023_paper.pdf)]
    * Title: Self-Pair: Synthesizing Changes From Single Source for Object Change Detection in Remote Sensing Imagery
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Minseok Seo, Hakjin Lee, Yongjin Jeon, Junghoon Seo
    * Abstract: For change detection in remote sensing, constructing a training dataset for deep learning models is quite difficult due to the requirements of bi-temporal supervision. To overcome this issue, single-temporal supervision which treats change labels as the difference of two semantic masks has been proposed. This novel method trains a change detector using two spatially unrelated images with corresponding semantic labels. However, training with unpaired dataset shows not enough performance compared with other methods based on bi-temporal supervision. We suspect this phenomenon caused by ignorance of meaningful information in the actual bi-temporal pairs.In this paper, we emphasize that the change originates from the source image and show that manipulating the source image as an after-image is crucial to the performance of change detection. Our method achieves state-of-the-art performance in a large gap than existing methods.

count=3
* ENIGMA-51: Towards a Fine-Grained Understanding of Human Behavior in Industrial Scenarios
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Ragusa_ENIGMA-51_Towards_a_Fine-Grained_Understanding_of_Human_Behavior_in_Industrial_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Ragusa_ENIGMA-51_Towards_a_Fine-Grained_Understanding_of_Human_Behavior_in_Industrial_WACV_2024_paper.pdf)]
    * Title: ENIGMA-51: Towards a Fine-Grained Understanding of Human Behavior in Industrial Scenarios
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Francesco Ragusa, Rosario Leonardi, Michele Mazzamuto, Claudia Bonanno, Rosario Scavo, Antonino Furnari, Giovanni Maria Farinella
    * Abstract: ENIGMA-51 is a new egocentric dataset acquired in an industrial scenario by 19 subjects who followed instructions to complete the repair of electrical boards using industrial tools (e.g., electric screwdriver) and equipments (e.g., oscilloscope). The 51 egocentric video sequences are densely annotated with a rich set of labels that enable the systematic study of human behavior in the industrial domain. We provide benchmarks on four tasks related to human behavior: 1) untrimmed temporal detection of human-object interactions, 2) egocentric human-object interaction detection, 3) short-term object interaction anticipation and 4) natural language understanding of intents and entities. Baseline results show that the ENIGMA-51 dataset poses a challenging benchmark to study human behavior in industrial scenarios. We publicly release the dataset at https://iplab.dmi.unict.it/ENIGMA-51.

count=3
* D^2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/d2cc447db9e56c13b993c11b45956281-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/d2cc447db9e56c13b993c11b45956281-Paper-Conference.pdf)]
    * Title: D^2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Tianhao Wu, Fangcheng Zhong, Andrea Tagliasacchi, Forrester Cole, Cengiz Oztireli
    * Abstract: Given a monocular video, segmenting and decoupling dynamic objects while recovering the static environment is a widely studied problem in machine intelligence. Existing solutions usually approach this problem in the image domain, limiting their performance and understanding of the environment. We introduce Decoupled Dynamic Neural Radiance Field (D^2NeRF), a self-supervised approach that takes a monocular video and learns a 3D scene representation which decouples moving objects, including their shadows, from the static background. Our method represents the moving objects and the static background by two separate neural radiance fields with only one allowing for temporal changes. A naive implementation of this approach leads to the dynamic component taking over the static one as the representation of the former is inherently more general and prone to overfitting. To this end, we propose a novel loss to promote correct separation of phenomena. We further propose a shadow field network to detect and decouple dynamically moving shadows. We introduce a new dataset containing various dynamic objects and shadows and demonstrate that our method can achieve better performance than state-of-the-art approaches in decoupling dynamic and static 3D objects, occlusion and shadow removal, and image segmentation for moving objects. Project page: https://d2nerf.github.io/

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
* Large-Scale Damage Detection Using Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gueguen_Large-Scale_Damage_Detection_2015_CVPR_paper.pdf)]
    * Title: Large-Scale Damage Detection Using Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Lionel Gueguen, Raffay Hamid
    * Abstract: Satellite imagery is a valuable source of information for assessing damages in distressed areas undergoing a calamity, such as an earthquake or an armed conflict. However, the sheer amount of data required to be inspected for this assessment makes it impractical to do it manually. To address this problem, we present a semi-supervised learning framework for large-scale damage detection in satellite imagery. We present a comparative evaluation of our framework using over 88 million images collected from 4,665 square kilometers from 12 different locations around the world. To enable accurate and efficient damage detection, we introduce a novel use of hierarchical shape features in the bags-of-visual words setting. We analyze how practical factors such as sun, sensor-resolution, and satellite-angle differences impact the effectiveness of our proposed representation, and compare it to five alternative features in multiple learning settings. Finally, we demonstrate through a user-study that our semi-supervised framework results in a ten-fold reduction in human annotation time at a minimal loss in detection accuracy compared to an exhaustive manual inspection.

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
* Embedded Vision System for Atmospheric Turbulence Mitigation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/html/Deshmukh_Embedded_Vision_System_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w14/papers/Deshmukh_Embedded_Vision_System_CVPR_2016_paper.pdf)]
    * Title: Embedded Vision System for Atmospheric Turbulence Mitigation
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Ajinkya Deshmukh, Gaurav Bhosale, Swarup Shanti, Karthik Reddy, Hemanthkumar P., Chandrasekhar A., Kirankumar P., Vijaysagar K.
    * Abstract: Outdoor surveillance systems that involve farfield operations often encounter atmospheric turbulence perturbations due to a series of randomized reflections and refraction effecting incoming light rays. The resulting distortions make it hard to discriminate between true moving objects and turbulence induced motion. Current algorithms are not effective in detecting true moving objects in the scene and also rely on computationally complex warping methods. In this paper, we describe a real time embedded solution connected with traditional cameras to both rectify turbulence distortions and reliably detect and track true moving targets. Our comparisons with other methods shows better turbulence rectification with less false and miss detections. FPGA-DSP based embedded realization of our algorithm achieves nearly 15x speed-up along with lesser memory requirement over a quad core PC implementation. The proposed system is suitable for persistence surveillance systems and optical sight devices.

count=2
* The TUM-DLR Multimodal Earth Observation Evaluation Benchmark
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w19/html/Koch_The_TUM-DLR_Multimodal_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w19/papers/Koch_The_TUM-DLR_Multimodal_CVPR_2016_paper.pdf)]
    * Title: The TUM-DLR Multimodal Earth Observation Evaluation Benchmark
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Tobias Koch, Pablo d'Angelo, Franz Kurz, Friedrich Fraundorfer, Peter Reinartz, Marco Korner
    * Abstract: We present a new dataset for development, benchmarking, and evaluation of remote sensing and earth observation approaches with special focus on converging perspectives. In order to provide data with different modalities, we observed the same scene using satellites, airplanes, unmanned aerial vehicles (UAV), and smartphones. The dataset is further complemented by ground-truth information and baseline results for different application scenarios. The provided data can be freely used by anybody interested in remote sensing and earth observation and will be continuously augmented and updated.

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
* Minimum Delay Moving Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Lao_Minimum_Delay_Moving_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lao_Minimum_Delay_Moving_CVPR_2017_paper.pdf)]
    * Title: Minimum Delay Moving Object Detection
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Dong Lao, Ganesh Sundaramoorthi
    * Abstract: We present a general framework and method for detection of an object in a video based on apparent motion. The object moves relative to background motion at some unknown time in the video, and the goal is to detect and segment the object as soon it moves in an online manner. Due to unreliability of motion between frames, more than two frames are needed to reliably detect the object. Our method is designed to detect the object(s) with minimum delay, i.e., frames after the object moves, constraining the false alarms. Experiments on a new extensive dataset for moving object detection show that our method achieves less delay for all false alarm constraints than existing state-of-the-art.

count=2
* Going From Image to Video Saliency: Augmenting Image Salience With Dynamic Attentional Push
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Gorji_Going_From_Image_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gorji_Going_From_Image_CVPR_2018_paper.pdf)]
    * Title: Going From Image to Video Saliency: Augmenting Image Salience With Dynamic Attentional Push
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Siavash Gorji, James J. Clark
    * Abstract: We present a novel method to incorporate the recent advent in static saliency models to predict the saliency in videos. Our model augments the static saliency models with the Attentional Push effect of the photographer and the scene actors in a shared attention setting. We demonstrate that not only it is imperative to use static Attentional Push cues, noticeable performance improvement is achievable by learning the time-varying nature of Attentional Push. We propose a multi-stream Convolutional Long Short-Term Memory network (ConvLSTM) structure which augments state-of-the-art in static saliency models with dynamic Attentional Push. Our network contains four pathways, a saliency pathway and three Attentional Push pathways. The multi-pathway structure is followed by an augmenting convnet that learns to combine the complementary and time-varying outputs of the ConvLSTMs by minimizing the relative entropy between the augmented saliency and viewers fixation patterns on videos. We evaluate our model by comparing the performance of several augmented static saliency models with state-of-the-art in spatiotemporal saliency on three largest dynamic eye tracking datasets, HOLLYWOOD2, UCF-Sport and DIEM. Experimental results illustrates that solid performance gain is achievable using the proposed methodology.

count=2
* Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Video_Rain_Streak_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Video_Rain_Streak_CVPR_2018_paper.pdf)]
    * Title: Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Minghan Li, Qi Xie, Qian Zhao, Wei Wei, Shuhang Gu, Jing Tao, Deyu Meng
    * Abstract: Videos captured by outdoor surveillance equipments sometimes contain unexpected rain streaks, which brings difficulty in subsequent video processing tasks. Rain streak removal from a video is thus an important topic in recent computer vision research. In this paper, we raise two intrinsic characteristics specifically possessed by rain streaks. Firstly, the rain streaks in a video contain repetitive local patterns sparsely scattered over different positions of the video. Secondly, the rain streaks are with multiscale configurations due to their occurrence on positions with different distances to the cameras. Based on such understanding, we specifically formulate both characteristics into a multiscale convolutional sparse coding (MS-CSC) model for the video rain streak removal task. Specifically, we use multiple convolutional filters convolved on the sparse feature maps to deliver the former characteristic, and further use multiscale filters to represent different scales of rain streaks. Such a new encoding manner makes the proposed method capable of properly extracting rain streaks from videos, thus getting fine video deraining effects. Experiments implemented on synthetic and real videos verify the superiority of the proposed method, as compared with the state-of-the-art ones along this research line, both visually and quantitatively.

count=2
* Object Tracking by Reconstruction With View-Specific Discriminative Correlation Filters
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Kart_Object_Tracking_by_Reconstruction_With_View-Specific_Discriminative_Correlation_Filters_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kart_Object_Tracking_by_Reconstruction_With_View-Specific_Discriminative_Correlation_Filters_CVPR_2019_paper.pdf)]
    * Title: Object Tracking by Reconstruction With View-Specific Discriminative Correlation Filters
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Ugur Kart,  Alan Lukezic,  Matej Kristan,  Joni-Kristian Kamarainen,  Jiri Matas
    * Abstract: Standard RGB-D trackers treat the target as a 2D structure, which makes modelling appearance changes related even to out-of-plane rotation challenging. This limitation is addressed by the proposed long-term RGB-D tracker called OTR - Object Tracking by Reconstruction. OTR performs online 3D target reconstruction to facilitate robust learning of a set of view-specific discriminative correlation filters (DCFs). The 3D reconstruction supports two performance- enhancing features: (i) generation of an accurate spatial support for constrained DCF learning from its 2D projection and (ii) point-cloud based estimation of 3D pose change for selection and storage of view-specific DCFs which robustly localize the target after out-of-view rotation or heavy occlusion. Extensive evaluation on the Princeton RGB-D tracking and STC Benchmarks shows OTR outperforms the state-of-the-art by a large margin.

count=2
* Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.pdf)]
    * Title: Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Vivien Sainte Fare Garnot,  Loic Landrieu,  Sebastien Giordano,  Nesrine Chehata
    * Abstract: Satellite image time series, bolstered by their growing availability, are at the forefront of an extensive effort towards automated Earth monitoring by international institutions. In particular, large-scale control of agricultural parcels is an issue of major political and economic importance. In this regard, hybrid convolutional-recurrent neural architectures have shown promising results for the automated classification of satellite image time series. We propose an alternative approach in which the convolutional layers are advantageously replaced with encoders operating on unordered sets of pixels to exploit the typically coarse resolution of publicly available satellite images. We also propose to extract temporal features using a bespoke neural architecture based on self-attention instead of recurrent networks. We demonstrate experimentally that our method not only outperforms previous state-of-the-art approaches in terms of precision, but also significantly decreases processing time and memory requirements. Lastly, we release a large open-access annotated dataset as a benchmark for future work on satellite image time series.

count=2
* Learning Compositional Representation for 4D Captures With Neural ODE
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.pdf)]
    * Title: Learning Compositional Representation for 4D Captures With Neural ODE
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Boyan Jiang, Yinda Zhang, Xingkui Wei, Xiangyang Xue, Yanwei Fu
    * Abstract: Learning based representation has become the key to the success of many computer vision systems. While many 3D representations have been proposed, it is still an unaddressed problem how to represent a dynamically changing 3D object. In this paper, we introduce a compositional representation for 4D captures, i.e. a deforming 3D object over a temporal span, that disentangles shape, initial state, and motion respectively. Each component is represented by a latent code via a trained encoder. To model the motion, a neural Ordinary Differential Equation (ODE) is trained to update the initial state conditioned on the learned motion code, and a decoder takes the shape code and the updated state code to reconstruct the 3D model at each time stamp. To this end, we propose an Identity Exchange Training (IET) strategy to encourage the network to learn effectively decoupling each component. Extensive experiments demonstrate that the proposed method outperforms existing state-of-the-art deep learning based methods on 4D reconstruction, and significantly improves on various tasks, including motion transfer and completion.

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
* Change-Aware Sampling and Contrastive Learning for Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf)]
    * Title: Change-Aware Sampling and Contrastive Learning for Satellite Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Utkarsh Mall, Bharath Hariharan, Kavita Bala
    * Abstract: Automatic remote sensing tools can help inform many large-scale challenges such as disaster management, climate change, etc. While a vast amount of spatio-temporal satellite image data is readily available, most of it remains unlabelled. Without labels, this data is not very useful for supervised learning algorithms. Self-supervised learning instead provides a way to learn effective representations for various downstream tasks without labels. In this work, we leverage characteristics unique to satellite images to learn better self-supervised features. Specifically, we use the temporal signal to contrast images with long-term and short-term differences, and we leverage the fact that satellite images do not change frequently. Using these characteristics, we formulate a new loss contrastive loss called Change-Aware Contrastive (CACo) Loss. Further, we also present a novel method of sampling different geographical regions. We show that leveraging these properties leads to better performance on diverse downstream tasks. For example, we see a 6.5% relative improvement for semantic segmentation and an 8.5% relative improvement for change detection over the best-performing baseline with our method.

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
* Density Invariant Contrast Maximization for Neuromorphic Earth Observations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/html/Arja_Density_Invariant_Contrast_Maximization_for_Neuromorphic_Earth_Observations_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Arja_Density_Invariant_Contrast_Maximization_for_Neuromorphic_Earth_Observations_CVPRW_2023_paper.pdf)]
    * Title: Density Invariant Contrast Maximization for Neuromorphic Earth Observations
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Sami Arja, Alexandre Marcireau, Richard L. Balthazor, Matthew G. McHarg, Saeed Afshar, Gregory Cohen
    * Abstract: Contrast maximization (CMax) techniques are widely used in event-based vision systems to estimate the motion parameters of the camera and generate high-contrast images. However, these techniques are noise-intolerance and suffer from the multiple extrema problem which arises when the scene contains more noisy events than structure, causing the contrast to be higher at multiple locations. This makes the task of estimating the camera motion extremely challenging, which is a problem for neuromorphic earth observation, because, without a proper estimation of the motion parameters, it is not possible to generate a map with high contrast, causing important details to be lost. Similar methods that use CMax addressed this problem by changing or augmenting the objective function to enable it to converge to the correct motion parameters. Our proposed solution overcomes the multiple extrema and noise-intolerance problems by correcting the warped event before calculating the contrast and offers the following advantages: it does not depend on the event data, it does not require a prior about the camera motion and keeps the rest of the CMax pipeline unchanged. This is to ensure that the contrast is only high around the correct motion parameters. Our approach enables the creation of better motion-compensated maps through an analytical compensation technique using a novel dataset from the International Space Station (ISS). Code is available at https://github.com/neuromorphicsystems/event_warping

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
* Empowering Resampling Operation for Ultra-High-Definition Image Enhancement with Model-Aware Guidance
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Empowering_Resampling_Operation_for_Ultra-High-Definition_Image_Enhancement_with_Model-Aware_Guidance_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Empowering_Resampling_Operation_for_Ultra-High-Definition_Image_Enhancement_with_Model-Aware_Guidance_CVPR_2024_paper.pdf)]
    * Title: Empowering Resampling Operation for Ultra-High-Definition Image Enhancement with Model-Aware Guidance
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Wei Yu, Jie Huang, Bing Li, Kaiwen Zheng, Qi Zhu, Man Zhou, Feng Zhao
    * Abstract: Image enhancement algorithms have made remarkable advancements in recent years but directly applying them to Ultra-high-definition (UHD) images presents intractable computational overheads. Therefore previous straightforward solutions employ resampling techniques to reduce the resolution by adopting a "Downsampling-Enhancement-Upsampling" processing paradigm. However this paradigm disentangles the resampling operators and inner enhancement algorithms which results in the loss of information that is favored by the model further leading to sub-optimal outcomes. In this paper we propose a novel method of Learning Model-Aware Resampling (LMAR) which learns to customize resampling by extracting model-aware information from the UHD input image under the guidance of model knowledge. Specifically our method consists of two core designs namely compensatory kernel estimation and steganographic resampling. At the first stage we dynamically predict compensatory kernels tailored to the specific input and resampling scales. At the second stage the image-wise compensatory information is derived with the compensatory kernels and embedded into the rescaled input images. This promotes the representation of the newly derived downscaled inputs to be more consistent with the full-resolution UHD inputs as perceived by the model. Our LMAR enables model-aware and model-favored resampling while maintaining compatibility with existing resampling operators. Extensive experiments on multiple UHD image enhancement datasets and different backbones have shown consistent performance gains after correlating resizer and enhancer e.g. up to 1.2dB PSNR gain for x1.8 resampling scale on UHD-LOL4K. The code is available at \href https://github.com/YPatrickW/LMAR https://github.com/YPatrickW/LMAR .

count=2
* Unmixing Diffusion for Self-Supervised Hyperspectral Image Denoising
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zeng_Unmixing_Diffusion_for_Self-Supervised_Hyperspectral_Image_Denoising_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zeng_Unmixing_Diffusion_for_Self-Supervised_Hyperspectral_Image_Denoising_CVPR_2024_paper.pdf)]
    * Title: Unmixing Diffusion for Self-Supervised Hyperspectral Image Denoising
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Haijin Zeng, Jiezhang Cao, Kai Zhang, Yongyong Chen, Hiep Luong, Wilfried Philips
    * Abstract: Hyperspectral images (HSIs) have extensive applications in various fields such as medicine agriculture and industry. Nevertheless acquiring high signal-to-noise ratio HSI poses a challenge due to narrow-band spectral filtering. Consequently the importance of HSI denoising is substantial especially for snapshot hyperspectral imaging technology. While most previous HSI denoising methods are supervised creating supervised training datasets for the diverse scenes hyperspectral cameras and scan parameters is impractical. In this work we present Diff-Unmix a self-supervised denoising method for HSI using diffusion denoising generative models. Specifically Diff-Unmix addresses the challenge of recovering noise-degraded HSI through a fusion of Spectral Unmixing and conditional abundance generation. Firstly it employs a learnable block-based spectral unmixing strategy complemented by a pure transformer-based backbone. Then we introduce a self-supervised generative diffusion network to enhance abundance maps from the spectral unmixing block. This network reconstructs noise-free Unmixing probability distributions effectively mitigating noise-induced degradations within these components. Finally the reconstructed HSI is reconstructed through unmixing reconstruction by blending the diffusion-adjusted abundance map with the spectral endmembers. Experimental results on both simulated and real-world noisy datasets show that Diff-Unmix achieves state-of-the-art performance.

count=2
* Flexible Background Subtraction With Self-Balanced Local Sensitivity
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf)]
    * Title: Flexible Background Subtraction With Self-Balanced Local Sensitivity
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: Most background subtraction approaches offer decent results in baseline scenarios, but adaptive and flexible solutions are still uncommon as many require scenario-specific parameter tuning to achieve optimal performance. In this paper, we introduce a new strategy to tackle this problem that focuses on balancing the inner workings of a non-parametric model based on pixel-level feedback loops. Pixels are modeled using a spatiotemporal feature descriptor for increased sensitivity. Using the video sequences and ground truth annotations of the 2012 and 2014 CVPR Change Detection Workshops, we demonstrate that our approach outperforms all previously ranked methods in the original dataset while achieving good results in the most recent one.

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
* Simultaneous Detection and Removal of High Altitude Clouds From an Image
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Sandhan_Simultaneous_Detection_and_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Sandhan_Simultaneous_Detection_and_ICCV_2017_paper.pdf)]
    * Title: Simultaneous Detection and Removal of High Altitude Clouds From an Image
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Tushar Sandhan, Jin Young Choi
    * Abstract: Interestingly, shape of the high-altitude clouds serves as a beacon for weather forecasting, so its detection is of vital importance. Besides these clouds often cause hindrance in an endeavor of satellites to inspect our world. Even thin clouds produce the undesired superposition of visual information, whose decomposition into the clear background and cloudy layer using a single satellite image is a highly ill-posed problem. In this work, we derive sophisticated image priors by thoroughly analyzing the properties of high-altitude clouds and geological images; and formulate a non-convex optimization scheme, which simultaneously detects and removes the clouds within a few seconds. Experimental results on real world RGB images demonstrate that the proposed method outperforms the other competitive methods by retaining the comprehensive background details and producing the precise shape of the cloudy layer.

count=2
* Learning to Detect Fine-Grained Change Under Variant Imaging Conditions
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w42/html/Huang_Learning_to_Detect_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w42/Huang_Learning_to_Detect_ICCV_2017_paper.pdf)]
    * Title: Learning to Detect Fine-Grained Change Under Variant Imaging Conditions
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Rui Huang, Wei Feng, Zezheng Wang, Mingyuan Fan, Liang Wan, Jizhou Sun
    * Abstract: Fine-grained change detection under variant imaging conditions is an important and challenging task for high-value scene monitoring in culture heritage. In this paper, we show that after a simple coarse alignment of lighting and camera differences, fine-grained change detection can be reliably solved by a deep network model, which is specifically composed of three functional parts, i.e., camera pose correction network (PCN), fine-grained change detection network (FCDN), and detection confidence boosting. Since our model is properly pre-trained and fine-tuned on both general and specialized data, it exhibits very good generalization capability to produce high-quality minute change detection on real-world scenes under varied imaging conditions. Extensive experiments validate the superior effectiveness and reliability over state-of-the-art methods. We have achieved 67.41% relative F1-measure improvement over the best competitor on real-world benchmark dataset.

count=2
* Sparse-to-Dense Feature Matching: Intra and Inter Domain Cross-Modal Learning in Domain Adaptation for 3D Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Sparse-to-Dense_Feature_Matching_Intra_and_Inter_Domain_Cross-Modal_Learning_in_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Peng_Sparse-to-Dense_Feature_Matching_Intra_and_Inter_Domain_Cross-Modal_Learning_in_ICCV_2021_paper.pdf)]
    * Title: Sparse-to-Dense Feature Matching: Intra and Inter Domain Cross-Modal Learning in Domain Adaptation for 3D Semantic Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Duo Peng, Yinjie Lei, Wen Li, Pingping Zhang, Yulan Guo
    * Abstract: Domain adaptation is critical for success when confronting with the lack of annotations in a new domain. As the huge time consumption of labeling process on 3D point cloud, domain adaptation for 3D semantic segmentation is of great expectation. With the rise of multi-modal datasets, large amount of 2D images are accessible besides 3D point clouds. In light of this, we propose to further leverage 2D data for 3D domain adaptation by intra and inter domain cross modal learning. As for intra-domain cross modal learning, most existing works sample the dense 2D pixel-wise features into the same size with sparse 3D point-wise features, resulting in the abandon of numerous useful 2D features. To address this problem, we propose Dynamic sparse-to-dense Cross Modal Learning (DsCML) to increase the sufficiency of multi-modality information interaction for domain adaptation. For inter-domain cross modal learning, we further advance Cross Modal Adversarial Learning (CMAL) on 2D and 3D data which contains different semantic content aiming to promote high-level modal complementarity. We evaluate our model under various multi-modality domain adaptation settings including day-to-night, country-to-country and dataset-to-dataset, brings large improvements over both uni-modal and multi-modal domain adaptation methods on all settings.

count=2
* Single Satellite Optical Imagery Dehazing using SAR Image Prior Based on conditional Generative Adversarial Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.pdf)]
    * Title: Single Satellite Optical Imagery Dehazing using SAR Image Prior Based on conditional Generative Adversarial Networks
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Binghui Huang,  Li Zhi,  Chao Yang,  Fuchun Sun,  Yixu Song
    * Abstract: Satellite image dehazing aims at precisely retrieving the real situations of the obscured parts from the hazy remote sensing (RS) images, which is a challenging task since the hazy regions contain both ground features and haze components. Many approaches of removing haze focus on processing multi-spectral or RGB images, whereas few of them utilize multi-sensor data. The multi-sensor data fusion is significant to provide auxiliary information since RGB images are sensitive to atmospheric conditions. In this paper, a dataset called SateHaze1k is established and composed of 1200 pairs clear Synthetic Aperture Radar (SAR), hazy RGB, and corresponding ground truth images, which are divided into three degrees of the haze, i.e. thin, moderate, and thick fog. Moreover, we propose a novel fusion dehazing method to directly restore the haze-free RS images by using an end-to-end conditional generative adversarial network(cGAN). The proposed network combines the information of both RGB and SAR images to eliminate the image blurring. Besides, the dilated residual blocks of the generator can also sufficiently improve the dehazing effects. Our experiments demonstrate that the proposed method, which fuses the information of different sensors applied to the cloudy conditions, can achieve more precise results than other baseline models.

count=2
* MotionRec: A Unified Deep Framework for Moving Object Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Mandal_MotionRec_A_Unified_Deep_Framework_for_Moving_Object_Recognition_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Mandal_MotionRec_A_Unified_Deep_Framework_for_Moving_Object_Recognition_WACV_2020_paper.pdf)]
    * Title: MotionRec: A Unified Deep Framework for Moving Object Recognition
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Murari Mandal,  Lav Kush Kumar,  Mahipal Singh Saran,  Santosh Kumar vipparthi
    * Abstract: In this paper we present a novel deep learning framework to perform online moving object recognition (MOR) in streaming videos. The existing methods for moving object detection (MOD) only computes class-agnostic pixel-wise binary segmentation of video frames. On the other hand, the object detection techniques do not differentiate between static and moving objects. To the best of our knowledge, this is a first attempt for simultaneous localization and classification of moving objects in a video, i.e. MOR in a single-stage deep learning framework. We achieve this by labelling axis-aligned bounding boxes for moving objects which requires less computational resources than producing pixel-level estimates. In the proposed MotionRec, both temporal and spatial features are learned using past history and current frames respectively. First, the background is estimated with a temporal depth reductionist (TDR) block. Then the estimated background, current frame and temporal median of recent observations are assimilated to encode spatiotemporal motion saliency. Moreover, feature pyramids are generated from these motion saliency maps to perform regression and classification at multiple levels of feature abstractions. MotionRec works online at inference as it requires only few past frames for MOR. Moreover, it doesn't require predefined target initialization from user. We also annotated axis-aligned bounding boxes (42,614 objects (14,814 cars and 27,800 person) in 24,923 video frames of CDnet 2014 dataset) due to lack of available benchmark datasets for MOR. The performance is observed qualitatively and quantitatively in terms of mAP over a defined unseen test set. Experiments show that the proposed MotionRec significantly improves over strong baselines with RetinaNet architectures for MOR.

count=2
* Ev-NeRF: Event Based Neural Radiance Field
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Hwang_Ev-NeRF_Event_Based_Neural_Radiance_Field_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Hwang_Ev-NeRF_Event_Based_Neural_Radiance_Field_WACV_2023_paper.pdf)]
    * Title: Ev-NeRF: Event Based Neural Radiance Field
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Inwoo Hwang, Junho Kim, Young Min Kim
    * Abstract: We present Ev-NeRF, a Neural Radiance Field derived from event data. While event cameras can measure subtle brightness changes in high frame rates, the measurements in low lighting or extreme motion suffer from significant domain discrepancy with complex noise. As a result, the performance of event-based vision tasks does not transfer to challenging environments, where the event cameras are expected to thrive over normal cameras. We find that the multi-view consistency of NeRF provides a powerful self-supervision signal for eliminating spurious measurements and extracting the consistent underlying structure despite highly noisy input. Instead of posed images of the original NeRF, the input to Ev-NeRF is the event measurements accompanied by the movements of the sensors. Using the loss function that reflects the measurement model of the sensor, Ev-NeRF creates an integrated neural volume that summarizes the unstructured and sparse data points captured for about 2-4 seconds. The generated neural volume can also produce intensity images from novel views with reasonable depth estimates, which can serve as a high-quality input to various vision-based tasks. Our results show that Ev-NeRF achieves competitive performance for intensity image reconstruction under extreme noise conditions and high-dynamic-range imaging.

count=2
* ZRG: A Dataset for Multimodal 3D Residential Rooftop Understanding
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Corley_ZRG_A_Dataset_for_Multimodal_3D_Residential_Rooftop_Understanding_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Corley_ZRG_A_Dataset_for_Multimodal_3D_Residential_Rooftop_Understanding_WACV_2024_paper.pdf)]
    * Title: ZRG: A Dataset for Multimodal 3D Residential Rooftop Understanding
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Isaac Corley, Jonathan Lwowski, Peyman Najafirad
    * Abstract: A crucial part of any home is the roof over our heads to protect us from the elements. In this paper we present the Zeitview Rooftop Geometry (ZRG) dataset for residential rooftop understanding. ZRG is a large-scale residential rooftop inspection dataset of over 20k properties from across the U.S. and includes high resolution aerial orthomosaics, digital surface models (DSM), colored point clouds, and 3D roof wireframe annotations. We provide an in-depth analysis and perform several experimental baselines including roof outline extraction, monocular height estimation, and planar roof structure extraction, to illustrate a few of the numerous applications unlocked by this dataset.

count=2
* Spatio-Temporal Filter Analysis Improves 3D-CNN for Action Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Kobayashi_Spatio-Temporal_Filter_Analysis_Improves_3D-CNN_for_Action_Classification_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kobayashi_Spatio-Temporal_Filter_Analysis_Improves_3D-CNN_for_Action_Classification_WACV_2024_paper.pdf)]
    * Title: Spatio-Temporal Filter Analysis Improves 3D-CNN for Action Classification
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Takumi Kobayashi, Jiaxing Ye
    * Abstract: As 2D-CNNs are growing in image recognition literature, 3D-CNNs are enthusiastically applied to video action recognition. While spatio-temporal (3D) convolution successfully stems from spatial (2D) convolution, it is still unclear how the convolution works for encoding temporal motion patterns in 3D-CNNs. In this paper, we shed light on the mechanism of feature extraction through analyzing the spatio-temporal filters from a temporal viewpoint. The analysis not only describes characteristics of the two action datasets, Something-Something-v2 (SSv2) and Kinetics-400, but also reveals how temporal dynamics are characterized through stacked spatio-temporal convolutions. Based on the analysis, we propose methods to improve temporal feature extraction, covering temporal filter representation and temporal data augmentation. The proposed method contributes to enlarging temporal receptive field of 3D-CNN without touching its fundamental architecture, thus keeping the computation cost. In the experiments on action classification using SSv2 and Kinetics-400, it produces favorable performance improvement of 3D-CNNs.

count=2
* Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/b01153e7112b347d8ed54f317840d8af-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Utkarsh Mall, Bharath Hariharan, Kavita Bala
    * Abstract: Satellite imagery is increasingly available, high resolution, and temporally detailed. Changes in spatio-temporal datasets such as satellite images are particularly interesting as they reveal the many events and forces that shape our world. However, finding such interesting and meaningful change events from the vast data is challenging. In this paper, we present new datasets for such change events that include semantically meaningful events like road construction. Instead of manually annotating the very large corpus of satellite images, we introduce a novel unsupervised approach that takes a large spatio-temporal dataset from satellite images and finds interesting change events. To evaluate the meaningfulness on these datasets we create 2 benchmarks namely CaiRoad and CalFire which capture the events of road construction and forest fires. These new benchmarks can be used to evaluate semantic retrieval/classification performance. We explore these benchmarks qualitatively and quantitatively by using several methods and show that these new datasets are indeed challenging for many existing methods.

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
* City-Scale Change Detection in Cadastral 3D Models Using Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Taneja_City-Scale_Change_Detection_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Taneja_City-Scale_Change_Detection_2013_CVPR_paper.pdf)]
    * Title: City-Scale Change Detection in Cadastral 3D Models Using Images
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Aparna Taneja, Luca Ballan, Marc Pollefeys
    * Abstract: In this paper, we propose a method to detect changes in the geometry of a city using panoramic images captured by a car driving around the city. We designed our approach to account for all the challenges involved in a large scale application of change detection, such as, inaccuracies in the input geometry, errors in the geo-location data of the images, as well as, the limited amount of information due to sparse imagery. We evaluated our approach on an area of 6 square kilometers inside a city, using 3420 images downloaded from Google StreetView. These images besides being publicly available, are also a good example of panoramic images captured with a driving vehicle, and hence demonstrating all the possible challenges resulting from such an acquisition. We also quantitatively compared the performance of our approach with respect to a ground truth, as well as to prior work. This evaluation shows that our approach outperforms the current state of the art.

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
* Semantic 3D Reconstruction With Continuous Regularization and Ray Potentials Using a Visibility Consistency Constraint
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Savinov_Semantic_3D_Reconstruction_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Savinov_Semantic_3D_Reconstruction_CVPR_2016_paper.pdf)]
    * Title: Semantic 3D Reconstruction With Continuous Regularization and Ray Potentials Using a Visibility Consistency Constraint
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Nikolay Savinov, Christian Hane, Lubor Ladicky, Marc Pollefeys
    * Abstract: We propose an approach for dense semantic 3D reconstruction which uses a data term that is defined as potentials over viewing rays, combined with continuous surface area penalization. Our formulation is a convex relaxation which we augment with a crucial non-convex constraint that ensures exact handling of visibility. To tackle the non-convex minimization problem, we propose a majorize-minimize type strategy which converges to a critical point. We demonstrate the benefits of using the non-convex constraint experimentally. For the geometry-only case, we set a new state of the art on two datasets of the commonly used Middlebury multi-view stereo benchmark. Moreover, our general-purpose formulation directly reconstructs thin objects, which are usually treated with specialized algorithms. A qualitative evaluation on the dense semantic 3D reconstruction task shows that we improve significantly over previous methods.

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
* Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Makihara_Joint_Intensity_and_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Makihara_Joint_Intensity_and_CVPR_2017_paper.pdf)]
    * Title: Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yasushi Makihara, Atsuyuki Suzuki, Daigo Muramatsu, Xiang Li, Yasushi Yagi
    * Abstract: This paper describes a joint intensity metric learning method to improve the robustness of gait recognition with silhouette-based descriptors such as gait energy images. Because existing methods often use the difference of image intensities between a matching pair (e.g., the absolute difference of gait energies for the l_1-norm) to measure a dissimilarity, large intrasubject differences derived from covariate conditions (e.g., large gait energies caused by carried objects vs. small gait energies caused by the background), may wash out subtle intersubject differences (e.g., the difference of middle-level gait energies derived from motion differences). We therefore introduce a metric on joint intensity to mitigate the large intrasubject differences as well as leverage the subtle intersubject differences. More specifically, we formulate the joint intensity and spatial metric learning in a unified framework and alternately optimize it by linear or ranking support vector machines. Experiments using the OU-ISIR treadmill data set B with the largest clothing variation and large population data set with bag, b version containing carrying status in the wild demonstrate the effectiveness of the proposed method.

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
* Joint Learning From Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Audebert_Joint_Learning_From_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Audebert_Joint_Learning_From_CVPR_2017_paper.pdf)]
    * Title: Joint Learning From Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Nicolas Audebert, Bertrand Le Saux, Sebastien Lefevre
    * Abstract: We investigate the use of OSM data for semantic labeling of EO images. Deep neural networks have been used in the past for remote sensing data classification from various sensors, including multispectral, hyperspectral, Radar and Lidar data. However, OSM is an abundant data source that has already been used as ground truth data, but rarely exploited as an input information layer. We study different use cases and deep network architectures to leverage this OSM data for semantic labeling of aerial and satellite images. Especially, we look into fusion based architectures and coarse-to-fine segmentation to include the OSM layer into multispectral-based deep fully convolutional networks. We illustrate how these methods can be used successfully on two public datasets: the ISPRS Potsdam and the DFC2017. We show that OSM data can efficiently be integrated into the vision-based deep learning models and that it significantly improves both the accuracy performance and the convergence.

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
* HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Sironi_HATS_Histograms_of_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf)]
    * Title: HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Amos Sironi, Manuele Brambilla, Nicolas Bourdis, Xavier Lagorce, Ryad Benosman
    * Abstract: Event-based cameras have recently drawn the attention of the Computer Vision community thanks to their advantages in terms of high temporal resolution, low power consumption and high dynamic range, compared to traditional frame-based cameras. These properties make event-based cameras an ideal choice for autonomous vehicles, robot navigation or UAV vision, among others. However, the accuracy of event-based object classification algorithms, which is of crucial importance for any reliable system working in real-world conditions, is still far behind their frame-based counterparts. Two main reasons for this performance gap are: 1. The lack of effective low-level representations and architectures for event-based object classification and 2. The absence of large real-world event-based datasets. In this paper we address both problems. First, we introduce a novel event-based feature representation together with a new machine learning architecture. Compared to previous approaches, we use local memory units to efficiently leverage past temporal information and build a robust event-based representation. Second, we release the first large real-world event-based dataset for object classification. We compare our method to the state-of-the-art with extensive experiments, showing better classification performance and real-time computation.

count=1
* Crowd Activity Change Point Detection in Videos via Graph Stream Mining
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w6/html/Yang_Crowd_Activity_Change_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Yang_Crowd_Activity_Change_CVPR_2018_paper.pdf)]
    * Title: Crowd Activity Change Point Detection in Videos via Graph Stream Mining
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Meng Yang, Lida Rashidi, Sutharshan Rajasegarar, Christopher Leckie, Aravinda S. Rao, Marimuthu Palaniswami
    * Abstract: In recent years, there has been a growing interest in detecting anomalous behavioral patterns in video. In this work, we address this task by proposing a novel activity change point detection method to identify crowd movement anomalies for video surveillance. In our proposed novel framework, a hyperspherical clustering algorithm is utilized for the automatic identification of interesting regions, then the density of pedestrian flows between every pair of interesting regions over consecutive time intervals is monitored and represented as a sequence of adjacency matrices where the direction and density of flows are captured through a directed graph. Finally, we use graph edit distance as well as a cumulative sum test to detect change points in the graph sequence. We conduct experiments on four real-world video datasets: Dublin, New Orleans, Abbey Road and MCG Datasets. We observe that our proposed approach achieves a high F-measure, i.e., in the range [0.7, 1], for these datasets. The evaluation reveals that our proposed method can successfully detect the change points in all datasets at both global and local levels. Our results also demonstrate the efficiency and effectiveness of our proposed algorithm for change point detection and segmentation tasks.

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
* Event Probability Mask (EPM) and Event Denoising Convolutional Neural Network (EDnCNN) for Neuromorphic Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Baldwin_Event_Probability_Mask_EPM_and_Event_Denoising_Convolutional_Neural_Network_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Baldwin_Event_Probability_Mask_EPM_and_Event_Denoising_Convolutional_Neural_Network_CVPR_2020_paper.pdf)]
    * Title: Event Probability Mask (EPM) and Event Denoising Convolutional Neural Network (EDnCNN) for Neuromorphic Cameras
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: R. Wes Baldwin,  Mohammed Almatrafi,  Vijayan Asari,  Keigo Hirakawa
    * Abstract: This paper presents a novel method for labeling real-world neuromorphic camera sensor data by calculating the likelihood of generating an event at each pixel within a short time window, which we refer to as "event probability mask" or EPM. Its applications include (i) objective benchmarking of event denoising performance, (ii) training convolutional neural networks for noise removal called "event denoising convolutional neural network" (EDnCNN), and (iii) estimating internal neuromorphic camera parameters. We provide the first dataset (DVSNOISE20) of real-world labeled neuromorphic camera events for noise removal.

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
* Joint Filtering of Intensity Images and Neuromorphic Events for High-Resolution Noise-Robust Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Joint_Filtering_of_Intensity_Images_and_Neuromorphic_Events_for_High-Resolution_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Joint_Filtering_of_Intensity_Images_and_Neuromorphic_Events_for_High-Resolution_CVPR_2020_paper.pdf)]
    * Title: Joint Filtering of Intensity Images and Neuromorphic Events for High-Resolution Noise-Robust Imaging
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zihao W. Wang,  Peiqi Duan,  Oliver Cossairt,  Aggelos Katsaggelos,  Tiejun Huang,  Boxin Shi
    * Abstract: We present a novel computational imaging system with high resolution and low noise. Our system consists of a traditional video camera which captures high-resolution intensity images, and an event camera which encodes high-speed motion as a stream of asynchronous binary events. To process the hybrid input, we propose a unifying framework that first bridges the two sensing modalities via a noise-robust motion compensation model, and then performs joint image filtering. The filtered output represents the temporal gradient of the captured space-time volume, which can be viewed as motion-compensated event frames with high resolution and low noise. Therefore, the output can be widely applied to many existing event-based algorithms that are highly dependent on spatial resolution and noise robustness. In experimental results performed on both publicly available datasets as well as our contributing RGB-DAVIS dataset, we show systematic performance improvement in applications such as high frame-rate video synthesis, feature/corner detection and tracking, as well as high dynamic range image reconstruction.

count=1
* The Multi-Temporal Urban Development SpaceNet Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.pdf)]
    * Title: The Multi-Temporal Urban Development SpaceNet Dataset
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Adam Van Etten, Daniel Hogan, Jesus Martinez Manso, Jacob Shermeyer, Nicholas Weir, Ryan Lewis
    * Abstract: Satellite imagery analytics have numerous human development and disaster response applications, particularly when time series methods are involved. For example, quantifying population statistics is fundamental to 67 of the 231 United Nations Sustainable Development Goals Indicators, but the World Bank estimates that over 100 countries currently lack effective Civil Registration systems. To help address this deficit and develop novel computer vision methods for time series data, we present the Multi-Temporal Urban Development SpaceNet (MUDS, also known as SpaceNet 7) dataset. This open source dataset consists of medium resolution (4.0m) satellite imagery mosaics, which includes 24 images (one per month) covering >100 unique geographies, and comprises >40,000 km2 of imagery and exhaustive polygon labels of building footprints therein, totaling over 11M individual annotations. Each building is assigned a unique identifier (i.e. address), which permits tracking of individual objects over time. Label fidelity exceeds image resolution; this "omniscient labeling" is a unique feature of the dataset, and enables surprisingly precise algorithmic models to be crafted. We demonstrate methods to track building footprint construction (or demolition) over time, thereby directly assessing urbanization. Performance is measured with the newly developed SpaceNet Change and Object Tracking (SCOT) metric, which quantifies both object tracking as well as change detection. We demonstrate that despite the moderate resolution of the data, we are able to track individual building identifiers over time.

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
* Perceptual Loss for Robust Unsupervised Homography Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/IMW/html/Koguciuk_Perceptual_Loss_for_Robust_Unsupervised_Homography_Estimation_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/IMW/papers/Koguciuk_Perceptual_Loss_for_Robust_Unsupervised_Homography_Estimation_CVPRW_2021_paper.pdf)]
    * Title: Perceptual Loss for Robust Unsupervised Homography Estimation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Daniel Koguciuk, Elahe Arani, Bahram Zonooz
    * Abstract: Homography estimation is often an indispensable step in many computer vision tasks. The existing approaches, however, are not robust to illumination and/or larger viewpoint changes. In this paper, we propose bidirectional implicit Homography Estimation (biHomE) loss for unsupervised homography estimation. biHomE minimizes the distance in the feature space between the warped image from the source viewpoint and the corresponding image from the target viewpoint. Since we use a fixed pre-trained feature extractor and the only learnable component of our framework is the homography network, we effectively decouple the homography estimation from representation learning. We use an additional photometric distortion step in the synthetic COCO dataset generation to better represent the illumination variation of the real-world scenarios. We show that biHomE achieves state-of-the-art performance on synthetic COCO dataset, which is also comparable or better compared to supervised approaches. Furthermore, the empirical results demonstrate the robustness of our approach to illumination variation compared to existing methods.

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
* Deep Anomaly Discovery From Unlabeled Videos via Normality Advantage and Self-Paced Refinement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.pdf)]
    * Title: Deep Anomaly Discovery From Unlabeled Videos via Normality Advantage and Self-Paced Refinement
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Guang Yu, Siqi Wang, Zhiping Cai, Xinwang Liu, Chuanfu Xu, Chengkun Wu
    * Abstract: While classic video anomaly detection (VAD) requires labeled normal videos for training, emerging unsupervised VAD (UVAD) aims to discover anomalies directly from fully unlabeled videos. However, existing UVAD methods still rely on shallow models to perform detection or initialization, and they are evidently inferior to classic VAD methods. This paper proposes a full deep neural network (DNN) based solution that can realize highly effective UVAD. First, we, for the first time, point out that deep reconstruction can be surprisingly effective for UVAD, which inspires us to unveil a property named "normality advantage", i.e., normal events will enjoy lower reconstruction loss when DNN learns to reconstruct unlabeled videos. With this property, we propose Localization based Reconstruction (LBR) as a strong UVAD baseline and a solid foundation of our solution. Second, we propose a novel self-paced refinement (SPR) scheme, which is synthesized into LBR to conduct UVAD. Unlike ordinary self-paced learning that injects more samples in an easy-to-hard manner, the proposed SPR scheme gradually drops samples so that suspicious anomalies can be removed from the learning process. In this way, SPR consolidates normality advantage and enables better UVAD in a more proactive way. Finally, we further design a variant solution that explicitly takes the motion cues into account. The solution evidently enhances the UVAD performance, and it sometimes even surpasses the best classic VAD methods. Experiments show that our solution not only significantly outperforms existing UVAD methods by a wide margin (5% to 9% AUROC), but also enables UVAD to catch up with the mainstream performance of classic VAD.

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
* Pix2map: Cross-Modal Retrieval for Inferring Street Maps From Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Pix2map_Cross-Modal_Retrieval_for_Inferring_Street_Maps_From_Images_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Pix2map_Cross-Modal_Retrieval_for_Inferring_Street_Maps_From_Images_CVPR_2023_paper.pdf)]
    * Title: Pix2map: Cross-Modal Retrieval for Inferring Street Maps From Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Xindi Wu, KwunFung Lau, Francesco Ferroni, Aljoša Ošep, Deva Ramanan
    * Abstract: Self-driving vehicles rely on urban street maps for autonomous navigation. In this paper, we introduce Pix2Map, a method for inferring urban street map topology directly from ego-view images, as needed to continually update and expand existing maps. This is a challenging task, as we need to infer a complex urban road topology directly from raw image data. The main insight of this paper is that this problem can be posed as cross-modal retrieval by learning a joint, cross-modal embedding space for images and existing maps, represented as discrete graphs that encode the topological layout of the visual surroundings. We conduct our experimental evaluation using the Argoverse dataset and show that it is indeed possible to accurately retrieve street maps corresponding to both seen and unseen roads solely from image data. Moreover, we show that our retrieved maps can be used to update or expand existing maps and even show proof-of-concept results for visual localization and image retrieval from spatial graphs.

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
* HIR-Diff: Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.pdf)]
    * Title: HIR-Diff: Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Li Pang, Xiangyu Rui, Long Cui, Hongzhong Wang, Deyu Meng, Xiangyong Cao
    * Abstract: Hyperspectral image (HSI) restoration aims at recovering clean images from degraded observations and plays a vital role in downstream tasks. Existing model-based methods have limitations in accurately modeling the complex image characteristics with handcraft priors and deep learning-based methods suffer from poor generalization ability. To alleviate these issues this paper proposes an unsupervised HSI restoration framework with pre-trained diffusion model (HIR-Diff) which restores the clean HSIs from the product of two low-rank components i.e. the reduced image and the coefficient matrix. Specifically the reduced image which has a low spectral dimension lies in the image field and can be inferred from our improved diffusion model where a new guidance function with total variation (TV) prior is designed to ensure that the reduced image can be well sampled. The coefficient matrix can be effectively pre-estimated based on singular value decomposition (SVD) and rank-revealing QR (RRQR) factorization. Furthermore a novel exponential noise schedule is proposed to accelerate the restoration process (about 5xacceleration for denoising) with little performance decrease. Extensive experimental results validate the superiority of our method in both performance and speed on a variety of HSI restoration tasks including HSI denoising noisy HSI super-resolution and noisy HSI inpainting. The code is available at https://github.com/LiPang/HIRDiff.

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
* Revisiting Pre-trained Remote Sensing Model Benchmarks: Resizing and Normalization Matters
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Corley_Revisiting_Pre-trained_Remote_Sensing_Model_Benchmarks_Resizing_and_Normalization_Matters_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Corley_Revisiting_Pre-trained_Remote_Sensing_Model_Benchmarks_Resizing_and_Normalization_Matters_CVPRW_2024_paper.pdf)]
    * Title: Revisiting Pre-trained Remote Sensing Model Benchmarks: Resizing and Normalization Matters
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Isaac Corley, Caleb Robinson, Rahul Dodhia, Juan M. Lavista Ferres, Peyman Najafirad
    * Abstract: Research in self-supervised learning (SSL) with natural images has progressed rapidly in recent years and is now increasingly being applied to and benchmarked with datasets containing remotely sensed imagery. A common benchmark case is to evaluate SSL pre-trained model embeddings on datasets of remotely sensed imagery with small patch sizes e.g. 32 x 32 pixels whereas standard SSL pre-training takes place with larger patch sizes e.g. 224 x 224. Furthermore pre-training methods tend to use different image normalization preprocessing steps depending on the dataset. In this paper we show across seven satellite and aerial imagery datasets of varying resolution that by simply following the preprocessing steps used in pre-training (precisely image sizing and normalization methods) one can achieve significant performance improvements when evaluating the extracted features on downstream tasks -- an important detail overlooked in previous work in this space. We show that by following these steps ImageNet pre-training remains a competitive baseline for satellite imagery based transfer learning tasks -- for example we find that these steps give +32.28 to overall accuracy on the So2Sat random split dataset and +11.16 on the EuroSAT dataset. Finally we report comprehensive benchmark results with a variety of simple baseline methods for each of the seven datasets forming an initial benchmark suite for remote sensing imagery.

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
* iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.pdf)]
    * Title: iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Syed Waqas Zamir,  Aditya Arora,  Akshita Gupta,  Salman  Khan,  Guolei Sun,  Fahad Shahbaz Khan,  Fan Zhu,  Ling Shao,  Gui-Song Xia,  Xiang Bai
    * Abstract: Existing Earth Vision datasets are either suitable for semantic segmentation or object detection. In this work, we introduce the first benchmark dataset for instance segmentation in aerial imagery that combines instance-level object detection and pixel-level segmentation tasks. In comparison to instance segmentation in natural scenes, aerial images present unique challenges e.g., huge number of instances per image, large object-scale variations and abundant tiny objects. Our large-scale and densely annotated Instance Segmentation in Aerial Images Dataset (IS-AID) comes with 655,451 object instances for 15 categories across 2,806 high-resolution images. Such precise per-pixel annotations for each instance ensure accurate localization that is essential for detailed scene analysis. Compared to existing small-scale aerial image based instance segmentation datasets, IS-AID contains 15x the number of object categories and 5x the number of instances. We benchmark our dataset using two popular instance segmentation approaches for natural images, namely Mask R-CNN and PANet. In our experiments we show that direct application of off-the-shelf Mask R-CNN and PANet on aerial images provide sub-optimal instance segmentation results, thus requiring specialized solutions from the research community.

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
* HIDeGan: A Hyperspectral-Guided Image Dehazing GAN
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w14/Mehta_HIDeGan_A_Hyperspectral-Guided_Image_Dehazing_GAN_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w14/Mehta_HIDeGan_A_Hyperspectral-Guided_Image_Dehazing_GAN_CVPRW_2020_paper.pdf)]
    * Title: HIDeGan: A Hyperspectral-Guided Image Dehazing GAN
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Aditya Mehta, Harsh Sinha, Pratik Narang, Murari Mandal
    * Abstract: Haze removal in images captured from a diverse set of scenarios is a very challenging problem. The existing dehazing methods either reconstruct the transmission map or directly estimate the dehazed image in RGB color space. In this paper, we make a first attempt to propose a Hyperspectral-guided Image Dehazing Generative Adversarial Network (HIDEGAN). The HIDEGAN architecture is formulated by designing an enhanced version of CYCLEGAN named R2HCYCLE and an enhanced conditional GAN named H2RGAN. The R2HCYCLE makes use of the hyperspectral-image (HSI) in combination with cycle-consistency and skeleton losses in order to improve the quality of information recovery by analyzing the entire spectrum. The H2RGAN estimates the clean RGB image from the hazy hyperspectral image generated by the R2HCYCLE. The models designed for spatial-spectral-spatial mapping generate visually better haze-free images. To facilitate HSI generation, datasets from spectral reconstruction challenge at NTIRE 2018 and NTIRE 2020 are used. A comprehensive set of experiments were conducted on the D-Hazy, and the recent RESIDE-Standard (SOTS), RESIDE-b (OTS) and RESIDE-Standard (HSTS) datasets. The proposed HIDEGAN outperforms the existing state-of-the-art in all these datasets.

count=1
* iTASK - Intelligent Traffic Analysis Software Kit
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w35/Tran_iTASK_-_Intelligent_Traffic_Analysis_Software_Kit_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Tran_iTASK_-_Intelligent_Traffic_Analysis_Software_Kit_CVPRW_2020_paper.pdf)]
    * Title: iTASK - Intelligent Traffic Analysis Software Kit
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Minh-Triet Tran, Tam V. Nguyen, Trung-Hieu Hoang, Trung-Nghia Le, Khac-Tuan Nguyen, Dat-Thanh Dinh, Thanh-An Nguyen, Hai-Dang Nguyen, Xuan-Nhat Hoang, Trong-Tung Nguyen, Viet-Khoa Vo-Ho, Trong-Le Do, Lam Nguyen, Minh-Quan Le, Hoang-Phuc Nguyen-Dinh, Trong-Thang Pham, Xuan-Vy Nguyen, E-Ro Nguyen, Quoc-Cuong Tran, Hung Tran, Hieu Dao, Mai-Khiem Tran, Quang-Thuc Nguyen, Tien-Phat Nguyen, The-Anh Vu-Le, Gia-Han Diep, Minh N. Do
    * Abstract: Traffic flow analysis is essential for intelligent transportation systems. In this paper, we introduce our Intelligent Traffic Analysis Software Kit (iTASK) to tackle three challenging problems: vehicle flow counting, vehicle re-identification, and abnormal event detection. For the first problem, we propose to real-time track vehicles moving along the desired direction in corresponding motion-of-interests (MOIs). For the second problem, we consider each vehicle as a document with multiple semantic words (i.e., vehicle attributes) and transform the given problem to classical document retrieval. For the last problem, we propose to forward and backward refine anomaly detection using GAN-based future prediction and backward tracking completely stalled vehicle or sudden-change direction, respectively. Experiments on the datasets of traffic flow analysis from AI City Challenge 2020 show our competitive results, namely, S1 score of 0.8297 for vehicle flow counting in Track 1, mAP score of 0.3882 for vehicle re-identification in Track 2, and S4 score of 0.9059 for anomaly detection in Track 4. All data and source code are publicly available on our project page.

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
* A Multi-sensor Fusion Framework in 3-D
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/html/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/papers/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.pdf)]
    * Title: A Multi-sensor Fusion Framework in 3-D
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Vishal Jain, Andrew C. Miller, Joseph L. Mundy
    * Abstract: The majority of existing image fusion techniques operate in the 2-d image domain which perform well for imagery of planar regions but fails in presence of any 3-d relief and provides inaccurate alignment of imagery from different sensors. A framework for multi-sensor image fusion in 3-d is proposed in this paper. The imagery from different sensors, specifically EO and IR, are fused in a common 3-d reference coordinate frame. A dense probabilistic and volumetric 3-d model is reconstructed from each of the sensors. The imagery is registered by aligning the 3-d models as the underlying 3-d structure in the images is the true invariant information. The image intensities are back-projected onto a 3-d model and every discretized location (voxel) of the 3-d model stores an array of intensities from different modalities. This 3-d model is forward-projected to produce a fused image of EO and IR from any viewpoint.

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
* Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.pdf)]
    * Title: Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Maria Vakalopoulou, Konstantinos Karantzalos, Nikos Komodakis, Nikos Paragios
    * Abstract: In order to exploit the currently continuous streams of massive, multi-temporal, high-resolution remote sensing datasets there is an emerging need to address efficiently the image registration and change detection challenges. To this end, in this paper we propose a modular, scalable, metric free single shot change detection/registration method. The approach exploits a decomposed interconnected graphical model formulation where registration similarity constraints are relaxed in the presence of change detection. The deformation space is discretized, while efficient linear programming and duality principles are used to optimize a joint solution space where local consistency is imposed on the deformation and the detection space. Promising results on large scale experiments demonstrate the extreme potentials of our method.

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
* Fine-Grained Change Detection of Misaligned Scenes With Varied Illuminations
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Feng_Fine-Grained_Change_Detection_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Feng_Fine-Grained_Change_Detection_ICCV_2015_paper.pdf)]
    * Title: Fine-Grained Change Detection of Misaligned Scenes With Varied Illuminations
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Wei Feng, Fei-Peng Tian, Qian Zhang, Nan Zhang, Liang Wan, Jizhou Sun
    * Abstract: Detecting fine-grained subtle changes among a scene is critically important in practice. Previous change detection methods, focusing on detecting large-scale significant changes, cannot do this well. This paper proposes a feasible end-to-end approach to this challenging problem. We start from active camera relocation that quickly relocates camera to nearly the same pose and position of the last time observation. To guarantee detection sensitivity and accuracy of minute changes, in an observation, we capture a group of images under multiple illuminations, which need only to be roughly aligned to the last time lighting conditions. Given two times observations, we formulate fine-grained change detection as a joint optimization problem of three related factors, i.e., normal-aware lighting difference, camera geometry correction flow, and real scene change mask. We solve the three factors in a coarse-to-fine manner and achieve reliable change decision by rank minimization. We build three real-world datasets to benchmark fine-grained change detection of misaligned scenes under varied multiple lighting conditions. Extensive experiments show the superior performance of our approach over state-of-the-art change detection methods and its ability to distinguish real scene changes from false ones caused by lighting variations.

count=1
* Rolling Shutter Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Punnappurath_Rolling_Shutter_Super-Resolution_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Punnappurath_Rolling_Shutter_Super-Resolution_ICCV_2015_paper.pdf)]
    * Title: Rolling Shutter Super-Resolution
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Abhijith Punnappurath, Vijay Rengarajan, A.N. Rajagopalan
    * Abstract: Classical multi-image super-resolution (SR) algorithms, designed for CCD cameras, assume that the motion among the images is global. But CMOS sensors that have increasingly started to replace their more expensive CCD counterparts in many applications do not respect this assumption if there is a motion of the camera relative to the scene during the exposure duration of an image because of the row-wise acquisition mechanism. In this paper, we study the hitherto unexplored topic of multi-image SR in CMOS cameras. We initially develop an SR observation model that accounts for the row-wise distortions called the ``rolling shutter'' (RS) effect observed in images captured using non-stationary CMOS cameras. We then propose a unified RS-SR framework to obtain an RS-free high-resolution image (and the row-wise motion) from distorted low-resolution images. We demonstrate the efficacy of the proposed scheme using synthetic data as well as real images captured using a hand-held CMOS camera. Quantitative and qualitative assessments reveal that our method significantly advances the state-of-the-art.

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
* Scene Categorization With Spectral Features
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Khan_Scene_Categorization_With_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Khan_Scene_Categorization_With_ICCV_2017_paper.pdf)]
    * Title: Scene Categorization With Spectral Features
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Salman H. Khan, Munawar Hayat, Fatih Porikli
    * Abstract: Spectral signatures of natural scenes were earlier found to be distinctive for different scene types with varying spatial envelope properties such as openness, naturalness, ruggedness, and symmetry. Recently, such handcrafted features have been outclassed by deep learning based representations. This paper proposes a novel spectral description of convolution features, implemented efficiently as a unitary transformation within deep network architectures. To the best of our knowledge, this is the first attempt to use deep learning based spectral features explicitly for image classification task. We show that the spectral transformation decorrelates convolutional activations, which reduces co-adaptation between feature detections, thus acts as an effective regularizer. Our approach achieves significant improvements on three large-scale scene-centric datasets (MIT-67, SUN-397, and Places-205). Furthermore, we evaluated the proposed approach on the attribute detection task where its superior performance manifests its relevance to semantically meaningful characteristics of natural scenes.

count=1
* Space-Time Localization and Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Lee_Space-Time_Localization_and_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Space-Time_Localization_and_ICCV_2017_paper.pdf)]
    * Title: Space-Time Localization and Mapping
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Minhaeng Lee, Charless C. Fowlkes
    * Abstract: This paper addresses the problem of building a spatio-temporal model of the world from a stream of time-stamped data. Unlike traditional models for simultaneous localization and mapping (SLAM) and structure-from-motion (SfM) which focus on recovering a single rigid 3D model, we tackle the problem of mapping scenes in which dynamic components appear, move and disappear independently of each other over time. We introduce a simple generative probabilistic model of 4D structure which specifies location, spatial and temporal extent of rigid surface patches by local Gaussian mixtures. We fit this model to a time-stamped stream of input data using expectation-maximization to estimate the model structure parameters (mapping) and the alignment of the input data to the model (localization). By explicitly representing the temporal extent and observability of surfaces in a scene, our method yields superior localization and reconstruction relative to baselines that assume a static 3D scene. We carry out experiments on both synthetic RGB-D data streams as well as challenging real-world datasets, tracking scene dynamics in a human workspace over the course of several weeks.

count=1
* Can We Speed up 3D Scanning? A Cognitive and Geometric Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w40/html/Vaiapury_Can_We_Speed_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w40/Vaiapury_Can_We_Speed_ICCV_2017_paper.pdf)]
    * Title: Can We Speed up 3D Scanning? A Cognitive and Geometric Analysis
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Karthikeyan Vaiapury, Balamuralidhar Purushothaman, Arpan Pal, Swapna Agarwal, Brojeshwar Bhowmick
    * Abstract: The paper propose a cognitive inspired change detection method for the detection and localization of shape variations on point clouds. A well defined pipeline is introduced by proposing a coarse to fine approach: i) shape segmentation, ii) fine segment registration using attention blocks. Shape segmentation is obtained using covariance based method and fine segment registration is carried out using gravitational registration algorithm. In particular the introduction of this partition-based approach using visual attention mechanism improves the speed of deformation detection and localization. Some results are shown on synthetic data of house and aircraft models. Experimental results shows that this simple yet effective approach designed with an eye to scalability can detect and localize the deformation in a faster manner. A real world car usecase is also presented with some preliminary promising results useful for auditing and insurance claim tasks.

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
* SiamSTA: Spatio-Temporal Attention Based Siamese Tracker for Tracking UAVs
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/html/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/papers/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.pdf)]
    * Title: SiamSTA: Spatio-Temporal Attention Based Siamese Tracker for Tracking UAVs
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Bo Huang, Junjie Chen, Tingfa Xu, Ying Wang, Shenwang Jiang, Yuncheng Wang, Lei Wang, Jianan Li
    * Abstract: With the growing threat of unmanned aerial vehicle (UAV) intrusion, anti-UAV techniques are becoming increasingly demanding. Object tracking, especially in thermal infrared (TIR) videos, though provides a promising solution, struggles with challenges like small scale and fast movement that commonly occur in anti-UAV scenarios. To mitigate this, we propose a simple yet effective spatio-temporal attention based Siamese network, dubbed SiamSTA, to track UAV robustly by performing reliable local tracking and wide-range re-detection alternatively. Concretely, tracking is carried out by posing spatial and temporal constraints on generating candidate proposals within local neighborhoods, hence eliminating background distractors to better perceive small targets. Complementarily, in case of target lost from local regions due to fast movement, a three-stage re-detection mechanism is introduced to re-detect targets from a global view by exploiting valuable motion cues through a correlation filter based on change detection. Finally, a state-aware switching policy is adopted to adaptively integrate local tracking and global re-detection and take their complementary strengths for robust tracking. Extensive experiments on the 1st and 2nd anti-UAV datasets well demonstrate the superiority of SiamSTA over other competing counterparts. Notably, SiamSTA is the foundation of the 1st-place winning entry in the 2nd Anti-UAV Challenge.

count=1
* Background/Foreground Separation: Guided Attention Based Adversarial Modeling (GAAM) Versus Robust Subspace Learning Methods
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Sultana_BackgroundForeground_Separation_Guided_Attention_Based_Adversarial_Modeling_GAAM_Versus_Robust_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Sultana_BackgroundForeground_Separation_Guided_Attention_Based_Adversarial_Modeling_GAAM_Versus_Robust_ICCVW_2021_paper.pdf)]
    * Title: Background/Foreground Separation: Guided Attention Based Adversarial Modeling (GAAM) Versus Robust Subspace Learning Methods
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Maryam Sultana, Arif Mahmood, Thierry Bouwmans, Muhammad Haris Khan, Soon Ki Jung
    * Abstract: Background-Foreground separation and appearance generation is a fundamental step in many computer vision applications. Existing methods like Robust Subspace Learning (RSL) suffer performance degradation in the presence of challenges like bad weather, illumination variations, occlusion, dynamic backgrounds and intermittent object motion. In the current work we propose a more accurate deep neural network based model for background-foreground separation and complete appearance generation of the foreground objects. Our proposed model, Guided Attention based Adversarial Model (GAAM), can efficiently extract pixel-level boundaries of the foreground objects for improved appearance generation. Unlike RSL methods our model extracts the binary information of foreground objects labeled as attention map which guides our generator network to segment the foreground objects from the complex background information. Wide range of experiments performed on the benchmark CDnet2014 dataset demonstrate the excellent performance of our proposed model.

count=1
* JanusNet: Detection of Moving Objects From UAV Platforms
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/html/Zhao_JanusNet_Detection_of_Moving_Objects_From_UAV_Platforms_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/papers/Zhao_JanusNet_Detection_of_Moving_Objects_From_UAV_Platforms_ICCVW_2021_paper.pdf)]
    * Title: JanusNet: Detection of Moving Objects From UAV Platforms
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yuxiang Zhao, Khurram Shafique, Zeeshan Rasheed, Maoxu Li
    * Abstract: In this paper, we present JanusNet, an efficient CNN model that can perform online background subtraction and robustly detect moving targets using resource-constrained computational hardware on-board unmanned aerial vehicles (UAVs). Most of the existing work on background subtraction either assume that the camera is stationary or make limiting assumptions about the motion of the camera, the structure of the scene under observation, or the apparent motion of the background in video. JanusNet does not have these limitations and therefore, is applicable to a variety of UAV applications. JanusNet learns to extract and combine motion and appearance features to separate background and foreground to generate accurate pixel-wise masks of the moving objects. The network is trained using a simulated video dataset (generated using Unreal Engine 4) with ground-truth labels. Results on UCF Aerial and Kaggle Drone videos datasets show that the learned model transfers well to real UAV videos and can robustly detect moving targets in a wide variety of scenarios. Moreover, experiments on CDNet dataset demonstrate that even without explicitly assuming that the camera is stationary, the performance of JanusNet is comparable to traditional background subtraction methods.

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
* Large Selective Kernel Network for Remote Sensing Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.pdf)]
    * Title: Large Selective Kernel Network for Remote Sensing Object Detection
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yuxuan Li, Qibin Hou, Zhaohui Zheng, Ming-Ming Cheng, Jian Yang, Xiang Li
    * Abstract: Recent research on remote sensing object detection has largely focused on improving the representation of oriented bounding boxes but has overlooked the unique prior knowledge presented in remote sensing scenarios. Such prior knowledge can be useful because tiny remote sensing objects may be mistakenly detected without referencing a sufficiently long-range context, which can vary for different objects. This paper considers these priors and proposes the lightweight Large Selective Kernel Network (LSKNet). LSKNet can dynamically adjust its large spatial receptive field to better model the ranging context of various objects in remote sensing scenarios. To our knowledge, large and selective kernel mechanisms have not been previously explored in remote sensing object detection. Without bells and whistles, our lightweight LSKNet sets new state-of-the-art scores on standard benchmarks, i.e., HRSC2016 (98.46% mAP), DOTA-v1.0 (81.85% mAP), and FAIR1M-v1.0 (47.87% mAP).

count=1
* Towards Geospatial Foundation Models via Continual Pretraining
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.pdf)]
    * Title: Towards Geospatial Foundation Models via Continual Pretraining
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Matías Mendieta, Boran Han, Xingjian Shi, Yi Zhu, Chen Chen
    * Abstract: Geospatial technologies are becoming increasingly essential in our world for a wide range of applications, including agriculture, urban planning, and disaster response. To help improve the applicability and performance of deep learning models on these geospatial tasks, various works have begun investigating foundation models for this domain. Researchers have explored two prominent approaches for introducing such models in geospatial applications, but both have drawbacks in terms of limited performance benefit or prohibitive training cost. Therefore, in this work, we propose a novel paradigm for building highly effective geospatial foundation models with minimal resource cost and carbon impact. We first construct a compact yet diverse dataset from multiple sources to promote feature diversity, which we term GeoPile. Then, we investigate the potential of continual pretraining from large-scale ImageNet-22k models and propose a multi-objective continual pretraining paradigm, which leverages the strong representations of ImageNet while simultaneously providing the freedom to learn valuable in-domain features. Our approach outperforms previous state-of-the-art geospatial pretraining methods in an extensive evaluation on seven downstream datasets covering various tasks such as change detection, classification, multi-label classification, semantic segmentation, and super-resolution. Code is available at https://github.com/mmendiet/GFM.

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
* XNet: Wavelet-Based Low and High Frequency Fusion Networks for Fully- and Semi-Supervised Semantic Segmentation of Biomedical Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_XNet_Wavelet-Based_Low_and_High_Frequency_Fusion_Networks_for_Fully-_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_XNet_Wavelet-Based_Low_and_High_Frequency_Fusion_Networks_for_Fully-_ICCV_2023_paper.pdf)]
    * Title: XNet: Wavelet-Based Low and High Frequency Fusion Networks for Fully- and Semi-Supervised Semantic Segmentation of Biomedical Images
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yanfeng Zhou, Jiaxing Huang, Chenlong Wang, Le Song, Ge Yang
    * Abstract: Fully- and semi-supervised semantic segmentation of biomedical images have been advanced with the development of deep neural networks (DNNs). So far, however, DNN models are usually designed to support one of these two learning schemes, unified models that support both fully- and semi-supervised segmentation remain limited. Furthermore, few fully-supervised models focus on the intrinsic low frequency (LF) and high frequency (HF) information of images to improve performance. Perturbations in consistency-based semi-supervised models are often artificially designed. They may introduce negative learning bias that are not beneficial for training. In this study, we propose a wavelet-based LF and HF fusion model XNet, which supports both fully- and semi-supervised semantic segmentation and outperforms state-of-the-art models in both fields. It emphasizes extracting LF and HF information for consistency training to alleviate the learning bias caused by artificial perturbations. Extensive experiments on two 2D and two 3D datasets demonstrate the effectiveness of our model. Code is available at https://github.com/Yanfeng-Zhou/XNet.

count=1
* The Change You Want to See (Now in 3D)
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/html/Sachdeva_The_Change_You_Want_to_See_Now_in_3D_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/papers/Sachdeva_The_Change_You_Want_to_See_Now_in_3D_ICCVW_2023_paper.pdf)]
    * Title: The Change You Want to See (Now in 3D)
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ragav Sachdeva, Andrew Zisserman
    * Abstract: The goal of this paper is to detect what has changed, if anything, between two "in the wild" images of the same 3D scene acquired from different camera positions and at different temporal instances. The open-set nature of this problem, occlusions/dis-occlusions due to the shift in viewpoint, and the lack of suitable training datasets, presents substantial challenges in devising a solution. To address this problem, we contribute a change detection model that is trained entirely on synthetic data and is class-agnostic, yet it is performant out-of-the-box on real world images without requiring fine-tuning. Our solution entails a "register and difference" approach that leverages self-supervised frozen embeddings and feature differences, which allows the model to generalise to a wide variety of scenes and domains. The model is able to operate directly on two RGB images, without requiring access to ground truth camera intrinsics, extrinsics, depth maps, point clouds, or additional before-after images. Finally, we collect and release a new evaluation dataset consisting of real-world image pairs with human-annotated differences and demonstrate the efficacy of our method. The code, datasets and pre-trained model can be found at: https://github.com/ragavsachdeva/CYWS-3D

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
* TKD: Temporal Knowledge Distillation for Active Perception
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Bajestani_TKD_Temporal_Knowledge_Distillation_for_Active_Perception_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Bajestani_TKD_Temporal_Knowledge_Distillation_for_Active_Perception_WACV_2020_paper.pdf)]
    * Title: TKD: Temporal Knowledge Distillation for Active Perception
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Mohammad Farhadi Bajestani,  Yezhou Yang
    * Abstract: Deep neural network-based methods have been proved to achieve outstanding performance on object detection and classification tasks. Despite the significant performance improvement using the deep structures, they still require prohibitive runtime to process images and maintain the highest possible performance for real-time applications. Observing the phenomenon that human visual system (HVS) relies heavily on the temporal dependencies among frames from the visual input to conduct recognition efficiently, we propose a novel framework dubbed as TKD: temporal knowledge distillation. This framework distills the temporal knowledge from a heavy neural network-based model over selected video frames (the perception of the moments) to a light-weight model. To enable the distillation, we put forward two novel procedures: 1) a Long-short Term Memory (LSTM)-based key frame selection method; and 2) a novel teacher-bounded loss design. To validate our approach, we conduct comprehensive empirical evaluations using different object detection methods over multiple datasets including Youtube Objects and Hollywood scene dataset. Our results show consistent improvement in accuracy-speed trade-offs for object detection over the frames of the dynamic scene, compared to other modern object recognition methods. It can maintain the desired accuracy with the throughput of around 220 images per second. Implementation: https://github.com/mfarhadi/TKD-Cloud.

count=1
* BSUV-Net: A Fully-Convolutional Neural Network for Background Subtraction of Unseen Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Tezcan_BSUV-Net_A_Fully-Convolutional_Neural_Network_for_Background_Subtraction_of_Unseen_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Tezcan_BSUV-Net_A_Fully-Convolutional_Neural_Network_for_Background_Subtraction_of_Unseen_WACV_2020_paper.pdf)]
    * Title: BSUV-Net: A Fully-Convolutional Neural Network for Background Subtraction of Unseen Videos
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Ozan Tezcan,  Prakash Ishwar,  Janusz Konrad
    * Abstract: Background subtraction is a basic task in computer vision and video processing often applied as a pre-processing step for object tracking, people recognition, etc. Recently, a number of successful background-subtraction algorithms have been proposed, however nearly all of the top-performing ones are supervised. Crucially, their success relies upon the availability of some annotated frames of the test video during training. Consequently, their performance on completely "unseen" videos is undocumented in the literature. In this work, we propose a new, supervised, background-subtraction algorithm for unseen videos (BSUV-Net) based on a fully-convolutional neural network. The input to our network consists of the current frame and two background frames captured at different time scales along with their semantic segmentation maps. In order to reduce the chance of overfitting, we also introduce a new data-augmentation technique which mitigates the impact of illumination difference between the background frames and the current frame. On the CDNet-2014 dataset, BSUV-Net outperforms state-of-the-art algorithms evaluated on unseen videos in terms of several metrics including F-measure, recall and precision.

count=1
* Do Adaptive Active Attacks Pose Greater Risk Than Static Attacks?
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Drenkow_Do_Adaptive_Active_Attacks_Pose_Greater_Risk_Than_Static_Attacks_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Drenkow_Do_Adaptive_Active_Attacks_Pose_Greater_Risk_Than_Static_Attacks_WACV_2023_paper.pdf)]
    * Title: Do Adaptive Active Attacks Pose Greater Risk Than Static Attacks?
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Nathan Drenkow, Max Lennon, I-Jeng Wang, Philippe Burlina
    * Abstract: In contrast to perturbation-based attacks, patch-based attacks are physically realizable, and are therefore increasingly studied. However, prior work neglects the possibility of adaptive attacks optimized for 3D pose. For the first time, to our knowledge, we consider the challenge of designing and evaluating attacks on image sequences using 3D optimization along entire 3D kinematic trajectories. In this context, we study a type of dynamic attack, referred to as "adaptive active attacks" (AAA), that takes into consideration the pose of the observer being targeted. To better address the threat and risk posed by AAA attacks, we develop several novel risk-based and trajectory-based metrics. These are designed to capture the risk of attack success for attacking earlier in the trajectory to derail autonomous driving systems as well as tradeoffs that may arise given the possibility of additional detection. We evaluate performance of white-box targeted attacks using a subset of ImageNet classes, and demonstrate, in aggregate, that AAA attacks can pose threats beyond static attacks in kinematic settings in situations of predominantly looming motion (i.,e., a prevalent use case in automated vehicular navigation). Results demonstrate that AAA attacks can exhibit targeted attack success exceeding 10% in aggregate, and for some specific classes, up to 15% over their static counterparts. However, taking into consideration the probability of detection by the defender shows a more nuanced risk pattern. These new insights are important for guiding future adversarial machine learning studies and suggest researchers should consider defense against novel threats posed by dynamic attacks for full trajectories and videos.

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
* The Change You Want To See
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Sachdeva_The_Change_You_Want_To_See_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Sachdeva_The_Change_You_Want_To_See_WACV_2023_paper.pdf)]
    * Title: The Change You Want To See
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Ragav Sachdeva, Andrew Zisserman
    * Abstract: We live in a dynamic world where things change all the time. Given two images of the same scene, being able to automatically detect the changes in them has practical applications in a variety of domains. In this paper, we tackle the change detection problem with the goal of detecting "object-level" changes in an image pair despite differences in their viewpoint and illumination. To this end, we make the following four contributions: (i) we propose a scalable methodology for obtaining a large-scale change detection training dataset by leveraging existing object segmentation benchmarks; (ii) we introduce a co-attention based novel architecture that is able to implicitly determine correspondences between an image pair and find changes in the form of bounding box predictions; (iii) we contribute four evaluation datasets that cover a variety of domains and transformations, including synthetic image changes, real surveillance images of a 3D scene, and synthetic 3D scenes with camera motion; (iv) we evaluate our model on these four datasets and demonstrate zero-shot and beyond training transformation generalization. The code, datasets and pre-trained model can be found at our project page: https://www.robots.ox.ac.uk/ vgg/research/cyws/

count=1
* Autoencoder-Based Background Reconstruction and Foreground Segmentation With Background Noise Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.pdf)]
    * Title: Autoencoder-Based Background Reconstruction and Foreground Segmentation With Background Noise Estimation
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Bruno Sauvalle, Arnaud de La Fortelle
    * Abstract: Even after decades of research, dynamic scene background reconstruction and foreground object segmentation are still considered as open problems due to various challenges such as illumination changes, camera movements, or background noise caused by air turbulence or moving trees. We propose in this paper to model the background of a frame sequence as a low dimensional manifold using an autoencoder and compare the reconstructed background provided by this autoencoder with the original image to compute the foreground/background segmentation masks. The main novelty of the proposed model is that the autoencoder is also trained to predict the background noise, which allows to compute for each frame a pixel-dependent threshold to perform the foreground segmentation. Although the proposed model does not use any temporal or motion information, it exceeds the state of the art for unsupervised background subtraction on the CDnet 2014 and LASIESTA datasets, with a significant improvement on videos where the camera is moving. It is also able to perform background reconstruction on some non-video image datasets.

count=1
* WATCH: Wide-Area Terrestrial Change Hypercube
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Greenwell_WATCH_Wide-Area_Terrestrial_Change_Hypercube_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Greenwell_WATCH_Wide-Area_Terrestrial_Change_Hypercube_WACV_2024_paper.pdf)]
    * Title: WATCH: Wide-Area Terrestrial Change Hypercube
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Connor Greenwell, Jon Crall, Matthew Purri, Kristin Dana, Nathan Jacobs, Armin Hadzic, Scott Workman, Matt Leotta
    * Abstract: Monitoring Earth activity using data collected from multiple satellite imaging platforms in a unified way is a significant challenge, especially with large variability in image resolution, spectral bands, and revisit rates. Further, the availability of sensor data varies across time as new platforms are launched. In this work, we introduce an adaptable framework and network architecture capable of predicting on subsets of the available platforms, bands, or temporal ranges it was trained on. Our system, called WATCH, is highly general and can be applied to a variety of geospatial tasks. In this work, we analyze the performance of WATCH using the recent IARPA SMART public dataset and metrics. We focus primarily on the problem of broad area search for heavy construction sites. Experiments validate the robustness of WATCH during inference to limited sensor availability, as well the the ability to alter inference-time spatial or temporal sampling. WATCH is open source and available for use on this or other remote sensing problems. Code and model weights are available at: https://gitlab.kitware.com/computer-vision/geowatch

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
* ICF-SRSR: Invertible Scale-Conditional Function for Self-Supervised Real-World Single Image Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Neshatavar_ICF-SRSR_Invertible_Scale-Conditional_Function_for_Self-Supervised_Real-World_Single_Image_Super-Resolution_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Neshatavar_ICF-SRSR_Invertible_Scale-Conditional_Function_for_Self-Supervised_Real-World_Single_Image_Super-Resolution_WACV_2024_paper.pdf)]
    * Title: ICF-SRSR: Invertible Scale-Conditional Function for Self-Supervised Real-World Single Image Super-Resolution
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Reyhaneh Neshatavar, Mohsen Yavartanoo, Sanghyun Son, Kyoung Mu Lee
    * Abstract: Single image super-resolution (SISR) is a challenging ill-posed problem that aims to up-sample a given low-resolution (LR) image to a high-resolution (HR) counterpart. Due to the difficulty in obtaining real LR-HR training pairs, recent approaches are trained on simulated LR images degraded by simplified down-sampling operators, e.g., bicubic. Such an approach can be problematic in practice due to the large gap between the synthesized and real-world LR images. To alleviate the issue, we propose a novel Invertible scale-Conditional Function (ICF), which can scale an input image and then restore the original input with different scale conditions. Using the proposed ICF, we construct a novel self-supervised SISR framework (ICF-SRSR) to handle the real-world SR task without using any paired/unpaired training data. Furthermore, our ICF-SRSR can generate realistic and feasible LR-HR pairs, which can make existing supervised SISR networks more robust. Extensive experiments demonstrate the effectiveness of our method in handling SISR in a fully self-supervised manner. Our ICF-SRSR demonstrates superior performance compared to the existing methods trained on synthetic paired images in real-world scenarios and exhibits comparable performance compared to state-of-the-art supervised/unsupervised methods on public benchmark datasets. The code is available from this link.

count=1
* Controlled Recognition Bounds for Visual Learning and Exploration
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/2a50e9c2d6b89b95bcb416d6857f8b45-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/2a50e9c2d6b89b95bcb416d6857f8b45-Paper.pdf)]
    * Title: Controlled Recognition Bounds for Visual Learning and Exploration
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Vasiliy Karasev, Alessandro Chiuso, Stefano Soatto
    * Abstract: We describe the tradeoff between the performance in a visual recognition problem and the control authority that the agent can exercise on the sensing process. We focus on the problem of “visual search” of an object in an otherwise known and static scene, propose a measure of control authority, and relate it to the expected risk and its proxy (conditional entropy of the posterior density). We show this analytically, as well as empirically by simulation using the simplest known model that captures the phenomenology of image formation, including scaling and occlusions. We show that a “passive” agent given a training set can provide no guarantees on performance beyond what is afforded by the priors, and that an “omnipotent” agent, capable of infinite control authority, can achieve arbitrarily good performance (asymptotically).

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
* Trimmed Density Ratio Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/ea204361fe7f024b130143eb3e189a18-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/ea204361fe7f024b130143eb3e189a18-Paper.pdf)]
    * Title: Trimmed Density Ratio Estimation
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Song Liu, Akiko Takeda, Taiji Suzuki, Kenji Fukumizu
    * Abstract: Density ratio estimation is a vital tool in both machine learning and statistical community. However, due to the unbounded nature of density ratio, the estimation proceudre can be vulnerable to corrupted data points, which often pushes the estimated ratio toward infinity. In this paper, we present a robust estimator which automatically identifies and trims outliers. The proposed estimator has a convex formulation, and the global optimum can be obtained via subgradient descent. We analyze the parameter estimation error of this estimator under high-dimensional settings. Experiments are conducted to verify the effectiveness of the estimator.

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
* Structure Learning with Side Information: Sample Complexity
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/e025b6279c1b88d3ec0eca6fcb6e6280-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/e025b6279c1b88d3ec0eca6fcb6e6280-Paper.pdf)]
    * Title: Structure Learning with Side Information: Sample Complexity
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Saurabh Sihag, Ali Tajer
    * Abstract: Graphical models encode the stochastic dependencies among random variables (RVs). The vertices represent the RVs, and the edges signify the conditional dependencies among the RVs. Structure learning is the process of inferring the edges by observing realizations of the RVs, and it has applications in a wide range of technological, social, and biological networks. Learning the structure of graphs when the vertices are treated in isolation from inferential information known about them is well-investigated. In a wide range of domains, however, often there exist additional inferred knowledge about the structure, which can serve as valuable side information. For instance, the gene networks that represent different subtypes of the same cancer share similar edges across all subtypes and also have exclusive edges corresponding to each subtype, rendering partially similar graphical models for gene expression in different cancer subtypes. Hence, an inferential decision regarding a gene network can serve as side information for inferring other related gene networks. When such side information is leveraged judiciously, it can translate to significant improvement in structure learning. Leveraging such side information can be abstracted as inferring structures of distinct graphical models that are {\sl partially} similar. This paper focuses on Ising graphical models, and considers the problem of simultaneously learning the structures of two {\sl partially} similar graphs, where any inference about the structure of one graph offers side information for the other graph. The bounded edge subclass of Ising models is considered, and necessary conditions (information-theoretic ), as well as sufficient conditions (algorithmic) for the sample complexity for achieving a bounded probability of error, are established. Furthermore, specific regimes are identified in which the necessary and sufficient conditions coincide, rendering the optimal sample complexity.

count=1
* Spot the Difference: Detection of Topological Changes via Geometric Alignment
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/7867d6557b82ed3b5d61e6591a2a2fd3-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/7867d6557b82ed3b5d61e6591a2a2fd3-Paper.pdf)]
    * Title: Spot the Difference: Detection of Topological Changes via Geometric Alignment
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Per Steffen Czolbe, Aasa Feragen, Oswin Krause
    * Abstract: Geometric alignment appears in a variety of applications, ranging from domain adaptation, optimal transport, and normalizing flows in machine learning; optical flow and learned augmentation in computer vision and deformable registration within biomedical imaging. A recurring challenge is the alignment of domains whose topology is not the same; a problem that is routinely ignored, potentially introducing bias in downstream analysis. As a first step towards solving such alignment problems, we propose an unsupervised algorithm for the detection of changes in image topology. The model is based on a conditional variational auto-encoder and detects topological changes between two images during the registration step. We account for both topological changes in the image under spatial variation and unexpected transformations. Our approach is validated on two tasks and datasets: detection of topological changes in microscopy images of cells, and unsupervised anomaly detection brain imaging.

count=1
* CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf)]
    * Title: CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Anthony Fuller, Koreen Millard, James Green
    * Abstract: A vital and rapidly growing application, remote sensing offers vast yet sparsely labeled, spatially aligned multimodal data; this makes self-supervised learning algorithms invaluable. We present CROMA: a framework that combines contrastive and reconstruction self-supervised objectives to learn rich unimodal and multimodal representations. Our method separately encodes masked-out multispectral optical and synthetic aperture radar samples—aligned in space and time—and performs cross-modal contrastive learning. Another encoder fuses these sensors, producing joint multimodal encodings that are used to predict the masked patches via a lightweight decoder. We show that these objectives are complementary when leveraged on spatially aligned multimodal data. We also introduce X- and 2D-ALiBi, which spatially biases our cross- and self-attention matrices. These strategies improve representations and allow our models to effectively extrapolate to images up to $17.6\times$ larger at test-time. CROMA outperforms the current SoTA multispectral model, evaluated on: four classification benchmarks—finetuning (avg.$\uparrow$ 1.8%), linear (avg.$\uparrow$ 2.4%) and nonlinear (avg.$\uparrow$ 1.4%) probing, $k$NN classification (avg.$\uparrow$ 3.5%), and $K$-means clustering (avg.$\uparrow$ 8.4%); and three segmentation benchmarks (avg.$\uparrow$ 6.4%). CROMA’s rich, optionally multimodal representations can be widely leveraged across remote sensing applications.

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
* SSL4EO-L: Datasets and Foundation Models for Landsat Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/bbf7ee04e2aefec136ecf60e346c2e61-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/bbf7ee04e2aefec136ecf60e346c2e61-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: SSL4EO-L: Datasets and Foundation Models for Landsat Imagery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Adam Stewart, Nils Lehmann, Isaac Corley, Yi Wang, Yi-Chia Chang, Nassim Ait Ait Ali Braham, Shradha Sehgal, Caleb Robinson, Arindam Banerjee
    * Abstract: The Landsat program is the longest-running Earth observation program in history, with 50+ years of data acquisition by 8 satellites. The multispectral imagery captured by sensors onboard these satellites is critical for a wide range of scientific fields. Despite the increasing popularity of deep learning and remote sensing, the majority of researchers still use decision trees and random forests for Landsat image analysis due to the prevalence of small labeled datasets and lack of foundation models. In this paper, we introduce SSL4EO-L, the first ever dataset designed for Self-Supervised Learning for Earth Observation for the Landsat family of satellites (including 3 sensors and 2 product levels) and the largest Landsat dataset in history (5M image patches). Additionally, we modernize and re-release the L7 Irish and L8 Biome cloud detection datasets, and introduce the first ML benchmark datasets for Landsats 4–5 TM and Landsat 7 ETM+ SR. Finally, we pre-train the first foundation models for Landsat imagery using SSL4EO-L and evaluate their performance on multiple semantic segmentation tasks. All datasets and model weights are available via the TorchGeo library, making reproducibility and experimentation easy, and enabling scientific advancements in the burgeoning field of remote sensing for a multitude of downstream applications.

