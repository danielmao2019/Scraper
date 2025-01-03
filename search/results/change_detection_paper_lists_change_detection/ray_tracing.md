count=2
* Dynamic Probabilistic Volumetric Models
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Ulusoy_Dynamic_Probabilistic_Volumetric_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Ulusoy_Dynamic_Probabilistic_Volumetric_2013_ICCV_paper.pdf)]
    * Title: Dynamic Probabilistic Volumetric Models
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Ali Osman Ulusoy, Octavian Biris, Joseph L. Mundy
    * Abstract: This paper presents a probabilistic volumetric framework for image based modeling of general dynamic 3-d scenes. The framework is targeted towards high quality modeling of complex scenes evolving over thousands of frames. Extensive storage and computational resources are required in processing large scale space-time (4-d) data. Existing methods typically store separate 3-d models at each time step and do not address such limitations. A novel 4-d representation is proposed that adaptively subdivides in space and time to explain the appearance of 3-d dynamic surfaces. This representation is shown to achieve compression of 4-d data and provide efficient spatio-temporal processing. The advances of the proposed framework is demonstrated on standard datasets using free-viewpoint video and 3-d tracking applications.

count=2
* Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/html/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/papers/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.pdf)]
    * Title: Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Csaba Beleznai, Peter Gemeiner, Christian Zinner
    * Abstract: Reliable and timely detection of abandoned items in public places still represents an unsolved problem for automated visual surveillance. Typical surveilled scenarios are associated with high visual ambiguity such as shadows, occlusions, illumination changes and substantial clutter consisting of a mixture of dynamic and stationary objects. Motivated by these challenges we propose a reliable left item detection approach based on the combination of intensity and depth data from a passive stereo setup. The employed in-house developed stereo system consists of low-cost sensors and it is capable to perform detection in environments of up to 10m x 10m in size. The proposed algorithm is tested on a set of indoor sequences and compared to manually annotated ground truth data. Obtained results show that many failure modes of intensity-based approaches are absent and even small-sized objects such as a handbag can be reliably detected when left behind in a scene. The presented results display a very promising approach, which can robustly detect left luggage in dynamic environments at a close to real-time computational speed.

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
* Semantic Multi-View Stereo: Jointly Estimating Objects and Voxels
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Ulusoy_Semantic_Multi-View_Stereo_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ulusoy_Semantic_Multi-View_Stereo_CVPR_2017_paper.pdf)]
    * Title: Semantic Multi-View Stereo: Jointly Estimating Objects and Voxels
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Ali Osman Ulusoy, Michael J. Black, Andreas Geiger
    * Abstract: Dense 3D reconstruction from RGB images is a highly ill-posed problem due to occlusions, textureless or reflective surfaces, as well as other challenges. We propose object-level shape priors to address these ambiguities. Towards this goal, we formulate a probabilistic model that integrates multi-view image evidence with 3D shape information from multiple objects. Inference in this model yields a dense 3D reconstruction of the scene as well as the existence and precise 3D pose of the objects in it. Our approach is able to recover fine details not captured in the input shapes while defaulting to the input models in occluded regions where image evidence is weak. Due to its probabilistic nature, the approach is able to cope with the approximate geometry of the 3D models as well as input shapes that are not present in the scene. We evaluate the approach quantitatively on several challenging indoor and outdoor datasets.

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
* Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.pdf)]
    * Title: Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Olaf Wysocki, Yan Xia, Magdalena Wysocki, Eleonora Grilli, Ludwig Hoegner, Daniel Cremers, Uwe Stilla
    * Abstract: Reconstructing semantic 3D building models at the level of detail (LoD) 3 is a long-standing challenge. Unlike mesh-based models, they require watertight geometry and object-wise semantics at the facade level. The principal challenge of such demanding semantic 3D reconstruction is reliable facade-level semantic segmentation of 3D input data. We present a novel method, called Scan2LoD3, that accurately reconstructs semantic LoD3 building models by improving facade-level semantic 3D segmentation. To this end, we leverage laser physics and 3D building model priors to probabilistically identify model conflicts. These probabilistic physical conflicts propose locations of model openings: Their final semantics and shapes are inferred in a Bayesian network fusing multimodal probabilistic maps of conflicts, 3D point clouds, and 2D images. To fulfill demanding LoD3 requirements, we use the estimated shapes to cut openings in 3D building priors and fit semantic 3D objects from a library of facade objects. Extensive experiments on the TUM city campus datasets demonstrate the superior performance of the proposed Scan2LoD3 over the state-of-the-art methods in facade-level detection, semantic segmentation, and LoD3 building model reconstruction. We believe our method can foster the development of probability-driven semantic 3D reconstruction at LoD3 since not only the high-definition reconstruction but also reconstruction confidence becomes pivotal for various applications such as autonomous driving and urban simulations.

count=1
* Calibrated Vehicle Paint Signatures for Simulating Hyperspectral Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Mulhollan_Calibrated_Vehicle_Paint_Signatures_for_Simulating_Hyperspectral_Imagery_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w6/Mulhollan_Calibrated_Vehicle_Paint_Signatures_for_Simulating_Hyperspectral_Imagery_CVPRW_2020_paper.pdf)]
    * Title: Calibrated Vehicle Paint Signatures for Simulating Hyperspectral Imagery
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zachary Mulhollan, Aneesh Rangnekar, Timothy Bauch, Matthew J. Hoffman, Anthony Vodacek
    * Abstract: We investigate a procedure for rapidly adding calibrated vehicle visible-near infrared (VNIR) paint signatures to an existing hyperspectral simulator - The Digital Imaging and Remote Sensing Image Generation (DIRSIG) model - to create more diversity in simulated urban scenes. The DIRSIG model can produce synthetic hyperspectral imagery with user-specified geometry, atmospheric conditions, and ground target spectra. To render an object pixel's spectral signature, DIRSIG uses a large database of reflectance curves for the corresponding object material and a bidirectional reflectance model to introduce s due to orientation and surface structure. However, this database contains only a few spectral curves for vehicle paints and generates new paint signatures by combining these curves internally. In this paper we demonstrate a method to rapidly generate multiple paint spectra, flying a drone carrying a pushbroom hyperspectral camera to image a university parking lot. We then process the images to convert them from the digital count space to spectral reflectance without the need of calibration panels in the scene, and port the paint signatures into DIRSIG for successful integration into the newly rendered sets of synthetic VNIR hyperspectral scenes.

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
* Learning a Multi-View Stereo Machine
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/9c838d2e45b2ad1094d42f4ef36764f6-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/9c838d2e45b2ad1094d42f4ef36764f6-Paper.pdf)]
    * Title: Learning a Multi-View Stereo Machine
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Abhishek Kar, Christian Häne, Jitendra Malik
    * Abstract: We present a learnt system for multi-view stereopsis. In contrast to recent learning based methods for 3D reconstruction, we leverage the underlying 3D geometry of the problem through feature projection and unprojection along viewing rays. By formulating these operations in a differentiable manner, we are able to learn the system end-to-end for the task of metric 3D reconstruction. End-to-end learning allows us to jointly reason about shape priors while conforming to geometric constraints, enabling reconstruction from much fewer images (even a single image) than required by classical approaches as well as completion of unseen surfaces. We thoroughly evaluate our approach on the ShapeNet dataset and demonstrate the benefits over classical approaches and recent learning based methods.

