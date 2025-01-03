count=26
* HashPoint: Accelerated Point Searching and Sampling for Neural Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_HashPoint_Accelerated_Point_Searching_and_Sampling_for_Neural_Rendering_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_HashPoint_Accelerated_Point_Searching_and_Sampling_for_Neural_Rendering_CVPR_2024_paper.pdf)]
    * Title: HashPoint: Accelerated Point Searching and Sampling for Neural Rendering
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jiahao Ma, Miaomiao Liu, David Ahmedt-Aristizabal, Chuong Nguyen
    * Abstract: In this paper we address the problem of efficient point searching and sampling for volume neural rendering. Within this realm two typical approaches are employed: rasterization and ray tracing. The rasterization-based methods enable real-time rendering at the cost of increased memory and lower fidelity. In contrast the ray-tracing-based methods yield superior quality but demand longer rendering time. We solve this problem by our HashPoint method combining these two strategies leveraging rasterization for efficient point searching and sampling and ray marching for rendering. Our method optimizes point searching by rasterizing points within the camera's view organizing them in a hash table and facilitating rapid searches. Notably we accelerate the rendering process by adaptive sampling on the primary surface encountered by the ray. Our approach yields substantial speed-up for a range of state-of-the-art ray-tracing-based methods maintaining equivalent or superior accuracy across synthetic and real test datasets. The code will be available at https://jiahao-ma.github.io/hashpoint/

count=22
* The Differentiable Lens: Compound Lens Search Over Glass Surfaces and Materials for Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Cote_The_Differentiable_Lens_Compound_Lens_Search_Over_Glass_Surfaces_and_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Cote_The_Differentiable_Lens_Compound_Lens_Search_Over_Glass_Surfaces_and_CVPR_2023_paper.pdf)]
    * Title: The Differentiable Lens: Compound Lens Search Over Glass Surfaces and Materials for Object Detection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Geoffroi Côté, Fahim Mannan, Simon Thibault, Jean-François Lalonde, Felix Heide
    * Abstract: Most camera lens systems are designed in isolation, separately from downstream computer vision methods. Recently, joint optimization approaches that design lenses alongside other components of the image acquisition and processing pipeline--notably, downstream neural networks--have achieved improved imaging quality or better performance on vision tasks. However, these existing methods optimize only a subset of lens parameters and cannot optimize glass materials given their categorical nature. In this work, we develop a differentiable spherical lens simulation model that accurately captures geometrical aberrations. We propose an optimization strategy to address the challenges of lens design--notorious for non-convex loss function landscapes and many manufacturing constraints--that are exacerbated in joint optimization tasks. Specifically, we introduce quantized continuous glass variables to facilitate the optimization and selection of glass materials in an end-to-end design context, and couple this with carefully designed constraints to support manufacturability. In automotive object detection, we report improved detection performance over existing designs even when simplifying designs to two- or three-element lenses, despite significantly degrading the image quality.

count=22
* Towards High Fidelity Monocular Face Reconstruction With Rich Reflectance Using Self-Supervised Learning and Ray Tracing
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Dib_Towards_High_Fidelity_Monocular_Face_Reconstruction_With_Rich_Reflectance_Using_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Dib_Towards_High_Fidelity_Monocular_Face_Reconstruction_With_Rich_Reflectance_Using_ICCV_2021_paper.pdf)]
    * Title: Towards High Fidelity Monocular Face Reconstruction With Rich Reflectance Using Self-Supervised Learning and Ray Tracing
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Abdallah Dib, Cédric Thébault, Junghyun Ahn, Philippe-Henri Gosselin, Christian Theobalt, Louis Chevallier
    * Abstract: Robust face reconstruction from monocular image in general lighting conditions is challenging. Methods combining deep neural network encoders with differentiable rendering have opened up the path for very fast monocular reconstruction of geometry, lighting and reflectance. They can also be trained in self-supervised manner for increased robustness and better generalization. However, their differentiable rasterization based image formation models, as well as underlying scene parameterization, limit them to Lambertian face reflectance and to poor shape details. More recently, ray tracing was introduced for monocular face reconstruction within a classic optimization-based framework and enables state-of-the art results. However optimization-based approaches are inherently slow and lack robustness. In this paper, we build our work on the aforementioned approaches and propose a new method that greatly improves reconstruction quality and robustness in general scenes. We achieve this by combining a CNN encoder with a differentiable ray tracer, which enables us to base the reconstruction on much more advanced personalized diffuse and specular albedos, a more sophisticated illumination model and a plausible representation of self-shadows. This enables to take a big leap forward in reconstruction quality of shape, appearance and lighting even in scenes with difficult illumination. With consistent face attributes reconstruction, our method leads to practical applications such as relighting and self-shadows removal. Compared to state-of-the-art methods, our results show improved accuracy and validity of the approach.

count=18
* Road Scene Understanding by Occupancy Grid Learning from Sparse Radar Clusters using Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/CVRSUAD/Sless_Road_Scene_Understanding_by_Occupancy_Grid_Learning_from_Sparse_Radar_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Sless_Road_Scene_Understanding_by_Occupancy_Grid_Learning_from_Sparse_Radar_ICCVW_2019_paper.pdf)]
    * Title: Road Scene Understanding by Occupancy Grid Learning from Sparse Radar Clusters using Semantic Segmentation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Liat Sless, Bat El Shlomo, Gilad Cohen, Shaul Oron
    * Abstract: Occupancy grid mapping is an important component in road scene understanding for autonomous driving. It encapsulates information of the drivable area, road obstacles and enables safe autonomous driving. Radars are an emerging sensor in autonomous vehicle vision, becoming more widely used due to their long range sensing, low cost, and robustness to severe weather conditions. Despite recent advances in deep learning technology, occupancy grid mapping from radar data is still mostly done using classical filtering approaches. In this work, we propose learning the inverse sensor model used for occupancy grid mapping from clustered radar data. This is done in a data driven approach that leverages computer vision techniques. This task is very challenging due to data sparsity and noise characteristics of the radar sensor. The problem is formulated as a semantic segmentation task and we show how it can be learned using lidar data for generating ground truth. We show both qualitatively and quantitatively that our learned occupancy net outperforms classic methods by a large margin using the recently released NuScenes real-world driving data.

count=17
* IntrinsicAvatar: Physically Based Inverse Rendering of Dynamic Humans from Monocular Videos via Explicit Ray Tracing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_IntrinsicAvatar_Physically_Based_Inverse_Rendering_of_Dynamic_Humans_from_Monocular_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_IntrinsicAvatar_Physically_Based_Inverse_Rendering_of_Dynamic_Humans_from_Monocular_CVPR_2024_paper.pdf)]
    * Title: IntrinsicAvatar: Physically Based Inverse Rendering of Dynamic Humans from Monocular Videos via Explicit Ray Tracing
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Shaofei Wang, Bozidar Antic, Andreas Geiger, Siyu Tang
    * Abstract: We present IntrinsicAvatar a novel approach to recovering the intrinsic properties of clothed human avatars including geometry albedo material and environment lighting from only monocular videos. Recent advancements in human-based neural rendering have enabled high-quality geometry and appearance reconstruction of clothed humans from just monocular videos. However these methods bake intrinsic properties such as albedo material and environment lighting into a single entangled neural representation. On the other hand only a handful of works tackle the problem of estimating geometry and disentangled appearance properties of clothed humans from monocular videos. They usually achieve limited quality and disentanglement due to approximations of secondary shading effects via learned MLPs. In this work we propose to model secondary shading effects explicitly via Monte-Carlo ray tracing. We model the rendering process of clothed humans as a volumetric scattering process and combine ray tracing with body articulation. Our approach can recover high-quality geometry albedo material and lighting properties of clothed humans from a single monocular video without requiring supervised pre-training using ground truth materials. Furthermore since we explicitly model the volumetric scattering process and ray tracing our model naturally generalizes to novel poses enabling animation of the reconstructed avatar in novel lighting conditions.

count=17
* Efficient and Differentiable Shadow Computation for Inverse Problems
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Lyu_Efficient_and_Differentiable_Shadow_Computation_for_Inverse_Problems_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Lyu_Efficient_and_Differentiable_Shadow_Computation_for_Inverse_Problems_ICCV_2021_paper.pdf)]
    * Title: Efficient and Differentiable Shadow Computation for Inverse Problems
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Linjie Lyu, Marc Habermann, Lingjie Liu, Mallikarjun B R, Ayush Tewari, Christian Theobalt
    * Abstract: Differentiable rendering has received increasing interest in the solution of image-based inverse problems. It can benefit traditional optimization-based solutions to inverse problems, but also allows for self-supervision of learning-based approaches for which training data with ground truth annotation is hard to obtain. However, existing differentiable renderers either do not correctly model complex visibility responsible for shadows in the images, or are too slow for being used to train deep architectures over thousands of iterations. To this end, we propose an accurate yet efficient approach for differentiable visibility and soft shadow computation. Our approach is based on the spherical harmonics approximation of the scene illumination and visibility, where the occluding surface is approximated with spheres. This allows for significantly more efficient visibility computation compared to methods based on path tracing without sacrificing quality of generated images. As our formulation is differentiable, it can be used to solve various image-based inverse problems such as texture, lighting, geometry recovery from images using analysis-by-synthesis optimization.

count=16
* AR-NeRF: Unsupervised Learning of Depth and Defocus Effects From Natural Images With Aperture Rendering Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Kaneko_AR-NeRF_Unsupervised_Learning_of_Depth_and_Defocus_Effects_From_Natural_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Kaneko_AR-NeRF_Unsupervised_Learning_of_Depth_and_Defocus_Effects_From_Natural_CVPR_2022_paper.pdf)]
    * Title: AR-NeRF: Unsupervised Learning of Depth and Defocus Effects From Natural Images With Aperture Rendering Neural Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Takuhiro Kaneko
    * Abstract: Fully unsupervised 3D representation learning has gained attention owing to its advantages in data collection. A successful approach involves a viewpoint-aware approach that learns an image distribution based on generative models (e.g., generative adversarial networks (GANs)) while generating various view images based on 3D-aware models (e.g., neural radiance fields (NeRFs)). However, they require images with various views for training, and consequently, their application to datasets with few or limited viewpoints remains a challenge. As a complementary approach, an aperture rendering GAN (AR-GAN) that employs a defocus cue was proposed. However, an AR-GAN is a CNN-based model and represents a defocus independently from a viewpoint change despite its high correlation, which is one of the reasons for its performance. As an alternative to an AR-GAN, we propose an aperture rendering NeRF (AR-NeRF), which can utilize viewpoint and defocus cues in a unified manner by representing both factors in a common ray-tracing framework. Moreover, to learn defocus-aware and defocus-independent representations in a disentangled manner, we propose aperture randomized training, for which we learn to generate images while randomizing the aperture size and latent codes independently. During our experiments, we applied AR-NeRF to various natural image datasets, including flower, bird, and face images, the results of which demonstrate the utility of AR-NeRF for unsupervised learning of the depth and defocus effects.

count=16
* Pointersect: Neural Rendering With Cloud-Ray Intersection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Chang_Pointersect_Neural_Rendering_With_Cloud-Ray_Intersection_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Chang_Pointersect_Neural_Rendering_With_Cloud-Ray_Intersection_CVPR_2023_paper.pdf)]
    * Title: Pointersect: Neural Rendering With Cloud-Ray Intersection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jen-Hao Rick Chang, Wei-Yu Chen, Anurag Ranjan, Kwang Moo Yi, Oncel Tuzel
    * Abstract: We propose a novel method that renders point clouds as if they are surfaces. The proposed method is differentiable and requires no scene-specific optimization. This unique capability enables, out-of-the-box, surface normal estimation, rendering room-scale point clouds, inverse rendering, and ray tracing with global illumination. Unlike existing work that focuses on converting point clouds to other representations--e.g., surfaces or implicit functions--our key idea is to directly infer the intersection of a light ray with the underlying surface represented by the given point cloud. Specifically, we train a set transformer that, given a small number of local neighbor points along a light ray, provides the intersection point, the surface normal, and the material blending weights, which are used to render the outcome of this light ray. Localizing the problem into small neighborhoods enables us to train a model with only 48 meshes and apply it to unseen point clouds. Our model achieves higher estimation accuracy than state-of-the-art surface reconstruction and point-cloud rendering methods on three test sets. When applied to room-scale point clouds, without any scene-specific optimization, the model achieves competitive quality with the state-of-the-art novel-view rendering methods. Moreover, we demonstrate ability to render and manipulate Lidar-scanned point clouds such as lighting control and object insertion.

count=15
* Reflection and Diffraction-Aware Sound Source Localization
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/Sight_and_Sound/Inkyu_An_Reflection_and_Diffraction-Aware_Sound_Source_Localization_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Sight and Sound/Inkyu_An_Reflection_and_Diffraction-Aware_Sound_Source_Localization_CVPRW_2019_paper.pdf)]
    * Title: Reflection and Diffraction-Aware Sound Source Localization
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Inkyu An, Jung-Woo Choi, Dinesh Manocha, Sung-Eui Yoon
    * Abstract: 

count=13
* Inverse Rendering of Glossy Objects via the Neural Plenoptic Function and Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Inverse_Rendering_of_Glossy_Objects_via_the_Neural_Plenoptic_Function_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Inverse_Rendering_of_Glossy_Objects_via_the_Neural_Plenoptic_Function_CVPR_2024_paper.pdf)]
    * Title: Inverse Rendering of Glossy Objects via the Neural Plenoptic Function and Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Haoyuan Wang, Wenbo Hu, Lei Zhu, Rynson W.H. Lau
    * Abstract: Inverse rendering aims at recovering both geometry and materials of objects. It provides a more compatible reconstruction for conventional rendering engines compared with the neural radiance fields (NeRFs). On the other hand existing NeRF-based inverse rendering methods cannot handle glossy objects with local light interactions well as they typically oversimplify the illumination as a 2D environmental map which assumes infinite lights only. Observing the superiority of NeRFs in recovering radiance fields we propose a novel 5D Neural Plenoptic Function (NeP) based on NeRFs and ray tracing such that more accurate lighting-object interactions can be formulated via the rendering equation. We also design a material-aware cone sampling strategy to efficiently integrate lights inside the BRDF lobes with the help of pre-filtered radiance fields. Our method has two stages: the geometry of the target object and the pre-filtered environmental radiance fields are reconstructed in the first stage and materials of the target object are estimated in the second stage with the proposed NeP and material-aware cone sampling strategy. Extensive experiments on the proposed real-world and synthetic datasets demonstrate that our method can reconstruct high-fidelity geometry/materials of challenging glossy objects with complex lighting interactions from nearby objects. Project webpage: https://whyy.site/paper/nep

count=13
* Single View Refractive Index Tomography with Neural Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.pdf)]
    * Title: Single View Refractive Index Tomography with Neural Fields
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Brandon Zhao, Aviad Levis, Liam Connor, Pratul P. Srinivasan, Katherine L. Bouman
    * Abstract: Refractive Index Tomography is the inverse problem of reconstructing the continuously-varying 3D refractive index in a scene using 2D projected image measurements. Although a purely refractive field is not directly visible it bends light rays as they travel through space thus providing a signal for reconstruction. The effects of such fields appear in many scientific computer vision settings ranging from refraction due to transparent cells in microscopy to the lensing of distant galaxies caused by dark matter in astrophysics. Reconstructing these fields is particularly difficult due to the complex nonlinear effects of the refractive field on observed images. Furthermore while standard 3D reconstruction and tomography settings typically have access to observations of the scene from many viewpoints many refractive index tomography problem settings only have access to images observed from a single viewpoint. We introduce a method that leverages prior knowledge of light sources scattered throughout the refractive medium to help disambiguate the single-view refractive index tomography problem. We differentiably trace curved rays through a neural field representation of the refractive field and optimize its parameters to best reproduce the observed image. We demonstrate the efficacy of our approach by reconstructing simulated refractive fields analyze the effects of light source distribution on the recovered field and test our method on a simulated dark matter mapping problem where we successfully recover the 3D refractive field caused by a realistic dark matter distribution.

count=12
* Dynamic Multiview Refinement of 3D Hand Datasets Using Differentiable Ray Tracing
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/AMFG/html/Karvounas_Dynamic_Multiview_Refinement_of_3D_Hand_Datasets_Using_Differentiable_Ray_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/AMFG/papers/Karvounas_Dynamic_Multiview_Refinement_of_3D_Hand_Datasets_Using_Differentiable_Ray_ICCVW_2023_paper.pdf)]
    * Title: Dynamic Multiview Refinement of 3D Hand Datasets Using Differentiable Ray Tracing
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Giorgos Karvounas, Nikolaos Kyriazis, Iason Oikonomidis, Antonis Argyros
    * Abstract: With the increase of AI applications in the field of 3D estimation of hand state, the quality of the datasets used for training the relevant models is of utmost importance. Especially in the case of datasets consisting of real-world images, the quality of annotations, i.e., how accurately the provided ground truth reflects the true state of the scene, can greatly affect the performance of downstream applications. In this work, we propose a methodology with significant impact on improving ubiquitous 3D hand geometry datasets that contain real images with imperfect annotations. Our approach leverages multi-view imagery, temporal consistency, and a disentangled representation of hand shape, texture, and environment lighting. This allows to refine the hand geometry of existing datasets and also paves the way for texture extraction. Extensive experiments on synthetic and real-world data show that our method outperforms the current state of the art, resulting in more accurate and visually pleasing reconstructions of hand gestures.

count=11
* Face Relighting With Geometrically Consistent Shadows
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Hou_Face_Relighting_With_Geometrically_Consistent_Shadows_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Hou_Face_Relighting_With_Geometrically_Consistent_Shadows_CVPR_2022_paper.pdf)]
    * Title: Face Relighting With Geometrically Consistent Shadows
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Andrew Hou, Michel Sarkis, Ning Bi, Yiying Tong, Xiaoming Liu
    * Abstract: Most face relighting methods are able to handle diffuse shadows, but struggle to handle hard shadows, such as those cast by the nose. Methods that propose techniques for handling hard shadows often do not produce geometrically consistent shadows since they do not directly leverage the estimated face geometry while synthesizing them. We propose a novel differentiable algorithm for synthesizing hard shadows based on ray tracing, which we incorporate into training our face relighting model. Our proposed algorithm directly utilizes the estimated face geometry to synthesize geometrically consistent hard shadows. We demonstrate through quantitative and qualitative experiments on Multi-PIE and FFHQ that our method produces more geometrically consistent shadows than previous face relighting methods while also achieving state-of-the-art face relighting performance under directional lighting. In addition, we demonstrate that our differentiable hard shadow modeling improves the quality of the estimated face geometry over diffuse shading models.

count=11
* Seeing Through the Glass: Neural 3D Reconstruction of Object Inside a Transparent Container
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tong_Seeing_Through_the_Glass_Neural_3D_Reconstruction_of_Object_Inside_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Tong_Seeing_Through_the_Glass_Neural_3D_Reconstruction_of_Object_Inside_CVPR_2023_paper.pdf)]
    * Title: Seeing Through the Glass: Neural 3D Reconstruction of Object Inside a Transparent Container
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jinguang Tong, Sundaram Muthu, Fahira Afzal Maken, Chuong Nguyen, Hongdong Li
    * Abstract: In this paper, we define a new problem of recovering the 3D geometry of an object confined in a transparent enclosure. We also propose a novel method for solving this challenging problem. Transparent enclosures pose challenges of multiple light reflections and refractions at the interface between different propagation media e.g. air or glass. These multiple reflections and refractions cause serious image distortions which invalidate the single viewpoint assumption. Hence the 3D geometry of such objects cannot be reliably reconstructed using existing methods, such as traditional structure from motion or modern neural reconstruction methods. We solve this problem by explicitly modeling the scene as two distinct sub-spaces, inside and outside the transparent enclosure. We use an existing neural reconstruction method (NeuS) that implicitly represents the geometry and appearance of the inner subspace. In order to account for complex light interactions, we develop a hybrid rendering strategy that combines volume rendering with ray tracing. We then recover the underlying geometry and appearance of the model by minimizing the difference between the real and rendered images. We evaluate our method on both synthetic and real data. Experiment results show that our method outperforms the state-of-the-art (SOTA) methods. Codes and data will be available at https://github.com/hirotong/ReNeuS

count=10
* Multiview Detection with Cardboard Human Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Ma_Multiview_Detection_with_Cardboard_Human_Modeling_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Ma_Multiview_Detection_with_Cardboard_Human_Modeling_ACCV_2024_paper.pdf)]
    * Title: Multiview Detection with Cardboard Human Modeling
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Jiahao Ma, Zicheng Duan, Liang Zheng, Chuong Nguyen
    * Abstract: Multiview detection uses multiple calibrated cameras with overlapping fields of views to locate occluded pedestrians. In this field, existing methods typically adopt a "human modeling - aggregation" strategy. To find robust pedestrian representations, some intuitively incorporate 2D perception results from each frame, while others use entire frame features projected to the ground plane. However, the former does not consider the human appearance and leads to many ambiguities, and the latter suffers from projection errors due to the lack of accurate height of the human torso and head. In this paper, we propose a new pedestrian representation scheme based on human point clouds modeling. Specifically, using ray tracing for holistic human depth estimation, we model pedestrians as upright, thin cardboard point clouds on the ground. Then, we aggregate the point clouds of the pedestrian cardboard across multiple views for a final decision. Compared with existing representations, the proposed method explicitly leverages human appearance and reduces projection errors significantly by relatively accurate height estimation. On four standard evaluation benchmarks, our method achieves very competitive results. Our code and data will be open-sourced.

count=10
* Extreme Low-Light Environment-Driven Image Denoising Over Permanently Shadowed Lunar Regions With a Physical Noise Model
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Moseley_Extreme_Low-Light_Environment-Driven_Image_Denoising_Over_Permanently_Shadowed_Lunar_Regions_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Moseley_Extreme_Low-Light_Environment-Driven_Image_Denoising_Over_Permanently_Shadowed_Lunar_Regions_CVPR_2021_paper.pdf)]
    * Title: Extreme Low-Light Environment-Driven Image Denoising Over Permanently Shadowed Lunar Regions With a Physical Noise Model
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Ben Moseley, Valentin Bickel, Ignacio G. Lopez-Francos, Loveneesh Rana
    * Abstract: Recently, learning-based approaches have achieved impressive results in the field of low-light image denoising. Some state of the art approaches employ a rich physical model to generate realistic training data. However, the performance of these approaches ultimately depends on the realism of the physical model, and many works only concentrate on everyday photography. In this work we present a denoising approach for extremely low-light images of permanently shadowed regions (PSRs) on the lunar surface, taken by the Narrow Angle Camera on board the Lunar Reconnaissance Orbiter satellite. Our approach extends existing learning-based approaches by combining a physical noise model of the camera with real noise samples and training image scene selection based on 3D ray tracing to generate realistic training data. We also condition our denoising model on the camera's environmental metadata at the time of image capture (such as the camera's temperature and age), showing that this improves performance. Our quantitative and qualitative results show that our method strongly outperforms the existing calibration routine for the camera and other baselines. Our results could significantly impact lunar science and exploration, for example by aiding the identification of surface water-ice and reducing uncertainty in rover and human traverse planning into PSRs.

count=10
* RL-based Stateful Neural Adaptive Sampling and Denoising for Real-Time Path Tracing
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/d1422213c9f2bdd5178b77d166fba86a-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/d1422213c9f2bdd5178b77d166fba86a-Paper-Conference.pdf)]
    * Title: RL-based Stateful Neural Adaptive Sampling and Denoising for Real-Time Path Tracing
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Antoine Scardigli, Lukas Cavigelli, Lorenz K. Müller
    * Abstract: Monte-Carlo path tracing is a powerful technique for realistic image synthesis but suffers from high levels of noise at low sample counts, limiting its use in real-time applications. To address this, we propose a framework with end-to-end training of a sampling importance network, a latent space encoder network, and a denoiser network. Our approach uses reinforcement learning to optimize the sampling importance network, thus avoiding explicit numerically approximated gradients. Our method does not aggregate the sampled values per pixel by averaging but keeps all sampled values which are then fed into the latent space encoder. The encoder replaces handcrafted spatiotemporal heuristics by learned representations in a latent space. Finally, a neural denoiser is trained to refine the output image. Our approach increases visual quality on several challenging datasets and reduces rendering times for equal quality by a factor of 1.6x compared to the previous state-of-the-art, making it a promising solution for real-time applications.

count=9
* Iso-Points: Optimizing Neural Implicit Surfaces With Hybrid Representations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Yifan_Iso-Points_Optimizing_Neural_Implicit_Surfaces_With_Hybrid_Representations_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yifan_Iso-Points_Optimizing_Neural_Implicit_Surfaces_With_Hybrid_Representations_CVPR_2021_paper.pdf)]
    * Title: Iso-Points: Optimizing Neural Implicit Surfaces With Hybrid Representations
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Wang Yifan, Shihao Wu, Cengiz Oztireli, Olga Sorkine-Hornung
    * Abstract: Neural implicit functions have emerged as a powerful representation for surfaces in 3D. Such a function can encode a high quality surface with intricate details into the parameters of a deep neural network. However, optimizing for the parameters for accurate and robust reconstructions remains a challenge especially when the input data is noisy or incomplete. In this work, we develop a hybrid neural surface representation that allows us to impose geometry-aware sampling and regularization, which significantly improves the fidelity of reconstructions. We propose to use iso-points as an explicit representation for a neural implicit function. These points are computed and updated on-the-fly during training to capture important geometric features and impose geometric constraints on the optimization. We demonstrate that our method can be adopted to improve state-of-the-art techniques for reconstructing neural implicit surfaces from multi-view images or point clouds. Quantitative and qualitative evaluations show that, compared with existing sampling and optimization methods, our approach allows faster convergence, better generalization, and accurate recovery of details and topology.

count=9
* In-Situ Joint Light and Medium Estimation for Underwater Color Restoration
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/OceanVision/html/Nakath_In-Situ_Joint_Light_and_Medium_Estimation_for_Underwater_Color_Restoration_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/OceanVision/papers/Nakath_In-Situ_Joint_Light_and_Medium_Estimation_for_Underwater_Color_Restoration_ICCVW_2021_paper.pdf)]
    * Title: In-Situ Joint Light and Medium Estimation for Underwater Color Restoration
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: David Nakath, Mengkun She, Yifan Song, Kevin Köser
    * Abstract: The majority of Earth's surface is situated in the deep sea and thus remains deprived of natural light. Such adverse underwater environments have to be explored with powerful camera-light systems. In order to restore the colors in images taken by such systems, we need to jointly estimate physically-meaningful optical parameters of the light as well as the water column. We thus propose an integrated in-situ estimation approach and a complementary surface texture recovery strategy, which also removes shadows as a by-product. As we operate in a scattering medium under inhomogeneous lighting conditions, the volumetric effects are difficult to capture in closed-form solutions. Hence, we leverage the latest progress in Monte Carlo-based differentiable ray tracing that becomes tractable through recent GPU RTX-hardware acceleration. Evaluations on synthetic data and in a water tank show that we can estimate physically meaningful parameters, which enables color restoration. The approaches could also be employed to other camera-light systems (AUV, robot, car, endoscope) operating either in the dark, in fog - or - underwater.

count=8
* Neural Fields Meet Explicit Geometric Representations for Inverse Rendering of Urban Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Neural_Fields_Meet_Explicit_Geometric_Representations_for_Inverse_Rendering_of_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Neural_Fields_Meet_Explicit_Geometric_Representations_for_Inverse_Rendering_of_CVPR_2023_paper.pdf)]
    * Title: Neural Fields Meet Explicit Geometric Representations for Inverse Rendering of Urban Scenes
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zian Wang, Tianchang Shen, Jun Gao, Shengyu Huang, Jacob Munkberg, Jon Hasselgren, Zan Gojcic, Wenzheng Chen, Sanja Fidler
    * Abstract: Reconstruction and intrinsic decomposition of scenes from captured imagery would enable many applications such as relighting and virtual object insertion. Recent NeRF based methods achieve impressive fidelity of 3D reconstruction, but bake the lighting and shadows into the radiance field, while mesh-based methods that facilitate intrinsic decomposition through differentiable rendering have not yet scaled to the complexity and scale of outdoor scenes. We present a novel inverse rendering framework for large urban scenes capable of jointly reconstructing the scene geometry, spatially-varying materials, and HDR lighting from a set of posed RGB images with optional depth. Specifically, we use a neural field to account for the primary rays, and use an explicit mesh (reconstructed from the underlying neural field) for modeling secondary rays that produce higher-order lighting effects such as cast shadows. By faithfully disentangling complex geometry and materials from lighting effects, our method enables photorealistic relighting with specular and shadow effects on several outdoor datasets. Moreover, it supports physics-based scene manipulations such as virtual object insertion with ray-traced shadow casting.

count=8
* RadSimReal: Bridging the Gap Between Synthetic and Real Data in Radar Object Detection With Simulation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Bialer_RadSimReal_Bridging_the_Gap_Between_Synthetic_and_Real_Data_in_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Bialer_RadSimReal_Bridging_the_Gap_Between_Synthetic_and_Real_Data_in_CVPR_2024_paper.pdf)]
    * Title: RadSimReal: Bridging the Gap Between Synthetic and Real Data in Radar Object Detection With Simulation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Oded Bialer, Yuval Haitman
    * Abstract: Object detection in radar imagery with neural networks shows great potential for improving autonomous driving. However obtaining annotated datasets from real radar images crucial for training these networks is challenging especially in scenarios with long-range detection and adverse weather and lighting conditions where radar performance excels. To address this challenge we present RadSimReal an innovative physical radar simulation capable of generating synthetic radar images with accompanying annotations for various radar types and environmental conditions all without the need for real data collection. Remarkably our findings demonstrate that training object detection models on RadSimReal data and subsequently evaluating them on real-world data produce performance levels comparable to models trained and tested on real data from the same dataset and even achieves better performance when testing across different real datasets. RadSimReal offers advantages over other physical radar simulations that it does not necessitate knowledge of the radar design details which are often not disclosed by radar suppliers and has faster run-time. This innovative tool has the potential to advance the development of computer vision algorithms for radar-based autonomous driving applications.

count=7
* Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Khurana_Point_Cloud_Forecasting_as_a_Proxy_for_4D_Occupancy_Forecasting_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Khurana_Point_Cloud_Forecasting_as_a_Proxy_for_4D_Occupancy_Forecasting_CVPR_2023_paper.pdf)]
    * Title: Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Tarasha Khurana, Peiyun Hu, David Held, Deva Ramanan
    * Abstract: Predicting how the world can evolve in the future is crucial for motion planning in autonomous systems. Classical methods are limited because they rely on costly human annotations in the form of semantic class labels, bounding boxes, and tracks or HD maps of cities to plan their motion -- and thus are difficult to scale to large unlabeled datasets. One promising self-supervised task is 3D point cloud forecasting from unannotated LiDAR sequences. We show that this task requires algorithms to implicitly capture (1) sensor extrinsics (i.e., the egomotion of the autonomous vehicle), (2) sensor intrinsics (i.e., the sampling pattern specific to the particular LiDAR sensor), and (3) the shape and motion of other objects in the scene. But autonomous systems should make predictions about the world and not their sensors! To this end, we factor out (1) and (2) by recasting the task as one of spacetime (4D) occupancy forecasting. But because it is expensive to obtain ground-truth 4D occupancy, we "render" point cloud data from 4D occupancy predictions given sensor extrinsics and intrinsics, allowing one to train and test occupancy algorithms with unannotated LiDAR sequences. This also allows one to evaluate and compare point cloud forecasting algorithms across diverse datasets, sensors, and vehicles.

count=7
* Humans As Light Bulbs: 3D Human Reconstruction From Thermal Reflection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Humans_As_Light_Bulbs_3D_Human_Reconstruction_From_Thermal_Reflection_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Humans_As_Light_Bulbs_3D_Human_Reconstruction_From_Thermal_Reflection_CVPR_2023_paper.pdf)]
    * Title: Humans As Light Bulbs: 3D Human Reconstruction From Thermal Reflection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Ruoshi Liu, Carl Vondrick
    * Abstract: The relatively hot temperature of the human body causes people to turn into long-wave infrared light sources. Since this emitted light has a larger wavelength than visible light, many surfaces in typical scenes act as infrared mirrors with strong specular reflections. We exploit the thermal reflections of a person onto objects in order to locate their position and reconstruct their pose, even if they are not visible to a normal camera. We propose an analysis-by-synthesis framework that jointly models the objects, people, and their thermal reflections, which allows us to combine generative models with differentiable rendering of reflections. Quantitative and qualitative experiments show our approach works in highly challenging cases, such as with curved mirrors or when the person is completely unseen by a normal camera.

count=7
* Eclipse: Disambiguating Illumination and Materials using Unintended Shadows
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Verbin_Eclipse_Disambiguating_Illumination_and_Materials_using_Unintended_Shadows_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Verbin_Eclipse_Disambiguating_Illumination_and_Materials_using_Unintended_Shadows_CVPR_2024_paper.pdf)]
    * Title: Eclipse: Disambiguating Illumination and Materials using Unintended Shadows
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Dor Verbin, Ben Mildenhall, Peter Hedman, Jonathan T. Barron, Todd Zickler, Pratul P. Srinivasan
    * Abstract: Decomposing an object's appearance into representations of its materials and the surrounding illumination is difficult even when the object's 3D shape is known beforehand. This problem is especially challenging for diffuse objects: it is ill-conditioned because diffuse materials severely blur incoming light and it is ill-posed because diffuse materials under high-frequency lighting can be indistinguishable from shiny materials under low-frequency lighting. We show that it is possible to recover precise materials and illumination---even from diffuse objects---by exploiting unintended shadows like the ones cast onto an object by the photographer who moves around it. These shadows are a nuisance in most previous inverse rendering pipelines but here we exploit them as signals that improve conditioning and help resolve material-lighting ambiguities. We present a method based on differentiable Monte Carlo ray tracing that uses images of an object to jointly recover its spatially-varying materials the surrounding illumination environment and the shapes of the unseen light occluders who inadvertently cast shadows upon it.

count=7
* Physics Based Camera Privacy: Lens and Network Co-Design to the Rescue
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/html/Dufraisse_Physics_Based_Camera_Privacy_Lens_and_Network_Co-Design_to_the_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/papers/Dufraisse_Physics_Based_Camera_Privacy_Lens_and_Network_Co-Design_to_the_CVPRW_2024_paper.pdf)]
    * Title: Physics Based Camera Privacy: Lens and Network Co-Design to the Rescue
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Marius Dufraisse, Marcela Carvalho, Pauline Trouvé-Peloux, Frédéric Champagnat
    * Abstract: The wide presence of cameras using automatic image processing has a positive impact on smart cities and autonomous driving objectives but also poses privacy threats. In this context there has been an increasing interest in regulating the acquisition and use of public and private data. In this paper we propose a method to co-design a lens and an image processing pipeline to perform semantic segmentation of urban images while ensuring privacy by introducing optical aberrations directly on the lens. This particular design relies on a hardware level of protection that prevents an attacker from accessing sensitive information from the original images of a device. Related works rely on specific privacy threats requiring a large database for training. In contrast we propose to seek robustness by preventing deblurring during training in a self-supervised way thus without requiring additional annotations. Moreover we validate our approach by simulating attacks with deblurring and license plate detection and recognition to show that our model can fool these tasks with success while keeping a high score on the utility task.

count=6
* Acquiring Axially-Symmetric Transparent Objects Using Single-View Transmission Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Kim_Acquiring_Axially-Symmetric_Transparent_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Kim_Acquiring_Axially-Symmetric_Transparent_CVPR_2017_paper.pdf)]
    * Title: Acquiring Axially-Symmetric Transparent Objects Using Single-View Transmission Imaging
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Jaewon Kim, Ilya Reshetouski, Abhijeet Ghosh
    * Abstract: We propose a novel, practical solution for high quality reconstruction of axially-symmetric transparent objects. While a special case, such transparent objects are ubiquitous in the real world. Common examples of these are glasses, goblets, tumblers, carafes, etc., that can have very unique and visually appealing forms making their reconstruction interesting for vision and graphics applications. Our acquisition setup involves imaging such objects from a single viewpoint while illuminating them from directly behind with a few patterns emitted by an LCD panel. Our reconstruction step is then based on optimization of the object's geometry and its refractive index to minimize the difference between observed and simulated transmission/refraction of rays passing through the object. We exploit the object's axial symmetry as a strong shape prior which allows us to achieve robust reconstruction from a single viewpoint using a simple, commodity acquisition setup. We demonstrate high quality reconstruction of several common rotationally symmetric as well as more complex n-fold symmetric transparent objects with our approach.

count=6
* Neural Geometric Level of Detail: Real-Time Rendering With Implicit 3D Shapes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Takikawa_Neural_Geometric_Level_of_Detail_Real-Time_Rendering_With_Implicit_3D_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Takikawa_Neural_Geometric_Level_of_Detail_Real-Time_Rendering_With_Implicit_3D_CVPR_2021_paper.pdf)]
    * Title: Neural Geometric Level of Detail: Real-Time Rendering With Implicit 3D Shapes
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten Kreis, Charles Loop, Derek Nowrouzezahrai, Alec Jacobson, Morgan McGuire, Sanja Fidler
    * Abstract: Neural signed distance functions (SDFs) are emerging as an effective representation for 3D shapes. State-of-the-art methods typically encode the SDF with a large, fixed-size neural network to approximate complex shapes with implicit surfaces. Rendering with these large networks is, however, computationally expensive since it requires many forward passes through the network for every pixel, making these representations impractical for real-time graphics. We introduce an efficient neural representation that, for the first time, enables real-time rendering of high-fidelity neural SDFs, while achieving state-of-the-art geometry reconstruction quality. We represent implicit surfaces using an octree-based feature volume which adaptively fits shapes with multiple discrete levels of detail (LODs), and enables continuous LOD with SDF interpolation. We further develop an efficient algorithm to directly render our novelneural SDF representation in real-time by querying only the necessary LODswith sparse octree traversal. We show that our representation is 2-3 orders of magnitude more efficient in terms of rendering speed compared to previous works. Furthermore, it produces state-of-the-art reconstruction quality for complex shapes under both 3D geometric and 2D image-space metrics.

count=6
* Gravitationally Lensed Black Hole Emission Tomography
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Levis_Gravitationally_Lensed_Black_Hole_Emission_Tomography_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Levis_Gravitationally_Lensed_Black_Hole_Emission_Tomography_CVPR_2022_paper.pdf)]
    * Title: Gravitationally Lensed Black Hole Emission Tomography
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Aviad Levis, Pratul P. Srinivasan, Andrew A. Chael, Ren Ng, Katherine L. Bouman
    * Abstract: Measurements from the Event Horizon Telescope enabled the visualization of light emission around a black hole for the first time. So far, these measurements have been used to recover a 2D image under the assumption that the emission field is static over the period of acquisition. In this work, we propose BH-NeRF, a novel tomography approach that leverages gravitational lensing to recover the continuous 3D emission field near a black hole. Compared to other 3D reconstruction or tomography settings, this task poses two significant challenges: first, rays near black holes follow curved paths dictated by general relativity, and second, we only observe measurements from a single viewpoint. Our method captures the unknown emission field using a continuous volumetric function parameterized by a coordinate-based neural network, and uses knowledge of Keplerian orbital dynamics to establish correspondence between 3D points over time. Together, these enable BH-NeRF to recover accurate 3D emission fields, even in challenging situations with sparse measurements and uncertain orbital dynamics. This work takes the first steps in showing how future measurements from the Event Horizon Telescope could be used to recover evolving 3D emission around the supermassive black hole in our Galactic center.

count=6
* ActiveZero: Mixed Domain Learning for Active Stereovision With Zero Annotation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_ActiveZero_Mixed_Domain_Learning_for_Active_Stereovision_With_Zero_Annotation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_ActiveZero_Mixed_Domain_Learning_for_Active_Stereovision_With_Zero_Annotation_CVPR_2022_paper.pdf)]
    * Title: ActiveZero: Mixed Domain Learning for Active Stereovision With Zero Annotation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Isabella Liu, Edward Yang, Jianyu Tao, Rui Chen, Xiaoshuai Zhang, Qing Ran, Zhu Liu, Hao Su
    * Abstract: Traditional depth sensors generate accurate real world depth estimates that surpass even the most advanced learning approaches trained only on simulation domains. Since ground truth depth is readily available in the simulation domain but quite difficult to obtain in the real domain, we propose a method that leverages the best of both worlds. In this paper we present a new framework, ActiveZero, which is a mixed domain learning solution for active stereovision systems that requires no real world depth annotation. First, we demonstrate the transferability of our method to out-of-distribution real data by using a mixed domain learning strategy. In the simulation domain, we use a combination of supervised disparity loss and self-supervised losses on a shape primitives dataset. By contrast, in the real domain, we only use self-supervised losses on a dataset that is out-of-distribution from either training simulation data or test real data. Second, our method introduces a novel self-supervised loss called temporal IR reprojection to increase the robustness and accuracy of our reprojections in hard-to-perceive regions. Finally, we show how the method can be trained end-to-end and that each module is important for attaining the end result. Extensive qualitative and quantitative evaluations on real data demonstrate state of the art results that can even beat a commercial depth sensor.

count=6
* Physics-Based Differentiable Depth Sensor Simulation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Planche_Physics-Based_Differentiable_Depth_Sensor_Simulation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Planche_Physics-Based_Differentiable_Depth_Sensor_Simulation_ICCV_2021_paper.pdf)]
    * Title: Physics-Based Differentiable Depth Sensor Simulation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Benjamin Planche, Rajat Vikram Singh
    * Abstract: Gradient-based algorithms are crucial to modern computer-vision and graphics applications, enabling learning-based optimization and inverse problems. For example, photorealistic differentiable rendering pipelines for color images have been proven highly valuable to applications aiming to map 2D and 3D domains. However, to the best of our knowledge, no effort has been made so far towards extending these gradient-based methods to the generation of depth (2.5D) images, as simulating structured-light depth sensors implies solving complex light transport and stereo-matching problems. In this paper, we introduce a novel end-to-end differentiable simulation pipeline for the generation of realistic 2.5D scans, built on physics-based 3D rendering and custom block-matching algorithms. Each module can be differentiated w.r.t sensor and scene parameters; e.g., to automatically tune the simulation for new devices over some provided scans or to leverage the pipeline as a 3D-to-2.5D transformer within larger computer-vision applications. Applied to the training of deep-learning methods for various depth-based recognition tasks (classification, pose estimation, semantic segmentation), our simulation greatly improves the performance of the resulting models on real scans, thereby demonstrating the fidelity and value of its synthetic depth data compared to previous static simulations and learning-based domain adaptation schemes.

count=6
* Shape, Light, and Material Decomposition from Images using Monte Carlo Rendering and Denoising
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/8fcb27984bf16ca03cad643244ec470d-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/8fcb27984bf16ca03cad643244ec470d-Paper-Conference.pdf)]
    * Title: Shape, Light, and Material Decomposition from Images using Monte Carlo Rendering and Denoising
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Jon Hasselgren, Nikolai Hofmann, Jacob Munkberg
    * Abstract: Recent advances in differentiable rendering have enabled high-quality reconstruction of 3D scenes from multi-view images. Most methods rely on simple rendering algorithms: pre-filtered direct lighting or learned representations of irradiance. We show that a more realistic shading model, incorporating ray tracing and Monte Carlo integration, substantially improves decomposition into shape, materials & lighting. Unfortunately, Monte Carlo integration provides estimates with significant noise, even at large sample counts, which makes gradient-based inverse rendering very challenging. To address this, we incorporate multiple importance sampling and denoising in a novel inverse rendering pipeline. This improves convergence and enables gradient-based optimization at low sample counts. We present an efficient method to jointly reconstruct geometry (explicit triangle meshes), materials, and lighting, which substantially improves material and light separation compared to previous work. We argue that denoising can become an integral part of high quality inverse rendering pipelines.

count=5
* Depth Image Enhancement Using Local Tangent Plane Approximations
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Matsuo_Depth_Image_Enhancement_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Matsuo_Depth_Image_Enhancement_2015_CVPR_paper.pdf)]
    * Title: Depth Image Enhancement Using Local Tangent Plane Approximations
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Kiyoshi Matsuo, Yoshimitsu Aoki
    * Abstract: This paper describes a depth image enhancement method for consumer RGB-D cameras. Most existing methods use the pixel-coordinates of the aligned color image. Because the image plane generally has no relationship to the measured surfaces, the global coordinate system is not suitable to handle their local geometries. To improve enhancement accuracy, we use local tangent planes as local coordinates for the measured surfaces. Our method is composed of two steps, a calculation of the local tangents and surface reconstruction. To accurately estimate the local tangents, we propose a color heuristic calculation and an orientation correction using their positional relationships. Additionally, we propose a surface reconstruction method by ray-tracing to local tangents. In our method, accurate depth image enhancement is achieved by using the local geometries approximated by the local tangents. We demonstrate the effectiveness of our method using synthetic and real sensor data. Our method has a high completion rate and achieves the lowest errors in noisy cases when compared with existing techniques.

count=5
* Through Fog High-Resolution Imaging Using Millimeter Wave Radar
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Guan_Through_Fog_High-Resolution_Imaging_Using_Millimeter_Wave_Radar_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guan_Through_Fog_High-Resolution_Imaging_Using_Millimeter_Wave_Radar_CVPR_2020_paper.pdf)]
    * Title: Through Fog High-Resolution Imaging Using Millimeter Wave Radar
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Junfeng Guan,  Sohrab Madani,  Suraj Jog,  Saurabh Gupta,  Haitham Hassanieh
    * Abstract: This paper demonstrates high-resolution imaging using millimeter Wave (mmWave) radars that can function even in dense fog. We leverage the fact that mmWave signals have favorable propagation characteristics in low visibility conditions, unlike optical sensors like cameras and LiDARs which cannot penetrate through dense fog. Millimeter-wave radars, however, suffer from very low resolution, specularity, and noise artifacts. We introduce HawkEye, a system that leverages a cGAN architecture to recover high-frequency shapes from raw low-resolution mmWave heat-maps. We propose a novel design that addresses challenges specific to the structure and nature of the radar signals involved. We also develop a data synthesizer to aid with large-scale dataset generation for training. We implement our system on a custom-built mmWave radar platform and demonstrate performance improvement over both standard mmWave radars and other competitive baselines.

count=5
* DIST: Rendering Deep Implicit Signed Distance Function With Differentiable Sphere Tracing
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_DIST_Rendering_Deep_Implicit_Signed_Distance_Function_With_Differentiable_Sphere_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_DIST_Rendering_Deep_Implicit_Signed_Distance_Function_With_Differentiable_Sphere_CVPR_2020_paper.pdf)]
    * Title: DIST: Rendering Deep Implicit Signed Distance Function With Differentiable Sphere Tracing
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Shaohui Liu,  Yinda Zhang,  Songyou Peng,  Boxin Shi,  Marc Pollefeys,  Zhaopeng Cui
    * Abstract: We propose a differentiable sphere tracing algorithm to bridge the gap between inverse graphics methods and the recently proposed deep learning based implicit signed distance function. Due to the nature of the implicit function, the rendering process requires tremendous function queries, which is particularly problematic when the function is represented as a neural network. We optimize both the forward and backward pass of our rendering layer to make it run efficiently with affordable memory consumption on a commodity graphics card. Our rendering method is fully differentiable such that losses can be directly computed on the rendered 2D observations, and the gradients can be propagated backward to optimize the 3D geometry. We show that our rendering method can effectively reconstruct accurate 3D shapes from various inputs, such as sparse depth and multi-view images, through inverse optimization. With the geometry based reasoning, our 3D shape prediction methods show excellent generalization capability and robustness against various noises.

count=5
* Physically-Guided Disentangled Implicit Rendering for 3D Face Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Physically-Guided_Disentangled_Implicit_Rendering_for_3D_Face_Modeling_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Physically-Guided_Disentangled_Implicit_Rendering_for_3D_Face_Modeling_CVPR_2022_paper.pdf)]
    * Title: Physically-Guided Disentangled Implicit Rendering for 3D Face Modeling
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Zhenyu Zhang, Yanhao Ge, Ying Tai, Weijian Cao, Renwang Chen, Kunlin Liu, Hao Tang, Xiaoming Huang, Chengjie Wang, Zhifeng Xie, Dongjin Huang
    * Abstract: This paper presents a novel Physically-guided Disentangled Implicit Rendering (PhyDIR) framework for high-fidelity 3D face modeling. The motivation comes from two observations: widely-used graphics renderers yield excessive approximations against photo-realistic imaging, while neural rendering methods are highly entangled to perceive 3D-aware operations. Hence, we learn to disentangle the implicit rendering via explicit physical guidance, meanwhile guarantee the properties of (1) 3D-aware comprehension and (2) high-reality imaging. For the former one, PhyDIR explicitly adopts 3D shading and rasterizing modules to control the renderer, which disentangles the lighting, facial shape and view point from neural reasoning. Specifically, PhyDIR proposes a novel multi-image shading strategy to compensate the monocular limitation, so that the lighting variations are accessible to the neural renderer. For the latter one, PhyDIR learns the face-collection implicit texture to avoid ill-posed intrinsic factorization, then leverages a series of consistency losses to constrain the robustness. With the disentangled method, we make 3D face modeling benefit from both kinds of rendering strategies. Extensive experiments on benchmarks show that PhyDIR obtains superior performance than state-of-the-art explicit/implicit methods, on both geometry/texture modeling.

count=5
* SeSDF: Self-Evolved Signed Distance Field for Implicit 3D Clothed Human Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_SeSDF_Self-Evolved_Signed_Distance_Field_for_Implicit_3D_Clothed_Human_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_SeSDF_Self-Evolved_Signed_Distance_Field_for_Implicit_3D_Clothed_Human_CVPR_2023_paper.pdf)]
    * Title: SeSDF: Self-Evolved Signed Distance Field for Implicit 3D Clothed Human Reconstruction
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yukang Cao, Kai Han, Kwan-Yee K. Wong
    * Abstract: We address the problem of clothed human reconstruction from a single image or uncalibrated multi-view images. Existing methods struggle with reconstructing detailed geometry of a clothed human and often require a calibrated setting for multi-view reconstruction. We propose a flexible framework which, by leveraging the parametric SMPL-X model, can take an arbitrary number of input images to reconstruct a clothed human model under an uncalibrated setting. At the core of our framework is our novel self-evolved signed distance field (SeSDF) module which allows the framework to learn to deform the signed distance field (SDF) derived from the fitted SMPL-X model, such that detailed geometry reflecting the actual clothed human can be encoded for better reconstruction. Besides, we propose a simple method for self-calibration of multi-view images via the fitted SMPL-X parameters. This lifts the requirement of tedious manual calibration and largely increases the flexibility of our method. Further, we introduce an effective occlusion-aware feature fusion strategy to account for the most useful features to reconstruct the human model. We thoroughly evaluate our framework on public benchmarks, demonstrating significant superiority over the state-of-the-arts both qualitatively and quantitatively.

count=5
* Low Latency Point Cloud Rendering with Learned Splatting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Hu_Low_Latency_Point_Cloud_Rendering_with_Learned_Splatting_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Hu_Low_Latency_Point_Cloud_Rendering_with_Learned_Splatting_CVPRW_2024_paper.pdf)]
    * Title: Low Latency Point Cloud Rendering with Learned Splatting
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yueyu Hu, Ran Gong, Qi Sun, Yao Wang
    * Abstract: Point cloud is a critical 3D representation with many emerging applications. Because of the point sparsity and irregularity high-quality rendering of point clouds is challenging and often requires complex computations to recover the continuous surface representation. On the other hand to avoid visual discomfort the motion-to-photon latency has to be very short under 10 ms. Existing rendering solutions lack in either quality or speed. To tackle these challenges we present a framework that unlocks interactive free-viewing and high-fidelity point cloud rendering. We pre-train a neural network to estimate 3D elliptical Gaussians from arbitrary point clouds and use differentiable surface splatting to render smooth texture and surface normal for arbitrary views. Our approach does not require per-scene optimization and enable real-time rendering of dynamic point cloud. Experimental results demonstrate the proposed solution enjoys superior visual quality and speed as well as generalizability to different scene content and robustness to compression artifacts.

count=5
* FastNeRF
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Garbin_FastNeRF_High-Fidelity_Neural_Rendering_at_200FPS_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Garbin_FastNeRF_High-Fidelity_Neural_Rendering_at_200FPS_ICCV_2021_paper.pdf)]
    * Title: FastNeRF: High-Fidelity Neural Rendering at 200FPS
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, Julien Valentin
    * Abstract: Recent work on Neural Radiance Fields (NeRF) showed how neural networks can be used to encode complex 3D environments that can be rendered photorealistically from novel viewpoints. Rendering these images is very computationally demanding and recent improvements are still a long way from enabling interactive rates, even on high-end hardware. Motivated by scenarios on mobile and mixed reality devices, we propose FastNeRF, the first NeRF-based system capable of rendering high fidelity photorealistic images at 200Hz on a high-end consumer GPU. The core of our method is a graphics-inspired factorization that allows for (i) compactly caching a deep radiance map at each position in space, (ii) efficiently querying that map using ray directions to estimate the pixel values in the rendered image. Extensive experiments show that the proposed method is 3000 times faster than the original NeRF algorithm and at least an order of magnitude faster than existing work on accelerating NeRF, while maintaining visual quality and extensibility.

count=5
* Neural-PBIR Reconstruction of Shape, Material, and Illumination
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Sun_Neural-PBIR_Reconstruction_of_Shape_Material_and_Illumination_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Neural-PBIR_Reconstruction_of_Shape_Material_and_Illumination_ICCV_2023_paper.pdf)]
    * Title: Neural-PBIR Reconstruction of Shape, Material, and Illumination
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Cheng Sun, Guangyan Cai, Zhengqin Li, Kai Yan, Cheng Zhang, Carl Marshall, Jia-Bin Huang, Shuang Zhao, Zhao Dong
    * Abstract: Reconstructing the shape and spatially varying surface appearances of a physical-world object as well as its surrounding illumination based on 2D images (e.g., photographs) of the object has been a long-standing problem in computer vision and graphics. In this paper, we introduce an accurate and highly efficient object reconstruction pipeline combining neural based object reconstruction and physics-based inverse rendering (PBIR). Our pipeline firstly leverages a neural SDF based shape reconstruction to produce high-quality but potentially imperfect object shape. Then, we introduce a neural material and lighting distillation stage to achieve high-quality predictions for material and illumination. In the last stage, initialized by the neural predictions, we perform PBIR to refine the initial results and obtain the final high-quality reconstruction of object shape, material, and illumination. Experimental results demonstrate our pipeline significantly outperforms existing methods quality-wise and performance-wise. Code: https://neural-pbir.github.io/

count=5
* NeRFrac: Neural Radiance Fields through Refractive Surface
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhan_NeRFrac_Neural_Radiance_Fields_through_Refractive_Surface_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhan_NeRFrac_Neural_Radiance_Fields_through_Refractive_Surface_ICCV_2023_paper.pdf)]
    * Title: NeRFrac: Neural Radiance Fields through Refractive Surface
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yifan Zhan, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng
    * Abstract: Neural Radiance Fields (NeRF) is a popular neural expression for novel view synthesis. By querying spatial points and view directions, a multilayer perceptron (MLP) can be trained to output the volume density and radiance at each point, which lets us render novel views of the scene. The original NeRF and its recent variants, however, target opaque scenes dominated by diffuse reflection surfaces and cannot handle complex refractive surfaces well. We introduce NeRFrac to realize neural novel view synthesis of scenes captured through refractive surfaces, typically water surfaces. For each queried ray, an MLP-based Refractive Field is trained to estimate the distance from the ray origin to the refractive surface. A refracted ray at each intersection point is then computed by Snell's Law, given the input ray and the approximated local normal. Points of the scene are sampled along the refracted ray and are sent to a Radiance Field for further radiance estimation. We show that from a sparse set of images, our model achieves accurate novel view synthesis of the scene underneath the refractive surface and simultaneously reconstructs the refractive surface. We evaluate the effectiveness of our method with synthetic and real scenes seen through water surfaces. Experimental results demonstrate the accuracy of NeRFrac for modeling scenes seen through wavy refractive surfaces.

count=5
* RenderNet: A deep convolutional network for differentiable rendering from 3D shapes
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2018/hash/68d3743587f71fbaa5062152985aff40-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2018/file/68d3743587f71fbaa5062152985aff40-Paper.pdf)]
    * Title: RenderNet: A deep convolutional network for differentiable rendering from 3D shapes
    * Publisher: NeurIPS
    * Publication Date: `2018`
    * Authors: Thu H. Nguyen-Phuoc, Chuan Li, Stephen Balaban, Yongliang Yang
    * Abstract: Traditional computer graphics rendering pipelines are designed for procedurally generating 2D images from 3D shapes with high performance. The nondifferentiability due to discrete operations (such as visibility computation) makes it hard to explicitly correlate rendering parameters and the resulting image, posing a significant challenge for inverse rendering tasks. Recent work on differentiable rendering achieves differentiability either by designing surrogate gradients for non-differentiable operations or via an approximate but differentiable renderer. These methods, however, are still limited when it comes to handling occlusion, and restricted to particular rendering effects. We present RenderNet, a differentiable rendering convolutional network with a novel projection unit that can render 2D images from 3D shapes. Spatial occlusion and shading calculation are automatically encoded in the network. Our experiments show that RenderNet can successfully learn to implement different shaders, and can be used in inverse rendering tasks to estimate shape, pose, lighting and texture from a single image.

count=4
* SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_SDFDiff_Differentiable_Rendering_of_Signed_Distance_Fields_for_3D_Shape_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_SDFDiff_Differentiable_Rendering_of_Signed_Distance_Fields_for_3D_Shape_CVPR_2020_paper.pdf)]
    * Title: SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Yue Jiang,  Dantong Ji,  Zhizhong Han,  Matthias Zwicker
    * Abstract: We propose SDFDiff, a novel approach for image-based shape optimization using differentiable rendering of 3D shapes represented by signed distance functions (SDFs). Compared to other representations, SDFs have the advantage that they can represent shapes with arbitrary topology, and that they guarantee watertight surfaces. We apply our approach to the problem of multi-view 3D reconstruction, where we achieve high reconstruction quality and can capture complex topology of 3D objects. In addition, we employ a multi-resolution strategy to obtain a robust optimization algorithm. We further demonstrate that our SDF-based differentiable renderer can be integrated with deep learning models, which opens up options for learning approaches on 3D objects without 3D supervision. In particular, we apply our method to single-view 3D reconstruction and achieve state-of-the-art results.

count=4
* PointRend: Image Segmentation As Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf)]
    * Title: PointRend: Image Segmentation As Rendering
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Alexander Kirillov,  Yuxin Wu,  Kaiming He,  Ross Girshick
    * Abstract: We present a new method for efficient high-quality image segmentation of objects and scenes. By analogizing classical computer graphics methods for efficient rendering with over- and undersampling challenges faced in pixel labeling tasks, we develop a unique perspective of image segmentation as a rendering problem. From this vantage, we present the PointRend (Point-based Rendering) neural network module: a module that performs point-based segmentation predictions at adaptively selected locations based on an iterative subdivision algorithm. PointRend can be flexibly applied to both instance and semantic segmentation tasks by building on top of existing state-of-the-art models. While many concrete implementations of the general idea are possible, we show that a simple design already achieves excellent results. Qualitatively, PointRend outputs crisp object boundaries in regions that are over-smoothed by previous methods. Quantitatively, PointRend yields significant gains on COCO and Cityscapes, for both instance and semantic segmentation. PointRend's efficiency enables output resolutions that are otherwise impractical in terms of memory or computation compared to existing approaches. Code has been made available at https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend.

count=4
* RelightableHands: Efficient Neural Relighting of Articulated Hand Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Iwase_RelightableHands_Efficient_Neural_Relighting_of_Articulated_Hand_Models_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Iwase_RelightableHands_Efficient_Neural_Relighting_of_Articulated_Hand_Models_CVPR_2023_paper.pdf)]
    * Title: RelightableHands: Efficient Neural Relighting of Articulated Hand Models
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Shun Iwase, Shunsuke Saito, Tomas Simon, Stephen Lombardi, Timur Bagautdinov, Rohan Joshi, Fabian Prada, Takaaki Shiratori, Yaser Sheikh, Jason Saragih
    * Abstract: We present the first neural relighting approach for rendering high-fidelity personalized hands that can be animated in real-time under novel illumination. Our approach adopts a teacher-student framework, where the teacher learns appearance under a single point light from images captured in a light-stage, allowing us to synthesize hands in arbitrary illuminations but with heavy compute. Using images rendered by the teacher model as training data, an efficient student model directly predicts appearance under natural illuminations in real-time. To achieve generalization, we condition the student model with physics-inspired illumination features such as visibility, diffuse shading, and specular reflections computed on a coarse proxy geometry, maintaining a small computational overhead. Our key insight is that these features have strong correlation with subsequent global light transport effects, which proves sufficient as conditioning data for the neural relighting network. Moreover, in contrast to bottleneck illumination conditioning, these features are spatially aligned based on underlying geometry, leading to better generalization to unseen illuminations and poses. In our experiments, we demonstrate the efficacy of our illumination feature representations, outperforming baseline approaches. We also show that our approach can photorealistically relight two interacting hands at real-time speeds.

count=4
* Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Learning_Visibility_Field_for_Detailed_3D_Human_Reconstruction_and_Relighting_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_Learning_Visibility_Field_for_Detailed_3D_Human_Reconstruction_and_Relighting_CVPR_2023_paper.pdf)]
    * Title: Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Ruichen Zheng, Peng Li, Haoqian Wang, Tao Yu
    * Abstract: Detailed 3D reconstruction and photo-realistic relighting of digital humans are essential for various applications. To this end, we propose a novel sparse-view 3d human reconstruction framework that closely incorporates the occupancy field and albedo field with an additional visibility field--it not only resolves occlusion ambiguity in multiview feature aggregation, but can also be used to evaluate light attenuation for self-shadowed relighting. To enhance its training viability and efficiency, we discretize visibility onto a fixed set of sample directions and supply it with coupled geometric 3D depth feature and local 2D image feature. We further propose a novel rendering-inspired loss, namely TransferLoss, to implicitly enforce the alignment between visibility and occupancy field, enabling end-to-end joint training. Results and extensive experiments demonstrate the effectiveness of the proposed method, as it surpasses state-of-the-art in terms of reconstruction accuracy while achieving comparably accurate relighting to ray-traced ground truth.

count=4
* Gaussian Shadow Casting for Neural Characters
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Bolanos_Gaussian_Shadow_Casting_for_Neural_Characters_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Bolanos_Gaussian_Shadow_Casting_for_Neural_Characters_CVPR_2024_paper.pdf)]
    * Title: Gaussian Shadow Casting for Neural Characters
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Luis Bolanos, Shih-Yang Su, Helge Rhodin
    * Abstract: Neural character models can now reconstruct detailed geometry and texture from video but they lack explicit shadows and shading leading to artifacts when generating novel views and poses or during relighting. It is particularly difficult to include shadows as they are a global effect and the required casting of secondary rays is costly. We propose a new shadow model using a Gaussian density proxy that replaces sampling with a simple analytic formula. It supports dynamic motion and is tailored for shadow computation thereby avoiding the affine projection approximation and sorting required by the closely related Gaussian splatting. Combined with a deferred neural rendering model our Gaussian shadows enable Lambertian shading and shadow casting with minimal overhead. We demonstrate improved reconstructions with better separation of albedo shading and shadows in challenging outdoor scenes with direct sun light and hard shadows. Our method is able to optimize the light direction without any input from the user. As a result novel poses have fewer shadow artifacts and relighting in novel scenes is more realistic compared to the state-of-the-art methods providing new ways to pose neural characters in novel environments increasing their applicability. Code available at: https://github.com/LuisBolanos17/GaussianShadowCasting

count=4
* Differentiable Neural Surface Refinement for Modeling Transparent Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Deng_Differentiable_Neural_Surface_Refinement_for_Modeling_Transparent_Objects_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_Differentiable_Neural_Surface_Refinement_for_Modeling_Transparent_Objects_CVPR_2024_paper.pdf)]
    * Title: Differentiable Neural Surface Refinement for Modeling Transparent Objects
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Weijian Deng, Dylan Campbell, Chunyi Sun, Shubham Kanitkar, Matthew E. Shaffer, Stephen Gould
    * Abstract: Neural implicit surface reconstruction leveraging volume rendering has led to significant advances in multi-view reconstruction. However results for transparent objects can be very poor primarily because the rendering function fails to account for the intricate light transport induced by refraction and reflection. In this study we introduce transparent neural surface refinement (TNSR) a novel surface reconstruction framework that explicitly incorporates physical refraction and reflection tracing. Beginning with an initial approximate surface our method employs sphere tracing combined with Snell's law to cast both reflected and refracted rays. Central to our proposal is an innovative differentiable technique devised to allow signals from the photometric evidence to propagate back to the surface model by considering how the surface bends and reflects light rays. This allows us to connect surface refinement with volume rendering enabling end-to-end optimization solely on multi-view RGB images. In our experiments TNSR demonstrates significant improvements in novel view synthesis and geometry estimation of transparent objects without prior knowledge of the refractive index.

count=4
* DART: Implicit Doppler Tomography for Radar Novel View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_DART_Implicit_Doppler_Tomography_for_Radar_Novel_View_Synthesis_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DART_Implicit_Doppler_Tomography_for_Radar_Novel_View_Synthesis_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_DART_Implicit_Doppler_Tomography_for_Radar_Novel_View_Synthesis_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DART_Implicit_Doppler_Tomography_for_Radar_Novel_View_Synthesis_CVPR_2024_paper.pdf)]
    * Title: DART: Implicit Doppler Tomography for Radar Novel View Synthesis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tianshu Huang, John Miller, Akarsh Prabhakara, Tao Jin, Tarana Laroia, Zico Kolter, Anthony Rowe
    * Abstract: Simulation is an invaluable tool for radio-frequency system designers that enables rapid prototyping of various algorithms for imaging target detection classification and tracking. However simulating realistic radar scans is a challenging task that requires an accurate model of the scene radio frequency material properties and a corresponding radar synthesis function. Rather than specifying these models explicitly we propose DART - Doppler Aided Radar Tomography a Neural Radiance Field-inspired method which uses radar-specific physics to create a reflectance and transmittance-based rendering pipeline for range-Doppler images. We then evaluate DART by constructing a custom data collection platform and collecting a novel radar dataset together with accurate position and instantaneous velocity measurements from lidar-based localization. In comparison to state-of-the-art baselines DART synthesizes superior radar range-Doppler images from novel views across all datasets and additionally can be used to generate high quality tomographic images.

count=4
* Single-Shot Specular Surface Reconstruction With Gonio-Plenoptic Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Meng_Single-Shot_Specular_Surface_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Meng_Single-Shot_Specular_Surface_ICCV_2015_paper.pdf)]
    * Title: Single-Shot Specular Surface Reconstruction With Gonio-Plenoptic Imaging
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Lingfei Meng, Liyang Lu, Noah Bedard, Kathrin Berkner
    * Abstract: We present a gonio-plenoptic imaging system that realizes a single-shot shape measurement for specular surfaces. The system is comprised of a collimated illumination source and a plenoptic camera. Unlike a conventional plenoptic camera, our system captures the BRDF variation of the object surface in a single image in addition to the light field information from the scene, which allows us to recover very fine 3D structures of the surface. The shape of the surface is reconstructed based on the reflectance property of the material rather than the parallax between different views. Since only a single-shot is required to reconstruct the whole surface, our system is able to capture dynamic surface deformation in a video mode. We also describe a novel calibration technique that maps the light field viewing directions from the object space to subpixels on the sensor plane. The proposed system is evaluated using a concave mirror with known curvature, and is compared to a parabolic mirror scanning system as well as a multi-illumination photometric stereo approach based on simulations and experiments.

count=4
* A Differential Volumetric Approach to Multi-View Photometric Stereo
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Logothetis_A_Differential_Volumetric_Approach_to_Multi-View_Photometric_Stereo_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Logothetis_A_Differential_Volumetric_Approach_to_Multi-View_Photometric_Stereo_ICCV_2019_paper.pdf)]
    * Title: A Differential Volumetric Approach to Multi-View Photometric Stereo
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Fotios Logothetis,  Roberto Mecca,  Roberto Cipolla
    * Abstract: Highly accurate 3D volumetric reconstruction is still an open research topic where the main difficulty is usually related to merging some rough estimations with high frequency details. One of the most promising methods is the fusion between multi-view stereo and photometric stereo images. Beside the intrinsic difficulties that multi-view stereo and photometric stereo in order to work reliably, supplementary problems arise when considered together. In this work, we present a volumetric approach to the multi-view photometric stereo problem. The key point of our method is the signed distance field parameterisation and its relation to the surface normal. This is exploited in order to obtain a linear partial differential equation which is solved in a variational framework, that combines multiple images from multiple points of view in a single system. In addition, the volumetric approach is naturally implemented on an octree, which allows for fast ray-tracing that reliably alleviates occlusions and cast shadows. Our approach is evaluated on synthetic and real data-sets and achieves state-of-the-art results.

count=4
* ViewNet: Unsupervised Viewpoint Estimation From Conditional Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Mariotti_ViewNet_Unsupervised_Viewpoint_Estimation_From_Conditional_Generation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Mariotti_ViewNet_Unsupervised_Viewpoint_Estimation_From_Conditional_Generation_ICCV_2021_paper.pdf)]
    * Title: ViewNet: Unsupervised Viewpoint Estimation From Conditional Generation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Octave Mariotti, Oisin Mac Aodha, Hakan Bilen
    * Abstract: Understanding the 3D world without supervision is currently a major challenge in computer vision as the annotations required to supervise deep networks for tasks in this domain are expensive to obtain on a large scale. In this paper, we address the problem of unsupervised viewpoint estimation. We formulate this as a self-supervised learning task, where image reconstruction provides the supervision needed to predict the camera viewpoint. Specifically, we make use of pairs of images of the same object at training time, from unknown viewpoints, to self-supervise training by combining the viewpoint information from one image with the appearance information from the other. We demonstrate that using a perspective spatial transformer allows efficient viewpoint learning, outperforming existing unsupervised approaches on synthetic data, and obtains competitive results on the challenging PASCAL3D+ dataset.

count=4
* Learning Signed Distance Field for Multi-View Surface Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Learning_Signed_Distance_Field_for_Multi-View_Surface_Reconstruction_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learning_Signed_Distance_Field_for_Multi-View_Surface_Reconstruction_ICCV_2021_paper.pdf)]
    * Title: Learning Signed Distance Field for Multi-View Surface Reconstruction
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jingyang Zhang, Yao Yao, Long Quan
    * Abstract: Recent works on implicit neural representations have shown promising results for multi-view surface reconstruction. However, most approaches are limited to relatively simple geometries and usually require clean object masks for reconstructing complex and concave objects. In this work, we introduce a novel neural surface reconstruction framework that leverages the knowledge of stereo matching and feature consistency to optimize the implicit surface representation. More specifically, we apply a signed distance field (SDF) and a surface light field to represent the scene geometry and appearance respectively. The SDF is directly supervised by geometry from stereo matching, and is refined by optimizing the multi-view feature consistency and the fidelity of rendered images. Our method is able to improve the robustness of geometry estimation and support reconstruction of complex scene topologies. Extensive experiments have been conducted on DTU, EPFL and Tanks and Temples datasets. Compared to previous state-of-the-art methods, our method achieves better mesh reconstruction in wide open scenes without masks as input.

count=4
* Simulated Photorealistic Deep Learning Framework and Workflows To Accelerate Computer Vision and Unmanned Aerial Vehicle Research
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/html/Alvey_Simulated_Photorealistic_Deep_Learning_Framework_and_Workflows_To_Accelerate_Computer_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/papers/Alvey_Simulated_Photorealistic_Deep_Learning_Framework_and_Workflows_To_Accelerate_Computer_ICCVW_2021_paper.pdf)]
    * Title: Simulated Photorealistic Deep Learning Framework and Workflows To Accelerate Computer Vision and Unmanned Aerial Vehicle Research
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Brendan Alvey, Derek T. Anderson, Andrew Buck, Matthew Deardorff, Grant Scott, James M. Keller
    * Abstract: Deep learning (DL) is producing state-of-the-art results in a number of unmanned aerial vehicle (UAV) tasks from low level signal processing to object detection, 3D mapping, tracking, fusion, autonomy, control, and beyond. However, barriers exist. For example, most DL algorithms require big data, but supervised ground truth is a bottleneck, fueling topics like self-supervised learning. While it is well-known that hardware and data augmentation plays a significant role in performance, it is not well understood which data augmentations or what real data need be collected. Furthermore, existing datasets do not have sufficient ground truth nor variety to support adequate controlled experimental research into understanding and mitigating limitations in DL algorithms, models, data, and biases. In this article, we address the combination of photorealistic simulation, open source libraries, and high quality content (models, materials, and environments) to develop workflows to mitigate the above challenges and accelerate DL-enabled computer vision research. Herein, examples are provided relative to data collection, detection, passive ranging, and human-robot teaming. Online video tutorials are also provided at https://github.com/MizzouINDFUL/UEUAVSim.

count=4
* Relightify: Relightable 3D Faces from a Single Image via Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Papantoniou_Relightify_Relightable_3D_Faces_from_a_Single_Image_via_Diffusion_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Papantoniou_Relightify_Relightable_3D_Faces_from_a_Single_Image_via_Diffusion_ICCV_2023_paper.pdf)]
    * Title: Relightify: Relightable 3D Faces from a Single Image via Diffusion Models
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Foivos Paraperas Papantoniou, Alexandros Lattas, Stylianos Moschoglou, Stefanos Zafeiriou
    * Abstract: Following the remarkable success of diffusion models on image generation, recent works have also demonstrated their impressive ability to address a number of inverse problems in an unsupervised way, by properly constraining the sampling process based on a conditioning input. Motivated by this, in this paper, we present the first approach to use diffusion models as a prior for highly accurate 3D facial BRDF reconstruction from a single image. We start by leveraging a high-quality UV dataset of facial reflectance (diffuse and specular albedo and normals), which we render under varying illumination settings to simulate natural RGB textures and, then, train an unconditional diffusion model on concatenated pairs of rendered textures and reflectance components. At test time, we fit a 3D morphable model to the given image and unwrap the face in a partial UV texture. By sampling from the diffusion model, while retaining the observed texture part intact, the model inpaints not only the self-occluded areas but also the unknown reflectance components, in a single sequence of denoising steps. In contrast to existing methods, we directly acquire the observed texture from the input image, thus, resulting in more faithful and consistent reflectance estimation. Through a series of qualitative and quantitative comparisons, we demonstrate superior performance in both texture completion as well as reflectance reconstruction tasks.

count=4
* 3D Shape Reconstruction of Plant Roots in a Cylindrical Tank From Multiview Images
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/3DRW/Masuda_3D_Shape_Reconstruction_of_Plant_Roots_in_a_Cylindrical_Tank_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/3DRW/Masuda_3D_Shape_Reconstruction_of_Plant_Roots_in_a_Cylindrical_Tank_ICCVW_2019_paper.pdf)]
    * Title: 3D Shape Reconstruction of Plant Roots in a Cylindrical Tank From Multiview Images
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Takeshi Masuda
    * Abstract: We propose a method for reconstructing a 3D shape of live plant roots submerged in a transparent cylindrical hydroponic tank from multiple-view images for root phenotyping. The proposed method does not assume special devices and careful setups, but the geometry and material of the tank are assumed known. First, we estimate the intrinsic and extrinsic camera parameters by the SfM algorithm, and the scale and axis of the tank are estimated by chamfer matching. Second, we apply the ray tracing considering the refraction for each view, the input images are mapped to the voxels, and then multiview voxels at the same location are robustly merged to reconstruct the 3D shape. Finally, the root feature extracted from the voxel is binarized and thinned by applying the 3D Canny operator. The proposed method was applied to real a dataset, and the reconstruction results are presented.

count=4
* Recovering Fine Details for Neural Implicit Surface Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Chen_Recovering_Fine_Details_for_Neural_Implicit_Surface_Reconstruction_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Chen_Recovering_Fine_Details_for_Neural_Implicit_Surface_Reconstruction_WACV_2023_paper.pdf)]
    * Title: Recovering Fine Details for Neural Implicit Surface Reconstruction
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Decai Chen, Peng Zhang, Ingo Feldmann, Oliver Schreer, Peter Eisert
    * Abstract: Recent works on implicit neural representations have made significant strides. Learning implicit neural surfaces using volume rendering has gained popularity in multi-view reconstruction without 3D supervision. However, accurately recovering fine details is still challenging, due to the underlying ambiguity of geometry and appearance representation. In this paper, we present D-NeuS, a volume rendering-base neural implicit surface reconstruction method capable to recover fine geometry details, which extends NeuS by two additional loss functions targeting enhanced reconstruction quality. First, we encourage the rendered surface points from alpha compositing to have zero signed distance values, alleviating the geometry bias arising from transforming SDF to density for volume rendering. Second, we impose multi-view feature consistency on the surface points, derived by interpolating SDF zero-crossings from sampled points along rays. Extensive quantitative and qualitative results demonstrate that our method reconstructs high-accuracy surfaces with details, and outperforms the state of the art.

count=4
* SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/83fa5a432ae55c253d0e60dbfa716723-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/83fa5a432ae55c253d0e60dbfa716723-Paper.pdf)]
    * Title: SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Chen-Hsuan Lin, Chaoyang Wang, Simon Lucey
    * Abstract: Dense 3D object reconstruction from a single image has recently witnessed remarkable advances, but supervising neural networks with ground-truth 3D shapes is impractical due to the laborious process of creating paired image-shape datasets. Recent efforts have turned to learning 3D reconstruction without 3D supervision from RGB images with annotated 2D silhouettes, dramatically reducing the cost and effort of annotation. These techniques, however, remain impractical as they still require multi-view annotations of the same object instance during training. As a result, most experimental efforts to date have been limited to synthetic datasets. In this paper, we address this issue and propose SDF-SRN, an approach that requires only a single view of objects at training time, offering greater utility for real-world scenarios. SDF-SRN learns implicit 3D shape representations to handle arbitrary shape topologies that may exist in the datasets. To this end, we derive a novel differentiable rendering formulation for learning signed distance functions (SDF) from 2D silhouettes. Our method outperforms the state of the art under challenging single-view supervision settings on both synthetic and real-world datasets.

count=4
* Neural Unsigned Distance Fields for Implicit Function Learning
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/f69e505b08403ad2298b9f262659929a-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/f69e505b08403ad2298b9f262659929a-Paper.pdf)]
    * Title: Neural Unsigned Distance Fields for Implicit Function Learning
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Julian Chibane, Mohamad Aymen mir, Gerard Pons-Moll
    * Abstract: In this work we target a learnable output representation that allows continuous, high resolution outputs of arbitrary shape. Recent works represent 3D surfaces implicitly with a Neural Network, thereby breaking previous barriers in resolution, and ability to represent diverse topologies. However, neural implicit representations are limited to closed surfaces, which divide the space into inside and outside. Many real world objects such as walls of a scene scanned by a sensor, clothing, or a car with inner structures are not closed. This constitutes a significant barrier, in terms of data pre-processing (objects need to be artificially closed creating artifacts), and the ability to output open surfaces. In this work, we propose Neural Distance Fields (NDF), a neural network based model which predicts the unsigned distance field for arbitrary 3D shapes given sparse point clouds. NDF represent surfaces at high resolutions as prior implicit models, but do not require closed surface data, and significantly broaden the class of representable shapes in the output. NDF allow to extract the surface as very dense point clouds and as meshes. We also show that NDF allow for surface normal calculation and can be rendered using a slight modification of sphere tracing. We find NDF can be used for multi-target regression (multiple outputs for one input) with techniques that have been exclusively used for rendering in graphics. Experiments on ShapeNet show that NDF, while simple, is the state-of-the art, and allows to reconstruct shapes with inner structures, such as the chairs inside a bus. Notably, we show that NDF are not restricted to 3D shapes, and can approximate more general open surfaces such as curves, manifolds, and functions. Code is available for research at https://virtualhumans.mpi-inf.mpg.de/ndf/.

count=4
* SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/6d0c932802f6953f70eb20931645fa40-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/6d0c932802f6953f70eb20931645fa40-Paper.pdf)]
    * Title: SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Devendra Singh Chaplot, Murtaza Dalal, Saurabh Gupta, Jitendra Malik, Russ R. Salakhutdinov
    * Abstract: In this paper, we explore how we can build upon the data and models of Internet images and use them to adapt to robot vision without requiring any extra labels. We present a framework called Self-supervised Embodied Active Learning (SEAL). It utilizes perception models trained on internet images to learn an active exploration policy. The observations gathered by this exploration policy are labelled using 3D consistency and used to improve the perception model. We build and utilize 3D semantic maps to learn both action and perception in a completely self-supervised manner. The semantic map is used to compute an intrinsic motivation reward for training the exploration policy and for labelling the agent observations using spatio-temporal 3D consistency and label propagation. We demonstrate that the SEAL framework can be used to close the action-perception loop: it improves object detection and instance segmentation performance of a pretrained perception model by just moving around in training environments and the improved perception model can be used to improve Object Goal Navigation.

count=4
* Neural Relightable Participating Media Rendering
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/80f24ef493982c552b6943f1411f7e2c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/80f24ef493982c552b6943f1411f7e2c-Paper.pdf)]
    * Title: Neural Relightable Participating Media Rendering
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Quan Zheng, Gurprit Singh, Hans-peter Seidel
    * Abstract: Learning neural radiance fields of a scene has recently allowed realistic novel view synthesis of the scene, but they are limited to synthesize images under the original fixed lighting condition. Therefore, they are not flexible for the eagerly desired tasks like relighting, scene editing and scene composition. To tackle this problem, several recent methods propose to disentangle reflectance and illumination from the radiance field. These methods can cope with solid objects with opaque surfaces but participating media are neglected. Also, they take into account only direct illumination or at most one-bounce indirect illumination, thus suffer from energy loss due to ignoring the high-order indirect illumination. We propose to learn neural representations for participating media with a complete simulation of global illumination. We estimate direct illumination via ray tracing and compute indirect illumination with spherical harmonics. Our approach avoids computing the lengthy indirect bounces and does not suffer from energy loss. Our experiments on multiple scenes show that our approach achieves superior visual quality and numerical performance compared to state-of-the-art methods, and it can generalize to deal with solid objects with opaque surfaces as well.

count=4
* DIB-R++: Learning to Predict Lighting and Material with a Hybrid Differentiable Renderer
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/c0f971d8cd24364f2029fcb9ac7b71f5-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/c0f971d8cd24364f2029fcb9ac7b71f5-Paper.pdf)]
    * Title: DIB-R++: Learning to Predict Lighting and Material with a Hybrid Differentiable Renderer
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Wenzheng Chen, Joey Litalien, Jun Gao, Zian Wang, Clement Fuji Tsang, Sameh Khamis, Or Litany, Sanja Fidler
    * Abstract: We consider the challenging problem of predicting intrinsic object properties from a single image by exploiting differentiable renderers. Many previous learning-based approaches for inverse graphics adopt rasterization-based renderers and assume naive lighting and material models, which often fail to account for non-Lambertian, specular reflections commonly observed in the wild. In this work, we propose DIBR++, a hybrid differentiable renderer which supports these photorealistic effects by combining rasterization and ray-tracing, taking the advantage of their respective strengths---speed and realism. Our renderer incorporates environmental lighting and spatially-varying material models to efficiently approximate light transport, either through direct estimation or via spherical basis functions. Compared to more advanced physics-based differentiable renderers leveraging path tracing, DIBR++ is highly performant due to its compact and expressive shading model, which enables easy integration with learning frameworks for geometry, reflectance and lighting prediction from a single image without requiring any ground-truth. We experimentally demonstrate that our approach achieves superior material and lighting disentanglement on synthetic and real data compared to existing rasterization-based approaches and showcase several artistic applications including material editing and relighting.

count=4
* INRAS: Implicit Neural Representation for Audio Scenes
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/35d5ad984cc0ddd84c6f1c177a2066e5-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/35d5ad984cc0ddd84c6f1c177a2066e5-Paper-Conference.pdf)]
    * Title: INRAS: Implicit Neural Representation for Audio Scenes
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Kun Su, Mingfei Chen, Eli Shlizerman
    * Abstract: The spatial acoustic information of a scene, i.e., how sounds emitted from a particular location in the scene are perceived in another location, is key for immersive scene modeling. Robust representation of scene's acoustics can be formulated through a continuous field formulation along with impulse responses varied by emitter-listener locations. The impulse responses are then used to render sounds perceived by the listener. While such representation is advantageous, parameterization of impulse responses for generic scenes presents itself as a challenge. Indeed, traditional pre-computation methods have only implemented parameterization at discrete probe points and require large storage, while other existing methods such as geometry-based sound simulations still suffer from inability to simulate all wave-based sound effects. In this work, we introduce a novel neural network for light-weight Implicit Neural Representation for Audio Scenes (INRAS), which can render a high fidelity time-domain impulse responses at any arbitrary emitter-listener positions by learning a continuous implicit function. INRAS disentangles scene’s geometry features with three modules to generate independent features for the emitter, the geometry of the scene, and the listener respectively. These lead to an efficient reuse of scene-dependent features and support effective multi-condition training for multiple scenes. Our experimental results show that INRAS outperforms existing approaches for representation and rendering of sounds for varying emitter-listener locations in all aspects, including the impulse response quality, inference speed, and storage requirements.

count=3
* Efficient Computation of Shortest Path-Concavity for 3D Meshes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Zimmer_Efficient_Computation_of_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Zimmer_Efficient_Computation_of_2013_CVPR_paper.pdf)]
    * Title: Efficient Computation of Shortest Path-Concavity for 3D Meshes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Henrik Zimmer, Marcel Campen, Leif Kobbelt
    * Abstract: In the context of shape segmentation and retrieval object-wide distributions of measures are needed to accurately evaluate and compare local regions of shapes. Lien et al. [16] proposed two point-wise concavity measures in the context of Approximate Convex Decompositions of polygons measuring the distance from a point to the polygon's convex hull: an accurate Shortest Path-Concavity (SPC) measure and a Straight Line-Concavity (SLC) approximation of the same. While both are practicable on 2D shapes, the exponential costs of SPC in 3D makes it inhibitively expensive for a generalization to meshes [14]. In this paper we propose an efficient and straight forward approximation of the Shortest Path-Concavity measure to 3D meshes. Our approximation is based on discretizing the space between mesh and convex hull, thereby reducing the continuous Shortest Path search to an efficiently solvable graph problem. Our approach works outof-the-box on complex mesh topologies and requires no complicated handling of genus. Besides presenting a rigorous evaluation of our method on a variety of input meshes, we also define an SPC-based Shape Descriptor and show its superior retrieval and runtime performance compared with the recently presented results on the Convexity Distribution by Lian et al. [12].

count=3
* Two-View Camera Housing Parameters Calibration for Multi-Layer Flat Refractive Interface
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Chen_Two-View_Camera_Housing_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Chen_Two-View_Camera_Housing_2014_CVPR_paper.pdf)]
    * Title: Two-View Camera Housing Parameters Calibration for Multi-Layer Flat Refractive Interface
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Xida Chen, Yee-Hong Yang
    * Abstract: In this paper, we present a novel refractive calibration method for an underwater stereo camera system where both cameras are looking through multiple parallel flat refractive interfaces. At the heart of our method is an important finding that the thickness of the interface can be estimated from a set of pixel correspondences in the stereo images when the refractive axis is given. To our best knowledge, such a finding has not been studied or reported. Moreover, by exploring the search space for the refractive axis and using reprojection error as a measure, both the refractive axis and the thickness of the interface can be recovered simultaneously. Our method does not require any calibration target such as a checkerboard pattern which may be difficult to manipulate when the cameras are deployed deep undersea. The implementation of our method is simple. In particular, it only requires solving a set of linear equations of the form Ax = b and applies sparse bundle adjustment to refine the initial estimated results. Extensive experiments have been carried out which include simulations with and without outliers to verify the correctness of our method as well as to test its robustness to noise and outliers. The results of real experiments are also provided. The accuracy of our results is comparable to that of a state-of-the-art method that requires known 3D geometry of a scene.

count=3
* Stereo-Based 3D Reconstruction of Dynamic Fluid Surfaces by Global Optimization
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Qian_Stereo-Based_3D_Reconstruction_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qian_Stereo-Based_3D_Reconstruction_CVPR_2017_paper.pdf)]
    * Title: Stereo-Based 3D Reconstruction of Dynamic Fluid Surfaces by Global Optimization
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yiming Qian, Minglun Gong, Yee-Hong Yang
    * Abstract: 3D reconstruction of dynamic fluid surfaces is an open and challenging problem in computer vision. Unlike previous approaches that reconstruct each surface point independently and often return noisy depth maps, we propose a novel global optimization-based approach that recovers both depths and normals of all 3D points simultaneously. Using the traditional refraction stereo setup, we capture the wavy appearance of a pre-generated random pattern, and then estimate the correspondences between the captured images and the known background by tracking the pattern. Assuming that the light is refracted only once through the fluid interface, we minimize an objective function that incorporates both the cross-view normal consistency constraint and the single-view normal consistency constraints. The key idea is that the normals required for light refraction based on Snell's law from one view should agree with not only the ones from the second view, but also the ones estimated from local 3D geometry. Moreover, an effective reconstruction error metric is designed for estimating the refractive index of the fluid. We report experimental results on both synthetic and real data demonstrating that the proposed approach is accurate and shows superiority over the conventional stereo-based method.

count=3
* Infrared Variation Optimized Deep Convolutional Neural Network for Robust Automatic Ground Target Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w3/html/Kim_Infrared_Variation_Optimized_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w3/papers/Kim_Infrared_Variation_Optimized_CVPR_2017_paper.pdf)]
    * Title: Infrared Variation Optimized Deep Convolutional Neural Network for Robust Automatic Ground Target Recognition
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Sungho Kim, Woo-Jin Song, So-Hyun Kim
    * Abstract: ATR is a traditionally unsolved problem in military applications because of the wide range of infrared (IR) image variations and limited number of training images. Recently, deep convolutional neural network-based approaches in RGB images (RGB-CNN) showed breakthrough performance in computer vision problems. The direct use of the RGB-CNN to IR ATR problem fails to work because of the IR database problems. This paper presents a novel infrared variation-optimized deep convolutional neural network (IVO-CNN) by considering database management, such as increasing the database by a thermal simulator, controlling the image contrast automatically and suppressing the thermal noise to reduce the effects of infrared image variations in deep convolutional neural network-based automatic ground target recognition. The experimental results on the synthesized infrared images generated by the thermal simulator (OKTAL-SE) validated the feasibility of IVO-CNN for military ATR applications.

count=3
* Neural Point Cloud Rendering via Multi-Plane Projection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Dai_Neural_Point_Cloud_Rendering_via_Multi-Plane_Projection_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dai_Neural_Point_Cloud_Rendering_via_Multi-Plane_Projection_CVPR_2020_paper.pdf)]
    * Title: Neural Point Cloud Rendering via Multi-Plane Projection
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Peng Dai,  Yinda Zhang,  Zhuwen Li,  Shuaicheng Liu,  Bing Zeng
    * Abstract: We present a new deep point cloud rendering pipeline through multi-plane projections. The input to the network is the raw point cloud of a scene and the output are image or image sequences from a novel view or along a novel camera trajectory. Unlike previous approaches that directly project features from 3D points onto 2D image domain, we propose to project these features into a layered volume of camera frustum. In this way, the visibility of 3D points can be automatically learnt by the network, such that ghosting effects due to false visibility check as well as occlusions caused by noise interferences are both avoided successfully. Next, the 3D feature volume is fed into a 3D CNN to produce multiple planes of images w.r.t. the space division in the depth directions. The multi-plane images are then blended based on learned weights to produce the final rendering results. Experiments show that our network produces more stable renderings compared to previous methods, especially near the object boundaries. Moreover, our pipeline is robust to noisy and relatively sparse point cloud for a variety of challenging scenes.

count=3
* ARCH: Animatable Reconstruction of Clothed Humans
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_ARCH_Animatable_Reconstruction_of_Clothed_Humans_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_ARCH_Animatable_Reconstruction_of_Clothed_Humans_CVPR_2020_paper.pdf)]
    * Title: ARCH: Animatable Reconstruction of Clothed Humans
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zeng Huang,  Yuanlu Xu,  Christoph Lassner,  Hao Li,  Tony Tung
    * Abstract: In this paper, we propose ARCH (Animatable Reconstruction of Clothed Humans), a novel end-to-end framework for accurate reconstruction of animation-ready 3D clothed humans from a monocular image. Existing approaches to digitize 3D humans struggle to handle pose variations and recover details. Also, they do not produce models that are animation ready. In contrast, ARCH is a learnedpose-awaremodelthatproducesdetailed3Drigged full-body human avatars from a single unconstrained RGB image. A Semantic Space and a Semantic Deformation Field are created using a parametric 3D body estimator. They allow the transformation of 2D/3D clothed humans into a canonical space, reducing ambiguities in geometry caused by pose variations and occlusions in training data. Detailed surface geometry and appearance are learned using an implicit function representation with spatial local features. Furthermore, we propose additional per-pixel supervisiononthe3Dreconstructionusingopacity-awaredifferentiablerendering. OurexperimentsindicatethatARCH increases the fidelity of the reconstructed humans. We obtain more than 50% lower reconstruction errors for standard metrics compared to state-of-the-art methods on public datasets. We also show numerous qualitative examples of animated, high-quality reconstructed avatars unseen in the literature so far.

count=3
* Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Through_the_Looking_Glass_Neural_3D_Reconstruction_of_Transparent_Shapes_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Through_the_Looking_Glass_Neural_3D_Reconstruction_of_Transparent_Shapes_CVPR_2020_paper.pdf)]
    * Title: Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zhengqin Li,  Yu-Ying Yeh,  Manmohan Chandraker
    * Abstract: Recovering the 3D shape of transparent objects using a small number of unconstrained natural images is an ill-posed problem. Complex light paths induced by refraction and reflection have prevented both traditional and deep multiview stereo from solving this challenge. We propose a physically-based network to recover 3D shape of transparent objects using a few images acquired with a mobile phone camera, under a known but arbitrary environment map. Our novel contributions include a normal representation that enables the network to model complex light transport through local computation, a rendering layer that models refractions and reflections, a cost volume specifically designed for normal refinement of transparent shapes and a feature mapping based on predicted normals for 3D point cloud reconstruction. We render a synthetic dataset to encourage the model to learn refractive light transport across different views. Our experiments show successful recovery of high-quality 3D geometry for complex transparent shapes using as few as 5-12 natural images. Code and data will be publicly released.Recovering the 3D shape of transparent objects using a small number of unconstrained natural images is an ill-posed problem. Complex light paths induced by refraction and reflection have prevented both traditional and deep multiview stereo from solving this challenge. We propose a physically-based network to recover 3D shape of transparent objects using a few images acquired with a mobile phone camera, under a known but arbitrary environment map. Our novel contributions include a normal representation that enables the network to model complex light transport through local computation, a rendering layer that models refractions and reflections, a cost volume specifically designed for normal refinement of transparent shapes and a feature mapping based on predicted normals for 3D point cloud reconstruction. We render a synthetic dataset to encourage the model to learn refractive light transport across different views. Our experiments show successful recovery of high-quality 3D geometry for complex transparent shapes using as few as 5-12 natural images. Code and data will be publicly released.

count=3
* Globally Optimal Contrast Maximisation for Event-Based Motion Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Globally_Optimal_Contrast_Maximisation_for_Event-Based_Motion_Estimation_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Globally_Optimal_Contrast_Maximisation_for_Event-Based_Motion_Estimation_CVPR_2020_paper.pdf)]
    * Title: Globally Optimal Contrast Maximisation for Event-Based Motion Estimation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Daqi Liu,  Alvaro Parra,  Tat-Jun Chin
    * Abstract: Contrast maximisation estimates the motion captured in an event stream by maximising the sharpness of the motion-compensated event image. To carry out contrast maximisation, many previous works employ iterative optimisation algorithms, such as conjugate gradient, which require good initialisation to avoid converging to bad local minima. To alleviate this weakness, we propose a new globally optimal event-based motion estimation algorithm. Based on branch-and-bound (BnB), our method solves rotational (3DoF) motion estimation on event streams, which supports practical applications such as video stabilisation and attitude estimation. Underpinning our method are novel bounding functions for contrast maximisation, whose theoretical validity is rigorously established. We show concrete examples from public datasets where globally optimal solutions are vital to the success of contrast maximisation. Despite its exact nature, our algorithm is currently able to process a 50,000-event input in approx 300 seconds (a locally optimal solver takes approx 30 seconds on the same input), and has the potential to be further speeded-up using GPUs.

count=3
* SAPIEN: A SimulAted Part-Based Interactive ENvironment
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Xiang_SAPIEN_A_SimulAted_Part-Based_Interactive_ENvironment_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xiang_SAPIEN_A_SimulAted_Part-Based_Interactive_ENvironment_CVPR_2020_paper.pdf)]
    * Title: SAPIEN: A SimulAted Part-Based Interactive ENvironment
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Fanbo Xiang,  Yuzhe Qin,  Kaichun Mo,  Yikuan Xia,  Hao Zhu,  Fangchen Liu,  Minghua Liu,  Hanxiao Jiang,  Yifu Yuan,  He Wang,  Li Yi,  Angel X. Chang,  Leonidas J. Guibas,  Hao Su
    * Abstract: Building home assistant robots has long been a goal for vision and robotics researchers. To achieve this task, a simulated environment with physically realistic simulation, sufficient articulated objects, and transferability to the real robot is indispensable. Existing environments achieve these requirements for robotics simulation with different levels of simplification and focus. We take one step further in constructing an environment that supports household tasks for training robot learning algorithm. Our work, SAPIEN, is a realistic and physics-rich simulated environment that hosts a large-scale set of articulated objects. SAPIEN enables various robotic vision and interaction tasks that require detailed part-level understanding.We evaluate state-of-the-art vision algorithms for part detection and motion attribute recognition as well as demonstrate robotic interaction tasks using heuristic approaches and reinforcement learning algorithms. We hope that SAPIEN will open research directions yet to be explored, including learning cognition through interaction, part motion discovery, and construction of robotics-ready simulated game environment.

count=3
* Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Jafarian_Learning_High_Fidelity_Depths_of_Dressed_Humans_by_Watching_Social_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Jafarian_Learning_High_Fidelity_Depths_of_Dressed_Humans_by_Watching_Social_CVPR_2021_paper.pdf)]
    * Title: Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yasamin Jafarian, Hyun Soo Park
    * Abstract: A key challenge of learning the geometry of dressed humans lies in the limited availability of the ground truth data (e.g., 3D scanned models), which results in the performance degradation of 3D human reconstruction when applying to real world imagery. We address this challenge by leveraging a new data resource: a number of social media dance videos that span diverse appearance, clothing styles, performances, and identities. Each video depicts dynamic movements of the body and clothes of a single person while lacking the 3D ground truth geometry. To utilize these videos, we present a new method to use the local transformation that warps the predicted local geometry of the person from an image to that of the other image at a different time instant. With the transformation, the predicted geometry can be self-supervised by the warped geometry from the other image. In addition, we jointly learn the depth along with the surface normals, which are highly responsive to local texture, wrinkle, and shade by maximizing their geometric consistency. Our method is end-to-end trainable, resulting in high fidelity depth estimation that predicts fine geometry faithful to the input real image. We demonstrate that our method outperforms the state-of-the-art human depth estimation and human shape recovery approaches on both real and rendered images.

count=3
* Kubric: A Scalable Dataset Generator
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.pdf)]
    * Title: Kubric: A Scalable Dataset Generator
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J. Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, Thomas Kipf, Abhijit Kundu, Dmitry Lagun, Issam Laradji, Hsueh-Ti (Derek) Liu, Henning Meyer, Yishu Miao, Derek Nowrouzezahrai, Cengiz Oztireli, Etienne Pot, Noha Radwan, Daniel Rebain, Sara Sabour, Mehdi S. M. Sajjadi, Matan Sela, Vincent Sitzmann, Austin Stone, Deqing Sun, Suhani Vora, Ziyu Wang, Tianhao Wu, Kwang Moo Yi, Fangcheng Zhong, Andrea Tagliasacchi
    * Abstract: Data is the driving force of machine learning, with the amount and quality of training data often being more important for the performance of a system than architecture and training details. But collecting, processing and annotating real data at scale is difficult, expensive, and frequently raises additional privacy, fairness and legal concerns. Synthetic data is a powerful tool with the potential to address these shortcomings: 1) it is cheap 2) supports rich ground-truth annotations 3) offers full control over data and 4) can circumvent or mitigate problems regarding bias, privacy and licensing. Unfortunately, software tools for effective data generation are less mature than those for architecture design and training, which leads to fragmented generation efforts. To address these problems we introduce Kubric, an open-source Python framework that interfaces with PyBullet and Blender to generate photo-realistic scenes, with rich annotations, and seamlessly scales to large jobs distributed over thousands of machines, and generating TBs of data. We demonstrate the effectiveness of Kubric by presenting a series of 11 different generated datasets for tasks ranging from studying 3D NeRF models to optical flow estimation. We release Kubric, the used assets, all of the generation code, as well as the rendered datasets for reuse and modification.

count=3
* BokehMe: When Neural Rendering Meets Classical Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_BokehMe_When_Neural_Rendering_Meets_Classical_Rendering_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_BokehMe_When_Neural_Rendering_Meets_Classical_Rendering_CVPR_2022_paper.pdf)]
    * Title: BokehMe: When Neural Rendering Meets Classical Rendering
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Juewen Peng, Zhiguo Cao, Xianrui Luo, Hao Lu, Ke Xian, Jianming Zhang
    * Abstract: We propose BokehMe, a hybrid bokeh rendering framework that marries a neural renderer with a classical physically motivated renderer. Given a single image and a potentially imperfect disparity map, BokehMe generates high-resolution photo-realistic bokeh effects with adjustable blur size, focal plane, and aperture shape. To this end, we analyze the errors from the classical scattering-based method and derive a formulation to calculate an error map. Based on this formulation, we implement the classical renderer by a scattering-based method and propose a two-stage neural renderer to fix the erroneous areas from the classical renderer. The neural renderer employs a dynamic multi-scale scheme to efficiently handle arbitrary blur sizes, and it is trained to handle imperfect disparity input. Experiments show that our method compares favorably against previous methods on both synthetic image data and real image data with predicted disparity. A user study is further conducted to validate the advantage of our method.

count=3
* Deep 3D-to-2D Watermarking: Embedding Messages in 3D Meshes and Extracting Them From 2D Renderings
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yoo_Deep_3D-to-2D_Watermarking_Embedding_Messages_in_3D_Meshes_and_Extracting_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yoo_Deep_3D-to-2D_Watermarking_Embedding_Messages_in_3D_Meshes_and_Extracting_CVPR_2022_paper.pdf)]
    * Title: Deep 3D-to-2D Watermarking: Embedding Messages in 3D Meshes and Extracting Them From 2D Renderings
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Innfarn Yoo, Huiwen Chang, Xiyang Luo, Ondrej Stava, Ce Liu, Peyman Milanfar, Feng Yang
    * Abstract: Digital watermarking is widely used for copyright protection. Traditional 3D watermarking approaches or commercial software are typically designed to embed messages into 3D meshes, and later retrieve the messages directly from distorted/undistorted watermarked 3D meshes. However, in many cases, users only have access to rendered 2D images instead of 3D meshes. Unfortunately, retrieving messages from 2D renderings of 3D meshes is still challenging and underexplored. We introduce a novel end-to-end learning framework to solve this problem through: 1) an encoder to covertly embed messages in both mesh geometry and textures; 2) a differentiable renderer to render watermarked 3D objects from different camera angles and under varied lighting conditions; 3) a decoder to recover the messages from 2D rendered images. From our experiments, we show that our model can learn to embed information visually imperceptible to humans, and to retrieve the embedded information from 2D renderings that undergo 3D distortions. In addition, we demonstrate that our method can also work with other renderers, such as ray tracers and real-time renderers with and without fine-tuning.

count=3
* Modeling Indirect Illumination for Inverse Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Modeling_Indirect_Illumination_for_Inverse_Rendering_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Modeling_Indirect_Illumination_for_Inverse_Rendering_CVPR_2022_paper.pdf)]
    * Title: Modeling Indirect Illumination for Inverse Rendering
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, Xiaowei Zhou
    * Abstract: Recent advances in implicit neural representations and differentiable rendering make it possible to simultaneously recover the geometry and materials of an object from multi-view RGB images captured under unknown static illumination. Despite the promising results achieved, indirect illumination is rarely modeled in previous methods, as it requires expensive recursive path tracing which makes the inverse rendering computationally intractable. In this paper, we propose a novel approach to efficiently recovering spatially-varying indirect illumination. The key insight is that indirect illumination can be conveniently derived from the neural radiance field learned from input images instead of being estimated jointly with direct illumination and materials. By properly modeling the indirect illumination and visibility of direct illumination, interreflection- and shadow-free albedo can be recovered. The experiments on both synthetic and real data demonstrate the superior performance of our approach compared to previous work and its capability to synthesize realistic renderings under novel viewpoints and illumination. Our code and data are available at https://zju3dv.github.io/invrender/.

count=3
* Look, Radiate, and Learn: Self-Supervised Localisation via Radio-Visual Correspondence
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Alloulah_Look_Radiate_and_Learn_Self-Supervised_Localisation_via_Radio-Visual_Correspondence_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Alloulah_Look_Radiate_and_Learn_Self-Supervised_Localisation_via_Radio-Visual_Correspondence_CVPR_2023_paper.pdf)]
    * Title: Look, Radiate, and Learn: Self-Supervised Localisation via Radio-Visual Correspondence
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Mohammed Alloulah, Maximilian Arnold
    * Abstract: Next generation cellular networks will implement radio sensing functions alongside customary communications, thereby enabling unprecedented worldwide sensing coverage outdoors. Deep learning has revolutionised computer vision but has had limited application to radio perception tasks, in part due to lack of systematic datasets and benchmarks dedicated to the study of the performance and promise of radio sensing. To address this gap, we present MaxRay: a synthetic radio-visual dataset and benchmark that facilitate precise target localisation in radio. We further propose to learn to localise targets in radio without supervision by extracting self-coordinates from radio-visual correspondence. We use such self-supervised coordinates to train a radio localiser network. We characterise our performance against a number of state-of-the-art baselines. Our results indicate that accurate radio target localisation can be automatically learned from paired radio-visual data without labels, which is important for empirical data. This opens the door for vast data scalability and may prove key to realising the promise of robust radio sensing atop a unified communication-perception cellular infrastructure. Dataset will be hosted on IEEE DataPort.

count=3
* High-Res Facial Appearance Capture From Polarized Smartphone Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Azinovic_High-Res_Facial_Appearance_Capture_From_Polarized_Smartphone_Images_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Azinovic_High-Res_Facial_Appearance_Capture_From_Polarized_Smartphone_Images_CVPR_2023_paper.pdf)]
    * Title: High-Res Facial Appearance Capture From Polarized Smartphone Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Dejan Azinović, Olivier Maury, Christophe Hery, Matthias Nießner, Justus Thies
    * Abstract: We propose a novel method for high-quality facial texture reconstruction from RGB images using a novel capturing routine based on a single smartphone which we equip with an inexpensive polarization foil. Specifically, we turn the flashlight into a polarized light source and add a polarization filter on top of the camera. Leveraging this setup, we capture the face of a subject with cross-polarized and parallel-polarized light. For each subject, we record two short sequences in a dark environment under flash illumination with different light polarization using the modified smartphone. Based on these observations, we reconstruct an explicit surface mesh of the face using structure from motion. We then exploit the camera and light co-location within a differentiable renderer to optimize the facial textures using an analysis-by-synthesis approach. Our method optimizes for high-resolution normal textures, diffuse albedo, and specular albedo using a coarse-to-fine optimization scheme. We show that the optimized textures can be used in a standard rendering pipeline to synthesize high-quality photo-realistic 3D digital humans in novel environments.

count=3
* Learning a 3D Morphable Face Reflectance Model From Low-Cost Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Han_Learning_a_3D_Morphable_Face_Reflectance_Model_From_Low-Cost_Data_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_Learning_a_3D_Morphable_Face_Reflectance_Model_From_Low-Cost_Data_CVPR_2023_paper.pdf)]
    * Title: Learning a 3D Morphable Face Reflectance Model From Low-Cost Data
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yuxuan Han, Zhibo Wang, Feng Xu
    * Abstract: Modeling non-Lambertian effects such as facial specularity leads to a more realistic 3D Morphable Face Model. Existing works build parametric models for diffuse and specular albedo using Light Stage data. However, only diffuse and specular albedo cannot determine the full BRDF. In addition, the requirement of Light Stage data is hard to fulfill for the research communities. This paper proposes the first 3D morphable face reflectance model with spatially varying BRDF using only low-cost publicly-available data. We apply linear shiness weighting into parametric modeling to represent spatially varying specular intensity and shiness. Then an inverse rendering algorithm is developed to reconstruct the reflectance parameters from non-Light Stage data, which are used to train an initial morphable reflectance model. To enhance the model's generalization capability and expressive power, we further propose an update-by-reconstruction strategy to finetune it on an in-the-wild dataset. Experimental results show that our method obtains decent rendering results with plausible facial specularities. Our code is released at https://yxuhan.github.io/ReflectanceMM/index.html.

count=3
* HARP: Personalized Hand Reconstruction From a Monocular RGB Video
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Karunratanakul_HARP_Personalized_Hand_Reconstruction_From_a_Monocular_RGB_Video_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Karunratanakul_HARP_Personalized_Hand_Reconstruction_From_a_Monocular_RGB_Video_CVPR_2023_paper.pdf)]
    * Title: HARP: Personalized Hand Reconstruction From a Monocular RGB Video
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Korrawe Karunratanakul, Sergey Prokudin, Otmar Hilliges, Siyu Tang
    * Abstract: We present HARP (HAnd Reconstruction and Personalization), a personalized hand avatar creation approach that takes a short monocular RGB video of a human hand as input and reconstructs a faithful hand avatar exhibiting a high-fidelity appearance and geometry. In contrast to the major trend of neural implicit representations, HARP models a hand with a mesh-based parametric hand model, a vertex displacement map, a normal map, and an albedo without any neural components. The explicit nature of our representation enables a truly scalable, robust, and efficient approach to hand avatar creation as validated by our experiments. HARP is optimized via gradient descent from a short sequence captured by a hand-held mobile phone and can be directly used in AR/VR applications with real-time rendering capability. To enable this, we carefully design and implement a shadow-aware differentiable rendering scheme that is robust to high degree articulations and self-shadowing regularly present in hand motions, as well as challenging lighting conditions. It also generalizes to unseen poses and novel viewpoints, producing photo-realistic renderings of hand animations. Furthermore, the learned HARP representation can be used for improving 3D hand pose estimation quality in challenging viewpoints. The key advantages of HARP are validated by the in-depth analyses on appearance reconstruction, novel view and novel pose synthesis, and 3D hand pose refinement. It is an AR/VR-ready personalized hand representation that shows superior fidelity and scalability.

count=3
* FitMe: Deep Photorealistic 3D Morphable Model Avatars
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lattas_FitMe_Deep_Photorealistic_3D_Morphable_Model_Avatars_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Lattas_FitMe_Deep_Photorealistic_3D_Morphable_Model_Avatars_CVPR_2023_paper.pdf)]
    * Title: FitMe: Deep Photorealistic 3D Morphable Model Avatars
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Alexandros Lattas, Stylianos Moschoglou, Stylianos Ploumpis, Baris Gecer, Jiankang Deng, Stefanos Zafeiriou
    * Abstract: In this paper, we introduce FitMe, a facial reflectance model and a differentiable rendering optimization pipeline, that can be used to acquire high-fidelity renderable human avatars from single or multiple images. The model consists of a multi-modal style-based generator, that captures facial appearance in terms of diffuse and specular reflectance, and a PCA-based shape model. We employ a fast differentiable rendering process that can be used in an optimization pipeline, while also achieving photorealistic facial shading. Our optimization process accurately captures both the facial reflectance and shape in high-detail, by exploiting the expressivity of the style-based latent representation and of our shape model. FitMe achieves state-of-the-art reflectance acquisition and identity preservation on single "in-the-wild" facial images, while it produces impressive scan-like results, when given multiple unconstrained facial images pertaining to the same identity. In contrast with recent implicit avatar reconstructions, FitMe requires only one minute and produces relightable mesh and texture-based avatars, that can be used by end-user applications.

count=3
* PixHt-Lab: Pixel Height Based Light Effect Generation for Image Compositing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Sheng_PixHt-Lab_Pixel_Height_Based_Light_Effect_Generation_for_Image_Compositing_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Sheng_PixHt-Lab_Pixel_Height_Based_Light_Effect_Generation_for_Image_Compositing_CVPR_2023_paper.pdf)]
    * Title: PixHt-Lab: Pixel Height Based Light Effect Generation for Image Compositing
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yichen Sheng, Jianming Zhang, Julien Philip, Yannick Hold-Geoffroy, Xin Sun, He Zhang, Lu Ling, Bedrich Benes
    * Abstract: Lighting effects such as shadows or reflections are key in making synthetic images realistic and visually appealing. To generate such effects, traditional computer graphics uses a physically-based renderer along with 3D geometry. To compensate for the lack of geometry in 2D Image compositing, recent deep learning-based approaches introduced a pixel height representation to generate soft shadows and reflections. However, the lack of geometry limits the quality of the generated soft shadows and constrains reflections to pure specular ones. We introduce PixHt-Lab, a system leveraging an explicit mapping from pixel height representation to 3D space. Using this mapping, PixHt-Lab reconstructs both the cutout and background geometry and renders realistic, diverse, lighting effects for image compositing. Given a surface with physically-based materials, we can render reflections with varying glossiness. To generate more realistic soft shadows, we further propose to use 3D-aware buffer channels to guide a neural renderer. Both quantitative and qualitative evaluations demonstrate that PixHt-Lab significantly improves soft shadow generation.

count=3
* Differentiable Shadow Mapping for Efficient Inverse Graphics
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Worchel_Differentiable_Shadow_Mapping_for_Efficient_Inverse_Graphics_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Worchel_Differentiable_Shadow_Mapping_for_Efficient_Inverse_Graphics_CVPR_2023_paper.pdf)]
    * Title: Differentiable Shadow Mapping for Efficient Inverse Graphics
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Markus Worchel, Marc Alexa
    * Abstract: We show how shadows can be efficiently generated in differentiable rendering of triangle meshes. Our central observation is that pre-filtered shadow mapping, a technique for approximating shadows based on rendering from the perspective of a light, can be combined with existing differentiable rasterizers to yield differentiable visibility information. We demonstrate at several inverse graphics problems that differentiable shadow maps are orders of magnitude faster than differentiable light transport simulation with similar accuracy -- while differentiable rasterization without shadows often fails to converge.

count=3
* NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yan_NeRF-DS_Neural_Radiance_Fields_for_Dynamic_Specular_Objects_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_NeRF-DS_Neural_Radiance_Fields_for_Dynamic_Specular_Objects_CVPR_2023_paper.pdf)]
    * Title: NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zhiwen Yan, Chen Li, Gim Hee Lee
    * Abstract: Dynamic Neural Radiance Field (NeRF) is a powerful algorithm capable of rendering photo-realistic novel view images from a monocular RGB video of a dynamic scene. Although it warps moving points across frames from the observation spaces to a common canonical space for rendering, dynamic NeRF does not model the change of the reflected color during the warping. As a result, this approach often fails drastically on challenging specular objects in motion. We address this limitation by reformulating the neural radiance field function to be conditioned on surface position and orientation in the observation space. This allows the specular surface at different poses to keep the different reflected colors when mapped to the common canonical space. Additionally, we add the mask of moving objects to guide the deformation field. As the specular surface changes color during motion, the mask mitigates the problem of failure to find temporal correspondences with only RGB supervision. We evaluate our model based on the novel view synthesis quality with a self-collected dataset of different moving specular objects in realistic environments. The experimental results demonstrate that our method significantly improves the reconstruction quality of moving specular objects from monocular RGB videos compared to the existing NeRF models. Our code and data are available at the project website https://github.com/JokerYan/NeRF-DS.

count=3
* MoSAR: Monocular Semi-Supervised Model for Avatar Reconstruction using Differentiable Shading
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Dib_MoSAR_Monocular_Semi-Supervised_Model_for_Avatar_Reconstruction_using_Differentiable_Shading_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Dib_MoSAR_Monocular_Semi-Supervised_Model_for_Avatar_Reconstruction_using_Differentiable_Shading_CVPR_2024_paper.pdf)]
    * Title: MoSAR: Monocular Semi-Supervised Model for Avatar Reconstruction using Differentiable Shading
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Abdallah Dib, Luiz Gustavo Hafemann, Emeline Got, Trevor Anderson, Amin Fadaeinejad, Rafael M. O. Cruz, Marc-André Carbonneau
    * Abstract: Reconstructing an avatar from a portrait image has many applications in multimedia but remains a challenging research problem. Extracting reflectance maps and geometry from one image is ill-posed: recovering geometry is a one-to-many mapping problem and reflectance and light are difficult to disentangle. Accurate geometry and reflectance can be captured under the controlled conditions of a light stage but it is costly to acquire large datasets in this fashion. Moreover training solely with this type of data leads to poor generalization with in-the-wild images. This motivates the introduction of MoSAR a method for 3D avatar generation from monocular images. We propose a semi-supervised training scheme that improves generalization by learning from both light stage and in-the-wild datasets. This is achieved using a novel differentiable shading formulation. We show that our approach effectively disentangles the intrinsic face parameters producing relightable avatars. As a result MoSAR estimates a richer set of skin reflectance maps and generates more realistic avatars than existing state-of-the-art methods. We also release a new dataset that provides intrinsic face attributes (diffuse specular ambient occlusion and translucency maps) for 10k subjects.

count=3
* Differentiable Micro-Mesh Construction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Dou_Differentiable_Micro-Mesh_Construction_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Dou_Differentiable_Micro-Mesh_Construction_CVPR_2024_paper.pdf)]
    * Title: Differentiable Micro-Mesh Construction
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yishun Dou, Zhong Zheng, Qiaoqiao Jin, Rui Shi, Yuhan Li, Bingbing Ni
    * Abstract: Micro-mesh (u-mesh) is a new graphics primitive for compact representation of extreme geometry consisting of a low-polygon base mesh enriched by per micro-vertex displacement. A new generation of GPUs supports this structure with hardware evolution on u-mesh ray tracing achieving real-time rendering in pixel level geometric details. In this article we present a differentiable framework to convert standard meshes into this efficient format offering a holistic scheme in contrast to the previous stage-based methods. In our construction context a u-mesh is defined where each base triangle is a parametric primitive which is then reparameterized with Laplacian operators for efficient geometry optimization. Our framework offers numerous advantages for high-quality u-mesh production: (i) end-to-end geometry optimization and displacement baking; (ii) enabling the differentiation of renderings with respect to umesh for faithful reprojectability; (iii) high scalability for integrating useful features for u-mesh production and rendering such as minimizing shell volume maintaining the isotropy of the base mesh and visual-guided adaptive level of detail. Extensive experiments on u-mesh construction for a large set of high-resolution meshes demonstrate the superior quality achieved by the proposed scheme.

count=3
* GS-IR: 3D Gaussian Splatting for Inverse Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_GS-IR_3D_Gaussian_Splatting_for_Inverse_Rendering_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_GS-IR_3D_Gaussian_Splatting_for_Inverse_Rendering_CVPR_2024_paper.pdf)]
    * Title: GS-IR: 3D Gaussian Splatting for Inverse Rendering
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, Kui Jia
    * Abstract: We propose GS-IR a novel inverse rendering approach based on 3D Gaussian Splatting (GS) that leverages forward mapping volume rendering to achieve photorealistic novel view synthesis and relighting results. Unlike previous works that use implicit neural representations and volume rendering (e.g. NeRF) which suffer from low expressive power and high computational complexity we extend GS a top-performance representation for novel view synthesis to estimate scene geometry surface material and environment illumination from multi-view images captured under unknown lighting conditions. There are two main problems when introducing GS to inverse rendering: 1) GS does not support producing plausible normal natively; 2) forward mapping (e.g. rasterization and splatting) cannot trace the occlusion like backward mapping (e.g. ray tracing). To address these challenges our GS-IR proposes an efficient optimization scheme that incorporates a depth-derivation-based regularization for normal estimation and a baking-based occlusion to model indirect lighting. The flexible and expressive GS representation allows us to achieve fast and compact geometry reconstruction photorealistic novel view synthesis and effective physically-based rendering. We demonstrate the superiority of our method over baseline methods through qualitative and quantitative evaluations on various challenging scenes.

count=3
* Dr. Bokeh: DiffeRentiable Occlusion-aware Bokeh Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sheng_Dr._Bokeh_DiffeRentiable_Occlusion-aware_Bokeh_Rendering_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sheng_Dr._Bokeh_DiffeRentiable_Occlusion-aware_Bokeh_Rendering_CVPR_2024_paper.pdf)]
    * Title: Dr. Bokeh: DiffeRentiable Occlusion-aware Bokeh Rendering
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yichen Sheng, Zixun Yu, Lu Ling, Zhiwen Cao, Xuaner Zhang, Xin Lu, Ke Xian, Haiting Lin, Bedrich Benes
    * Abstract: Bokeh is widely used in photography to draw attention to the subject while effectively isolating distractions in the background. Computational methods can simulate bokeh effects without relying on a physical camera lens but the inaccurate lens modeling in existing filtering-based methods leads to artifacts that need post-processing or learning-based methods to fix. We propose Dr.Bokeh a novel rendering method that addresses the issue by directly correcting the defect that violates the physics in the current filtering-based bokeh rendering equation. Dr.Bokeh first preprocesses the input RGBD to obtain a layered scene representation. Dr.Bokeh then takes the layered representation and user-defined lens parameters to render photo-realistic lens blur based on the novel occlusion-aware bokeh rendering method. Experiments show that the non-learning based renderer Dr.Bokeh outperforms state-of-the-art bokeh rendering algorithms in terms of photo-realism. In addition extensive quantitative and qualitative evaluations show the more accurate lens model further pushes the limit of a closely related field depth-from-defocus.

count=3
* Mip-Splatting: Alias-free 3D Gaussian Splatting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf)]
    * Title: Mip-Splatting: Alias-free 3D Gaussian Splatting
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, Andreas Geiger
    * Abstract: Recently 3D Gaussian Splatting has demonstrated impressive novel view synthesis results reaching high fidelity and efficiency. However strong artifacts can be observed when changing the sampling rate e.g. by changing focal length or camera distance. We find that the source for this phenomenon can be attributed to the lack of 3D frequency constraints and the usage of a 2D dilation filter. To address this problem we introduce a 3D smoothing filter to constrains the size of the 3D Gaussian primitives based on the maximal sampling frequency induced by the input views. It eliminates high-frequency artifacts when zooming in. Moreover replacing 2D dilation with a 2D Mip filter which simulates a 2D box filter effectively mitigates aliasing and dilation issues. Our evaluation including scenarios such a training on single-scale images and testing on multiple scales validates the effectiveness of our approach.

count=3
* Reflection Modeling for Passive Stereo
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Nair_Reflection_Modeling_for_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Nair_Reflection_Modeling_for_ICCV_2015_paper.pdf)]
    * Title: Reflection Modeling for Passive Stereo
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Rahul Nair, Andrew Fitzgibbon, Daniel Kondermann, Carsten Rother
    * Abstract: Stereo reconstruction in presence of reality faces many challenges that still need to be addressed. This paper considers reflections, which introduce incorrect matches due to the observation violating the diffuse-world assumption underlying the majority of stereo techniques. Unlike most existing work, which employ regularization or robust data terms to suppress such errors, we derive two least squares models from first principles that generalize diffuse world stereo and explicitly take reflections into account. These models are parametrized by depth, orientation and material properties, resulting in a total of up to 5 parameters per pixel that have to be estimated. Additionally large non-local interactions between viewed and reflected surface have to be taken into account. These two properties make inference of the model appear prohibitive, but we present evidence that inference is actually possible using a variant of patch match stereo.

count=3
* SceneNet RGB-D: Can 5M Synthetic Images Beat Generic ImageNet Pre-Training on Indoor Segmentation?
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/McCormac_SceneNet_RGB-D_Can_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/McCormac_SceneNet_RGB-D_Can_ICCV_2017_paper.pdf)]
    * Title: SceneNet RGB-D: Can 5M Synthetic Images Beat Generic ImageNet Pre-Training on Indoor Segmentation?
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: John McCormac, Ankur Handa, Stefan Leutenegger, Andrew J. Davison
    * Abstract: We introduce SceneNet RGB-D, a dataset providing pixel-perfect ground truth for scene understanding problems such as semantic segmentation, instance segmentation, and object detection. It also provides perfect camera poses and depth data, allowing investigation into geometric computer vision problems such as optical flow, camera pose estimation, and 3D scene labelling tasks. Random sampling permits virtually unlimited scene configurations, and here we provide 5M rendered RGB-D images from 16K randomly generated 3D trajectories in synthetic layouts, with random but physically simulated object configurations. We compare the semantic segmentation performance of network weights produced from pre-training on RGB images from our dataset against generic VGG-16 ImageNet weights. After fine-tuning on the SUN RGB-D and NYUv2 real-world datasets we find in both cases that the synthetically pre-trained network outperforms the VGG-16 weights. When synthetic pre-training includes a depth channel (something ImageNet cannot natively provide) the performance is greater still. This suggests that large-scale high-quality synthetic RGB datasets with task-specific labels can be more useful for pre-training than real-world generic pre-training such as ImageNet. We host the dataset at http://robotvault.bitbucket.io/scenenet-rgbd.html

count=3
* Efficient Global Illumination for Morphable Models
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Schneider_Efficient_Global_Illumination_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Schneider_Efficient_Global_Illumination_ICCV_2017_paper.pdf)]
    * Title: Efficient Global Illumination for Morphable Models
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Andreas Schneider, Sandro Schonborn, Lavrenti Frobeen, Bernhard Egger, Thomas Vetter
    * Abstract: We propose an efficient self-shadowing illumination model for Morphable Models. Simulating self-shadowing with ray casting is computationally expensive which makes them impractical in Analysis-by-Synthesis methods for object reconstruction from single images. Therefore, we propose to learn self-shadowing for Morphable Model parameters directly with a linear model. Radiance transfer functions are a powerful way to represent self-shadowing used within the precomputed radiance transfer framework (PRT). We build on PRT to render deforming objects with self-shadowing at interactive frame rates. It can be illuminated efficiently by environment maps represented with spherical harmonics. The result is an efficient global illumination method for Morphable Models, exploiting an approximated radiance transfer. We apply the method to fitting Morphable Model parameters to a single image of a face and demonstrate that considering self-shadowing improves shape reconstruction.

count=3
* Neural Inverse Rendering of an Indoor Scene From a Single Image
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Sengupta_Neural_Inverse_Rendering_of_an_Indoor_Scene_From_a_Single_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sengupta_Neural_Inverse_Rendering_of_an_Indoor_Scene_From_a_Single_ICCV_2019_paper.pdf)]
    * Title: Neural Inverse Rendering of an Indoor Scene From a Single Image
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Soumyadip Sengupta,  Jinwei Gu,  Kihwan Kim,  Guilin Liu,  David W. Jacobs,  Jan Kautz
    * Abstract: Inverse rendering aims to estimate physical attributes of a scene, e.g., reflectance, geometry, and lighting, from image(s). Inverse rendering has been studied primarily for single objects or with methods that solve for only one of the scene attributes. We propose the first learning based approach that jointly estimates albedo, normals, and lighting of an indoor scene from a single image. Our key contribution is the Residual Appearance Renderer (RAR), which can be trained to synthesize complex appearance effects (e.g., inter-reflection, cast shadows, near-field illumination, and realistic shading), which would be neglected otherwise. This enables us to perform self-supervised learning on real data using a reconstruction loss, based on re-synthesizing the input image from the estimated components. We finetune with real data after pretraining with synthetic data. To this end, we use physically-based rendering to create a large-scale synthetic dataset, named SUNCG-PBR, which is a significant improvement over prior datasets. Experimental results show that our approach outperforms state-of-the-art methods that estimate one or more scene attributes.

count=3
* SurfGen: Adversarial 3D Shape Synthesis With Explicit Surface Discriminators
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Luo_SurfGen_Adversarial_3D_Shape_Synthesis_With_Explicit_Surface_Discriminators_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Luo_SurfGen_Adversarial_3D_Shape_Synthesis_With_Explicit_Surface_Discriminators_ICCV_2021_paper.pdf)]
    * Title: SurfGen: Adversarial 3D Shape Synthesis With Explicit Surface Discriminators
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Andrew Luo, Tianqin Li, Wen-Hao Zhang, Tai Sing Lee
    * Abstract: Recent advances in deep generative models have led to immense progress in 3D shape synthesis. While existing models are able to synthesize shapes represented as voxels, point-clouds, or implicit functions, these methods only indirectly enforce the plausibility of the final 3D shape surface. Here we present a 3D shape synthesis framework (SurfGen) that directly applies adversarial training to the object surface. Our approach uses a differentiable spherical projection layer to capture and represent the explicit zero isosurface of an implicit 3D generator as functions defined on the unit sphere. By processing the spherical representation of 3D object surfaces with a spherical CNN in an adversarial setting, our generator can better learn the statistics of natural shape surfaces. We evaluate our model on large-scale shape datasets, and demonstrate that the end-to-end trained model is capable of generating high fidelity 3D shapes with diverse topology

count=3
* Accelerating Atmospheric Turbulence Simulation via Learned Phase-to-Space Transform
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Mao_Accelerating_Atmospheric_Turbulence_Simulation_via_Learned_Phase-to-Space_Transform_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Mao_Accelerating_Atmospheric_Turbulence_Simulation_via_Learned_Phase-to-Space_Transform_ICCV_2021_paper.pdf)]
    * Title: Accelerating Atmospheric Turbulence Simulation via Learned Phase-to-Space Transform
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zhiyuan Mao, Nicholas Chimitt, Stanley H. Chan
    * Abstract: Fast and accurate simulation of imaging through atmospheric turbulence is essential for developing turbulence mitigation algorithms. Recognizing the limitations of previous approaches, we introduce a new concept known as the phase-to-space (P2S) transform to significantly speed up the simulation. P2S is built upon three ideas: (1) reformulating the spatially varying convolution as a set of invariant convolutions with basis functions, (2) learning the basis function via the known turbulence statistics models, (3) implementing the P2S transform via a light-weight network that directly converts the phase representation to spatial representation. The new simulator offers 300x - 1000x speed up compared to the mainstream split-step simulators while preserving the essential turbulence statistics.

count=3
* Deep Implicit Surface Point Prediction Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Venkatesh_Deep_Implicit_Surface_Point_Prediction_Networks_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Venkatesh_Deep_Implicit_Surface_Point_Prediction_Networks_ICCV_2021_paper.pdf)]
    * Title: Deep Implicit Surface Point Prediction Networks
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Rahul Venkatesh, Tejan Karmali, Sarthak Sharma, Aurobrata Ghosh, R. Venkatesh Babu, László A. Jeni, Maneesh Singh
    * Abstract: Deep neural representations of 3D shapes as implicit functions have been shown to produce high fidelity models surpassing the resolution-memory trade-off faced by the explicit representations using meshes and point clouds. However, most such approaches focus on representing closed shapes. Unsigned distance function (UDF) based approaches have been proposed recently as a promising alternative to represent both open and closed shapes. However, since the gradients of UDFs vanish on the surface, it is challenging to estimate local (differential) geometric properties like the normals and tangent planes which are needed for many downstream applications in vision and graphics. There are additional challenges in computing these properties efficiently with a low-memory footprint. This paper presents a novel approach that models such surfaces using a new class of implicit representations called the closest surface-point CSP representation. We show that CSP allows us to represent complex surfaces of any topology (open or closed) with high fidelity. It also allows for accurate and efficient computation of local geometric properties. We further demonstrate that it leads to efficient implementation of downstream algorithms like sphere-tracing for rendering the 3D surface as well as to create explicit mesh-based representations. Extensive experimental evaluation on the ShapeNet dataset validate the above contributions with results surpassing the state-of-the-art. Code and data are available at https://sites.google.com/view/cspnet

count=3
* Strata-NeRF : Neural Radiance Fields for Stratified Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Dhiman_Strata-NeRF__Neural_Radiance_Fields_for_Stratified_Scenes_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Dhiman_Strata-NeRF__Neural_Radiance_Fields_for_Stratified_Scenes_ICCV_2023_paper.pdf)]
    * Title: Strata-NeRF : Neural Radiance Fields for Stratified Scenes
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ankit Dhiman, R Srinath, Harsh Rangwani, Rishubh Parihar, Lokesh R Boregowda, Srinath Sridhar, R Venkatesh Babu
    * Abstract: Neural Radiance Fields (NeRF) approaches learn the underlying 3D representation of a scene and generate photo-realistic novel views with high fidelity. However, most proposed settings concentrate on 3D modelling a single object or a single level of a scene. However, in the real world, a person captures a structure at multiple levels, resulting in layered capture. For example, tourists usually capture a monument's exterior structure before capturing the inner structure. Modelling such scenes in 3D with seamless switching between levels can drastically improve the Virtual Reality (VR) experience. However, most of the existing techniques struggle in modelling such scenes. Hence, we propose Strata-NeRF, a single radiance field that can implicitly learn the 3D representation of outer, inner, and subsequent levels. Strata-NeRF achieves this by conditioning the NeRFs on Vector Quantized (VQ) latents which allows sudden changes in scene structure with changes in levels due to their discrete nature. We first investigate the proposed approach's effectiveness by modelling a novel multilayered synthetic dataset comprising diverse scenes and then further validate its generalization on the real-world RealEstate dataset. We find that Strata-NeRF effectively models the scene structure, minimizes artefacts and synthesizes high-fidelity views compared to existing state-of-the-art approaches in the literature.

count=3
* ARNOLD: A Benchmark for Language-Grounded Task Learning with Continuous States in Realistic 3D Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Gong_ARNOLD_A_Benchmark_for_Language-Grounded_Task_Learning_with_Continuous_States_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Gong_ARNOLD_A_Benchmark_for_Language-Grounded_Task_Learning_with_Continuous_States_ICCV_2023_paper.pdf)]
    * Title: ARNOLD: A Benchmark for Language-Grounded Task Learning with Continuous States in Realistic 3D Scenes
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ran Gong, Jiangyong Huang, Yizhou Zhao, Haoran Geng, Xiaofeng Gao, Qingyang Wu, Wensi Ai, Ziheng Zhou, Demetri Terzopoulos, Song-Chun Zhu, Baoxiong Jia, Siyuan Huang
    * Abstract: Understanding the continuous states of objects is essential for task learning and planning in the real world. However, most existing task learning benchmarks assume discrete (e.g., binary) object states, which poses challenges for learning complex tasks and transferring learned policy from the simulated environment to the real world. Furthermore, the robot's ability to follow human instructions based on grounding the actions and states is limited. To tackle these challenges, we present ARNOLD, a benchmark that evaluates language-grounded task learning with continuous states in realistic 3D scenes. ARNOLD consists of 8 language-conditioned tasks that involve understanding object states and learning policies for continuous goals. To promote language-instructed learning, we provide expert demonstrations with template-generated language descriptions. We assess task performance by utilizing the latest language-conditioned policy learning models. Our results indicate that current models for language-conditioned manipulations continue to experience significant challenges when it comes to novel goal-state generalizations, scene generalizations, and object generalizations. These findings highlight the need to develop new algorithms to address this gap and underscore the potential for further research in this area.

count=3
* Neural Microfacet Fields for Inverse Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Mai_Neural_Microfacet_Fields_for_Inverse_Rendering_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Mai_Neural_Microfacet_Fields_for_Inverse_Rendering_ICCV_2023_paper.pdf)]
    * Title: Neural Microfacet Fields for Inverse Rendering
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Alexander Mai, Dor Verbin, Falko Kuester, Sara Fridovich-Keil
    * Abstract: We present Neural Microfacet Fields, a method for recovering materials, geometry (volumetric density), and environmental illumination from a collection of images of a scene. Our method applies a microfacet reflectance model within a volumetric setting by treating each sample along the ray as a surface, rather than an emitter. Using surface-based Monte Carlo rendering in a volumetric setting enables our method to perform inverse rendering efficiently and enjoy recent advances in volume rendering. Our approach obtains similar performance as state-of-the-art methods for novel view synthesis and outperforms prior work in inverse rendering, capturing high fidelity geometry and high frequency illumination details.

count=3
* DiFaReli: Diffusion Face Relighting
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Ponglertnapakorn_DiFaReli_Diffusion_Face_Relighting_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Ponglertnapakorn_DiFaReli_Diffusion_Face_Relighting_ICCV_2023_paper.pdf)]
    * Title: DiFaReli: Diffusion Face Relighting
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Puntawat Ponglertnapakorn, Nontawat Tritrong, Supasorn Suwajanakorn
    * Abstract: We present a novel approach to single-view face relighting in the wild. Handling non-diffuse effects, such as global illumination or cast shadows, has long been a challenge in face relighting. Prior work often assumes Lambertian surfaces, simplified lighting models or involves estimating 3D shape, albedo, or a shadow map. This estimation, however, is error-prone and requires many training examples with lighting ground truth to generalize well. Our work bypasses the need for accurate estimation of intrinsic components and can be trained solely on 2D images without any light stage data, multi-view images, or lighting ground truth. Our key idea is to leverage a conditional diffusion implicit model (DDIM) for decoding a disentangled light encoding along with other encodings related to 3D shape and facial identity inferred from off-the-shelf estimators. We also propose a novel conditioning technique that eases the modeling of the complex interaction between light and geometry by using a rendered shading reference to spatially modulate the DDIM. We achieve state-of-the-art performance on standard benchmark Multi-PIE and can photorealistically relight in-the-wild images. Please visit our page: https://diffusion-face-relighting.github.io.

count=3
* Factorized Inverse Path Tracing for Efficient and Accurate Material-Lighting Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Factorized_Inverse_Path_Tracing_for_Efficient_and_Accurate_Material-Lighting_Estimation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Factorized_Inverse_Path_Tracing_for_Efficient_and_Accurate_Material-Lighting_Estimation_ICCV_2023_paper.pdf)]
    * Title: Factorized Inverse Path Tracing for Efficient and Accurate Material-Lighting Estimation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Liwen Wu, Rui Zhu, Mustafa B. Yaldiz, Yinhao Zhu, Hong Cai, Janarbek Matai, Fatih Porikli, Tzu-Mao Li, Manmohan Chandraker, Ravi Ramamoorthi
    * Abstract: Inverse path tracing has recently been applied to joint material and lighting estimation, given geometry and multi-view HDR observations of an indoor scene. However, it has two major limitations: path tracing is expensive to compute, and ambiguities exist between reflection and emission. Our Factorized Inverse Path Tracing (FIPT) addresses these challenges by using a factored light transport formulation and finds emitters driven by rendering errors. Our algorithm enables accurate material and lighting optimization faster than previous work, and is more effective at resolving ambiguities. The exhaustive experiments on synthetic scenes show that our method (1) outperforms state-of-the-art indoor inverse rendering and relighting methods particularly in the presence of complex illumination effects; (2) speeds up inverse path tracing optimization to less than an hour. We further demonstrate robustness to noisy inputs through material and lighting estimates that allow plausible relighting in a real scene. The source code is available at: https://github.com/lwwu2/fipt

count=3
* Self-Annotated 3D Geometric Learning for Smeared Points Removal
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Wang_Self-Annotated_3D_Geometric_Learning_for_Smeared_Points_Removal_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Wang_Self-Annotated_3D_Geometric_Learning_for_Smeared_Points_Removal_WACV_2024_paper.pdf)]
    * Title: Self-Annotated 3D Geometric Learning for Smeared Points Removal
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Miaowei Wang, Daniel Morris
    * Abstract: There has been significant progress in improving the accuracy and quality of consumer-level dense depth sensors. Nevertheless, there remains a common depth pixel artifact which we call smeared points. These are points not on any 3D surface and typically occur as interpolations between foreground and background objects. As they cause fictitious surfaces, these points have the potential to harm applications dependent on the depth maps. Statistical outlier removal methods fare poorly in removing these points as they tend also to remove actual surface points. Trained network-based point removal faces difficulty in obtaining sufficient annotated data. To address this, we propose a fully self-annotated method to train a smeared point removal classifier. Our approach relies on gathering 3D geometric evidence from multiple perspectives to automatically detect and annotate smeared points and valid points. To validate the effectiveness of our method, we present a new benchmark dataset: the Real Azure-Kinect dataset. Experimental results and ablation studies show that our method outperforms traditional filters and other self-annotated methods. Our work is publicly available at https://github.com/wangmiaowei/wacv2024_smearedremover.git.

count=3
* Learning to Infer Implicit Surfaces without 3D Supervision
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/bdf3fd65c81469f9b74cedd497f2f9ce-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/bdf3fd65c81469f9b74cedd497f2f9ce-Paper.pdf)]
    * Title: Learning to Infer Implicit Surfaces without 3D Supervision
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Shichen Liu, Shunsuke Saito, Weikai Chen, Hao Li
    * Abstract: Recent advances in 3D deep learning have shown that it is possible to train highly effective deep models for 3D shape generation, directly from 2D images. This is particularly interesting since the availability of 3D models is still limited compared to the massive amount of accessible 2D images, which is invaluable for training. The representation of 3D surfaces itself is a key factor for the quality and resolution of the 3D output. While explicit representations, such as point clouds and voxels, can span a wide range of shape variations, their resolutions are often limited. Mesh-based representations are more efficient but are limited by their ability to handle varying topologies. Implicit surfaces, however, can robustly handle complex shapes, topologies, and also provide flexible resolution control. We address the fundamental problem of learning implicit surfaces for shape inference without the need of 3D supervision. Despite their advantages, it remains nontrivial to (1) formulate a differentiable connection between implicit surfaces and their 2D renderings, which is needed for image-based supervision; and (2) ensure precise geometric properties and control, such as local smoothness. In particular, sampling implicit surfaces densely is also known to be a computationally demanding and very slow operation. To this end, we propose a novel ray-based field probing technique for efficient image-to-field supervision, as well as a general geometric regularizer for implicit surfaces, which provides natural shape priors in unconstrained regions. We demonstrate the effectiveness of our framework on the task of single-view image-based 3D shape digitization and show how we outperform state-of-the-art techniques both quantitatively and qualitatively.

count=3
* Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/f5ac21cd0ef1b88e9848571aeb53551a-Paper.pdf)]
    * Title: Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Wenzheng Chen, Huan Ling, Jun Gao, Edward Smith, Jaakko Lehtinen, Alec Jacobson, Sanja Fidler
    * Abstract: Many machine learning models operate on images, but ignore the fact that images are 2D projections formed by 3D geometry interacting with light, in a process called rendering. Enabling ML models to understand image formation might be key for generalization. However, due to an essential rasterization step involving discrete assignment operations, rendering pipelines are non-differentiable and thus largely inaccessible to gradient-based ML techniques. In this paper, we present DIB-Render, a novel rendering framework through which gradients can be analytically computed. Key to our approach is to view rasterization as a weighted interpolation, allowing image gradients to back-propagate through various standard vertex shaders within a single framework. Our approach supports optimizing over vertex positions, colors, normals, light directions and texture coordinates, and allows us to incorporate various well-known lighting models from graphics. We showcase our approach in two ML applications: single-image 3D object prediction, and 3D textured object generation, both trained using exclusively 2D supervision.

count=2
* A Theory of Refractive Photo-Light-Path Triangulation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Chari_A_Theory_of_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Chari_A_Theory_of_2013_CVPR_paper.pdf)]
    * Title: A Theory of Refractive Photo-Light-Path Triangulation
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Visesh Chari, Peter Sturm
    * Abstract: 3D reconstruction of transparent refractive objects like a plastic bottle is challenging: they lack appearance related visual cues and merely reflect and refract light from the surrounding environment. Amongst several approaches to reconstruct such objects, the seminal work of Light-Path triangulation [17] is highly popular because of its general applicability and analysis of minimal scenarios. A lightpath is defined as the piece-wise linear path taken by a ray of light as it passes from source, through the object and into the camera. Transparent refractive objects not only affect the geometric configuration of light-paths but also their radiometric properties. In this paper, we describe a method that combines both geometric and radiometric information to do reconstruction. We show two major consequences of the addition of radiometric cues to the light-path setup. Firstly, we extend the case of scenarios in which reconstruction is plausible while reducing the minimal requirements for a unique reconstruction. This happens as a consequence of the fact that radiometric cues add an additional known variable to the already existing system of equations. Secondly, we present a simple algorithm for reconstruction, owing to the nature of the radiometric cue. We present several synthetic experiments to validate our theories, and show high quality reconstructions in challenging scenarios.

count=2
* Improving the Visual Comprehension of Point Sets
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Katz_Improving_the_Visual_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Katz_Improving_the_Visual_2013_CVPR_paper.pdf)]
    * Title: Improving the Visual Comprehension of Point Sets
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Sagi Katz, Ayellet Tal
    * Abstract: Point sets are the standard output of many 3D scanning systems and depth cameras. Presenting the set of points as is, might "hide" the prominent features of the object from which the points are sampled. Our goal is to reduce the number of points in a point set, for improving the visual comprehension from a given viewpoint. This is done by controlling the density of the reduced point set, so as to create bright regions (low density) and dark regions (high density), producing an effect of shading. This data reduction is achieved by leveraging a limitation of a solution to the classical problem of determining visibility from a viewpoint. In addition, we introduce a new dual problem, for determining visibility of a point from infinity, and show how a limitation of its solution can be leveraged in a similar way.

count=2
* Recovering Transparent Shape From Time-Of-Flight Distortion
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Tanaka_Recovering_Transparent_Shape_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Tanaka_Recovering_Transparent_Shape_CVPR_2016_paper.pdf)]
    * Title: Recovering Transparent Shape From Time-Of-Flight Distortion
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Kenichiro Tanaka, Yasuhiro Mukaigawa, Hiroyuki Kubo, Yasuyuki Matsushita, Yasushi Yagi
    * Abstract: This paper presents a method for recovering shape and normal of a transparent object from a single viewpoint using a Time-of-Flight (ToF) camera. Our method is built upon the fact that the speed of light varies with the refractive index of the medium and therefore the depth measurement of a transparent object with a ToF camera may be distorted. We show that, from this ToF distortion, the refractive light path can be uniquely determined by estimating a single parameter. We estimate this parameter by introducing a surface normal consistency between the one determined by a light path candidate and the other computed from the corresponding shape. The proposed method is evaluated by both simulation and real-world experiments and shows faithful transparent shape recovery.

count=2
* The Surfacing of Multiview 3D Drawings via Lofting and Occlusion Reasoning
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Usumezbas_The_Surfacing_of_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Usumezbas_The_Surfacing_of_CVPR_2017_paper.pdf)]
    * Title: The Surfacing of Multiview 3D Drawings via Lofting and Occlusion Reasoning
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Anil Usumezbas, Ricardo Fabbri, Benjamin B. Kimia
    * Abstract: The three-dimensional reconstruction of scenes from multiple views has made impressive strides in recent years, chiefly by methods correlating isolated feature points, intensities, or curvilinear structure. In the general setting, i.e., without requiring controlled acquisition, limited number of objects, abundant patterns on objects, or object curves to follow particular models, the majority of these methods produce unorganized point clouds, meshes, or voxel representations of the reconstructed scene, with some exceptions producing 3D drawings as networks of curves. Many applications, e.g., robotics, urban planning, industrial design, and hard surface modeling, however, require structured representations which make explicit 3D curves, surfaces, and their spatial relationships. Reconstructing surface representations can now be constrained by the 3D drawing acting like a scaffold to hang on the computed representations, leading to increased robustness and quality of reconstruction. This paper presents one way of completing such 3D drawings with surface reconstructions, by exploring occlusion reasoning through lofting algorithms.

count=2
* Learning 3D Shape Completion From Laser Scan Data With Weak Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Stutz_Learning_3D_Shape_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Stutz_Learning_3D_Shape_CVPR_2018_paper.pdf)]
    * Title: Learning 3D Shape Completion From Laser Scan Data With Weak Supervision
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: David Stutz, Andreas Geiger
    * Abstract: 3D shape completion from partial point clouds is a fundamental problem in computer vision and computer graphics. Recent approaches can be characterized as either data-driven or learning-based. Data-driven approaches rely on a shape model whose parameters are optimized to fit the observations. Learning-based approaches, in contrast, avoid the expensive optimization step and instead directly predict the complete shape from the incomplete observations using deep neural networks. However, full supervision is required which is often not available in practice. In this work, we propose a weakly-supervised learning-based approach to 3D shape completion which neither requires slow optimization nor direct supervision. While we also learn a shape prior on synthetic data, we amortize, i.e., learn, maximum likelihood fitting using deep neural networks resulting in efficient shape completion without sacrificing accuracy. Tackling 3D shape completion of cars on ShapeNet and KITTI, we demonstrate that the proposed amortized maximum likelihood approach is able to compete with a fully supervised baseline and a state-of-the-art data-driven approach while being significantly faster. On ModelNet, we additionally show that the approach is able to generalize to other object categories as well.

count=2
* Learning Biomimetic Perception for Human Sensorimotor Control
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w39/html/Nakada_Learning_Biomimetic_Perception_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w39/Nakada_Learning_Biomimetic_Perception_CVPR_2018_paper.pdf)]
    * Title: Learning Biomimetic Perception for Human Sensorimotor Control
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Masaki Nakada, Honglin Chen, Demetri Terzopoulos
    * Abstract: We introduce a biomimetic simulation framework for human perception and sensorimotor control. Our framework features a biomechanically simulated musculoskeletal human model actuated by numerous skeletal muscles, with two human-like eyes whose retinas contain spatially nonuniform distributions of photoreceptors. Its prototype sensorimotor system comprises a set of 20 automatically-trained deep neural networks (DNNs), half of which comprise the neuromuscular motor control subsystem, whereas the other half are devoted to the visual perception subsystem. Directly from the photoreceptor responses, 2 perception DNNs control eye and head movements, while 8 DNNs extract the perceptual information needed to control the arms and legs. Thus, driven exclusively by its egocentric, active visual perception, our virtual human is capable of learning efficient, online visuomotor control of its eyes, head, and four limbs to perform a nontrivial task involving the foveation and visual persuit of a moving target object coupled with visuallyguided reaching actions to intercept the incoming target.

count=2
* Deep Defocus Map Estimation Using Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Deep_Defocus_Map_Estimation_Using_Domain_Adaptation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Deep_Defocus_Map_Estimation_Using_Domain_Adaptation_CVPR_2019_paper.pdf)]
    * Title: Deep Defocus Map Estimation Using Domain Adaptation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Junyong Lee,  Sungkil Lee,  Sunghyun Cho,  Seungyong Lee
    * Abstract: In this paper, we propose the first end-to-end convolutional neural network (CNN) architecture, Defocus Map Estimation Network (DMENet), for spatially varying defocus map estimation. To train the network, we produce a novel depth-of-field (DOF) dataset, SYNDOF, where each image is synthetically blurred with a ground-truth depth map. Due to the synthetic nature of SYNDOF, the feature characteristics of images in SYNDOF can differ from those of real defocused photos. To address this gap, we use domain adaptation that transfers the features of real defocused photos into those of synthetically blurred ones. Our DMENet consists of four subnetworks: blur estimation, domain adaptation, content preservation, and sharpness calibration networks. The subnetworks are connected to each other and jointly trained with their corresponding supervisions in an end-to-end manner. Our method is evaluated on publicly available blur detection and blur estimation datasets and the results show the state-of-the-art performance.In this paper, we propose the first end-to-end convolutional neural network (CNN) architecture, Defocus Map Estimation Network (DMENet), for spatially varying defocus map estimation. To train the network, we produce a novel depth-of-field (DOF) dataset, SYNDOF, where each image is synthetically blurred with a ground-truth depth map. Due to the synthetic nature of SYNDOF, the feature characteristics of images in SYNDOF can differ from those of real defocused photos. To address this gap, we use domain adaptation that transfers the features of real defocused photos into those of synthetically blurred ones. Our DMENet consists of four subnetworks: blur estimation, domain adaptation, content preservation, and sharpness calibration networks. The subnetworks are connected to each other and jointly trained with their corresponding supervisions in an end-to-end manner. Our method is evaluated on publicly available blur detection and blur estimation datasets and the results show the state-of-the-art performance.

count=2
* DeepSDF
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf)]
    * Title: DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jeong Joon Park,  Peter Florence,  Julian Straub,  Richard Newcombe,  Steven Lovegrove
    * Abstract: Computer graphics, 3D computer vision and robotics communities have produced multiple approaches to representing 3D geometry for rendering and reconstruction. These provide trade-offs across fidelity, efficiency and compression capabilities. In this work, we introduce DeepSDF, a learned continuous Signed Distance Function (SDF) representation of a class of shapes that enables high quality shape representation, interpolation and completion from partial and noisy 3D input data. DeepSDF, like its classical counterpart, represents a shape's surface by a continuous volumetric field: the magnitude of a point in the field represents the distance to the surface boundary and the sign indicates whether the region is inside (-) or outside (+) of the shape, hence our representation implicitly encodes a shape's boundary as the zero-level-set of the learned function while explicitly representing the classification of space as being part of the shapes interior or not. While classical SDF's both in analytical or discretized voxel form typically represent the surface of a single shape, DeepSDF can represent an entire class of shapes. Furthermore, we show state-of-the-art performance for learned 3D shape representation and completion while reducing the model size by an order of magnitude compared with previous work.

count=2
* Beyond Volumetric Albedo -- A Surface Optimization Framework for Non-Line-Of-Sight Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Tsai_Beyond_Volumetric_Albedo_--_A_Surface_Optimization_Framework_for_Non-Line-Of-Sight_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tsai_Beyond_Volumetric_Albedo_--_A_Surface_Optimization_Framework_for_Non-Line-Of-Sight_CVPR_2019_paper.pdf)]
    * Title: Beyond Volumetric Albedo -- A Surface Optimization Framework for Non-Line-Of-Sight Imaging
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Chia-Yin Tsai,  Aswin C. Sankaranarayanan,  Ioannis Gkioulekas
    * Abstract: Non-line-of-sight (NLOS) imaging is the problem of reconstructing properties of scenes occluded from a sensor, using measurements of light that indirectly travels from the occluded scene to the sensor through intermediate diffuse reflections. We introduce an analysis-by-synthesis framework that can reconstruct complex shape and reflectance of an NLOS object. Our framework deviates from prior work on NLOS reconstruction, by directly optimizing for a surface representation of the NLOS object, in place of commonly employed volumetric representations. At the core of our framework is a new rendering formulation that efficiently computes derivatives of radiometric measurements with respect to NLOS geometry and reflectance, while accurately modeling the underlying light transport physics. By coupling this with stochastic optimization and geometry processing techniques, we are able to reconstruct NLOS surface at a level of detail significantly exceeding what is possible with previous volumetric reconstruction methods.

count=2
* Intelligent Home 3D: Automatic 3D-House Design From Linguistic Descriptions Only
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Intelligent_Home_3D_Automatic_3D-House_Design_From_Linguistic_Descriptions_Only_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Intelligent_Home_3D_Automatic_3D-House_Design_From_Linguistic_Descriptions_Only_CVPR_2020_paper.pdf)]
    * Title: Intelligent Home 3D: Automatic 3D-House Design From Linguistic Descriptions Only
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Qi Chen,  Qi Wu,  Rui Tang,  Yuhan Wang,  Shuai Wang,  Mingkui Tan
    * Abstract: Home design is a complex task that normally requires architects to finish with their professional skills and tools. It will be fascinating that if one can produce a house plan intuitively without knowing much knowledge about home design and experience of using complex designing tools, for example, via natural language. In this paper, we formulate it as a language conditioned visual content generation problem that is further divided into a floor plan generation and an interior texture (such as floor and wall) synthesis task. The only control signal of the generation process is the linguistic expression given by users that describe the house details. To this end, we propose a House Plan Generative Model (HPGM) that first translates the language input to a structural graph representation and then predicts the layout of rooms with a Graph Conditioned Layout Prediction Network (GC-LPN) and generates the interior texture with a Language Conditioned Texture GAN (LCT-GAN). With some post-processing, the final product of this task is a 3D house model. To train and evaluate our model, we build the first Text--to--3D House Model dataset, which will be released at: https:// hidden-link-for-submission.

count=2
* Scene Recomposition by Learning-Based ICP
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Izadinia_Scene_Recomposition_by_Learning-Based_ICP_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Izadinia_Scene_Recomposition_by_Learning-Based_ICP_CVPR_2020_paper.pdf)]
    * Title: Scene Recomposition by Learning-Based ICP
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Hamid Izadinia,  Steven M. Seitz
    * Abstract: By moving a depth sensor around a room, we compute a 3D CAD model of the environment, capturing the room shape and contents such as chairs, desks, sofas, and tables. Rather than reconstructing geometry, we match, place, and align each object in the scene to thousands of CAD models of objects. In addition to the fully automatic system, the key technical contribution is a novel approach for aligning CAD models to 3D scans, based on deep reinforcement learning. This approach, which we call Learning-based ICP, outperforms prior ICP methods in the literature, by learning the best points to match and conditioning on object viewpoint. LICP learns to align using only synthetic data and does not require ground truth annotation of object pose or keypoint pair matching in real scene scans. While LICP is trained on synthetic data and without 3D real scene annotations, it outperforms both learned local deep feature matching and geometric based alignment methods in real scenes. The proposed method is evaluated on real scenes datasets of SceneNN and ScanNet as well as synthetic scenes of SUNCG. High quality results are demonstrated on a range of real world scenes, with robustness to clutter, viewpoint, and occlusion.

count=2
* Geometric Structure Based and Regularized Depth Estimation From 360 Indoor Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_Geometric_Structure_Based_and_Regularized_Depth_Estimation_From_360_Indoor_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Geometric_Structure_Based_and_Regularized_Depth_Estimation_From_360_Indoor_CVPR_2020_paper.pdf)]
    * Title: Geometric Structure Based and Regularized Depth Estimation From 360 Indoor Imagery
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Lei Jin,  Yanyu Xu,  Jia Zheng,  Junfei Zhang,  Rui Tang,  Shugong Xu,  Jingyi Yu,  Shenghua Gao
    * Abstract: Motivated by the correlation between the depth and the geometric structure of a 360 indoor image, we propose a novel learning-based depth estimation framework that leverages the geometric structure of a scene to conduct depth estimation. Specifically, we represent the geometric structure of an indoor scene as a collection of corners, boundaries and planes. On the one hand, once a depth map is estimated, this geometric structure can be inferred from the estimated depth map; thus, the geometric structure functions as a regularizer for depth estimation. On the other hand, this estimation also benefits from the geometric structure of a scene estimated from an image where the structure functions as a prior. However, furniture in indoor scenes makes it challenging to infer geometric structure from depth or image data. An attention map is inferred to facilitate both depth estimation from features of the geometric structure and also geometric inferences from the estimated depth map. To validate the effectiveness of each component in our framework under controlled conditions, we render a synthetic dataset, Shanghaitech-Kujiale Indoor 360 dataset with 3550 360 indoor images. Extensive experiments on popular datasets validate the effectiveness of our solution. We also demonstrate that our method can also be applied to counterfactual depth.

count=2
* Autolabeling 3D Objects With Differentiable Rendering of SDF Shape Priors
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zakharov_Autolabeling_3D_Objects_With_Differentiable_Rendering_of_SDF_Shape_Priors_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zakharov_Autolabeling_3D_Objects_With_Differentiable_Rendering_of_SDF_Shape_Priors_CVPR_2020_paper.pdf)]
    * Title: Autolabeling 3D Objects With Differentiable Rendering of SDF Shape Priors
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Sergey Zakharov,  Wadim Kehl,  Arjun Bhargava,  Adrien Gaidon
    * Abstract: We present an automatic annotation pipeline to recover 9D cuboids and 3D shapes from pre-trained off-the-shelf 2D detectors and sparse LIDAR data. Our autolabeling method solves an ill-posed inverse problem by considering learned shape priors and optimizing geometric and physical parameters. To address this challenging problem, we apply a novel differentiable shape renderer to signed distance fields (SDF), leveraged together with normalized object coordinate spaces (NOCS). Initially trained on synthetic data to predict shape and coordinates, our method uses these predictions for projective and geometric alignment over real samples. Moreover, we also propose a curriculum learning strategy, iteratively retraining on samples of increasing difficulty in subsequent self-improving annotation rounds. Our experiments on the KITTI3D dataset show that we can recover a substantial amount of accurate cuboids, and that these autolabels can be used to train 3D vehicle detectors with state-of-the-art results.

count=2
* NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Srinivasan_NeRV_Neural_Reflectance_and_Visibility_Fields_for_Relighting_and_View_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Srinivasan_NeRV_Neural_Reflectance_and_Visibility_Fields_for_Relighting_and_View_CVPR_2021_paper.pdf)]
    * Title: NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Pratul P. Srinivasan, Boyang Deng, Xiuming Zhang, Matthew Tancik, Ben Mildenhall, Jonathan T. Barron
    * Abstract: We present a method that takes as input a set of images of a scene illuminated by unconstrained known lighting, and produces as output a 3D representation that can be rendered from novel viewpoints under arbitrary lighting conditions. Our method represents the scene as a continuous volumetric function parameterized as MLPs whose inputs are a 3D location and whose outputs are the following scene properties at that input location: volume density, surface normal, material parameters, distance to the first surface intersection in any direction, and visibility of the external environment in any direction. Together, these allow us to render novel views of the object under arbitrary lighting, including indirect illumination effects. The predicted visibility and surface intersection fields are critical to our model's ability to simulate direct and indirect illumination during training, because the brute-force techniques used by prior work are intractable for lighting conditions outside of controlled setups with a single light. Our method outperforms alternative approaches for recovering relightable 3D scene representations, and performs well in complex lighting settings that have posed a significant challenge to prior work.

count=2
* Inferring CAD Modeling Sequences Using Zone Graphs
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_Inferring_CAD_Modeling_Sequences_Using_Zone_Graphs_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Inferring_CAD_Modeling_Sequences_Using_Zone_Graphs_CVPR_2021_paper.pdf)]
    * Title: Inferring CAD Modeling Sequences Using Zone Graphs
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Xianghao Xu, Wenzhe Peng, Chin-Yi Cheng, Karl D.D. Willis, Daniel Ritchie
    * Abstract: In computer-aided design (CAD), the ability to "reverse engineer" the modeling steps used to create 3D shapes is a long-sought-after goal. This process can be decomposed into two sub-problems: converting an input mesh or point cloud into a boundary representation (or B-rep), and then inferring modeling operations which construct this B-rep. In this paper, we present a new system for solving the second sub-problem. Central to our approach is a new geometric representation: the zone graph. Zones are the set of solid regions formed by extending all B-Rep faces and partitioning space with them; a zone graph has these zones as its nodes, with edges denoting geometric adjacencies between them. Zone graphs allow us to tractably work with industry-standard CAD operations, unlike prior work using CSG with parametric primitives. We focus on CAD programs consisting of sketch + extrude + Boolean operations, which are common in CAD practice. We phrase our problem as search in the space of such extrusions permitted by the zone graph, and we train a graph neural network to score potential extrusions in order to accelerate the search. We show that our approach outperforms an existing CSG inference baseline in terms of geometric reconstruction accuracy and reconstruction time, while also creating more plausible modeling sequences.

count=2
* PhySG: Inverse Rendering With Spherical Gaussians for Physics-Based Material Editing and Relighting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_PhySG_Inverse_Rendering_With_Spherical_Gaussians_for_Physics-Based_Material_Editing_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_PhySG_Inverse_Rendering_With_Spherical_Gaussians_for_Physics-Based_Material_Editing_CVPR_2021_paper.pdf)]
    * Title: PhySG: Inverse Rendering With Spherical Gaussians for Physics-Based Material Editing and Relighting
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, Noah Snavely
    * Abstract: We present an end-to-end inverse rendering pipeline that includes a fully differentiable renderer, and can reconstruct geometry, materials, and illumination from scratch from a set of images. Our rendering framework represents specular BRDFs and environmental illumination using mixtures of spherical Gaussians, and represents geometry as a signed distance function parameterized as a Multi-Layer Perceptron. The use of spherical Gaussians allows us to efficiently solve for approximate light transport, and our method works on scenes with challenging non-Lambertian reflectance captured under natural, static illumination. We demonstrate, with both synthetic and real data, that our reconstruction not only can render novel viewpoints, but also enables physics-based appearance editing of materials and illumination.

count=2
* RGB-D Local Implicit Function for Depth Completion of Transparent Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_RGB-D_Local_Implicit_Function_for_Depth_Completion_of_Transparent_Objects_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_RGB-D_Local_Implicit_Function_for_Depth_Completion_of_Transparent_Objects_CVPR_2021_paper.pdf)]
    * Title: RGB-D Local Implicit Function for Depth Completion of Transparent Objects
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Luyang Zhu, Arsalan Mousavian, Yu Xiang, Hammad Mazhar, Jozef van Eenbergen, Shoubhik Debnath, Dieter Fox
    * Abstract: Majority of the perception methods in robotics require depth information provided by RGB-D cameras. However, standard 3D sensors fail to capture depth of transparent objects due to refraction and absorption of light. In this paper, we introduce a new approach for depth completion of transparent objects from a single RGB-D image. Key to our approach is a local implicit neural representation built on ray-voxel pairs that allows our method to generalize to unseen objects and achieve fast inference speed. Based on this representation, we present a novel framework that can complete missing depth given noisy RGB-D input. We further improve the depth estimation iteratively using a self-correcting refinement model. To train the whole pipeline, we build a large scale synthetic dataset with transparent objects. Experiments demonstrate that our method performs significantly better than the current state-of-the-art methods on both synthetic and real world data. In addition, our approach improves the inference speed by a factor of 20 compared to the previous best method, ClearGrasp. Code will be released at https://research.nvidia.com/publication/2021-03_RGB-D-Local-Implicit.

count=2
* A Monocular Pose Estimation Case Study: The Hayabusa2 Minerva-II2 Deployment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Price_A_Monocular_Pose_Estimation_Case_Study_The_Hayabusa2_Minerva-II2_Deployment_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Price_A_Monocular_Pose_Estimation_Case_Study_The_Hayabusa2_Minerva-II2_Deployment_CVPRW_2021_paper.pdf)]
    * Title: A Monocular Pose Estimation Case Study: The Hayabusa2 Minerva-II2 Deployment
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Andrew Price, Kazuya Yoshida
    * Abstract: In an environment of increasing orbital debris and remote operation, visual data acquisition methods are becoming a core competency of the next generation of spacecraft. However, deep space missions often generate limited data and noisy images, necessitating complex data analysis methods. Here, a state-of-the-art convolutional neural network (CNN) pose estimation pipeline is applied to the Hayabusa2 Minerva-II2 rover deployment; a challenging case with noisy images and a symmetric target. To enable training of this CNN, a custom dataset is created. The deployment velocity is estimated as 0.1908 m/s using a projective geometry approach and 0.1934 m/s using a CNN landmark detector approach, as compared to the official JAXA estimation of 0.1924 m/s (relative to the spacecraft). Additionally, the attitude estimation results from the real deployment images are shared and the associated tumble estimation is discussed.

count=2
* FS6D: Few-Shot 6D Pose Estimation of Novel Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/He_FS6D_Few-Shot_6D_Pose_Estimation_of_Novel_Objects_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/He_FS6D_Few-Shot_6D_Pose_Estimation_of_Novel_Objects_CVPR_2022_paper.pdf)]
    * Title: FS6D: Few-Shot 6D Pose Estimation of Novel Objects
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yisheng He, Yao Wang, Haoqiang Fan, Jian Sun, Qifeng Chen
    * Abstract: 6D object pose estimation networks are limited in their capability to scale to large numbers of object instances due to the close-set assumption and their reliance on high-fidelity object CAD models. In this work, we study a new open set problem; the few-shot 6D object poses estimation: estimating the 6D pose of an unknown object by a few support views without extra training. To tackle the problem, we point out the importance of fully exploring the appearance and geometric relationship between the given support views and query scene patches and propose a dense prototypes matching framework by extracting and matching dense RGBD prototypes with transformers. Moreover, we show that the priors from diverse appearances and shapes are crucial to the generalization capability under the problem setting and thus propose a large-scale RGBD photorealistic dataset (ShapeNet6D) for network pre-training. A simple and effective online texture blending approach is also introduced to eliminate the domain gap from the synthesis dataset, which enriches appearance diversity at a low cost. Finally, we discuss possible solutions to this problem and establish benchmarks on popular datasets to facilitate future research.

count=2
* Investigating the Impact of Multi-LiDAR Placement on Object Detection for Autonomous Driving
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Investigating_the_Impact_of_Multi-LiDAR_Placement_on_Object_Detection_for_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Investigating_the_Impact_of_Multi-LiDAR_Placement_on_Object_Detection_for_CVPR_2022_paper.pdf)]
    * Title: Investigating the Impact of Multi-LiDAR Placement on Object Detection for Autonomous Driving
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Hanjiang Hu, Zuxin Liu, Sharad Chitlangia, Akhil Agnihotri, Ding Zhao
    * Abstract: The past few years have witnessed an increasing interest in improving the perception performance of LiDARs on autonomous vehicles. While most of the existing works focus on developing new deep learning algorithms or model architectures, we study the problem from the physical design perspective, i.e., how different placements of multiple LiDARs influence the learning-based perception. To this end, we introduce an easy-to-compute information-theoretic surrogate metric to quantitatively and fast evaluate LiDAR placement for 3D detection of different types of objects. We also present a new data collection, detection model training and evaluation framework in the realistic CARLA simulator to evaluate disparate multi-LiDAR configurations. Using several prevalent placements inspired by the designs of self-driving companies, we show the correlation between our surrogate metric and object detection performance of different representative algorithms on KITTI through extensive experiments, validating the effectiveness of our LiDAR placement evaluation approach. Our results show that sensor placement is non-negligible in 3D point cloud-based object detection, which will contribute to 5% 10% performance discrepancy in terms of average precision in challenging 3D object detection settings. We believe that this is one of the first studies to quantitatively investigate the influence of LiDAR placement on perception performance.

count=2
* NeRF in the Dark: High Dynamic Range View Synthesis From Noisy Raw Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Mildenhall_NeRF_in_the_Dark_High_Dynamic_Range_View_Synthesis_From_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Mildenhall_NeRF_in_the_Dark_High_Dynamic_Range_View_Synthesis_From_CVPR_2022_paper.pdf)]
    * Title: NeRF in the Dark: High Dynamic Range View Synthesis From Noisy Raw Images
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P. Srinivasan, Jonathan T. Barron
    * Abstract: Neural Radiance Fields (NeRF) is a technique for high quality novel view synthesis from a collection of posed input images. Like most view synthesis methods, NeRF uses tonemapped low dynamic range (LDR) as input; these images have been processed by a lossy camera pipeline that smooths detail, clips highlights, and distorts the simple noise distribution of raw sensor data. We modify NeRF to instead train directly on linear raw images, preserving the scene's full dynamic range. By rendering raw output images from the resulting NeRF, we can perform novel high dynamic range (HDR) view synthesis tasks. In addition to changing the camera viewpoint, we can manipulate focus, exposure, and tonemapping after the fact. Although a single raw image appears significantly more noisy than a postprocessed one, we show that NeRF is highly robust to the zero-mean distribution of raw noise. When optimized over many noisy raw inputs (25-200), NeRF produces a scene representation so accurate that its rendered novel views outperform dedicated single and multi-image deep raw denoisers run on the same wide baseline input images. As a result, our method, which we call RawNeRF, can reconstruct scenes from extremely noisy images captured in near-darkness.

count=2
* CrossLoc: Scalable Aerial Localization Assisted by Multimodal Synthetic Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yan_CrossLoc_Scalable_Aerial_Localization_Assisted_by_Multimodal_Synthetic_Data_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yan_CrossLoc_Scalable_Aerial_Localization_Assisted_by_Multimodal_Synthetic_Data_CVPR_2022_paper.pdf)]
    * Title: CrossLoc: Scalable Aerial Localization Assisted by Multimodal Synthetic Data
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Qi Yan, Jianhao Zheng, Simon Reding, Shanci Li, Iordan Doytchinov
    * Abstract: We present a visual localization system that learns to estimate camera poses in the real world with the help of synthetic data. Despite significant progress in recent years, most learning-based approaches to visual localization target at a single domain and require a dense database of geo-tagged images to function well. To mitigate the data scarcity issue and improve the scalability of the neural localization models, we introduce TOPO-DataGen, a versatile synthetic data generation tool that traverses smoothly between the real and virtual world, hinged on the geographic camera viewpoint. New large-scale sim-to-real benchmark datasets are proposed to showcase and evaluate the utility of the said synthetic data. Our experiments reveal that synthetic data generically enhances the neural network performance on real data. Furthermore, we introduce CrossLoc, a cross-modal visual representation learning approach to pose estimation that makes full use of the scene coordinate ground truth via self-supervision. Without any extra data, CrossLoc significantly outperforms the state-of-the-art methods and achieves substantially higher real-data sample efficiency. Our code and datasets are all available at crossloc.github.io.

count=2
* IRON: Inverse Rendering by Optimizing Neural SDFs and Materials From Photometric Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_IRON_Inverse_Rendering_by_Optimizing_Neural_SDFs_and_Materials_From_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_IRON_Inverse_Rendering_by_Optimizing_Neural_SDFs_and_Materials_From_CVPR_2022_paper.pdf)]
    * Title: IRON: Inverse Rendering by Optimizing Neural SDFs and Materials From Photometric Images
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Kai Zhang, Fujun Luan, Zhengqi Li, Noah Snavely
    * Abstract: We propose a neural inverse rendering pipeline called IRON that operates on photometric images and outputs high-quality 3D content in the format of triangle meshes and material textures readily deployable in existing graphics pipelines. We propose a neural inverse rendering pipeline called IRON that operates on photometric images and outputs high-quality 3D content in the format of triangle meshes and material textures readily deployable in existing graphics pipelines. Our method adopts neural representations for geometry as signed distance fields (SDFs) and materials during optimization to enjoy their flexibility and compactness, and features a hybrid optimization scheme for neural SDFs: first, optimize using a volumetric radiance field approach to recover correct topology, then optimize further using edge-aware physics-based surface rendering for geometry refinement and disentanglement of materials and lighting. In the second stage, we also draw inspiration from mesh-based differentiable rendering, and design a novel edge sampling algorithm for neural SDFs to further improve performance. We show that our IRON achieves significantly better inverse rendering quality compared to prior works.

count=2
* A New Non-Central Model for Fisheye Calibration
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/OmniCV/html/Tezaur_A_New_Non-Central_Model_for_Fisheye_Calibration_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/OmniCV/papers/Tezaur_A_New_Non-Central_Model_for_Fisheye_Calibration_CVPRW_2022_paper.pdf)]
    * Title: A New Non-Central Model for Fisheye Calibration
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Radka Tezaur, Avinash Kumar, Oscar Nestares
    * Abstract: A new non-central model suitable for calibrating fisheye cameras is proposed. It is a direct extension of the popular central model developed by Scaramuzza et al., used by Matlab Computer Vision Toolbox fisheye calibration tool. It allows adapting existing applications that are using this central model to a non-central projection that is more accurate, especially when objects captured in the images are close to the camera, and it makes it possible to switch easily between the more accurate non-central characterization of the fisheye camera and the more convenient central approximation, as needed. It is shown that the algorithms proposed by Scaramuzza et al. for their central model can be modified to accommodate the angle dependent axial viewpoint shift. This means, besides other, that a similar process can be used for calibration involving the viewpoint shift characterization and a user-friendly calibration tool can be produced with this new non-central model that does not require the user to provide detailed lens design specifications or an educated guess for the initial parameter values. Several other improvements to the Scaramuzza's central model are also introduced, helping to improve the performance of both the central model, and its non-central extension.

count=2
* Neural Kaleidoscopic Space Sculpting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ahn_Neural_Kaleidoscopic_Space_Sculpting_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Ahn_Neural_Kaleidoscopic_Space_Sculpting_CVPR_2023_paper.pdf)]
    * Title: Neural Kaleidoscopic Space Sculpting
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Byeongjoo Ahn, Michael De Zeeuw, Ioannis Gkioulekas, Aswin C. Sankaranarayanan
    * Abstract: We introduce a method that recovers full-surround 3D reconstructions from a single kaleidoscopic image using a neural surface representation. Full-surround 3D reconstruction is critical for many applications, such as augmented and virtual reality. A kaleidoscope, which uses a single camera and multiple mirrors, is a convenient way of achieving full-surround coverage, as it redistributes light directions and thus captures multiple viewpoints in a single image. This enables single-shot and dynamic full-surround 3D reconstruction. However, using a kaleidoscopic image for multi-view stereo is challenging, as we need to decompose the image into multi-view images by identifying which pixel corresponds to which virtual camera, a process we call labeling. To address this challenge, pur approach avoids the need to explicitly estimate labels, but instead "sculpts" a neural surface representation through the careful use of silhouette, background, foreground, and texture information present in the kaleidoscopic image. We demonstrate the advantages of our method in a range of simulated and real experiments, on both static and dynamic scenes.

count=2
* HandNeRF: Neural Radiance Fields for Animatable Interacting Hands
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Guo_HandNeRF_Neural_Radiance_Fields_for_Animatable_Interacting_Hands_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_HandNeRF_Neural_Radiance_Fields_for_Animatable_Interacting_Hands_CVPR_2023_paper.pdf)]
    * Title: HandNeRF: Neural Radiance Fields for Animatable Interacting Hands
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zhiyang Guo, Wengang Zhou, Min Wang, Li Li, Houqiang Li
    * Abstract: We propose a novel framework to reconstruct accurate appearance and geometry with neural radiance fields (NeRF) for interacting hands, enabling the rendering of photo-realistic images and videos for gesture animation from arbitrary views. Given multi-view images of a single hand or interacting hands, an off-the-shelf skeleton estimator is first employed to parameterize the hand poses. Then we design a pose-driven deformation field to establish correspondence from those different poses to a shared canonical space, where a pose-disentangled NeRF for one hand is optimized. Such unified modeling efficiently complements the geometry and texture cues in rarely-observed areas for both hands. Meanwhile, we further leverage the pose priors to generate pseudo depth maps as guidance for occlusion-aware density learning. Moreover, a neural feature distillation method is proposed to achieve cross-domain alignment for color optimization. We conduct extensive experiments to verify the merits of our proposed HandNeRF and report a series of state-of-the-art results both qualitatively and quantitatively on the large-scale InterHand2.6M dataset.

count=2
* NeFII: Inverse Rendering for Reflectance Decomposition With Near-Field Indirect Illumination
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_NeFII_Inverse_Rendering_for_Reflectance_Decomposition_With_Near-Field_Indirect_Illumination_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_NeFII_Inverse_Rendering_for_Reflectance_Decomposition_With_Near-Field_Indirect_Illumination_CVPR_2023_paper.pdf)]
    * Title: NeFII: Inverse Rendering for Reflectance Decomposition With Near-Field Indirect Illumination
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Haoqian Wu, Zhipeng Hu, Lincheng Li, Yongqiang Zhang, Changjie Fan, Xin Yu
    * Abstract: Inverse rendering methods aim to estimate geometry, materials and illumination from multi-view RGB images. In order to achieve better decomposition, recent approaches attempt to model indirect illuminations reflected from different materials via Spherical Gaussians (SG), which, however, tends to blur the high-frequency reflection details. In this paper, we propose an end-to-end inverse rendering pipeline that decomposes materials and illumination from multi-view images, while considering near-field indirect illumination. In a nutshell, we introduce the Monte Carlo sampling based path tracing and cache the indirect illumination as neural radiance, enabling a physics-faithful and easy-to-optimize inverse rendering method. To enhance efficiency and practicality, we leverage SG to represent the smooth environment illuminations and apply importance sampling techniques. To supervise indirect illuminations from unobserved directions, we develop a novel radiance consistency constraint between implicit neural radiance and path tracing results of unobserved rays along with the joint optimization of materials and illuminations, thus significantly improving the decomposition performance. Extensive experiments demonstrate that our method outperforms the state-of-the-art on multiple synthetic and real datasets, especially in terms of inter-reflection decomposition.

count=2
* UniSim: A Neural Closed-Loop Sensor Simulator
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.pdf)]
    * Title: UniSim: A Neural Closed-Loop Sensor Simulator
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, Raquel Urtasun
    * Abstract: Rigorously testing autonomy systems is essential for making safe self-driving vehicles (SDV) a reality. It requires one to generate safety critical scenarios beyond what can be collected safely in the world, as many scenarios happen rarely on our roads. To accurately evaluate performance, we need to test the SDV on these scenarios in closed-loop, where the SDV and other actors interact with each other at each timestep. Previously recorded driving logs provide a rich resource to build these new scenarios from, but for closed loop evaluation, we need to modify the sensor data based on the new scene configuration and the SDV's decisions, as actors might be added or removed and the trajectories of existing actors and the SDV will differ from the original log. In this paper, we present UniSim, a neural sensor simulator that takes a single recorded log captured by a sensor-equipped vehicle and converts it into a realistic closed-loop multi-sensor simulation. UniSim builds neural feature grids to reconstruct both the static background and dynamic actors in the scene, and composites them together to simulate LiDAR and camera data at new viewpoints, with actors added or removed and at new placements. To better handle extrapolated views, we incorporate learnable priors for dynamic objects, and leverage a convolutional network to complete unseen regions. Our experiments show UniSim can simulate realistic sensor data with small domain gap on downstream tasks. With UniSim, we demonstrate, for the first time, closed-loop evaluation of an autonomy system on safety-critical scenarios as if it were in the real world.

count=2
* Learning Neural Proto-Face Field for Disentangled 3D Face Modeling in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Learning_Neural_Proto-Face_Field_for_Disentangled_3D_Face_Modeling_in_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Learning_Neural_Proto-Face_Field_for_Disentangled_3D_Face_Modeling_in_CVPR_2023_paper.pdf)]
    * Title: Learning Neural Proto-Face Field for Disentangled 3D Face Modeling in the Wild
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zhenyu Zhang, Renwang Chen, Weijian Cao, Ying Tai, Chengjie Wang
    * Abstract: Generative models show good potential for recovering 3D faces beyond limited shape assumptions. While plausible details and resolutions are achieved, these models easily fail under extreme conditions of pose, shadow or appearance, due to the entangled fitting or lack of multi-view priors. To address this problem, this paper presents a novel Neural Proto-face Field (NPF) for unsupervised robust 3D face modeling. Instead of using constrained images as Neural Radiance Field (NeRF), NPF disentangles the common/specific facial cues, i.e., ID, expression and scene-specific details from in-the-wild photo collections. Specifically, NPF learns a face prototype to aggregate 3D-consistent identity via uncertainty modeling, extracting multi-image priors from a photo collection. NPF then learns to deform the prototype with the appropriate facial expressions, constrained by a loss of expression consistency and personal idiosyncrasies. Finally, NPF is optimized to fit a target image in the collection, recovering specific details of appearance and geometry. In this way, the generative model benefits from multi-image priors and meaningful facial structures. Extensive experiments on benchmarks show that NPF recovers superior or competitive facial shapes and textures, compared to state-of-the-art methods.

count=2
* I2-SDF: Intrinsic Indoor Scene Reconstruction and Editing via Raytracing in Neural SDFs
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_I2-SDF_Intrinsic_Indoor_Scene_Reconstruction_and_Editing_via_Raytracing_in_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_I2-SDF_Intrinsic_Indoor_Scene_Reconstruction_and_Editing_via_Raytracing_in_CVPR_2023_paper.pdf)]
    * Title: I2-SDF: Intrinsic Indoor Scene Reconstruction and Editing via Raytracing in Neural SDFs
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jingsen Zhu, Yuchi Huo, Qi Ye, Fujun Luan, Jifan Li, Dianbing Xi, Lisha Wang, Rui Tang, Wei Hua, Hujun Bao, Rui Wang
    * Abstract: In this work, we present I^2-SDF, a new method for intrinsic indoor scene reconstruction and editing using differentiable Monte Carlo raytracing on neural signed distance fields (SDFs). Our holistic neural SDF-based framework jointly recovers the underlying shapes, incident radiance and materials from multi-view images. We introduce a novel bubble loss for fine-grained small objects and error-guided adaptive sampling scheme to largely improve the reconstruction quality on large-scale indoor scenes. Further, we propose to decompose the neural radiance field into spatially-varying material of the scene as a neural field through surface-based, differentiable Monte Carlo raytracing and emitter semantic segmentations, which enables physically based and photorealistic scene relighting and editing applications. Through a number of qualitative and quantitative experiments, we demonstrate the superior quality of our method on indoor scene reconstruction, novel view synthesis, and scene editing compared to state-of-the-art baselines. Our project page is at https://jingsenzhu.github.io/i2-sdf.

count=2
* PeanutNeRF: 3D Radiance Field for Peanuts
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/AgriVision/html/Saeed_PeanutNeRF_3D_Radiance_Field_for_Peanuts_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/AgriVision/papers/Saeed_PeanutNeRF_3D_Radiance_Field_for_Peanuts_CVPRW_2023_paper.pdf)]
    * Title: PeanutNeRF: 3D Radiance Field for Peanuts
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Farah Saeed, Jin Sun, Peggy Ozias-Akins, Ye (Juliet) Chu, Changying (Charlie) Li
    * Abstract: Accurate phenotypic analysis can help plant breeders efficiently identify and analyze suitable plant traits to enhance crop yield. While 2D images from RGB cameras are easily accessible, their trait estimation performance is limited due to occlusion and the absence of depth information. On the other hand, 3D data from LiDAR sensors are noisy and limited in their ability to capture very thin plant parts such as peanut plant pegs. To combine the merits of both the 2D and 3D data analysis, the 2D images were used to capture thin parts in peanut plants, and deep learning-based 3D reconstruction using captured 2D images was performed to obtain 3D point clouds with information about the scene from different angles. The neural radiance fields were optimized for implicit 3D representation of the plants. The trained radiance fields were queried for 3D reconstruction to achieve point clouds for a 360-degree view and frontal view of the plant. With frontal-view reconstruction and the corresponding 2D images, we used Frustum PVCNN to perform 3D detection of peanut pods. We showed the effectiveness of PeanutNeRF on peanut plants with and without foliage: it showed negligible noise and a chamfer distance of less than 0.0004 from a manually cleaned version. The pod detection showed a precision of around 0.7 at the IoU threshold of 0.5 on the validation set. This method can assist in accurate plant phenotypic studies of peanuts and other important crops.

count=2
* VMINer: Versatile Multi-view Inverse Rendering with Near- and Far-field Light Sources
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Fei_VMINer_Versatile_Multi-view_Inverse_Rendering_with_Near-_and_Far-field_Light_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Fei_VMINer_Versatile_Multi-view_Inverse_Rendering_with_Near-_and_Far-field_Light_CVPR_2024_paper.pdf)]
    * Title: VMINer: Versatile Multi-view Inverse Rendering with Near- and Far-field Light Sources
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Fan Fei, Jiajun Tang, Ping Tan, Boxin Shi
    * Abstract: This paper introduces a versatile multi-view inverse rendering framework with near- and far-field light sources. Tackling the fundamental challenge of inherent ambiguity in inverse rendering our framework adopts a lightweight yet inclusive lighting model for different near- and far-field lights thus is able to make use of input images under varied lighting conditions available during capture. It leverages observations under each lighting to disentangle the intrinsic geometry and material from the external lighting using both neural radiance field rendering and physically-based surface rendering on the 3D implicit fields. After training the reconstructed scene is extracted to a textured triangle mesh for seamless integration into industrial rendering software for various applications. Quantitatively and qualitatively tested on synthetic and real-world scenes our method shows superiority to state-of-the-art multi-view inverse rendering methods in both speed and quality.

count=2
* MonoNPHM: Dynamic Head Reconstruction from Monocular Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Giebenhain_MonoNPHM_Dynamic_Head_Reconstruction_from_Monocular_Videos_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Giebenhain_MonoNPHM_Dynamic_Head_Reconstruction_from_Monocular_Videos_CVPR_2024_paper.pdf)]
    * Title: MonoNPHM: Dynamic Head Reconstruction from Monocular Videos
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Simon Giebenhain, Tobias Kirschstein, Markos Georgopoulos, Martin Rünz, Lourdes Agapito, Matthias Nießner
    * Abstract: We present Monocular Neural Parametric Head Models (MonoNPHM) for dynamic 3D head reconstructions from monocular RGB videos. To this end we propose a latent appearance space that parameterizes a texture field on top of a neural parametric model. We constrain predicted color values to be correlated with the underlying geometry such that gradients from RGB effectively influence latent geometry codes during inverse rendering. To increase the representational capacity of our expression space we augment our backward deformation field with hyper-dimensions thus improving color and geometry representation in topologically challenging expressions. Using MonoNPHM as a learned prior we approach the task of 3D head reconstruction using signed distance field based volumetric rendering. By numerically inverting our backward deformation field we incorporated a landmark loss using facial anchor points that are closely tied to our canonical geometry representation. We incorporate a facial landmark loss by numerically inverting our backward deformation field tied with our canonical geometry to observed 2D facial landmarks in posed space. To evaluate the task of dynamic face reconstruction from monocular RGB videos we record 20 challenging Kinect sequences under casual conditions. MonoNPHM outperforms all baselines with a significant margin and makes an important step towards easily accessible neural parametric face models through RGB tracking.

count=2
* ESR-NeRF: Emissive Source Reconstruction Using LDR Multi-view Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_ESR-NeRF_Emissive_Source_Reconstruction_Using_LDR_Multi-view_Images_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Jeong_ESR-NeRF_Emissive_Source_Reconstruction_Using_LDR_Multi-view_Images_CVPR_2024_paper.pdf)]
    * Title: ESR-NeRF: Emissive Source Reconstruction Using LDR Multi-view Images
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jinseo Jeong, Junseo Koo, Qimeng Zhang, Gunhee Kim
    * Abstract: Existing NeRF-based inverse rendering methods suppose that scenes are exclusively illuminated by distant light sources neglecting the potential influence of emissive sources within a scene. In this work we confront this limitation using LDR multi-view images captured with emissive sources turned on and off. Two key issues must be addressed: 1) ambiguity arising from the limited dynamic range along with unknown lighting details and 2) the expensive computational cost in volume rendering to backtrace the paths leading to final object colors. We present a novel approach ESR-NeRF leveraging neural networks as learnable functions to represent ray-traced fields. By training networks to satisfy light transport segments we regulate outgoing radiances progressively identifying emissive sources while being aware of reflection areas. The results on scenes encompassing emissive sources with various properties demonstrate the superiority of ESR-NeRF in qualitative and quantitative ways. Our approach also extends its applicability to the scenes devoid of emissive sources achieving lower CD metrics on the DTU dataset.

count=2
* L0-Sampler: An L0 Model Guided Volume Sampling for NeRF
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_L0-Sampler_An_L0_Model_Guided_Volume_Sampling_for_NeRF_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_L0-Sampler_An_L0_Model_Guided_Volume_Sampling_for_NeRF_CVPR_2024_paper.pdf)]
    * Title: L0-Sampler: An L0 Model Guided Volume Sampling for NeRF
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Liangchen Li, Juyong Zhang
    * Abstract: Since its proposal Neural Radiance Fields (NeRF) has achieved great success in related tasks mainly adopting the hierarchical volume sampling (HVS) strategy for volume rendering. However the HVS of NeRF approximates distributions using piecewise constant functions which provides a relatively rough estimation. Based on the observation that a well-trained weight function w(t) and the L_0 distance between points and the surface have very high similarity we propose L_0-Sampler by incorporating the L_0 model into w(t) to guide the sampling process. Specifically we propose using piecewise exponential functions rather than piecewise constant functions for interpolation which can not only approximate quasi-L_0 weight distributions along rays quite well but can be easily implemented with a few lines of code change without additional computational burden. Stable performance improvements can be achieved by applying L_0-Sampler to NeRF and related tasks like 3D reconstruction. Code is available at https://ustc3dv.github.io/L0-Sampler/.

count=2
* Neural Super-Resolution for Real-time Rendering with Radiance Demodulation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Neural_Super-Resolution_for_Real-time_Rendering_with_Radiance_Demodulation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Neural_Super-Resolution_for_Real-time_Rendering_with_Radiance_Demodulation_CVPR_2024_paper.pdf)]
    * Title: Neural Super-Resolution for Real-time Rendering with Radiance Demodulation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jia Li, Ziling Chen, Xiaolong Wu, Lu Wang, Beibei Wang, Lei Zhang
    * Abstract: It is time-consuming to render high-resolution images in applications such as video games and virtual reality and thus super-resolution technologies become increasingly popular for real-time rendering. However it is challenging to preserve sharp texture details keep the temporal stability and avoid the ghosting artifacts in real-time super-resolution rendering. To address this issue we introduce radiance demodulation to separate the rendered image or radiance into a lighting component and a material component considering the fact that the light component is smoother than the rendered image so that the high-resolution material component with detailed textures can be easily obtained. We perform the super-resolution on the lighting component only and re-modulate it with the high-resolution material component to obtain the final super-resolution image with more texture details. A reliable warping module is proposed by explicitly marking the occluded regions to avoid the ghosting artifacts. To further enhance the temporal stability we design a frame-recurrent neural network and a temporal loss to aggregate the previous and current frames which can better capture the spatial-temporal consistency among reconstructed frames. As a result our method is able to produce temporally stable results in real-time rendering with high-quality details even in the challenging 4 x4 super-resolution scenarios. Code is available at: \href https://github.com/Riga2/NSRD https://github.com/Riga2/NSRD .

count=2
* HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_HumanGaussian_Text-Driven_3D_Human_Generation_with_Gaussian_Splatting_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_HumanGaussian_Text-Driven_3D_Human_Generation_with_Gaussian_Splatting_CVPR_2024_paper.pdf)]
    * Title: HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xian Liu, Xiaohang Zhan, Jiaxiang Tang, Ying Shan, Gang Zeng, Dahua Lin, Xihui Liu, Ziwei Liu
    * Abstract: Realistic 3D human generation from text prompts is a desirable yet challenging task. Existing methods optimize 3D representations like mesh or neural fields via score distillation sampling (SDS) which suffers from inadequate fine details or excessive training time. In this paper we propose an efficient yet effective framework HumanGaussian that generates high-quality 3D humans with fine-grained geometry and realistic appearance. Our key insight is that 3D Gaussian Splatting is an efficient renderer with periodic Gaussian shrinkage or growing where such adaptive density control can be naturally guided by intrinsic human structures. Specifically 1) we first propose a Structure-Aware SDS that simultaneously optimizes human appearance and geometry. The multi-modal score function from both RGB and depth space is leveraged to distill the Gaussian densification and pruning process. 2) Moreover we devise an Annealed Negative Prompt Guidance by decomposing SDS into a noisier generative score and a cleaner classifier score which well addresses the over-saturation issue. The floating artifacts are further eliminated based on Gaussian size in a prune-only phase to enhance generation smoothness. Extensive experiments demonstrate the superior efficiency and competitive quality of our framework rendering vivid 3D humans under diverse scenarios.

count=2
* Unsigned Orthogonal Distance Fields: An Accurate Neural Implicit Representation for Diverse 3D Shapes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Unsigned_Orthogonal_Distance_Fields_An_Accurate_Neural_Implicit_Representation_for_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_Unsigned_Orthogonal_Distance_Fields_An_Accurate_Neural_Implicit_Representation_for_CVPR_2024_paper.pdf)]
    * Title: Unsigned Orthogonal Distance Fields: An Accurate Neural Implicit Representation for Diverse 3D Shapes
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yujie Lu, Long Wan, Nayu Ding, Yulong Wang, Shuhan Shen, Shen Cai, Lin Gao
    * Abstract: Neural implicit representation of geometric shapes has witnessed considerable advancements in recent years. However common distance field based implicit representations specifically signed distance field (SDF) for watertight shapes or unsigned distance field (UDF) for arbitrary shapes routinely suffer from degradation of reconstruction accuracy when converting to explicit surface points and meshes. In this paper we introduce a novel neural implicit representation based on unsigned orthogonal distance fields (UODFs). In UODFs the minimal unsigned distance from any spatial point to the shape surface is defined solely in one orthogonal direction contrasting with the multi-directional determination made by SDF and UDF. Consequently every point in the 3D UODFs can directly access its closest surface points along three orthogonal directions. This distinctive feature leverages the accurate reconstruction of surface points without interpolation errors. We verify the effectiveness of UODFs through a range of reconstruction examples extending from simple watertight or non-watertight shapes to complex shapes that include hollows internal or assembling structures.

count=2
* SpecNeRF: Gaussian Directional Encoding for Specular Reflections
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_SpecNeRF_Gaussian_Directional_Encoding_for_Specular_Reflections_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_SpecNeRF_Gaussian_Directional_Encoding_for_Specular_Reflections_CVPR_2024_paper.pdf)]
    * Title: SpecNeRF: Gaussian Directional Encoding for Specular Reflections
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Li Ma, Vasu Agrawal, Haithem Turki, Changil Kim, Chen Gao, Pedro Sander, Michael Zollhöfer, Christian Richardt
    * Abstract: Neural radiance fields have achieved remarkable performance in modeling the appearance of 3D scenes. However existing approaches still struggle with the view-dependent appearance of glossy surfaces especially under complex lighting of indoor environments. Unlike existing methods which typically assume distant lighting like an environment map we propose a learnable Gaussian directional encoding to better model the view-dependent effects under near-field lighting conditions. Importantly our new directional encoding captures the spatially-varying nature of near-field lighting and emulates the behavior of prefiltered environment maps. As a result it enables the efficient evaluation of preconvolved specular color at any 3D location with varying roughness coefficients. We further introduce a data-driven geometry prior that helps alleviate the shape radiance ambiguity in reflection modeling. We show that our Gaussian directional encoding and geometry prior significantly improve the modeling of challenging specular reflections in neural radiance fields which helps decompose appearance into more physically meaningful components.

count=2
* Learning Occupancy for Monocular 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_Learning_Occupancy_for_Monocular_3D_Object_Detection_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Peng_Learning_Occupancy_for_Monocular_3D_Object_Detection_CVPR_2024_paper.pdf)]
    * Title: Learning Occupancy for Monocular 3D Object Detection
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Liang Peng, Junkai Xu, Haoran Cheng, Zheng Yang, Xiaopei Wu, Wei Qian, Wenxiao Wang, Boxi Wu, Deng Cai
    * Abstract: Monocular 3D detection is a challenging task due to the lack of accurate 3D information. Existing approaches typically rely on geometry constraints and dense depth estimates to facilitate the learning but often fail to fully exploit the benefits of three-dimensional feature extraction in frustum and 3D space. In this paper we propose OccupancyM3D a method of learning occupancy for monocular 3D detection. It directly learns occupancy in frustum and 3D space leading to more discriminative and informative 3D features and representations. Specifically by using synchronized raw sparse LiDAR point clouds we define the space status and generate voxel-based occupancy labels. We formulate occupancy prediction as a simple classification problem and design associated occupancy losses. Resulting occupancy estimates are employed to enhance original frustum/3D features. As a result experiments on KITTI and Waymo open datasets demonstrate that the proposed method achieves a new state of the art and surpasses other methods by a significant margin.

count=2
* DiffHuman: Probabilistic Photorealistic 3D Reconstruction of Humans
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sengupta_DiffHuman_Probabilistic_Photorealistic_3D_Reconstruction_of_Humans_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sengupta_DiffHuman_Probabilistic_Photorealistic_3D_Reconstruction_of_Humans_CVPR_2024_paper.pdf)]
    * Title: DiffHuman: Probabilistic Photorealistic 3D Reconstruction of Humans
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Akash Sengupta, Thiemo Alldieck, Nikos Kolotouros, Enric Corona, Andrei Zanfir, Cristian Sminchisescu
    * Abstract: We present DiffHuman a probabilistic method for photorealistic 3D human reconstruction from a single RGB image. Despite the ill-posed nature of this problem most methods are deterministic and output a single solution often resulting in a lack of geometric detail and blurriness in unseen or uncertain regions. In contrast DiffHuman predicts a probability distribution over 3D reconstructions conditioned on an input 2D image which allows us to sample multiple detailed 3D avatars that are consistent with the image. DiffHuman is implemented as a conditional diffusion model that denoises pixel-aligned 2D observations of an underlying 3D shape representation. During inference we may sample 3D avatars by iteratively denoising 2D renders of the predicted 3D representation. Furthermore we introduce a generator neural network that approximates rendering with considerably reduced runtime (55x speed up) resulting in a novel dual-branch diffusion framework. Our experiments show that DiffHuman can produce diverse and detailed reconstructions for the parts of the person that are unseen or uncertain in the input image while remaining competitive with the state-of-the-art when reconstructing visible surfaces.

count=2
* TutteNet: Injective 3D Deformations by Composition of 2D Mesh Deformations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_TutteNet_Injective_3D_Deformations_by_Composition_of_2D_Mesh_Deformations_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_TutteNet_Injective_3D_Deformations_by_Composition_of_2D_Mesh_Deformations_CVPR_2024_paper.pdf)]
    * Title: TutteNet: Injective 3D Deformations by Composition of 2D Mesh Deformations
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Bo Sun, Thibault Groueix, Chen Song, Qixing Huang, Noam Aigerman
    * Abstract: This work proposes a novel representation of injective deformations of 3D space which overcomes existing limitations of injective methods namely inaccuracy lack of robustness and incompatibility with general learning and optimization frameworks. Our core idea is to reduce the problem to a "deep" composition of multiple 2D mesh-based piecewise-linear maps. Namely we build differentiable layers that produce mesh deformations through Tutte's embedding (guaranteed to be injective in 2D) and compose these layers over different planes to create complex 3D injective deformations of the 3D volume. We show our method provides the ability to ef?ciently and accurately optimize and learn complex deformations outperforming other injective approaches. As a main application we produce complex and artifact-free NeRF and SDF deformations.

count=2
* Generating Material-Aware 3D Models from Sparse Views
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/html/Mao_Generating_Material-Aware_3D_Models_from_Sparse_Views_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/papers/Mao_Generating_Material-Aware_3D_Models_from_Sparse_Views_CVPRW_2024_paper.pdf)]
    * Title: Generating Material-Aware 3D Models from Sparse Views
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Shi Mao, Chenming Wu, Ran Yi, Zhelun Shen, Liangjun Zhang, Wolfgang Heidrich
    * Abstract: Image-to-3D diffusion models have significantly advanced 3D content generation. However existing methods often struggle to disentangle material and illumination from coupled appearance as they primarily focus on modeling geometry and appearance. This paper introduces a novel approach to generate material-aware 3D models from sparse-view images using generative models and efficient pre-integrated rendering. The output of our method is a relightable model that independently models geometry material and lighting enabling downstream tasks to manipulate these components separately. To fully leverage information from limited sparse views we propose a mixed supervision framework that simultaneously exploits view-consistency via captured views and diffusion prior via generating views. Additionally a view selection mechanism is proposed to mitigate the degenerated diffusion prior. We adapt an efficient yet powerful pre-integrated rendering pipeline to factorize the scene into a differentiable environment illumination a spatially varying material field and an implicit SDF field. Our experiments on both real-world and synthetic datasets demonstrate the effectiveness of our approach in decomposing each component as well as manipulating the illumination. Source codes are available at https://github.com/Sheldonmao/MatSparse3D

count=2
* Modeling the Calibration Pipeline of the Lytro Camera for High Quality Light-Field Image Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Cho_Modeling_the_Calibration_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Cho_Modeling_the_Calibration_2013_ICCV_paper.pdf)]
    * Title: Modeling the Calibration Pipeline of the Lytro Camera for High Quality Light-Field Image Reconstruction
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Donghyeon Cho, Minhaeng Lee, Sunyeong Kim, Yu-Wing Tai
    * Abstract: Light-field imaging systems have got much attention recently as the next generation camera model. A light-field imaging system consists of three parts: data acquisition, manipulation, and application. Given an acquisition system, it is important to understand how a light-field camera converts from its raw image to its resulting refocused image. In this paper, using the Lytro camera as an example, we describe step-by-step procedures to calibrate a raw light-field image. In particular, we are interested in knowing the spatial and angular coordinates of the micro lens array and the resampling process for image reconstruction. Since Lytro uses a hexagonal arrangement of a micro lens image, additional treatments in calibration are required. After calibration, we analyze and compare the performances of several resampling methods for image reconstruction with and without calibration. Finally, a learning based interpolation method is proposed which demonstrates a higher quality image reconstruction than previous interpolation methods including a method used in Lytro software.

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
* Semantic Pose Using Deep Networks Trained on Synthetic RGB-D
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Papon_Semantic_Pose_Using_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Papon_Semantic_Pose_Using_ICCV_2015_paper.pdf)]
    * Title: Semantic Pose Using Deep Networks Trained on Synthetic RGB-D
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Jeremie Papon, Markus Schoeler
    * Abstract: In this work we address the problem of indoor scene understanding from RGB-D images. Specifically, we propose to find instances of common furniture classes, their spatial extent, and their pose with respect to generalized class models. To accomplish this, we use a deep, wide, multi-output convolutional neural network (CNN) that predicts class, pose, and location of possible objects simultaneously. To overcome the lack of large annotated RGB-D training sets (especially those with pose), we use an on-the-fly rendering pipeline that generates realistic cluttered room scenes in parallel to training. We then perform transfer learning on the relatively small amount of publicly available annotated RGB-D data, and find that our model is able to successfully annotate even highly challenging real scenes. Importantly, our trained network is able to understand noisy and sparse observations of highly cluttered scenes with a remarkable degree of accuracy, inferring class and pose from a very limited set of cues. Additionally, our neural network is only moderately deep and computes class, pose and position in tandem, so the overall run-time is significantly faster than existing methods, estimating all output parameters simultaneously in parallel.

count=2
* Extended Depth of Field Catadioptric Imaging Using Focal Sweep
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Yokoya_Extended_Depth_of_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Yokoya_Extended_Depth_of_ICCV_2015_paper.pdf)]
    * Title: Extended Depth of Field Catadioptric Imaging Using Focal Sweep
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Ryunosuke Yokoya, Shree K. Nayar
    * Abstract: Catadioptric imaging systems use curved mirrors to capture wide fields of view. However, due to the curvature of the mirror, these systems tend to have very limited depth of field (DOF), with the point spread function (PSF) varying dramatically over the field of view and as a function of scene depth. In recent years, focal sweep has been used extensively to extend the DOF of conventional imaging systems. It has been shown that focal sweep produces an integrated point spread function (IPSF) that is nearly space-invariant and depth-invariant, enabling the recovery of an extended depth of field (EDOF) image by deconvolving the captured focal sweep image with a single IPSF. In this paper, we use focal sweep to extend the DOF of a catadioptric imaging system. We show that while the IPSF is spatially varying when a curved mirror is used, it remains quasi depth-invariant over the wide field of view of the imaging system. We have developed a focal sweep system where mirrors of different shapes can be used to capture wide field of view EDOF images. In particular, we show experimental results using spherical and paraboloidal mirrors.

count=2
* Unrolled Memory Inner-Products: An Abstract GPU Operator for Efficient Vision-Related Computations
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Unrolled_Memory_Inner-Products_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Unrolled_Memory_Inner-Products_ICCV_2017_paper.pdf)]
    * Title: Unrolled Memory Inner-Products: An Abstract GPU Operator for Efficient Vision-Related Computations
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Yu-Sheng Lin, Wei-Chao Chen, Shao-Yi Chien
    * Abstract: Recently, convolutional neural networks (CNNs) have achieved great success in fields such as computer vision, natural language processing, and artificial intelligence. Many of these applications utilize parallel processing in GPUs to achieve higher performance. However, it remains a daunting task to optimize for GPUs, and most researchers have to rely on vendor-provided libraries for such purposes. In this paper, we discuss an operator that can be used to succinctly express computational kernels in CNNs and various scientific and vision applications. This operator, called Unrolled-Memory-Inner-Product (UMI), is a computationally-efficient operator with smaller code token requirement. Since a naive UMI implementation would increase memory requirement through input data unrolling, we propose a method to achieve optimal memory fetch performance in modern GPUs. We demonstrate this operator by converting several popular applications into the UMI representation and achieve 1.3x-26.4x speedup against frameworks such as OpenCV and Caffe.

count=2
* On Tablet 3D Structured Light Reconstruction and Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w35/html/Donlic_On_Tablet_3D_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w35/Donlic_On_Tablet_3D_ICCV_2017_paper.pdf)]
    * Title: On Tablet 3D Structured Light Reconstruction and Registration
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Matea Donlic, Tomislav Petkovic, Tomislav Pribanic
    * Abstract: One of the very first tablets with a built-in DLP projector has recently appeared on the market while smartphones with a built-in projector have been available around for quite a while. Interestingly, 3D reconstruction solutions on mobile devices never considered exploiting a built-in projector for the implementation of a powerful active stereo concept, structured light (SL), whose main component is a camera-projector pair. In this work we demonstrate a 3D reconstruction framework implementing SL on a tablet. In addition, we propose a 3D registration method by taking the advantage in a novel way of two commonly available sensors on mobile devices, an accelerometer and a magnetometer. The proposed solution provides robust and accurate 3D reconstruction and 3D registration results.

count=2
* Deep Meta Functionals for Shape Representation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Littwin_Deep_Meta_Functionals_for_Shape_Representation_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Littwin_Deep_Meta_Functionals_for_Shape_Representation_ICCV_2019_paper.pdf)]
    * Title: Deep Meta Functionals for Shape Representation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Gidi Littwin,  Lior Wolf
    * Abstract: We present a new method for 3D shape reconstruction from a single image, in which a deep neural network directly maps an image to a vector of network weights. The network parametrized by these weights represents a 3D shape by classifying every point in the volume as either within or outside the shape. The new representation has virtually unlimited capacity and resolution, and can have an arbitrary topology. Our experiments show that it leads to more accurate shape inference from a 2D projection than the existing methods, including voxel-, silhouette-, and mesh-based methods. The code will be available at: https: //github.com/gidilittwin/Deep-Meta.

count=2
* PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Saito_PIFu_Pixel-Aligned_Implicit_Function_for_High-Resolution_Clothed_Human_Digitization_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Saito_PIFu_Pixel-Aligned_Implicit_Function_for_High-Resolution_Clothed_Human_Digitization_ICCV_2019_paper.pdf)]
    * Title: PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Shunsuke Saito,  Zeng Huang,  Ryota Natsume,  Shigeo Morishima,  Angjoo Kanazawa,  Hao Li
    * Abstract: We introduce Pixel-aligned Implicit Function (PIFu), an implicit representation that locally aligns pixels of 2D images with the global context of their corresponding 3D object. Using PIFu, we propose an end-to-end deep learning method for digitizing highly detailed clothed humans that can infer both 3D surface and texture from a single image, and optionally, multiple input images. Highly intricate shapes, such as hairstyles, clothing, as well as their variations and deformations can be digitized in a unified way. Compared to existing representations used for 3D deep learning, PIFu produces high-resolution surfaces including largely unseen regions such as the back of a person. In particular, it is memory efficient unlike the voxel representation, can handle arbitrary topology, and the resulting surface is spatially aligned with the input image. Furthermore, while previous techniques are designed to process either a single image or multiple views, PIFu extends naturally to arbitrary number of views. We demonstrate high-resolution and robust reconstructions on real world images from the DeepFashion dataset, which contains a variety of challenging clothing types. Our method achieves state-of-the-art performance on a public benchmark and outperforms the prior work for clothed human digitization from a single image.

count=2
* Adaptive Surface Reconstruction With Multiscale Convolutional Kernels
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Ummenhofer_Adaptive_Surface_Reconstruction_With_Multiscale_Convolutional_Kernels_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Ummenhofer_Adaptive_Surface_Reconstruction_With_Multiscale_Convolutional_Kernels_ICCV_2021_paper.pdf)]
    * Title: Adaptive Surface Reconstruction With Multiscale Convolutional Kernels
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Benjamin Ummenhofer, Vladlen Koltun
    * Abstract: We propose generalized convolutional kernels for 3D reconstruction with ConvNets from point clouds. Our method uses multiscale convolutional kernels that can be applied to adaptive grids as generated with octrees. In addition to standard kernels in which each element has a distinct spatial location relative to the center, our elements have a distinct relative location as well as a relative scale level. Making our kernels span multiple resolutions allows us to apply ConvNets to adaptive grids for large problem sizes where the input data is sparse but the entire domain needs to be processed. Our ConvNet architecture can predict the signed and unsigned distance fields for large data sets with millions of input points and is faster and more accurate than classic energy minimization or recent learning approaches. We demonstrate this in a zero-shot setting where we only train on synthetic data and evaluate on the Tanks and Temples dataset of real-world large-scale 3D scenes.

count=2
* Be Everywhere - Hear Everything (BEE): Audio Scene Reconstruction by Sparse Audio-Visual Samples
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Be_Everywhere_-_Hear_Everything_BEE_Audio_Scene_Reconstruction_by_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Be_Everywhere_-_Hear_Everything_BEE_Audio_Scene_Reconstruction_by_ICCV_2023_paper.pdf)]
    * Title: Be Everywhere - Hear Everything (BEE): Audio Scene Reconstruction by Sparse Audio-Visual Samples
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Mingfei Chen, Kun Su, Eli Shlizerman
    * Abstract: Fully immersive and interactive audio-visual scenes are dynamic such that the listeners and the sound emitters move and interact with each other. Reconstruction of an immersive sound experience, as it happens in the scene, requires detailed reconstruction of the audio perceived by the listener at an arbitrary location. The audio at the listener location is a complex outcome of sound propagation through the scene geometry and interacting with surfaces and also the locations of the emitters and the sounds they emit. Due to these aspects, detailed audio reconstruction requires extensive sampling of audio at any potential listener location. This is usually difficult to implement in realistic real-time dynamic scenes. In this work, we propose to circumvent the need for extensive sensors by leveraging audio and visual samples from only a handful of A/V receivers placed in the scene. In particular, we introduce a novel method and end-to-end integrated rendering pipeline which allows the listener to be everywhere and hear everything (BEE) in a dynamic scene in real-time. BEE reconstructs the audio with two main modules, Joint Audio-Visual Representation, and Integrated Rendering Head. The first module extracts the informative audio-visual features of the scene from sparse A/V reference samples, while the second module integrates the audio samples with learned time-frequency transformations to obtain the target sound. Our experiments indicate that BEE outperforms existing methods by a large margin in terms of quality of sound reconstruction, can generalize to scenes not seen in training and runs in real-time speed.

count=2
* Multiscale Representation for Real-Time Anti-Aliasing Neural Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Multiscale_Representation_for_Real-Time_Anti-Aliasing_Neural_Rendering_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Multiscale_Representation_for_Real-Time_Anti-Aliasing_Neural_Rendering_ICCV_2023_paper.pdf)]
    * Title: Multiscale Representation for Real-Time Anti-Aliasing Neural Rendering
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Dongting Hu, Zhenkai Zhang, Tingbo Hou, Tongliang Liu, Huan Fu, Mingming Gong
    * Abstract: The rendering scheme in neural radiance field (NeRF) is effective in rendering a pixel by casting a ray into the scene. However, NeRF yields blurred rendering results when the training images are captured at non-uniform scales, and produces aliasing artifacts if the test images are taken in distant views. To address this issue, Mip-NeRF proposes a multiscale representation as a conical frustum to encode scale information. Nevertheless, this approach is only suitable for offline rendering since it relies on integrated positional encoding (IPE) to query a multilayer perceptron (MLP). To overcome this limitation, we propose mip voxel grids (Mip-VoG), an explicit multiscale representation with a deferred architecture for real-time anti-aliasing rendering. Our approach includes a density Mip-VoG for scene geometry and a feature Mip-VoG with a small MLP for view-dependent color. Mip-VoG represents scene scale using the level of detail (LOD) derived from ray differentials and uses quadrilinear interpolation to map a queried 3D location to its features and density from two neighboring down-sampled voxel grids. To our knowledge, our approach is the first to offer multiscale training and real-time anti-aliasing rendering simultaneously. We conducted experiments on multiscale dataset, results show that our approach outperforms state-of-the-art real-time rendering baselines.

count=2
* ClimateNeRF: Extreme Weather Synthesis in Neural Radiance Field
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_ClimateNeRF_Extreme_Weather_Synthesis_in_Neural_Radiance_Field_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_ClimateNeRF_Extreme_Weather_Synthesis_in_Neural_Radiance_Field_ICCV_2023_paper.pdf)]
    * Title: ClimateNeRF: Extreme Weather Synthesis in Neural Radiance Field
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yuan Li, Zhi-Hao Lin, David Forsyth, Jia-Bin Huang, Shenlong Wang
    * Abstract: Physical simulations produce excellent predictions of weather effects. Neural radiance fields produce SOTA scene models. We describe a novel NeRF-editing procedure that can fuse physical simulations with NeRF models of scenes, producing realistic movies of physical phenomena in those scenes. Our application -- Climate NeRF -- allows people to visualize what climate change outcomes will do to them. ClimateNeRF allows us to render realistic weather effects, including smog, snow, and flood. Results can be controlled with physically meaningful variables like water level. Qualitative and quantitative studies show that our simulated results are significantly more realistic than those from SOTA 2D image editing and SOTA 3D NeRF stylization.

count=2
* RenderIH: A Large-Scale Synthetic Dataset for 3D Interacting Hand Pose Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_RenderIH_A_Large-Scale_Synthetic_Dataset_for_3D_Interacting_Hand_Pose_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_RenderIH_A_Large-Scale_Synthetic_Dataset_for_3D_Interacting_Hand_Pose_ICCV_2023_paper.pdf)]
    * Title: RenderIH: A Large-Scale Synthetic Dataset for 3D Interacting Hand Pose Estimation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Lijun Li, Linrui Tian, Xindi Zhang, Qi Wang, Bang Zhang, Liefeng Bo, Mengyuan Liu, Chen Chen
    * Abstract: The current interacting hand (IH) datasets are relatively simplistic in terms of background and texture, with hand joints being annotated by a machine annotator, which may result in inaccuracies, and the diversity of pose distribution is limited. However, the variability of background, pose distribution, and texture can greatly influence the generalization ability. Therefore, we present a large-scale synthetic dataset --RenderIH-- for interacting hands with accurate and diverse pose annotations. The dataset contains 1M photo-realistic images with varied backgrounds, perspectives, and hand textures. To generate natural and diverse interacting poses, we propose a new pose optimization algorithm. Additionally, for better pose estimation accuracy, we introduce a transformer-based pose estimation network, TransHand, to leverage the correlation between interacting hands and verify the effectiveness of RenderIH in improving results. Our dataset is model-agnostic and can improve more accuracy of any hand pose estimation method in comparison to other real or synthetic datasets. Experiments have shown that pretraining on our synthetic data can significantly decrease the error from 6.76mm to 5.79mm, and our Transhand surpasses contemporary methods. Our dataset and code are available at https://github.com/adwardlee/RenderIH.

count=2
* Dynamic Mesh-Aware Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Qiao_Dynamic_Mesh-Aware_Radiance_Fields_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Qiao_Dynamic_Mesh-Aware_Radiance_Fields_ICCV_2023_paper.pdf)]
    * Title: Dynamic Mesh-Aware Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yi-Ling Qiao, Alexander Gao, Yiran Xu, Yue Feng, Jia-Bin Huang, Ming C. Lin
    * Abstract: Embedding polygonal mesh assets within photorealistic Neural Radience Fields (NeRF) volumes, such that they can be rendered and their dynamics simulated in a physically consistent manner with the NeRF, is under-explored from the system perspective of integrating NeRF into the traditional graphics pipeline. This paper designs a two-way coupling between mesh and NeRF during rendering and simulation. We first review the light transport equations for both mesh and NeRF, then distill them into an efficient algorithm for updating radiance and throughput along a cast ray with an arbitrary number of bounces. To resolve the discrepancy between the linear color space that the path tracer assumes and the sRGB color space that standard NeRF uses, we train NeRF with High Dynamic Range (HDR) images. We also present a strategy to estimate light sources and cast shadows on the NeRF. Finally, we consider how the hybrid surface-volumetric formulation can be efficiently integrated with a high-performance physics simulator that supports cloth, rigid and soft bodies. The full rendering and simulation system can be run on a GPU at interactive rates. We show that a hybrid system approach outperforms alternatives in visual realism for mesh insertion, because it allows realistic light transport from volumetric NeRF media onto surfaces, which affects the appearance of reflective/refractive surfaces and illumination of diffuse surfaces informed by the dynamic scene.

count=2
* Multi-task View Synthesis with Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Multi-task_View_Synthesis_with_Neural_Radiance_Fields_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Multi-task_View_Synthesis_with_Neural_Radiance_Fields_ICCV_2023_paper.pdf)]
    * Title: Multi-task View Synthesis with Neural Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Shuhong Zheng, Zhipeng Bao, Martial Hebert, Yu-Xiong Wang
    * Abstract: Multi-task visual learning is a critical aspect of computer vision. Current research, however, predominantly concentrates on the multi-task dense prediction setting, which overlooks the intrinsic 3D world and its multi-view consistent structures, and lacks the capacity for versatile imagination. In response to these limitations, we present a novel problem setting -- multi-task view synthesis (MTVS), which reinterprets multi-task prediction as a set of novel-view synthesis tasks for multiple scene properties, including RGB. To tackle the MTVS problem, we propose MuvieNeRF, a framework that incorporates both multi-task and cross-view knowledge to simultaneously synthesize multiple scene properties. MuvieNeRF integrates two key modules, the Cross-Task Attention (CTA) and Cross-View Attention (CVA) modules, enabling the efficient use of information across multiple views and tasks. Extensive evaluations on both synthetic and realistic benchmarks demonstrate that MuvieNeRF is capable of simultaneously synthesizing different scene properties with promising visual quality, even outperforming conventional discriminative models in various settings. Notably, we show that MuvieNeRF exhibits universal applicability across a range of NeRF backbones. Our code is available at https://github.com/zsh2000/MuvieNeRF.

count=2
* Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/html/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/papers/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.pdf)]
    * Title: Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Csaba Beleznai, Peter Gemeiner, Christian Zinner
    * Abstract: Reliable and timely detection of abandoned items in public places still represents an unsolved problem for automated visual surveillance. Typical surveilled scenarios are associated with high visual ambiguity such as shadows, occlusions, illumination changes and substantial clutter consisting of a mixture of dynamic and stationary objects. Motivated by these challenges we propose a reliable left item detection approach based on the combination of intensity and depth data from a passive stereo setup. The employed in-house developed stereo system consists of low-cost sensors and it is capable to perform detection in environments of up to 10m x 10m in size. The proposed algorithm is tested on a set of indoor sequences and compared to manually annotated ground truth data. Obtained results show that many failure modes of intensity-based approaches are absent and even small-sized objects such as a handbag can be reliably detected when left behind in a scene. The presented results display a very promising approach, which can robustly detect left luggage in dynamic environments at a close to real-time computational speed.

count=2
* Shape From Caustics: Reconstruction of 3D-Printed Glass From Simulated Caustic Images
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Kassubeck_Shape_From_Caustics_Reconstruction_of_3D-Printed_Glass_From_Simulated_Caustic_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Kassubeck_Shape_From_Caustics_Reconstruction_of_3D-Printed_Glass_From_Simulated_Caustic_WACV_2021_paper.pdf)]
    * Title: Shape From Caustics: Reconstruction of 3D-Printed Glass From Simulated Caustic Images
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Marc Kassubeck, Florian Burgel, Susana Castillo, Sebastian Stiller, Marcus Magnor
    * Abstract: We present an efficient and effective computational framework for the inverse rendering problem of reconstructing the 3D shape of a piece of glass from its caustic image. Our approach is motivated by the needs of 3D glass printing, a nascent additive manufacturing technique that promises to revolutionize the production of optics elements, from lightweight mirrors to waveguides and lenses. One important problem is the reliable control of the manufacturing process by inferring the printed 3D glass shape from its caustic image. Towards this goal, we propose a novel general-purpose reconstruction algorithm based on differentiable light propagation simulation followed by a regularization scheme that takes the deposited glass volume into account. This enables incorporating arbitrary measurements of caustics into an efficient reconstruction framework. We demonstrate the effectiveness of our method and establish the influence of our hyperparameters using several sample shapes and parameter configurations.

count=2
* LensNeRF: Rethinking Volume Rendering Based on Thin-Lens Camera Model
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Kim_LensNeRF_Rethinking_Volume_Rendering_Based_on_Thin-Lens_Camera_Model_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_LensNeRF_Rethinking_Volume_Rendering_Based_on_Thin-Lens_Camera_Model_WACV_2024_paper.pdf)]
    * Title: LensNeRF: Rethinking Volume Rendering Based on Thin-Lens Camera Model
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Min-Jung Kim, Gyojung Gu, Jaegul Choo
    * Abstract: Recent advances in Neural Radiance Field (NeRF) show promising results in rendering realistic novel view images. However, NeRF and its variants assume that input images are captured using a pinhole camera and that subjects in images are always all-in-focus by tacit agreement. In this paper, we propose aperture-aware NeRF optimization and rendering methods using a thin-lens model (dubbed LensNeRF), which allows defocus images of any aperture size as input and output. To generalize a pinhole camera model to a thin-lens camera model in NeRF framework, we define multiple rays originating from the aperture area, solving world-to-pixel scale ambiguity. Also, we propose in-focus loss that assigns the given pixel color to points on the focus plane to alleviate the color ambiguity caused by the use of multiple rays. For the rigorous evaluation of the proposed method, we collect a real forward-facing dataset with different F-numbers for each viewpoint. Experimental results demonstrate that our method successfully fuses an aperture-size adjustable thin-lens camera model into the NeRF architecture, showing favorable qualitative and quantitative results compared to baseline models. The dataset will be made available.

count=2
* SLoSH: Set Locality Sensitive Hashing via Sliced-Wasserstein Embeddings
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Lu_SLoSH_Set_Locality_Sensitive_Hashing_via_Sliced-Wasserstein_Embeddings_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Lu_SLoSH_Set_Locality_Sensitive_Hashing_via_Sliced-Wasserstein_Embeddings_WACV_2024_paper.pdf)]
    * Title: SLoSH: Set Locality Sensitive Hashing via Sliced-Wasserstein Embeddings
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Yuzhe Lu, Xinran Liu, Andrea Soltoggio, Soheil Kolouri
    * Abstract: Learning from set-structured data is an essential problem with many applications in machine learning and computer vision. This paper focuses on non-parametric and data-independent learning from set-structured data using approximate nearest neighbor (ANN) solutions, particularly locality-sensitive hashing. We consider the problem of set retrieval from an input set query. Such a retrieval problem requires: 1) an efficient mechanism to calculate the distances/dissimilarities between sets, and 2) an appropriate data structure for fast nearest-neighbor search. To that end, we propose to use Sliced-Wasserstein embedding as a computationally efficient set-2-vector operator that enables downstream ANN, with theoretical guarantees. The set elements are treated as samples from an unknown underlying distribution, and the Sliced-Wasserstein distance is used to compare sets. We demonstrate the effectiveness of our algorithm, denoted as Set Locality Sensitive Hashing (SLoSH), on various set retrieval datasets and compare our proposed embedding with standard set embedding approaches, including Generalized Mean (GeM) embedding/pooling, Featurewise Sort Pooling (FSPool), Covariance Pooling, and Wasserstein embedding and show consistent improvement in retrieval results.

count=2
* Self-Supervised Intrinsic Image Decomposition
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/c8862fc1a32725712838863fb1a260b9-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/c8862fc1a32725712838863fb1a260b9-Paper.pdf)]
    * Title: Self-Supervised Intrinsic Image Decomposition
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Michael Janner, Jiajun Wu, Tejas D. Kulkarni, Ilker Yildirim, Josh Tenenbaum
    * Abstract: Intrinsic decomposition from a single image is a highly challenging task, due to its inherent ambiguity and the scarcity of training data. In contrast to traditional fully supervised learning approaches, in this paper we propose learning intrinsic image decomposition by explaining the input image. Our model, the Rendered Intrinsics Network (RIN), joins together an image decomposition pipeline, which predicts reflectance, shape, and lighting conditions given a single image, with a recombination function, a learned shading model used to recompose the original input based off of intrinsic image predictions. Our network can then use unsupervised reconstruction error as an additional signal to improve its intermediate representations. This allows large-scale unlabeled data to be useful during training, and also enables transferring learned knowledge to images of unseen object categories, lighting conditions, and shapes. Extensive experiments demonstrate that our method performs well on both intrinsic image decomposition and knowledge transfer.

count=2
* Unsupervised Learning of Shape and Pose with Differentiable Point Clouds
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2018/hash/4e8412ad48562e3c9934f45c3e144d48-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2018/file/4e8412ad48562e3c9934f45c3e144d48-Paper.pdf)]
    * Title: Unsupervised Learning of Shape and Pose with Differentiable Point Clouds
    * Publisher: NeurIPS
    * Publication Date: `2018`
    * Authors: Eldar Insafutdinov, Alexey Dosovitskiy
    * Abstract: We address the problem of learning accurate 3D shape and camera pose from a collection of unlabeled category-specific images. We train a convolutional network to predict both the shape and the pose from a single image by minimizing the reprojection error: given several views of an object, the projections of the predicted shapes to the predicted camera poses should match the provided views. To deal with pose ambiguity, we introduce an ensemble of pose predictors which we then distill to a single "student" model. To allow for efficient learning of high-fidelity shapes, we represent the shapes by point clouds and devise a formulation allowing for differentiable projection of these. Our experiments show that the distilled ensemble of pose predictors learns to estimate the pose accurately, while the point cloud representation allows to predict detailed shape models.

count=2
* Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/1a77befc3b608d6ed363567685f70e1e-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/1a77befc3b608d6ed363567685f70e1e-Paper.pdf)]
    * Title: Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, Yaron Lipman
    * Abstract: In this work we address the challenging problem of multiview 3D surface reconstruction. We introduce a neural network architecture that simultaneously learns the unknown geometry, camera parameters, and a neural renderer that approximates the light reflected from the surface towards the camera. The geometry is represented as a zero level-set of a neural network, while the neural renderer, derived from the rendering equation, is capable of (implicitly) modeling a wide set of lighting conditions and materials. We trained our network on real world 2D images of objects with different material properties, lighting conditions, and noisy camera initializations from the DTU MVS dataset. We found our model to produce state of the art 3D surface reconstructions with high fidelity, resolution and detail.

count=2
* NSVF
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/b4b758962f17808746e9bb832a6fa4b8-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/b4b758962f17808746e9bb832a6fa4b8-Paper.pdf)]
    * Title: Neural Sparse Voxel Fields
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, Christian Theobalt
    * Abstract: Photo-realistic free-viewpoint rendering of real-world scenes using classical computer graphics techniques is challenging, because it requires the difficult step of capturing detailed appearance and geometry models. Recent studies have demonstrated promising results by learning scene representations that implicitly encodes both geometry and appearance without 3D supervision. However, existing approaches in practice often show blurry renderings caused by the limited network capacity or the difficulty in finding accurate intersections of camera rays with the scene geometry. Synthesizing high-resolution imagery from these representations often requires time-consuming optical ray marching. In this work, we introduce Neural Sparse Voxel Fields (NSVF), a new neural scene representation for fast and high-quality free-viewpoint rendering. The NSVF defines a series of voxel-bounded implicit fields organized in a sparse voxel octree to model local properties in each cell. We progressively learn the underlying voxel structures with a differentiable ray-marching operation from only a set of posed RGB images. With the sparse voxel octree structure, rendering novel views at inference time can be accelerated by skipping the voxels without relevant scene content. Our method is over 10 times faster than the state-of-the-art while achieving higher quality results. Furthermore, by utilizing an explicit sparse voxel representation, our method can be easily applied to scene editing and scene composition. we also demonstrate various kinds of challenging tasks, including multi-object learning, free-viewpoint rendering of a moving human, and large-scale scene rendering.

count=2
* MeshSDF: Differentiable Iso-Surface Extraction
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/fe40fb944ee700392ed51bfe84dd4e3d-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/fe40fb944ee700392ed51bfe84dd4e3d-Paper.pdf)]
    * Title: MeshSDF: Differentiable Iso-Surface Extraction
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Edoardo Remelli, Artem Lukoianov, Stephan Richter, Benoit Guillard, Timur Bagautdinov, Pierre Baque, Pascal Fua
    * Abstract: Geometric Deep Learning has recently made striking progress with the advent of continuous Deep Implicit Fields. They allow for detailed modeling of watertight surfaces of arbitrary topology while not relying on a 3D Euclidean grid, resulting in a learnable parameterization that is not limited in resolution. Unfortunately, these methods are often not suitable for applications that require an explicit mesh-based surface representation because converting an implicit field to such a representation relies on the Marching Cubes algorithm, which cannot be differentiated with respect to the underlying implicit field. In this work, we remove this limitation and introduce a differentiable way to produce explicit surface mesh representations from Deep Signed Distance Functions. Our key insight is that by reasoning on how implicit field perturbations impact local surface geometry, one can ultimately differentiate the 3D location of surface samples with respect to the underlying deep implicit field. We exploit this to define MeshSDF, an end-to-end differentiable mesh representation which can vary its topology. We use two different applications to validate our theoretical insight: Single-View Reconstruction via Differentiable Rendering and Physically-Driven Shape Optimization. In both cases our differentiable parameterization gives us an edge over state-of-the-art algorithms.

count=2
* Differentiable rendering with perturbed optimizers
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/ab233b682ec355648e7891e66c54191b-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/ab233b682ec355648e7891e66c54191b-Paper.pdf)]
    * Title: Differentiable rendering with perturbed optimizers
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Quentin Le Lidec, Ivan Laptev, Cordelia Schmid, Justin Carpentier
    * Abstract: Reasoning about 3D scenes from their 2D image projections is one of the core problems in computer vision. Solutions to this inverse and ill-posed problem typically involve a search for models that best explain observed image data. Notably, images depend both on the properties of observed scenes and on the process of image formation. Hence, if optimization techniques should be used to explain images, it is crucial to design differentable functions for the projection of 3D scenes into images, also known as differentiable rendering. Previous approaches to differentiable rendering typically replace non-differentiable operations by smooth approximations, impacting the subsequent 3D estimation. In this paper, we take a more general approach and study differentiable renderers through the prism of randomized optimization and the related notion of perturbed optimizers. In particular, our work highlights the link between some well-known differentiable renderer formulations and randomly smoothed optimizers, and introduces differentiable perturbed renderers. We also propose a variance reduction mechanism to alleviate the computational burden inherent to perturbed optimizers and introduce an adaptive scheme to automatically adjust the smoothing parameters of the rendering process. We apply our method to 3D scene reconstruction and demonstrate its advantages on the tasks of 6D pose estimation and 3D mesh reconstruction. By providing informative gradients that can be used as a strong supervisory signal, we demonstrate the benefits of perturbed renderers to obtain more accurate solutions when compared to the state-of-the-art alternatives using smooth gradient approximations.

count=2
* Learning Neural Acoustic Fields
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/151f4dfc71f025ae387e2d7a4ea1639b-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/151f4dfc71f025ae387e2d7a4ea1639b-Paper-Conference.pdf)]
    * Title: Learning Neural Acoustic Fields
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Andrew Luo, Yilun Du, Michael Tarr, Josh Tenenbaum, Antonio Torralba, Chuang Gan
    * Abstract: Our environment is filled with rich and dynamic acoustic information. When we walk into a cathedral, the reverberations as much as appearance inform us of the sanctuary's wide open space. Similarly, as an object moves around us, we expect the sound emitted to also exhibit this movement. While recent advances in learned implicit functions have led to increasingly higher quality representations of the visual world, there have not been commensurate advances in learning spatial auditory representations. To address this gap, we introduce Neural Acoustic Fields (NAFs), an implicit representation that captures how sounds propagate in a physical scene. By modeling acoustic propagation in a scene as a linear time-invariant system, NAFs learn to continuously map all emitter and listener location pairs to a neural impulse response function that can then be applied to arbitrary sounds. We demonstrate NAFs on both synthetic and real data, and show that the continuous nature of NAFs enables us to render spatial acoustics for a listener at arbitrary locations. We further show that the representation learned by NAFs can help improve visual learning with sparse views. Finally we show that a representation informative of scene structure emerges during the learning of NAFs.

count=2
* FlyView: a bio-informed optical flow truth dataset for visual navigation using panoramic stereo vision
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/b4005da5affc3ba527dcb992495ecd20-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/b4005da5affc3ba527dcb992495ecd20-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: FlyView: a bio-informed optical flow truth dataset for visual navigation using panoramic stereo vision
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Alix Leroy, Graham W. Taylor
    * Abstract: Flying at speed through complex environments is a challenging task that has been performed successfully by insects since the Carboniferous, but which remains a challenge for robotic and autonomous systems. Insects navigate the world using optical flow sensed by their compound eyes, which they process using a deep neural network weighing just a few milligrams. Deploying an insect-inspired network architecture in computer vision could therefore enable more efficient and effective ways of estimating structure and self-motion using optical flow. Training a bio-informed deep network to implement these tasks requires biologically relevant training, test, and validation data. To this end, we introduce FlyView, a novel bio-informed truth dataset for visual navigation. This simulated dataset is rendered using open source 3D scenes in which the observer's position is known at every frame, and is accompanied by truth data on depth, self-motion, and motion flow. This dataset comprising 42,475 frames has several key features that are missing from existing optical flow datasets, including: (i) panoramic cameras with a monocular and binocular field of view matched to that of a fly's compound eyes; (ii) dynamically meaningful self-motion modelled on motion primitives, or the 3D trajectories of drones and flies; and (iii) complex natural and indoor environments including reflective surfaces.

count=2
* Active-Passive SimStereo - Benchmarking the Cross-Generalization Capabilities of Deep Learning-based Stereo Methods
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/bc3a68a20e5c8ba5cbefc1ecf74bfaaa-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/bc3a68a20e5c8ba5cbefc1ecf74bfaaa-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Active-Passive SimStereo - Benchmarking the Cross-Generalization Capabilities of Deep Learning-based Stereo Methods
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Laurent Jospin, Allen Antony, Lian Xu, Hamid Laga, Farid Boussaid, Mohammed Bennamoun
    * Abstract: In stereo vision, self-similar or bland regions can make it difficult to match patches between two images. Active stereo-based methods mitigate this problem by projecting a pseudo-random pattern on the scene so that each patch of an image pair can be identified without ambiguity. However, the projected pattern significantly alters the appearance of the image. If this pattern acts as a form of adversarial noise, it could negatively impact the performance of deep learning-based methods, which are now the de-facto standard for dense stereo vision. In this paper, we propose the Active-Passive SimStereo dataset and a corresponding benchmark to evaluate the performance gap between passive and active stereo images for stereo matching algorithms. Using the proposed benchmark and an additional ablation study, we show that the feature extraction and matching modules of a selection of twenty selected deep learning-based stereo matching methods generalize to active stereo without a problem. However, the disparity refinement modules of three of the twenty architectures (ACVNet, CascadeStereo, and StereoNet) are negatively affected by the active stereo patterns due to their reliance on the appearance of the input images.

count=2
* Neural Lighting Simulation for Urban Scenes
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/3d7259031023c5aa463187c4a31c95c8-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/3d7259031023c5aa463187c4a31c95c8-Paper-Conference.pdf)]
    * Title: Neural Lighting Simulation for Urban Scenes
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Ava Pun, Gary Sun, Jingkang Wang, Yun Chen, Ze Yang, Sivabalan Manivasagam, Wei-Chiu Ma, Raquel Urtasun
    * Abstract: Different outdoor illumination conditions drastically alter the appearance of urban scenes, and they can harm the performance of image-based robot perception systems if not seen during training. Camera simulation provides a cost-effective solution to create a large dataset of images captured under different lighting conditions. Towards this goal, we propose LightSim, a neural lighting camera simulation system that enables diverse, realistic, and controllable data generation. LightSim automatically builds lighting-aware digital twins at scale from collected raw sensor data and decomposes the scene into dynamic actors and static background with accurate geometry, appearance, and estimated scene lighting. These digital twins enable actor insertion, modification, removal, and rendering from a new viewpoint, all in a lighting-aware manner. LightSim then combines physically-based and learnable deferred rendering to perform realistic relighting of modified scenes, such as altering the sun location and modifying the shadows or changing the sun brightness, producing spatially- and temporally-consistent camera videos. Our experiments show that LightSim generates more realistic relighting results than prior work. Importantly, training perception models on data generated by LightSim can significantly improve their performance. Our project page is available at https://waabi.ai/lightsim/.

count=2
* AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/760dff0f9c0e9ed4d7e22918c73351d4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/760dff0f9c0e9ed4d7e22918c73351d4-Paper-Conference.pdf)]
    * Title: AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, Chenliang Xu
    * Abstract: Can machines recording an audio-visual scene produce realistic, matching audio-visual experiences at novel positions and novel view directions? We answer it by studying a new task---real-world audio-visual scene synthesis---and a first-of-its-kind NeRF-based approach for multimodal learning. Concretely, given a video recording of an audio-visual scene, the task is to synthesize new videos with spatial audios along arbitrary novel camera trajectories in that scene. We propose an acoustic-aware audio generation module that integrates prior knowledge of audio propagation into NeRF, in which we implicitly associate audio generation with the 3D geometry and material properties of a visual environment. Furthermore, we present a coordinate transformation module that expresses a view direction relative to the sound source, enabling the model to learn sound source-centric acoustic fields. To facilitate the study of this new task, we collect a high-quality Real-World Audio-Visual Scene (RWAVS) dataset. We demonstrate the advantages of our method on this real-world dataset and the simulation-based SoundSpaces dataset. Notably, we refer readers to view our demo videos for convincing comparisons.

count=2
* ReTR: Modeling Rendering Via Transformer for Generalizable Neural Surface Reconstruction
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/c47ec10bc135be5c3663ba344d29a6a5-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/c47ec10bc135be5c3663ba344d29a6a5-Paper-Conference.pdf)]
    * Title: ReTR: Modeling Rendering Via Transformer for Generalizable Neural Surface Reconstruction
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Yixun Liang, Hao He, Yingcong Chen
    * Abstract: Generalizable neural surface reconstruction techniques have attracted great attention in recent years. However, they encounter limitations of low confidence depth distribution and inaccurate surface reasoning due to the oversimplified volume rendering process employed. In this paper, we present Reconstruction TRansformer (ReTR), a novel framework that leverages the transformer architecture to redesign the rendering process, enabling complex render interaction modeling. It introduces a learnable $\textit{meta-ray token}$ and utilizes the cross-attention mechanism to simulate the interaction of rendering process with sampled points and render the observed color. Meanwhile, by operating within a high-dimensional feature space rather than the color space, ReTR mitigates sensitivity to projected colors in source views. Such improvements result in accurate surface assessment with high confidence. We demonstrate the effectiveness of our approach on various datasets, showcasing how our method outperforms the current state-of-the-art approaches in terms of reconstruction quality and generalization ability. $\textit{Our code is available at }$ https://github.com/YixunLiang/ReTR.

count=1
* Multi-View Consistency Loss for Improved Single-Image 3D Reconstruction of Clothed People
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Caliskan_Multi-View_Consistency_Loss_for_Improved_Single-Image_3D_Reconstruction_of_Clothed_ACCV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2020/papers/Caliskan_Multi-View_Consistency_Loss_for_Improved_Single-Image_3D_Reconstruction_of_Clothed_ACCV_2020_paper.pdf)]
    * Title: Multi-View Consistency Loss for Improved Single-Image 3D Reconstruction of Clothed People
    * Publisher: ACCV
    * Publication Date: `2020`
    * Authors: Akin Caliskan, Armin Mustafa, Evren Imre, Adrian Hilton
    * Abstract: We present a novel method to improve the accuracy of the 3D reconstruction of clothed human shape from a single image. Recent work has introduced volumetric, implicit and model-based shape learning frameworks for reconstruction of objects and people from one or more images. However, the accuracy and completeness for reconstruction of clothed people is limited due to the large variation in shape resulting from clothing, hair, body size, pose and camera viewpoint. This paper introduces two advances to overcome this limitation: firstly a new synthetic dataset of realistic clothed people, 3DVH;and secondly, a novel multiple-view loss function for training of monocular volumetric shape estimation, which is demonstrated to significantly improve generalisation and reconstruction accuracy. The 3DVH dataset of realistic clothed 3D human models rendered with diverse natural backgrounds is demonstrated to allows transfer to reconstruction from real images of people. Comprehensive comparative performance evaluation on both synthetic and real images of people demonstrates that the proposed method significantly outperforms the previous state-of-the-art learning-based single image 3D human shape estimation approaches achieving significant improvement of reconstruction accuracy, completeness, and quality. An ablation study shows that this is due to both the proposed multiple-view training and the new 3DVH dataset. The code and the dataset can be found at the project website: https://akincaliskan3d.github.io/MV3DH/.

count=1
* The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2022/html/Bonfiglioli_The_Eyecandies_Dataset_for_Unsupervised_Multimodal_Anomaly_Detection_and_Localization_ACCV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2022/papers/Bonfiglioli_The_Eyecandies_Dataset_for_Unsupervised_Multimodal_Anomaly_Detection_and_Localization_ACCV_2022_paper.pdf)]
    * Title: The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization
    * Publisher: ACCV
    * Publication Date: `2022`
    * Authors: Luca Bonfiglioli, Marco Toschi, Davide Silvestri, Nicola Fioraio, Daniele De Gregorio
    * Abstract: We present Eyecandies, a novel synthetic dataset for unsupervised anomaly detection and localization. Photo-realistic images of procedurally generated candies are rendered in a controlled environment under multiple lightning conditions, also providing depth and normal maps in an industrial conveyor scenario. We make available anomaly-free samples for model training and validation, while anomalous instances with precise ground-truth annotations are provided only in the test set. The dataset comprises ten classes of candies, each showing different challenges, such as complex textures, self-occlusions and specularities. Furthermore, we achieve large intra-class variation by randomly drawing key parameters of a procedural rendering pipeline, which enables the creation of an arbitrary number of instances with photo-realistic appearance. Likewise, anomalies are injected into the rendering graph and pixel-wise annotations are automatically generated, overcoming human-biases and possible inconsistencies. We believe this dataset may encourage the exploration of original approaches to solve the anomaly detection task, e.g. by combining color, depth and normal maps, as they are not provided by most of the existing datasets. Indeed, in order to demonstrate how exploiting additional information may actually lead to higher detection performance, we show the results obtained by training a deep convolutional autoencoder to reconstruct different combinations of inputs.

count=1
* Neural Deformable Voxel Grid for Fast Optimization of Dynamic View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2022/html/Guo_Neural_Deformable_Voxel_Grid_for_Fast_Optimization_of_Dynamic_View_ACCV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2022/papers/Guo_Neural_Deformable_Voxel_Grid_for_Fast_Optimization_of_Dynamic_View_ACCV_2022_paper.pdf)]
    * Title: Neural Deformable Voxel Grid for Fast Optimization of Dynamic View Synthesis
    * Publisher: ACCV
    * Publication Date: `2022`
    * Authors: Xiang Guo, Guanying CHEN, Yuchao Dai, Xiaoqing Ye, Jiadai Sun, Xiao Tan, Errui Ding
    * Abstract: Recently, Neural Radiance Fields (NeRF) is revolutionizing the task of novel view synthesis (NVS) for its superior performance. In this paper, we propose to synthesize dynamic scenes. Extending the methods for static scenes to dynamic scenes is not straightforward as both the scene geometry and appearance change over time, especially under monocular setup. Also, the existing dynamic NeRF methods gen- erally require a lengthy per-scene training procedure, where multi-layer perceptrons (MLP) are fitted to model both motions and radiance. In this paper, built on top of the recent advances in voxel-grid optimization, we propose a fast deformable radiance field method to handle dynamic scenes. Our method consists of two modules. The first module adopts a deformation grid to store 3D dynamic features, and a light-weight MLP for decoding the deformation that maps a 3D point in the observation space to the canonical space using the interpolated features. The second module contains a density and a color grid to model the geometry and density of the scene. The occlusion is explicitly modeled to further im- prove the rendering quality. Experimental results show that our method achieves comparable performance to D-NeRF using only 20 minutes for training, which is more than 70x faster than D-NeRF, clearly demon- strating the efficiency of our proposed method.

count=1
* SymmNeRF: Learning to Explore Symmetry Prior for Single-View View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2022/html/Li_SymmNeRF_Learning_to_Explore_Symmetry_Prior_for_Single-View_View_Synthesis_ACCV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2022/papers/Li_SymmNeRF_Learning_to_Explore_Symmetry_Prior_for_Single-View_View_Synthesis_ACCV_2022_paper.pdf)]
    * Title: SymmNeRF: Learning to Explore Symmetry Prior for Single-View View Synthesis
    * Publisher: ACCV
    * Publication Date: `2022`
    * Authors: Xingyi Li, Chaoyi Hong, Yiran Wang, Zhiguo Cao, Ke Xian, Guosheng Lin
    * Abstract: We study the problem of novel view synthesis of objects from a single image. Existing methods have demonstrated the potential in single-view view synthesis. However, they still fail to recover the fine appearance details, especially in self-occluded areas. This is because a single view only provides limited information. We observe that manmade objects usually exhibit symmetric appearances, which introduce additional prior knowledge. Motivated by this, we investigate the potential performance gains of explicitly embedding symmetry into the scene representation. In this paper, we propose SymmNeRF, a neural radiance field (NeRF) based framework that combines local and global conditioning under the introduction of symmetry priors. In particular, SymmNeRF takes the pixel-aligned image features and the corresponding symmetric features as extra inputs to the NeRF, whose parameters are generated by a hypernetwork. As the parameters are conditioned on the image-encoded latent codes, SymmNeRF is thus scene-independent and can generalize to new scenes. Experiments on synthetic and real-world datasets show that SymmNeRF synthesizes novel views with more details regardless of the pose transformation, and demonstrates good generalization when applied to unseen objects. Code is available at: https://github.com/xingyi-li/SymmNeRF.

count=1
* InstantGeoAvatar: Effective Geometry and Appearance Modeling of Animatable Avatars from Monocular Video
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Budria_InstantGeoAvatar_Effective_Geometry_and_Appearance_Modeling_of_Animatable_Avatars_from_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Budria_InstantGeoAvatar_Effective_Geometry_and_Appearance_Modeling_of_Animatable_Avatars_from_ACCV_2024_paper.pdf)]
    * Title: InstantGeoAvatar: Effective Geometry and Appearance Modeling of Animatable Avatars from Monocular Video
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Alvaro Budria, Adrian Lopez-Rodriguez, Òscar Lorente, Francesc Moreno-Noguer
    * Abstract: We present InstantGeoAvatar, a method for efficient and effective learning from monocular video of detailed 3D geometry and appearance of animatable implicit human avatars. Our key observation is that the optimization of a hash grid encoding to represent a signed distance function (SDF) of the human subject is fraught with instabilities and bad local minima. We thus propose a principled geometry-aware SDF regularization scheme that seamlessly fits into the volume rendering pipeline and adds negligible computational overhead. Our regularization scheme significantly outperforms previous approaches for training SDFs on hash grids. We obtain competitive results in geometry reconstruction and novel view synthesis in as little as five minutes of training time, a significant reduction from the several hours required by previous work. InstantGeoAvatar represents a significant leap forward towards achieving interactive reconstruction of virtual avatars.

count=1
* Efficient Implicit SDF and Color Reconstruction via Shared Feature Field
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Fang_Efficient_Implicit_SDF_and_Color_Reconstruction_via_Shared_Feature_Field_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Fang_Efficient_Implicit_SDF_and_Color_Reconstruction_via_Shared_Feature_Field_ACCV_2024_paper.pdf)]
    * Title: Efficient Implicit SDF and Color Reconstruction via Shared Feature Field
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Shuangkang Fang, Dacheng Qi, Weixin Xu, Yufeng Wang, Zehao Zhang, Xiaorong Zhang, Huayu Zhang, Zeqi Shao, Wenrui Ding
    * Abstract: Recent advancements in neural implicit 3D representations have enabled simultaneous surface reconstruction and novel view synthesis using only 2D RGB images. However, these methods often struggle with textureless and minimally visible areas. In this study, we introduce a simple yet effective encoder-decoder framework that encodes positional and viewpoint coordinates into a shared feature field (SFF). This feature field is then decoded into an implicit signed distance field (SDF) and a color field. By employing a weight-sharing encoder, we enhance the joint optimization of the SDF and color field, enabling better utilization of the limited information in the scene. Additionally, we incorporate a periodic sine function as an activation function, eliminating the need for a positional encoding layer and significantly reducing rippling artifacts on surfaces. Empirical results demonstrate that our method more effectively reconstructs textureless and minimally visible surfaces, synthesizes higher-quality novel views, and achieves superior multi-view reconstruction with fewer input images.

count=1
* Joint 3D Scene Reconstruction and Class Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Hane_Joint_3D_Scene_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Hane_Joint_3D_Scene_2013_CVPR_paper.pdf)]
    * Title: Joint 3D Scene Reconstruction and Class Segmentation
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Christian Hane, Christopher Zach, Andrea Cohen, Roland Angst, Marc Pollefeys
    * Abstract: Both image segmentation and dense 3D modeling from images represent an intrinsically ill-posed problem. Strong regularizers are therefore required to constrain the solutions from being 'too noisy'. Unfortunately, these priors generally yield overly smooth reconstructions and/or segmentations in certain regions whereas they fail in other areas to constrain the solution sufficiently. In this paper we argue that image segmentation and dense 3D reconstruction contribute valuable information to each other's task. As a consequence, we propose a rigorous mathematical framework to formulate and solve a joint segmentation and dense reconstruction problem. Image segmentations provide geometric cues about which surface orientations are more likely to appear at a certain location in space whereas a dense 3D reconstruction yields a suitable regularization for the segmentation problem by lifting the labeling from 2D images to 3D space. We show how appearance-based cues and 3D surface orientation priors can be learned from training data and subsequently used for class-specific regularization. Experimental results on several real data sets highlight the advantages of our joint formulation.

count=1
* Reconstructing Gas Flows Using Light-Path Approximation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Ji_Reconstructing_Gas_Flows_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ji_Reconstructing_Gas_Flows_2013_CVPR_paper.pdf)]
    * Title: Reconstructing Gas Flows Using Light-Path Approximation
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Yu Ji, Jinwei Ye, Jingyi Yu
    * Abstract: Transparent gas flows are difficult to reconstruct: the refractive index field (RIF) within the gas volume is uneven and rapidly evolving, and correspondence matching under distortions is challenging. We present a novel computational imaging solution by exploiting the light field probe (LFProbe). A LF-probe resembles a view-dependent pattern where each pixel on the pattern maps to a unique ray. By observing the LF-probe through the gas flow, we acquire a dense set of ray-ray correspondences and then reconstruct their light paths. To recover the RIF, we use Fermat's Principle to correlate each light path with the RIF via a Partial Differential Equation (PDE). We then develop an iterative optimization scheme to solve for all light-path PDEs in conjunction. Specifically, we initialize the light paths by fitting Hermite splines to ray-ray correspondences, discretize their PDEs onto voxels, and solve a large, over-determined PDE system for the RIF. The RIF can then be used to refine the light paths. Finally, we alternate the RIF and light-path estimations to improve the reconstruction. Experiments on synthetic and real data show that our approach can reliably reconstruct small to medium scale gas flows. In particular, when the flow is acquired by a small number of cameras, the use of ray-ray correspondences can greatly improve the reconstruction.

count=1
* Light Field Distortion Feature for Transparent Object Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Maeno_Light_Field_Distortion_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Maeno_Light_Field_Distortion_2013_CVPR_paper.pdf)]
    * Title: Light Field Distortion Feature for Transparent Object Recognition
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Kazuki Maeno, Hajime Nagahara, Atsushi Shimada, Rin-Ichiro Taniguchi
    * Abstract: Current object-recognition algorithms use local features, such as scale-invariant feature transform (SIFT) and speeded-up robust features (SURF), for visually learning to recognize objects. These approaches though cannot apply to transparent objects made of glass or plastic, as such objects take on the visual features of background objects, and the appearance of such objects dramatically varies with changes in scene background. Indeed, in transmitting light, transparent objects have the unique characteristic of distorting the background by refraction. In this paper, we use a single-shot light Aeld image as an input and model the distortion of the light Aeld caused by the refractive property of a transparent object. We propose a new feature, called the light Aeld distortion (LFD) feature, for identifying a transparent object. The proposal incorporates this LFD feature into the bag-of-features approach for recognizing transparent objects. We evaluated its performance in laboratory and real settings.

count=1
* Underwater Camera Calibration Using Wavelength Triangulation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Yau_Underwater_Camera_Calibration_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Yau_Underwater_Camera_Calibration_2013_CVPR_paper.pdf)]
    * Title: Underwater Camera Calibration Using Wavelength Triangulation
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Timothy Yau, Minglun Gong, Yee-Hong Yang
    * Abstract: In underwater imagery, the image formation process includes refractions that occur when light passes from water into the camera housing, typically through a flat glass port. We extend the existing work on physical refraction models by considering the dispersion of light, and derive new constraints on the model parameters for use in calibration. This leads to a novel calibration method that achieves improved accuracy compared to existing work. We describe how to construct a novel calibration device for our method and evaluate the accuracy of the method through synthetic and real experiments.

count=1
* Large Scale Multi-view Stereopsis Evaluation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2014/html/Jensen_Large_Scale_Multi-view_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Jensen_Large_Scale_Multi-view_2014_CVPR_paper.pdf)]
    * Title: Large Scale Multi-view Stereopsis Evaluation
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, Henrik Aanaes
    * Abstract: The seminal multiple view stereo benchmark evaluations from Middlebury and by Strecha et al. have played a major role in propelling the development of multi-view stereopsis methodology. Although seminal, these benchmark datasets are limited in scope with few reference scenes. Here, we try to take these works a step further by proposing a new multi-view stereo dataset, which is an order of magnitude larger in number of scenes and with a significant increase in diversity. Specifically, we propose a dataset containing 80 scenes of large variability. Each scene consists of 49 or 64 accurate camera positions and reference structured light scans, all acquired by a 6-axis industrial robot. To apply this dataset we propose an extension of the evaluation protocol from the Middlebury evaluation, reflecting the more complex geometry of some of our scenes. The proposed dataset is used to evaluate the state of the art multiview stereo algorithms of Tola et al., Campbell et al. and Furukawa et al. Hereby we demonstrate the usability of the dataset as well as gain insight into the workings and challenges of multi-view stereopsis. Through these experiments we empirically validate some of the central hypotheses of multi-view stereopsis, as well as determining and reaffirming some of the central challenges.

count=1
* A Fixed Viewpoint Approach for Dense Reconstruction of Transparent Objects
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Han_A_Fixed_Viewpoint_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Han_A_Fixed_Viewpoint_2015_CVPR_paper.pdf)]
    * Title: A Fixed Viewpoint Approach for Dense Reconstruction of Transparent Objects
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Kai Han, Kwan-Yee K. Wong, Miaomiao Liu
    * Abstract: This paper addresses the problem of reconstructing the surface shape of transparent objects. The difficulty of this problem originates from the viewpoint dependent appearance of a transparent object, which quickly makes reconstruction methods tailored for diffuse surfaces fail disgracefully. In this paper, we develop a fixed viewpoint approach for dense surface reconstruction of transparent objects based on refraction of light. We introduce a simple setup that allows us alter the incident light paths before light rays enter the object, and develop a method for recovering the object surface based on reconstructing and triangulating such incident light paths. Our proposed approach does not need to model the complex interactions of light as it travels through the object, neither does it assume any parametric form for the shape of the object nor the exact number of refractions and reflections taken place along the light paths. It can therefore handle transparent objects with a complex shape and structure, with unknown and even inhomogeneous refractive index. Experimental results on both synthetic and real data are presented which demonstrate the feasibility and accuracy of our proposed approach.

count=1
* First-Person Pose Recognition Using Egocentric Workspaces
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Rogez_First-Person_Pose_Recognition_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Rogez_First-Person_Pose_Recognition_2015_CVPR_paper.pdf)]
    * Title: First-Person Pose Recognition Using Egocentric Workspaces
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Gregory Rogez, James S. Supancic III, Deva Ramanan
    * Abstract: We tackle the problem of estimating the 3D pose of an individual's upper limbs (arms+hands) from a chest mounted depth-camera. Importantly, we consider pose estimation during everyday interactions with objects. Past work shows that strong pose+viewpoint priors and depth-based features are crucial for robust performance. In egocentric views, hands and arms are observable within a well defined volume in front of the camera. We call this volume an egocentric workspace. A notable property is that hand appearance correlates with workspace location. To exploit this correlation, we classify arm+hand configurations in a global egocentric coordinate frame, rather than a local scanning window. This greatly simplify the architecture and improves performance. We propose an efficient pipeline which 1) generates synthetic workspace exemplars for training using a virtual chest-mounted camera whose intrinsic parameters match our physical camera, 2) computes perspective-aware depth features on this entire volume and 3) recognizes discrete arm+hand pose classes through a sparse multi-class SVM. We achieve state-of-the-art hand pose recognition performance from egocentric RGB-D images in real-time.

count=1
* Discrete Optimization of Ray Potentials for Semantic 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Savinov_Discrete_Optimization_of_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Savinov_Discrete_Optimization_of_2015_CVPR_paper.pdf)]
    * Title: Discrete Optimization of Ray Potentials for Semantic 3D Reconstruction
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Nikolay Savinov, Lubor Ladicky, Christian Hane, Marc Pollefeys
    * Abstract: Dense semantic 3D reconstruction is typically formulated as a discrete or continuous problem over label assignments in a voxel grid, combining semantic and depth likelihoods in a Markov Random Field framework. The depth and semantic information is incorporated as a unary potential, smoothed by a pairwise regularizer. However, modelling likelihoods as a unary potential does not model the problem correctly leading to various undesirable visibility artifacts. We propose to formulate an optimization problem that directly optimizes the reprojection error of the 3D model with respect to the image estimates, which corresponds to the optimization over rays, where the cost function depends on the semantic class and depth of the first occupied voxel along the ray. The 2-label formulation is made feasible by transforming it into a graph-representable form under QPBO relaxation, solvable using graph cut. The multi-label problem is solved by applying $\alpha$-expansion using the same relaxation in each expansion move. Our method was indeed shown to be feasible in practice, running comparably fast to the competing methods, while not suffering from ray potential approximation artifacts.

count=1
* Shape and Light Directions From Shading and Polarization
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Thanh_Shape_and_Light_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Thanh_Shape_and_Light_2015_CVPR_paper.pdf)]
    * Title: Shape and Light Directions From Shading and Polarization
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Trung Ngo Thanh, Hajime Nagahara, Rin-ichiro Taniguchi
    * Abstract: We introduce a method to recover the shape of a smooth dielectric object from polarization images taken with a light source from different directions. We present two constraints on shading and polarization and use both in a single optimization scheme. This integration is motivated by the fact that photometric stereo and polarization-based methods have complementary abilities. The polarization-based method can give strong cues for the surface orientation and refractive index, which are independent of the light direction. However, it has ambiguities in selecting between two ambiguous choices of the surface orientation, in the relationship between refractive index and zenith angle (observing angle), and limited performance for surface points with small zenith angles, where the polarization effect is weak. In contrast, photometric stereo method with multiple light sources can disambiguate the surface orientation and give a strong relationship between the surface normals and light directions. However, it has limited performance for large zenith angles, refractive index estimation, and faces the ambiguity in case the light direction is unknown. Taking their advantages, our proposed method can recover the surface normals for both small and large zenith angles, the light directions, and the refractive indexes of the object. The proposed method is successfully evaluated by simulation and real-world experiments.

count=1
* 24/7 Place Recognition by View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Torii_247_Place_Recognition_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Torii_247_Place_Recognition_2015_CVPR_paper.pdf)]
    * Title: 24/7 Place Recognition by View Synthesis
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Akihiko Torii, Relja Arandjelovic, Josef Sivic, Masatoshi Okutomi, Tomas Pajdla
    * Abstract: We address the problem of large-scale visual place recognition for situations where the scene undergoes a major change in appearance, for example, due to illumination (day/night), change of seasons, aging, or structural modifications over time such as buildings built or destroyed. Such situations represent a major challenge for current large-scale place recognition methods. This work has the following three principal contributions. First, we demonstrate that matching across large changes in the scene appearance becomes much easier when both the query image and the database image depict the scene from approximately the same viewpoint. Second, based on this observation, we develop a new place recognition approach that combines (i) an efficient synthesis of novel views with (ii) a compact indexable image representation. Third, we introduce a new challenging dataset of 1,125 camera-phone query images of Tokyo that contain major changes in illumination (day, sunset, night) as well as structural changes in the scene. We demonstrate that the proposed approach significantly outperforms other large-scale place recognition techniques on this challenging data.

count=1
* Panoramic Stereo Videos With a Single Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Aggarwal_Panoramic_Stereo_Videos_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Aggarwal_Panoramic_Stereo_Videos_CVPR_2016_paper.pdf)]
    * Title: Panoramic Stereo Videos With a Single Camera
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Rajat Aggarwal, Amrisha Vohra, Anoop M. Namboodiri
    * Abstract: We present a practical solution for generating 360 degree stereo panoramic videos using a single camera. Current approaches either use a moving camera that captures multiple images of a scene, which are then stitched together to form the final panorama, or use multiple cameras that are synchronized. A moving camera limits the solution to static scenes, while multi-camera solutions require dedicated calibrated setups. Our approach improves upon the existing solutions in two significant ways: It solves the problem using a single camera, thus minimizing the calibration problem and providing us the ability to convert any digital camera into a panoramic stereo capture device. It captures all the light rays required for stereo panoramas in a single frame using a compact custom designed mirror, thus making the design practical to manufacture and easier to use. We analyze several properties of the design as well as present panoramic stereo and depth estimation results.

count=1
* Mirror Surface Reconstruction Under an Uncalibrated Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Han_Mirror_Surface_Reconstruction_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Han_Mirror_Surface_Reconstruction_CVPR_2016_paper.pdf)]
    * Title: Mirror Surface Reconstruction Under an Uncalibrated Camera
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Kai Han, Kwan-Yee K. Wong, Dirk Schnieders, Miaomiao Liu
    * Abstract: This paper addresses the problem of mirror surface reconstruction, and a solution based on observing the reflections of a moving reference plane on the mirror surface is proposed. Unlike previous approaches which require tedious work to calibrate the camera, our method can recover both the camera intrinsics and extrinsics together with the mirror surface from reflections of the reference plane under at least three unknown distinct poses. Our previous work has demonstrated that 3D poses of the reference plane can be registered in a common coordinate system using reflection correspondences established across images. This leads to a bunch of registered 3D lines formed from the reflection correspondences. Given these lines, we first derive an analytical solution to recover the camera projection matrix through estimating the line projection matrix. We then optimize the camera projection matrix by minimizing reprojection errors computed based on a cross-ratio formulation. The mirror surface is finally reconstructed based on the optimized cross-ratio constraint. Experimental results on both synthetic and real data are presented, which demonstrate the feasibility and accuracy of our method.

count=1
* 3D Reconstruction of Transparent Objects With Position-Normal Consistency
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Qian_3D_Reconstruction_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Qian_3D_Reconstruction_of_CVPR_2016_paper.pdf)]
    * Title: 3D Reconstruction of Transparent Objects With Position-Normal Consistency
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Yiming Qian, Minglun Gong, Yee Hong Yang
    * Abstract: Estimating the shape of transparent and refractive objects is one of the few open problems in 3D reconstruction. Under the assumption that the rays refract only twice when traveling through the object, we present the first approach to simultaneously reconstructing the 3D positions and normals of the object's surface at both refraction locations. Our acquisition setup requires only two cameras and one monitor, which serves as the light source. After acquiring the ray-ray correspondences between each camera and the monitor, we solve an optimization function which enforces a new position-normal consistency constraint. That is, the 3D positions of surface points shall agree with the normals required to refract the rays under Snell's law. Experimental results using both synthetic and real data demonstrate the robustness and accuracy of the proposed approach.

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
* Real Time Complete Dense Depth Reconstruction for a Monocular Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w17/html/Huang_Real_Time_Complete_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w17/papers/Huang_Real_Time_Complete_CVPR_2016_paper.pdf)]
    * Title: Real Time Complete Dense Depth Reconstruction for a Monocular Camera
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Xiaoshui Huang, Lixin Fan, Jian Zhang, Qiang Wu, Chun Yuan
    * Abstract: In this paper, we aim to solve the problem of estimating complete dense depth maps from a monocular moving camera. By 'complete', we mean depth information is estimated for every pixel and detailed reconstruction is achieved. Although this problem has previously been attempted, the accuracy of complete dense depth reconstruction is a remaining problem. We propose a novel system which produces accurate complete dense depth map. The new system consists of two subsystems running in separated threads, namely, dense mapping and sparse patch-based tracking. For dense mapping, a new projection error computation method is proposed to enhance the gradient component in estimated depth maps. For tracking, a new sparse patch-based tracking method estimates camera pose by minimizing a normalized error term. The experiments demonstrate that the proposed method obtains improved performance in terms of completeness and accuracy compared to three state-of-the-art dense reconstruction methods.

count=1
* UAV-Based Autonomous Image Acquisition With Multi-View Stereo Quality Assurance by Confidence Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w3/html/Mostegel_UAV-Based_Autonomous_Image_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w3/papers/Mostegel_UAV-Based_Autonomous_Image_CVPR_2016_paper.pdf)]
    * Title: UAV-Based Autonomous Image Acquisition With Multi-View Stereo Quality Assurance by Confidence Prediction
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Christian Mostegel, Markus Rumpler, Friedrich Fraundorfer, Horst Bischof
    * Abstract: In this paper we present an autonomous system for acquiring close-range high-resolution images that maximize the quality of a later-on 3D reconstruction with respect to coverage, ground resolution and 3D uncertainty. In contrast to previous work, our system uses the already acquired images to predict the confidence in the output of a dense multi-view stereo approach without executing it. This confidence encodes the likelihood of a successful reconstruction with respect to the observed scene and potential camera constellations. Our prediction module runs in real-time and can be trained without any externally recorded ground truth. We use the confidence prediction for on-site quality assurance and for planning further views that are tailored for a specific multi-view stereo approach with respect to the given scene. We demonstrate the capabilities of our approach with an autonomous Unmanned Aerial Vehicle (UAV) in a challenging outdoor scenario.

count=1
* Shape Completion Using 3D-Encoder-Predictor CNNs and Shape Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Dai_Shape_Completion_Using_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_Shape_Completion_Using_CVPR_2017_paper.pdf)]
    * Title: Shape Completion Using 3D-Encoder-Predictor CNNs and Shape Synthesis
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Angela Dai, Charles Ruizhongtai Qi, Matthias Niessner
    * Abstract: We introduce a data-driven approach to complete partial 3D shapes through a combination of volumetric deep neural networks and 3D shape synthesis. From a partially-scanned input shape, our method first infers a low-resolution -- but complete -- output. To this end, we introduce a 3D-Encoder-Predictor Network (3D-EPN) which is composed of 3D convolutional layers. The network is trained to predict and fill in missing data, and operates on an implicit surface representation that encodes both known and unknown space. This allows us to predict global structure in unknown areas at high accuracy. We then correlate these intermediary results with 3D geometry from a shape database at test time. In a final pass, we propose a patch-based 3D shape synthesis method that imposes the 3D geometry from these retrieved shapes as constraints on the coarsely-completed mesh. This synthesis process enables us to reconstruct fine-scale detail and generate high-resolution output while respecting the global mesh structure obtained by the 3D-EPN. Although our 3D-EPN outperforms state-of-the-art completion method, the main contribution in our work lies in the combination of a data-driven shape predictor and analytic 3D shape synthesis. In our results, we show extensive evaluations on a newly-introduced shape completion benchmark for both real-world and synthetic data.

count=1
* Real-Time 3D Model Tracking in Color and Depth on a Single CPU Core
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Kehl_Real-Time_3D_Model_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Kehl_Real-Time_3D_Model_CVPR_2017_paper.pdf)]
    * Title: Real-Time 3D Model Tracking in Color and Depth on a Single CPU Core
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Wadim Kehl, Federico Tombari, Slobodan Ilic, Nassir Navab
    * Abstract: We present a novel method to track 3D models in color and depth data. To this end, we introduce approximations that accelerate the state-of-the-art in region-based tracking by an order of magnitude while retaining similar accuracy. Furthermore, we show how the method can be made more robust in the presence of depth data and consequently formulate a new joint contour and ICP tracking energy. We present better results than the state-of-the-art while being much faster then most other methods and achieving all of the above on a single CPU core.

count=1
* Deep Supervision With Shape Concepts for Occlusion-Aware 3D Object Parsing
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Deep_Supervision_With_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Deep_Supervision_With_CVPR_2017_paper.pdf)]
    * Title: Deep Supervision With Shape Concepts for Occlusion-Aware 3D Object Parsing
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Chi Li, M. Zeeshan Zia, Quoc-Huy Tran, Xiang Yu, Gregory D. Hager, Manmohan Chandraker
    * Abstract: Monocular 3D object parsing is highly desirable in various scenarios including occlusion reasoning and holistic scene interpretation. We present a deep convolutional neural network (CNN) architecture to localize semantic parts in 2D image and 3D space while inferring their visibility states, given a single RGB image. Our key insight is to exploit domain knowledge to regularize the network by deeply supervising its hidden layers, in order to sequentially infer intermediate concepts associated with the final task. To acquire training data in desired quantities with ground truth 3D shape and relevant concepts, we render 3D object CAD models to generate large-scale synthetic data and simulate challenging occlusion configurations between objects. We train the network only on synthetic data and demonstrate state-of-the-art performances on real image benchmarks including an extended version of KITTI, PASCAL VOC, PASCAL3D+ and IKEA for 2D and 3D keypoint localization and instance segmentation. The empirical results substantiate the utility of our deep supervision scheme by demonstrating effective transfer of knowledge from synthetic data to real images, resulting in less overfitting compared to standard end-to-end training.

count=1
* Semi-Calibrated Near Field Photometric Stereo
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Logothetis_Semi-Calibrated_Near_Field_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Logothetis_Semi-Calibrated_Near_Field_CVPR_2017_paper.pdf)]
    * Title: Semi-Calibrated Near Field Photometric Stereo
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Fotios Logothetis, Roberto Mecca, Roberto Cipolla
    * Abstract: 3D reconstruction from shading information through Photometric Stereo is considered a very challenging problem in Computer Vision. Although this technique can potentially provide highly detailed shape recovery, its accuracy is critically dependent on a numerous set of factors among them the reliability of the light sources in emitting a constant amount of light. In this work, we propose a novel variational approach to solve the so called semi-calibrated near field Photometric Stereo problem, where the positions but not the brightness of the light sources are known. Additionally, we take into account realistic modeling features such as perspective viewing geometry and heterogeneous scene composition, containing both diffuse and specular objects. Furthermore, we also relax the point light source assumption that usually constraints the near field formulation by explicitly calculating the light attenuation maps. Synthetic experiments are performed for quantitative evaluation for a wide range of cases whilst real experiments provide comparisons, qualitatively outperforming the state of the art.

count=1
* Multi-View Supervision for Single-View Reconstruction via Differentiable Ray Consistency
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Tulsiani_Multi-View_Supervision_for_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tulsiani_Multi-View_Supervision_for_CVPR_2017_paper.pdf)]
    * Title: Multi-View Supervision for Single-View Reconstruction via Differentiable Ray Consistency
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Shubham Tulsiani, Tinghui Zhou, Alexei A. Efros, Jitendra Malik
    * Abstract: We study the notion of consistency between a 3D shape and a 2D observation and propose a differentiable formulation which allows computing gradients of the 3D shape given an observation from an arbitrary view. We do so by reformulating view consistency using a differentiable ray consistency (DRC) term. We show that this formulation can be incorporated in a learning framework to leverage different types of multi-view observations e.g. foreground masks, depth, color images, semantics etc. as supervision for learning single-view 3D prediction. We present empirical analysis of our technique in a controlled setting. We also show that this approach allows us to improve over existing techniques for single-view reconstruction of objects from the PASCAL VOC dataset.

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
* Earth Observation Using SAR and Social Media Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Wang_Earth_Observation_Using_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Wang_Earth_Observation_Using_CVPR_2017_paper.pdf)]
    * Title: Earth Observation Using SAR and Social Media Images
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yuanyuan Wang, Xiao Xiang Zhu
    * Abstract: Earth Observation (EO) is mostly carried out through centralized optical and synthetic aperture radar (SAR) missions. Despite the controlled quality of their products, such observation is restricted by the characteristics of the sensor platform, e.g. the revisit time. Over the last decade, the rapid development of social media has accumulated vast amount of online images. Despite their uncontrolled quality, the sheer volume may contain useful information that can complement the EO missions, especially the SAR missions. This paper presents a preliminary work of fusing social media and SAR images. They have distinct imaging geometries, which are nearly impossible to even coregister without a precise 3-D model. We describe a general approach to coregister them without using external 3-D model. We demonstrate that, one can obtain a new kind of 3-D city model that includes the optical texture for better scene understanding and the precise deformation retrieved from SAR interferometry.

count=1
* A Logarithmic X-Ray Imaging Model for Baggage Inspection: Simulation and Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w3/html/Mery_A_Logarithmic_X-Ray_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w3/papers/Mery_A_Logarithmic_X-Ray_CVPR_2017_paper.pdf)]
    * Title: A Logarithmic X-Ray Imaging Model for Baggage Inspection: Simulation and Object Detection
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Domingo Mery, Aggelos K. Katsaggelos
    * Abstract: In baggage inspection the aim is to detect automatically threat objects. The progress in automated baggage inspection, however, is modest and very limited. In this work, we present an X-ray imaging model that can separate foreground from background in baggage screening. In our model, rather than a multiplication of foreground and background, we propose the addition of logarithmic images. This allows the use of linear strategies to superimpose images of threat objects onto X-ray images (simulation) and the use of sparse representations in order segment target objects (detection). In our experiments, we simulate new X-ray images of handguns, shuriken and razor blades, in which it is impossible to distinguish simulated and real X-ray images. In addition, we show in our experiments the effective detection of shuriken, razor blades and handguns using the proposed algorithm.

count=1
* Reconstructing Thin Structures of Manifold Surfaces by Integrating Spatial Curves
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Reconstructing_Thin_Structures_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Reconstructing_Thin_Structures_CVPR_2018_paper.pdf)]
    * Title: Reconstructing Thin Structures of Manifold Surfaces by Integrating Spatial Curves
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Shiwei Li, Yao Yao, Tian Fang, Long Quan
    * Abstract: The manifold surface reconstruction in multi-view stereo often fails in retaining thin structures due to incomplete and noisy reconstructed point clouds. In this paper, we address this problem by leveraging spatial curves. The curve representation in nature is advantageous in modeling thin and elongated structures, implying topology and connectivity information of the underlying geometry, which exactly compensates the weakness of scattered point clouds. We present a novel surface reconstruction method using both curves and point clouds. First, we propose a 3D curve reconstruction algorithm based on the initialize-optimize-expand strategy. Then, tetrahedra are constructed from points and curves, where the volumes of thin structures are robustly preserved by the Curve-conformed Delaunay Refinement. Finally, the mesh surface is extracted from tetrahedra by a graph optimization. The method has been intensively evaluated on both synthetic and real-world datasets, showing significant improvements over state-of-the-art methods.

count=1
* Divide and Conquer for Full-Resolution Light Field Deblurring
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Mohan_Divide_and_Conquer_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mohan_Divide_and_Conquer_CVPR_2018_paper.pdf)]
    * Title: Divide and Conquer for Full-Resolution Light Field Deblurring
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: M. R. Mahesh Mohan, A. N. Rajagopalan
    * Abstract: The increasing popularity of computational light field (LF) cameras has necessitated the need for tackling motion blur which is a ubiquitous phenomenon in hand-held photography. The state-of-the-art method for blind deblurring of LFs of general 3D scenes is limited to handling only downsampled LF, both in spatial and angular resolution. This is due to the computational overhead involved in processing data-hungry full-resolution 4D LF altogether. Moreover, the method warrants high-end GPUs for optimization and is ineffective for wide-angle settings and irregular camera motion. In this paper, we introduce a new blind motion deblurring strategy for LFs which alleviates these limitations significantly. Our model achieves this by isolating 4D LF motion blur across the 2D subaperture images, thus paving the way for independent deblurring of these subaperture images. Furthermore, our model accommodates common camera motion parameterization across the subaperture images. Consequently, blind deblurring of any single subaperture image elegantly paves the way for cost-effective non-blind deblurring of the other subaperture images. Our approach is CPU-efficient computationally and can effectively deblur full-resolution LFs.

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
* Augmenting Crowd-Sourced 3D Reconstructions Using Semantic Detections
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Price_Augmenting_Crowd-Sourced_3D_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Price_Augmenting_Crowd-Sourced_3D_CVPR_2018_paper.pdf)]
    * Title: Augmenting Crowd-Sourced 3D Reconstructions Using Semantic Detections
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: True Price, Johannes L. Schönberger, Zhen Wei, Marc Pollefeys, Jan-Michael Frahm
    * Abstract: Image-based 3D reconstruction for Internet photo collections has become a robust technology to produce impressive virtual representations of real-world scenes. However, several fundamental challenges remain for Structure-from-Motion (SfM) pipelines, namely: the placement and reconstruction of transient objects only observed in single views, estimating the absolute scale of the scene, and (suprisingly often) recovering ground surfaces in the scene. We propose a method to jointly address these remaining open problems of SfM. In particular, we focus on detecting people in individual images and accurately placing them into an existing 3D model. As part of this placement, our method also estimates the absolute scale of the scene from object semantics, which in this case constitutes the height distribution of the population. Further, we obtain a smooth approximation of the ground surface and recover the gravity vector of the scene directly from the individual person detections. We demonstrate the results of our approach on a number of unordered Internet photo collections, and we quantitatively evaluate the obtained absolute scene scales.

count=1
* Pixels, Voxels, and Views: A Study of Shape Representations for Single View 3D Object Shape Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Shin_Pixels_Voxels_and_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shin_Pixels_Voxels_and_CVPR_2018_paper.pdf)]
    * Title: Pixels, Voxels, and Views: A Study of Shape Representations for Single View 3D Object Shape Prediction
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Daeyun Shin, Charless C. Fowlkes, Derek Hoiem
    * Abstract: The goal of this paper is to compare surface-based and volumetric 3D object shape representations, as well as viewer-centered and object-centered reference frames for single-view 3D shape prediction. We propose a new algorithm for predicting depth maps from multiple viewpoints, with a single depth or RGB image as input. By modifying the network and the way models are evaluated, we can directly compare the merits of voxels vs. surfaces and viewer-centered vs. object-centered for familiar vs. unfamiliar objects, as predicted from RGB or depth images. Among our findings, we show that surface-based methods outperform voxel representations for objects from novel classes and produce higher resolution outputs. We also find that using viewer-centered coordinates is advantageous for novel objects, while object-centered representations are better for more familiar objects. Interestingly, the coordinate frame significantly affects the shape representation learned, with object-centered placing more importance on implicitly recognizing the object category and viewer-centered producing shape representations with less dependence on category recognition.

count=1
* Aperture Supervision for Monocular Depth Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Srinivasan_Aperture_Supervision_for_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Srinivasan_Aperture_Supervision_for_CVPR_2018_paper.pdf)]
    * Title: Aperture Supervision for Monocular Depth Estimation
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Pratul P. Srinivasan, Rahul Garg, Neal Wadhwa, Ren Ng, Jonathan T. Barron
    * Abstract: We present a novel method to train machine learning algorithms to estimate scene depths from a single image, by using the information provided by a camera's aperture as supervision. Prior works use a depth sensor's outputs or images of the same scene from alternate viewpoints as supervision, while our method instead uses images from the same viewpoint taken with a varying camera aperture. To enable learning algorithms to use aperture effects as supervision, we introduce two differentiable aperture rendering functions that use the input image and predicted depths to simulate the depth-of-field effects caused by real camera apertures. We train a monocular depth estimation network end-to-end to predict the scene depths that best explain these finite aperture images as defocus-blurred renderings of the input all-in-focus image.

count=1
* Light-Weight Head Pose Invariant Gaze Tracking
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w41/html/Ranjan_Light-Weight_Head_Pose_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w41/Ranjan_Light-Weight_Head_Pose_CVPR_2018_paper.pdf)]
    * Title: Light-Weight Head Pose Invariant Gaze Tracking
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Rajeev Ranjan, Shalini De Mello, Jan Kautz
    * Abstract: Unconstrained remote gaze tracking using off-the-shelf cameras is a challenging problem. Recently, promising algorithms for appearance-based gaze estimation using convolutional neural networks (CNN) have been proposed. Improving their robustness to various confounding factors including variable head pose, subject identity, illumination and image quality remain open problems. In this work, we study the effect of variable head pose on machine learning regressors trained to estimate gaze direction. We propose a novel branched CNN architecture that improves the robustness of gaze classifiers to variable head pose, without increasing computational cost. We also present various procedures to effectively train our gaze network including transfer learning from the more closely related task of object viewpoint estimation and from a large high-fidelity synthetic gaze dataset, which enable our ten times faster gaze network to achieve competitive accuracy to its current state-of-the-art direct competitor.

count=1
* Inverse Path Tracing for Joint Material and Lighting Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Azinovic_Inverse_Path_Tracing_for_Joint_Material_and_Lighting_Estimation_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Azinovic_Inverse_Path_Tracing_for_Joint_Material_and_Lighting_Estimation_CVPR_2019_paper.pdf)]
    * Title: Inverse Path Tracing for Joint Material and Lighting Estimation
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Dejan Azinovic,  Tzu-Mao Li,  Anton Kaplanyan,  Matthias Niessner
    * Abstract: Modern computer vision algorithms have brought significant advancement to 3D geometry reconstruction. However, illumination and material reconstruction remain less studied, with current approaches assuming very simplified models for materials and illumination. We introduce Inverse Path Tracing, a novel approach to jointly estimate the material properties of objects and light sources in indoor scenes by using an invertible light transport simulation. We assume a coarse geometry scan, along with corresponding images and camera poses. The key contribution of this work is an accurate and simultaneous retrieval of light sources and physically based material properties (e.g., diffuse reflectance, specular reflectance, roughness, etc.) for the purpose of editing and re-rendering the scene under new conditions. To this end, we introduce a novel optimization method using a differentiable Monte Carlo renderer that computes derivatives with respect to the estimated unknown illumination and material properties. This enables joint optimization for physically correct light transport and material models using a tailored stochastic gradient descent.

count=1
* High-Quality Face Capture Using Anatomical Muscles
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Bao_High-Quality_Face_Capture_Using_Anatomical_Muscles_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bao_High-Quality_Face_Capture_Using_Anatomical_Muscles_CVPR_2019_paper.pdf)]
    * Title: High-Quality Face Capture Using Anatomical Muscles
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Michael Bao,  Matthew Cong,  Stephane Grabli,  Ronald Fedkiw
    * Abstract: Muscle-based systems have the potential to provide both anatomical accuracy and semantic interpretability as compared to blendshape models; however, a lack of expressivity and differentiability has limited their impact. Thus, we propose modifying a recently developed rather expressive muscle-based system in order to make it fully-differentiable; in fact, our proposed modifications allow this physically robust and anatomically accurate muscle model to conveniently be driven by an underlying blendshape basis. Our formulation is intuitive, natural, as well as monolithically and fully coupled such that one can differentiate the model from end to end, which makes it viable for both optimization and learning-based approaches for a variety of applications. We illustrate this with a number of examples including both shape matching of three-dimensional geometry as as well as the automatic determination of a three-dimensional facial pose from a single two-dimensional RGB image without using markers or depth information.

count=1
* Steady-State Non-Line-Of-Sight Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Steady-State_Non-Line-Of-Sight_Imaging_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Steady-State_Non-Line-Of-Sight_Imaging_CVPR_2019_paper.pdf)]
    * Title: Steady-State Non-Line-Of-Sight Imaging
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Wenzheng Chen,  Simon Daneau,  Fahim Mannan,  Felix Heide
    * Abstract: Conventional intensity cameras recover objects in the direct line-of-sight of the camera, whereas occluded scene parts are considered lost in this process. Non-line-of-sight imaging (NLOS) aims at recovering these occluded objects by analyzing their indirect reflections on visible scene surfaces. Existing NLOS methods temporally probe the indirect light transport to unmix light paths based on their travel time, which mandates specialized instrumentation that suffers from low photon efficiency, high cost, and mechanical scanning. We depart from temporal probing and demonstrate steady-state NLOS imaging using conventional intensity sensors and continuous illumination. Instead of assuming perfectly isotropic scattering, the proposed method exploits directionality in the hidden surface reflectance, resulting in (small) spatial variation of their indirect reflections for varying illumination. To tackle the shape-dependence of these variations, we propose a trainable architecture which learns to map diffuse indirect reflections to scene reflectance using only synthetic training data. Relying on consumer color image sensors, with high fill factor, high quantum efficiency and low read-out noise, we demonstrate high-fidelity color NLOS imaging for scene configurations tackled before with picosecond time resolution.

count=1
* Incremental Object Learning From Contiguous Views
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Stojanov_Incremental_Object_Learning_From_Contiguous_Views_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Stojanov_Incremental_Object_Learning_From_Contiguous_Views_CVPR_2019_paper.pdf)]
    * Title: Incremental Object Learning From Contiguous Views
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Stefan Stojanov,  Samarth Mishra,  Ngoc Anh Thai,  Nikhil Dhanda,  Ahmad Humayun,  Chen Yu,  Linda B. Smith,  James M. Rehg
    * Abstract: In this work, we present CRIB (Continual Recognition Inspired by Babies), a synthetic incremental object learning environment that can produce data that models visual imagery produced by object exploration in early infancy. CRIB is coupled with a new 3D object dataset, Toys-200, that contains 200 unique toy-like object instances, and is also compatible with existing 3D datasets. Through extensive empirical evaluation of state-of-the-art incremental learning algorithms, we find the novel empirical result that repetition can significantly ameliorate the effects of catastrophic forgetting. Furthermore, we find that in certain cases repetition allows for performance approaching that of batch learning algorithms. Finally, we propose an unsupervised incremental learning task with intriguing baseline results.

count=1
* Self-Supervised 3D Hand Pose Estimation Through Training by Fitting
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Wan_Self-Supervised_3D_Hand_Pose_Estimation_Through_Training_by_Fitting_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wan_Self-Supervised_3D_Hand_Pose_Estimation_Through_Training_by_Fitting_CVPR_2019_paper.pdf)]
    * Title: Self-Supervised 3D Hand Pose Estimation Through Training by Fitting
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Chengde Wan,  Thomas Probst,  Luc Van Gool,  Angela Yao
    * Abstract: We present a self-supervision method for 3D hand pose estimation from depth maps. We begin with a neural network initialized with synthesized data and fine-tune it on real but unlabelled depth maps by minimizing a set of data-fitting terms. By approximating the hand surface with a set of spheres, we design a differentiable hand renderer to align estimates by comparing the rendered and input depth maps. In addition, we place a set of priors including a data-driven term to further regulate the estimate's kinematic feasibility. Our method makes highly accurate estimates comparable to current supervised methods which require large amounts of labelled training samples, thereby advancing state-of-the-art in unsupervised learning for hand pose estimation.

count=1
* MeshAdv: Adversarial Meshes for Visual Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Xiao_MeshAdv_Adversarial_Meshes_for_Visual_Recognition_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xiao_MeshAdv_Adversarial_Meshes_for_Visual_Recognition_CVPR_2019_paper.pdf)]
    * Title: MeshAdv: Adversarial Meshes for Visual Recognition
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Chaowei Xiao,  Dawei Yang,  Bo Li,  Jia Deng,  Mingyan Liu
    * Abstract: Highly expressive models such as deep neural networks (DNNs) have been widely applied to various applications. However, recent studies show that DNNs are vulnerable to adversarial examples, which are carefully crafted inputs aiming to mislead the predictions. Currently, the majority of these studies have focused on perturbation added to image pixels, while such manipulation is not physically realistic. Some works have tried to overcome this limitation by attaching printable 2D patches or painting patterns onto surfaces, but can be potentially defended because 3D shape features are intact. In this paper, we propose meshAdv to generate "adversarial 3D meshes" from objects that have rich shape features but minimal textural variation. To manipulate the shape or texture of the objects, we make use of a differentiable renderer to compute accurate shading on the shape and propagate the gradient. Extensive experiments show that the generated 3D meshes are effective in attacking both classifiers and object detectors. We evaluate the attack under different viewpoints. In addition, we design a pipeline to perform black-box attack on a photorealistic renderer with unknown rendering parameters.

count=1
* InverseRenderNet: Learning Single Image Inverse Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Yu_InverseRenderNet_Learning_Single_Image_Inverse_Rendering_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_InverseRenderNet_Learning_Single_Image_Inverse_Rendering_CVPR_2019_paper.pdf)]
    * Title: InverseRenderNet: Learning Single Image Inverse Rendering
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Ye Yu,  William A. P. Smith
    * Abstract: We show how to train a fully convolutional neural network to perform inverse rendering from a single, uncontrolled image. The network takes an RGB image as input, regresses albedo and normal maps from which we compute lighting coefficients. Our network is trained using large uncontrolled image collections without ground truth. By incorporating a differentiable renderer, our network can learn from self-supervision. Since the problem is ill-posed we introduce additional supervision: 1. We learn a statistical natural illumination prior, 2. Our key insight is to perform offline multiview stereo (MVS) on images containing rich illumination variation. From the MVS pose and depth maps, we can cross project between overlapping views such that Siamese training can be used to ensure consistent estimation of photometric invariants. MVS depth also provides direct coarse supervision for normal map estimation. We believe this is the first attempt to use MVS supervision for learning inverse rendering.

count=1
* Deep 3D Capture: Geometry and Reflectance From Sparse Multi-View Images
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Bi_Deep_3D_Capture_Geometry_and_Reflectance_From_Sparse_Multi-View_Images_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bi_Deep_3D_Capture_Geometry_and_Reflectance_From_Sparse_Multi-View_Images_CVPR_2020_paper.pdf)]
    * Title: Deep 3D Capture: Geometry and Reflectance From Sparse Multi-View Images
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Sai Bi,  Zexiang Xu,  Kalyan Sunkavalli,  David Kriegman,  Ravi Ramamoorthi
    * Abstract: We introduce a novel learning-based method to reconstruct the high-quality geometry and complex, spatially-varying BRDF of an arbitrary object from a sparse set of only six images captured by wide-baseline cameras under collocated point lighting. We first estimate per-view depth maps using a deep multi-view stereo network; these depth maps are used to coarsely align the different views. We propose a novel multi-view reflectance estimation network architecture that is trained to pool features from these coarsely aligned images and predict per-view spatially-varying diffuse albedo, surface normals, specular roughness and specular albedo. We do this by jointly optimizing the latent space of our multi-view reflectance network to minimize the photometric error between images rendered with our predictions and the input images. While previous state-of-the-art methods fail on such sparse acquisition setups, we demonstrate, via extensive experiments on synthetic and real data, that our method produces high-quality reconstructions that can be used to render photorealistic images.

count=1
* A Neural Rendering Framework for Free-Viewpoint Relighting
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_A_Neural_Rendering_Framework_for_Free-Viewpoint_Relighting_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Neural_Rendering_Framework_for_Free-Viewpoint_Relighting_CVPR_2020_paper.pdf)]
    * Title: A Neural Rendering Framework for Free-Viewpoint Relighting
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zhang Chen,  Anpei Chen,  Guli Zhang,  Chengyuan Wang,  Yu Ji,  Kiriakos N. Kutulakos,  Jingyi Yu
    * Abstract: We present a novel Relightable Neural Renderer (RNR) for simultaneous view synthesis and relighting using multi-view image inputs. Existing neural rendering (NR) does not explicitly model the physical rendering process and hence has limited capabilities on relighting. RNR instead models image formation in terms of environment lighting, object intrinsic attributes, and light transport function (LTF), each corresponding to a learnable component. In particular, the incorporation of a physically based rendering process not only enables relighting but also improves the quality of view synthesis. Comprehensive experiments on synthetic and real data show that RNR provides a practical and effective solution for conducting free-viewpoint relighting.

count=1
* Deep Stereo Using Adaptive Thin Volume Representation With Uncertainty Awareness
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Deep_Stereo_Using_Adaptive_Thin_Volume_Representation_With_Uncertainty_Awareness_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Deep_Stereo_Using_Adaptive_Thin_Volume_Representation_With_Uncertainty_Awareness_CVPR_2020_paper.pdf)]
    * Title: Deep Stereo Using Adaptive Thin Volume Representation With Uncertainty Awareness
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Shuo Cheng,  Zexiang Xu,  Shilin Zhu,  Zhuwen Li,  Li Erran Li,  Ravi Ramamoorthi,  Hao Su
    * Abstract: We present Uncertainty-aware Cascaded Stereo Network (UCS-Net) for 3D reconstruction from multiple RGB images. Multi-view stereo (MVS) aims to reconstruct fine-grained scene geometry from multi-view images. Previous learning-based MVS methods estimate per-view depth using plane sweep volumes (PSVs) with a fixed depth hypothesis at each plane; this requires densely sampled planes for high accuracy, which is impractical for high-resolution depth because of limited memory. In contrast, we propose adaptive thin volumes (ATVs); in an ATV, the depth hypothesis of each plane is spatially varying, which adapts to the uncertainties of previous per-pixel depth predictions. Our UCS-Net has three stages: the first stage processes a small PSV to predict low-resolution depth; two ATVs are then used in the following stages to refine the depth with higher resolution and higher accuracy. Our ATV consists of only a small number of planes with low memory and computation costs; yet, it efficiently partitions local depth ranges within learned small uncertainty intervals. We propose to use variance-based uncertainty estimates to adaptively construct ATVs; this differentiable process leads to reasonable and fine-grained spatial partitioning. Our multi-stage framework progressively sub-divides the vast scene space with increasing depth resolution and precision, which enables reconstruction with high completeness and accuracy in a coarse-to-fine fashion. We demonstrate that our method achieves superior performance compared with other learning-based MVS methods on various challenging datasets.

count=1
* What You See is What You Get: Exploiting Visibility for 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_What_You_See_is_What_You_Get_Exploiting_Visibility_for_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_What_You_See_is_What_You_Get_Exploiting_Visibility_for_CVPR_2020_paper.pdf)]
    * Title: What You See is What You Get: Exploiting Visibility for 3D Object Detection
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Peiyun Hu,  Jason Ziglar,  David Held,  Deva Ramanan
    * Abstract: Recent advances in 3D sensing have created unique challenges for computer vision. One fundamental challenge is finding a good representation for 3D sensor data. Most popular representations (such as PointNet) are proposed in the context of processing truly 3D data (e.g. points sampled from mesh models), ignoring the fact that 3D sensored data such as a LiDAR sweep is in fact 2.5D. We argue that representing 2.5D data as collections of (x,y,z) points fundamentally destroys hidden information about freespace. In this paper, we demonstrate such knowledge can be efficiently recovered through 3D raycasting and readily incorporated into batch-based gradient learning. We describe a simple approach to augmenting voxel-based networks with visibility: we add a voxelized visibility map as an additional input stream. In addition, we show that visibility can be combined with two crucial modifications common to state-of-the-art 3D detectors: synthetic data augmentation of virtual objects and temporal aggregation of LiDAR sweeps over multiple time frames. On the NuScenes 3D detection benchmark, we show that, by adding an additional stream for visibility input, we can significantly improve the overall detection accuracy of a state-of-the-art 3D detector.

count=1
* Inverse Rendering for Complex Indoor Scenes: Shape, Spatially-Varying Lighting and SVBRDF From a Single Image
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Inverse_Rendering_for_Complex_Indoor_Scenes_Shape_Spatially-Varying_Lighting_and_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Inverse_Rendering_for_Complex_Indoor_Scenes_Shape_Spatially-Varying_Lighting_and_CVPR_2020_paper.pdf)]
    * Title: Inverse Rendering for Complex Indoor Scenes: Shape, Spatially-Varying Lighting and SVBRDF From a Single Image
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zhengqin Li,  Mohammad Shafiei,  Ravi Ramamoorthi,  Kalyan Sunkavalli,  Manmohan Chandraker
    * Abstract: We propose a deep inverse rendering framework for indoor scenes. From a single RGB image of an arbitrary indoor scene, we obtain a complete scene reconstruction, estimating shape, spatially-varying lighting, and spatially-varying, non-Lambertian surface reflectance. Our novel inverse rendering network incorporates physical insights -- including a spatially-varying spherical Gaussian lighting representation, a differentiable rendering layer to model scene appearance, a cascade structure to iteratively refine the predictions and a bilateral solver for refinement -- allowing us to jointly reason about shape, lighting, and reflectance. Since no existing dataset provides ground truth high quality spatially-varying material and spatially-varying lighting, we propose novel methods to map complex materials to existing indoor scene datasets and a new physically-based GPU renderer to create a large-scale, photorealistic indoor dataset. Experiments show that our framework outperforms previous methods and enables various novel applications like photorealistic object insertion and material editing.

count=1
* KeyPose: Multi-View 3D Labeling and Keypoint Estimation for Transparent Objects
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_KeyPose_Multi-View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_KeyPose_Multi-View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects_CVPR_2020_paper.pdf)]
    * Title: KeyPose: Multi-View 3D Labeling and Keypoint Estimation for Transparent Objects
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Xingyu Liu,  Rico Jonschkowski,  Anelia Angelova,  Kurt Konolige
    * Abstract: Estimating the 3D pose of desktop objects is crucial for applications such as robotic manipulation. Many existing approaches to this problem require a depth map of the object for both training and prediction, which restricts them to opaque, lambertian objects that produce good returns in an RGBD sensor. In this paper we forgo using a depth sensor in favor of raw stereo input. We address two problems: first, we establish an easy method for capturing and labeling 3D keypoints on desktop objects with an RGB camera; and second, we develop a deep neural network, called KeyPose, that learns to accurately predict object poses using 3D keypoints, from stereo input, and works even for transparent objects. To evaluate the performance of our method, we create a dataset of 15 clear objects in five classes, with 48K 3D-keypoint labeled images. We train both instance and category models, and show generalization to new textures, poses, and objects. KeyPose surpasses state-of-the-art performance in 3D pose estimation on this dataset by factors of 1.5 to 3.5, even in cases where the competing method is provided with ground-truth depth. Stereo input is essential for this performance as it improves results compared to using monocular input by a factor of 2. We will release a public version of the data capture and labeling pipeline, the transparent object database, and the KeyPose models and evaluation code. Project website: https://sites.google.com/corp/view/keypose.

count=1
* End-to-End Optimization of Scene Layout
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Luo_End-to-End_Optimization_of_Scene_Layout_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_End-to-End_Optimization_of_Scene_Layout_CVPR_2020_paper.pdf)]
    * Title: End-to-End Optimization of Scene Layout
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Andrew Luo,  Zhoutong Zhang,  Jiajun Wu,  Joshua B. Tenenbaum
    * Abstract: We propose an end-to-end variational generative model for scene layout synthesis conditioned on scene graphs. Unlike unconditional scene layout generation, we use scene graphs as an abstract but general representation to guide the synthesis of diverse scene layouts that satisfy relationships included in the scene graph. This gives rise to more flexible control over the synthesis process, allowing various forms of inputs such as scene layouts extracted from sentences or inferred from a single color image. Using our conditional layout synthesizer, we can generate various layouts that share the same structure of the input example. In addition to this conditional generation design, we also integrate a differentiable rendering module that enables layout refinement using only 2D projections of the scene. Given a depth and a semantics map, the differentiable rendering module enables optimizing over the synthesized layout to fit the given input in an analysis-by-synthesis fashion. Experiments suggest that our model achieves higher accuracy and diversity in conditional scene synthesis and allows exemplar-based scene generation from various input forms.

count=1
* Single-Shot Monocular RGB-D Imaging Using Uneven Double Refraction
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Meuleman_Single-Shot_Monocular_RGB-D_Imaging_Using_Uneven_Double_Refraction_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Meuleman_Single-Shot_Monocular_RGB-D_Imaging_Using_Uneven_Double_Refraction_CVPR_2020_paper.pdf)]
    * Title: Single-Shot Monocular RGB-D Imaging Using Uneven Double Refraction
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Andreas Meuleman,  Seung-Hwan Baek,  Felix Heide,  Min H. Kim
    * Abstract: Cameras that capture color and depth information have become an essential imaging modality for applications in robotics, autonomous driving, virtual, and augmented reality. Existing RGB-D cameras rely on multiple sensors or active illumination with specialized sensors. In this work, we propose a method for monocular single-shot RGB-D imaging. Instead of learning depth from single-image depth cues, we revisit double-refraction imaging using a birefractive medium, measuring depth as the displacement of differently refracted images superimposed in a single capture. However, existing double-refraction methods are orders of magnitudes too slow to be used in real-time applications, e.g., in robotics, and provide only inaccurate depth due to correspondence ambiguity in double reflection. We resolve this ambiguity optically by leveraging the orthogonality of the two linearly polarized rays in double refraction -- introducing uneven double refraction by adding a linear polarizer to the birefractive medium. Doing so makes it possible to develop a real-time method for reconstructing sparse depth and color simultaneously in real-time. We validate the proposed method, both synthetically and experimentally, and demonstrate 3D object detection and photographic applications.

count=1
* TetraTSDF: 3D Human Reconstruction From a Single Image With a Tetrahedral Outer Shell
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Onizuka_TetraTSDF_3D_Human_Reconstruction_From_a_Single_Image_With_a_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Onizuka_TetraTSDF_3D_Human_Reconstruction_From_a_Single_Image_With_a_CVPR_2020_paper.pdf)]
    * Title: TetraTSDF: 3D Human Reconstruction From a Single Image With a Tetrahedral Outer Shell
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Hayato Onizuka,  Zehra Hayirci,  Diego Thomas,  Akihiro Sugimoto,  Hideaki Uchiyama,  Rin-ichiro Taniguchi
    * Abstract: Recovering the 3D shape of a person from its 2D appearance is ill-posed due to ambiguities. Nevertheless, with the help of convolutional neural networks (CNN) and prior knowledge on the 3D human body, it is possible to overcome such ambiguities to recover detailed 3D shapes of human bodies from single images. Current solutions, however, fail to reconstruct all the details of a person wearing loose clothes. This is because of either (a) huge memory requirement that cannot be maintained even on modern GPUs or (b) the compact 3D representation that cannot encode all the details. In this paper, we propose the tetrahedral outer shell volumetric truncated signed distance function (TetraTSDF) model for the human body, and its corresponding part connection network (PCN) for 3D human body shape regression. Our proposed model is compact, dense, accurate, and yet well suited for CNN-based regression task. Our proposed PCN allows us to learn the distribution of the TSDF in the tetrahedral volume from a single image in an end-to-end manner. Results show that our proposed method allows to reconstruct detailed shapes of humans wearing loose clothes from single RGB images.

count=1
* LatentFusion: End-to-End Differentiable Reconstruction and Rendering for Unseen Object Pose Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Park_LatentFusion_End-to-End_Differentiable_Reconstruction_and_Rendering_for_Unseen_Object_Pose_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_LatentFusion_End-to-End_Differentiable_Reconstruction_and_Rendering_for_Unseen_Object_Pose_CVPR_2020_paper.pdf)]
    * Title: LatentFusion: End-to-End Differentiable Reconstruction and Rendering for Unseen Object Pose Estimation
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Keunhong Park,  Arsalan Mousavian,  Yu Xiang,  Dieter Fox
    * Abstract: Current 6D object pose estimation methods usually require a 3D model for each object. These methods also require additional training in order to incorporate new objects. As a result, they are difficult to scale to a large number of objects and cannot be directly applied to unseen objects. We propose a novel framework for 6D pose estimation of unseen objects. We present a network that reconstructs a latent 3D representation of an object using a small number of reference views at inference time. Our network is able to render the latent 3D representation from arbitrary views. Using this neural renderer, we directly optimize for pose given an input image. By training our network with a large number of 3D shapes for reconstruction and rendering, our network generalizes well to unseen objects. We present a new dataset for unseen object pose estimation--MOPED. We evaluate the performance of our method for unseen object pose estimation on MOPED as well as the ModelNet and LINEMOD datasets. Our method performs competitively to supervised methods that are trained on those objects. Code and data will be available at https://keunhong.com/publications/latentfusion/

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
* 3D-ZeF: A 3D Zebrafish Tracking Benchmark Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Pedersen_3D-ZeF_A_3D_Zebrafish_Tracking_Benchmark_Dataset_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pedersen_3D-ZeF_A_3D_Zebrafish_Tracking_Benchmark_Dataset_CVPR_2020_paper.pdf)]
    * Title: 3D-ZeF: A 3D Zebrafish Tracking Benchmark Dataset
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Malte Pedersen,  Joakim Bruslund Haurum,  Stefan Hein Bengtson,  Thomas B. Moeslund
    * Abstract: In this work we present a novel publicly available stereo based 3D RGB dataset for multi-object zebrafish tracking, called 3D-ZeF. Zebrafish is an increasingly popular model organism used for studying neurological disorders, drug addiction, and more. Behavioral analysis is often a critical part of such research. However, visual similarity, occlusion, and erratic movement of the zebrafish makes robust 3D tracking a challenging and unsolved problem. The proposed dataset consists of eight sequences with a duration between 15-120 seconds and 1-10 free moving zebrafish. The videos have been annotated with a total of 86,400 points and bounding boxes. Furthermore, we present a complexity score and a novel open-source modular baseline system for 3D tracking of zebrafish. The performance of the system is measured with respect to two detectors: a naive approach and a Faster R-CNN based fish head detector. The system reaches a MOTA of up to 77.6%. Links to the code and dataset is available at the project page http://vap.aau.dk/3d-zef

count=1
* Neural Voxel Renderer: Learning an Accurate and Controllable Rendering Tool
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Rematas_Neural_Voxel_Renderer_Learning_an_Accurate_and_Controllable_Rendering_Tool_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Rematas_Neural_Voxel_Renderer_Learning_an_Accurate_and_Controllable_Rendering_Tool_CVPR_2020_paper.pdf)]
    * Title: Neural Voxel Renderer: Learning an Accurate and Controllable Rendering Tool
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Konstantinos Rematas,  Vittorio Ferrari
    * Abstract: We present a neural rendering framework that maps a voxelized scene into a high quality image. Highly-textured objects and scene element interactions are realistically rendered by our method, despite having a rough representation as an input. Moreover, our approach allows controllable rendering: geometric and appearance modifications in the input are accurately propagated to the output. The user can move, rotate and scale an object, change its appearance and texture or modify the position of the light and all these edits are represented in the final rendering. We demonstrate the effectiveness of our approach by rendering scenes with varying appearance, from single color per object to complex, high-frequency textures. We show that our rerendering network can generate very detailed images that represent precisely the appearance of the input scene. Our experiments illustrate that our approach achieves more accurate image synthesis results compared to alternatives and can also handle low voxel grid resolutions. Finally, we show how our neural rendering framework can capture and faithfully render objects from real images and from a diverse set of classes.

count=1
* Self-Supervised Human Depth Estimation From Monocular Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_Self-Supervised_Human_Depth_Estimation_From_Monocular_Videos_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_Self-Supervised_Human_Depth_Estimation_From_Monocular_Videos_CVPR_2020_paper.pdf)]
    * Title: Self-Supervised Human Depth Estimation From Monocular Videos
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Feitong Tan,  Hao Zhu,  Zhaopeng Cui,  Siyu Zhu,  Marc Pollefeys,  Ping Tan
    * Abstract: Previous methods on estimating detailed human depth often require supervised training with 'ground truth' depth data. This paper presents a self-supervised method that can be trained on YouTube videos without known depth, which makes training data collection simple and improves the generalization of the learned network. The self-supervised learning is achieved by minimizing a photo-consistency loss, which is evaluated between a video frame and its neighboring frames warped according to the estimated depth and the 3D non-rigid motion of the human body. To solve this non-rigid motion, we first estimate a rough SMPL model at each video frame and compute the non-rigid body motion accordingly, which enables self-supervised learning on estimating the shape details. Experiments demonstrate that our method enjoys better generalization, and performs much better on data in the wild.

count=1
* Learning to Generate 3D Training Data Through Hybrid Gradient
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Learning_to_Generate_3D_Training_Data_Through_Hybrid_Gradient_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Learning_to_Generate_3D_Training_Data_Through_Hybrid_Gradient_CVPR_2020_paper.pdf)]
    * Title: Learning to Generate 3D Training Data Through Hybrid Gradient
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Dawei Yang,  Jia Deng
    * Abstract: Synthetic images rendered by graphics engines are a promising source for training deep networks. However, it is challenging to ensure that they can help train a network to perform well on real images, because a graphics-based generation pipeline requires numerous design decisions such as the selection of 3D shapes and the placement of the camera. In this work, we propose a new method that optimizes the generation of 3D training data based on what we call "hybrid gradient". We parametrize the design decisions as a real vector, and combine the approximate gradient and the analytical gradient to obtain the hybrid gradient of the network performance with respect to this vector. We evaluate our approach on the task of estimating surface normal, depth or intrinsic decomposition from a single image. Experiments on standard benchmarks show that our approach can outperform the prior state of the art on optimizing the generation of 3D training data, particularly in terms of computational efficiency.

count=1
* SurfelGAN: Synthesizing Realistic Sensor Data for Autonomous Driving
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_SurfelGAN_Synthesizing_Realistic_Sensor_Data_for_Autonomous_Driving_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_SurfelGAN_Synthesizing_Realistic_Sensor_Data_for_Autonomous_Driving_CVPR_2020_paper.pdf)]
    * Title: SurfelGAN: Synthesizing Realistic Sensor Data for Autonomous Driving
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Zhenpei Yang,  Yuning Chai,  Dragomir Anguelov,  Yin Zhou,  Pei Sun,  Dumitru Erhan,  Sean Rafferty,  Henrik Kretzschmar
    * Abstract: Autonomous driving system development is critically dependent on the ability to replay complex and diverse traffic scenarios in simulation. In such scenarios, the ability to accurately simulate the vehicle sensors such as cameras, lidar or radar is hugely helpful. However, current sensor simulators leverage gaming engines such as Unreal or Unity, requiring manual creation of environments, objects, and material properties. Such approaches have limited scalability and fail to produce realistic approximations of camera, lidar, and radar data without significant additional work. In this paper, we present a simple yet effective approach to generate realistic scenario sensor data, based only on a limited amount of lidar and camera data collected by an autonomous vehicle. Our approach uses texture-mapped surfels to efficiently reconstruct the scene from an initial vehicle pass or set of passes, preserving rich information about object 3D geometry and appearance, as well as the scene conditions. We then leverage a SurfelGAN network to reconstruct realistic camera images for novel positions and orientations of the self-driving vehicle and moving objects in the scene. We demonstrate our approach on the Waymo Open Dataset and show that it can synthesize realistic camera data for simulated scenarios. We also create a novel dataset that contains cases in which two self-driving vehicles observe the same scene at the same time. We use this dataset to provide additional evaluation and demonstrate the usefulness of our SurfelGAN model.

count=1
* Pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Chan_Pi-GAN_Periodic_Implicit_Generative_Adversarial_Networks_for_3D-Aware_Image_Synthesis_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Chan_Pi-GAN_Periodic_Implicit_Generative_Adversarial_Networks_for_3D-Aware_Image_Synthesis_CVPR_2021_paper.pdf)]
    * Title: Pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Eric R. Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, Gordon Wetzstein
    * Abstract: We have witnessed rapid progress on 3D-aware image synthesis, leveraging recent advances in generative visual models and neural rendering. Existing approaches however fall short in two ways: first, they may lack an underlying 3D representation or rely on view-inconsistent rendering, hence synthesizing images that are not multi-view consistent; second, they often depend upon representation network architectures that are not expressive enough, and their results thus lack in image quality. We propose a novel generative model, named Periodic Implicit Generative Adversarial Networks (p-GAN or pi-GAN), for high-quality 3D-aware image synthesis. p-GAN leverages neural representations with periodic activation functions and volumetric rendering to represent scenes as view-consistent radiance fields. The proposed approach obtains state-of-the-art results for 3D-aware image synthesis with multiple real and synthetic datasets.

count=1
* High-Fidelity Face Tracking for AR/VR via Deep Lighting Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_High-Fidelity_Face_Tracking_for_ARVR_via_Deep_Lighting_Adaptation_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_High-Fidelity_Face_Tracking_for_ARVR_via_Deep_Lighting_Adaptation_CVPR_2021_paper.pdf)]
    * Title: High-Fidelity Face Tracking for AR/VR via Deep Lighting Adaptation
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Lele Chen, Chen Cao, Fernando De la Torre, Jason Saragih, Chenliang Xu, Yaser Sheikh
    * Abstract: 3D video avatars can empower virtual communications by providing compression, privacy, entertainment, and a sense of presence in AR/VR. Best 3D photo-realistic AR/VR avatars driven by video, that can minimize uncanny effects, rely on person-specific models. However, existing person-specific photo-realistic 3D models are not robust to lighting, hence their results typically miss subtle facial behaviors and cause artifacts in the avatar. This is a major drawback for the scalability of these models in communication systems (e.g., Messenger, Skype, FaceTime) and AR/VR. This paper addresses previous limitations by learning a deep learning lighting model, that in combination with a high-quality 3D face tracking algorithm, provides a method for subtle and robust facial motion transfer from a regular video to a 3D photo-realistic avatar. Extensive experimental validation and comparisons to other state-of-the-art methods demonstrate the effectiveness of the proposed framework in real-world scenarios with variability in pose, expression, and illumination. Our project page can be found at https://www.cs.rochester.edu/ cxu22/r/wild-avatar/.

count=1
* Global Transport for Fluid Reconstruction With Learned Self-Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Franz_Global_Transport_for_Fluid_Reconstruction_With_Learned_Self-Supervision_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Franz_Global_Transport_for_Fluid_Reconstruction_With_Learned_Self-Supervision_CVPR_2021_paper.pdf)]
    * Title: Global Transport for Fluid Reconstruction With Learned Self-Supervision
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Erik Franz, Barbara Solenthaler, Nils Thuerey
    * Abstract: We propose a novel method to reconstruct volumetric flows from sparse views via a global transport formulation. Instead of obtaining the space-time function of the observations, we reconstruct its motion based on a single initial state. In addition we introduce a learned self-supervision that constrains observations from unseen angles. These visual constraints are coupled via the transport constraints and a differentiable rendering step to arrive at a robust end-to-end reconstruction algorithm. This makes the reconstruction of highly realistic flow motions possible, even from only a single input view. We show with a variety of synthetic and real flows that the proposed global reconstruction of the transport process yields an improved reconstruction of the fluid motion.

count=1
* Neural Lumigraph Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Kellnhofer_Neural_Lumigraph_Rendering_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Kellnhofer_Neural_Lumigraph_Rendering_CVPR_2021_paper.pdf)]
    * Title: Neural Lumigraph Rendering
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Petr Kellnhofer, Lars C. Jebe, Andrew Jones, Ryan Spicer, Kari Pulli, Gordon Wetzstein
    * Abstract: Novel view synthesis is a challenging and ill-posed inverse rendering problem. Neural rendering techniques have recently achieved photorealistic image quality for this task. State-of-the-art (SOTA) neural volume rendering approaches, however, are slow to train and require minutes of inference (i.e., rendering) time for high image resolutions. We adopt high-capacity neural scene representations with periodic activations for jointly optimizing an implicit surface and a radiance field of a scene supervised exclusively with posed 2D images. Our neural rendering pipeline accelerates SOTA neural volume rendering by about two orders of magnitude and our implicit surface representation is unique in allowing us to export a mesh with view-dependent texture information. Thus, like other implicit surface representations, ours is compatible with traditional graphics pipelines, enabling real-time rendering rates, while achieving unprecedented image quality compared to other surface methods. We assess the quality of our approach using existing datasets as well as high-quality 3D face data captured with a custom multi-camera rig.

count=1
* Pulsar: Efficient Sphere-Based Neural Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lassner_Pulsar_Efficient_Sphere-Based_Neural_Rendering_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lassner_Pulsar_Efficient_Sphere-Based_Neural_Rendering_CVPR_2021_paper.pdf)]
    * Title: Pulsar: Efficient Sphere-Based Neural Rendering
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Christoph Lassner, Michael Zollhofer
    * Abstract: We propose Pulsar, an efficient sphere-based differentiable rendering module that is orders of magnitude faster than competing techniques, modular, and easy-to-use due to its tight integration with PyTorch. Differentiable rendering is the foundation for modern neural rendering approaches, since it enables end-to-end training of 3D scene representations from image observations. However, gradient-based optimization of neural mesh, voxel, or function representations suffers from multiple challenges, i.e., topological in-consistencies, high memory footprints, or slow rendering speeds. To alleviate these problems, Pulsar employs: 1) asphere-based scene representation, 2) a modular, efficient differentiable projection operation, and 3) (optional) neural shading. Pulsar executes orders of magnitude faster than existing techniques and allows real-time rendering and optimization of representations with millions of spheres. Using spheres for the scene representation, unprecedented speed is obtained while avoiding topology problems. Pulsar is fully differentiable and thus enables a plethora of applications, ranging from 3D reconstruction to neural rendering.

count=1
* Blocks-World Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Blocks-World_Cameras_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Blocks-World_Cameras_CVPR_2021_paper.pdf)]
    * Title: Blocks-World Cameras
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Jongho Lee, Mohit Gupta
    * Abstract: For several vision and robotics applications, 3D geometry of man-made environments such as indoor scenes can be represented with a small number of dominant planes. However, conventional 3D vision techniques typically first acquire dense 3D point clouds before estimating the compact piece-wise planar representations (e.g., by plane-fitting). This approach is costly, both in terms of acquisition and computational requirements, and potentially unreliable due to noisy point clouds. We propose Blocks-World Cameras, a class of imaging systems which directly recover dominant planes of piece-wise planar scenes (Blocks-World), without requiring point clouds. The Blocks-World Cameras are based on a structured-light system projecting a single pattern with a sparse set of cross-shaped features. We develop a novel geometric algorithm for recovering scene planes without explicit correspondence matching, thereby avoiding computationally intensive search or optimization routines. The proposed approach has low device and computational complexity, and requires capturing only one or two images. We demonstrate highly efficient and precise planar-scene sensing with simulations and real experiments, across various imaging conditions, including defocus blur, large lighting variations, ambient illumination, and scene clutter.

count=1
* AutoInt: Automatic Integration for Fast Neural Volume Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Lindell_AutoInt_Automatic_Integration_for_Fast_Neural_Volume_Rendering_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lindell_AutoInt_Automatic_Integration_for_Fast_Neural_Volume_Rendering_CVPR_2021_paper.pdf)]
    * Title: AutoInt: Automatic Integration for Fast Neural Volume Rendering
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: David B. Lindell, Julien N. P. Martel, Gordon Wetzstein
    * Abstract: Numerical integration is a foundational technique in scientific computing and is at the core of many computer vision applications. Among these applications, neural volume rendering has recently been proposed as a new paradigm for view synthesis, achieving photorealistic image quality. However, a fundamental obstacle to making these methods practical is the extreme computational and memory requirements caused by the required volume integrations along the rendered rays during training and inference. Millions of rays, each requiring hundreds of forward passes through a neural network are needed to approximate those integrations with Monte Carlo sampling. Here, we propose automatic integration, a new framework for learning efficient, closed-form solutions to integrals using coordinate-based neural networks. For training, we instantiate the computational graph corresponding to the derivative of the coordinate-based network. The graph is fitted to the signal to integrate. After optimization, we reassemble the graph to obtain a network that represents the antiderivative. By the fundamental theorem of calculus, this enables the calculation of any definite integral in two evaluations of the network. Applying this approach to neural rendering, we improve a tradeoff between rendering speed and image quality: improving render times by greater than 10x with a tradeoff of reduced image quality.

count=1
* Deep Implicit Moving Least-Squares Functions for 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Deep_Implicit_Moving_Least-Squares_Functions_for_3D_Reconstruction_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Deep_Implicit_Moving_Least-Squares_Functions_for_3D_Reconstruction_CVPR_2021_paper.pdf)]
    * Title: Deep Implicit Moving Least-Squares Functions for 3D Reconstruction
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Shi-Lin Liu, Hao-Xiang Guo, Hao Pan, Peng-Shuai Wang, Xin Tong, Yang Liu
    * Abstract: Point set is a flexible and lightweight representation widely used for 3D deep learning. However, their discrete nature prevents them from representing continuous and fine geometry, posing a major issue for learning-based shape generation. In this work, we turn the discrete point sets into smooth surfaces by introducing the well-known implicit moving least-squares (IMLS) surface formulation, which naturally defines locally implicit functions on point sets. We incorporate IMLS surface generation into deep neural networks for inheriting both the flexibility of point sets and the high quality of implicit surfaces. Our IMLSNet predicts an octree structure as a scaffold for generating MLS points where needed and characterizes shape geometry with learned local priors. Furthermore, our implicit function evaluation is independent of the neural network once the MLS points are predicted, thus enabling fast runtime evaluation. Our experiments on 3D object reconstruction demonstrate that IMLSNets outperform state-of-the-art learning-based methods in terms of reconstruction quality and computational efficiency. Extensive ablation tests also validate our network design and loss functions.

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
* GIRAFFE: Representing Scenes As Compositional Generative Neural Feature Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Niemeyer_GIRAFFE_Representing_Scenes_As_Compositional_Generative_Neural_Feature_Fields_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Niemeyer_GIRAFFE_Representing_Scenes_As_Compositional_Generative_Neural_Feature_Fields_CVPR_2021_paper.pdf)]
    * Title: GIRAFFE: Representing Scenes As Compositional Generative Neural Feature Fields
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Michael Niemeyer, Andreas Geiger
    * Abstract: Deep generative models allow for photorealistic image synthesis at high resolutions. But for many applications, this is not enough: content creation also needs to be controllable. While several recent works investigate how to disentangle underlying factors of variation in the data, most of them operate in 2D and hence ignore that our world is three-dimensional. Further, only few works consider the compositional nature of scenes. Our key hypothesis is that incorporating a compositional 3D scene representation into the generative model leads to more controllable image synthesis. Representing scenes as compositional generative neural feature fields allows us to disentangle one or multiple objects from the background as well as individual objects' shapes and appearances while learning from unstructured and unposed image collections without any additional supervision. Combining this scene representation with a neural rendering pipeline yields a fast and realistic image synthesis model. As evidenced by our experiments, our model is able to disentangle individual objects and allows for translating and rotating them in the scene as well as changing the camera pose.

count=1
* AGORA: Avatars in Geography Optimized for Regression Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Patel_AGORA_Avatars_in_Geography_Optimized_for_Regression_Analysis_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Patel_AGORA_Avatars_in_Geography_Optimized_for_Regression_Analysis_CVPR_2021_paper.pdf)]
    * Title: AGORA: Avatars in Geography Optimized for Regression Analysis
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Priyanka Patel, Chun-Hao P. Huang, Joachim Tesch, David T. Hoffmann, Shashank Tripathi, Michael J. Black
    * Abstract: While the accuracy of 3D human pose estimation from images has steadily improved on benchmark datasets, the best methods still fail in many real-world scenarios. This suggests that there is a domain gap between current datasets and common scenes containing people. To obtain ground-truth 3D pose, current datasets limit the complexity of clothing, environmental conditions, number of subjects, and occlusion. Moreover, current datasets evaluate sparse 3D joint locations corresponding to the major joints of the body, ignoring the hand pose and the face shape. To evaluate the current state-of-the-art methods on more challenging images, and to drive the field to address new problems, we introduce AGORA, a synthetic dataset with high realism and highly accurate ground truth. Here we use 4240 commercially-available, high-quality, textured human scans in diverse poses and natural clothing; this includes 257 scans of children. We create reference 3D poses and body shapes by fitting the SMPL-X body model (with face and hands) to the 3D scans, taking into account clothing. We create around 14K training and 3K test images by rendering between 5 and 15 people per image using either image-based lighting or rendered 3D environments, taking care to make the images physically plausible and photoreal. In total, AGORA consists of 173K individual person crops. We evaluate existing state-of-the-art methods for 3D human pose estimation on this dataset and find that most methods perform poorly on images of children. Hence, we extend the SMPL-X model to better capture the shape of children. Additionally, we fine-tune methods on AGORA and show improved performance on both AGORA and 3DPW, confirming the realism of the dataset. We provide all the registered 3D reference training data, rendered images, and a web-based evaluation site at https://agora.is.tue.mpg.de/.

count=1
* Neural Body: Implicit Neural Representations With Structured Latent Codes for Novel View Synthesis of Dynamic Humans
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.pdf)]
    * Title: Neural Body: Implicit Neural Representations With Structured Latent Codes for Novel View Synthesis of Dynamic Humans
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, Xiaowei Zhou
    * Abstract: This paper addresses the challenge of novel view synthesis for a human performer from a very sparse set of camera views. Some recent works have shown that learning implicit neural representations of 3D scenes achieves remarkable view synthesis quality given dense input views. However, the representation learning will be ill-posed if the views are highly sparse. To solve this ill-posed problem, our key idea is to integrate observations over video frames. To this end, we propose Neural Body, a new human body representation which assumes that the learned neural representations at different frames share the same set of latent codes anchored to a deformable mesh, so that the observations across frames can be naturally integrated. The deformable mesh also provides geometric guidance for the network to learn 3D representations more efficiently. Experiments on a newly collected multi-view dataset show that our approach outperforms prior works by a large margin in terms of the novel view synthesis quality. We also demonstrate the capability of our approach to reconstruct a moving person from a monocular video on the People-Snapshot dataset. We will release the code and dataset for reproducibility.

count=1
* SSN: Soft Shadow Network for Image Compositing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Sheng_SSN_Soft_Shadow_Network_for_Image_Compositing_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Sheng_SSN_Soft_Shadow_Network_for_Image_Compositing_CVPR_2021_paper.pdf)]
    * Title: SSN: Soft Shadow Network for Image Compositing
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yichen Sheng, Jianming Zhang, Bedrich Benes
    * Abstract: We introduce an interactive Soft Shadow Network (SSN) to generates controllable soft shadows for image compositing. SSN takes a 2D object mask as input and thus is agnostic to image types such as painting and vector art. An environment light map is used to control the shadow's characteristics, such as angle and softness. SSN employs an Ambient Occlusion Prediction module to predict an intermediate ambient occlusion map, which can be further refined by the user to provides geometric cues to modulate the shadow generation. To train our model, we design an efficient pipeline to produce diverse soft shadow training data using 3D object models. In addition, we propose an inverse shadow map representation to improve model training. We demonstrate that our model produces realistic soft shadows in real-time. Our user studies show that the generated shadows are often indistinguishable from shadows calculated by a physics-based renderer and users can easily use SSN through an interactive application to generate specific shadow effects in minutes.

count=1
* Using Shape To Categorize: Low-Shot Learning With an Explicit Shape Bias
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Stojanov_Using_Shape_To_Categorize_Low-Shot_Learning_With_an_Explicit_Shape_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Stojanov_Using_Shape_To_Categorize_Low-Shot_Learning_With_an_Explicit_Shape_CVPR_2021_paper.pdf)]
    * Title: Using Shape To Categorize: Low-Shot Learning With an Explicit Shape Bias
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Stefan Stojanov, Anh Thai, James M. Rehg
    * Abstract: It is widely accepted that reasoning about object shape is important for object recognition. However, the most powerful object recognition methods today do not explicitly make use of object shape during learning. In this work, motivated by recent developments in low-shot learning, findings in developmental psychology, and the increased use of synthetic data in computer vision research, we investigate how reasoning about 3D shape can be used to improve low-shot learning methods' generalization performance. We propose a new way to improve existing low-shot learning approaches by learning a discriminative embedding space using 3D object shape, and using this embedding by learning how to map images into it. Our new approach improves the performance of image-only low-shot learning approaches on multiple datasets. We also introduce Toys4K, a 3D object dataset with the largest number of object categories currently available, which supports low-shot learning.

count=1
* Learning View Selection for 3D Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Learning_View_Selection_for_3D_Scenes_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Learning_View_Selection_for_3D_Scenes_CVPR_2021_paper.pdf)]
    * Title: Learning View Selection for 3D Scenes
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yifan Sun, Qixing Huang, Dun-Yu Hsiao, Li Guan, Gang Hua
    * Abstract: Efficient 3D space sampling to represent an underlying3D object/scene is essential for 3D vision, robotics, and be-yond. A standard approach is to explicitly sample a densecollection of views and formulate it as a view selection prob-lem, or, more generally, a set cover problem. In this paper,we introduce a novel approach that avoids dense view sam-pling. The key idea is to learn a view prediction networkand a trainable aggregation module that takes the predictedviews as input and outputs an approximation of their genericscores (e.g., surface coverage, viewing angle from surfacenormals). This methodology allows us to turn the set coverproblem (or multi-view representation optimization) into acontinuous optimization problem. We then explain how toeffectively solve the induced optimization problem using con-tinuation, i.e., aggregating a hierarchy of smoothed scoringmodules. Experimental results show that our approach ar-rives at similar or better solutions with about 10 x speed upin running time, comparing with the standard methods.

count=1
* Learned Initializations for Optimizing Coordinate-Based Neural Representations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Tancik_Learned_Initializations_for_Optimizing_Coordinate-Based_Neural_Representations_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Tancik_Learned_Initializations_for_Optimizing_Coordinate-Based_Neural_Representations_CVPR_2021_paper.pdf)]
    * Title: Learned Initializations for Optimizing Coordinate-Based Neural Representations
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Matthew Tancik, Ben Mildenhall, Terrance Wang, Divi Schmidt, Pratul P. Srinivasan, Jonathan T. Barron, Ren Ng
    * Abstract: Coordinate-based neural representations have shown significant promise as an alternative to discrete, array-based representations for complex low dimensional signals. However, optimizing a coordinate-based network from randomly initialized weights for each new signal is inefficient. We propose applying standard meta-learning algorithms to learn the initial weight parameters for these fully-connected networks based on the underlying class of signals being represented (e.g., images of faces or 3D models of chairs). Despite requiring only a minor change in implementation, using these learned initial weights enables faster convergence during optimization and can serve as a strong prior over the signal class being modeled, resulting in better generalization when only partial observations of a given signal are available. We explore these benefits across a variety of tasks, including representing 2D images, reconstructing CT scans, and recovering 3D shapes and scenes from 2D image observations.

count=1
* IBRNet: Learning Multi-View Image-Based Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_IBRNet_Learning_Multi-View_Image-Based_Rendering_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_IBRNet_Learning_Multi-View_Image-Based_Rendering_CVPR_2021_paper.pdf)]
    * Title: IBRNet: Learning Multi-View Image-Based Rendering
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P. Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, Thomas Funkhouser
    * Abstract: We present a method that synthesizes novel views of complex scenes by interpolating a sparse set of nearby views. The core of our method is a network architecture that includes a multilayer perceptron and a ray transformer that estimates radiance and volume density at continuous 5D locations (3D spatial locations and 2D viewing directions), drawing appearance information on the fly from multiple source views. By drawing on source views at render time, our method hearkens back to classic work on image-based rendering (IBR), and allows us to render high-resolution imagery. Unlike neural scene representation work that optimizes per-scene functions for rendering, we learn a generic view interpolation function that generalizes to novel scenes. We render images using classic volume rendering, which is fully differentiable and allows us to train using only multi-view posed images as supervision. Experiments show that our method outperforms recent novel view synthesis methods that also seek to generalize to novel scenes. Further, if fine-tuned on each scene, our method is competitive with state-of-the-art single-scene neural rendering methods.

count=1
* S3: Neural Shape, Skeleton, and Skinning Fields for 3D Human Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_S3_Neural_Shape_Skeleton_and_Skinning_Fields_for_3D_Human_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_S3_Neural_Shape_Skeleton_and_Skinning_Fields_for_3D_Human_CVPR_2021_paper.pdf)]
    * Title: S3: Neural Shape, Skeleton, and Skinning Fields for 3D Human Modeling
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Ze Yang, Shenlong Wang, Sivabalan Manivasagam, Zeng Huang, Wei-Chiu Ma, Xinchen Yan, Ersin Yumer, Raquel Urtasun
    * Abstract: Constructing and animating humans is an important component for building virtual worlds in a wide variety of applications such as virtual reality or robotics testing in simulation. As there are exponentially many variations of humans with different shape, pose and clothing, it is critical to develop methods that can automatically reconstruct and animate humans at scale from real world data. Towards this goal, we represent the pedestrian's shape, pose and skinning weights as neural implicit functions that are directly learned from data. This representation enables us to handle a wide variety of different pedestrian shapes and poses without explicitly fitting a human parametric body model, allowing us to handle a wider range of human geometries and topologies. We demonstrate the effectiveness of our approach on various datasets and show that our reconstructions outperform existing state-of-the-art methods. Furthermore, our re-animation experiments show that we can generate 3D human animations at scale from a single RGB image (and/or an optional LiDAR sweep) as input.

count=1
* Stylized Neural Painting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zou_Stylized_Neural_Painting_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zou_Stylized_Neural_Painting_CVPR_2021_paper.pdf)]
    * Title: Stylized Neural Painting
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Zhengxia Zou, Tianyang Shi, Shuang Qiu, Yi Yuan, Zhenwei Shi
    * Abstract: This paper proposes an image-to-painting translation method that generates vivid and realistic painting artworks with controllable styles. Different from previous image-to-image translation methods that formulate the translation as pixel-wise prediction, we deal with such an artistic creation process in a vectorized environment and produce a sequence of physically meaningful stroke parameters that can be further used for rendering. Since a typical vector render is not differentiable, we design a novel neural renderer which imitates the behavior of the vector renderer and then frame the stroke prediction as a parameter searching process that maximizes the similarity between the input and the rendering output. We explored the zero-gradient problem on parameter searching and propose to solve this problem from an optimal transportation perspective. We also show that previous neural renderers have a parameter coupling problem and we re-design the rendering network with a rasterization network and a shading network that better handles the disentanglement of shape and color. Experiments show that the paintings generated by our method have a high degree of fidelity in both global appearance and local textures. Our method can be also jointly optimized with neural style transfer that further transfers visual style from other images. Our code and animated results are available at https://jiupinjia.github.io/neuralpainter/.

count=1
* Vision-Based Neural Scene Representations for Spacecraft
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Mergy_Vision-Based_Neural_Scene_Representations_for_Spacecraft_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Mergy_Vision-Based_Neural_Scene_Representations_for_Spacecraft_CVPRW_2021_paper.pdf)]
    * Title: Vision-Based Neural Scene Representations for Spacecraft
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Anne Mergy, Gurvan Lecuyer, Dawa Derksen, Dario Izzo
    * Abstract: In advanced mission concepts with high levels of autonomy, spacecraft need to internally model the pose and shape of nearby orbiting objects. Recent works in neural scene representations show promising results for inferring generic three-dimensional scenes from optical images. Neural Radiance Fields (NeRF) have shown success in rendering highly specular surfaces using a large number of images and their pose. More recently, Generative Radiance Fields (GRAF) achieved full volumetric reconstruction of a scene from unposed images only, thanks to the use of an adversarial framework to train a NeRF. In this paper, we compare and evaluate the potential of NeRF and GRAF to render novel views and extract the 3D shape of two different spacecraft, the Soil Moisture and Ocean Salinity satellite of ESA's Living Planet Programme and a generic cube sat. Considering the best performances of both models, we observe that NeRF has the ability to render more accurate images regarding the material specularity of the spacecraft and its pose. For its part, GRAF generates precise novel views with accurate details even when parts of the satellites are shadowed while having the significant advantage of not needing any information about the relative pose.

count=1
* Photorealistic Monocular 3D Reconstruction of Humans Wearing Clothing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Alldieck_Photorealistic_Monocular_3D_Reconstruction_of_Humans_Wearing_Clothing_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Alldieck_Photorealistic_Monocular_3D_Reconstruction_of_Humans_Wearing_Clothing_CVPR_2022_paper.pdf)]
    * Title: Photorealistic Monocular 3D Reconstruction of Humans Wearing Clothing
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Thiemo Alldieck, Mihai Zanfir, Cristian Sminchisescu
    * Abstract: We present PHORHUM, a novel, end-to-end trainable, deep neural network methodology for photorealistic 3D human reconstruction given just a monocular RGB image. Our pixel-aligned method estimates detailed 3D geometry and, for the first time, the unshaded surface color together with the scene illumination. Observing that 3D supervision alone is not sufficient for high fidelity color reconstruction, we introduce patch-based rendering losses that enable reliable color reconstruction on visible parts of the human, and detailed and plausible color estimation for the non-visible parts. Moreover, our method specifically addresses methodological and practical limitations of prior work in terms of representing geometry, albedo, and illumination effects, in an end-to-end model where factors can be effectively disentangled. In extensive experiments, we demonstrate the versatility and robustness of our approach. Our state-of-the-art results validate the method qualitatively and for different metrics, for both geometric and color reconstruction.

count=1
* RigNeRF: Fully Controllable Neural 3D Portraits
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Athar_RigNeRF_Fully_Controllable_Neural_3D_Portraits_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Athar_RigNeRF_Fully_Controllable_Neural_3D_Portraits_CVPR_2022_paper.pdf)]
    * Title: RigNeRF: Fully Controllable Neural 3D Portraits
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: ShahRukh Athar, Zexiang Xu, Kalyan Sunkavalli, Eli Shechtman, Zhixin Shu
    * Abstract: Volumetric neural rendering methods, such as neural ra-diance fields (NeRFs), have enabled photo-realistic novel view synthesis. However, in their standard form, NeRFs do not support the editing of objects, such as a human head,within a scene. In this work, we propose RigNeRF, a system that goes beyond just novel view synthesis and enables full control of head pose and facial expressions learned from a single portrait video. We model changes in head pose and facial expressions using a deformation field that is guided by a 3D morphable face model (3DMM). The 3DMM effectively acts as a prior for RigNeRF that learns to predict only residuals to the 3DMM deformations and allows us to render novel (rigid) poses and (non-rigid) expressions that were not present in the input sequence. Using only a smartphone-captured short video of a subject for training,we demonstrate the effectiveness of our method on free view synthesis of a portrait scene with explicit head pose and expression controls.

count=1
* Mip-NeRF 360
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.pdf)]
    * Title: Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, Peter Hedman
    * Abstract: Though neural radiance fields ("NeRF") have demonstrated impressive view synthesis results on objects and small bounded regions of space, they struggle on "unbounded" scenes, where the camera may point in any direction and content may exist at any distance. In this setting, existing NeRF-like models often produce blurry or low-resolution renderings (due to the unbalanced detail and scale of nearby and distant objects), are slow to train, and may exhibit artifacts due to the inherent ambiguity of the task of reconstructing a large scene from a small set of images. We present an extension of mip-NeRF (a NeRF variant that addresses sampling and aliasing) that uses a non-linear scene parameterization, online distillation, and a novel distortion-based regularizer to overcome the challenges presented by unbounded scenes. Our model, which we dub "mip-NeRF 360" as we target scenes in which the camera rotates 360 degrees around a point, reduces mean-squared error by 57% compared to mip-NeRF, and is able to produce realistic synthesized views and detailed depth maps for highly intricate, unbounded real-world scenes.

count=1
* Improving Neural Implicit Surfaces Geometry With Patch Warping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Darmon_Improving_Neural_Implicit_Surfaces_Geometry_With_Patch_Warping_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Darmon_Improving_Neural_Implicit_Surfaces_Geometry_With_Patch_Warping_CVPR_2022_paper.pdf)]
    * Title: Improving Neural Implicit Surfaces Geometry With Patch Warping
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: François Darmon, Bénédicte Bascle, Jean-Clément Devaux, Pascal Monasse, Mathieu Aubry
    * Abstract: Neural implicit surfaces have become an important technique for multi-view 3D reconstruction but their accuracy remains limited. In this paper, we argue that this comes from the difficulty to learn and render high frequency textures with neural networks. We thus propose to add to the standard neural rendering optimization a direct photo-consistency term across the different views. Intuitively, we optimize the implicit geometry so that it warps views on each other in a consistent way. We demonstrate that two elements are key to the success of such an approach: (i) warping entire patches, using the predicted occupancy and normals of the 3D points along each ray, and measuring their similarity with a robust structural similarity (SSIM); (ii) handling visibility and occlusion in such a way that incorrect warps are not given too much importance while encouraging a reconstruction as complete as possible. We evaluate our approach, dubbed NeuralWarp, on the standard DTU and EPFL benchmarks and show it outperforms state of the art unsupervised implicit surfaces reconstructions by over 20% on both datasets.

count=1
* GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_GRAM_Generative_Radiance_Manifolds_for_3D-Aware_Image_Generation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_GRAM_Generative_Radiance_Manifolds_for_3D-Aware_Image_Generation_CVPR_2022_paper.pdf)]
    * Title: GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yu Deng, Jiaolong Yang, Jianfeng Xiang, Xin Tong
    * Abstract: 3D-aware image generative modeling aims to generate 3D-consistent images with explicitly controllable camera poses. Recent works have shown promising results by training neural radiance field (NeRF) generators on unstructured 2D images, but still cannot generate highly-realistic images with fine details. A critical reason is that the high memory and computation cost of volumetric representation learning greatly restricts the number of point samples for radiance integration during training. Deficient sampling not only limits the expressive power of the generator to handle fine details but also impedes effective GAN training due to the noise caused by unstable Monte Carlo sampling. We propose a novel approach that regulates point sampling and radiance field learning on 2D manifolds, embodied as a set of learned implicit surfaces in the 3D volume. For each viewing ray, we calculate ray-surface intersections and accumulate their radiance generated by the network. By training and rendering such radiance manifolds, our generator can produce high quality images with realistic fine details and strong visual 3D consistency.

count=1
* Topologically-Aware Deformation Fields for Single-View 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Duggal_Topologically-Aware_Deformation_Fields_for_Single-View_3D_Reconstruction_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Duggal_Topologically-Aware_Deformation_Fields_for_Single-View_3D_Reconstruction_CVPR_2022_paper.pdf)]
    * Title: Topologically-Aware Deformation Fields for Single-View 3D Reconstruction
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Shivam Duggal, Deepak Pathak
    * Abstract: We present a new framework to learn dense 3D reconstruction and correspondence from a single 2D image. The shape is represented implicitly as deformation over a category-level occupancy field and learned in an unsupervised manner from an unaligned image collection without using any 3D supervision. However, image collections usually contain large intra-category topological variation, e.g. images of different chair instances, posing a major challenge. Hence, prior methods are either restricted only to categories with no topological variation for estimating shape and correspondence or focus only on learning shape independently for each instance without any correspondence. To address this issue, we propose a topologically-aware deformation field that maps 3D points in object space to a higher-dimensional canonical space. Given a single image, we first implicitly deform a 3D point in the object space to a learned category-specific canonical space using the topologically-aware field and then learn the 3D shape in the canonical space. Both the canonical shape and deformation field are trained end-to-end using differentiable rendering via learned recurrent ray marcher. Our approach, dubbed TARS, achieves state-of-the-art reconstruction fidelity on several datasets: ShapeNet, Pascal3D+, CUB, and Pix3D chairs.

count=1
* Plenoxels: Radiance Fields Without Neural Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Fridovich-Keil_Plenoxels_Radiance_Fields_Without_Neural_Networks_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Fridovich-Keil_Plenoxels_Radiance_Fields_Without_Neural_Networks_CVPR_2022_paper.pdf)]
    * Title: Plenoxels: Radiance Fields Without Neural Networks
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, Angjoo Kanazawa
    * Abstract: We introduce Plenoxels (plenoptic voxels), a system for photorealistic view synthesis. Plenoxels represent a scene as a sparse 3D grid with spherical harmonics. This representation can be optimized from calibrated images via gradient methods and regularization without any neural components. On standard, benchmark tasks, Plenoxels are optimized two orders of magnitude faster than Neural Radiance Fields with no loss in visual quality. For video and code, please see https://alexyu.net/plenoxels.

count=1
* ObjectFolder 2.0: A Multisensory Object Dataset for Sim2Real Transfer
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Gao_ObjectFolder_2.0_A_Multisensory_Object_Dataset_for_Sim2Real_Transfer_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_ObjectFolder_2.0_A_Multisensory_Object_Dataset_for_Sim2Real_Transfer_CVPR_2022_paper.pdf)]
    * Title: ObjectFolder 2.0: A Multisensory Object Dataset for Sim2Real Transfer
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Ruohan Gao, Zilin Si, Yen-Yu Chang, Samuel Clarke, Jeannette Bohg, Li Fei-Fei, Wenzhen Yuan, Jiajun Wu
    * Abstract: Objects play a crucial role in our everyday activities. Though multisensory object-centric learning has shown great potential lately, the modeling of objects in prior work is rather unrealistic. ObjectFolder 1.0 is a recent dataset that introduces 100 virtualized objects with visual, auditory, and tactile sensory data. However, the dataset is small in scale and the multisensory data is of limited quality, hampering generalization to real-world scenarios. We present ObjectFolder 2.0, a large-scale, multisensory dataset of common household objects in the form of implicit neural representations that significantly enhances ObjectFolder 1.0 in three aspects. First, our dataset is 10 times larger in the amount of objects and orders of magnitude faster in rendering time. Second, we significantly improve the multisensory rendering quality for all three modalities. Third, we show that models learned from virtual objects in our dataset successfully transfer to their real-world counterparts in three challenging tasks: object scale estimation, contact localization, and shape reconstruction. ObjectFolder 2.0 offers a new path and testbed for multisensory learning in computer vision and robotics. The dataset is available at https://github.com/rhgao/ObjectFolder.

count=1
* Learning 3D Object Shape and Layout Without 3D Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Gkioxari_Learning_3D_Object_Shape_and_Layout_Without_3D_Supervision_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Gkioxari_Learning_3D_Object_Shape_and_Layout_Without_3D_Supervision_CVPR_2022_paper.pdf)]
    * Title: Learning 3D Object Shape and Layout Without 3D Supervision
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Georgia Gkioxari, Nikhila Ravi, Justin Johnson
    * Abstract: A 3D scene consists of a set of objects, each with a shape and a layout giving their position in space. Understanding 3D scenes from 2D images is an important goal, with applications in robotics and graphics. While there have been recent advances in predicting 3D shape and layout from a single image, most approaches rely on 3D ground truth for training which is expensive to collect at scale. We overcome these limitations and propose a method that learns to predict 3D shape and layout for objects without any ground truth shape or layout information: instead we rely on multi-view images with 2D supervision which can more easily be collected at scale. Through extensive experiments on ShapeNet, Hypersim, and ScanNet we demonstrate that our approach scales to large datasets of realistic images, and compares favorably to methods relying on 3D ground truth. On Hypersim and ScanNet where reliable 3D ground truth is not available, our approach outperforms supervised approaches trained on smaller and less diverse datasets.

count=1
* IFOR: Iterative Flow Minimization for Robotic Object Rearrangement
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Goyal_IFOR_Iterative_Flow_Minimization_for_Robotic_Object_Rearrangement_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Goyal_IFOR_Iterative_Flow_Minimization_for_Robotic_Object_Rearrangement_CVPR_2022_paper.pdf)]
    * Title: IFOR: Iterative Flow Minimization for Robotic Object Rearrangement
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Ankit Goyal, Arsalan Mousavian, Chris Paxton, Yu-Wei Chao, Brian Okorn, Jia Deng, Dieter Fox
    * Abstract: Accurate object rearrangement from vision is a crucial problem for a wide variety of real-world robotics applications in unstructured environments. We propose IFOR, Iterative Flow Minimization for Robotic Object Rearrangement, an end-to-end method for the challenging problem of object rearrangement for unknown objects given an RGBD image of the original and final scenes. First, we learn an optical flow model based on RAFT to estimate the relative transformation of the objects purely from synthetic data. This flow is then used in an iterative minimization algorithm to achieve accurate positioning of previously unseen objects. Crucially, we show that our method applies to cluttered scenes, and in the real world, while training only on synthetic data. Videos are available at https://imankgoyal.github.io/ifor.html.

count=1
* NeRFReN: Neural Radiance Fields With Reflections
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_NeRFReN_Neural_Radiance_Fields_With_Reflections_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_NeRFReN_Neural_Radiance_Fields_With_Reflections_CVPR_2022_paper.pdf)]
    * Title: NeRFReN: Neural Radiance Fields With Reflections
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yuan-Chen Guo, Di Kang, Linchao Bao, Yu He, Song-Hai Zhang
    * Abstract: Neural Radiance Fields (NeRF) has achieved unprecedented view synthesis quality using coordinate-based neural scene representations. However, NeRF's view dependency can only handle simple reflections like highlights but cannot deal with complex reflections such as those from glass and mirrors. In these scenarios, NeRF models the virtual image as real geometries which leads to inaccurate depth estimation, and produces blurry renderings when the multi-view consistency is violated as the reflected objects may only be seen under some of the viewpoints. To overcome these issues, we introduce NeRFReN, which is built upon NeRF to model scenes with reflections. Specifically, we propose to split a scene into transmitted and reflected components, and model the two components with separate neural radiance fields. Considering that this decomposition is highly under-constrained, we exploit geometric priors and apply carefully-designed training strategies to achieve reasonable decomposition results. Experiments on various self-captured scenes show that our method achieves high-quality novel view synthesis and physically sound depth estimation results while enabling scene editing applications.

count=1
* HDR-NeRF
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_HDR-NeRF_High_Dynamic_Range_Neural_Radiance_Fields_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_HDR-NeRF_High_Dynamic_Range_Neural_Radiance_Fields_CVPR_2022_paper.pdf)]
    * Title: HDR-NeRF: High Dynamic Range Neural Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Xin Huang, Qi Zhang, Ying Feng, Hongdong Li, Xuan Wang, Qing Wang
    * Abstract: We present High Dynamic Range Neural Radiance Fields (HDR-NeRF) to recover an HDR radiance field from a set of low dynamic range (LDR) views with different exposures. Using the HDR-NeRF, we are able to generate both novel HDR views and novel LDR views under different exposures. The key to our method is to model the physical imaging process, which dictates that the radiance of a scene point transforms to a pixel value in the LDR image with two implicit functions: a radiance field and a tone mapper. The radiance field encodes the scene radiance (values vary from 0 to +infty), which outputs the density and radiance of a ray by giving corresponding ray origin and ray direction. The tone mapper models the mapping process that a ray hitting on the camera sensor becomes a pixel value. The color of the ray is predicted by feeding the radiance and the corresponding exposure time into the tone mapper. We use the classic volume rendering technique to project the output radiance, colors, and densities into HDR and LDR images, while only the input LDR images are used as the supervision. We collect a new forward-facing HDR dataset to evaluate the proposed method. Experimental results on synthetic and real-world scenes validate that our method can not only accurately control the exposures of synthesized views but also render views with a high dynamic range.

count=1
* SelfRecon: Self Reconstruction Your Digital Avatar From Monocular Video
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_SelfRecon_Self_Reconstruction_Your_Digital_Avatar_From_Monocular_Video_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_SelfRecon_Self_Reconstruction_Your_Digital_Avatar_From_Monocular_Video_CVPR_2022_paper.pdf)]
    * Title: SelfRecon: Self Reconstruction Your Digital Avatar From Monocular Video
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Boyi Jiang, Yang Hong, Hujun Bao, Juyong Zhang
    * Abstract: We propose SelfRecon, a clothed human body reconstruction method that combines implicit and explicit representations to recover space-time coherent geometries from a monocular self-rotating human video. Explicit methods require a predefined template mesh for a given sequence, while the template is hard to acquire for a specific subject. Meanwhile, the fixed topology limits the reconstruction accuracy and clothing types. Implicit representation supports arbitrary topology and can represent high-fidelity geometry shapes due to its continuous nature. However, it is difficult to integrate multi-frame information to produce a consistent registration sequence for downstream applications. We propose to combine the advantages of both representations. We utilize differential mask loss of the explicit mesh to obtain the coherent overall shape, while the details on the implicit surface are refined with the differentiable neural rendering. Meanwhile, the explicit mesh is updated periodically to adjust its topology changes, and a consistency loss is designed to match both representations. Compared with existing methods, SelfRecon can produce high-fidelity surfaces for arbitrary clothed humans with self-supervised optimization. Extensive experimental results demonstrate its effectiveness on real captured monocular videos. The source code is available at https://github.com/jby1993/SelfReconCode.

count=1
* CoNeRF: Controllable Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Kania_CoNeRF_Controllable_Neural_Radiance_Fields_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Kania_CoNeRF_Controllable_Neural_Radiance_Fields_CVPR_2022_paper.pdf)]
    * Title: CoNeRF: Controllable Neural Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Kacper Kania, Kwang Moo Yi, Marek Kowalski, Tomasz Trzciński, Andrea Tagliasacchi
    * Abstract: We extend neural 3D representations to allow for intuitive and interpretable user control beyond novel view rendering (i.e. camera control). We allow the user to annotate which part of the scene one wishes to control with just a small number of mask annotations in the training images. Our key idea is to treat the attributes as latent variables that are regressed by the neural network given the scene encoding. This leads to a few-shot learning framework, where attributes are discovered automatically by the framework when annotations are not provided. We apply our method to various scenes with different types of controllable attributes (e.g. expression control on human faces, or state control in the movement of inanimate objects). Overall, we demonstrate, to the best of our knowledge, for the first time novel view and novel attribute re-rendering of scenes from a single video.

count=1
* RegNeRF: Regularizing Neural Radiance Fields for View Synthesis From Sparse Inputs
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Niemeyer_RegNeRF_Regularizing_Neural_Radiance_Fields_for_View_Synthesis_From_Sparse_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Niemeyer_RegNeRF_Regularizing_Neural_Radiance_Fields_for_View_Synthesis_From_Sparse_CVPR_2022_paper.pdf)]
    * Title: RegNeRF: Regularizing Neural Radiance Fields for View Synthesis From Sparse Inputs
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall, Mehdi S. M. Sajjadi, Andreas Geiger, Noha Radwan
    * Abstract: Neural Radiance Fields (NeRF) have emerged as a powerful representation for the task of novel view synthesis due to their simplicity and state-of-the-art performance. Though NeRF can produce photorealistic renderings of unseen viewpoints when many input views are available, its performance drops significantly when this number is reduced. We observe that the majority of artifacts in sparse input scenarios are caused by errors in the estimated scene geometry, and by divergent behavior at the start of training. We address this by regularizing the geometry and appearance of patches rendered from unobserved viewpoints, and annealing the ray sampling space during training. We additionally use a normalizing flow model to regularize the color of unobserved viewpoints. Our model outperforms not only other methods that optimize over a single scene, but in many cases also conditional models that are extensively pre-trained on large multi-view datasets.

count=1
* StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Or-El_StyleSDF_High-Resolution_3D-Consistent_Image_and_Geometry_Generation_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Or-El_StyleSDF_High-Resolution_3D-Consistent_Image_and_Geometry_Generation_CVPR_2022_paper.pdf)]
    * Title: StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Roy Or-El, Xuan Luo, Mengyi Shan, Eli Shechtman, Jeong Joon Park, Ira Kemelmacher-Shlizerman
    * Abstract: We introduce a high resolution, 3D-consistent image and shape generation technique which we call StyleSDF. Our method is trained on single view RGB data only, and stands on the shoulders of StyleGAN2 for image generation, while solving two main challenges in 3D-aware GANs: 1) high-resolution, view-consistent generation of the RGB images, and 2) detailed 3D shape. We achieve this by merging an SDF-based 3D representation with a style-based 2D generator. Our 3D implicit network renders low-resolution feature maps, from which the style-based network generates view-consistent, 1024x1024 images. Notably, our SDF-based 3D modeling defines detailed 3D surfaces, leading to consistent volume rendering. Our method shows higher quality results compared to state of the art in terms of visual and geometric quality.

count=1
* GenDR: A Generalized Differentiable Renderer
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Petersen_GenDR_A_Generalized_Differentiable_Renderer_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Petersen_GenDR_A_Generalized_Differentiable_Renderer_CVPR_2022_paper.pdf)]
    * Title: GenDR: A Generalized Differentiable Renderer
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Felix Petersen, Bastian Goldluecke, Christian Borgelt, Oliver Deussen
    * Abstract: In this work, we present and study a generalized family of differentiable renderers. We discuss from scratch which components are necessary for differentiable rendering and formalize the requirements for each component.We instantiate our general differentiable renderer, which generalizes existing differentiable renderers like SoftRas and DIB-R, with an array of different smoothing distributions to cover a large spectrum of reasonable settings. We evaluate an array of differentiable renderer instantiations on the popular ShapeNet 3D reconstruction benchmark and analyze the implications of our results. Surprisingly, the simple uniform distribution yields the best overall results when averaged over 13 classes; in general, however, the optimal choice of distribution heavily depends on the task.

count=1
* LOLNerf: Learn From One Look
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Rebain_LOLNerf_Learn_From_One_Look_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Rebain_LOLNerf_Learn_From_One_Look_CVPR_2022_paper.pdf)]
    * Title: LOLNerf: Learn From One Look
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Daniel Rebain, Mark Matthews, Kwang Moo Yi, Dmitry Lagun, Andrea Tagliasacchi
    * Abstract: We present a method for learning a generative 3D model based on neural radiance fields, trained solely from data with only single views of each object. While generating realistic images is no longer a difficult task, producing the corresponding 3D structure such that they can be rendered from different views is non-trivial. We show that, unlike existing methods, one does not need multi-view data to achieve this goal. Specifically, we show that by reconstructing many images aligned to an approximate canonical pose with a single network conditioned on a shared latent space, you can learn a space of radiance fields that models shape and appearance for a class of objects. We demonstrate this by training models to reconstruct object categories using datasets that contain only one view of each subject without depth or geometry information. Our experiments show that we achieve state-of-the-art results in novel view synthesis and high-quality results for monocular depth prediction.

count=1
* SRT
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Sajjadi_Scene_Representation_Transformer_Geometry-Free_Novel_View_Synthesis_Through_Set-Latent_Scene_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Sajjadi_Scene_Representation_Transformer_Geometry-Free_Novel_View_Synthesis_Through_Set-Latent_Scene_CVPR_2022_paper.pdf)]
    * Title: Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Mehdi S. M. Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Lučić, Daniel Duckworth, Alexey Dosovitskiy, Jakob Uszkoreit, Thomas Funkhouser, Andrea Tagliasacchi
    * Abstract: A classical problem in computer vision is to infer a 3D scene representation from few images that can be used to render novel views at interactive rates. Previous work focuses on reconstructing pre-defined 3D representations, e.g. textured meshes, or implicit representations, e.g. radiance fields, and often requires input images with precise camera poses and long processing times for each novel scene. In this work, we propose the Scene Representation Transformer (SRT), a method which processes posed or unposed RGB images of a new area, infers a "set-latent scene representation", and synthesises novel views, all in a single feed-forward pass. To calculate the scene representation, we propose a generalization of the Vision Transformer to sets of images, enabling global information integration, and hence 3D reasoning. An efficient decoder transformer parameterizes the light field by attending into the scene representation to render novel views. Learning is supervised end-to-end by minimizing a novel-view reconstruction error. We show that this method outperforms recent baselines in terms of PSNR and speed on synthetic datasets, including a new dataset created for the paper. Further, we demonstrate that SRT scales to support interactive visualization and semantic segmentation of real-world outdoor environments using Street View imagery.

count=1
* Multi-View Mesh Reconstruction With Neural Deferred Shading
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Worchel_Multi-View_Mesh_Reconstruction_With_Neural_Deferred_Shading_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Worchel_Multi-View_Mesh_Reconstruction_With_Neural_Deferred_Shading_CVPR_2022_paper.pdf)]
    * Title: Multi-View Mesh Reconstruction With Neural Deferred Shading
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Markus Worchel, Rodrigo Diaz, Weiwen Hu, Oliver Schreer, Ingo Feldmann, Peter Eisert
    * Abstract: We propose an analysis-by-synthesis method for fast multi-view 3D reconstruction of opaque objects with arbitrary materials and illumination. State-of-the-art methods use both neural surface representations and neural rendering. While flexible, neural surface representations are a significant bottleneck in optimization runtime. Instead, we represent surfaces as triangle meshes and build a differentiable rendering pipeline around triangle rasterization and neural shading. The renderer is used in a gradient descent optimization where both a triangle mesh and a neural shader are jointly optimized to reproduce the multi-view images. We evaluate our method on a public 3D reconstruction dataset and show that it can match the reconstruction accuracy of traditional baselines and neural approaches while surpassing them in optimization runtime. Additionally, we investigate the shader and find that it learns an interpretable representation of appearance, enabling applications such as 3D material editing.

count=1
* DIVeR: Real-Time and Accurate Neural Radiance Fields With Deterministic Integration for Volume Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_DIVeR_Real-Time_and_Accurate_Neural_Radiance_Fields_With_Deterministic_Integration_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_DIVeR_Real-Time_and_Accurate_Neural_Radiance_Fields_With_Deterministic_Integration_CVPR_2022_paper.pdf)]
    * Title: DIVeR: Real-Time and Accurate Neural Radiance Fields With Deterministic Integration for Volume Rendering
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Liwen Wu, Jae Yong Lee, Anand Bhattad, Yu-Xiong Wang, David Forsyth
    * Abstract: DIVeR builds on the key ideas of NeRF and its variants -- density models and volume rendering -- to learn 3D object models that can be rendered realistically from small numbers of images. In contrast to all previous NeRF methods, DIVeR uses deterministic rather than stochastic estimates of the volume rendering integral. DIVeR's representation is a voxel based field of features. To compute the volume rendering integral, a ray is broken into intervals, one per voxel; components of the volume rendering integral are estimated from the features for each interval using an MLP, and the components are aggregated. As a result, DIVeR can render thin translucent structures that are missed by other integrators. Furthermore, DIVeR's representation has semantics that is relatively exposed compared to other such methods -- moving feature vectors around in the voxel space results in natural edits. Extensive qualitative and quantitative comparisons to current state-of-the-art methods show that DIVeR produces models that (1) render at or above state-of-the-art quality, (2) are very small without being baked, (3) render very fast without being baked, and (4) can be edited in natural ways.

count=1
* FvOR: Robust Joint Shape and Pose Optimization for Few-View Object Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_FvOR_Robust_Joint_Shape_and_Pose_Optimization_for_Few-View_Object_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_FvOR_Robust_Joint_Shape_and_Pose_Optimization_for_Few-View_Object_CVPR_2022_paper.pdf)]
    * Title: FvOR: Robust Joint Shape and Pose Optimization for Few-View Object Reconstruction
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Zhenpei Yang, Zhile Ren, Miguel Angel Bautista, Zaiwei Zhang, Qi Shan, Qixing Huang
    * Abstract: Reconstructing an accurate 3D object model from a few image observations remains a challenging problem in computer vision. State-of-the-art approaches typically assume accurate camera poses as input, which could be difficult to obtain in realistic settings. In this paper, we present FvOR, a learning-based object reconstruction method that predicts accurate 3D models given a few images with noisy input poses. The core of our approach is a fast and robust multi-view reconstruction algorithm to jointly refine 3D geometry and camera pose estimation using learnable neural network modules. We provide a thorough benchmark of state-of-the-art approaches for this problem on ShapeNet. Our approach achieves best-in-class results. It is also two orders of magnitude faster than the recent optimization-based approach IDR.

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
* NeRF-Editing: Geometry Editing of Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yuan_NeRF-Editing_Geometry_Editing_of_Neural_Radiance_Fields_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yuan_NeRF-Editing_Geometry_Editing_of_Neural_Radiance_Fields_CVPR_2022_paper.pdf)]
    * Title: NeRF-Editing: Geometry Editing of Neural Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yu-Jie Yuan, Yang-Tian Sun, Yu-Kun Lai, Yuewen Ma, Rongfei Jia, Lin Gao
    * Abstract: Implicit neural rendering, especially Neural Radiance Field (NeRF), has shown great potential in novel view synthesis of a scene. However, current NeRF-based methods cannot enable users to perform user-controlled shape deformation in the scene. While existing works have proposed some approaches to modify the radiance field according to the user's constraints, the modification is limited to color editing or object translation and rotation. In this paper, we propose a method that allows users to perform controllable shape deformation on the implicit representation of the scene, and synthesizes the novel view images of the edited scene without re-training the network. Specifically, we establish a correspondence between the extracted explicit mesh representation and the implicit neural representation of the target scene. Users can first utilize well-developed mesh-based deformation methods to deform the mesh representation of the scene. Our method then utilizes user edits from the mesh representation to bend the camera rays by introducing a tetrahedra mesh as a proxy, obtaining the rendering results of the edited scene. Extensive experiments demonstrate that our framework can achieve ideal editing results not only on synthetic data, but also on real scenes captured by users.

count=1
* Critical Regularizations for Neural Surface Reconstruction in the Wild
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Critical_Regularizations_for_Neural_Surface_Reconstruction_in_the_Wild_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Critical_Regularizations_for_Neural_Surface_Reconstruction_in_the_Wild_CVPR_2022_paper.pdf)]
    * Title: Critical Regularizations for Neural Surface Reconstruction in the Wild
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Jingyang Zhang, Yao Yao, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan
    * Abstract: Neural implicit functions have recently shown promising results on surface reconstructions from multiple views. However, current methods still suffer from excessive time complexity and poor robustness when reconstructing unbounded or complex scenes. In this paper, we present RegSDF, which shows that proper point cloud supervisions and geometry regularizations are sufficient to produce high-quality and robust reconstruction results. Specifically, RegSDF takes an additional oriented point cloud as input, and optimizes a signed distance field and a surface light field within a differentiable rendering framework. We also introduce the two critical regularizations for this optimization. The first one is the Hessian regularization that smoothly diffuses the signed distance values to the entire distance field given noisy and incomplete input. And the second one is the minimal surface regularization that compactly interpolates and extrapolates the missing geometry. Extensive experiments are conducted on DTU, BlendedMVS, and Tanks and Temples datasets. Compared with recent neural surface reconstruction approaches, RegSDF is able to reconstruct surfaces with fine details even for open scenes with complex topologies and unstructured camera trajectories.

count=1
* HumanNeRF: Efficiently Generated Human Radiance Field From Sparse Inputs
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_HumanNeRF_Efficiently_Generated_Human_Radiance_Field_From_Sparse_Inputs_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_HumanNeRF_Efficiently_Generated_Human_Radiance_Field_From_Sparse_Inputs_CVPR_2022_paper.pdf)]
    * Title: HumanNeRF: Efficiently Generated Human Radiance Field From Sparse Inputs
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Fuqiang Zhao, Wei Yang, Jiakai Zhang, Pei Lin, Yingliang Zhang, Jingyi Yu, Lan Xu
    * Abstract: Recent neural human representations can produce high-quality multi-view rendering but require using dense multi-view inputs and costly training. They are hence largely limited to static models as training each frame is infeasible. We present HumanNeRF - a neural representation with efficient generalization ability - for high-fidelity free-view synthesis of dynamic humans. Analogous to how IBRNet assists NeRF by avoiding per-scene training, HumanNeRF employs an aggregated pixel-alignment feature across multi-view inputs along with a pose embedded non-rigid deformation field for tackling dynamic motions. The raw HumanNeRF can already produce reasonable rendering on sparse video inputs of unseen subjects and camera settings. To further improve the rendering quality, we augment our solution with in-hour scene-specific fine-tuning, and an appearance blending module for combining the benefits of both neural volumetric rendering and neural texture blending. Extensive experiments on various multi-view dynamic human datasets demonstrate effectiveness of our approach in synthesizing photo-realistic free-view humans under challenging motions and with very sparse camera view inputs.

count=1
* I M Avatar: Implicit Morphable Head Avatars From Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_I_M_Avatar_Implicit_Morphable_Head_Avatars_From_Videos_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_I_M_Avatar_Implicit_Morphable_Head_Avatars_From_Videos_CVPR_2022_paper.pdf)]
    * Title: I M Avatar: Implicit Morphable Head Avatars From Videos
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Yufeng Zheng, Victoria Fernández Abrevaya, Marcel C. Bühler, Xu Chen, Michael J. Black, Otmar Hilliges
    * Abstract: Traditional 3D morphable face models (3DMMs) provide fine-grained control over expression but cannot easily capture geometric and appearance details. Neural volumetric representations approach photorealism but are hard to animate and do not generalize well to unseen expressions. To tackle this problem, we propose IMavatar (Implicit Morphable avatar), a novel method for learning implicit head avatars from monocular videos. Inspired by the fine-grained control mechanisms afforded by conventional 3DMMs, we represent the expression- and pose- related deformations via learned blendshapes and skinning fields. These attributes are pose-independent and can be used to morph the canonical geometry and texture fields given novel expression and pose parameters. We employ ray marching and iterative root-finding to locate the canonical surface intersection for each pixel. A key contribution is our novel analytical gradient formulation that enables end- to-end training of IMavatars from videos. We show quantitatively and qualitatively that our method improves geometry and covers a more complete expression space compared to state-of-the-art methods. Code and data can be found at https://ait.ethz.ch/projects/2022/IMavatar/.

count=1
* Real-Time Neural Light Field on Mobile Devices
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Real-Time_Neural_Light_Field_on_Mobile_Devices_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Real-Time_Neural_Light_Field_on_Mobile_Devices_CVPR_2023_paper.pdf)]
    * Title: Real-Time Neural Light Field on Mobile Devices
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Junli Cao, Huan Wang, Pavlo Chemerys, Vladislav Shakhrai, Ju Hu, Yun Fu, Denys Makoviichuk, Sergey Tulyakov, Jian Ren
    * Abstract: Recent efforts in Neural Rendering Fields (NeRF) have shown impressive results on novel view synthesis by utilizing implicit neural representation to represent 3D scenes. Due to the process of volumetric rendering, the inference speed for NeRF is extremely slow, limiting the application scenarios of utilizing NeRF on resource-constrained hardware, such as mobile devices. Many works have been conducted to reduce the latency of running NeRF models. However, most of them still require high-end GPU for acceleration or extra storage memory, which is all unavailable on mobile devices. Another emerging direction utilizes the neural light field (NeLF) for speedup, as only one forward pass is performed on a ray to predict the pixel color. Nevertheless, to reach a similar rendering quality as NeRF, the network in NeLF is designed with intensive computation, which is not mobile-friendly. In this work, we propose an efficient network that runs in real-time on mobile devices for neural rendering. We follow the setting of NeLF to train our network. Unlike existing works, we introduce a novel network architecture that runs efficiently on mobile devices with low latency and small size, i.e., saving 15x 24x storage compared with MobileNeRF. Our model achieves high-resolution generation while maintaining real-time inference for both synthetic and real-world scenes on mobile devices, e.g., 18.04ms (iPhone 13) for rendering one 1008x756 image of real 3D scenes. Additionally, we achieve similar image quality as NeRF and better quality than MobileNeRF (PSNR 26.15 vs. 25.91 on the real-world forward-facing dataset).

count=1
* UV Volumes for Real-Time Rendering of Editable Free-View Human Performance
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_UV_Volumes_for_Real-Time_Rendering_of_Editable_Free-View_Human_Performance_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_UV_Volumes_for_Real-Time_Rendering_of_Editable_Free-View_Human_Performance_CVPR_2023_paper.pdf)]
    * Title: UV Volumes for Real-Time Rendering of Editable Free-View Human Performance
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yue Chen, Xuan Wang, Xingyu Chen, Qi Zhang, Xiaoyu Li, Yu Guo, Jue Wang, Fei Wang
    * Abstract: Neural volume rendering enables photo-realistic renderings of a human performer in free-view, a critical task in immersive VR/AR applications. But the practice is severely limited by high computational costs in the rendering process. To solve this problem, we propose the UV Volumes, a new approach that can render an editable free-view video of a human performer in real-time. It separates the high-frequency (i.e., non-smooth) human appearance from the 3D volume, and encodes them into 2D neural texture stacks (NTS). The smooth UV volumes allow much smaller and shallower neural networks to obtain densities and texture coordinates in 3D while capturing detailed appearance in 2D NTS. For editability, the mapping between the parameterized human model and the smooth texture coordinates allows us a better generalization on novel poses and shapes. Furthermore, the use of NTS enables interesting applications, e.g., retexturing. Extensive experiments on CMU Panoptic, ZJU Mocap, and H36M datasets show that our model can render 960 x 540 images in 30FPS on average with comparable photo-realism to state-of-the-art methods.

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
* Hybrid Neural Rendering for Large-Scale Scenes With Motion Blur
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Dai_Hybrid_Neural_Rendering_for_Large-Scale_Scenes_With_Motion_Blur_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Dai_Hybrid_Neural_Rendering_for_Large-Scale_Scenes_With_Motion_Blur_CVPR_2023_paper.pdf)]
    * Title: Hybrid Neural Rendering for Large-Scale Scenes With Motion Blur
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Peng Dai, Yinda Zhang, Xin Yu, Xiaoyang Lyu, Xiaojuan Qi
    * Abstract: Rendering novel view images is highly desirable for many applications. Despite recent progress, it remains challenging to render high-fidelity and view-consistent novel views of large-scale scenes from in-the-wild images with inevitable artifacts (e.g., motion blur). To this end, we develop a hybrid neural rendering model that makes image-based representation and neural 3D representation join forces to render high-quality, view-consistent images. Besides, images captured in the wild inevitably contain artifacts, such as motion blur, which deteriorates the quality of rendered images. Accordingly, we propose strategies to simulate blur effects on the rendered images to mitigate the negative influence of blurriness images and reduce their importance during training based on precomputed quality-aware weights. Extensive experiments on real and synthetic data demonstrate our model surpasses state-of-the-art point-based methods for novel view synthesis. The code is available at https://daipengwa.github.io/Hybrid-Rendering-ProjectPage.

count=1
* 3D-Aware Conditional Image Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Deng_3D-Aware_Conditional_Image_Synthesis_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Deng_3D-Aware_Conditional_Image_Synthesis_CVPR_2023_paper.pdf)]
    * Title: 3D-Aware Conditional Image Synthesis
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Kangle Deng, Gengshan Yang, Deva Ramanan, Jun-Yan Zhu
    * Abstract: We propose pix2pix3D, a 3D-aware conditional generative model for controllable photorealistic image synthesis. Given a 2D label map, such as a segmentation or edge map, our model learns to synthesize a corresponding image from different viewpoints. To enable explicit 3D user control, we extend conditional generative models with neural radiance fields. Given widely-available posed monocular image and label map pairs, our model learns to assign a label to every 3D point in addition to color and density, which enables it to render the image and pixel-aligned label map simultaneously. Finally, we build an interactive system that allows users to edit the label map from different viewpoints and generate outputs accordingly.

count=1
* NeRDi: Single-View NeRF Synthesis With Language-Guided Diffusion As General Image Priors
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Deng_NeRDi_Single-View_NeRF_Synthesis_With_Language-Guided_Diffusion_As_General_Image_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Deng_NeRDi_Single-View_NeRF_Synthesis_With_Language-Guided_Diffusion_As_General_Image_CVPR_2023_paper.pdf)]
    * Title: NeRDi: Single-View NeRF Synthesis With Language-Guided Diffusion As General Image Priors
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Congyue Deng, Chiyu “Max” Jiang, Charles R. Qi, Xinchen Yan, Yin Zhou, Leonidas Guibas, Dragomir Anguelov
    * Abstract: 2D-to-3D reconstruction is an ill-posed problem, yet humans are good at solving this problem due to their prior knowledge of the 3D world developed over years. Driven by this observation, we propose NeRDi, a single-view NeRF synthesis framework with general image priors from 2D diffusion models. Formulating single-view reconstruction as an image-conditioned 3D generation problem, we optimize the NeRF representations by minimizing a diffusion loss on its arbitrary view renderings with a pretrained image diffusion model under the input-view constraint. We leverage off-the-shelf vision-language models and introduce a two-section language guidance as conditioning inputs to the diffusion model. This is essentially helpful for improving multiview content coherence as it narrows down the general image prior conditioned on the semantic and visual features of the single-view input image. Additionally, we introduce a geometric loss based on estimated depth maps to regularize the underlying 3D geometry of the NeRF. Experimental results on the DTU MVS dataset show that our method can synthesize novel views with higher quality even compared to existing methods trained on this dataset. We also demonstrate our generalizability in zero-shot NeRF synthesis for in-the-wild images.

count=1
* Plateau-Reduced Differentiable Path Tracing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Fischer_Plateau-Reduced_Differentiable_Path_Tracing_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Fischer_Plateau-Reduced_Differentiable_Path_Tracing_CVPR_2023_paper.pdf)]
    * Title: Plateau-Reduced Differentiable Path Tracing
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Michael Fischer, Tobias Ritschel
    * Abstract: Current differentiable renderers provide light transport gradients with respect to arbitrary scene parameters. However, the mere existence of these gradients does not guarantee useful update steps in an optimization. Instead, inverse rendering might not converge due to inherent plateaus, i.e., regions of zero gradient, in the objective function. We propose to alleviate this by convolving the high-dimensional rendering function that maps scene parameters to images with an additional kernel that blurs the parameter space. We describe two Monte Carlo estimators to compute plateau-free gradients efficiently, i.e., with low variance, and show that these translate into net-gains in optimization error and runtime performance. Our approach is a straightforward extension to both black-box and differentiable renderers and enables the successful optimization of problems with intricate light transport, such as caustics or global illumination, that existing differentiable path tracers do not converge on. Our code is at github.com/mfischer-ucl/prdpt.

count=1
* SurfelNeRF: Neural Surfel Radiance Fields for Online Photorealistic Reconstruction of Indoor Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Gao_SurfelNeRF_Neural_Surfel_Radiance_Fields_for_Online_Photorealistic_Reconstruction_of_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_SurfelNeRF_Neural_Surfel_Radiance_Fields_for_Online_Photorealistic_Reconstruction_of_CVPR_2023_paper.pdf)]
    * Title: SurfelNeRF: Neural Surfel Radiance Fields for Online Photorealistic Reconstruction of Indoor Scenes
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yiming Gao, Yan-Pei Cao, Ying Shan
    * Abstract: Online reconstructing and rendering of large-scale indoor scenes is a long-standing challenge. SLAM-based methods can reconstruct 3D scene geometry progressively in real time but can not render photorealistic results. While NeRF-based methods produce promising novel view synthesis results, their long offline optimization time and lack of geometric constraints pose challenges to efficiently handling online input. Inspired by the complementary advantages of classical 3D reconstruction and NeRF, we thus investigate marrying explicit geometric representation with NeRF rendering to achieve efficient online reconstruction and high-quality rendering. We introduce SurfelNeRF, a variant of neural radiance field which employs a flexible and scalable neural surfel representation to store geometric attributes and extracted appearance features from input images. We further extend conventional surfel-based fusion scheme to progressively integrate incoming input frames into the reconstructed global neural scene representation. In addition, we propose a highly-efficient differentiable rasterization scheme for rendering neural surfel radiance fields, which helps SurfelNeRF achieve 10x speedups in training and inference time, respectively. Experimental results show that our method achieves the state-of-the-art 23.82 PSNR and 29.58 PSNR on ScanNet in feedforward inference and per-scene optimization settings, respectively.

count=1
* Learning Neural Volumetric Representations of Dynamic Humans in Minutes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Geng_Learning_Neural_Volumetric_Representations_of_Dynamic_Humans_in_Minutes_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Geng_Learning_Neural_Volumetric_Representations_of_Dynamic_Humans_in_Minutes_CVPR_2023_paper.pdf)]
    * Title: Learning Neural Volumetric Representations of Dynamic Humans in Minutes
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Chen Geng, Sida Peng, Zhen Xu, Hujun Bao, Xiaowei Zhou
    * Abstract: This paper addresses the challenge of efficiently reconstructing volumetric videos of dynamic humans from sparse multi-view videos. Some recent works represent a dynamic human as a canonical neural radiance field (NeRF) and a motion field, which are learned from input videos through differentiable rendering. But the per-scene optimization generally requires hours. Other generalizable NeRF models leverage learned prior from datasets to reduce the optimization time by only finetuning on new scenes at the cost of visual fidelity. In this paper, we propose a novel method for learning neural volumetric representations of dynamic humans in minutes with competitive visual quality. Specifically, we define a novel part-based voxelized human representation to better distribute the representational power of the network to different human parts. Furthermore, we propose a novel 2D motion parameterization scheme to increase the convergence rate of deformation field learning. Experiments demonstrate that our model can be learned 100 times faster than previous per-scene optimization methods while being competitive in the rendering quality. Training our model on a 512x512 video with 100 frames typically takes about 5 minutes on a single RTX 3090 GPU. The code is available on our project page: https://zju3dv.github.io/instant_nvr

count=1
* High-Fidelity 3D Human Digitization From Single 2K Resolution Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Han_High-Fidelity_3D_Human_Digitization_From_Single_2K_Resolution_Images_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_High-Fidelity_3D_Human_Digitization_From_Single_2K_Resolution_Images_CVPR_2023_paper.pdf)]
    * Title: High-Fidelity 3D Human Digitization From Single 2K Resolution Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Sang-Hun Han, Min-Gyu Park, Ju Hong Yoon, Ju-Mi Kang, Young-Jae Park, Hae-Gon Jeon
    * Abstract: High-quality 3D human body reconstruction requires high-fidelity and large-scale training data and appropriate network design that effectively exploits the high-resolution input images. To tackle these problems, we propose a simple yet effective 3D human digitization method called 2K2K, which constructs a large-scale 2K human dataset and infers 3D human models from 2K resolution images. The proposed method separately recovers the global shape of a human and its details. The low-resolution depth network predicts the global structure from a low-resolution image, and the part-wise image-to-normal network predicts the details of the 3D human body structure. The high-resolution depth network merges the global 3D shape and the detailed structures to infer the high-resolution front and back side depth maps. Finally, an off-the-shelf mesh generator reconstructs the full 3D human model, which are available at https://github.com/SangHunHan92/2K2K. In addition, we also provide 2,050 3D human models, including texture maps, 3D joints, and SMPL parameters for research purposes. In experiments, we demonstrate competitive performance over the recent works on various datasets.

count=1
* Multiscale Tensor Decomposition and Rendering Equation Encoding for View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Han_Multiscale_Tensor_Decomposition_and_Rendering_Equation_Encoding_for_View_Synthesis_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_Multiscale_Tensor_Decomposition_and_Rendering_Equation_Encoding_for_View_Synthesis_CVPR_2023_paper.pdf)]
    * Title: Multiscale Tensor Decomposition and Rendering Equation Encoding for View Synthesis
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Kang Han, Wei Xiang
    * Abstract: Rendering novel views from captured multi-view images has made considerable progress since the emergence of the neural radiance field. This paper aims to further advance the quality of view rendering by proposing a novel approach dubbed the neural radiance feature field (NRFF). We first propose a multiscale tensor decomposition scheme to organize learnable features so as to represent scenes from coarse to fine scales. We demonstrate many benefits of the proposed multiscale representation, including more accurate scene shape and appearance reconstruction, and faster convergence compared with the single-scale representation. Instead of encoding view directions to model view-dependent effects, we further propose to encode the rendering equation in the feature space by employing the anisotropic spherical Gaussian mixture predicted from the proposed multiscale representation. The proposed NRFF improves state-of-the-art rendering results by over 1 dB in PSNR on both the NeRF and NSVF synthetic datasets. A significant improvement has also been observed on the real-world Tanks & Temples dataset. Code can be found at https://github.com/imkanghan/nrff.

count=1
* RefSR-NeRF: Towards High Fidelity and Super Resolution View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_RefSR-NeRF_Towards_High_Fidelity_and_Super_Resolution_View_Synthesis_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_RefSR-NeRF_Towards_High_Fidelity_and_Super_Resolution_View_Synthesis_CVPR_2023_paper.pdf)]
    * Title: RefSR-NeRF: Towards High Fidelity and Super Resolution View Synthesis
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Xudong Huang, Wei Li, Jie Hu, Hanting Chen, Yunhe Wang
    * Abstract: We present Reference-guided Super-Resolution Neural Radiance Field (RefSR-NeRF) that extends NeRF to super resolution and photorealistic novel view synthesis. Despite NeRF's extraordinary success in the neural rendering field, it suffers from blur in high resolution rendering because its inherent multilayer perceptron struggles to learn high frequency details and incurs a computational explosion as resolution increases. Therefore, we propose RefSR-NeRF, an end-to-end framework that first learns a low resolution NeRF representation, and then reconstructs the high frequency details with the help of a high resolution reference image. We observe that simply introducing the pre-trained models from the literature tends to produce unsatisfied artifacts due to the divergence in the degradation model. To this end, we design a novel lightweight RefSR model to learn the inverse degradation process from NeRF renderings to target HR ones. Extensive experiments on multiple benchmarks demonstrate that our method exhibits an impressive trade-off among rendering quality, speed, and memory usage, outperforming or on par with NeRF and its variants while being 52x speedup with minor extra memory usage.

count=1
* Exact-NeRF: An Exploration of a Precise Volumetric Parameterization for Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Isaac-Medina_Exact-NeRF_An_Exploration_of_a_Precise_Volumetric_Parameterization_for_Neural_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Isaac-Medina_Exact-NeRF_An_Exploration_of_a_Precise_Volumetric_Parameterization_for_Neural_CVPR_2023_paper.pdf)]
    * Title: Exact-NeRF: An Exploration of a Precise Volumetric Parameterization for Neural Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Brian K. S. Isaac-Medina, Chris G. Willcocks, Toby P. Breckon
    * Abstract: Neural Radiance Fields (NeRF) have attracted significant attention due to their ability to synthesize novel scene views with great accuracy. However, inherent to their underlying formulation, the sampling of points along a ray with zero width may result in ambiguous representations that lead to further rendering artifacts such as aliasing in the final scene. To address this issue, the recent variant mip-NeRF proposes an Integrated Positional Encoding (IPE) based on a conical view frustum. Although this is expressed with an integral formulation, mip-NeRF instead approximates this integral as the expected value of a multivariate Gaussian distribution. This approximation is reliable for short frustums but degrades with highly elongated regions, which arises when dealing with distant scene objects under a larger depth of field. In this paper, we explore the use of an exact approach for calculating the IPE by using a pyramid-based integral formulation instead of an approximated conical-based one. We denote this formulation as Exact-NeRF and contribute the first approach to offer a precise analytical solution to the IPE within the NeRF domain. Our exploratory work illustrates that such an exact formulation (Exact-NeRF) matches the accuracy of mip-NeRF and furthermore provides a natural extension to more challenging scenarios without further modification, such as in the case of unbounded scenes. Our contribution aims to both address the hitherto unexplored issues of frustum approximation in earlier NeRF work and additionally provide insight into the potential future consideration of analytical solutions in future NeRF extensions.

count=1
* NeuralField-LDM: Scene Generation With Hierarchical Latent Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_NeuralField-LDM_Scene_Generation_With_Hierarchical_Latent_Diffusion_Models_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_NeuralField-LDM_Scene_Generation_With_Hierarchical_Latent_Diffusion_Models_CVPR_2023_paper.pdf)]
    * Title: NeuralField-LDM: Scene Generation With Hierarchical Latent Diffusion Models
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Seung Wook Kim, Bradley Brown, Kangxue Yin, Karsten Kreis, Katja Schwarz, Daiqing Li, Robin Rombach, Antonio Torralba, Sanja Fidler
    * Abstract: Automatically generating high-quality real world 3D scenes is of enormous interest for applications such as virtual reality and robotics simulation. Towards this goal, we introduce NeuralField-LDM, a generative model capable of synthesizing complex 3D environments. We leverage Latent Diffusion Models that have been successfully utilized for efficient high-quality 2D content creation. We first train a scene auto-encoder to express a set of image and pose pairs as a neural field, represented as density and feature voxel grids that can be projected to produce novel views of the scene. To further compress this representation, we train a latent-autoencoder that maps the voxel grids to a set of latent representations. A hierarchical diffusion model is then fit to the latents to complete the scene generation pipeline. We achieve a substantial improvement over existing state-of-the-art scene generation models. Additionally, we show how NeuralField-LDM can be used for a variety of 3D content creation applications, including conditional scene generation, scene inpainting and scene style manipulation.

count=1
* DP-NeRF: Deblurred Neural Radiance Field With Physical Scene Priors
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_DP-NeRF_Deblurred_Neural_Radiance_Field_With_Physical_Scene_Priors_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Lee_DP-NeRF_Deblurred_Neural_Radiance_Field_With_Physical_Scene_Priors_CVPR_2023_paper.pdf)]
    * Title: DP-NeRF: Deblurred Neural Radiance Field With Physical Scene Priors
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Dogyoon Lee, Minhyeok Lee, Chajin Shin, Sangyoun Lee
    * Abstract: Neural Radiance Field (NeRF) has exhibited outstanding three-dimensional (3D) reconstruction quality via the novel view synthesis from multi-view images and paired calibrated camera parameters. However, previous NeRF-based systems have been demonstrated under strictly controlled settings, with little attention paid to less ideal scenarios, including with the presence of noise such as exposure, illumination changes, and blur. In particular, though blur frequently occurs in real situations, NeRF that can handle blurred images has received little attention. The few studies that have investigated NeRF for blurred images have not considered geometric and appearance consistency in 3D space, which is one of the most important factors in 3D reconstruction. This leads to inconsistency and the degradation of the perceptual quality of the constructed scene. Hence, this paper proposes a DP-NeRF, a novel clean NeRF framework for blurred images, which is constrained with two physical priors. These priors are derived from the actual blurring process during image acquisition by the camera. DP-NeRF proposes rigid blurring kernel to impose 3D consistency utilizing the physical priors and adaptive weight proposal to refine the color composition error in consideration of the relationship between depth and blur. We present extensive experimental results for synthetic and real scenes with two types of blur: camera motion blur and defocus blur. The results demonstrate that DP-NeRF successfully improves the perceptual quality of the constructed NeRF ensuring 3D geometric and appearance consistency. We further demonstrate the effectiveness of our model with comprehensive ablation analysis.

count=1
* A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction From In-the-Wild Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lei_A_Hierarchical_Representation_Network_for_Accurate_and_Detailed_Face_Reconstruction_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Lei_A_Hierarchical_Representation_Network_for_Accurate_and_Detailed_Face_Reconstruction_CVPR_2023_paper.pdf)]
    * Title: A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction From In-the-Wild Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Biwen Lei, Jianqiang Ren, Mengyang Feng, Miaomiao Cui, Xuansong Xie
    * Abstract: Limited by the nature of the low-dimensional representational capacity of 3DMM, most of the 3DMM-based face reconstruction (FR) methods fail to recover high-frequency facial details, such as wrinkles, dimples, etc. Some attempt to solve the problem by introducing detail maps or non-linear operations, however, the results are still not vivid. To this end, we in this paper present a novel hierarchical representation network (HRN) to achieve accurate and detailed face reconstruction from a single image. Specifically, we implement the geometry disentanglement and introduce the hierarchical representation to fulfill detailed face modeling. Meanwhile, 3D priors of facial details are incorporated to enhance the accuracy and authenticity of the reconstruction results. We also propose a de-retouching module to achieve better decoupling of the geometry and appearance. It is noteworthy that our framework can be extended to a multi-view fashion by considering detail consistency of different views. Extensive experiments on two single-view and two multi-view FR benchmarks demonstrate that our method outperforms the existing methods in both reconstruction accuracy and visual effects. Finally, we introduce a high-quality 3D face dataset FaceHD-100 to boost the research of high-fidelity face reconstruction. The project homepage is at https://younglbw.github.io/HRN-homepage/.

count=1
* DANI-Net: Uncalibrated Photometric Stereo by Differentiable Shadow Handling, Anisotropic Reflectance Modeling, and Neural Inverse Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Li_DANI-Net_Uncalibrated_Photometric_Stereo_by_Differentiable_Shadow_Handling_Anisotropic_Reflectance_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_DANI-Net_Uncalibrated_Photometric_Stereo_by_Differentiable_Shadow_Handling_Anisotropic_Reflectance_CVPR_2023_paper.pdf)]
    * Title: DANI-Net: Uncalibrated Photometric Stereo by Differentiable Shadow Handling, Anisotropic Reflectance Modeling, and Neural Inverse Rendering
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zongrui Li, Qian Zheng, Boxin Shi, Gang Pan, Xudong Jiang
    * Abstract: Uncalibrated photometric stereo (UPS) is challenging due to the inherent ambiguity brought by the unknown light. Although the ambiguity is alleviated on non-Lambertian objects, the problem is still difficult to solve for more general objects with complex shapes introducing irregular shadows and general materials with complex reflectance like anisotropic reflectance. To exploit cues from shadow and reflectance to solve UPS and improve performance on general materials, we propose DANI-Net, an inverse rendering framework with differentiable shadow handling and anisotropic reflectance modeling. Unlike most previous methods that use non-differentiable shadow maps and assume isotropic material, our network benefits from cues of shadow and anisotropic reflectance through two differentiable paths. Experiments on multiple real-world datasets demonstrate our superior and robust performance.

count=1
* Inverse Rendering of Translucent Objects Using Physical and Neural Renderers
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Inverse_Rendering_of_Translucent_Objects_Using_Physical_and_Neural_Renderers_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Inverse_Rendering_of_Translucent_Objects_Using_Physical_and_Neural_Renderers_CVPR_2023_paper.pdf)]
    * Title: Inverse Rendering of Translucent Objects Using Physical and Neural Renderers
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Chenhao Li, Trung Thanh Ngo, Hajime Nagahara
    * Abstract: In this work, we propose an inverse rendering model that estimates 3D shape, spatially-varying reflectance, homogeneous subsurface scattering parameters, and an environment illumination jointly from only a pair of captured images of a translucent object. In order to solve the ambiguity problem of inverse rendering, we use a physically-based renderer and a neural renderer for scene reconstruction and material editing. Because two renderers are differentiable, we can compute a reconstruction loss to assist parameter estimation. To enhance the supervision of the proposed neural renderer, we also propose an augmented loss. In addition, we use a flash and no-flash image pair as the input. To supervise the training, we constructed a large-scale synthetic dataset of translucent objects, which consists of 117K scenes. Qualitative and quantitative results on both synthetic and real-world datasets demonstrated the effectiveness of the proposed model.

count=1
* Multi-View Inverse Rendering for Large-Scale Real-World Indoor Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Multi-View_Inverse_Rendering_for_Large-Scale_Real-World_Indoor_Scenes_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Multi-View_Inverse_Rendering_for_Large-Scale_Real-World_Indoor_Scenes_CVPR_2023_paper.pdf)]
    * Title: Multi-View Inverse Rendering for Large-Scale Real-World Indoor Scenes
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zhen Li, Lingli Wang, Mofang Cheng, Cihui Pan, Jiaqi Yang
    * Abstract: We present a efficient multi-view inverse rendering method for large-scale real-world indoor scenes that reconstructs global illumination and physically-reasonable SVBRDFs. Unlike previous representations, where the global illumination of large scenes is simplified as multiple environment maps, we propose a compact representation called Texture-based Lighting (TBL). It consists of 3D mesh and HDR textures, and efficiently models direct and infinite-bounce indirect lighting of the entire large scene. Based on TBL, we further propose a hybrid lighting representation with precomputed irradiance, which significantly improves the efficiency and alleviates the rendering noise in the material optimization. To physically disentangle the ambiguity between materials, we propose a three-stage material optimization strategy based on the priors of semantic segmentation and room segmentation. Extensive experiments show that the proposed method outperforms the state-of-the-art quantitatively and qualitatively, and enables physically-reasonable mixed-reality applications such as material editing, editable novel view synthesis and relighting. The project page is at https://lzleejean.github.io/TexIR.

count=1
* ShadowNeuS: Neural SDF Reconstruction by Shadow Ray Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ling_ShadowNeuS_Neural_SDF_Reconstruction_by_Shadow_Ray_Supervision_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Ling_ShadowNeuS_Neural_SDF_Reconstruction_by_Shadow_Ray_Supervision_CVPR_2023_paper.pdf)]
    * Title: ShadowNeuS: Neural SDF Reconstruction by Shadow Ray Supervision
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jingwang Ling, Zhibo Wang, Feng Xu
    * Abstract: By supervising camera rays between a scene and multi-view image planes, NeRF reconstructs a neural scene representation for the task of novel view synthesis. On the other hand, shadow rays between the light source and the scene have yet to be considered. Therefore, we propose a novel shadow ray supervision scheme that optimizes both the samples along the ray and the ray location. By supervising shadow rays, we successfully reconstruct a neural SDF of the scene from single-view images under multiple lighting conditions. Given single-view binary shadows, we train a neural network to reconstruct a complete scene not limited by the camera's line of sight. By further modeling the correlation between the image colors and the shadow rays, our technique can also be effectively extended to RGB inputs. We compare our method with previous works on challenging tasks of shape reconstruction from single-view binary shadow or RGB images and observe significant improvements. The code and data are available at https://github.com/gerwang/ShadowNeuS.

count=1
* Robust Model-Based Face Reconstruction Through Weakly-Supervised Outlier Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Robust_Model-Based_Face_Reconstruction_Through_Weakly-Supervised_Outlier_Segmentation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Robust_Model-Based_Face_Reconstruction_Through_Weakly-Supervised_Outlier_Segmentation_CVPR_2023_paper.pdf)]
    * Title: Robust Model-Based Face Reconstruction Through Weakly-Supervised Outlier Segmentation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Chunlu Li, Andreas Morel-Forster, Thomas Vetter, Bernhard Egger, Adam Kortylewski
    * Abstract: In this work, we aim to enhance model-based face reconstruction by avoiding fitting the model to outliers, i.e. regions that cannot be well-expressed by the model such as occluders or make-up. The core challenge for localizing outliers is that they are highly variable and difficult to annotate. To overcome this challenging problem, we introduce a joint Face-autoencoder and outlier segmentation approach (FOCUS).In particular, we exploit the fact that the outliers cannot be fitted well by the face model and hence can be localized well given a high-quality model fitting. The main challenge is that the model fitting and the outlier segmentation are mutually dependent on each other, and need to be inferred jointly. We resolve this chicken-and-egg problem with an EM-type training strategy, where a face autoencoder is trained jointly with an outlier segmentation network. This leads to a synergistic effect, in which the segmentation network prevents the face encoder from fitting to the outliers, enhancing the reconstruction quality. The improved 3D face reconstruction, in turn, enables the segmentation network to better predict the outliers. To resolve the ambiguity between outliers and regions that are difficult to fit, such as eyebrows, we build a statistical prior from synthetic data that measures the systematic bias in model fitting. Experiments on the NoW testset demonstrate that FOCUS achieves SOTA 3D face reconstruction performance among all baselines that are trained without 3D annotation. Moreover, our results on CelebA-HQ and the AR database show that the segmentation network can localize occluders accurately despite being trained without any segmentation annotation.

count=1
* Robust Dynamic Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Robust_Dynamic_Radiance_Fields_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Robust_Dynamic_Radiance_Fields_CVPR_2023_paper.pdf)]
    * Title: Robust Dynamic Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yu-Lun Liu, Chen Gao, Andréas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, Jia-Bin Huang
    * Abstract: Dynamic radiance field reconstruction methods aim to model the time-varying structure and appearance of a dynamic scene. Existing methods, however, assume that accurate camera poses can be reliably estimated by Structure from Motion (SfM) algorithms. These methods, thus, are unreliable as SfM algorithms often fail or produce erroneous poses on challenging videos with highly dynamic objects, poorly textured surfaces, and rotating camera motion. We address this issue by jointly estimating the static and dynamic radiance fields along with the camera parameters (poses and focal length). We demonstrate the robustness of our approach via extensive quantitative and qualitative experiments. Our results show favorable performance over the state-of-the-art dynamic view synthesis methods.

count=1
* Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.pdf)]
    * Title: Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, Andrés Bruhn
    * Abstract: While recent methods for motion and stereo estimation recover an unprecedented amount of details, such highly detailed structures are neither adequately reflected in the data of existing benchmarks nor their evaluation methodology. Hence, we introduce Spring -- a large, high-resolution, high-detail, computer-generated benchmark for scene flow, optical flow, and stereo. Based on rendered scenes from the open-source Blender movie "Spring", it provides photo-realistic HD datasets with state-of-the-art visual effects and ground truth training data. Furthermore, we provide a website to upload, analyze and compare results. Using a novel evaluation methodology based on a super-resolved UHD ground truth, our Spring benchmark can assess the quality of fine structures and provides further detailed performance statistics on different image regions. Regarding the number of ground truth frames, Spring is 60x larger than the only scene flow benchmark, KITTI 2015, and 15x larger than the well-established MPI Sintel optical flow benchmark. Initial results for recent methods on our benchmark show that estimating fine details is indeed challenging, as their accuracy leaves significant room for improvement. The Spring benchmark and the corresponding datasets are available at http://spring-benchmark.org.

count=1
* Progressively Optimized Local Radiance Fields for Robust View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Meuleman_Progressively_Optimized_Local_Radiance_Fields_for_Robust_View_Synthesis_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Meuleman_Progressively_Optimized_Local_Radiance_Fields_for_Robust_View_Synthesis_CVPR_2023_paper.pdf)]
    * Title: Progressively Optimized Local Radiance Fields for Robust View Synthesis
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Andréas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H. Kim, Johannes Kopf
    * Abstract: We present an algorithm for reconstructing the radiance field of a large-scale scene from a single casually captured video. The task poses two core challenges. First, most existing radiance field reconstruction approaches rely on accurate pre-estimated camera poses from Structure-from-Motion algorithms, which frequently fail on in-the-wild videos. Second, using a single, global radiance field with finite representational capacity does not scale to longer trajectories in an unbounded scene. For handling unknown poses, we jointly estimate the camera poses with radiance field in a progressive manner. We show that progressive optimization significantly improves the robustness of the reconstruction. For handling large unbounded scenes, we dynamically allocate new local radiance fields trained with frames within a temporal window. This further improves robustness (e.g., performs well even under moderate pose drifts) and allows us to scale to large scenes. Our extensive evaluation on the Tanks and Temples dataset and our collected outdoor dataset, Static Hikes, show that our approach compares favorably with the state-of-the-art.

count=1
* Looking Through the Glass: Neural Surface Reconstruction Against High Specular Reflections
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Qiu_Looking_Through_the_Glass_Neural_Surface_Reconstruction_Against_High_Specular_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Qiu_Looking_Through_the_Glass_Neural_Surface_Reconstruction_Against_High_Specular_CVPR_2023_paper.pdf)]
    * Title: Looking Through the Glass: Neural Surface Reconstruction Against High Specular Reflections
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jiaxiong Qiu, Peng-Tao Jiang, Yifan Zhu, Ze-Xin Yin, Ming-Ming Cheng, Bo Ren
    * Abstract: Neural implicit methods have achieved high-quality 3D object surfaces under slight specular highlights. However, high specular reflections (HSR) often appear in front of target objects when we capture them through glasses. The complex ambiguity in these scenes violates the multi-view consistency, then makes it challenging for recent methods to reconstruct target objects correctly. To remedy this issue, we present a novel surface reconstruction framework, NeuS-HSR, based on implicit neural rendering. In NeuS-HSR, the object surface is parameterized as an implicit signed distance function (SDF). To reduce the interference of HSR, we propose decomposing the rendered image into two appearances: the target object and the auxiliary plane. We design a novel auxiliary plane module by combining physical assumptions and neural networks to generate the auxiliary plane appearance. Extensive experiments on synthetic and real-world datasets demonstrate that NeuS-HSR outperforms state-of-the-art approaches for accurate and robust target surface reconstruction against HSR.

count=1
* NeRFLight: Fast and Light Neural Radiance Fields Using a Shared Feature Grid
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Rivas-Manzaneque_NeRFLight_Fast_and_Light_Neural_Radiance_Fields_Using_a_Shared_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Rivas-Manzaneque_NeRFLight_Fast_and_Light_Neural_Radiance_Fields_Using_a_Shared_CVPR_2023_paper.pdf)]
    * Title: NeRFLight: Fast and Light Neural Radiance Fields Using a Shared Feature Grid
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Fernando Rivas-Manzaneque, Jorge Sierra-Acosta, Adrian Penate-Sanchez, Francesc Moreno-Noguer, Angela Ribeiro
    * Abstract: While original Neural Radiance Fields (NeRF) have shown impressive results in modeling the appearance of a scene with compact MLP architectures, they are not able to achieve real-time rendering. This has been recently addressed by either baking the outputs of NeRF into a data structure or arranging trainable parameters in an explicit feature grid. These strategies, however, significantly increase the memory footprint of the model which prevents their deployment on bandwidth-constrained applications. In this paper, we extend the grid-based approach to achieve real-time view synthesis at more than 150 FPS using a lightweight model. Our main contribution is a novel architecture in which the density field of NeRF-based representations is split into N regions and the density is modeled using N different decoders which reuse the same feature grid. This results in a smaller grid where each feature is located in more than one spatial position, forcing them to learn a compact representation that is valid for different parts of the scene. We further reduce the size of the final model by disposing of the features symmetrically on each region, which favors feature pruning after training while also allowing smooth gradient transitions between neighboring voxels. An exhaustive evaluation demonstrates that our method achieves real-time performance and quality metrics on a pair with state-of-the-art with an improvement of more than 2x in the FPS/MB ratio.

count=1
* EventNeRF: Neural Radiance Fields From a Single Colour Event Camera
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Rudnev_EventNeRF_Neural_Radiance_Fields_From_a_Single_Colour_Event_Camera_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Rudnev_EventNeRF_Neural_Radiance_Fields_From_a_Single_Colour_Event_Camera_CVPR_2023_paper.pdf)]
    * Title: EventNeRF: Neural Radiance Fields From a Single Colour Event Camera
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, Vladislav Golyanik
    * Abstract: Asynchronously operating event cameras find many applications due to their high dynamic range, vanishingly low motion blur, low latency and low data bandwidth. The field saw remarkable progress during the last few years, and existing event-based 3D reconstruction approaches recover sparse point clouds of the scene. However, such sparsity is a limiting factor in many cases, especially in computer vision and graphics, that has not been addressed satisfactorily so far. Accordingly, this paper proposes the first approach for 3D-consistent, dense and photorealistic novel view synthesis using just a single colour event stream as input. At its core is a neural radiance field trained entirely in a self-supervised manner from events while preserving the original resolution of the colour event channels. Next, our ray sampling strategy is tailored to events and allows for data-efficient training. At test, our method produces results in the RGB space at unprecedented quality. We evaluate our method qualitatively and numerically on several challenging synthetic and real scenes and show that it produces significantly denser and more visually appealing renderings than the existing methods. We also demonstrate robustness in challenging scenarios with fast motion and under low lighting conditions. We release the newly recorded dataset and our source code to facilitate the research field, see https://4dqv.mpi-inf.mpg.de/EventNeRF.

count=1
* Parameter Efficient Local Implicit Image Function Network for Face Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Sarkar_Parameter_Efficient_Local_Implicit_Image_Function_Network_for_Face_Segmentation_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Sarkar_Parameter_Efficient_Local_Implicit_Image_Function_Network_for_Face_Segmentation_CVPR_2023_paper.pdf)]
    * Title: Parameter Efficient Local Implicit Image Function Network for Face Segmentation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Mausoom Sarkar, Nikitha SR, Mayur Hemani, Rishabh Jain, Balaji Krishnamurthy
    * Abstract: Face parsing is defined as the per-pixel labeling of images containing human faces. The labels are defined to identify key facial regions like eyes, lips, nose, hair, etc. In this work, we make use of the structural consistency of the human face to propose a lightweight face-parsing method using a Local Implicit Function network, FP-LIIF. We propose a simple architecture having a convolutional encoder and a pixel MLP decoder that uses 1/26th number of parameters compared to the state-of-the-art models and yet matches or outperforms state-of-the-art models on multiple datasets, like CelebAMask-HQ and LaPa. We do not use any pretraining, and compared to other works, our network can also generate segmentation at different resolutions without any changes in the input resolution. This work enables the use of facial segmentation on low-compute or low-bandwidth devices because of its higher FPS and smaller model size.

count=1
* Panoptic Lifting for 3D Scene Understanding With Neural Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Siddiqui_Panoptic_Lifting_for_3D_Scene_Understanding_With_Neural_Fields_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Siddiqui_Panoptic_Lifting_for_3D_Scene_Understanding_With_Neural_Fields_CVPR_2023_paper.pdf)]
    * Title: Panoptic Lifting for 3D Scene Understanding With Neural Fields
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulò, Norman Müller, Matthias Nießner, Angela Dai, Peter Kontschieder
    * Abstract: We propose Panoptic Lifting, a novel approach for learning panoptic 3D volumetric representations from images of in-the-wild scenes. Once trained, our model can render color images together with 3D-consistent panoptic segmentation from novel viewpoints. Unlike existing approaches which use 3D input directly or indirectly, our method requires only machine-generated 2D panoptic segmentation masks inferred from a pre-trained network. Our core contribution is a panoptic lifting scheme based on a neural field representation that generates a unified and multi-view consistent, 3D panoptic representation of the scene. To account for inconsistencies of 2D instance identifiers across views, we solve a linear assignment with a cost based on the model's current predictions and the machine-generated segmentation masks, thus enabling us to lift 2D instances to 3D in a consistent way. We further propose and ablate contributions that make our method more robust to noisy, machine-generated labels, including test-time augmentations for confidence estimates, segment consistency loss, bounded segmentation fields, and gradient stopping. Experimental results validate our approach on the challenging Hypersim, Replica, and ScanNet datasets, improving by 8.4, 13.8, and 10.6% in scene-level PQ over state of the art.

count=1
* Generating Part-Aware Editable 3D Shapes Without 3D Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tertikas_Generating_Part-Aware_Editable_3D_Shapes_Without_3D_Supervision_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Tertikas_Generating_Part-Aware_Editable_3D_Shapes_Without_3D_Supervision_CVPR_2023_paper.pdf)]
    * Title: Generating Part-Aware Editable 3D Shapes Without 3D Supervision
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Konstantinos Tertikas, Despoina Paschalidou, Boxiao Pan, Jeong Joon Park, Mikaela Angelina Uy, Ioannis Emiris, Yannis Avrithis, Leonidas Guibas
    * Abstract: Impressive progress in generative models and implicit representations gave rise to methods that can generate 3D shapes of high quality. However, being able to locally control and edit shapes is another essential property that can unlock several content creation applications. Local control can be achieved with part-aware models, but existing methods require 3D supervision and cannot produce textures. In this work, we devise PartNeRF, a novel part-aware generative model for editable 3D shape synthesis that does not require any explicit 3D supervision. Our model generates objects as a set of locally defined NeRFs, augmented with an affine transformation. This enables several editing operations such as applying transformations on parts, mixing parts from different objects etc. To ensure distinct, manipulable parts we enforce a hard assignment of rays to parts that makes sure that the color of each ray is only determined by a single NeRF. As a result, altering one part does not affect the appearance of the others. Evaluations on various ShapeNet categories demonstrate the ability of our model to generate editable 3D objects of improved fidelity, compared to previous part-based generative approaches that require 3D supervision or models relying on NeRFs.

count=1
* BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_BAD-NeRF_Bundle_Adjusted_Deblur_Neural_Radiance_Fields_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_BAD-NeRF_Bundle_Adjusted_Deblur_Neural_Radiance_Fields_CVPR_2023_paper.pdf)]
    * Title: BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Peng Wang, Lingzhe Zhao, Ruijie Ma, Peidong Liu
    * Abstract: Neural Radiance Fields (NeRF) have received considerable attention recently, due to its impressive capability in photo-realistic 3D reconstruction and novel view synthesis, given a set of posed camera images. Earlier work usually assumes the input images are of good quality. However, image degradation (e.g. image motion blur in low-light conditions) can easily happen in real-world scenarios, which would further affect the rendering quality of NeRF. In this paper, we present a novel bundle adjusted deblur Neural Radiance Fields (BAD-NeRF), which can be robust to severe motion blurred images and inaccurate camera poses. Our approach models the physical image formation process of a motion blurred image, and jointly learns the parameters of NeRF and recovers the camera motion trajectories during exposure time. In experiments, we show that by directly modeling the real physical image formation process, BAD-NeRF achieves superior performance over prior works on both synthetic and real datasets. Code and data are available at https://github.com/WU-CVGL/BAD-NeRF.

count=1
* FrustumFormer: Adaptive Instance-Aware Resampling for Multi-View 3D Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_FrustumFormer_Adaptive_Instance-Aware_Resampling_for_Multi-View_3D_Detection_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_FrustumFormer_Adaptive_Instance-Aware_Resampling_for_Multi-View_3D_Detection_CVPR_2023_paper.pdf)]
    * Title: FrustumFormer: Adaptive Instance-Aware Resampling for Multi-View 3D Detection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang
    * Abstract: The transformation of features from 2D perspective space to 3D space is essential to multi-view 3D object detection. Recent approaches mainly focus on the design of view transformation, either pixel-wisely lifting perspective view features into 3D space with estimated depth or grid-wisely constructing BEV features via 3D projection, treating all pixels or grids equally. However, choosing what to transform is also important but has rarely been discussed before. The pixels of a moving car are more informative than the pixels of the sky. To fully utilize the information contained in images, the view transformation should be able to adapt to different image regions according to their contents. In this paper, we propose a novel framework named FrustumFormer, which pays more attention to the features in instance regions via adaptive instance-aware resampling. Specifically, the model obtains instance frustums on the bird's eye view by leveraging image view object proposals. An adaptive occupancy mask within the instance frustum is learned to refine the instance location. Moreover, the temporal frustum intersection could further reduce the localization uncertainty of objects. Comprehensive experiments on the nuScenes dataset demonstrate the effectiveness of FrustumFormer, and we achieve a new state-of-the-art performance on the benchmark. Codes and models will be made available at https://github.com/Robertwyq/Frustum.

count=1
* Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Score_Jacobian_Chaining_Lifting_Pretrained_2D_Diffusion_Models_for_3D_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Score_Jacobian_Chaining_Lifting_Pretrained_2D_Diffusion_Models_for_3D_CVPR_2023_paper.pdf)]
    * Title: Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, Greg Shakhnarovich
    * Abstract: A diffusion model learns to predict a vector field of gradients. We propose to apply chain rule on the learned gradients, and back-propagate the score of a diffusion model through the Jacobian of a differentiable renderer, which we instantiate to be a voxel radiance field. This setup aggregates 2D scores at multiple camera viewpoints into a 3D score, and repurposes a pretrained 2D model for 3D data generation. We identify a technical challenge of distribution mismatch that arises in this application, and propose a novel estimation mechanism to resolve it. We run our algorithm on several off-the-shelf diffusion image generative models, including the recently released Stable Diffusion trained on the large-scale LAION dataset.

count=1
* SunStage: Portrait Reconstruction and Relighting Using the Sun as a Light Stage
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_SunStage_Portrait_Reconstruction_and_Relighting_Using_the_Sun_as_a_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_SunStage_Portrait_Reconstruction_and_Relighting_Using_the_Sun_as_a_CVPR_2023_paper.pdf)]
    * Title: SunStage: Portrait Reconstruction and Relighting Using the Sun as a Light Stage
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yifan Wang, Aleksander Holynski, Xiuming Zhang, Xuaner Zhang
    * Abstract: A light stage uses a series of calibrated cameras and lights to capture a subject's facial appearance under varying illumination and viewpoint. This captured information is crucial for facial reconstruction and relighting. Unfortunately, light stages are often inaccessible: they are expensive and require significant technical expertise for construction and operation. In this paper, we present SunStage: a lightweight alternative to a light stage that captures comparable data using only a smartphone camera and the sun. Our method only requires the user to capture a selfie video outdoors, rotating in place, and uses the varying angles between the sun and the face as guidance in joint reconstruction of facial geometry, reflectance, camera pose, and lighting parameters. Despite the in-the-wild un-calibrated setting, our approach is able to reconstruct detailed facial appearance and geometry, enabling compelling effects such as relighting, novel view synthesis, and reflectance editing.

count=1
* BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.pdf)]
    * Title: BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Bowen Wen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas Müller, Alex Evans, Dieter Fox, Jan Kautz, Stan Birchfield
    * Abstract: We present a near real-time (10Hz) method for 6-DoF tracking of an unknown object from a monocular RGBD video sequence, while simultaneously performing neural 3D reconstruction of the object. Our method works for arbitrary rigid objects, even when visual texture is largely absent. The object is assumed to be segmented in the first frame only. No additional information is required, and no assumption is made about the interaction agent. Key to our method is a Neural Object Field that is learned concurrently with a pose graph optimization process in order to robustly accumulate information into a consistent 3D representation capturing both geometry and appearance. A dynamic pool of posed memory frames is automatically maintained to facilitate communication between these threads. Our approach handles challenging sequences with large pose changes, partial and full occlusion, untextured surfaces, and specular highlights. We show results on HO3D, YCBInEOAT, and BEHAVE datasets, demonstrating that our method significantly outperforms existing approaches. Project page: https://bundlesdf.github.io/

count=1
* Behind the Scenes: Density Fields for Single View Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wimbauer_Behind_the_Scenes_Density_Fields_for_Single_View_Reconstruction_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wimbauer_Behind_the_Scenes_Density_Fields_for_Single_View_Reconstruction_CVPR_2023_paper.pdf)]
    * Title: Behind the Scenes: Density Fields for Single View Reconstruction
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Felix Wimbauer, Nan Yang, Christian Rupprecht, Daniel Cremers
    * Abstract: Inferring a meaningful geometric scene representation from a single image is a fundamental problem in computer vision. Approaches based on traditional depth map prediction can only reason about areas that are visible in the image. Currently, neural radiance fields (NeRFs) can capture true 3D including color, but are too complex to be generated from a single image. As an alternative, we propose to predict an implicit density field from a single image. It maps every location in the frustum of the image to volumetric density. By directly sampling color from the available views instead of storing color in the density field, our scene representation becomes significantly less complex compared to NeRFs, and a neural network can predict it in a single forward pass. The network is trained through self-supervision from only video data. Our formulation allows volume rendering to perform both depth prediction and novel view synthesis. Through experiments, we show that our method is able to predict meaningful geometry for regions that are occluded in the input image. Additionally, we demonstrate the potential of our approach on three datasets for depth prediction and novel-view synthesis.

count=1
* High-Fidelity 3D Face Generation From Natural Language Descriptions
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_High-Fidelity_3D_Face_Generation_From_Natural_Language_Descriptions_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_High-Fidelity_3D_Face_Generation_From_Natural_Language_Descriptions_CVPR_2023_paper.pdf)]
    * Title: High-Fidelity 3D Face Generation From Natural Language Descriptions
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Menghua Wu, Hao Zhu, Linjia Huang, Yiyu Zhuang, Yuanxun Lu, Xun Cao
    * Abstract: Synthesizing high-quality 3D face models from natural language descriptions is very valuable for many applications, including avatar creation, virtual reality, and telepresence. However, little research ever tapped into this task. We argue the major obstacle lies in 1) the lack of high-quality 3D face data with descriptive text annotation, and 2) the complex mapping relationship between descriptive language space and shape/appearance space. To solve these problems, we build DESCRIBE3D dataset, the first large-scale dataset with fine-grained text descriptions for text-to-3D face generation task. Then we propose a two-stage framework to first generate a 3D face that matches the concrete descriptions, then optimize the parameters in the 3D shape and texture space with abstract description to refine the 3D face model. Extensive experimental results show that our method can produce a faithful 3D face that conforms to the input descriptions with higher accuracy and quality than previous methods. The code and DESCRIBE3D dataset are released at https://github.com/zhuhao-nju/describe3d.

count=1
* Multiview Compressive Coding for 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Multiview_Compressive_Coding_for_3D_Reconstruction_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Multiview_Compressive_Coding_for_3D_Reconstruction_CVPR_2023_paper.pdf)]
    * Title: Multiview Compressive Coding for 3D Reconstruction
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Chao-Yuan Wu, Justin Johnson, Jitendra Malik, Christoph Feichtenhofer, Georgia Gkioxari
    * Abstract: A central goal of visual recognition is to understand objects and scenes from a single image. 2D recognition has witnessed tremendous progress thanks to large-scale learning and general-purpose representations. But, 3D poses new challenges stemming from occlusions not depicted in the image. Prior works try to overcome these by inferring from multiple views or rely on scarce CAD models and category-specific priors which hinder scaling to novel settings. In this work, we explore single-view 3D reconstruction by learning generalizable representations inspired by advances in self-supervised learning. We introduce a simple framework that operates on 3D points of single objects or whole scenes coupled with category-agnostic large-scale training from diverse RGB-D videos. Our model, Multiview Compressive Coding (MCC), learns to compress the input appearance and geometry to predict the 3D structure by querying a 3D-aware decoder. MCC's generality and efficiency allow it to learn from large-scale and diverse data sources with strong generalization to novel objects imagined by DALL*E 2 or captured in-the-wild with an iPhone.

count=1
* Level-S$^2$fM: Structure From Motion on Neural Level Set of Implicit Surfaces
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Xiao_Level-S2fM_Structure_From_Motion_on_Neural_Level_Set_of_Implicit_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_Level-S2fM_Structure_From_Motion_on_Neural_Level_Set_of_Implicit_CVPR_2023_paper.pdf)]
    * Title: Level-S$^2$fM: Structure From Motion on Neural Level Set of Implicit Surfaces
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yuxi Xiao, Nan Xue, Tianfu Wu, Gui-Song Xia
    * Abstract: This paper presents a neural incremental Structure-from-Motion (SfM) approach, Level-S2fM, which estimates the camera poses and scene geometry from a set of uncalibrated images by learning coordinate MLPs for the implicit surfaces and the radiance fields from the established keypoint correspondences. Our novel formulation poses some new challenges due to inevitable two-view and few-view configurations in the incremental SfM pipeline, which complicates the optimization of coordinate MLPs for volumetric neural rendering with unknown camera poses. Nevertheless, we demonstrate that the strong inductive basis conveying in the 2D correspondences is promising to tackle those challenges by exploiting the relationship between the ray sampling schemes. Based on this, we revisit the pipeline of incremental SfM and renew the key components, including two-view geometry initialization, the camera poses registration, the 3D points triangulation, and Bundle Adjustment, with a fresh perspective based on neural implicit surfaces. By unifying the scene geometry in small MLP networks through coordinate MLPs, our Level-S2fM treats the zero-level set of the implicit surface as an informative top-down regularization to manage the reconstructed 3D points, reject the outliers in correspondences via querying SDF, and refine the estimated geometries by NBA (Neural BA). Not only does our Level-S2fM lead to promising results on camera pose estimation and scene geometry reconstruction, but it also shows a promising way for neural implicit rendering without knowing camera extrinsic beforehand.

count=1
* NeRFVS: Neural Radiance Fields for Free View Synthesis via Geometry Scaffolds
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_NeRFVS_Neural_Radiance_Fields_for_Free_View_Synthesis_via_Geometry_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_NeRFVS_Neural_Radiance_Fields_for_Free_View_Synthesis_via_Geometry_CVPR_2023_paper.pdf)]
    * Title: NeRFVS: Neural Radiance Fields for Free View Synthesis via Geometry Scaffolds
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Chen Yang, Peihao Li, Zanwei Zhou, Shanxin Yuan, Bingbing Liu, Xiaokang Yang, Weichao Qiu, Wei Shen
    * Abstract: We present NeRFVS, a novel neural radiance fields (NeRF) based method to enable free navigation in a room. NeRF achieves impressive performance in rendering images for novel views similar to the input views while suffering for novel views that are significantly different from the training views. To address this issue, we utilize the holistic priors, including pseudo depth maps and view coverage information, from neural reconstruction to guide the learning of implicit neural representations of 3D indoor scenes. Concretely, an off-the-shelf neural reconstruction method is leveraged to generate a geometry scaffold. Then, two loss functions based on the holistic priors are proposed to improve the learning of NeRF: 1) A robust depth loss that can tolerate the error of the pseudo depth map to guide the geometry learning of NeRF; 2) A variance loss to regularize the variance of implicit neural representations to reduce the geometry and color ambiguity in the learning procedure. These two loss functions are modulated during NeRF optimization according to the view coverage information to reduce the negative influence brought by the view coverage imbalance. Extensive results demonstrate that our NeRFVS outperforms state-of-the-art view synthesis methods quantitatively and qualitatively on indoor scenes, achieving high-fidelity free navigation results.

count=1
* Self-Supervised Super-Plane for Neural 3D Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ye_Self-Supervised_Super-Plane_for_Neural_3D_Reconstruction_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_Self-Supervised_Super-Plane_for_Neural_3D_Reconstruction_CVPR_2023_paper.pdf)]
    * Title: Self-Supervised Super-Plane for Neural 3D Reconstruction
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Botao Ye, Sifei Liu, Xueting Li, Ming-Hsuan Yang
    * Abstract: Neural implicit surface representation methods show impressive reconstruction results but struggle to handle texture-less planar regions that widely exist in indoor scenes. Existing approaches addressing this leverage image prior that requires assistive networks trained with large-scale annotated datasets. In this work, we introduce a self-supervised super-plane constraint by exploring the free geometry cues from the predicted surface, which can further regularize the reconstruction of plane regions without any other ground truth annotations. Specifically, we introduce an iterative training scheme, where (i) grouping of pixels to formulate a super-plane (analogous to super-pixels), and (ii) optimizing of the scene reconstruction network via a super-plane constraint, are progressively conducted. We demonstrate that the model trained with super-planes surprisingly outperforms the one using conventional annotated planes, as individual super-plane statistically occupies a larger area and leads to more stable training. Extensive experiments show that our self-supervised super-plane constraint significantly improves 3D reconstruction quality even better than using ground truth plane segmentation. Additionally, the plane reconstruction results from our model can be used for auto-labeling for other vision tasks. The code and models are available at https: //github.com/botaoye/S3PRecon.

count=1
* Accidental Light Probes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Accidental_Light_Probes_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Accidental_Light_Probes_CVPR_2023_paper.pdf)]
    * Title: Accidental Light Probes
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Hong-Xing Yu, Samir Agarwala, Charles Herrmann, Richard Szeliski, Noah Snavely, Jiajun Wu, Deqing Sun
    * Abstract: Recovering lighting in a scene from a single image is a fundamental problem in computer vision. While a mirror ball light probe can capture omnidirectional lighting, light probes are generally unavailable in everyday images. In this work, we study recovering lighting from accidental light probes (ALPs)---common, shiny objects like Coke cans, which often accidentally appear in daily scenes. We propose a physically-based approach to model ALPs and estimate lighting from their appearances in single images. The main idea is to model the appearance of ALPs by photogrammetrically principled shading and to invert this process via differentiable rendering to recover incidental illumination. We demonstrate that we can put an ALP into a scene to allow high-fidelity lighting estimation. Our model can also recover lighting for existing images that happen to contain an ALP.

count=1
* Ref-NPR: Reference-Based Non-Photorealistic Radiance Fields for Controllable Scene Stylization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Ref-NPR_Reference-Based_Non-Photorealistic_Radiance_Fields_for_Controllable_Scene_Stylization_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Ref-NPR_Reference-Based_Non-Photorealistic_Radiance_Fields_for_Controllable_Scene_Stylization_CVPR_2023_paper.pdf)]
    * Title: Ref-NPR: Reference-Based Non-Photorealistic Radiance Fields for Controllable Scene Stylization
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yuechen Zhang, Zexin He, Jinbo Xing, Xufeng Yao, Jiaya Jia
    * Abstract: Current 3D scene stylization methods transfer textures and colors as styles using arbitrary style references, lacking meaningful semantic correspondences. We introduce Reference-Based Non-Photorealistic Radiance Fields (Ref-NPR) to address this limitation. This controllable method stylizes a 3D scene using radiance fields with a single stylized 2D view as a reference. We propose a ray registration process based on the stylized reference view to obtain pseudo-ray supervision in novel views. Then we exploit semantic correspondences in content images to fill occluded regions with perceptually similar styles, resulting in non-photorealistic and continuous novel view sequences. Our experimental results demonstrate that Ref-NPR outperforms existing scene and video stylization methods regarding visual quality and semantic correspondence. The code and data are publicly available on the project page at https://ref-npr.github.io.

count=1
* Multi-View Reconstruction Using Signed Ray Distance Functions (SRDF)
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zins_Multi-View_Reconstruction_Using_Signed_Ray_Distance_Functions_SRDF_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zins_Multi-View_Reconstruction_Using_Signed_Ray_Distance_Functions_SRDF_CVPR_2023_paper.pdf)]
    * Title: Multi-View Reconstruction Using Signed Ray Distance Functions (SRDF)
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Pierre Zins, Yuanlu Xu, Edmond Boyer, Stefanie Wuhrer, Tony Tung
    * Abstract: In this paper, we investigate a new optimization framework for multi-view 3D shape reconstructions. Recent differentiable rendering approaches have provided breakthrough performances with implicit shape representations though they can still lack precision in the estimated geometries. On the other hand multi-view stereo methods can yield pixel wise geometric accuracy with local depth predictions along viewing rays. Our approach bridges the gap between the two strategies with a novel volumetric shape representation that is implicit but parameterized with pixel depths to better materialize the shape surface with consistent signed distances along viewing rays. The approach retains pixel-accuracy while benefiting from volumetric integration in the optimization. To this aim, depths are optimized by evaluating, at each 3D location within the volumetric discretization, the agreement between the depth prediction consistency and the photometric consistency for the corresponding pixels. The optimization is agnostic to the associated photo-consistency term which can vary from a median-based baseline to more elaborate criteria, learned functions. Our experiments demonstrate the benefit of the volumetric integration with depth predictions. They also show that our approach outperforms existing approaches over standard 3D benchmarks with better geometry estimations.

count=1
* Efficient Multi-Lens Bokeh Effect Rendering and Transformation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Seizinger_Efficient_Multi-Lens_Bokeh_Effect_Rendering_and_Transformation_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Seizinger_Efficient_Multi-Lens_Bokeh_Effect_Rendering_and_Transformation_CVPRW_2023_paper.pdf)]
    * Title: Efficient Multi-Lens Bokeh Effect Rendering and Transformation
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Tim Seizinger, Marcos V. Conde, Manuel Kolmet, Tom E. Bishop, Radu Timofte
    * Abstract: Many advancements of mobile cameras aim to reach the visual quality of professional DSLR cameras. Great progress was shown over the last years in optimizing the sharp regions of an image and in creating virtual portrait effects with artificially blurred backgrounds. Bokeh is the aesthetic quality of the blur in out-of-focus areas of an image. This is a popular technique among professional photographers, and for this reason, a new goal in computational photography is to optimize the Bokeh effect itself. This paper introduces EBokehNet, a efficient state-of-the-art solution for Bokeh effect transformation and rendering. Our method can render Bokeh from an all-in-focus image, or transform the Bokeh of one lens to the effect of another lens without harming the sharp foreground regions in the image. Moreover we can control the shape and strength of the effect by feeding the lens properties i.e. type (Sony or Canon) and aperture, into the neural network as an additional input. Our method is a winning solution at the NTIRE 2023 Lens-to-Lens Bokeh Effect Transformation Challenge, and state-of-the-art at the EBB benchmark.

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
* UnO: Unsupervised Occupancy Fields for Perception and Forecasting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.pdf)]
    * Title: UnO: Unsupervised Occupancy Fields for Perception and Forecasting
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ben Agro, Quinlan Sykora, Sergio Casas, Thomas Gilles, Raquel Urtasun
    * Abstract: Perceiving the world and forecasting its future state is a critical task for self-driving. Supervised approaches leverage annotated object labels to learn a model of the world --- traditionally with object detections and trajectory predictions or temporal bird's-eye-view (BEV) occupancy fields. However these annotations are expensive and typically limited to a set of predefined categories that do not cover everything we might encounter on the road. Instead we learn to perceive and forecast a continuous 4D (spatio-temporal) occupancy field with self-supervision from LiDAR data. This unsupervised world model can be easily and effectively transferred to downstream tasks. We tackle point cloud forecasting by adding a lightweight learned renderer and achieve state-of-the-art performance in Argoverse 2 nuScenes and KITTI. To further showcase its transferability we fine-tune our model for BEV semantic occupancy forecasting and show that it outperforms the fully supervised state-of-the-art especially when labeled data is scarce. Finally when compared to prior state-of-the-art on spatio-temporal geometric occupancy prediction our 4D world model achieves a much higher recall of objects from classes relevant to self-driving.

count=1
* SceneFun3D: Fine-Grained Functionality and Affordance Understanding in 3D Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Delitzas_SceneFun3D_Fine-Grained_Functionality_and_Affordance_Understanding_in_3D_Scenes_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Delitzas_SceneFun3D_Fine-Grained_Functionality_and_Affordance_Understanding_in_3D_Scenes_CVPR_2024_paper.pdf)]
    * Title: SceneFun3D: Fine-Grained Functionality and Affordance Understanding in 3D Scenes
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Alexandros Delitzas, Ayca Takmaz, Federico Tombari, Robert Sumner, Marc Pollefeys, Francis Engelmann
    * Abstract: Existing 3D scene understanding methods are heavily focused on 3D semantic and instance segmentation. However identifying objects and their parts only constitutes an intermediate step towards a more fine-grained goal which is effectively interacting with the functional interactive elements (e.g. handles knobs buttons) in the scene to accomplish diverse tasks. To this end we introduce SceneFun3D a large-scale dataset with more than 14.8k highly accurate interaction annotations for 710 high-resolution real-world 3D indoor scenes. We accompany the annotations with motion parameter information describing how to interact with these elements and a diverse set of natural language descriptions of tasks that involve manipulating them in the scene context. To showcase the value of our dataset we introduce three novel tasks namely functionality segmentation task-driven affordance grounding and 3D motion estimation and adapt existing state-of-the-art methods to tackle them. Our experiments show that solving these tasks in real 3D scenes remains challenging despite recent progress in closed-set and open-set 3D scene understanding methods.

count=1
* Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Deng_Portrait4D_Learning_One-Shot_4D_Head_Avatar_Synthesis_using_Synthetic_Data_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_Portrait4D_Learning_One-Shot_4D_Head_Avatar_Synthesis_using_Synthetic_Data_CVPR_2024_paper.pdf)]
    * Title: Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yu Deng, Duomin Wang, Xiaohang Ren, Xingyu Chen, Baoyuan Wang
    * Abstract: Existing one-shot 4D head synthesis methods usually learn from monocular videos with the aid of 3DMM reconstruction yet the latter is evenly challenging which restricts them from reasonable 4D head synthesis. We present a method to learn one-shot 4D head synthesis via large-scale synthetic data. The key is to first learn a part-wise 4D generative model from monocular images via adversarial learning to synthesize multi-view images of diverse identities and full motions as training data; then leverage a transformer-based animatable triplane reconstructor to learn 4D head reconstruction using the synthetic data. A novel learning strategy is enforced to enhance the generalizability to real images by disentangling the learning process of 3D reconstruction and reenactment. Experiments demonstrate our superiority over the prior art.

count=1
* NARUTO: Neural Active Reconstruction from Uncertain Target Observations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Feng_NARUTO_Neural_Active_Reconstruction_from_Uncertain_Target_Observations_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Feng_NARUTO_Neural_Active_Reconstruction_from_Uncertain_Target_Observations_CVPR_2024_paper.pdf)]
    * Title: NARUTO: Neural Active Reconstruction from Uncertain Target Observations
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ziyue Feng, Huangying Zhan, Zheng Chen, Qingan Yan, Xiangyu Xu, Changjiang Cai, Bing Li, Qilun Zhu, Yi Xu
    * Abstract: We present NARUTO a neural active reconstruction system that combines a hybrid neural representation with uncertainty learning enabling high-fidelity surface reconstruction. Our approach leverages a multi-resolution hash-grid as the mapping backbone chosen for its exceptional convergence speed and capacity to capture high-frequency local features. The centerpiece of our work is the incorporation of an uncertainty learning module that dynamically quantifies reconstruction uncertainty while actively reconstructing the environment. By harnessing learned uncertainty we propose a novel uncertainty aggregation strategy for goal searching and efficient path planning. Our system autonomously explores by targeting uncertain observations and reconstructs environments with remarkable completeness and fidelity. We also demonstrate the utility of this uncertainty-aware approach by enhancing SOTA neural SLAM systems through an active ray sampling strategy. Extensive evaluations of NARUTO in various environments using an indoor scene simulator confirm its superior performance and state-of-the-art status in active reconstruction as evidenced by its impressive results on benchmark datasets like Replica and MP3D.

count=1
* QUADify: Extracting Meshes with Pixel-level Details and Materials from Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Fruhauf_QUADify_Extracting_Meshes_with_Pixel-level_Details_and_Materials_from_Images_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Fruhauf_QUADify_Extracting_Meshes_with_Pixel-level_Details_and_Materials_from_Images_CVPR_2024_paper.pdf)]
    * Title: QUADify: Extracting Meshes with Pixel-level Details and Materials from Images
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Maximilian Frühauf, Hayko Riemenschneider, Markus Gross, Christopher Schroers
    * Abstract: Despite exciting progress in automatic 3D reconstruction from images excessive and irregular triangular faces in the resulting meshes still constitute a significant challenge when it comes to adoption in practical artist workflows. Therefore we propose a method to extract regular quad-dominant meshes from posed images. More specifically we generate a high-quality 3D model through decomposition into an easily editable quad-dominant mesh with pixel-level details such as displacement materials and lighting. To enable end-to-end learning of shape and quad topology we QUADify a neural implicit representation using our novel differentiable re-meshing objective. Distinct from previous work our method exploits artifact-free Catmull-Clark subdivision combined with vertex displacement to extract pixel-level details linked to the base geometry. Finally we apply differentiable rendering techniques for material and lighting decomposition to optimize for image reconstruction. Our experiments show the benefits of end-to-end re-meshing and that our method yields state-of-the-art geometric accuracy while providing lightweight meshes with displacements and textures that are directly compatible with professional renderers and game engines.

count=1
* COLMAP-Free 3D Gaussian Splatting
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Fu_COLMAP-Free_3D_Gaussian_Splatting_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Fu_COLMAP-Free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf)]
    * Title: COLMAP-Free 3D Gaussian Splatting
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A. Efros, Xiaolong Wang
    * Abstract: While neural rendering has led to impressive advances in scene reconstruction and novel view synthesis it relies heavily on accurately pre-computed camera poses. To relax this constraint multiple efforts have been made to train Neural Radiance Fields (NeRFs) without pre-processed camera poses. However the implicit representations of NeRFs provide extra challenges to optimize the 3D structure and camera poses at the same time. On the other hand the recently proposed 3D Gaussian Splatting provides new opportunities given its explicit point cloud representations. This paper leverages both the explicit geometric representation and the continuity of the input video stream to perform novel view synthesis without any SfM preprocessing. We process the input frames in a sequential manner and progressively grow the 3D Gaussians set by taking one input frame at a time without the need to pre-compute the camera poses. Our method significantly improves over previous approaches in view synthesis and camera pose estimation under large motion changes. Our project page is: https: //oasisyang.github.io/colmap-free-3dgs.

count=1
* Multiplane Prior Guided Few-Shot Aerial Scene Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Multiplane_Prior_Guided_Few-Shot_Aerial_Scene_Rendering_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_Multiplane_Prior_Guided_Few-Shot_Aerial_Scene_Rendering_CVPR_2024_paper.pdf)]
    * Title: Multiplane Prior Guided Few-Shot Aerial Scene Rendering
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zihan Gao, Licheng Jiao, Lingling Li, Xu Liu, Fang Liu, Puhua Chen, Yuwei Guo
    * Abstract: Neural Radiance Fields (NeRF) have been successfully applied in various aerial scenes yet they face challenges with sparse views due to limited supervision. The acquisition of dense aerial views is often prohibitive as unmanned aerial vehicles (UAVs) may encounter constraints in perspective range and energy constraints. In this work we introduce Multiplane Prior guided NeRF (MPNeRF) a novel approach tailored for few-shot aerial scene rendering--marking a pioneering effort in this domain. Our key insight is that the intrinsic geometric regularities specific to aerial imagery could be leveraged to enhance NeRF in sparse aerial scenes. By investigating NeRF's and Multiplane Image (MPI)'s behavior we propose to guide the training process of NeRF with a Multiplane Prior. The proposed Multiplane Prior draws upon MPI's benefits and incorporates advanced image comprehension through a SwinV2 Transformer pre-trained via SimMIM. Our extensive experiments demonstrate that MPNeRF outperforms existing state-of-the-art methods applied in non-aerial contexts by tripling the performance in SSIM and LPIPS even with three views available. We hope our work offers insights into the development of NeRF-based applications in aerial scenes with limited data.

count=1
* Intraoperative 2D/3D Image Registration via Differentiable X-ray Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Gopalakrishnan_Intraoperative_2D3D_Image_Registration_via_Differentiable_X-ray_Rendering_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Gopalakrishnan_Intraoperative_2D3D_Image_Registration_via_Differentiable_X-ray_Rendering_CVPR_2024_paper.pdf)]
    * Title: Intraoperative 2D/3D Image Registration via Differentiable X-ray Rendering
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Vivek Gopalakrishnan, Neel Dey, Polina Golland
    * Abstract: Surgical decisions are informed by aligning rapid portable 2D intraoperative images (e.g. X-rays) to a high-fidelity 3D preoperative reference scan (e.g. CT). However 2D/3D registration can often fail in practice: conventional optimization methods are prohibitively slow and susceptible to local minima while neural networks trained on small datasets fail on new patients or require impractical landmark supervision. We present DiffPose a self-supervised approach that leverages patient-specific simulation and differentiable physics-based rendering to achieve accurate 2D/3D registration without relying on manually labeled data. Preoperatively a CNN is trained to regress the pose of a randomly oriented synthetic X-ray rendered from the preoperative CT. The CNN then initializes rapid intraoperative test-time optimization that uses the differentiable X-ray renderer to refine the solution. Our work further proposes several geometrically principled methods for sampling camera poses from SE(3) for sparse differentiable rendering and for driving registration in the tangent space se(3) with geodesic and multiscale locality-sensitive losses. DiffPose achieves sub-millimeter accuracy across surgical datasets at intraoperative speeds improving upon existing unsupervised methods by an order of magnitude and even outperforming supervised baselines. Our implementation is at https://github.com/eigenvivek/DiffPose.

count=1
* DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Gu_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Gu_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_CVPR_2024_paper.pdf)]
    * Title: DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yuming Gu, Hongyi Xu, You Xie, Guoxian Song, Yichun Shi, Di Chang, Jing Yang, Linjie Luo
    * Abstract: We present DiffPortrait3D a conditional diffusion model that is capable of synthesizing 3D-consistent photo-realistic novel views from as few as a single in-the-wild portrait. Specifically given a single RGB input we aim to synthesize plausible but consistent facial details rendered from novel camera views with retained both identity and facial expression. In lieu of time-consuming optimization and fine-tuning our zero-shot method generalizes well to arbitrary face portraits with unposed camera views extreme facial expressions and diverse artistic depictions. At its core we leverage the generative prior of 2D diffusion models pre-trained on large-scale image datasets as our rendering backbone while the denoising is guided with disentangled attentive control of appearance and camera pose. To achieve this we first inject the appearance context from the reference image into the self-attention layers of the frozen UNets. The rendering view is then manipulated with a novel conditional control module that interprets the camera pose by watching a condition image of a crossed subject from the same view. Furthermore we insert a trainable cross-view attention module to enhance view consistency which is further strengthened with a novel 3D-aware noise generation process during inference. We demonstrate state-of-the-art results both qualitatively and quantitatively on our challenging in-the-wild and multi-view benchmarks.

count=1
* SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.pdf)]
    * Title: SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Antoine Guédon, Vincent Lepetit
    * Abstract: We propose a method to allow precise and extremely fast mesh extraction from 3D Gaussian Splatting. Gaussian Splatting has recently become very popular as it yields realistic rendering while being significantly faster to train than NeRFs. It is however challenging to extract a mesh from the millions of tiny 3D Gaussians as these Gaussians tend to be unorganized after optimization and no method has been proposed so far. Our first key contribution is a regularization term that encourages the Gaussians to align well with the surface of the scene. We then introduce a method that exploits this alignment to extract a mesh from the Gaussians using Poisson reconstruction which is fast scalable and preserves details in contrast to the Marching Cubes algorithm usually applied to extract meshes from Neural SDFs. Finally we introduce an optional refinement strategy that binds Gaussians to the surface of the mesh and jointly optimizes these Gaussians and the mesh through Gaussian splatting rendering. This enables easy editing sculpting animating and relighting of the Gaussians by manipulating the mesh instead of the Gaussians themselves. Retrieving such an editable mesh for realistic rendering is done within minutes with our method compared to hours with the state-of-the-art method on SDFs while providing a better rendering quality.

count=1
* High-Quality Facial Geometry and Appearance Capture at Home
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Han_High-Quality_Facial_Geometry_and_Appearance_Capture_at_Home_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_High-Quality_Facial_Geometry_and_Appearance_Capture_at_Home_CVPR_2024_paper.pdf)]
    * Title: High-Quality Facial Geometry and Appearance Capture at Home
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yuxuan Han, Junfeng Lyu, Feng Xu
    * Abstract: Facial geometry and appearance capture have demonstrated tremendous success in 3D scanning real humans in studios. Recent works propose to democratize this technique while keeping the results high quality. However they are still inconvenient for daily usage. In addition they focus on an easier problem of only capturing facial skin. This paper proposes a novel method for high-quality face capture featuring an easy-to-use system and the capability to model the complete face with skin mouth interior hair and eyes. We reconstruct facial geometry and appearance from a single co-located smartphone flashlight sequence captured in a dim room where the flashlight is the dominant light source (e.g. rooms with curtains or at night). To model the complete face we propose a novel hybrid representation to effectively model both eyes and other facial regions along with novel techniques to learn it from images. We apply a combined lighting model to compactly represent real illuminations and exploit a morphable face albedo model as a reflectance prior to disentangle diffuse and specular. Experiments show that our method can capture high-quality 3D relightable scans. Our code will be released.

count=1
* GauHuman: Articulated Gaussian Splatting from Monocular Human Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_GauHuman_Articulated_Gaussian_Splatting_from_Monocular_Human_Videos_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_GauHuman_Articulated_Gaussian_Splatting_from_Monocular_Human_Videos_CVPR_2024_paper.pdf)]
    * Title: GauHuman: Articulated Gaussian Splatting from Monocular Human Videos
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Shoukang Hu, Tao Hu, Ziwei Liu
    * Abstract: We present GauHuman a 3D human model with Gaussian Splatting for both fast training (1 2 minutes) and real-time rendering (up to 189 FPS) compared with existing NeRF-based implicit representation modelling frameworks demanding hours of training and seconds of rendering per frame. Specifically GauHuman encodes Gaussian Splatting in the canonical space and transforms 3D Gaussians from canonical space to posed space with linear blend skinning (LBS) in which effective pose and LBS refinement modules are designed to learn fine details of 3D humans under negligible computational cost. Moreover to enable fast optimization of GauHuman we initialize and prune 3D Gaussians with 3D human prior while splitting/cloning via KL divergence guidance along with a novel merge operation for further speeding up. Extensive experiments on ZJU_Mocap and MonoCap datasets demonstrate that GauHuman achieves state-of-the-art performance quantitatively and qualitatively with fast training and real-time rendering speed. Notably without sacrificing rendering quality GauHuman can fast model the 3D human performer with 13k 3D Gaussians. Our code is available at https://github.com/skhu101/GauHuman.

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
* WateRF: Robust Watermarks in Radiance Fields for Protection of Copyrights
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Jang_WateRF_Robust_Watermarks_in_Radiance_Fields_for_Protection_of_Copyrights_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Jang_WateRF_Robust_Watermarks_in_Radiance_Fields_for_Protection_of_Copyrights_CVPR_2024_paper.pdf)]
    * Title: WateRF: Robust Watermarks in Radiance Fields for Protection of Copyrights
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Youngdong Jang, Dong In Lee, MinHyuk Jang, Jong Wook Kim, Feng Yang, Sangpil Kim
    * Abstract: The advances in the Neural Radiance Fields (NeRF) research offer extensive applications in diverse domains but protecting their copyrights has not yet been researched in depth. Recently NeRF watermarking has been considered one of the pivotal solutions for safely deploying NeRF-based 3D representations. However existing methods are designed to apply only to implicit or explicit NeRF representations. In this work we introduce an innovative watermarking method that can be employed in both representations of NeRF. This is achieved by fine-tuning NeRF to embed binary messages in the rendering process. In detail we propose utilizing the discrete wavelet transform in the NeRF space for watermarking. Furthermore we adopt a deferred back-propagation technique and introduce a combination with the patch-wise loss to improve rendering quality and bit accuracy with minimum trade-offs. We evaluate our method in three different aspects: capacity invisibility and robustness of the embedded watermarks in the 2D-rendered images. Our method achieves state-of-the-art performance with faster training speed over the compared state-of-the-art methods. Project page: https://kuai-lab.github.io/cvpr2024waterf/

count=1
* Diffeomorphic Template Registration for Atmospheric Turbulence Mitigation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lao_Diffeomorphic_Template_Registration_for_Atmospheric_Turbulence_Mitigation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lao_Diffeomorphic_Template_Registration_for_Atmospheric_Turbulence_Mitigation_CVPR_2024_paper.pdf)]
    * Title: Diffeomorphic Template Registration for Atmospheric Turbulence Mitigation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Dong Lao, Congli Wang, Alex Wong, Stefano Soatto
    * Abstract: We describe a method for recovering the irradiance underlying a collection of images corrupted by atmospheric turbulence. Since supervised data is often technically impossible to obtain assumptions and biases have to be imposed to solve this inverse problem and we choose to model them explicitly. Rather than initializing a latent irradiance ("template") by heuristics to estimate deformation we select one of the images as a reference and model the deformation in this image by the aggregation of the optical flow from it to other images exploiting a prior imposed by Central Limit Theorem. Then with a novel flow inversion module the model registers each image TO the template but WITHOUT the template avoiding artifacts related to poor template initialization. To illustrate the robustness of the method we simply (i) select the first frame as the reference and (ii) use the simplest optical flow to estimate the warpings yet the improvement in registration is decisive in the final reconstruction as we achieve state-of-the-art performance despite its simplicity. The method establishes a strong baseline that can be further improved by integrating it seamlessly into more sophisticated pipelines or with domain-specific methods if so desired.

count=1
* NeISF: Neural Incident Stokes Field for Geometry and Material Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_NeISF_Neural_Incident_Stokes_Field_for_Geometry_and_Material_Estimation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_NeISF_Neural_Incident_Stokes_Field_for_Geometry_and_Material_Estimation_CVPR_2024_paper.pdf)]
    * Title: NeISF: Neural Incident Stokes Field for Geometry and Material Estimation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chenhao Li, Taishi Ono, Takeshi Uemori, Hajime Mihara, Alexander Gatto, Hajime Nagahara, Yusuke Moriuchi
    * Abstract: Multi-view inverse rendering is the problem of estimating the scene parameters such as shapes materials or illuminations from a sequence of images captured under different viewpoints. Many approaches however assume single light bounce and thus fail to recover challenging scenarios like inter-reflections. On the other hand simply extending those methods to consider multi-bounced light requires more assumptions to alleviate the ambiguity. To address this problem we propose Neural Incident Stokes Fields (NeISF) a multi-view inverse rendering framework that reduces ambiguities using polarization cues. The primary motivation for using polarization cues is that it is the accumulation of multi-bounced light providing rich information about geometry and material. Based on this knowledge the proposed incident Stokes field efficiently models the accumulated polarization effect with the aid of an original physically-based differentiable polarimetric renderer. Lastly experimental results show that our method outperforms the existing works in synthetic and real scenarios.

count=1
* Instantaneous Perception of Moving Objects in 3D
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Instantaneous_Perception_of_Moving_Objects_in_3D_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Instantaneous_Perception_of_Moving_Objects_in_3D_CVPR_2024_paper.pdf)]
    * Title: Instantaneous Perception of Moving Objects in 3D
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Di Liu, Bingbing Zhuang, Dimitris N. Metaxas, Manmohan Chandraker
    * Abstract: The perception of 3D motion of surrounding traffic participants is crucial for driving safety. While existing works primarily focus on general large motions we contend that the instantaneous detection and quantification of subtle motions is equally important as they indicate the nuances in driving behavior that may be safety critical such as behaviors near a stop sign of parking positions. We delve into this under-explored task examining its unique challenges and developing our solution accompanied by a carefully designed benchmark. Specifically due to the lack of correspondences between consecutive frames of sparse Lidar point clouds static objects might appear to be moving - the so-called swimming effect. This intertwines with the true object motion thereby posing ambiguity in accurate estimation especially for subtle motion. To address this we propose to leverage local occupancy completion of object point clouds to densify the shape cue and mitigate the impact of swimming artifacts. The occupancy completion is learned in an end-to-end fashion together with the detection of moving objects and the estimation of their motion instantaneously as soon as objects start to move. Extensive experiments demonstrate superior performance compared to standard 3D motion estimation approaches particularly highlighting our method's specialized treatment of subtle motion.

count=1
* PI3D: Efficient Text-to-3D Generation with Pseudo-Image Diffusion
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_PI3D_Efficient_Text-to-3D_Generation_with_Pseudo-Image_Diffusion_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_PI3D_Efficient_Text-to-3D_Generation_with_Pseudo-Image_Diffusion_CVPR_2024_paper.pdf)]
    * Title: PI3D: Efficient Text-to-3D Generation with Pseudo-Image Diffusion
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ying-Tian Liu, Yuan-Chen Guo, Guan Luo, Heyi Sun, Wei Yin, Song-Hai Zhang
    * Abstract: Diffusion models trained on large-scale text-image datasets have demonstrated a strong capability of controllable high-quality image generation from arbitrary text prompts. However the generation quality and generalization ability of 3D diffusion models is hindered by the scarcity of high-quality and large-scale 3D datasets. In this paper we present PI3D a framework that fully leverages the pre-trained text-to-image diffusion models' ability to generate high-quality 3D shapes from text prompts in minutes. The core idea is to connect the 2D and 3D domains by representing a 3D shape as a set of Pseudo RGB Images. We fine-tune an existing text-to-image diffusion model to produce such pseudo-images using a small number of text-3D pairs. Surprisingly we find that it can already generate meaningful and consistent 3D shapes given complex text descriptions. We further take the generated shapes as the starting point for a lightweight iterative refinement using score distillation sampling to achieve high-quality generation under a low budget. PI3D generates a single 3D shape from text in only 3 minutes and the quality is validated to outperform existing 3D generative models by a large margin.

count=1
* UV-IDM: Identity-Conditioned Latent Diffusion Model for Face UV-Texture Generation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_UV-IDM_Identity-Conditioned_Latent_Diffusion_Model_for_Face_UV-Texture_Generation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_UV-IDM_Identity-Conditioned_Latent_Diffusion_Model_for_Face_UV-Texture_Generation_CVPR_2024_paper.pdf)]
    * Title: UV-IDM: Identity-Conditioned Latent Diffusion Model for Face UV-Texture Generation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Hong Li, Yutang Feng, Song Xue, Xuhui Liu, Bohan Zeng, Shanglin Li, Boyu Liu, Jianzhuang Liu, Shumin Han, Baochang Zhang
    * Abstract: 3D face reconstruction aims at generating high-fidelity 3D face shapes and textures from single-view or multi-view images. However current prevailing facial texture generation methods generally suffer from low-quality texture identity information loss and inadequate handling of occlusions. To solve these problems we introduce an Identity-Conditioned Latent Diffusion Model for face UV-texture generation (UV-IDM) to generate photo-realistic textures based on the Basel Face Model (BFM). UV-IDM leverages the powerful texture generation capacity of a latent diffusion model (LDM) to obtain detailed facial textures. To preserve the identity during the reconstruction procedure we design an identity-conditioned module that can utilize any in-the-wild image as a robust condition for the LDM to guide texture generation. UV-IDM can be easily adapted to different BFM-based methods as a high-fidelity texture generator. Furthermore in light of the limited accessibility of most existing UV-texture datasets we build a large-scale and publicly available UV-texture dataset based on BFM termed BFM-UV. Extensive experiments show that our UV-IDM can generate high-fidelity textures in 3D face reconstruction within seconds while maintaining image consistency bringing new state-of-the-art performance in facial texture generation.

count=1
* Total-Decom: Decomposed 3D Scene Reconstruction with Minimal Interaction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lyu_Total-Decom_Decomposed_3D_Scene_Reconstruction_with_Minimal_Interaction_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lyu_Total-Decom_Decomposed_3D_Scene_Reconstruction_with_Minimal_Interaction_CVPR_2024_paper.pdf)]
    * Title: Total-Decom: Decomposed 3D Scene Reconstruction with Minimal Interaction
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xiaoyang Lyu, Chirui Chang, Peng Dai, Yang-Tian Sun, Xiaojuan Qi
    * Abstract: Scene reconstruction from multi-view images is a fundamental problem in computer vision and graphics. Recent neural implicit surface reconstruction methods have achieved high-quality results; however editing and manipulating the 3D geometry of reconstructed scenes remains challenging due to the absence of naturally decomposed object entities and complex object/background compositions. In this paper we present Total-Decom a novel method for decomposed 3D reconstruction with minimal human interaction. Our approach seamlessly integrates the Segment Anything Model (SAM) with hybrid implicit-explicit neural surface representations and a mesh-based region-growing technique for accurate 3D object decomposition. Total-Decom requires minimal human annotations while providing users with real-time control over the granularity and quality of decomposition. We extensively evaluate our method on benchmark datasets and demonstrate its potential for downstream applications such as animation and scene editing.

count=1
* iToF-flow-based High Frame Rate Depth Imaging
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Meng_iToF-flow-based_High_Frame_Rate_Depth_Imaging_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Meng_iToF-flow-based_High_Frame_Rate_Depth_Imaging_CVPR_2024_paper.pdf)]
    * Title: iToF-flow-based High Frame Rate Depth Imaging
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yu Meng, Zhou Xue, Xu Chang, Xuemei Hu, Tao Yue
    * Abstract: iToF is a prevalent cost-effective technology for 3D perception. While its reliance on multi-measurement commonly leads to reduced performance in dynamic environments. Based on the analysis of the physical iToF imaging process we propose the iToF flow composed of crossmode transformation and uni-mode photometric correction to model the variation of measurements caused by different measurement modes and 3D motion respectively. We propose a local linear transform (LLT) based cross-mode transfer module (LCTM) for mode-varying and pixel shift compensation of cross-mode flow and uni-mode photometric correct module (UPCM) for estimating the depth-wise motion caused photometric residual of uni-mode flow. The iToF flow-based depth extraction network is proposed which could facilitate the estimation of the 4-phase measurements at each individual time for high framerate and accurate depth estimation. Extensive experiments including both simulation and real-world experiments are conducted to demonstrate the effectiveness of the proposed methods. Compared with the SOTA method our approach reduces the computation time by 75% while improving the performance by 38%. The code and database are available at https://github.com/ComputationalPerceptionLab/iToF_flow.

count=1
* Entity-NeRF: Detecting and Removing Moving Entities in Urban Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Otonari_Entity-NeRF_Detecting_and_Removing_Moving_Entities_in_Urban_Scenes_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Otonari_Entity-NeRF_Detecting_and_Removing_Moving_Entities_in_Urban_Scenes_CVPR_2024_paper.pdf)]
    * Title: Entity-NeRF: Detecting and Removing Moving Entities in Urban Scenes
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Takashi Otonari, Satoshi Ikehata, Kiyoharu Aizawa
    * Abstract: Recent advancements in the study of Neural Radiance Fields (NeRF) for dynamic scenes often involve explicit modeling of scene dynamics. However this approach faces challenges in modeling scene dynamics in urban environments where moving objects of various categories and scales are present. In such settings it becomes crucial to effectively eliminate moving objects to accurately reconstruct static backgrounds. Our research introduces an innovative method termed here as Entity-NeRF which combines the strengths of knowledge-based and statistical strategies. This approach utilizes entity-wise statistics leveraging entity segmentation and stationary entity classification through thing/stuff segmentation. To assess our methodology we created an urban scene dataset masked with moving objects. Our comprehensive experiments demonstrate that Entity-NeRF notably outperforms existing techniques in removing moving objects and reconstructing static urban backgrounds both quantitatively and qualitatively.

count=1
* Unsupervised Occupancy Learning from Sparse Point Cloud
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ouasfi_Unsupervised_Occupancy_Learning_from_Sparse_Point_Cloud_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ouasfi_Unsupervised_Occupancy_Learning_from_Sparse_Point_Cloud_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ouasfi_Unsupervised_Occupancy_Learning_from_Sparse_Point_Cloud_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ouasfi_Unsupervised_Occupancy_Learning_from_Sparse_Point_Cloud_CVPR_2024_paper.pdf)]
    * Title: Unsupervised Occupancy Learning from Sparse Point Cloud
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Amine Ouasfi, Adnane Boukhayma
    * Abstract: Implicit Neural Representations have gained prominence as a powerful framework for capturing complex data modalities encompassing a wide range from 3D shapes to images and audio. Within the realm of 3D shape representation Neural Signed Distance Functions (SDF) have demonstrated remarkable potential in faithfully encoding intricate shape geometry. However learning SDFs from 3D point clouds in the absence of ground truth supervision remains a very challenging task. In this paper we propose a method to infer occupancy fields instead of SDFs as they are easier to learn from sparse inputs. We leverage a margin-based uncertainty measure to differentiably sample from the decision boundary of the occupancy function and supervise the sampled boundary points using the input point cloud. We further stabilise the optimization process at the early stages of the training by biasing the occupancy function towards minimal entropy fields while maximizing its entropy at the input point cloud. Through extensive experiments and evaluations we illustrate the efficacy of our proposed method highlighting its capacity to improve implicit shape inference with respect to baselines and the state-of-the-art using synthetic and real data.

count=1
* Gated Fields: Learning Scene Reconstruction from Gated Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ramazzina_Gated_Fields_Learning_Scene_Reconstruction_from_Gated_Videos_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ramazzina_Gated_Fields_Learning_Scene_Reconstruction_from_Gated_Videos_CVPR_2024_paper.pdf)]
    * Title: Gated Fields: Learning Scene Reconstruction from Gated Videos
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Andrea Ramazzina, Stefanie Walz, Pragyan Dahal, Mario Bijelic, Felix Heide
    * Abstract: Reconstructing outdoor 3D scenes from temporal observations is a challenge that recent work on neural fields has offered a new avenue for. However existing methods that recover scene properties such as geometry appearance or radiance solely from RGB captures often fail when handling poorly-lit or texture-deficient regions. Similarly recovering scenes with scanning lidar sensors is also difficult due to their low angular sampling rate which makes recovering expansive real-world scenes difficult. Tackling these gaps we introduce Gated Fields - a neural scene reconstruction method that utilizes active gated video sequences. To this end we propose a neural rendering approach that seamlessly incorporates time-gated capture and illumination. Our method exploits the intrinsic depth cues in the gated videos achieving precise and dense geometry reconstruction irrespective of ambient illumination conditions. We validate the method across day and night scenarios and find that Gated Fields compares favorably to RGB and LiDAR reconstruction methods

count=1
* AV-RIR: Audio-Visual Room Impulse Response Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ratnarajah_AV-RIR_Audio-Visual_Room_Impulse_Response_Estimation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ratnarajah_AV-RIR_Audio-Visual_Room_Impulse_Response_Estimation_CVPR_2024_paper.pdf)]
    * Title: AV-RIR: Audio-Visual Room Impulse Response Estimation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Anton Ratnarajah, Sreyan Ghosh, Sonal Kumar, Purva Chiniya, Dinesh Manocha
    * Abstract: Accurate estimation of Room Impulse Response (RIR) which captures an environment's acoustic properties is important for speech processing and AR/VR applications. We propose AV-RIR a novel multi-modal multi-task learning approach to accurately estimate the RIR from a given reverberant speech signal and the visual cues of its corresponding environment. AV-RIR builds on a novel neural codec-based architecture that effectively captures environment geometry and materials properties and solves speech dereverberation as an auxiliary task by using multi-task learning. We also propose Geo-Mat features that augment material information into visual cues and CRIP that improves late reverberation components in the estimated RIR via image-to-RIR retrieval by 86%. Empirical results show that AV-RIR quantitatively outperforms previous audio-only and visual-only approaches by achieving 36% - 63% improvement across various acoustic metrics in RIR estimation. Additionally it also achieves higher preference scores in human evaluation. As an auxiliary benefit dereverbed speech from AV-RIR shows competitive performance with the state-of-the-art in various spoken language processing tasks and outperforms reverberation time error score in the real-world AVSpeech dataset. Qualitative examples of both synthesized reverberant speech and enhanced speech are available online https://www.youtube.com/watch?v=tTsKhviukAE.

count=1
* Monocular Identity-Conditioned Facial Reflectance Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ren_Monocular_Identity-Conditioned_Facial_Reflectance_Reconstruction_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_Monocular_Identity-Conditioned_Facial_Reflectance_Reconstruction_CVPR_2024_paper.pdf)]
    * Title: Monocular Identity-Conditioned Facial Reflectance Reconstruction
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xingyu Ren, Jiankang Deng, Yuhao Cheng, Jia Guo, Chao Ma, Yichao Yan, Wenhan Zhu, Xiaokang Yang
    * Abstract: Recent 3D face reconstruction methods have made remarkable advancements yet there remain huge challenges in monocular high-quality facial reflectance reconstruction. Existing methods rely on a large amount of light-stage captured data to learn facial reflectance models. However the lack of subject diversity poses challenges in achieving good generalization and widespread applicability. In this paper we learn the reflectance prior in image space rather than UV space and present a framework named ID2Reflectance. Our framework can directly estimate the reflectance maps of a single image while using limited reflectance data for training. Our key insight is that reflectance data shares facial structures with RGB faces which enables obtaining expressive facial prior from inexpensive RGB data thus reducing the dependency on reflectance data. We first learn a high-quality prior for facial reflectance. Specifically we pretrain multi-domain facial feature codebooks and design a codebook fusion method to align the reflectance and RGB domains. Then we propose an identity-conditioned swapping module that injects facial identity from the target image into the pre-trained auto-encoder to modify the identity of the source reflectance image. Finally we stitch multi-view swapped reflectance images to obtain renderable assets. Extensive experiments demonstrate that our method exhibits excellent generalization capability and achieves state-of-the-art facial reflectance reconstruction results for in-the-wild faces.

count=1
* Unbiased Estimator for Distorted Conics in Camera Calibration
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Song_Unbiased_Estimator_for_Distorted_Conics_in_Camera_Calibration_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Song_Unbiased_Estimator_for_Distorted_Conics_in_Camera_Calibration_CVPR_2024_paper.pdf)]
    * Title: Unbiased Estimator for Distorted Conics in Camera Calibration
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chaehyeon Song, Jaeho Shin, Myung-Hwan Jeon, Jongwoo Lim, Ayoung Kim
    * Abstract: In the literature points and conics have been major features for camera geometric calibration. Although conics are more informative features than points the loss of the conic property under distortion has critically limited the utility of conic features in camera calibration. Many existing approaches addressed conic-based calibration by ignoring distortion or introducing 3D spherical targets to circumvent this limitation. In this paper we present a novel formulation for conic-based calibration using moments. Our derivation is based on the mathematical finding that the first moment can be estimated without bias even under distortion. This allows us to track moment changes during projection and distortion ensuring the preservation of the first moment of the distorted conic. With an unbiased estimator the circular patterns can be accurately detected at the sub-pixel level and can now be fully exploited for an entire calibration pipeline resulting in significantly improved calibration. The entire code is readily available from https://github.com/ChaehyeonSong/discocal.

count=1
* Self-Supervised Dual Contouring
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sundararaman_Self-Supervised_Dual_Contouring_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sundararaman_Self-Supervised_Dual_Contouring_CVPR_2024_paper.pdf)]
    * Title: Self-Supervised Dual Contouring
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ramana Sundararaman, Roman Klokov, Maks Ovsjanikov
    * Abstract: Learning-based isosurface extraction methods have recently emerged as a robust and efficient alternative to axiomatic techniques. However the vast majority of such approaches rely on supervised training with axiomatically computed ground truths thus potentially inheriting biases and data artefacts of the corresponding axiomatic methods. Steering away from such dependencies we propose a self-supervised training scheme to the Neural Dual Contouring meshing framework resulting in our method: Self-Supervised Dual Contouring (SDC). Instead of optimizing predicted mesh vertices with supervised training we use two novel self-supervised loss functions that encourage the consistency between distances to the generated mesh up to the first order. Meshes reconstructed by SDC surpass existing data-driven methods in capturing intricate details while being more robust to possible irregularities in the input. Furthermore we use the same self-supervised training objective linking inferred mesh and input SDF to regularize the training process of Deep Implicit Networks (DINs). We demonstrate that the resulting DINs produce higher-quality implicit functions ultimately leading to more accurate and detail-preserving surfaces compared to prior baselines for different input modalities. Finally we demonstrate that our self-supervised losses improve meshing performance in the single-view reconstruction task by enabling joint training of predicted SDF and resulting output mesh.

count=1
* DyBluRF: Dynamic Neural Radiance Fields from Blurry Monocular Video
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_DyBluRF_Dynamic_Neural_Radiance_Fields_from_Blurry_Monocular_Video_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_DyBluRF_Dynamic_Neural_Radiance_Fields_from_Blurry_Monocular_Video_CVPR_2024_paper.pdf)]
    * Title: DyBluRF: Dynamic Neural Radiance Fields from Blurry Monocular Video
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Huiqiang Sun, Xingyi Li, Liao Shen, Xinyi Ye, Ke Xian, Zhiguo Cao
    * Abstract: Recent advancements in dynamic neural radiance field methods have yielded remarkable outcomes. However these approaches rely on the assumption of sharp input images. When faced with motion blur existing dynamic NeRF methods often struggle to generate high-quality novel views. In this paper we propose DyBluRF a dynamic radiance field approach that synthesizes sharp novel views from a monocular video affected by motion blur. To account for motion blur in input images we simultaneously capture the camera trajectory and object Discrete Cosine Transform (DCT) trajectories within the scene. Additionally we employ a global cross-time rendering approach to ensure consistent temporal coherence across the entire scene. We curate a dataset comprising diverse dynamic scenes that are specifically tailored for our task. Experimental results on our dataset demonstrate that our method outperforms existing approaches in generating sharp novel views from motion-blurred inputs while maintaining spatial-temporal consistency of the scene.

count=1
* LidaRF: Delving into Lidar for Neural Radiance Field on Street Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_LidaRF_Delving_into_Lidar_for_Neural_Radiance_Field_on_Street_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_LidaRF_Delving_into_Lidar_for_Neural_Radiance_Field_on_Street_CVPR_2024_paper.pdf)]
    * Title: LidaRF: Delving into Lidar for Neural Radiance Field on Street Scenes
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Shanlin Sun, Bingbing Zhuang, Ziyu Jiang, Buyu Liu, Xiaohui Xie, Manmohan Chandraker
    * Abstract: Photorealistic simulation plays a crucial role in applications such as autonomous driving where advances in neural radiance fields (NeRFs) may allow better scalability through the automatic creation of digital 3D assets. However reconstruction quality suffers on street scenes due to largely collinear camera motions and sparser samplings at higher speeds. On the other hand the application often demands rendering from camera views that deviate from the inputs to accurately simulate behaviors like lane changes. In this paper we propose several insights that allow a better utilization of Lidar data to improve NeRF quality on street scenes. First our framework learns a geometric scene representation from Lidar which are fused with the implicit grid-based representation for radiance decoding thereby supplying stronger geometric information offered by explicit point cloud. Second we put forth a robust occlusion-aware depth supervision scheme which allows utilizing densified Lidar points by accumulation. Third we generate augmented training views from Lidar points for further improvement. Our insights translate to largely improved novel view synthesis under real driving scenes.

count=1
* Seg2Reg: Differentiable 2D Segmentation to 1D Regression Rendering for 360 Room Layout Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Seg2Reg_Differentiable_2D_Segmentation_to_1D_Regression_Rendering_for_360_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_Seg2Reg_Differentiable_2D_Segmentation_to_1D_Regression_Rendering_for_360_CVPR_2024_paper.pdf)]
    * Title: Seg2Reg: Differentiable 2D Segmentation to 1D Regression Rendering for 360 Room Layout Reconstruction
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Cheng Sun, Wei-En Tai, Yu-Lin Shih, Kuan-Wei Chen, Yong-Jing Syu, Kent Selwyn The, Yu-Chiang Frank Wang, Hwann-Tzong Chen
    * Abstract: State-of-the-art single-view 360 room layout reconstruction methods formulate the problem as a high-level 1D (per-column) regression task. On the other hand traditional low-level 2D layout segmentation is simpler to learn and can represent occluded regions but it requires complex post-processing for the targeting layout polygon and sacrifices accuracy. We present Seg2Reg to render 1D layout depth regression from the 2D segmentation map in a differentiable and occlusion-aware way marrying the merits of both sides. Specifically our model predicts floor-plan density for the input equirectangular 360 image. Formulating the 2D layout representation as a density field enables us to employ 'flattened' volume rendering to form 1D layout depth regression. In addition we propose a novel 3D warping augmentation on layout to improve generalization. Finally we re-implement recent room layout reconstruction methods into our codebase for benchmarking and explore modern backbones and training techniques to serve as the strong baseline. The code is at https: //PanoLayoutStudio.github.io .

count=1
* Hearing Anything Anywhere
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Hearing_Anything_Anywhere_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Hearing_Anything_Anywhere_CVPR_2024_paper.pdf)]
    * Title: Hearing Anything Anywhere
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Mason Long Wang, Ryosuke Sawata, Samuel Clarke, Ruohan Gao, Shangzhe Wu, Jiajun Wu
    * Abstract: Recent years have seen immense progress in 3D computer vision and computer graphics with emerging tools that can virtualize real-world 3D environments for numerous Mixed Reality (XR) applications. However alongside immersive visual experiences immersive auditory experiences are equally vital to our holistic perception of an environment. In this paper we aim to reconstruct the spatial acoustic characteristics of an arbitrary environment given only a sparse set of (roughly 12) room impulse response (RIR) recordings and a planar reconstruction of the scene a setup that is easily achievable by ordinary users. To this end we introduce DiffRIR a differentiable RIR rendering framework with interpretable parametric models of salient acoustic features of the scene including sound source directivity and surface reflectivity. This allows us to synthesize novel auditory experiences through the space with any source audio. To evaluate our method we collect a dataset of RIR recordings and music in four diverse real environments. We show that our model outperforms state-of-the-art baselines on rendering monaural and binaural RIRs and music at unseen locations and learns physically interpretable parameters characterizing acoustic properties of the sound source and surfaces in the scene.

count=1
* PanoOcc: Unified Occupancy Representation for Camera-based 3D Panoptic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_PanoOcc_Unified_Occupancy_Representation_for_Camera-based_3D_Panoptic_Segmentation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_PanoOcc_Unified_Occupancy_Representation_for_Camera-based_3D_Panoptic_Segmentation_CVPR_2024_paper.pdf)]
    * Title: PanoOcc: Unified Occupancy Representation for Camera-based 3D Panoptic Segmentation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yuqi Wang, Yuntao Chen, Xingyu Liao, Lue Fan, Zhaoxiang Zhang
    * Abstract: Comprehensive modeling of the surrounding 3D world is crucial for the success of autonomous driving. However existing perception tasks like object detection road structure segmentation depth & elevation estimation and open-set object localization each only focus on a small facet of the holistic 3D scene understanding task. This divide-and-conquer strategy simplifies the algorithm development process but comes at the cost of losing an end-to-end unified solution to the problem. In this work we address this limitation by studying camera-based 3D panoptic segmentation aiming to achieve a unified occupancy representation for camera-only 3D scene understanding. To achieve this we introduce a novel method called PanoOcc which utilizes voxel queries to aggregate spatiotemporal information from multi-frame and multi-view images in a coarse-to-fine scheme integrating feature learning and scene representation into a unified occupancy representation. We have conducted extensive ablation studies to validate the effectiveness and efficiency of the proposed method. Our approach achieves new state-of-the-art results for camera-based semantic segmentation and panoptic segmentation on the nuScenes dataset. Furthermore our method can be easily extended to dense occupancy prediction and has demonstrated promising performance on the Occ3D benchmark. The code will be made available at https://github.com/Robertwyq/PanoOcc.

count=1
* Effective Video Mirror Detection with Inconsistent Motion Cues
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Warren_Effective_Video_Mirror_Detection_with_Inconsistent_Motion_Cues_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Warren_Effective_Video_Mirror_Detection_with_Inconsistent_Motion_Cues_CVPR_2024_paper.pdf)]
    * Title: Effective Video Mirror Detection with Inconsistent Motion Cues
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Alex Warren, Ke Xu, Jiaying Lin, Gary K.L. Tam, Rynson W.H. Lau
    * Abstract: Image-based mirror detection has recently undergone rapid research due to its significance in applications such as robotic navigation semantic segmentation and scene reconstruction. Recently VMD-Net was proposed as the first video mirror detection technique by modeling dual correspondences between the inside and outside of the mirror both spatially and temporally. However this approach is not reliable as correspondences can occur completely inside or outside of the mirrors. In addition the proposed dataset VMD-D contains many small mirrors limiting its applicability to real-world scenarios. To address these problems we developed a more challenging dataset that includes mirrors of various shapes and sizes at different locations of the frames providing a better reflection of real-world scenarios. Next we observed that the motions between the inside and outside of the mirror are often inconsistent. For instance when moving in front of a mirror the motion inside the mirror is often much smaller than the motion outside due to increased depth perception. With these observations we propose modeling inconsistent motion cues to detect mirrors and a new network with two novel modules. The Motion Attention Module (MAM) explicitly models inconsistent motions around mirrors via optical flow and the Motion-Guided Edge Detection Module (MEDM) uses motions to guide mirror edge feature learning. Experimental results on our proposed dataset show that our method outperforms state-of-the-arts. The code and dataset are available at https://github.com/AlexAnthonyWarren/MG-VMD.

count=1
* FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.pdf)]
    * Title: FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield
    * Abstract: We present FoundationPose a unified foundation model for 6D object pose estimation and tracking supporting both model-based and model-free setups. Our approach can be instantly applied at test-time to a novel object without finetuning as long as its CAD model is given or a small number of reference images are captured. Thanks to the unified framework the downstream pose estimation modules are the same in both setups with a neural implicit representation used for efficient novel view synthesis when no CAD model is available. Strong generalizability is achieved via large-scale synthetic training aided by a large language model (LLM) a novel transformer-based architecture and contrastive learning formulation. Extensive evaluation on multiple public datasets involving challenging scenarios and objects indicate our unified approach outperforms existing methods specialized for each task by a large margin. In addition it even achieves comparable results to instance-level methods despite the reduced assumptions. Project page: https://nvlabs.github.io/FoundationPose/

count=1
* GoMAvatar: Efficient Animatable Human Modeling from Monocular Video Using Gaussians-on-Mesh
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_GoMAvatar_Efficient_Animatable_Human_Modeling_from_Monocular_Video_Using_Gaussians-on-Mesh_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_GoMAvatar_Efficient_Animatable_Human_Modeling_from_Monocular_Video_Using_Gaussians-on-Mesh_CVPR_2024_paper.pdf)]
    * Title: GoMAvatar: Efficient Animatable Human Modeling from Monocular Video Using Gaussians-on-Mesh
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jing Wen, Xiaoming Zhao, Zhongzheng Ren, Alexander G. Schwing, Shenlong Wang
    * Abstract: We introduce GoMAvatar a novel approach for real-time memory-efficient high-quality animatable human modeling. GoMAvatar takes as input a single monocular video to create a digital avatar capable of re-articulation in new poses and real-time rendering from novel viewpoints while seamlessly integrating with rasterization-based graphics pipelines. Central to our method is the Gaussians-on-Mesh (GoM) representation a hybrid 3D model combining rendering quality and speed of Gaussian splatting with geometry modeling and compatibility of deformable meshes. We assess GoMAvatar on ZJU-MoCap PeopleSnapshot and various YouTube videos. GoMAvatar matches or surpasses current monocular human modeling algorithms in rendering quality and significantly outperforms them in computational efficiency (43 FPS) while being memory-efficient (3.63 MB per subject).

count=1
* Neural Directional Encoding for Efficient and Accurate View-Dependent Appearance Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Neural_Directional_Encoding_for_Efficient_and_Accurate_View-Dependent_Appearance_Modeling_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Neural_Directional_Encoding_for_Efficient_and_Accurate_View-Dependent_Appearance_Modeling_CVPR_2024_paper.pdf)]
    * Title: Neural Directional Encoding for Efficient and Accurate View-Dependent Appearance Modeling
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Liwen Wu, Sai Bi, Zexiang Xu, Fujun Luan, Kai Zhang, Iliyan Georgiev, Kalyan Sunkavalli, Ravi Ramamoorthi
    * Abstract: Novel-view synthesis of specular objects like shiny metals or glossy paints remains a significant challenge. Not only the glossy appearance but also global illumination effects including reflections of other objects in the environment are critical components to faithfully reproduce a scene. In this paper we present Neural Directional Encoding (NDE) a view-dependent appearance encoding of neural radiance fields (NeRF) for rendering specular objects. NDE transfers the concept of feature-grid-based spatial encoding to the angular domain significantly improving the ability to model high-frequency angular signals. In contrast to previous methods that use encoding functions with only angular input we additionally cone-trace spatial features to obtain a spatially varying directional encoding which addresses the challenging interreflection effects. Extensive experiments on both synthetic and real datasets show that a NeRF model with NDE (1) outperforms the state of the art on view synthesis of specular objects and (2) works with small networks to allow fast (real-time) inference. The source code is available at: https://github.com/lwwu2/nde

count=1
* NECA: Neural Customizable Human Avatar
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_NECA_Neural_Customizable_Human_Avatar_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_NECA_Neural_Customizable_Human_Avatar_CVPR_2024_paper.pdf)]
    * Title: NECA: Neural Customizable Human Avatar
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Junjin Xiao, Qing Zhang, Zhan Xu, Wei-Shi Zheng
    * Abstract: Human avatar has become a novel type of 3D asset with various applications. Ideally a human avatar should be fully customizable to accommodate different settings and environments. In this work we introduce NECA an approach capable of learning versatile human representation from monocular or sparse-view videos enabling granular customization across aspects such as pose shadow shape lighting and texture. The core of our approach is to represent humans in complementary dual spaces and predict disentangled neural fields of geometry albedo shadow as well as an external lighting from which we are able to derive realistic rendering with high-frequency details via volumetric rendering. Extensive experiments demonstrate the advantage of our method over the state-of-the-art methods in photorealistic rendering as well as various editing tasks such as novel pose synthesis and relighting. Our code is available at https://github.com/iSEE-Laboratory/NECA.

count=1
* WaveMo: Learning Wavefront Modulations to See Through Scattering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_WaveMo_Learning_Wavefront_Modulations_to_See_Through_Scattering_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_WaveMo_Learning_Wavefront_Modulations_to_See_Through_Scattering_CVPR_2024_paper.pdf)]
    * Title: WaveMo: Learning Wavefront Modulations to See Through Scattering
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Mingyang Xie, Haiyun Guo, Brandon Y. Feng, Lingbo Jin, Ashok Veeraraghavan, Christopher A. Metzler
    * Abstract: Imaging through scattering media is a fundamental and pervasive challenge in fields ranging from medical diagnostics to astronomy. A promising strategy to overcome this challenge is wavefront modulation which induces measurement diversity during image acquisition. Despite its importance designing optimal wavefront modulations to image through scattering remains under-explored. This paper introduces a novel learning-based framework to address the gap. Our approach jointly optimizes wavefront modulations and a computationally lightweight feedforward "proxy" reconstruction network. This network is trained to recover scenes obscured by scattering using measurements that are modified by these modulations. The learned modulations produced by our framework generalize effectively to unseen scattering scenarios and exhibit remarkable versatility. During deployment the learned modulations can be decoupled from the proxy network to augment other more computationally expensive restoration algorithms. Through extensive experiments we demonstrate our approach significantly advances the state of the art in imaging through scattering media. Our project webpage is at https://wavemo-2024.github.io/.

count=1
* Relightable and Animatable Neural Avatar from Sparse-View Video
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Relightable_and_Animatable_Neural_Avatar_from_Sparse-View_Video_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Relightable_and_Animatable_Neural_Avatar_from_Sparse-View_Video_CVPR_2024_paper.pdf)]
    * Title: Relightable and Animatable Neural Avatar from Sparse-View Video
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zhen Xu, Sida Peng, Chen Geng, Linzhan Mou, Zihan Yan, Jiaming Sun, Hujun Bao, Xiaowei Zhou
    * Abstract: This paper tackles the problem of creating relightable and animatable neural avatars from sparse-view (or monocular) videos of dynamic humans under unknown illumination. Previous neural human reconstruction methods produce animatable avatars from sparse views using deformed Signed Distance Fields (SDF) but are non-relightable. While differentiable inverse rendering methods have succeeded in the material recovery of static objects it is not straightforward to extend them to dynamic humans since it is computationally intensive to compute pixel-surface intersection and light visibility on deformed SDFs for relighting. To solve this challenge we propose a Hierarchical Distance Query (HDQ) algorithm to approximate the world space SDF under arbitrary human poses. Specifically we estimate coarse SDF based on a parametric human model and compute fine SDF by exploiting the invariance of SDF w.r.t. local deformation. Based on HDQ we leverage sphere tracing to efficiently estimate the surface intersection and light visibility. This allows us to develop the first system to recover relightable and animatable neural avatars from sparse or monocular inputs. Experiments show that our approach produces superior results compared to state-of-the-art methods. Our project page is available at https://zju3dv.github.io/relightable_avatar.

count=1
* UniPAD: A Universal Pre-training Paradigm for Autonomous Driving
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_UniPAD_A_Universal_Pre-training_Paradigm_for_Autonomous_Driving_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_UniPAD_A_Universal_Pre-training_Paradigm_for_Autonomous_Driving_CVPR_2024_paper.pdf)]
    * Title: UniPAD: A Universal Pre-training Paradigm for Autonomous Driving
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Honghui Yang, Sha Zhang, Di Huang, Xiaoyang Wu, Haoyi Zhu, Tong He, Shixiang Tang, Hengshuang Zhao, Qibo Qiu, Binbin Lin, Xiaofei He, Wanli Ouyang
    * Abstract: In the context of autonomous driving the significance of effective feature learning is widely acknowledged. While conventional 3D self-supervised pre-training methods have shown widespread success most methods follow the ideas originally designed for 2D images. In this paper we present UniPAD a novel self-supervised learning paradigm applying 3D volumetric differentiable rendering. UniPAD implicitly encodes 3D space facilitating the reconstruction of continuous 3D shape structures and the intricate appearance characteristics of their 2D projections. The flexibility of our method enables seamless integration into both 2D and 3D frameworks enabling a more holistic comprehension of the scenes. We manifest the feasibility and effectiveness of UniPAD by conducting extensive experiments on various 3D perception tasks. Our method significantly improves lidar- camera- and lidar-camera-based baseline by 9.1 7.7 and 6.9 NDS respectively. Notably our pre-training pipeline achieves 73.2 NDS for 3D object detection and 79.4 mIoU for 3D semantic segmentation on the nuScenes validation set achieving state-of-the-art results in comparison with previous methods.

count=1
* EventPS: Real-Time Photometric Stereo Using an Event Camera
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_EventPS_Real-Time_Photometric_Stereo_Using_an_Event_Camera_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_EventPS_Real-Time_Photometric_Stereo_Using_an_Event_Camera_CVPR_2024_paper.pdf)]
    * Title: EventPS: Real-Time Photometric Stereo Using an Event Camera
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Bohan Yu, Jieji Ren, Jin Han, Feishi Wang, Jinxiu Liang, Boxin Shi
    * Abstract: Photometric stereo is a well-established technique to estimate the surface normal of an object. However the requirement of capturing multiple high dynamic range images under different illumination conditions limits the speed and real-time applications. This paper introduces EventPS a novel approach to real-time photometric stereo using an event camera. Capitalizing on the exceptional temporal resolution dynamic range and low bandwidth characteristics of event cameras EventPS estimates surface normal only from the radiance changes significantly enhancing data efficiency. EventPS seamlessly integrates with both optimization-based and deep-learning-based photometric stereo techniques to offer a robust solution for non-Lambertian surfaces. Extensive experiments validate the effectiveness and efficiency of EventPS compared to frame-based counterparts. Our algorithm runs at over 30 fps in real-world scenarios unleashing the potential of EventPS in time-sensitive and high-speed downstream applications.

count=1
* Functional Diffusion
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Functional_Diffusion_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Functional_Diffusion_CVPR_2024_paper.pdf)]
    * Title: Functional Diffusion
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Biao Zhang, Peter Wonka
    * Abstract: We propose functional diffusion a generative diffusion model focused on infinite-dimensional function data samples. In contrast to previous work functional diffusion works on samples that are represented by functions with a continuous domain. Functional diffusion can be seen as an extension of classical diffusion models to an infinite-dimensional domain. Functional diffusion is very versatile as images videos audio 3D shapes deformations etc. can be handled by the same framework with minimal changes. In addition functional diffusion is especially suited for irregular data or data defined in non-standard domains. In our work we derive the necessary foundations for functional diffusion and propose a first implementation based on the transformer architecture. We show generative results on complicated signed distance functions and deformation functions defined on 3D surfaces.

count=1
* Learning Dynamic Tetrahedra for High-Quality Talking Head Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Learning_Dynamic_Tetrahedra_for_High-Quality_Talking_Head_Synthesis_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Learning_Dynamic_Tetrahedra_for_High-Quality_Talking_Head_Synthesis_CVPR_2024_paper.pdf)]
    * Title: Learning Dynamic Tetrahedra for High-Quality Talking Head Synthesis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Zicheng Zhang, Ruobing Zheng, Bonan Li, Congying Han, Tianqi Li, Meng Wang, Tiande Guo, Jingdong Chen, Ziwen Liu, Ming Yang
    * Abstract: Recent works in implicit representations such as Neural Radiance Fields (NeRF) have advanced the generation of realistic and animatable head avatars from video sequences. These implicit methods are still confronted by visual artifacts and jitters since the lack of explicit geometric constraints poses a fundamental challenge in accurately modeling complex facial deformations. In this paper we introduce Dynamic Tetrahedra (DynTet) a novel hybrid representation that encodes explicit dynamic meshes by neural networks to ensure geometric consistency across various motions and viewpoints. DynTet is parameterized by the coordinate-based networks which learn signed distance deformation and material texture anchoring the training data into a predefined tetrahedra grid. Leveraging Marching Tetrahedra DynTet efficiently decodes textured meshes with a consistent topology enabling fast rendering through a differentiable rasterizer and supervision via a pixel loss. To enhance training efficiency we incorporate classical 3D Morphable Models to facilitate geometry learning and define a canonical space for simplifying texture learning. These advantages are readily achievable owing to the effective geometric representation employed in DynTet. Compared with prior works DynTet demonstrates significant improvements in fidelity lip synchronization and real-time performance according to various metrics. Beyond producing stable and visually appealing synthesis videos our method also outputs the dynamic meshes which is promising to enable many emerging applications. Code is available at https://github.com/zhangzc21/DynTet.

count=1
* Improving Bird's Eye View Semantic Segmentation by Task Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Improving_Birds_Eye_View_Semantic_Segmentation_by_Task_Decomposition_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Improving_Birds_Eye_View_Semantic_Segmentation_by_Task_Decomposition_CVPR_2024_paper.pdf)]
    * Title: Improving Bird's Eye View Semantic Segmentation by Task Decomposition
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tianhao Zhao, Yongcan Chen, Yu Wu, Tianyang Liu, Bo Du, Peilun Xiao, Shi Qiu, Hongda Yang, Guozhen Li, Yi Yang, Yutian Lin
    * Abstract: Semantic segmentation in bird's eye view (BEV) plays a crucial role in autonomous driving. Previous methods usually follow an end-to-end pipeline directly predicting the BEV segmentation map from monocular RGB inputs. However the challenge arises when the RGB inputs and BEV targets from distinct perspectives making the direct point-to-point predicting hard to optimize. In this paper we decompose the original BEV segmentation task into two stages namely BEV map reconstruction and RGB-BEV feature alignment. In the first stage we train a BEV autoencoder to reconstruct the BEV segmentation maps given corrupted noisy latent representation which urges the decoder to learn fundamental knowledge of typical BEV patterns. The second stage involves mapping RGB input images into the BEV latent space of the first stage directly optimizing the correlations between the two views at the feature level. Our approach simplifies the complexity of combining perception and generation into distinct steps equipping the model to handle intricate and challenging scenes effectively. Besides we propose to transform the BEV segmentation map from the Cartesian to the polar coordinate system to establish the column-wise correspondence between RGB images and BEV maps. Moreover our method requires neither multi-scale features nor camera intrinsic parameters for depth estimation and saves computational overhead. Extensive experiments on nuScenes and Argoverse show the effectiveness and efficiency of our method. Code is available at https://github.com/happytianhao/TaDe.

count=1
* 3D LiDAR Mapping in Dynamic Environments using a 4D Implicit Neural Representation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhong_3D_LiDAR_Mapping_in_Dynamic_Environments_using_a_4D_Implicit_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhong_3D_LiDAR_Mapping_in_Dynamic_Environments_using_a_4D_Implicit_CVPR_2024_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhong_3D_LiDAR_Mapping_in_Dynamic_Environments_using_a_4D_Implicit_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhong_3D_LiDAR_Mapping_in_Dynamic_Environments_using_a_4D_Implicit_CVPR_2024_paper.pdf)]
    * Title: 3D LiDAR Mapping in Dynamic Environments using a 4D Implicit Neural Representation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xingguang Zhong, Yue Pan, Cyrill Stachniss, Jens Behley
    * Abstract: Building accurate maps is a key building block to enable reliable localization planning and navigation of autonomous vehicles. We propose a novel approach for building accurate 3D maps of dynamic environments utilizing a sequence of LiDAR scans. To this end we propose encoding the 4D scene into a novel spatio-temporal implicit neural map representation by fitting a time-dependent truncated signed distance function to each point. Using our representation we can extract the static map by filtering the dynamic parts. Our neural representation is based on sparse feature grids a globally shared decoder and time-dependent basis functions which can be jointly optimized in an unsupervised fashion. To learn this representation from a sequence of LiDAR scans we design a simple yet efficient loss function to supervise the map optimization in a piecewise way. We evaluate our approach on various scenes containing moving objects in terms of the reconstruction quality of static maps and the segmentation of dynamic point clouds. The experimental results demonstrate that our method is capable of removing the dynamic part of the input point clouds while reconstructing accurate and complete large-scale 3D maps outperforming several state-of-the-art methods for static map generation and scene reconstruction.

count=1
* Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.pdf)]
    * Title: Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, Achuta Kadambi
    * Abstract: 3D scene representations have gained immense popularity in recent years. Methods that use Neural Radiance fields are versatile for traditional tasks such as novel view synthesis. In recent times some work has emerged that aims to extend the functionality of NeRF beyond view synthesis for semantically aware tasks such as editing and segmentation using 3D feature field distillation from 2D foundation models. However these methods have two major limitations: (a) they are limited by the rendering speed of NeRF pipelines and (b) implicitly represented feature fields suffer from continuity artifacts reducing feature quality. Recently 3D Gaussian Splatting has shown state-of-the-art performance on real-time radiance field rendering. In this work we go one step further: in addition to radiance field rendering we enable 3D Gaussian splatting on arbitrary-dimension semantic features via 2D foundation model distillation. This translation is not straightforward: naively incorporating feature fields in the 3DGS framework encounters significant challenges notably the disparities in spatial resolution and channel consistency between RGB images and feature maps. We propose architectural and training changes to efficiently avert this problem. Our proposed method is general and our experiments showcase novel view semantic segmentation language-guided editing and segment anything through learning feature fields from state-of-the-art 2D foundation models such as SAM and CLIP-LSeg. Across experiments our distillation method is able to provide comparable or better results while being significantly faster to both train and render. Additionally to the best of our knowledge we are the first method to enable point and bounding-box prompting for radiance field manipulation by leveraging the SAM model. Project website at: https://feature-3dgs.github.io/

count=1
* 3D Clothed Human Reconstruction from Sparse Multi-view Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/3DMV/html/Hong_3D_Clothed_Human_Reconstruction_from_Sparse_Multi-view_Images_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/3DMV/papers/Hong_3D_Clothed_Human_Reconstruction_from_Sparse_Multi-view_Images_CVPRW_2024_paper.pdf)]
    * Title: 3D Clothed Human Reconstruction from Sparse Multi-view Images
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jin Gyu Hong, Seung Young Noh, Hee Kyung Lee, Won Sik Cheong, Ju Yong Chang
    * Abstract: Clothed human reconstruction based on implicit functions has recently received considerable attention. In this study we explore the most effective 2D feature fusion method from multi-view inputs experimentally and propose a method utilizing the 3D coarse volume predicted by the network to provide a better 3D prior. We fuse 2D features using an attention-based method to obtain detailed geometric predictions. In addition we propose depth and color projection networks that predict the coarse depth volume and the coarse color volume from the input RGB images and depth maps respectively. Coarse depth volume and coarse color volume are used as 3D priors to predict occupancy and texture respectively. Further we combine the fused 2D features and 3D features extracted from our 3D prior to predict occupancy and propose a technique to adjust the influence of 2D and 3D features using learnable weights. The effectiveness of our method is demonstrated through qualitative and quantitative comparisons with recent multi-view clothed human reconstruction models.

count=1
* Benchmarking Robustness in Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AdvML/html/Wang_Benchmarking_Robustness_in_Neural_Radiance_Fields_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AdvML/papers/Wang_Benchmarking_Robustness_in_Neural_Radiance_Fields_CVPRW_2024_paper.pdf)]
    * Title: Benchmarking Robustness in Neural Radiance Fields
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chen Wang, Angtian Wang, Junbo Li, Alan Yuille, Cihang Xie
    * Abstract: Neural Radiance Field (NeRF) has demonstrated excellent quality in novel view synthesis thanks to its ability to model 3D object geometries in a concise formulation. However current approaches to NeRF-based models rely on clean images with accurate camera calibration which can be difficult to obtain in the real world where data is often subject to corruption and distortion. In this work we provide the first comprehensive analysis of the robustness of NeRF-based novel view synthesis algorithms in the presence of different types of corruptions. We find that NeRF-based models are significantly degraded in the presence of corruption and are more sensitive to a different set of corruptions than image recognition models. Furthermore we analyze the robustness of the feature encoder in generalizable methods which synthesize images using neural features extracted via convolutional neural networks or transformers and find that it only contributes marginally to robustness. Finally we reveal that standard data augmentation techniques which can significantly improve the robustness of recognition models do not help the robustness of NeRF-based models. We hope our findings will attract more researchers to study the robustness of NeRF-based approaches and help improve their performance in the real world.

count=1
* Exploring AI-Based Satellite Pose Estimation: from Novel Synthetic Dataset to In-Depth Performance Evaluation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Space/html/Gallet_Exploring_AI-Based_Satellite_Pose_Estimation_from_Novel_Synthetic_Dataset_to_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Space/papers/Gallet_Exploring_AI-Based_Satellite_Pose_Estimation_from_Novel_Synthetic_Dataset_to_CVPRW_2024_paper.pdf)]
    * Title: Exploring AI-Based Satellite Pose Estimation: from Novel Synthetic Dataset to In-Depth Performance Evaluation
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Fabien Gallet, Christophe Marabotto, Thomas Chambon
    * Abstract: Vision-based pose estimation using deep learning offers a promising cost effective and versatile solution for relative satellite navigation purposes. Using such a solution in closed loop to control spacecraft position is challenging from validation and performance verification viewpoint because of the complex specification and development process. The validation task entails bridging the gap between the dataset and real-world data. In particular modelling of Sun power and spectrum Earth albedo and atmospheric absence effects is costly to replicate on ground. This article suggests a novel approach to produce synthetic space scene images. Fine statistical balancing is ensured to train and assess pose stimation solutions. A physically based camera model is used. Synthetic images incorporate realistic light flux radiometric properties and texture scatterings. The dataset comprises 120000 images supplemented with masks distance maps celestial body positions and precise camera parameters (dataset publicly available https://www.irt-saintexupery.com/space_rendezvous/ created in the frame of a project called RAPTOR: Robotic and Artificial intelligence Processing Test On Representative target). An analysis method using a dedicated metric library has been developed to help the assessment of the solution performance and robustness. A deeper comprehension of algorithm behavior through distribution law fitting and outlier identification is then facilitated. Finally it is shown that implementing Region-of-Interest (RoI) training can drastically increase the performance of the Convolutional Neural Networks (CNNs) for long-range satellite pose estimation tasks.

count=1
* Adaptive Render-Video Streaming for Virtual Environments
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Lim_Adaptive_Render-Video_Streaming_for_Virtual_Environments_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Lim_Adaptive_Render-Video_Streaming_for_Virtual_Environments_CVPRW_2024_paper.pdf)]
    * Title: Adaptive Render-Video Streaming for Virtual Environments
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jia-Jie Lim, Matthias S. Treder, Aaron Chadha, Yiannis Andreopoulos
    * Abstract: In cloud-based gaming and virtual reality (G&VR) scene content is rendered in a cloud server and streamed as low-latency encoded video to the client device. In this context distributed rendering aims to offload parts of the rendering to the client. An adaptive approach is proposed which dynamically assigns assets to client-side vs. server-side rendering according to varying rendering time and bitrate targets. This is achieved by streaming perceptually-optimized scene control weights to the client which are compressed with a composable autoencoder in conjunction with select video segments. This creates an adaptive render-video (REVI) streaming framework which allows for substantial tradeoffs between client rendering time and the bitrate required to stream visually-lossless video from the server to the client. In order to estimate and control the client rendering time and the required bitrate of each subset of each scene a random-forest based regressor is proposed in conjunction with the use of AIMD (additive-increase/multiplicative-decrease) to ensure predetermined average bitrate or render-time targets are met. Experiments are presented based on typical sets of G&VR scenes rendered in Blender and HEVC low-latency encoding. A key result is that when the client is providing for 50% of the rendering time needed to render the whole scene up to 60% average bitrate saving is achieved versus streaming the entire scene to the client as video.

count=1
* Training Neural Networks to Classify Chiral Nanoparticle Stereopairs Using Weakly Labelled Datasets
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/html/Pattison_Training_Neural_Networks_to_Classify_Chiral_Nanoparticle_Stereopairs_Using_Weakly_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/papers/Pattison_Training_Neural_Networks_to_Classify_Chiral_Nanoparticle_Stereopairs_Using_Weakly_CVPRW_2024_paper.pdf)]
    * Title: Training Neural Networks to Classify Chiral Nanoparticle Stereopairs Using Weakly Labelled Datasets
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Alexander J Pattison, Mary Scott, Peter Ercius, Wolfgang Theis
    * Abstract: As the interest in using machine learning techniques for electron microscopy grows so does the need for labeled datasets and automated labeling strategies. Here we exploit the nature of chirality to weakly label datasets of scanning transmission electron microscope stereopairs of chiral tellurium nanoparticles. We then use these datasets to train convolutional neural networks to classify the handedness of these particles. The methods and results can be applied to other machine learning methods involving weak labeling and stereo imaging.

count=1
* Beyond Respiratory Models: A Physics-enhanced Synthetic Data Generation Method for 2D-3D Deformable Registration
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/html/Lecomte_Beyond_Respiratory_Models_A_Physics-enhanced_Synthetic_Data_Generation_Method_for_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/papers/Lecomte_Beyond_Respiratory_Models_A_Physics-enhanced_Synthetic_Data_Generation_Method_for_CVPRW_2024_paper.pdf)]
    * Title: Beyond Respiratory Models: A Physics-enhanced Synthetic Data Generation Method for 2D-3D Deformable Registration
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: François Lecomte, Pablo Alvarez, Stéphane Cotin, Jean-Louis Dillenseger
    * Abstract: Deformable image registration is crucial in aligning medical images for various clinical applications yet enhancing its efficiency and robustness remains a challenge. Deep Learning methods have shown very promising results for addressing the registration process however acquiring sufficient and diverse data for training remains a hurdle. Synthetic data generation strategies have emerged as a solution yet existing methods often lack versatility and often do not represent well certain types of deformation. This work focuses on X-ray to CT 2D-3D deformable image registration for abdominal interventions where tissue deformation can arise from multiple sources. Due to the scarcity of real-world data for this task synthetic data generation is unavoidable. Unlike previous approaches relying on statistical models extracted from 4DCT images our method leverages a single 3D CT image and physically corrected randomized Displacement Vector Fields (DVF) to enable 2D-3D registration for a variety of clinical scenarios. We believe that our approach represents a significant step towards overcoming data scarcity challenges and enhancing the effectiveness of DL-based DIR in a variety of clinical settings.

count=1
* Multi-angle Consistent Generative NeRF with Additive Angular Margin Momentum Contrastive Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/FAS2024/html/Zou_Multi-angle_Consistent_Generative_NeRF_with_Additive_Angular_Margin_Momentum_Contrastive_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/FAS2024/papers/Zou_Multi-angle_Consistent_Generative_NeRF_with_Additive_Angular_Margin_Momentum_Contrastive_CVPRW_2024_paper.pdf)]
    * Title: Multi-angle Consistent Generative NeRF with Additive Angular Margin Momentum Contrastive Learning
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Hang Zou, Hui Zhang, Yuan Zhang, Hui Ma, Dexin Zhao, Qi Zhang, Qi Li
    * Abstract: The NeRF and GAN-based GIRAFFE algorithm has drawn a lot of attention because of its controllable image production capacity. However the consistency of GIRAFFE rendering results from different perspectives of the same object is not stable. The reasons are twofold: First the optimization goal of GIRAFFE is only concerned with whether the generated image resembles the real image or not. Second GIRAFFE could learn knowledge implicitly to complement the feature deformation of large camera angle change which may introduce uncontrollable generation mode resulting in low consistency of the 3D object. This limits its application in fields such as digital person generation and biometric identity. In this paper We introduce an additional Encoder to form a momentum-based Contrastive Learning with the Discriminator of GAN. In addition we propose an AamNCE loss to train our model which introduces an additive angular margin to the positive sample pairs. In brief the proposed framework could be regarded as a new paradigm of GAN and Contrastive Learning. The Contrastive Learning improves the characteristic expression ability of the model and the AamNCE loss makes the category boundaries of the generated images more explicit. The experimental results demonstrate that our method maintains the consistency of face identity well in the multi-angle rotation of the face dataset.

count=1
* Connecting NeRFs Images and Text
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/INRV/html/Ballerini_Connecting_NeRFs_Images_and_Text_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/INRV/papers/Ballerini_Connecting_NeRFs_Images_and_Text_CVPRW_2024_paper.pdf)]
    * Title: Connecting NeRFs Images and Text
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Francesco Ballerini, Pierluigi Zama Ramirez, Roberto Mirabella, Samuele Salti, Luigi Di Stefano
    * Abstract: Neural Radiance Fields (NeRFs) have emerged as a standard framework for representing 3D scenes and objects introducing a novel data type for information exchange and storage. Concurrently significant progress has been made in multimodal representation learning for text and image data. This paper explores a novel research direction that aims to connect the NeRF modality with other modalities similar to established methodologies for images and text. To this end we propose a simple framework that exploits pre-trained models for NeRF representations alongside multimodal models for text and image processing. Our framework learns a bidirectional mapping between NeRF embeddings and those obtained from corresponding images and text. This mapping unlocks several novel and useful applications including NeRF zero-shot classification and NeRF retrieval from images or text.

count=1
* SLAIM: Robust Dense Neural SLAM for Online Tracking and Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NRI/html/Cartillier_SLAIM_Robust_Dense_Neural_SLAM_for_Online_Tracking_and_Mapping_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NRI/papers/Cartillier_SLAIM_Robust_Dense_Neural_SLAM_for_Online_Tracking_and_Mapping_CVPRW_2024_paper.pdf)]
    * Title: SLAIM: Robust Dense Neural SLAM for Online Tracking and Mapping
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Vincent Cartillier, Grant Schindler, Irfan Essa
    * Abstract: We present SLAIM - Simultaneous Localization and Implicit Mapping. We propose a novel coarse-to-fine tracking model tailored for Neural Radiance Field SLAM (NeRF-SLAM) to achieve state-of-the-art tracking performance. Notably existing NeRF-SLAM systems consistently exhibit inferior tracking performance compared to traditional SLAM algorithms. NeRF-SLAM methods solve camera tracking via image alignment and photometric bundle-adjustment. Such optimization processes are difficult to optimize due to the narrow basin of attraction of the optimization loss in image space (local minima) and the lack of initial correspondences. We mitigate these limitations by implementing a Gaussian pyramid filter on top of NeRF facilitating a coarse-to-fine tracking optimization strategy. Furthermore NeRF systems encounter challenges in converging to the right geometry with limited input views. While prior approaches use a Signed-Distance Function (SDF)-based NeRF and directly supervise SDF values by approximating ground truth SDF through depth measurements this often results in suboptimal geometry. In contrast our method employs a volume density representation and introduces a novel KL regularizer on the ray termination distribution constraining scene geometry to consist of empty space and opaque surfaces. Our solution implements both local and global bundle-adjustment to produce a robust (coarse-to-fine) and accurate (KL regularizer) SLAM solution. We conduct experiments on multiple datasets (ScanNet TUM Replica) showing state-of-the-art results in tracking and in reconstruction accuracy.

count=1
* Unveiling the Ambiguity in Neural Inverse Rendering: A Parameter Compensation Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NRI/html/Kouros_Unveiling_the_Ambiguity_in_Neural_Inverse_Rendering_A_Parameter_Compensation_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/NRI/papers/Kouros_Unveiling_the_Ambiguity_in_Neural_Inverse_Rendering_A_Parameter_Compensation_CVPRW_2024_paper.pdf)]
    * Title: Unveiling the Ambiguity in Neural Inverse Rendering: A Parameter Compensation Analysis
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Georgios Kouros, Minye Wu, Sushruth Nagesh, Xianling Zhang, Tinne Tuytelaars
    * Abstract: Inverse rendering aims to reconstruct the scene properties of objects solely from multiview images. However it is an ill-posed problem prone to producing ambiguous estimations deviating from physically accurate representations. In this paper we utilize Neural Microfacet Fields (NMF) a state-of-the-art neural inverse rendering method to illustrate the inherent ambiguity. We propose an evaluation framework to assess the degree of compensation or interaction between the estimated scene properties aiming to explore the mechanisms behind this ill-posed problem and potential mitigation strategies. Specifically we introduce artificial perturbations to one scene property and examine how adjusting another property can compensate for these perturbations. To facilitate such experiments we introduce a disentangled NMF where material properties are independent. The experimental findings underscore the intrinsic ambiguity present in neural inverse rendering and highlight the importance of providing additional guidance through geometry material and illumination priors.

count=1
* Multi-layer Depth and Epipolar Feature Transformers for 3D Scene Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/SUMO/Shin_Multi-layer_Depth_and_Epipolar_Feature_Transformers_for_3D_Scene_Reconstruction_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/SUMO/Shin_Multi-layer_Depth_and_Epipolar_Feature_Transformers_for_3D_Scene_Reconstruction_CVPRW_2019_paper.pdf)]
    * Title: Multi-layer Depth and Epipolar Feature Transformers for 3D Scene Reconstruction
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Daeyun Shin,  Zhile Ren,  Erik B. Sudderth,  Charless C. Fowlkes
    * Abstract: We tackle the problem of automatically reconstructing a complete 3D model of a scene from a single RGB image. This challenging task requires inferring the shape of both visible and occluded surfaces. Our approach utilizes viewer-centered, multi-layer representation of scene geometry adapted from recent methods for single object shape completion. To improve the accuracy of view-centered representations for complex scenes, we introduce a novel "Epipolar Feature Transformer" that transfers convolutional network features from an input view to other virtual camera viewpoints, and thus better covers the 3D scene geometry. Unlike existing approaches that first detect and localize objects in 3D, and then infer object shape using category-specific models, our approach is fully convolutional, end-to-end differentiable, and avoids the resolution and memory limitations of voxel representations. We demonstrate the advantages of multi-layer depth representations and epipolar feature transformers on the reconstruction of a large database of indoor scenes.

count=1
* Mid-Air: A Multi-Modal Dataset for Extremely Low Altitude Drone Flights
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/UAVision/Fonder_Mid-Air_A_Multi-Modal_Dataset_for_Extremely_Low_Altitude_Drone_Flights_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/UAVision/Fonder_Mid-Air_A_Multi-Modal_Dataset_for_Extremely_Low_Altitude_Drone_Flights_CVPRW_2019_paper.pdf)]
    * Title: Mid-Air: A Multi-Modal Dataset for Extremely Low Altitude Drone Flights
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Michael Fonder,  Marc Van Droogenbroeck
    * Abstract: Flying a drone in unstructured environments with varying conditions is challenging. To help producing better algorithms, we present Mid-Air, a multi-purpose synthetic dataset for low altitude drone flights in unstructured environments. It contains synchronized data of multiple sensors for a total of 54 trajectories and more than 420k video frames simulated in various climate conditions. In this work, we motivate design choices, explain how the data was simulated, and present the content of the dataset. Finally, a benchmark for positioning and a benchmark for image generation tasks show how Mid-Air can be used to set up a standard evaluation method for assessing computer vision algorithms in terms of robustness and generalization. We illustrate this by providing a baseline for depth estimation and by comparing it with results obtained on an existing dataset. The Mid-Air dataset is publicly downloadable, with additional details on the data format and organization, at http://midair.ulg.ac.be.

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
* An Analysis of Focus Sweep for Improved 2D Motion Invariance
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W18/html/Bando_An_Analysis_of_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W18/papers/Bando_An_Analysis_of_2013_CVPR_paper.pdf)]
    * Title: An Analysis of Focus Sweep for Improved 2D Motion Invariance
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Yosuke Bando
    * Abstract: Recent research on computational cameras has shown that it is possible to make motion blur nearly invariant to object speed and 2D (i.e., in-plane) motion direction, with a method called focus sweep that moves the plane of focus through a range of scene depth during exposure. Nevertheless, the focus sweep point-spread function (PSF) slightly changes its shape for different object speeds, deviating from perfect 2D motion invariance. In this paper we perform a time-varying light field analysis of the focus sweep PSF to derive a uniform frequency power assignment for varying motions, leading to a finding that perfect 2D motion invariance is possible in theory, in the limit of infinite exposure time, by designing a custom lens bokeh used for focus sweep. With simulation experiments, we verify that the use of the custom lens bokeh improves motion invariance also in practice, and show that it produces better worst-case performance than the conventional focus sweep method.

count=1
* The Sightfield: Visualizing Computer Vision, and Seeing Its Capacity to "See"
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W17/html/Mann_The_Sightfield_Visualizing_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W17/papers/Mann_The_Sightfield_Visualizing_2014_CVPR_paper.pdf)]
    * Title: The Sightfield: Visualizing Computer Vision, and Seeing Its Capacity to "See"
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Steve Mann
    * Abstract: Computer vision is embedded in toilets, urinals, hand-wash faucets (e.g. Delta Faucet's 128 or 1024 pixel linear arrays), doors, lightswitches, thermostats, and many other objects that "watch" us. Camera-based motion sensing streetlights are installed throughout entire cities, making embedded vision ubiquitous. Technological advancement is leading to increased sensory and computational performance combined with miniaturization that is making vision sensors less visible. In that sense, computer vision is "seeing" better while it is becoming harder for us to see it. I will introduce and describe the concept of a "sight-field", a time-reversed lightﬁeld that can be visualized with time-exposure photography, to make vision (i.e. the capacity to see) visible. In particular, I will describe a special wand that changes color when it is being observed. The wand has an array of light sources that each change color when being watched. The intensity of each light source increases in proportion to the degree to which it is observed. The wand is a surveillometer/sousveillometer array sensing, measuring, and making visible sur/sousveillance. Moving the wand through space, while tracking its exact 3D position in space, makes visible the otherwise invisible "rays of sight" that emenate from cameras. This capacity to sense, measure, and visualize vision, is useful in liability, insurance, safety, and risk assessment, as well as privacy//priveillance assessment, criminology, urban planning, design, and (sur/sous)veillance studies.

count=1
* SparkleVision: Seeing the World Through Random Specular Microfacets
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W10/html/Zhang_SparkleVision_Seeing_the_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W10/papers/Zhang_SparkleVision_Seeing_the_2015_CVPR_paper.pdf)]
    * Title: SparkleVision: Seeing the World Through Random Specular Microfacets
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Zhengdong Zhang, Phillip Isola, Edward H. Adelson
    * Abstract: In this paper, we study the problem of reproducing the world lighting from a single image of an object covered with random specular microfacets on the surface. We show that such reflectors can be interpreted as a randomized mapping from the lighting to the image. Such specular objects have very different optical properties from both diffuse surfaces and smooth specular objects like metals, so we design special imaging system to robustly and effectively photograph them. We present simple yet reliable algorithms to calibrate the proposed system and do the inference. We conduct experiments to verify the correctness of our model assumptions and prove the effectiveness of our pipeline.

count=1
* Multi-view Normal Field Integration for 3D Reconstruction of Mirroring Objects
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Weinmann_Multi-view_Normal_Field_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Weinmann_Multi-view_Normal_Field_2013_ICCV_paper.pdf)]
    * Title: Multi-view Normal Field Integration for 3D Reconstruction of Mirroring Objects
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Michael Weinmann, Aljosa Osep, Roland Ruiters, Reinhard Klein
    * Abstract: In this paper, we present a novel, robust multi-view normal field integration technique for reconstructing the full 3D shape of mirroring objects. We employ a turntablebased setup with several cameras and displays. These are used to display illumination patterns which are reflected by the object surface. The pattern information observed in the cameras enables the calculation of individual volumetric normal fields for each combination of camera, display and turntable angle. As the pattern information might be blurred depending on the surface curvature or due to nonperfect mirroring surface characteristics, we locally adapt the decoding to the finest still resolvable pattern resolution. In complex real-world scenarios, the normal fields contain regions without observations due to occlusions and outliers due to interreflections and noise. Therefore, a robust reconstruction using only normal information is challenging. Via a non-parametric clustering of normal hypotheses derived for each point in the scene, we obtain both the most likely local surface normal and a local surface consistency estimate. This information is utilized in an iterative mincut based variational approach to reconstruct the surface geometry.

count=1
* Blur-Aware Disparity Estimation From Defocus Stereo Images
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Chen_Blur-Aware_Disparity_Estimation_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_Blur-Aware_Disparity_Estimation_ICCV_2015_paper.pdf)]
    * Title: Blur-Aware Disparity Estimation From Defocus Stereo Images
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Ching-Hui Chen, Hui Zhou, Timo Ahonen
    * Abstract: Defocus blur usually causes performance degradation in establishing the visual correspondence between stereo images. We propose a blur-aware disparity estimation method that is robust to the mismatch of focus in stereo images. The relative blur resulting from the mismatch of focus between stereo images is approximated as the difference of the square diameters of the blur kernels. Based on the defocus and stereo model, we propose the relative blur versus disparity (RBD) model that characterizes the relative blur as a second-order polynomial function of disparity. Our method alternates between RBD model update and disparity update in each iteration. The RBD model in return refines the disparity estimation by updating the matching cost and aggregation weight to compensate the mismatch of focus. Experiments using both synthesized and real datasets demonstrate the effectiveness of our proposed algorithm.

count=1
* High Quality Structure From Small Motion for Rolling Shutter Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Im_High_Quality_Structure_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015/papers/Im_High_Quality_Structure_ICCV_2015_paper.pdf)]
    * Title: High Quality Structure From Small Motion for Rolling Shutter Cameras
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: Sunghoon Im, Hyowon Ha, Gyeongmin Choe, Hae-Gon Jeon, Kyungdon Joo, In So Kweon
    * Abstract: We present a practical 3D reconstruction method to obtain a high-quality dense depth map from narrow-baseline image sequences captured by commercial digital cameras, such as DSLRs or mobile phones. Depth estimation from small motion has gained interest as a means of various photographic editing, but important limitations present themselves in the form of depth uncertainty due to a narrow baseline and rolling shutter. To address these problems, we introduce a novel 3D reconstruction method from narrow-baseline image sequences that effectively handles the effects of a rolling shutter that occur from most of commercial digital cameras. Additionally, we present a depth propagation method to fill in the holes associated with the unknown pixels based on our novel geometric guidance model. Both qualitative and quantitative experimental results show that our new algorithm consistently generates better 3D depth maps than those by the state-of-the-art method.

count=1
* Neural EPI-Volume Networks for Shape From Light Field
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Heber_Neural_EPI-Volume_Networks_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Heber_Neural_EPI-Volume_Networks_ICCV_2017_paper.pdf)]
    * Title: Neural EPI-Volume Networks for Shape From Light Field
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Stefan Heber, Wei Yu, Thomas Pock
    * Abstract: This paper presents a novel deep regression network to extract geometric information from Light Field (LF) data. Our network builds upon u-shaped network architectures. Those networks involve two symmetric parts, an encoding and a decoding part. In the first part the network encodes relevant information from the given input into a set of high-level feature maps. In the second part the generated feature maps are then decoded to the desired output. To predict reliable and robust depth information the proposed network examines 3D subsets of the 4D LF called Epipolar Plane Image (EPI) volumes. An important aspect of our network is the use of 3D convolutional layers, that allow to propagate information from two spatial dimensions and one directional dimension of the LF. Compared to previous work this allows for an additional spatial regularization, which reduces depth artifacts and simultaneously maintains clear depth discontinuities. Experimental results show that our approach allows to create high-quality reconstruction results, which outperform current state-of-the-art Shape from Light Field (SfLF) techniques. The main advantage of the proposed approach is the ability to provide those high-quality reconstructions at a low computation time.

count=1
* Click Here: Human-Localized Keypoints as Guidance for Viewpoint Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Szeto_Click_Here_Human-Localized_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Szeto_Click_Here_Human-Localized_ICCV_2017_paper.pdf)]
    * Title: Click Here: Human-Localized Keypoints as Guidance for Viewpoint Estimation
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Ryan Szeto, Jason J. Corso
    * Abstract: We motivate and address a human-in-the-loop variant of the monocular viewpoint estimation task in which the location and class of one semantic object keypoint is available at test time. In order to leverage the keypoint information, we devise a Convolutional Neural Network called Click-Here CNN (CH-CNN) that integrates the keypoint information with activations from the layers that process the image. It transforms the keypoint information into a 2D map that can be used to weigh features from certain parts of the image more heavily. The weighted sum of these spatial features is combined with global image features to provide relevant information to the prediction layers. To train our network, we collect a novel dataset of 3D keypoint annotations on thousands of CAD models, and synthetically render millions of images with 2D keypoint information. On test instances from PASCAL 3D+, our model achieves a mean class accuracy of 90.7%, whereas the state-of-the-art baseline only obtains 85.7% mean class accuracy, justifying our argument for human-in-the-loop inference.

count=1
* Rethinking Reprojection: Closing the Loop for Pose-Aware Shape Reconstruction From a Single Image
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Rethinking_Reprojection_Closing_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Rethinking_Reprojection_Closing_ICCV_2017_paper.pdf)]
    * Title: Rethinking Reprojection: Closing the Loop for Pose-Aware Shape Reconstruction From a Single Image
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Rui Zhu, Hamed Kiani Galoogahi, Chaoyang Wang, Simon Lucey
    * Abstract: An emerging problem in computer vision is the reconstruction of 3D shape and pose of an object from a single image. Hitherto, the problem has been addressed through the application of canonical deep learning methods to regress from the image directly to the 3D shape and pose labels. These approaches, however, are problematic from two perspectives. First, they are minimizing the error between 3D shapes and pose labels - with little thought about the nature of this "label error" when reprojecting the shape back onto the image. Second, they rely on the onerous and ill-posed task of hand labeling natural images with respect to 3D shape and pose. In this paper we define the new task of pose-aware shape reconstruction from a single image, and we advocate that cheaper 2D annotations of objects silhouettes in natural images can be utilized. We design architectures of pose-aware shape reconstruction which reproject the predicted shape back on to the image using the predicted pose. Our evaluation on several object categories demonstrates the superiority of our method for predicting pose-aware 3D shapes from natural images.

count=1
* 3D Morphable Models as Spatial Transformer Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w17/html/Bas_3D_Morphable_Models_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Bas_3D_Morphable_Models_ICCV_2017_paper.pdf)]
    * Title: 3D Morphable Models as Spatial Transformer Networks
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Anil Bas, Patrik Huber, William A. P. Smith, Muhammad Awais, Josef Kittler
    * Abstract: In this paper, we show how a 3D Morphable Model (i.e. a statistical model of the 3D shape of a class of objects such as faces) can be used to spatially transform input data as a module (a 3DMM-STN) within a convolutional neural network. This is an extension of the original spatial transformer network in that we are able to interpret and normalise 3D pose changes and self-occlusions. The trained localisation part of the network is independently useful since it learns to fit a 3D morphable model to a single image. We show that the localiser can be trained using only simple geometric loss functions on a relatively small dataset yet is able to perform robust normalisation on highly uncontrolled images including occlusion, self-occlusion and large pose changes.

count=1
* PVNN: A Neural Network Library for Photometric Vision
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w9/html/Yu_PVNN_A_Neural_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w9/Yu_PVNN_A_Neural_ICCV_2017_paper.pdf)]
    * Title: PVNN: A Neural Network Library for Photometric Vision
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Ye Yu, William A. P. Smith
    * Abstract: In this paper we show how a differentiable, physics-based renderer suitable for photometric vision tasks can be implemented as layers in a deep neural network. The layers include geometric operations for representation transformations, reflectance evaluations with arbitrary numbers of light sources and statistical bidirectional reflectance distribution function (BRDF) models. We make an implementation of these layers available as a neural network library (pvnn) for Theano. The layers can be incorporated into any neural network architecture, allowing parts of the photometric image formation process to be explicitly modelled in a network that is trained end to end via backpropagation. As an exemplar application, we show how to train a network with encoder-decoder architecture that learns to estimate BRDF parameters from a single image in an unsupervised manner.

count=1
* Moulding Humans: Non-Parametric 3D Human Shape Estimation From Single Images
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Gabeur_Moulding_Humans_Non-Parametric_3D_Human_Shape_Estimation_From_Single_Images_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gabeur_Moulding_Humans_Non-Parametric_3D_Human_Shape_Estimation_From_Single_Images_ICCV_2019_paper.pdf)]
    * Title: Moulding Humans: Non-Parametric 3D Human Shape Estimation From Single Images
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Valentin Gabeur,  Jean-Sebastien Franco,  Xavier Martin,  Cordelia Schmid,  Gregory Rogez
    * Abstract: In this paper, we tackle the problem of 3D human shape estimation from single RGB images. While the recent progress in convolutional neural networks has allowed impressive results for 3D human pose estimation, estimating the full 3D shape of a person is still an open issue. Model-based approaches can output precise meshes of naked under-cloth human bodies but fail to estimate details and un-modelled elements such as hair or clothing. On the other hand, non-parametric volumetric approaches can potentially estimate complete shapes but, in practice, they are limited by the resolution of the output grid and cannot produce detailed estimates. In this work, we propose a non-parametric approach that employs a double depth map to represent the 3D shape of a person: a visible depth map and a "hidden" depth map are estimated and combined, to reconstruct the human 3D shape as done with a "mould". This representation through 2D depth maps allows a higher resolution output with a much lower dimension than voxel-based volumetric representations. Additionally, our fully derivable depth-based model allows us to efficiently incorporate a discriminator in an adversarial fashion to improve the accuracy and "humanness" of the 3D output. We train and quantitatively validate our approach on SURREAL and on 3D-HUMANS, a new photorealistic dataset made of semi-synthetic in-house videos annotated with 3D ground truth surfaces.

count=1
* Deep Parametric Indoor Lighting Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Gardner_Deep_Parametric_Indoor_Lighting_Estimation_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gardner_Deep_Parametric_Indoor_Lighting_Estimation_ICCV_2019_paper.pdf)]
    * Title: Deep Parametric Indoor Lighting Estimation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Marc-Andre Gardner,  Yannick Hold-Geoffroy,  Kalyan Sunkavalli,  Christian Gagne,  Jean-Francois Lalonde
    * Abstract: We present a method to estimate lighting from a single image of an indoor scene. Previous work has used an environment map representation that does not account for the localized nature of indoor lighting. Instead, we represent lighting as a set of discrete 3D lights with geometric and photometric parameters. We train a deep neural network to regress these parameters from a single image, on a dataset of environment maps annotated with depth. We propose a differentiable layer to convert these parameters to an environment map to compute our loss; this bypasses the challenge of establishing correspondences between estimated and ground truth lights. We demonstrate, via quantitative and qualitative evaluations, that our representation and training scheme lead to more accurate results compared to previous work, while allowing for more realistic 3D object compositing with spatially-varying lighting.

count=1
* Physics-Based Rendering for Improving Robustness to Rain
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Halder_Physics-Based_Rendering_for_Improving_Robustness_to_Rain_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Halder_Physics-Based_Rendering_for_Improving_Robustness_to_Rain_ICCV_2019_paper.pdf)]
    * Title: Physics-Based Rendering for Improving Robustness to Rain
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Shirsendu Sukanta Halder,  Jean-Francois Lalonde,  Raoul de Charette
    * Abstract: To improve the robustness to rain, we present a physically-based rain rendering pipeline for realistically inserting rain into clear weather images. Our rendering relies on a physical particle simulator, an estimation of the scene lighting and an accurate rain photometric modeling to augment images with arbitrary amount of realistic rain or fog. We validate our rendering with a user study, proving our rain is judged 40% more realistic that state-of-the-art. Using our generated weather augmented Kitti and Cityscapes dataset, we conduct a thorough evaluation of deep object detection and semantic segmentation algorithms and show that their performance decreases in degraded weather, on the order of 15% for object detection and 60% for semantic segmentation. Furthermore, we show refining existing networks with our augmented images improves the robustness of both object detection and semantic segmentation algorithms. We experiment on nuScenes and measure an improvement of 15% for object detection and 35% for semantic segmentation compared to original rainy performance. Augmented databases and code are available on the project page.

count=1
* No Fear of the Dark: Image Retrieval Under Varying Illumination Conditions
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Jenicek_No_Fear_of_the_Dark_Image_Retrieval_Under_Varying_Illumination_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jenicek_No_Fear_of_the_Dark_Image_Retrieval_Under_Varying_Illumination_ICCV_2019_paper.pdf)]
    * Title: No Fear of the Dark: Image Retrieval Under Varying Illumination Conditions
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Tomas Jenicek,  Ondrej Chum
    * Abstract: Image retrieval under varying illumination conditions, such as day and night images, is addressed by image preprocessing, both hand-crafted and learned. Prior to extracting image descriptors by a convolutional neural network, images are photometrically normalised in order to reduce the descriptor sensitivity to illumination changes. We propose a learnable normalisation based on the U-Net architecture, which is trained on a combination of single-camera multi-exposure images and a newly constructed collection of similar views of landmarks during day and night. We experimentally show that both hand-crafted normalisation based on local histogram equalisation and the learnable normalisation outperform standard approaches in varying illumination conditions, while staying on par with the state-of-the-art methods on daylight illumination benchmarks, such as Oxford or Paris datasets.

count=1
* DDSL: Deep Differentiable Simplex Layer for Learning Geometric Signals
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Jiang_DDSL_Deep_Differentiable_Simplex_Layer_for_Learning_Geometric_Signals_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_DDSL_Deep_Differentiable_Simplex_Layer_for_Learning_Geometric_Signals_ICCV_2019_paper.pdf)]
    * Title: DDSL: Deep Differentiable Simplex Layer for Learning Geometric Signals
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Chiyu "Max" Jiang,  Dana Lansigan,  Philip Marcus,  Matthias Niessner
    * Abstract: We present a Deep Differentiable Simplex Layer (DDSL) for neural networks for geometric deep learning. The DDSL is a differentiable layer compatible with deep neural networks for bridging simplex mesh-based geometry representations (point clouds, line mesh, triangular mesh, tetrahedral mesh) with raster images (e.g., 2D/3D grids). The DDSL uses Non-Uniform Fourier Transform (NUFT) to perform differentiable, efficient, anti- aliased rasterization of simplex-based signals. We present a complete theoretical framework for the process as well as an efficient backpropagation algorithm. Compared to previous differentiable renderers and rasterizers, the DDSL generalizes to arbitrary simplex degrees and dimensions. In particular, we explore its applications to 2D shapes and illustrate two applications of this method: (1) mesh editing and optimization guided by neural network outputs, and (2) using DDSL for a differentiable rasterization loss to facilitate end-to-end training of polygon generators. We are able to validate the effectiveness of gradient-based shape optimization with the example of airfoil optimization, and using the differentiable rasterization loss to facilitate end-to-end training, we surpass state of the art for polygonal image segmentation given ground-truth bounding boxes.

count=1
* Meta-Sim: Learning to Generate Synthetic Datasets
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Kar_Meta-Sim_Learning_to_Generate_Synthetic_Datasets_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kar_Meta-Sim_Learning_to_Generate_Synthetic_Datasets_ICCV_2019_paper.pdf)]
    * Title: Meta-Sim: Learning to Generate Synthetic Datasets
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Amlan Kar,  Aayush Prakash,  Ming-Yu Liu,  Eric Cameracci,  Justin Yuan,  Matt Rusiniak,  David Acuna,  Antonio Torralba,  Sanja Fidler
    * Abstract: Training models to high-end performance requires availability of large labeled datasets, which are expensive to get. The goal of our work is to automatically synthesize labeled datasets that are relevant for a downstream task. We propose Meta-Sim, which learns a generative model of synthetic scenes, and obtain images as well as its corresponding ground-truth via a graphics engine. We parametrize our dataset generator with a neural network, which learns to modify attributes of scene graphs obtained from probabilistic scene grammars, so as to minimize the distribution gap between its rendered outputs and target data. If the real dataset comes with a small labeled validation set, we additionally aim to optimize a meta-objective, i.e. downstream task performance. Experiments show that the proposed method can greatly improve content generation quality over a human-engineered probabilistic scene grammar, both qualitatively and quantitatively as measured by performance on a downstream task.

count=1
* Soft Rasterizer: A Differentiable Renderer for Image-Based 3D Reasoning
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf)]
    * Title: Soft Rasterizer: A Differentiable Renderer for Image-Based 3D Reasoning
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Shichen Liu,  Tianye Li,  Weikai Chen,  Hao Li
    * Abstract: Rendering bridges the gap between 2D vision and 3D scenes by simulating the physical process of image formation. By inverting such renderer, one can think of a learning approach to infer 3D information from 2D images. However, standard graphics renderers involve a fundamental discretization step called rasterization, which prevents the rendering process to be differentiable, hence able to be learned. Unlike the state-of-the-art differentiable renderers, which only approximate the rendering gradient in the back propagation, we propose a truly differentiable rendering framework that is able to (1) directly render colorized mesh using differentiable functions and (2) back-propagate efficient supervision signals to mesh vertices and their attributes from various forms of image representations, including silhouette, shading and color images. The key to our framework is a novel formulation that views rendering as an aggregation function that fuses the probabilistic contributions of all mesh triangles with respect to the rendered pixels. Such formulation enables our framework to flow gradients to the occluded and far-range vertices, which cannot be achieved by the previous state-of-the-arts. We show that by using the proposed renderer, one can achieve significant improvement in 3D unsupervised single-view reconstruction both qualitatively and quantitatively. Experiments also demonstrate that our approach is able to handle the challenging tasks in image-based shape fitting, which remain nontrivial to existing differentiable renderers. Code is available at https://github.com/ShichenLiu/SoftRas.

count=1
* GraphX-Convolution for Point Cloud Deformation in 2D-to-3D Conversion
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_GraphX-Convolution_for_Point_Cloud_Deformation_in_2D-to-3D_Conversion_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_GraphX-Convolution_for_Point_Cloud_Deformation_in_2D-to-3D_Conversion_ICCV_2019_paper.pdf)]
    * Title: GraphX-Convolution for Point Cloud Deformation in 2D-to-3D Conversion
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Anh-Duc Nguyen,  Seonghwa Choi,  Woojae Kim,  Sanghoon Lee
    * Abstract: In this paper, we present a novel deep method to reconstruct a point cloud of an object from a single still image. Prior arts in the field struggle to reconstruct an accurate and scalable 3D model due to either the inefficient and expensive 3D representations, the dependency between the output and number of model parameters or the lack of a suitable computing operation. We propose to overcome these by deforming a random point cloud to the object shape through two steps: feature blending and deformation. In the first step, the global and point-specific shape features extracted from a 2D object image are blended with the encoded feature of a randomly generated point cloud, and then this mixture is sent to the deformation step to produce the final representative point set of the object. In the deformation process, we introduce a new layer termed as GraphX that considers the inter-relationship between points like common graph convolutions but operates on unordered sets. Moreover, with a simple trick, the proposed model can generate an arbitrary-sized point cloud, which is the first deep method to do so. Extensive experiments verify that we outperform existing models and halve the state-of-the-art distance score in single image 3D reconstruction.

count=1
* HoloGAN: Unsupervised Learning of 3D Representations From Natural Images
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen-Phuoc_HoloGAN_Unsupervised_Learning_of_3D_Representations_From_Natural_Images_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen-Phuoc_HoloGAN_Unsupervised_Learning_of_3D_Representations_From_Natural_Images_ICCV_2019_paper.pdf)]
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Nguyen-Phuoc_HoloGAN_Unsupervised_Learning_of_3D_Representations_From_Natural_Images_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Nguyen-Phuoc_HoloGAN_Unsupervised_Learning_of_3D_Representations_From_Natural_Images_ICCVW_2019_paper.pdf)]
    * Title: HoloGAN: Unsupervised Learning of 3D Representations From Natural Images
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Thu Nguyen-Phuoc,  Chuan Li,  Lucas Theis,  Christian Richardt,  Yong-Liang Yang
    * Abstract: We propose a novel generative adversarial network (GAN) for the task of unsupervised learning of 3D representations from natural images. Most generative models rely on 2D kernels to generate images and make few assumptions about the 3D world. These models therefore tend to create blurry images or artefacts in tasks that require a strong 3D understanding, such as novel-view synthesis. HoloGAN instead learns a 3D representation of the world, and to render this representation in a realistic manner. Unlike other GANs, HoloGAN provides explicit control over the pose of generated objects through rigid-body transformations of the learnt 3D features. Our experiments show that using explicit 3D features enables HoloGAN to disentangle 3D pose and identity, which is further decomposed into shape and appearance, while still being able to generate images with similar or higher visual quality than other generative models. HoloGAN can be trained end-to-end from unlabelled 2D images only. Particularly, we do not require pose labels, 3D shapes, or multiple views of the same objects. This shows that HoloGAN is the first generative model that learns 3D representations from natural images in an entirely unsupervised manner.

count=1
* 3D Scene Reconstruction With Multi-Layer Depth and Epipolar Transformers
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Shin_3D_Scene_Reconstruction_With_Multi-Layer_Depth_and_Epipolar_Transformers_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shin_3D_Scene_Reconstruction_With_Multi-Layer_Depth_and_Epipolar_Transformers_ICCV_2019_paper.pdf)]
    * Title: 3D Scene Reconstruction With Multi-Layer Depth and Epipolar Transformers
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Daeyun Shin,  Zhile Ren,  Erik B. Sudderth,  Charless C. Fowlkes
    * Abstract: We tackle the problem of automatically reconstructing a complete 3D model of a scene from a single RGB image. This challenging task requires inferring the shape of both visible and occluded surfaces. Our approach utilizes viewer-centered, multi-layer representation of scene geometry adapted from recent methods for single object shape completion. To improve the accuracy of view-centered representations for complex scenes, we introduce a novel "Epipolar Feature Transformer" that transfers convolutional network features from an input view to other virtual camera viewpoints, and thus better covers the 3D scene geometry. Unlike existing approaches that first detect and localize objects in 3D, and then infer object shape using category-specific models, our approach is fully convolutional, end-to-end differentiable, and avoids the resolution and memory limitations of voxel representations. We demonstrate the advantages of multi-layer depth representations and epipolar feature transformers on the reconstruction of a large database of indoor scenes.

count=1
* Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Wen_Pixel2Mesh_Multi-View_3D_Mesh_Generation_via_Deformation_ICCV_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wen_Pixel2Mesh_Multi-View_3D_Mesh_Generation_via_Deformation_ICCV_2019_paper.pdf)]
    * Title: Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Chao Wen,  Yinda Zhang,  Zhuwen Li,  Yanwei Fu
    * Abstract: We study the problem of shape generation in 3D mesh representation from a few color images with known camera poses. While many previous works learn to hallucinate the shape directly from priors, we resort to further improving the shape quality by leveraging cross-view information with a graph convolutional network. Instead of building a direct mapping function from images to 3D shape, our model learns to predict series of deformations to improve a coarse shape iteratively. Inspired by traditional multiple view geometry methods, our network samples nearby area around the initial mesh's vertex locations and reasons an optimal deformation using perceptual feature statistics built from multiple input images. Extensive experiments show that our model produces accurate 3D shape that are not only visually plausible from the input perspectives, but also well aligned to arbitrary viewpoints. With the help of physically driven architecture, our model also exhibits generalization capability across different semantic categories, number of input images, and quality of mesh initialization.

count=1
* Learning To Reduce Defocus Blur by Realistically Modeling Dual-Pixel Data
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Abuolaim_Learning_To_Reduce_Defocus_Blur_by_Realistically_Modeling_Dual-Pixel_Data_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Abuolaim_Learning_To_Reduce_Defocus_Blur_by_Realistically_Modeling_Dual-Pixel_Data_ICCV_2021_paper.pdf)]
    * Title: Learning To Reduce Defocus Blur by Realistically Modeling Dual-Pixel Data
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Abdullah Abuolaim, Mauricio Delbracio, Damien Kelly, Michael S. Brown, Peyman Milanfar
    * Abstract: Recent work has shown impressive results on data-driven defocus deblurring using the two-image views available on modern dual-pixel (DP) sensors. One significant challenge in this line of research is access to DP data. Despite many cameras having DP sensors, only a limited number provide access to the low-level DP sensor images. In addition, capturing training data for defocus deblurring involves a time-consuming and tedious setup requiring the camera's aperture to be adjusted. Some cameras with DP sensors (e.g., smartphones) do not have adjustable apertures, further limiting the ability to produce the necessary training data. We address the data capture bottleneck by proposing a procedure to generate realistic DP data synthetically. Our synthesis approach mimics the optical image formation found on DP sensors and can be applied to virtual scenes rendered with standard computer software. Leveraging these realistic synthetic DP images, we introduce a recurrent convolutional network (RCN) architecture that improves deblurring results and is suitable for use with single-frame and multi-frame data (e.g., video) captured by DP sensors. Finally, we show that our synthetic DP data is useful for training DNN models targeting video deblurring applications where access to DP data remains challenging.

count=1
* Single-Shot Hyperspectral-Depth Imaging With Learned Diffractive Optics
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Baek_Single-Shot_Hyperspectral-Depth_Imaging_With_Learned_Diffractive_Optics_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Baek_Single-Shot_Hyperspectral-Depth_Imaging_With_Learned_Diffractive_Optics_ICCV_2021_paper.pdf)]
    * Title: Single-Shot Hyperspectral-Depth Imaging With Learned Diffractive Optics
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Seung-Hwan Baek, Hayato Ikoma, Daniel S. Jeon, Yuqi Li, Wolfgang Heidrich, Gordon Wetzstein, Min H. Kim
    * Abstract: Imaging depth and spectrum have been extensively studied in isolation from each other for decades. Recently, hyperspectral-depth (HS-D) imaging emerges to capture both information simultaneously by combining two different imaging systems; one for depth, the other for spectrum. While being accurate, this combinational approach induces increased form factor, cost, capture time, and alignment/registration problems. In this work, departing from the combinational principle, we propose a compact single-shot monocular HS-D imaging method. Our method uses a diffractive optical element (DOE), the point spread function of which changes with respect to both depth and spectrum. This enables us to reconstruct spectrum and depth from a single captured image. To this end, we develop a differentiable simulator and a neural-network-based reconstruction method that are jointly optimized via automatic differentiation. To facilitate learning the DOE, we present a first HS-D dataset by building a benchtop HS-D imager that acquires high-quality ground truth. We evaluate our method with synthetic and real experiments by building an experimental prototype and achieve state-of-the-art HS-D imaging results.

count=1
* Mip-NeRF
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.pdf)]
    * Title: Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, Pratul P. Srinivasan
    * Abstract: The rendering procedure used by neural radiance fields (NeRF) samples a scene with a single ray per pixel and may therefore produce renderings that are excessively blurred or aliased when training or testing images observe scene content at different resolutions. The straightforward solution of supersampling by rendering with multiple rays per pixel is impractical for NeRF, because rendering each ray requires querying a multilayer perceptron hundreds of times. Our solution, which we call "mip-NeRF" (a la "mipmap"), extends NeRF to represent the scene at a continuously-valued scale. By efficiently rendering anti-aliased conical frustums instead of rays, mip-NeRF reduces objectionable aliasing artifacts and significantly improves NeRF's ability to represent fine details, while also being 7% faster than NeRF and half the size. Compared to NeRF, mip-NeRF reduces average error rates by 17% on the dataset presented with NeRF and by 60% on a challenging multiscale variant of that dataset that we present. Mip-NeRF is also able to match the accuracy of a brute-force supersampled NeRF on our multiscale dataset while being 22x faster.

count=1
* STR-GQN: Scene Representation and Rendering for Unknown Cameras Based on Spatial Transformation Routing
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_STR-GQN_Scene_Representation_and_Rendering_for_Unknown_Cameras_Based_on_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_STR-GQN_Scene_Representation_and_Rendering_for_Unknown_Cameras_Based_on_ICCV_2021_paper.pdf)]
    * Title: STR-GQN: Scene Representation and Rendering for Unknown Cameras Based on Spatial Transformation Routing
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Wen-Cheng Chen, Min-Chun Hu, Chu-Song Chen
    * Abstract: Geometry-aware modules are widely applied in recent deep learning architectures for scene representation and rendering. However, these modules require intrinsic camera information that might not be obtained accurately. In this paper, we propose a Spatial Transformation Routing (STR) mechanism to model the spatial properties without applying any geometric prior. The STR mechanism treats the spatial transformation as the message passing process, and the relation between the view poses and the routing weights is modeled by an end-to-end trainable neural network. Besides, an Occupancy Concept Mapping (OCM) framework is proposed to provide explainable rationals for scene-fusion processes. We conducted experiments on several datasets and show that the proposed STR mechanism improves the performance of the Generative Query Network (GQN). The visualization results reveal that the routing process can pass the observed information from one location of some view to the associated location in the other view, which demonstrates the advantage of the proposed model in terms of spatial cognition.

count=1
* Differentiable Surface Rendering via Non-Differentiable Sampling
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Cole_Differentiable_Surface_Rendering_via_Non-Differentiable_Sampling_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Cole_Differentiable_Surface_Rendering_via_Non-Differentiable_Sampling_ICCV_2021_paper.pdf)]
    * Title: Differentiable Surface Rendering via Non-Differentiable Sampling
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Forrester Cole, Kyle Genova, Avneesh Sud, Daniel Vlasic, Zhoutong Zhang
    * Abstract: We present a method for differentiable rendering of 3D surfaces that supports both explicit and implicit representations, provides derivatives at occlusion boundaries, and is fast and simple to implement. The method first samples the surface using non-differentiable rasterization, then applies differentiable, depth-aware point splatting to produce the final image. Our approach requires no differentiable meshing or rasterization steps, making it efficient for large 3D models and applicable to isosurfaces extracted from implicit surface definitions. We demonstrate the effectiveness of our method for implicit-, mesh-, and parametric-surface-based inverse rendering and neural-network training applications. In particular, we show for the first time efficient, differentiable rendering of an isosurface extracted from a neural radiance field (NeRF), and demonstrate surface-based, rather than volume-based, rendering of a NeRF.

count=1
* Unconstrained Scene Generation With Locally Conditioned Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/DeVries_Unconstrained_Scene_Generation_With_Locally_Conditioned_Radiance_Fields_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/DeVries_Unconstrained_Scene_Generation_With_Locally_Conditioned_Radiance_Fields_ICCV_2021_paper.pdf)]
    * Title: Unconstrained Scene Generation With Locally Conditioned Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Terrance DeVries, Miguel Angel Bautista, Nitish Srivastava, Graham W. Taylor, Joshua M. Susskind
    * Abstract: We tackle the challenge of learning a distribution over complex, realistic, indoor scenes. In this paper, we introduce Generative Scene Networks (GSN), which learns to decompose scenes into a collection of many local radiance fields that can be rendered from a free moving camera. Our model can be used as a prior to generate new scenes, or to complete a scene given only sparse 2D observations. Recent work has shown that generative models of radiance fields can capture properties such as multi-view consistency and view-dependent lighting. However, these models are specialized for constrained viewing of single objects, such as cars or faces. Due to the size and complexity of realistic indoor environments, existing models lack the representational capacity to adequately capture them. Our decomposition scheme scales to larger and more complex scenes while preserving details and diversity, and the learned prior enables high-quality rendering from view-points that are significantly different from observed viewpoints. When compared to existing models, GSN produces quantitatively higher quality scene renderings across several different scene datasets.

count=1
* MVTN: Multi-View Transformation Network for 3D Shape Recognition
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Hamdi_MVTN_Multi-View_Transformation_Network_for_3D_Shape_Recognition_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Hamdi_MVTN_Multi-View_Transformation_Network_for_3D_Shape_Recognition_ICCV_2021_paper.pdf)]
    * Title: MVTN: Multi-View Transformation Network for 3D Shape Recognition
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Abdullah Hamdi, Silvio Giancola, Bernard Ghanem
    * Abstract: Multi-view projection methods have demonstrated their ability to reach state-of-the-art performance on 3D shape recognition. Those methods learn different ways to aggregate information from multiple views. However, the camera view-points for those views tend to be heuristically set and fixed for all shapes. To circumvent the lack of dynamism of current multi-view methods, we propose to learn those view-points. In particular, we introduce the Multi-View Transformation Network (MVTN) that regresses optimal view-points for 3D shape recognition, building upon advances in differentiable rendering. As a result, MVTN can be trained end-to-end along with any multi-view network for 3D shape classification. We integrate MVTN in a novel adaptive multi-view pipeline that can render either 3D meshes or point clouds. MVTN exhibits clear performance gains in the tasks of 3D shape classification and 3D shape retrieval without the need for extra training supervision. In these tasks, MVTN achieves state-of-the-art performance on ModelNet40, ShapeNet Core55, and the most recent and realistic ScanObjectNN dataset (up to 6 % improvement). Interestingly, we also show that MVTN can provide network robustness against rotation and occlusion in the 3D domain.

count=1
* GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_GANcraft_Unsupervised_3D_Neural_Rendering_of_Minecraft_Worlds_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Hao_GANcraft_Unsupervised_3D_Neural_Rendering_of_Minecraft_Worlds_ICCV_2021_paper.pdf)]
    * Title: GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zekun Hao, Arun Mallya, Serge Belongie, Ming-Yu Liu
    * Abstract: We present GANcraft, an unsupervised neural rendering framework for generating photorealistic images of large 3D block worlds such as those created in Minecraft. Our method takes a semantic block world as input, where each block is assigned a semantic label such as dirt, grass, or water. We represent the world as a continuous volumetric function and train our model to render view-consistent photorealistic images for a user-controlled camera. In the absence of paired ground truth real images for the block world, we devise a training technique based on pseudo-ground truth and adversarial training. This stands in contrast to prior work on neural rendering for view synthesis, which requires ground truth images to estimate scene geometry and view-dependent appearance. In addition to camera trajectory, GANcraft allows user control over both scene semantics and output style. Experimental results with comparison to strong baselines show the effectiveness of GANcraft on this novel task of photorealistic 3D block world synthesis.

count=1
* ARCH++: Animation-Ready Clothed Human Reconstruction Revisited
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/He_ARCH_Animation-Ready_Clothed_Human_Reconstruction_Revisited_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/He_ARCH_Animation-Ready_Clothed_Human_Reconstruction_Revisited_ICCV_2021_paper.pdf)]
    * Title: ARCH++: Animation-Ready Clothed Human Reconstruction Revisited
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Tong He, Yuanlu Xu, Shunsuke Saito, Stefano Soatto, Tony Tung
    * Abstract: We present ARCH++, an image-based method to reconstruct 3D avatars with arbitrary clothing styles. Our reconstructed avatars are animation-ready and highly realistic, in both the visible regions from input views and the unseen regions. While prior work shows great promise of reconstructing animatable clothed humans with various topologies, we observe that there exist fundamental limitations resulting in sub-optimal reconstruction quality. In this paper, we revisit the major steps of image-based avatar reconstruction and address the limitations with ARCH++. First, we introduce an end-to-end point based geometry encoder to better describe the semantics of the underlying 3D human body, in replacement of previous hand-crafted features. Second, in order to address the occupancy ambiguity caused by topological changes of clothed humans in the canonical pose, we propose a co-supervising framework with cross-space consistency to jointly estimate the occupancy in both the posed and canonical spaces. Last, we use image-to-image translation networks to further refine detailed geometry and texture on the reconstructed surface, which improves the fidelity and consistency across arbitrary viewpoints. In the experiments, we demonstrate improvements over the state of the art on both public benchmarks and user studies in reconstruction quality and realism.

count=1
* Baking Neural Radiance Fields for Real-Time View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Hedman_Baking_Neural_Radiance_Fields_for_Real-Time_View_Synthesis_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Hedman_Baking_Neural_Radiance_Fields_for_Real-Time_View_Synthesis_ICCV_2021_paper.pdf)]
    * Title: Baking Neural Radiance Fields for Real-Time View Synthesis
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall, Jonathan T. Barron, Paul Debevec
    * Abstract: Neural volumetric representations such as Neural Radiance Fields (NeRF) have emerged as a compelling technique for learning to represent 3D scenes from images with the goal of rendering photorealistic images of the scene from unobserved viewpoints. However, NeRF's computational requirements are prohibitive for real-time applications: rendering views from a trained NeRF requires querying a multilayer perceptron (MLP) hundreds of times per ray. We present a method to train a NeRF, then precompute and store (i.e. ""bake"") it as a novel representation called a Sparse Neural Radiance Grid (SNeRG) that enables real-time rendering on commodity hardware. To achieve this, we introduce 1) a reformulation of NeRF's architecture, and 2) a sparse voxel grid representation with learned feature vectors. The resulting scene representation retains NeRF's ability to render fine geometric details and view-dependent appearance, is compact (averaging less than 90 MB per scene), and can be rendered in real-time (higher than 30 frames per second on a laptop GPU). Actual screen captures are shown in our video.

count=1
* RePOSE: Fast 6D Object Pose Refinement via Deep Texture Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Iwase_RePOSE_Fast_6D_Object_Pose_Refinement_via_Deep_Texture_Rendering_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Iwase_RePOSE_Fast_6D_Object_Pose_Refinement_via_Deep_Texture_Rendering_ICCV_2021_paper.pdf)]
    * Title: RePOSE: Fast 6D Object Pose Refinement via Deep Texture Rendering
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Shun Iwase, Xingyu Liu, Rawal Khirodkar, Rio Yokota, Kris M. Kitani
    * Abstract: We present RePOSE, a fast iterative refinement method for 6D object pose estimation. Prior methods perform refinement by feeding zoomed-in input and rendered RGB images into a CNN and directly regressing an update of a refined pose. Their runtime is slow due to the computational cost of CNN, which is especially prominent in multiple-object pose refinement. To overcome this problem, RePOSE leverages image rendering for fast feature extraction using a 3D model with a learnable texture. We call this deep texture rendering, which uses a shallow multi-layer perceptron to directly regress a view-invariant image representation of an object. Furthermore, we utilize differentiable Levenberg-Marquard (LM) optimization to refine a pose fast and accurately by minimizing the feature-metric error between the input and rendered image representations without the need of zooming in. These image representations are trained such that differentiable LM optimization converges within few iterations. Consequently, RePOSE runs at 92 FPS and achieves state-of-the-art accuracy of 51.6% on the Occlusion LineMOD dataset - a 4.1% absolute improvement over the prior art, and comparable result on the YCB-Video dataset with a much faster runtime. The code is available at https://github.com/sh8/repose.

count=1
* Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Jain_Putting_NeRF_on_a_Diet_Semantically_Consistent_Few-Shot_View_Synthesis_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Jain_Putting_NeRF_on_a_Diet_Semantically_Consistent_Few-Shot_View_Synthesis_ICCV_2021_paper.pdf)]
    * Title: Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Ajay Jain, Matthew Tancik, Pieter Abbeel
    * Abstract: We present DietNeRF, a 3D neural scene representation estimated from a few images. Neural Radiance Fields (NeRF) learn a continuous volumetric representation of a scene through multi-view consistency, and can be rendered from novel viewpoints by ray casting. While NeRF has an impressive ability to reconstruct geometry and fine details given many images, up to 100 for challenging 360 degree scenes, it often finds a degenerate solution to its image reconstruction objective when only a few input views are available. To improve few-shot quality, we propose DietNeRF. We introduce an auxiliary semantic consistency loss that encourages realistic renderings at novel poses. DietNeRF is trained on individual scenes to (1) correctly render given input views from the same pose, and (2) match high-level semantic attributes across different, random poses. Our semantic loss allows us to supervise DietNeRF from arbitrary poses. We extract these semantics using a pre-trained visual encoder such as CLIP, a Vision Transformer trained on hundreds of millions of diverse single-view, 2D photographs mined from the web with natural language supervision. In experiments, DietNeRF improves the perceptual quality of few-shot view synthesis when learned from scratch, can render novel views with as few as one observed image when pre-trained on a multi-view dataset, and produces plausible completions of completely unobserved regions. Our project website is available at https://www.ajayj.com/dietnerf.

count=1
* CodeNeRF: Disentangled Neural Radiance Fields for Object Categories
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Jang_CodeNeRF_Disentangled_Neural_Radiance_Fields_for_Object_Categories_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Jang_CodeNeRF_Disentangled_Neural_Radiance_Fields_for_Object_Categories_ICCV_2021_paper.pdf)]
    * Title: CodeNeRF: Disentangled Neural Radiance Fields for Object Categories
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Wonbong Jang, Lourdes Agapito
    * Abstract: CodeNeRF is an implicit 3D neural representation that learns the variation of object shapes and textures across a category and can be trained, from a set of posed images, to synthesize novel views of unseen objects. Unlike the original NeRF, which is scene specific, CodeNeRF learns to disentangle shape and texture by learning separate embeddings. At test time, given a single unposed image of an unseen object, CodeNeRF jointly estimates camera viewpoint, and shape and appearance codes via optimization. Unseen objects can be reconstructed from a single image, and then rendered from new viewpoints or their shape and texture edited by varying the latent codes. We conduct experiments on the SRN benchmark, which show that CodeNeRF generalises well to unseen objects and achieves on-par performance with methods that require known camera pose at test time. Our results on real-world images demonstrate that CodeNeRF can bridge the sim-to-real gap. Project page: https://github.com/wayne1123/code-nerf

count=1
* SIMstack: A Generative Shape and Instance Model for Unordered Object Stacks
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Landgraf_SIMstack_A_Generative_Shape_and_Instance_Model_for_Unordered_Object_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Landgraf_SIMstack_A_Generative_Shape_and_Instance_Model_for_Unordered_Object_ICCV_2021_paper.pdf)]
    * Title: SIMstack: A Generative Shape and Instance Model for Unordered Object Stacks
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Zoe Landgraf, Raluca Scona, Tristan Laidlow, Stephen James, Stefan Leutenegger, Andrew J. Davison
    * Abstract: By estimating 3D shape and instances from a single view, we can capture information about the environment quickly, without the need for comprehensive scanning and multi-view fusion. Solving this task for composite scenes (such as object stacks) is challenging: occluded areas are not only ambiguous in shape but also in instance segmentation; multiple decompositions could be valid. We observe that physics constrains decomposition as well as shape in occluded regions and hypothesise that a latent space learned from scenes built under physics simulation can serve as a prior to better predict shape and instances in occluded regions. To this end we propose SIMstack, a depth-conditioned Variational Auto-Encoder (VAE), trained on a dataset of objects stacked under physics simulation. We formulate instance segmentation as a center voting task which allows for class-agnostic detection and doesn't require setting the maximum number of objects in the scene. At test time, our model can generate 3D shape and instance segmentation from a single depth view, probabilistically sampling proposals for the occluded region from the learned latent space. We argue that this method has practical applications in providing robots some of the ability humans have to make rapid intuitive inferences of partially observed scenes. We demonstrate an application for precise (non-disruptive) object grasping of unknown objects from a single depth view.

count=1
* MINE: Towards Continuous Depth MPI With NeRF for Novel View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Li_MINE_Towards_Continuous_Depth_MPI_With_NeRF_for_Novel_View_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_MINE_Towards_Continuous_Depth_MPI_With_NeRF_for_Novel_View_ICCV_2021_paper.pdf)]
    * Title: MINE: Towards Continuous Depth MPI With NeRF for Novel View Synthesis
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jiaxin Li, Zijian Feng, Qi She, Henghui Ding, Changhu Wang, Gim Hee Lee
    * Abstract: In this paper, we propose MINE to perform novel view synthesis and depth estimation via dense 3D reconstruction from a single image. Our approach is a continuous depth generalization of the Multiplane Images (MPI) by introducing the NEural radiance fields (NeRF). Given a single image as input, MINE predicts a 4-channel image (RGB and volume density) at arbitrary depth values to jointly reconstruct the camera frustum and fill in occluded contents. The reconstructed and inpainted frustum can then be easily rendered into novel RGB or depth views using differentiable rendering. Extensive experiments on RealEstate10K, KITTI and Flowers Light Fields show that our MINE outperforms state-of-the-art by a large margin in novel view synthesis. We also achieve competitive results in depth estimation on iBims-1 and NYU-v2 without annotated depth supervision. Our source code is available at https://github.com/vincentfung13/MINE

count=1
* BARF: Bundle-Adjusting Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Lin_BARF_Bundle-Adjusting_Neural_Radiance_Fields_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_BARF_Bundle-Adjusting_Neural_Radiance_Fields_ICCV_2021_paper.pdf)]
    * Title: BARF: Bundle-Adjusting Neural Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, Simon Lucey
    * Abstract: Neural Radiance Fields (NeRF) have recently gained a surge of interest within the computer vision community for its power to synthesize photorealistic novel views of real-world scenes. One limitation of NeRF, however, is its requirement of known camera poses to learn the scene representations. In this paper, we propose Bundle-Adjusting Neural Radiance Fields (BARF) for training NeRF from imperfect camera poses -- the joint problem of learning neural 3D representations and registering camera frames. We establish a theoretical connection to classical planar image registration and show that coarse-to-fine registration is also applicable to NeRF. Furthermore, we demonstrate mathematically that positional encoding has a direct impact on the basin of attraction for registration with a synthesis-based objective. Experiments on synthetic and real-world data show that BARF can effectively optimize the neural scene representations and resolve large camera pose misalignment at the same time. This enables applications of view synthesis and localization of video sequences from unknown camera poses, opening up new avenues for visual localization systems (e.g. SLAM) towards sequential registration with NeRF.

count=1
* Universal and Flexible Optical Aberration Correction Using Deep-Prior Based Deconvolution
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Universal_and_Flexible_Optical_Aberration_Correction_Using_Deep-Prior_Based_Deconvolution_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Universal_and_Flexible_Optical_Aberration_Correction_Using_Deep-Prior_Based_Deconvolution_ICCV_2021_paper.pdf)]
    * Title: Universal and Flexible Optical Aberration Correction Using Deep-Prior Based Deconvolution
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Xiu Li, Jinli Suo, Weihang Zhang, Xin Yuan, Qionghai Dai
    * Abstract: High quality imaging usually requires bulky and expensive lenses to compensate geometric and chromatic aberrations. This poses high constraints on the optical hash or low cost applications. Although one can utilize algorithmic reconstruction to remove the artifacts of low-end lenses, the degeneration from optical aberrations is spatially varying and the computation has to trade off efficiency for performance. For example, we need to conduct patch-wise optimization or train a large set of local deep neural networks to achieve high reconstruction performance across the whole image. In this paper, we propose a PSF aware plug-and-play deep network, which takes the aberrant image and PSF map as input and produces the latent high quality version via incorporating lens-specific deep priors, thus leading to a universal and flexible optical aberration correction method. Specifically, we pre-train a base model from a set of diverse lenses and then adapt it to a given lens by quickly refining the parameters, which largely alleviates the time and memory consumption of model learning. The approach is of high efficiency in both training and testing stages. Extensive results verify the promising applications of our proposed approach for compact low-end cameras.

count=1
* Unsupervised Non-Rigid Image Distortion Removal via Grid Deformation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Unsupervised_Non-Rigid_Image_Distortion_Removal_via_Grid_Deformation_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Unsupervised_Non-Rigid_Image_Distortion_Removal_via_Grid_Deformation_ICCV_2021_paper.pdf)]
    * Title: Unsupervised Non-Rigid Image Distortion Removal via Grid Deformation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Nianyi Li, Simron Thapa, Cameron Whyte, Albert W. Reed, Suren Jayasuriya, Jinwei Ye
    * Abstract: Many computer vision problems face difficulties when imaging through turbulent refractive media (e.g., air and water) due to the refraction and scattering of light. These effects cause geometric distortion that requires either handcrafted physical priors or supervised learning methods to remove. In this paper, we present a novel unsupervised network to recover the latent distortion-free image. The key idea is to model non-rigid distortions as deformable grids. Our network consists of a grid deformer that estimates the distortion field and an image generator that outputs the distortion-free image. By leveraging the positional encoding operator, we can simplify the network structure while maintaining fine spatial details in the recovered images. Our method doesn't need to be trained on labeled data and has good transferability across various turbulent image datasets with different types of distortions. Extensive experiments on both simulated and real-captured turbulent images demonstrate that our method can remove both air and water distortions without much customization.

count=1
* PX-NET: Simple and Efficient Pixel-Wise Training of Photometric Stereo Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Logothetis_PX-NET_Simple_and_Efficient_Pixel-Wise_Training_of_Photometric_Stereo_Networks_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Logothetis_PX-NET_Simple_and_Efficient_Pixel-Wise_Training_of_Photometric_Stereo_Networks_ICCV_2021_paper.pdf)]
    * Title: PX-NET: Simple and Efficient Pixel-Wise Training of Photometric Stereo Networks
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Fotios Logothetis, Ignas Budvytis, Roberto Mecca, Roberto Cipolla
    * Abstract: Retrieving accurate 3D reconstructions of objects from the way they reflect light is a very challenging task in computer vision. Despite more than four decades since the definition of the Photometric Stereo problem, most of the literature has had limited success when global illumination effects such as cast shadows, self-reflections and ambient light come into play, especially for specular surfaces. Recent approaches have leveraged the capabilities of deep learning in conjunction with computer graphics in order to cope with the need of a vast number of training data to invert the image irradiance equation and retrieve the geometry of the object. However, rendering global illumination effects is a slow process which can limit the amount of training data that can be generated. In this work we propose a novel pixel-wise training procedure for normal prediction by replacing the training data (observation maps) of globally rendered images with independent per-pixel generated data. We show that global physical effects can be approximated on the observation map domain and this simplifies and speeds up the data creation procedure. Our network, PX-NET, achieves state-of-the-art performance compared to other pixelwise methods on synthetic datasets, as well as the DiLiGenT real dataset on both dense and sparse light settings.

count=1
* GNeRF: GAN-Based Neural Radiance Field Without Posed Camera
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Meng_GNeRF_GAN-Based_Neural_Radiance_Field_Without_Posed_Camera_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Meng_GNeRF_GAN-Based_Neural_Radiance_Field_Without_Posed_Camera_ICCV_2021_paper.pdf)]
    * Title: GNeRF: GAN-Based Neural Radiance Field Without Posed Camera
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Quan Meng, Anpei Chen, Haimin Luo, Minye Wu, Hao Su, Lan Xu, Xuming He, Jingyi Yu
    * Abstract: We introduce GNeRF, a framework to marry Generative Adversarial Networks (GAN) with Neural Radiance Field (NeRF) reconstruction for the complex scenarios with unknown and even randomly initialized camera poses. Recent NeRF-based advances have gained popularity for remarkable realistic novel view synthesis. However, most of them heavily rely on accurate camera poses estimation, while few recent methods can only optimize the unknown camera poses in roughly forward-facing scenes with relatively short camera trajectories and require rough camera poses initialization. Differently, our GNeRF only utilizes randomly initialized poses for complex outside-in scenarios. We propose a novel two-phases end-to-end framework. The first phase takes the use of GANs into the new realm for optimizing coarse camera poses and radiance fields jointly, while the second phase refines them with additional photometric loss. We overcome local minima using a hybrid and iterative optimization scheme. Extensive experiments on a variety of synthetic and natural scenes demonstrate the effectiveness of GNeRF. More impressively, our approach outperforms the baselines favorably in those scenes with repeated patterns or even low textures that are regarded as extremely challenging before.

count=1
* Neural Articulated Radiance Field
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Noguchi_Neural_Articulated_Radiance_Field_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Noguchi_Neural_Articulated_Radiance_Field_ICCV_2021_paper.pdf)]
    * Title: Neural Articulated Radiance Field
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Atsuhiro Noguchi, Xiao Sun, Stephen Lin, Tatsuya Harada
    * Abstract: We present Neural Articulated Radiance Field (NARF), a novel deformable 3D representation for articulated objects learned from images. While recent advances in 3D implicit representation have made it possible to learn models of complex objects, learning pose-controllable representations of articulated objects remains a challenge, as current methods require 3D shape supervision and are unable to render appearance. In formulating an implicit representation of 3D articulated objects, our method considers only the rigid transformation of the most relevant object part in solving for the radiance field at each 3D location. In this way, the proposed method represents pose-dependent changes without significantly increasing the computational complexity. NARF is fully differentiable and can be trained from images with pose annotations. Moreover, through the use of an autoencoder, it can learn appearance variations over multiple instances of an object class. Experiments show that the proposed method is efficient and can generalize well to novel poses. The code is available for research purposes at https://github.com/nogu-atsu/NARF

count=1
* Audio-Visual Floorplan Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Purushwalkam_Audio-Visual_Floorplan_Reconstruction_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Purushwalkam_Audio-Visual_Floorplan_Reconstruction_ICCV_2021_paper.pdf)]
    * Title: Audio-Visual Floorplan Reconstruction
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Senthil Purushwalkam, Sebastià Vicenc Amengual Garí, Vamsi Krishna Ithapu, Carl Schissler, Philip Robinson, Abhinav Gupta, Kristen Grauman
    * Abstract: Given only a few glimpses of an environment, how much can we infer about its entire floorplan? Existing methods can map only what is visible or immediately apparent from context, and thus require substantial movements through a space to fully map it. We explore how both audio and visual sensing together can provide rapid floorplan reconstruction from limited viewpoints. Audio not only helps sense geometry outside the camera's field of view, but it also reveals the existence of distant freespace (e.g., a dog barking in another room) and suggests the presence of rooms not visible to the camera (e.g., a dishwasher humming in what must be the kitchen to the left). We introduce AV-Map, a novel multi-modal encoder-decoder framework that reasons jointly about audio and vision to reconstruct a floorplan from a short input video sequence. We train our model to predict both the interior structure of the environment and the associated rooms' semantic labels. Our results on 85 large real-world environments show the impact: with just a few glimpses spanning 26% of an area, we can estimate the whole area with 66% accuracy---substantially better than the state of the art approach for extrapolating visual maps.

count=1
* H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Ramon_H3D-Net_Few-Shot_High-Fidelity_3D_Head_Reconstruction_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Ramon_H3D-Net_Few-Shot_High-Fidelity_3D_Head_Reconstruction_ICCV_2021_paper.pdf)]
    * Title: H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Eduard Ramon, Gil Triginer, Janna Escur, Albert Pumarola, Jaime Garcia, Xavier Giró-i-Nieto, Francesc Moreno-Noguer
    * Abstract: Recent learning approaches that implicitly represent surface geometry using coordinate-based neural representations have shown impressive results in the problem of multi-view 3D reconstruction. The effectiveness of these techniques is, however, subject to the availability of a large number (several tens) of input views of the scene, and computationally demanding optimizations. In this paper, we tackle these limitations for the specific problem of few-shot full 3D head reconstruction, by endowing coordinate-based representations with a probabilistic shape prior that enables faster convergence and better generalization when using few input images (down to three). First, we learn a shape model of 3D heads from thousands of incomplete raw scans using implicit representations. At test time, we jointly overfit two coordinate-based neural networks to the scene, one modeling the geometry and another estimating the surface radiance, using implicit differentiable rendering. We devise a two-stage optimization strategy in which the learned prior is used to initialize and constrain the geometry during an initial optimization phase. Then, the prior is unfrozen and fine-tuned to the scene. By doing this, we achieve high-fidelity head reconstructions, including hair and shoulders, and with a high level of detail that consistently outperforms both state-of-the-art 3D Morphable Models methods in the few-shot scenario, and non-parametric methods when large sets of views are available.

count=1
* Image2Reverb: Cross-Modal Reverb Impulse Response Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Singh_Image2Reverb_Cross-Modal_Reverb_Impulse_Response_Synthesis_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Singh_Image2Reverb_Cross-Modal_Reverb_Impulse_Response_Synthesis_ICCV_2021_paper.pdf)]
    * Title: Image2Reverb: Cross-Modal Reverb Impulse Response Synthesis
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Nikhil Singh, Jeff Mentch, Jerry Ng, Matthew Beveridge, Iddo Drori
    * Abstract: Measuring the acoustic characteristics of a space is often done by capturing its impulse response (IR), a representation of how a full-range stimulus sound excites it. This work generates an IR from a single image, which can then be applied to other signals using convolution, simulating the reverberant characteristics of the space shown in the image. Recording these IRs is both time-intensive and expensive, and often infeasible for inaccessible locations. We use an end-to-end neural network architecture to generate plausible audio impulse responses from single images of acoustic environments. We evaluate our method both by comparisons to ground truth data and by human expert evaluation. We demonstrate our approach by generating plausible impulse responses from diverse settings and formats including well known places, musical halls, rooms in paintings, images from animations and computer games, synthetic environments generated from text, panoramic images, and video conference backgrounds.

count=1
* Objects As Cameras: Estimating High-Frequency Illumination From Shadows
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Swedish_Objects_As_Cameras_Estimating_High-Frequency_Illumination_From_Shadows_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Swedish_Objects_As_Cameras_Estimating_High-Frequency_Illumination_From_Shadows_ICCV_2021_paper.pdf)]
    * Title: Objects As Cameras: Estimating High-Frequency Illumination From Shadows
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Tristan Swedish, Connor Henley, Ramesh Raskar
    * Abstract: We recover high-frequency information encoded in the shadows cast by an object to estimate a hemispherical photograph from the viewpoint of the object, effectively turning objects into cameras. Estimating environment maps is useful for advanced image editing tasks such as relighting, object insertion or removal, and material parameter estimation. Because the problem is ill-posed, recent works in illumination recovery have tackled the problem of low-frequency lighting for object insertion, rely upon specular surface materials, or make use of data-driven methods that are susceptible to hallucination without physically plausible constraints. We incorporate an optimization scheme to update scene parameters that could enable practical capture of real-world scenes. Furthermore, we develop a methodology for evaluating expected recovery performance for different types and shapes of objects.

count=1
* Learning To Remove Refractive Distortions From Underwater Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Thapa_Learning_To_Remove_Refractive_Distortions_From_Underwater_Images_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Thapa_Learning_To_Remove_Refractive_Distortions_From_Underwater_Images_ICCV_2021_paper.pdf)]
    * Title: Learning To Remove Refractive Distortions From Underwater Images
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Simron Thapa, Nianyi Li, Jinwei Ye
    * Abstract: The fluctuation of the water surface causes refractive distortions that severely downgrade the image of an underwater scene. Here, we present the distortion-guided network (DG-Net) for restoring distortion-free underwater images. The key idea is to use a distortion map to guide network training. The distortion map models the pixel displacement caused by water refraction. We first use a physically constrained convolutional network to estimate the distortion map from the refracted image. We then use a generative adversarial network guided by the distortion map to restore the sharp distortion-free image. Since the distortion map indicates correspondences between the distorted image and the distortion-free one, it guides the network to make better predictions. We evaluate our network on several real and synthetic underwater image datasets and show that it out-performs the state-of-the-art algorithms, especially in presence of large distortions. We also show results of complex scenarios, including outdoor swimming pool images captured by the drone and indoor aquarium images taken by cellphone camera.

count=1
* GRF: Learning a General Radiance Field for 3D Representation and Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Trevithick_GRF_Learning_a_General_Radiance_Field_for_3D_Representation_and_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Trevithick_GRF_Learning_a_General_Radiance_Field_for_3D_Representation_and_ICCV_2021_paper.pdf)]
    * Title: GRF: Learning a General Radiance Field for 3D Representation and Rendering
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Alex Trevithick, Bo Yang
    * Abstract: We present a simple yet powerful neural network that implicitly represents and renders 3D objects and scenes only from 2D observations. The network models 3D geometries as a general radiance field, which takes a set of 2D images with camera poses and intrinsics as input, constructs an internal representation for each point of the 3D space, and then renders the corresponding appearance and geometry of that point viewed from an arbitrary position. The key to our approach is to learn local features for each pixel in 2D images and to then project these features to 3D points, thus yielding general and rich point representations. We additionally integrate an attention mechanism to aggregate pixel features from multiple 2D views, such that visual occlusions are implicitly taken into account. Extensive experiments demonstrate that our method can generate high-quality and realistic novel views for novel objects, unseen categories and challenging real-world scenes.

count=1
* Fake It Till You Make It: Face Analysis in the Wild Using Synthetic Data Alone
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Wood_Fake_It_Till_You_Make_It_Face_Analysis_in_the_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Wood_Fake_It_Till_You_Make_It_Face_Analysis_in_the_ICCV_2021_paper.pdf)]
    * Title: Fake It Till You Make It: Face Analysis in the Wild Using Synthetic Data Alone
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Erroll Wood, Tadas Baltrušaitis, Charlie Hewitt, Sebastian Dziadzio, Thomas J. Cashman, Jamie Shotton
    * Abstract: We demonstrate that it is possible to perform face-related computer vision in the wild using synthetic data alone. The community has long enjoyed the benefits of synthesizing training data with graphics, but the domain gap between real and synthetic data has remained a problem, especially for human faces. Researchers have tried to bridge this gap with data mixing, domain adaptation, and domain-adversarial training, but we show that it is possible to synthesize data with minimal domain gap, so that models trained on synthetic data generalize to real in-the-wild datasets. We describe how to combine a procedurally-generated parametric 3D face model with a comprehensive library of hand-crafted assets to render training images with unprecedented realism and diversity. We train machine learning systems for face-related tasks such as landmark localization and face parsing, showing that synthetic data can both match real data in accuracy, as well as open up new approaches where manual labeling would be impossible.

count=1
* In-the-Wild Single Camera 3D Reconstruction Through Moving Water Surfaces
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Xiong_In-the-Wild_Single_Camera_3D_Reconstruction_Through_Moving_Water_Surfaces_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiong_In-the-Wild_Single_Camera_3D_Reconstruction_Through_Moving_Water_Surfaces_ICCV_2021_paper.pdf)]
    * Title: In-the-Wild Single Camera 3D Reconstruction Through Moving Water Surfaces
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jinhui Xiong, Wolfgang Heidrich
    * Abstract: We present a method for reconstructing the 3D shape of underwater environments from a single, stationary camera placed above the water. We propose a novel differentiable framework, which, to our knowledge, is the first single-camera solution that is capable of simultaneously retrieving the structure of dynamic water surfaces and static underwater scene geometry in the wild. This framework integrates ray casting of Snell's law at the refractive interface, multi-view triangulation and specially designed loss functions. Our method is calibration-free, and thus it is easy to collect data outdoors in uncontrolled environments. Experimental results show that our method is able to realize robust and quality reconstructions on a variety of scenes, both in a laboratory environment and in the wild, and even in a salt water environment. We believe the method is promising for applications in surveying and environmental monitoring.

count=1
* 3DIAS: 3D Shape Reconstruction With Implicit Algebraic Surfaces
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Yavartanoo_3DIAS_3D_Shape_Reconstruction_With_Implicit_Algebraic_Surfaces_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Yavartanoo_3DIAS_3D_Shape_Reconstruction_With_Implicit_Algebraic_Surfaces_ICCV_2021_paper.pdf)]
    * Title: 3DIAS: 3D Shape Reconstruction With Implicit Algebraic Surfaces
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Mohsen Yavartanoo, Jaeyoung Chung, Reyhaneh Neshatavar, Kyoung Mu Lee
    * Abstract: 3D Shape representation has substantial effects on 3D shape reconstruction. Primitive-based representations approximate a 3D shape mainly by a set of simple implicit primitives, but the low geometrical complexity of the primitives limits the shape resolution. Moreover, setting a sufficient number of primitives for an arbitrary shape is challenging. To overcome these issues, we propose a constrained implicit algebraic surface as the primitive with few learnable coefficients and higher geometrical complexities and a deep neural network to produce these primitives. Our experiments demonstrate the superiorities of our method in terms of representation power compared to the state-of-the-art methods in single RGB image 3D shape reconstruction. Furthermore, we show that our method can semantically learn segments of 3D shapes in an unsupervised manner. The code is publicly available from this link.

count=1
* CryoDRGN2: Ab Initio Neural Reconstruction of 3D Protein Structures From Real Cryo-EM Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zhong_CryoDRGN2_Ab_Initio_Neural_Reconstruction_of_3D_Protein_Structures_From_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhong_CryoDRGN2_Ab_Initio_Neural_Reconstruction_of_3D_Protein_Structures_From_ICCV_2021_paper.pdf)]
    * Title: CryoDRGN2: Ab Initio Neural Reconstruction of 3D Protein Structures From Real Cryo-EM Images
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Ellen D. Zhong, Adam Lerer, Joseph H. Davis, Bonnie Berger
    * Abstract: Protein structure determination from cryo-EM data requires reconstructing a 3D volume (or distribution of volumes) from many noisy and randomly oriented 2D projection images. While the standard homogeneous reconstruction task aims to recover a single static structure, recently-proposed neural and non-neural methods can reconstruct distributions of structures, thereby enabling the study of protein complexes that possess intrinsic structural or conformational heterogeneity. These heterogeneous reconstruction methods, however, require fixed image poses, which are typically estimated from an upstream homogeneous reconstruction and are not guaranteed to be accurate under highly heterogeneous conditions. In this work we describe cryoDRGN2, an ab initio reconstruction algorithm, which can jointly estimate image poses and learn a neural model of a distribution of 3D structures on real heterogeneous cryo-EM data. To achieve this, we adapt search algorithms from the traditional cryo-EM literature, and describe the optimizations and design choices required to make such a search procedure computationally tractable in the neural model setting. We show that cryoDRGN2 is robust to the high noise levels of real cryo-EM images, trains faster than earlier neural methods, and achieves state-of-the-art performance on real cryo-EM datasets.

count=1
* ToFNest: Efficient Normal Estimation for Time-of-Flight Depth Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/ACVR/html/Molnar_ToFNest_Efficient_Normal_Estimation_for_Time-of-Flight_Depth_Cameras_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/ACVR/papers/Molnar_ToFNest_Efficient_Normal_Estimation_for_Time-of-Flight_Depth_Cameras_ICCVW_2021_paper.pdf)]
    * Title: ToFNest: Efficient Normal Estimation for Time-of-Flight Depth Cameras
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Szilárd Molnár, Benjamin Kelényi, Levente Tamás
    * Abstract: In this work, we propose an efficient normal estimation method for depth images acquired by Time-of-Flight (ToF) cameras based on feature pyramid networks (FPN). We perform the normal estimation starting from the 2D depth images, projecting the measured data into the 3D space and computing the loss function for the point cloud normal. Despite the simplicity of our method, which we call ToFNest, it proves to be efficient in terms of robustness and runtime. In order to validate ToFNest we performed extensive evaluations using both public and custom outdoor datasets. Compared with the state of the art methods, our algorithm is faster by an order of magnitude without losing precision on public datasets. The demo code, custom datasets and videos are available on the project website.

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
* Dynamic PlenOctree for Adaptive Sampling Refinement in Explicit NeRF
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Bai_Dynamic_PlenOctree_for_Adaptive_Sampling_Refinement_in_Explicit_NeRF_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Bai_Dynamic_PlenOctree_for_Adaptive_Sampling_Refinement_in_Explicit_NeRF_ICCV_2023_paper.pdf)]
    * Title: Dynamic PlenOctree for Adaptive Sampling Refinement in Explicit NeRF
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Haotian Bai, Yiqi Lin, Yize Chen, Lin Wang
    * Abstract: The explicit neural radiance field (NeRF) has gained considerable interest for its efficient training and fast inference capabilities, making it a promising direction such as virtual reality and gaming. In particular, PlenOctree (POT), an explicit hierarchical multi-scale octree representation, has emerged as a structural and influential framework. However, POT's fixed structure for direct optimization is sub-optimal as the scene complexity evolves continuously with updates to cached color and density, necessitating refining the sampling distribution to capture signal complexity accordingly. To address this issue, we propose the dynamic PlenOctree (DOT), which adaptively refines the sample distribution to adjust to changing scene complexity. Specifically, DOT proposes a concise yet novel hierarchical feature fusion strategy during the iterative rendering process. Firstly, it identifies the regions of interest through training signals to ensure adaptive and efficient refinement. Next, rather than directly filtering out valueless nodes, DOT introduces the sampling and pruning operations for octrees to aggregate features, enabling rapid parameter learning. Compared with POT, our DOT outperforms it by enhancing visual quality, reducing over 55.15/68.84% parameters, and providing 1.7/1.9 times FPS for NeRF-synthetic and Tanks & Temples, respectively.

count=1
* Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Barron_Zip-NeRF_Anti-Aliased_Grid-Based_Neural_Radiance_Fields_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Barron_Zip-NeRF_Anti-Aliased_Grid-Based_Neural_Radiance_Fields_ICCV_2023_paper.pdf)]
    * Title: Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, Peter Hedman
    * Abstract: Neural Radiance Field training can be accelerated through the use of grid-based representations in NeRF's learned mapping from spatial coordinates to colors and volumetric density. However, these grid-based approaches lack an explicit understanding of scale and therefore often introduce aliasing, usually in the form of jaggies or missing scene content. Anti-aliasing has previously been addressed by mip-NeRF 360, which reasons about sub-volumes along a cone rather than points along a ray, but this approach is not natively compatible with current grid-based techniques. We show how ideas from rendering and signal processing can be used to construct a technique that combines mip-NeRF 360 and grid-based models such as Instant NGP to yield error rates that are 8%-77% lower than either prior technique, and that trains 24x faster than mip-NeRF 360.

count=1
* HiFace: High-Fidelity 3D Face Reconstruction by Learning Static and Dynamic Details
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chai_HiFace_High-Fidelity_3D_Face_Reconstruction_by_Learning_Static_and_Dynamic_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chai_HiFace_High-Fidelity_3D_Face_Reconstruction_by_Learning_Static_and_Dynamic_ICCV_2023_paper.pdf)]
    * Title: HiFace: High-Fidelity 3D Face Reconstruction by Learning Static and Dynamic Details
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zenghao Chai, Tianke Zhang, Tianyu He, Xu Tan, Tadas Baltrusaitis, HsiangTao Wu, Runnan Li, Sheng Zhao, Chun Yuan, Jiang Bian
    * Abstract: 3D Morphable Models (3DMMs) demonstrate great potential for reconstructing faithful and animatable 3D facial surfaces from a single image. The facial surface is influenced by the coarse shape, as well as the static detail (e,g., person-specific appearance) and dynamic detail (e.g., expression-driven wrinkles). Previous work struggles to decouple the static and dynamic details through image-level supervision, leading to reconstructions that are not realistic. In this paper, we aim at high-fidelity 3D face reconstruction and propose HiFace to explicitly model the static and dynamic details. Specifically, the static detail is modeled as the linear combination of a displacement basis, while the dynamic detail is modeled as the linear interpolation of two displacement maps with polarized expressions. We exploit several loss functions to jointly learn the coarse shape and fine details with both synthetic and real-world datasets, which enable HiFace to reconstruct high-fidelity 3D shapes with animatable details. Extensive quantitative and qualitative experiments demonstrate that HiFace presents state-of-the-art reconstruction quality and faithfully recovers both the static and dynamic details.

count=1
* ReLeaPS : Reinforcement Learning-based Illumination Planning for Generalized Photometric Stereo
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chan_ReLeaPS__Reinforcement_Learning-based_Illumination_Planning_for_Generalized_Photometric_Stereo_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chan_ReLeaPS__Reinforcement_Learning-based_Illumination_Planning_for_Generalized_Photometric_Stereo_ICCV_2023_paper.pdf)]
    * Title: ReLeaPS : Reinforcement Learning-based Illumination Planning for Generalized Photometric Stereo
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Jun Hoong Chan, Bohan Yu, Heng Guo, Jieji Ren, Zongqing Lu, Boxin Shi
    * Abstract: Illumination planning in photometric stereo aims to find a balance between tween surface normal estimation accuracy and image capturing efficiency by selecting optimal light configurations. It depends on factors such as the unknown shape and general reflectance of the target object, global illumination, and the choice of photometric stereo backbones, which are too complex to be handled by existing methods based on handcrafted illumination planning rules. This paper proposes a learning-based illumination planning method that jointly considers these factors via integrating a neural network and a generalized image formation model. As it is impractical to supervise illumination planning due to the enormous search space for ground truth light configurations, we formulate illumination planning using reinforcement learning, which explores the light space in a photometric stereo-aware and reward-driven manner. Experiments on synthetic and real-world datasets demonstrate that photometric stereo under the 20-light configurations from our method is comparable to, or even surpasses that of using lights from all available directions.

count=1
* CuNeRF: Cube-Based Neural Radiance Field for Zero-Shot Medical Image Arbitrary-Scale Super Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_CuNeRF_Cube-Based_Neural_Radiance_Field_for_Zero-Shot_Medical_Image_Arbitrary-Scale_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_CuNeRF_Cube-Based_Neural_Radiance_Field_for_Zero-Shot_Medical_Image_Arbitrary-Scale_ICCV_2023_paper.pdf)]
    * Title: CuNeRF: Cube-Based Neural Radiance Field for Zero-Shot Medical Image Arbitrary-Scale Super Resolution
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zixuan Chen, Lingxiao Yang, Jian-Huang Lai, Xiaohua Xie
    * Abstract: Medical image arbitrary-scale super-resolution (MIASSR) has recently gained widespread attention, aiming to supersample medical volumes at arbitrary scales via a single model. However, existing MIASSR methods face two major limitations: (i) reliance on high-resolution (HR) volumes and (ii) limited generalization ability, which restricts their applications in various scenarios. To overcome these limitations, we propose Cube-based Neural Radiance Field (CuNeRF), a zero-shot MIASSR framework that is able to yield medical images at arbitrary scales and free viewpoints in a continuous domain. Unlike existing MISR methods that only fit the mapping between low-resolution (LR) and HR volumes, CuNeRF focuses on building a continuous volumetric representation from each LR volume without the knowledge from the corresponding HR one. This is achieved by the proposed differentiable modules: cube-based sampling, isotropic volume rendering, and cube-based hierarchical rendering. Through extensive experiments on magnetic resource imaging (MRI) and computed tomography (CT) modalities, we demonstrate that CuNeRF can synthesize high-quality SR medical images, which outperforms state-of-the-art MISR methods, achieving better visual verisimilitude and fewer objectionable artifacts. Compared to existing MISR methods, our CuNeRF is more applicable in practice.

count=1
* Mimic3D: Thriving 3D-Aware GANs via 3D-to-2D Imitation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Mimic3D_Thriving_3D-Aware_GANs_via_3D-to-2D_Imitation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Mimic3D_Thriving_3D-Aware_GANs_via_3D-to-2D_Imitation_ICCV_2023_paper.pdf)]
    * Title: Mimic3D: Thriving 3D-Aware GANs via 3D-to-2D Imitation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Xingyu Chen, Yu Deng, Baoyuan Wang
    * Abstract: Generating images with both photorealism and multiview 3D consistency is crucial for 3D-aware GANs, yet existing methods struggle to achieve them simultaneously. Improving the photorealism via CNN-based 2D super-resolution can break the strict 3D consistency, while keeping the 3D consistency by learning high-resolution 3D representations for direct rendering often compromises image quality. In this paper, we propose a novel learning strategy, namely 3D-to-2D imitation, which enables a 3D-aware GAN to generate high-quality images while maintaining their strict 3D consistency, by letting the images synthesized by the generator's 3D rendering branch mimic those generated by its 2D super-resolution branch. We also introduce 3D-aware convolutions into the generator for better 3D representation learning, which further improves the image generation quality. With the above strategies, our method reaches FID scores of 5.4 and 4.3 on FFHQ and AFHQ-v2 Cats, respectively, at 512x512 resolution, largely outperforming existing 3D-aware GANs using direct 3D rendering and coming very close to the previous state-of-the-art method that leverages 2D super-resolution. Project website: https://seanchenxy.github.io/Mimic3DWeb.

count=1
* SHIFT3D: Synthesizing Hard Inputs For Tricking 3D Detectors
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_SHIFT3D_Synthesizing_Hard_Inputs_For_Tricking_3D_Detectors_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_SHIFT3D_Synthesizing_Hard_Inputs_For_Tricking_3D_Detectors_ICCV_2023_paper.pdf)]
    * Title: SHIFT3D: Synthesizing Hard Inputs For Tricking 3D Detectors
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Hongge Chen, Zhao Chen, Gregory P. Meyer, Dennis Park, Carl Vondrick, Ashish Shrivastava, Yuning Chai
    * Abstract: We present SHIFT3D, a differentiable pipeline for generating 3D shapes that are structurally plausible yet challenging to 3D object detectors. In safety-critical applications like autonomous driving, discovering such novel challenging objects can offer insight into unknown vulnerabilities of 3D detectors. By representing objects with a signed distanced function (SDF), we show that gradient error signals allow us to smoothly deform the shape or pose of a 3D object in order to confuse a downstream 3D detector. Importantly, the objects generated by SHIFT3D physically differ from the baseline object yet retain a semantically recognizable shape. Our approach provides interpretable failure modes for modern 3D object detectors, and can aid in preemptive discovery of potential safety risks within 3D perception systems before these risks become critical failures.

count=1
* Spacetime Surface Regularization for Neural Dynamic Scene Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Choe_Spacetime_Surface_Regularization_for_Neural_Dynamic_Scene_Reconstruction_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Choe_Spacetime_Surface_Regularization_for_Neural_Dynamic_Scene_Reconstruction_ICCV_2023_paper.pdf)]
    * Title: Spacetime Surface Regularization for Neural Dynamic Scene Reconstruction
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Jaesung Choe, Christopher Choy, Jaesik Park, In So Kweon, Anima Anandkumar
    * Abstract: We propose an algorithm, 4DRegSDF, for the spacetime surface regularization to improve the fidelity of neural rendering and reconstruction in dynamic scenes. The key idea is to impose local rigidity on the deformable Signed Distance Function (SDF) for temporal coherency. Our approach works by (1) sampling points on the deformed surface by taking gradient steps toward the steepest direction along SDF, (2) extracting differential surface geometry, such as tangent plane or curvature, at each sample, and (3) adjusting the local rigidity at different timestamps. This enables our dynamic surface regularization to align 4D spacetime geometry via 3D canonical space more accurately. Experiments demonstrate that our 4DRegSDF achieves state-of-the-art performance in both reconstruction and rendering quality over synthetic and real-world datasets.

count=1
* NLOS-NeuS: Non-line-of-sight Neural Implicit Surface
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Fujimura_NLOS-NeuS_Non-line-of-sight_Neural_Implicit_Surface_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Fujimura_NLOS-NeuS_Non-line-of-sight_Neural_Implicit_Surface_ICCV_2023_paper.pdf)]
    * Title: NLOS-NeuS: Non-line-of-sight Neural Implicit Surface
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Yuki Fujimura, Takahiro Kushida, Takuya Funatomi, Yasuhiro Mukaigawa
    * Abstract: Non-line-of-sight (NLOS) imaging is conducted to infer invisible scenes from indirect light on visible objects. The neural transient field (NeTF) was proposed for representing scenes as neural radiance fields in NLOS scenes. We propose NLOS neural implicit surface (NLOS-NeuS), which extends the NeTF to neural implicit surfaces with a signed distance function (SDF) for reconstructing three-dimensional surfaces in NLOS scenes. We introduce two constraints as loss functions for correctly learning an SDF to avoid non-zero level-set surfaces. We also introduce a lower bound constraint of an SDF based on the geometry of the first-returning photons. The experimental results indicate that these constraints are essential for learning a correct SDF in NLOS scenes. Compared with previous methods with discretized representation, NLOS-NeuS with the neural continuous representation enables us to reconstruct smooth surfaces while preserving fine details in NLOS scenes. To the best of our knowledge, this is the first study on neural implicit surfaces with volume rendering in NLOS scenes. Project page: https://yfujimura. github.io/nlos-neus/

count=1
* Adaptive Positional Encoding for Bundle-Adjusting Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_Adaptive_Positional_Encoding_for_Bundle-Adjusting_Neural_Radiance_Fields_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Adaptive_Positional_Encoding_for_Bundle-Adjusting_Neural_Radiance_Fields_ICCV_2023_paper.pdf)]
    * Title: Adaptive Positional Encoding for Bundle-Adjusting Neural Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zelin Gao, Weichen Dai, Yu Zhang
    * Abstract: Neural Radiance Fields have shown great potential to synthesize novel views with only a few discrete image observations of the world. However, the requirement of accurate camera parameters to learn scene representations limits its further application. In this paper, we present adaptive positional encoding (APE) for bundle-adjusting neural radiance fields to reconstruct the neural radiance fields from unknown camera poses (or even intrinsics). Inspired by Fourier series regression, we investigate its relationship with the positional encoding method and therefore propose APE where all frequency bands are trainable. Furthermore, we introduce period-activated multilayer perceptrons (PMLPs) to construct the implicit network for the high-order scene representations and fine-grain gradients during backpropagation. Experimental results on public datasets demonstrate that the proposed method with APE and PMLPs can outperform the state-of-the-art methods in accurate camera poses and high-fidelity view synthesis.

count=1
* Ref-NeuS: Ambiguity-Reduced Neural Implicit Surface Learning for Multi-View Reconstruction with Reflection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Ge_Ref-NeuS_Ambiguity-Reduced_Neural_Implicit_Surface_Learning_for_Multi-View_Reconstruction_with_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Ge_Ref-NeuS_Ambiguity-Reduced_Neural_Implicit_Surface_Learning_for_Multi-View_Reconstruction_with_ICCV_2023_paper.pdf)]
    * Title: Ref-NeuS: Ambiguity-Reduced Neural Implicit Surface Learning for Multi-View Reconstruction with Reflection
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Wenhang Ge, Tao Hu, Haoyu Zhao, Shu Liu, Ying-Cong Chen
    * Abstract: Neural implicit surface learning has shown significant progress in multi-view 3D reconstruction, where an object is represented by multilayer perceptrons that provide continuous implicit surface representation and view-dependent radiance. However, current methods often fail to accurately reconstruct reflective surfaces, leading to severe ambiguity. To overcome this issue, we propose Ref-NeuS, which aims to reduce ambiguity by attenuating the effect of reflective surfaces. Specifically, we utilize an anomaly detector to estimate an explicit reflection score with the guidance of multi-view context to localize reflective surfaces. Afterward, we design a reflection-aware photometric loss that adaptively reduces ambiguity by modeling rendered color as a Gaussian distribution, with the reflection score representing the variance. We show that together with a reflection direction-dependent radiance, our model achieves high-quality surface reconstruction on reflective surfaces and outperforms the state-of-the-arts by a large margin. Besides, our model is also comparable on general surfaces.

count=1
* Neural LiDAR Fields for Novel View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Neural_LiDAR_Fields_for_Novel_View_Synthesis_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_Neural_LiDAR_Fields_for_Novel_View_Synthesis_ICCV_2023_paper.pdf)]
    * Title: Neural LiDAR Fields for Novel View Synthesis
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Shengyu Huang, Zan Gojcic, Zian Wang, Francis Williams, Yoni Kasten, Sanja Fidler, Konrad Schindler, Or Litany
    * Abstract: We present Neural Fields for LiDAR (NFL), a method to optimise a neural field scene representation from LiDAR measurements, with the goal of synthesizing realistic LiDAR scans from novel viewpoints. NFL combines the rendering power of neural fields with a detailed, physically motivated model of the LiDAR sensing process, thus enabling it to accurately reproduce key sensor behaviors like beam divergence, secondary returns, and ray dropping. We evaluate NFL on synthetic and real LiDAR scans and show that it outperforms explicit reconstruct-then-simulate methods as well as other NeRF-style methods on LiDAR novel view synthesis task. Moreover, we show that the improved realism of the synthesized views narrows the domain gap to real scans and translates to better registration and semantic segmentation performance.

count=1
* SHERF: Generalizable Human NeRF from a Single Image
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_SHERF_Generalizable_Human_NeRF_from_a_Single_Image_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_SHERF_Generalizable_Human_NeRF_from_a_Single_Image_ICCV_2023_paper.pdf)]
    * Title: SHERF: Generalizable Human NeRF from a Single Image
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Shoukang Hu, Fangzhou Hong, Liang Pan, Haiyi Mei, Lei Yang, Ziwei Liu
    * Abstract: Existing Human NeRF methods for reconstructing 3D humans typically rely on multiple 2D images from multi-view cameras or monocular videos captured from fixed camera views. However, in real-world scenarios, human images are often captured from random camera angles, presenting challenges for high-quality 3D human reconstruction. In this paper, we propose SHERF, the first generalizable Human NeRF model for recovering animatable 3D humans from a single input image. SHERF extracts and encodes 3D human representations in canonical space, enabling rendering and animation from free views and poses. To achieve high-fidelity novel view and pose synthesis, the encoded 3D human representations should capture both global appearance and local fine-grained textures. To this end, we propose a bank of 3D-aware hierarchical features, including global, point-level, and pixel-aligned features, to facilitate informative encoding. Global features enhance the information extracted from the single input image and complement the information missing from the partial 2D observation. Point-level features provide strong clues of 3D human structure, while pixel-aligned features preserve more fine-grained details. To effectively integrate the 3D-aware hierarchical feature bank, we design a feature fusion transformer. Extensive experiments on THuman, RenderPeople, ZJU_MoCap, and HuMMan datasets demonstrate that SHERF achieves state-of-the-art performance, with better generalizability for novel view and pose synthesis.

count=1
* FaceCLIPNeRF: Text-driven 3D Face Manipulation using Deformable Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Hwang_FaceCLIPNeRF_Text-driven_3D_Face_Manipulation_using_Deformable_Neural_Radiance_Fields_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Hwang_FaceCLIPNeRF_Text-driven_3D_Face_Manipulation_using_Deformable_Neural_Radiance_Fields_ICCV_2023_paper.pdf)]
    * Title: FaceCLIPNeRF: Text-driven 3D Face Manipulation using Deformable Neural Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Sungwon Hwang, Junha Hyung, Daejin Kim, Min-Jung Kim, Jaegul Choo
    * Abstract: As recent advances in Neural Radiance Fields (NeRF) have enabled high-fidelity 3D face reconstruction and novel view synthesis, its manipulation also became an essential task in 3D vision. However, existing manipulation methods require extensive human labor, such as a user-provided semantic mask and manual attribute search unsuitable for non-expert users. Instead, our approach is designed to require a single text to manipulate a face reconstructed with NeRF. To do so, we first train a scene manipulator, a latent code-conditional deformable NeRF, over a dynamic scene to control a face deformation using the latent code. However, representing a scene deformation with a single latent code is unfavorable for compositing local deformations observed in different instances. As so, our proposed Position-conditional Anchor Compositor (PAC) learns to represent a manipulated scene with spatially varying latent codes. Their renderings with the scene manipulator are then optimized to yield high cosine similarity to a target text in CLIP embedding space for text-driven manipulation. To the best of our knowledge, our approach is the first to address the text-driven manipulation of a face reconstructed with NeRF. Extensive results, comparisons, and ablation studies demonstrate the effectiveness of our approach.

count=1
* MIMO-NeRF: Fast Neural Rendering with Multi-input Multi-output Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Kaneko_MIMO-NeRF_Fast_Neural_Rendering_with_Multi-input_Multi-output_Neural_Radiance_Fields_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Kaneko_MIMO-NeRF_Fast_Neural_Rendering_with_Multi-input_Multi-output_Neural_Radiance_Fields_ICCV_2023_paper.pdf)]
    * Title: MIMO-NeRF: Fast Neural Rendering with Multi-input Multi-output Neural Radiance Fields
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Takuhiro Kaneko
    * Abstract: Neural radiance fields (NeRFs) have shown impressive results for novel view synthesis. However, they depend on the repetitive use of a single-input single-output multilayer perceptron (SISO MLP) that maps 3D coordinates and view direction to the color and volume density in a sample-wise manner, which slows the rendering. We propose a multi-input multi-output NeRF (MIMO-NeRF) that reduces the number of MLPs running by replacing the SISO MLP with a MIMO MLP and conducting mappings in a group-wise manner. One notable challenge with this approach is that the color and volume density of each point can differ according to a choice of input coordinates in a group, which can lead to some notable ambiguity. We also propose a self-supervised learning method that regularizes the MIMO MLP with multiple fast reformulated MLPs to alleviate this ambiguity without using pretrained models. The results of a comprehensive experimental evaluation including comparative and ablation studies are presented to show that MIMO-NeRF obtains a good trade-off between speed and quality with a reasonable training time. We then demonstrate that MIMO-NeRF is compatible with and complementary to previous advancements in NeRFs by applying it to two representative fast NeRFs, i.e., a NeRF with a sampling network (DONeRF) and a NeRF with alternative representations (TensoRF).

count=1
* CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Kim_CRN_Camera_Radar_Net_for_Accurate_Robust_Efficient_3D_Perception_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_CRN_Camera_Radar_Net_for_Accurate_Robust_Efficient_3D_Perception_ICCV_2023_paper.pdf)]
    * Title: CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Youngseok Kim, Juyeb Shin, Sanmin Kim, In-Jae Lee, Jun Won Choi, Dongsuk Kum
    * Abstract: Autonomous driving requires an accurate and fast 3D perception system that includes 3D object detection, tracking, and segmentation. Although recent low-cost camera-based approaches have shown promising results, they are susceptible to poor illumination or bad weather conditions and have a large localization error. Hence, fusing camera with low-cost radar, which provides precise long-range measurement and operates reliably in all environments, is promising but has not yet been thoroughly investigated. In this paper, we propose Camera Radar Net (CRN), a novel camera-radar fusion framework that generates a semantically rich and spatially accurate bird's-eye-view (BEV) feature map for various tasks. To overcome the lack of spatial information in an image, we transform perspective view image features to BEV with the help of sparse but accurate radar points. We further aggregate image and radar feature maps in BEV using multi-modal deformable attention designed to tackle the spatial misalignment between inputs. CRN with real-time setting operates at 20 FPS while achieving comparable performance to LiDAR detectors on nuScenes, and even outperforms at a far distance on 100m setting. Moreover, CRN with offline setting yields 62.4% NDS, 57.5% mAP on nuScenes test set and ranks first among all camera and camera-radar 3D object detectors.

count=1
* DISeR: Designing Imaging Systems with Reinforcement Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Klinghoffer_DISeR_Designing_Imaging_Systems_with_Reinforcement_Learning_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Klinghoffer_DISeR_Designing_Imaging_Systems_with_Reinforcement_Learning_ICCV_2023_paper.pdf)]
    * Title: DISeR: Designing Imaging Systems with Reinforcement Learning
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Tzofi Klinghoffer, Kushagra Tiwary, Nikhil Behari, Bhavya Agrawalla, Ramesh Raskar
    * Abstract: Imaging systems consist of cameras to encode visual information about the world and perception models to interpret this encoding. Cameras contain (1) illumination sources, (2) optical elements, and (3) sensors, while perception models use (4) algorithms. Directly searching over all combinations of these four building blocks to design an imaging system is challenging due to the size of the search space. Moreover, cameras and perception models are often designed independently, leading to sub-optimal task performance. In this paper, we formulate these four building blocks of imaging systems as a context-free grammar (CFG), which can be automatically searched over with a learned camera designer to jointly optimize the imaging system with task-specific perception models. By transforming the CFG to a state-action space, we then show how the camera designer can be implemented with reinforcement learning to intelligently search over the combinatorial space of possible imaging system configurations. We demonstrate our approach on two tasks, depth estimation and camera rig design for autonomous vehicles, showing that our method yields rigs that outperform industry-wide standards. We believe that our proposed approach is an important step towards automating imaging system design.

count=1
* Tetra-NeRF: Representing Neural Radiance Fields Using Tetrahedra
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Kulhanek_Tetra-NeRF_Representing_Neural_Radiance_Fields_Using_Tetrahedra_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Kulhanek_Tetra-NeRF_Representing_Neural_Radiance_Fields_Using_Tetrahedra_ICCV_2023_paper.pdf)]
    * Title: Tetra-NeRF: Representing Neural Radiance Fields Using Tetrahedra
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Jonas Kulhanek, Torsten Sattler
    * Abstract: Neural Radiance Fields (NeRFs) are a very recent and very popular approach for the problems of novel view synthesis and 3D reconstruction. A popular scene representation used by NeRFs is to combine a uniform, voxel-based subdivision of the scene with an MLP. Based on the observation that a (sparse) point cloud of the scene is often available, this paper proposes to use an adaptive representation based on tetrahedra obtained by Delaunay triangulation instead of uniform subdivision or point-based representations. We show that such a representation enables efficient training and leads to state-of-the-art results. Our approach elegantly combines concepts from 3D geometry processing, triangle-based rendering, and modern neural radiance fields. Compared to voxel-based representations, ours provides more detail around parts of the scene likely to be close to the surface. Compared to point-based representations, our approach achieves better performance. The source code is publicly available at: https://jkulhanek.com/tetra-nerf.

count=1
* ExBluRF: Efficient Radiance Fields for Extreme Motion Blurred Images
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_ExBluRF_Efficient_Radiance_Fields_for_Extreme_Motion_Blurred_Images_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_ExBluRF_Efficient_Radiance_Fields_for_Extreme_Motion_Blurred_Images_ICCV_2023_paper.pdf)]
    * Title: ExBluRF: Efficient Radiance Fields for Extreme Motion Blurred Images
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Dongwoo Lee, Jeongtaek Oh, Jaesung Rim, Sunghyun Cho, Kyoung Mu Lee
    * Abstract: We present ExBluRF, a novel view synthesis method for extreme motion blurred images based on efficient radiance fields optimization. Our approach consists of two main components: 6-DOF camera trajectory-based motion blur formulation and voxel-based radiance fields. From extremely blurred images, we optimize the sharp radiance fields by jointly estimating the camera trajectories that generate the blurry images. In training, multiple rays along the camera trajectory are accumulated to reconstruct single blurry color, which is equivalent to the physical motion blur operation. We minimize the photo-consistency loss on blurred image space and obtain the sharp radiance fields with camera trajectories that explain the blur of all images. The joint optimization on the blurred image space demands painfully increasing computation and resources proportional to the blur size. Our method solves this problem by replacing the MLP-based framework to low-dimensional 6-DOF camera poses and voxel-based radiance fields. Compared with the existing works, our approach restores much sharper 3D scenes from challenging motion blurred views with the order of 10x less training time and GPU memory consumption.

count=1
* CHORD: Category-level Hand-held Object Reconstruction via Shape Deformation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_CHORD_Category-level_Hand-held_Object_Reconstruction_via_Shape_Deformation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_CHORD_Category-level_Hand-held_Object_Reconstruction_via_Shape_Deformation_ICCV_2023_paper.pdf)]
    * Title: CHORD: Category-level Hand-held Object Reconstruction via Shape Deformation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Kailin Li, Lixin Yang, Haoyu Zhen, Zenan Lin, Xinyu Zhan, Licheng Zhong, Jian Xu, Kejian Wu, Cewu Lu
    * Abstract: In daily life, humans utilize hands to manipulate objects. Modeling the shape of objects that are manipulated by the hand is essential for AI to comprehend daily tasks and to learn manipulation skills. However, previous approaches have encountered difficulties in reconstructing the precise shapes of hand-held objects, primarily owing to a deficiency in prior shape knowledge and inadequate data for training. As illustrated, given a particular type of tool, such as a mug, despite its infinite variations in shape and appearance, humans have a limited number of 'effective' modes and poses for its manipulation. This can be attributed to the fact that humans have mastered the shape prior of the 'mug' category, and can quickly establish the corresponding relations between different mug instances and the prior, such as where the rim and handle are located. In light of this, we propose a new method, CHORD, for Category-level Hand-held Object Reconstruction via shape Deformation. CHORD deforms a categorical shape prior for reconstructing the intra-class objects. To ensure accurate reconstruction, we empower CHORD with three types of awareness: appearance, shape, and interacting pose. In addition, we have constructed a new dataset, COMIC, of category-level hand-object interaction. COMIC contains a rich array of object instances, materials, hand interactions, and viewing directions. Extensive evaluation shows that CHORD outperforms state-of-the-art approaches in both quantitative and qualitative measures. Code, model, and datasets are available at https://kailinli.github.io/CHORD.

count=1
* NeTO:Neural Reconstruction of Transparent Objects with Self-Occlusion Aware Refraction-Tracing
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_NeTONeural_Reconstruction_of_Transparent_Objects_with_Self-Occlusion_Aware_Refraction-Tracing_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_NeTONeural_Reconstruction_of_Transparent_Objects_with_Self-Occlusion_Aware_Refraction-Tracing_ICCV_2023_paper.pdf)]
    * Title: NeTO:Neural Reconstruction of Transparent Objects with Self-Occlusion Aware Refraction-Tracing
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zongcheng Li, Xiaoxiao Long, Yusen Wang, Tuo Cao, Wenping Wang, Fei Luo, Chunxia Xiao
    * Abstract: We present a novel method called NeTO, for capturing the 3D geometry of solid transparent objects from 2D images via volume rendering. Reconstructing transparent objects is a very challenging task, which is ill-suited for general-purpose reconstruction techniques due to the specular light transport phenomena. Although existing refraction-tracing-based methods, designed especially for this task, achieve impressive results, they still suffer from unstable optimization and loss of fine details since the explicit surface representation they adopted is difficult to be optimized, and the self-occlusion problem is ignored for refraction-tracing. In this paper, we propose to leverage implicit Signed Distance Function (SDF) as surface representation and optimize the SDF field via volume rendering with a self-occlusion aware refractive ray tracing. The implicit representation enables our method to be capable of reconstructing high-quality reconstruction even with a limited set of views, and the self-occlusion aware strategy makes it possible for our method to accurately reconstruct the self-occluded regions. Experiments show that our method achieves faithful reconstruction results and outperforms prior works by a large margin. Visit our project page at https://www.xxlong.site/NeTO/.

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
* Zero-1-to-3: Zero-shot One Image to 3D Object
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf)]
    * Title: Zero-1-to-3: Zero-shot One Image to 3D Object
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, Carl Vondrick
    * Abstract: We introduce Zero-1-to-3, a framework for changing the camera viewpoint of an object given just a single RGB image. To perform novel view synthesis in this underconstrained setting, we capitalize on the geometric priors that large-scale diffusion models learn about natural images. Our conditional diffusion model uses a synthetic dataset to learn controls of the relative camera viewpoint, which allow new images to be generated of the same object under a specified camera transformation. Even though it is trained on a synthetic dataset, our model retains a strong zero-shot generalization ability to out-of-distribution datasets as well as in-the-wild images, including impressionist paintings. Our viewpoint-conditioned diffusion approach can further be used for the task of 3D reconstruction from a single image. Qualitative and quantitative experiments show that our method significantly outperforms stateof- the-art single-view 3D reconstruction and novel view synthesis models by leveraging Internet-scale pre-training.

count=1
* Translating Images to Road Network: A Non-Autoregressive Sequence-to-Sequence Approach
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Lu_Translating_Images_to_Road_Network_A_Non-Autoregressive_Sequence-to-Sequence_Approach_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Translating_Images_to_Road_Network_A_Non-Autoregressive_Sequence-to-Sequence_Approach_ICCV_2023_paper.pdf)]
    * Title: Translating Images to Road Network: A Non-Autoregressive Sequence-to-Sequence Approach
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Jiachen Lu, Renyuan Peng, Xinyue Cai, Hang Xu, Hongyang Li, Feng Wen, Wei Zhang, Li Zhang
    * Abstract: The extraction of road network is essential for the generation of high-definition maps since it enables the precise localization of road landmarks and their interconnections. However, generating road network poses a significant challenge due to the conflicting underlying combination of Euclidean (e.g., road landmarks location) and non-Euclidean (e.g., road topological connectivity) structures. Existing methods struggle to merge the two types of data domains effectively, but few of them address it properly. Instead, our work establishes a unified representation of both types of data domain by projecting both Euclidean and non-Euclidean data into an integer series called RoadNet Sequence. Further than modeling an auto-regressive sequence-to-sequence Transformer model to understand RoadNet Sequence, we decouple the dependency of RoadNet Sequence into a mixture of auto-regressive and non-autoregressive dependency. Building on this, our proposed non-autoregressive sequence-to-sequence approach leverages non-autoregressive dependencies while fixing the gap towards auto-regressive dependencies, resulting in success on both efficiency and accuracy. Extensive experiments on nuScenes dataset demonstrate the superiority of RoadNet Sequence representation and the non-autoregressive approach compared to existing state-of-the-art alternatives.

count=1
* Towards Zero Domain Gap: A Comprehensive Study of Realistic LiDAR Simulation for Autonomy Testing
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Manivasagam_Towards_Zero_Domain_Gap_A_Comprehensive_Study_of_Realistic_LiDAR_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Manivasagam_Towards_Zero_Domain_Gap_A_Comprehensive_Study_of_Realistic_LiDAR_ICCV_2023_paper.pdf)]
    * Title: Towards Zero Domain Gap: A Comprehensive Study of Realistic LiDAR Simulation for Autonomy Testing
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Sivabalan Manivasagam, Ioan Andrei Bârsan, Jingkang Wang, Ze Yang, Raquel Urtasun
    * Abstract: Testing the full autonomy system in simulation is the safest and most scalable way to evaluate autonomous vehicle performance before deployment. This requires simulating sensor inputs such as LiDAR. To be effective, it is essential that the simulation has low domain gap with the real world. That is, the autonomy system in simulation should perform exactly the same way it would in the real world for the same scenario. To date, there has been limited analysis into what aspects of LiDAR phenomena affect autonomy performance. It is also difficult to evaluate the domain gap of existing LiDAR simulators, as they operate on fully synthetic scenes. In this paper, we propose a novel "paired-scenario" approach to evaluating the domain gap of a LiDAR simulator by reconstructing digital twins of real world scenarios. We can then simulate LiDAR in the scene and compare it to the real LiDAR. We leverage this setting to analyze what aspects of LiDAR simulation, such as pulse phenomena, scanning effects, and asset quality, affect the domain gap with respect to the autonomy system, including perception, prediction, and motion planning, and analyze how modifications to the simulated LiDAR influence each part. We identify key aspects that are important to model, such as motion blur, material reflectance, and the accurate geometric reconstruction of traffic participants. This helps provide research directions for improving LiDAR simulation and autonomy robustness to these effects. For more information, please visit the project website: https://waabi.ai/lidar-dg

count=1
* A Theory of Topological Derivatives for Inverse Rendering of Geometry
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Mehta_A_Theory_of_Topological_Derivatives_for_Inverse_Rendering_of_Geometry_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Mehta_A_Theory_of_Topological_Derivatives_for_Inverse_Rendering_of_Geometry_ICCV_2023_paper.pdf)]
    * Title: A Theory of Topological Derivatives for Inverse Rendering of Geometry
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ishit Mehta, Manmohan Chandraker, Ravi Ramamoorthi
    * Abstract: We introduce a theoretical framework for differentiable surface evolution that allows discrete topology changes through the use of topological derivatives for variational optimization of image functionals. While prior methods for inverse rendering of geometry rely on silhouette gradients for topology changes, such signals are sparse. In contrast, our theory derives topological derivatives that relate the introduction of vanishing holes and phases to changes in image intensity. As a result, we enable differentiable shape perturbations in the form of hole or phase nucleation. We validate the proposed theory with optimization of closed curves in 2D and surfaces in 3D to lend insights into limitations of current methods and enable improved applications such as image vectorization, vector-graphics generation from text prompts, single-image reconstruction of shape ambigrams and multiview 3D reconstruction.

count=1
* Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Qian_Sat2Density_Faithful_Density_Learning_from_Satellite-Ground_Image_Pairs_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Qian_Sat2Density_Faithful_Density_Learning_from_Satellite-Ground_Image_Pairs_ICCV_2023_paper.pdf)]
    * Title: Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Ming Qian, Jincheng Xiong, Gui-Song Xia, Nan Xue
    * Abstract: This paper aims to develop an accurate 3D geometry representation of satellite images using satellite-ground image pairs. Our focus is on the challenging problem of 3D-aware ground-views synthesis from a satellite image. We draw inspiration from the density field representation used in volumetric neural rendering and propose a new approach, called Sat2Density. Our method utilizes the properties of ground-view panoramas for the sky and non-sky regions to learn faithful density fields of 3D scenes in a geometric perspective. Unlike other methods that require extra depth information during training, our Sat2Density can automatically learn accurate and faithful 3D geometry via density representation without depth supervision. This advancement significantly improves the ground-view panorama synthesis task. Additionally, our study provides a new geometric perspective to understand the relationship between satellite and ground-view images in 3D space.

count=1
* 3DPPE: 3D Point Positional Encoding for Transformer-based Multi-Camera 3D Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Shu_3DPPE_3D_Point_Positional_Encoding_for_Transformer-based_Multi-Camera_3D_Object_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Shu_3DPPE_3D_Point_Positional_Encoding_for_Transformer-based_Multi-Camera_3D_Object_ICCV_2023_paper.pdf)]
    * Title: 3DPPE: 3D Point Positional Encoding for Transformer-based Multi-Camera 3D Object Detection
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Changyong Shu, Jiajun Deng, Fisher Yu, Yifan Liu
    * Abstract: Transformer-based methods have swept the benchmarks on 2D and 3D detection on images. Because tokenization before the attention mechanism drops the spatial information, positional encoding becomes critical for those methods. Recent works found that encodings based on samples of the 3D viewing rays can significantly improve the quality of multi-camera 3D object detection. We hypothesize that 3D point locations can provide more information than rays. Therefore, we introduce 3D point positional encoding, 3DPPE, to the 3D detection Transformer decoder. Although 3D measurements are not available at the inference time of monocular 3D object detection, 3DPPE uses predicted depth to approximate the real point positions. Our hybrid-depth module combines direct and categorical depth to estimate the refined depth of each pixel. Despite the approximation, 3DPPE achieves 46.0 mAP and 51.4 NDS on the competitive nuScenes dataset, significantly outperforming encodings based on ray samples. The codes are available at https://github.com/drilistbox/3DPPE.

count=1
* Neural Reconstruction of Relightable Human Model from Monocular Video
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Sun_Neural_Reconstruction_of_Relightable_Human_Model_from_Monocular_Video_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Neural_Reconstruction_of_Relightable_Human_Model_from_Monocular_Video_ICCV_2023_paper.pdf)]
    * Title: Neural Reconstruction of Relightable Human Model from Monocular Video
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Wenzhang Sun, Yunlong Che, Han Huang, Yandong Guo
    * Abstract: Creating relightable and animatable human characters from monocular video at a low cost is a critical task for digital human modeling and virtual reality applications. This task is complex due to intricate articulation motion, a wide range of ambient lighting conditions, and pose-dependent clothing deformations. In this paper, we introduce a novel self-supervised framework that takes a monocular video of a moving human as input and generates a 3D neural representation capable of being rendered with novel poses under arbitrary lighting conditions. Our framework decomposes dynamic humans under varying illumination into neural fields in canonical space, taking into account geometry and spatially varying BRDF material properties. Additionally, we introduce pose-driven deformation fields, enabling bidirectional mapping between canonical space and observation. Leveraging the proposed appearance decomposition and deformation fields, our framework learns in a self-supervised manner. Ultimately, based on pose-driven deformation, recovered appearance, and physically-based rendering, the reconstructed human figure becomes relightable and can be explicitly driven by novel poses. We demonstrate significant performance improvements over previous works and provide compelling examples of relighting from monocular videos of moving humans in challenging, uncontrolled capture scenarios.

count=1
* LoLep: Single-View View Synthesis with Locally-Learned Planes and Self-Attention Occlusion Inference
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_LoLep_Single-View_View_Synthesis_with_Locally-Learned_Planes_and_Self-Attention_Occlusion_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_LoLep_Single-View_View_Synthesis_with_Locally-Learned_Planes_and_Self-Attention_Occlusion_ICCV_2023_paper.pdf)]
    * Title: LoLep: Single-View View Synthesis with Locally-Learned Planes and Self-Attention Occlusion Inference
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Cong Wang, Yu-Ping Wang, Dinesh Manocha
    * Abstract: We propose a novel method, LoLep, which regresses Locally-Learned planes from a single RGB image to represent scenes accurately, thus generating better novel views. Without the depth information, regressing appropriate plane locations is a challenging problem. To solve this issue, we pre-partition the disparity space into bins and design a disparity sampler to regress local offsets for multiple planes in each bin. However, only using such a sampler makes the network not convergent; we further propose two optimizing strategies that combine with different disparity distributions of datasets and propose an occlusion-aware reprojection loss as a simple yet effective geometric supervision technique. We also introduce a self-attention mechanism to improve occlusion inference and present a Block-Sampling Self-Attention (BS-SA) module to address the problem of applying self-attention to large feature maps. We demonstrate the effectiveness of our approach and generate state-of-the-art results on different datasets. Compared to MINE, our approach has an LPIPS reduction of 4.8% 9.0% and an RV reduction of 74.9% 83.5%. We also evaluate the performance on real-world images and demonstrate the benefits. We will release the source code at the time of publication.

count=1
* NEMTO: Neural Environment Matting for Novel View and Relighting Synthesis of Transparent Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_NEMTO_Neural_Environment_Matting_for_Novel_View_and_Relighting_Synthesis_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_NEMTO_Neural_Environment_Matting_for_Novel_View_and_Relighting_Synthesis_ICCV_2023_paper.pdf)]
    * Title: NEMTO: Neural Environment Matting for Novel View and Relighting Synthesis of Transparent Objects
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Dongqing Wang, Tong Zhang, Sabine Süsstrunk
    * Abstract: We propose NEMTO, the first end-to-end neural rendering pipeline to model 3D transparent objects with complex geometry and unknown indices of refraction. Commonly used appearance modeling such as the Disney BSDF model cannot accurately address this challenging problem due to the complex light paths bending through refractions and the strong dependency of surface appearance on illumination. With 2D images of the transparent object as input, our method is capable of high-quality novel view and relighting synthesis. We leverage implicit Signed Distance Functions (SDF) to model the object geometry and propose a refraction-aware ray bending network to model the effects of light refraction within the object. Our ray bending network is more tolerant to geometric inaccuracies than traditional physically-based methods for rendering transparent objects. We provide extensive evaluations on both synthetic and real-world datasets to demonstrate our high-quality synthesis and the applicability of our method.

count=1
* ObjectSDF++: Improved Object-Compositional Neural Implicit Surfaces
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_ObjectSDF_Improved_Object-Compositional_Neural_Implicit_Surfaces_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_ObjectSDF_Improved_Object-Compositional_Neural_Implicit_Surfaces_ICCV_2023_paper.pdf)]
    * Title: ObjectSDF++: Improved Object-Compositional Neural Implicit Surfaces
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Qianyi Wu, Kaisiyuan Wang, Kejie Li, Jianmin Zheng, Jianfei Cai
    * Abstract: In recent years, neural implicit surface reconstruction has emerged as a popular paradigm for multi-view 3D reconstruction. Unlike traditional multi-view stereo approaches, the neural implicit surface-based methods leverage neural networks to represent 3D scenes as signed distance functions (SDFs). However, they tend to disregard the reconstruction of individual objects within the scene, which limits their performance and practical applications. To address this issue, previous work ObjectSDF introduced a nice framework of object-composition neural implicit surfaces, which utilizes 2D instance masks to supervise individual object SDFs. In this paper, we propose a new framework called ObjectSDF++ to overcome the limitations of ObjectSDF. First, in contrast to ObjectSDF whose performance is primarily restricted by its converted semantic field, the core component of our model is an occlusion-aware object opacity rendering formulation that directly volume-renders object opacity to be supervised with instance masks. Second, we design a novel regularization term for object distinction, which can effectively mitigate the issue that ObjectSDF may result in unexpected reconstruction in invisible regions due to the lack of constraint to prevent collisions. Our extensive experiments demonstrate that our novel framework not only produces superior object reconstruction results but also significantly improves the quality of scene reconstruction. Code and more resources can be found in https://qianyiwu.github.io/objectsdf++

count=1
* S-VolSDF: Sparse Multi-View Stereo Regularization of Neural Implicit Surfaces
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_S-VolSDF_Sparse_Multi-View_Stereo_Regularization_of_Neural_Implicit_Surfaces_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_S-VolSDF_Sparse_Multi-View_Stereo_Regularization_of_Neural_Implicit_Surfaces_ICCV_2023_paper.pdf)]
    * Title: S-VolSDF: Sparse Multi-View Stereo Regularization of Neural Implicit Surfaces
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Haoyu Wu, Alexandros Graikos, Dimitris Samaras
    * Abstract: Neural rendering of implicit surfaces performs well in 3D vision applications. However, it requires dense input views as supervision. When only sparse input images are available, output quality drops significantly due to the shape-radiance ambiguity problem. We note that this ambiguity can be constrained when a 3D point is visible in multiple views, as is the case in multi-view stereo (MVS). We thus propose to regularize neural rendering optimization with an MVS solution. The use of an MVS probability volume and a generalized cross entropy loss leads to a noise-tolerant optimization process. In addition, neural rendering provides global consistency constraints that guide the MVS depth hypothesis sampling and thus improves MVS performance. Given only three sparse input views, experiments show that our method not only outperforms generic neural rendering models by a large margin but also significantly increases the reconstruction quality of MVS models.

count=1
* Rendering Humans from Object-Occluded Monocular Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Xiang_Rendering_Humans_from_Object-Occluded_Monocular_Videos_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiang_Rendering_Humans_from_Object-Occluded_Monocular_Videos_ICCV_2023_paper.pdf)]
    * Title: Rendering Humans from Object-Occluded Monocular Videos
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Tiange Xiang, Adam Sun, Jiajun Wu, Ehsan Adeli, Li Fei-Fei
    * Abstract: 3D understanding and rendering of moving humans from monocular videos is a challenging task. Although recent progress has enabled this task to some extent, it is still difficult to guarantee satisfactory results in real-world scenarios, where obstacles may block the camera view and cause partial occlusions in the captured videos. Existing methods cannot handle such defects due to two reasons. Firstly, the standard rendering strategy relies on point-point mapping, which could lead to dramatic disparities between the visible and occluded areas of the body. Secondly, the naive direct regression approach does not consider any feasibility criteria (i.e., prior information) for rendering under occlusions. To tackle the above drawbacks, we present OccNeRF, a neural rendering method that achieves better rendering of humans in severely occluded scenes. As direct solutions to the two drawbacks, we propose surface-based rendering by integrating geometry and visibility priors. We validate our method on both simulated and real-world occlusions and demonstrate our method's superiority.

count=1
* IntrinsicNeRF: Learning Intrinsic Neural Radiance Fields for Editable Novel View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Ye_IntrinsicNeRF_Learning_Intrinsic_Neural_Radiance_Fields_for_Editable_Novel_View_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Ye_IntrinsicNeRF_Learning_Intrinsic_Neural_Radiance_Fields_for_Editable_Novel_View_ICCV_2023_paper.pdf)]
    * Title: IntrinsicNeRF: Learning Intrinsic Neural Radiance Fields for Editable Novel View Synthesis
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Weicai Ye, Shuo Chen, Chong Bao, Hujun Bao, Marc Pollefeys, Zhaopeng Cui, Guofeng Zhang
    * Abstract: Existing inverse rendering combined with neural rendering methods can only perform editable novel view synthesis on object-specific scenes, while we present intrinsic neural radiance fields, dubbed IntrinsicNeRF, which introduce intrinsic decomposition into the NeRF-based neural rendering method and can extend its application to room-scale scenes. Since intrinsic decomposition is a fundamentally under-constrained inverse problem, we propose a novel distance-aware point sampling and adaptive reflectance iterative clustering optimization method, which enables IntrinsicNeRF with traditional intrinsic decomposition constraints to be trained in an unsupervised manner, resulting in multi-view consistent intrinsic decomposition results. To cope with the problem that different adjacent instances of similar reflectance in a scene are incorrectly clustered together, we further propose a hierarchical clustering method with coarse-to-fine optimization to obtain a fast hierarchical indexing representation. It supports compelling real-time augmented applications such as recoloring and illumination variation. Extensive experiments and editing samples on both object-specific/room-scale scenes and synthetic/real-word data demonstrate that we can obtain consistent intrinsic decomposition results and high-fidelity novel view synthesis even for challenging sequences.

count=1
* NeILF++: Inter-Reflectable Light Fields for Geometry and Material Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_NeILF_Inter-Reflectable_Light_Fields_for_Geometry_and_Material_Estimation_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_NeILF_Inter-Reflectable_Light_Fields_for_Geometry_and_Material_Estimation_ICCV_2023_paper.pdf)]
    * Title: NeILF++: Inter-Reflectable Light Fields for Geometry and Material Estimation
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Jingyang Zhang, Yao Yao, Shiwei Li, Jingbo Liu, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan
    * Abstract: We present a novel differentiable rendering framework for joint geometry, material, and lighting estimation from multi-view images. In contrast to previous methods which assume a simplified environment map or co-located flashlights, in this work, we formulate the lighting of a static scene as one neural incident light field (NeILF) and one outgoing neural radiance field (NeRF). The key insight of the proposed method is the union of the incident and outgoing light fields through physically-based rendering and inter-reflections between surfaces, making it possible to disentangle the scene geometry, material, and lighting from image observations in a physically-based manner. The proposed incident light and inter-reflection framework can be easily applied to other NeRF systems. We show that our method can not only decompose the outgoing radiance into incident lights and surface materials, but also serve as a surface refinement module that further improves the reconstruction detail of the neural surface. We demonstrate on several datasets that the proposed method is able to achieve state-of-the-art results in terms of the geometry reconstruction quality, material estimation accuracy, and the fidelity of novel view rendering.

count=1
* Temporal Enhanced Training of Multi-view 3D Object Detector via Historical Object Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zong_Temporal_Enhanced_Training_of_Multi-view_3D_Object_Detector_via_Historical_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zong_Temporal_Enhanced_Training_of_Multi-view_3D_Object_Detector_via_Historical_ICCV_2023_paper.pdf)]
    * Title: Temporal Enhanced Training of Multi-view 3D Object Detector via Historical Object Prediction
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Zhuofan Zong, Dongzhi Jiang, Guanglu Song, Zeyue Xue, Jingyong Su, Hongsheng Li, Yu Liu
    * Abstract: In this paper, we propose a new paradigm, named Historical Object Prediction (HoP) for multi-view 3D detection to leverage temporal information more effectively. The HoP approach is straightforward: given the current timestamp t, we generate a pseudo Bird's-Eye View (BEV) feature of timestamp t-k from its adjacent frames and utilize this feature to predict the object set at timestamp t-k. Our approach is motivated by the observation that enforcing the detector to capture both the spatial location and temporal motion of objects occurring at historical timestamps can lead to more accurate BEV feature learning. First, we elaborately design short-term and long-term temporal decoders, which can generate the pseudo BEV feature for timestamp t-k without the involvement of its corresponding camera images. Second, an additional object decoder is flexibly attached to predict the object targets using the generated pseudo BEV feature. Note that we only perform HoP during training, thus the proposed method does not introduce extra overheads during inference. As a plug-and-play approach, HoP can be easily incorporated into state-of-the-art BEV detection frameworks, including BEVFormer and BEVDet series. Furthermore, the auxiliary HoP approach is complementary to prevalent temporal modeling methods, leading to significant performance gains. Extensive experiments are conducted to evaluate the effectiveness of the proposed HoP on the nuScenes dataset. We choose the representative methods, including BEVFormer and BEVDet4D-Depth to evaluate our method. Surprisingly, HoP achieves 68.2% NDS and 61.6% mAP with ViT-L on nuScenes test, outperforming all the 3D object detectors on the leaderboard by a large margin.

count=1
* Implicit Neural Representation in Medical Imaging: A Comparative Survey
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Molaei_Implicit_Neural_Representation_in_Medical_Imaging_A_Comparative_Survey_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Molaei_Implicit_Neural_Representation_in_Medical_Imaging_A_Comparative_Survey_ICCVW_2023_paper.pdf)]
    * Title: Implicit Neural Representation in Medical Imaging: A Comparative Survey
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Amirali Molaei, Amirhossein Aminimehr, Armin Tavakoli, Amirhossein Kazerouni, Bobby Azad, Reza Azad, Dorit Merhof
    * Abstract: Implicit neural representations (INRs) have emerged as a powerful paradigm in scene reconstruction and computer graphics, showcasing remarkable results. By utilizing neural networks to parameterize data through implicit continuous functions, INRs offer several benefits. Recognizing the potential of INRs beyond these domains, this survey aims to provide a comprehensive overview of INR models in the field of medical imaging. In medical settings, numerous challenging and ill-posed problems exist, making INRs an attractive solution. The survey explores the application of INRs in various medical imaging tasks, such as image reconstruction, segmentation, registration, novel view synthesis, and compression. It discusses the advantages and limitations of INRs, highlighting their resolution-agnostic nature, memory efficiency, ability to avoid locality biases, and differentiability, enabling adaptation to different tasks. Furthermore, the survey addresses the challenges and considerations specific to medical imaging data, such as data availability, computational complexity, and dynamic clinical scene analysis. It also identifies future research directions and opportunities, including integration with multi-modal imaging, real-time and interactive systems, and domain adaptation for clinical decision support. To facilitate further exploration and implementation of INRs in medical image analysis, we have provided a compilation of cited studies along with their available open-source implementations on Github, which will be available after acceptence. Finally, we aim to consistently incorporate the most recent and relevant papers regularly.

count=1
* SteReFo: Efficient Image Refocusing with Stereo Vision
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/AIM/Busam_SteReFo_Efficient_Image_Refocusing_with_Stereo_Vision_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/AIM/Busam_SteReFo_Efficient_Image_Refocusing_with_Stereo_Vision_ICCVW_2019_paper.pdf)]
    * Title: SteReFo: Efficient Image Refocusing with Stereo Vision
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Benjamin Busam, Matthieu Hog, Steven McDonagh, Gregory Slabaugh
    * Abstract: Whether to attract viewer attention to a particular object, give the impression of depth or simply reproduce human-like scene perception, shallow depth of field images are used extensively by professional and amateur photographers alike. To this end, high quality optical systems are used in DSLR cameras to focus on a specific depth plane while producing visually pleasing bokeh. We propose a physically motivated pipeline to mimic this effect from all-in-focus stereo images, typically retrieved by mobile cameras. It is capable to change the focal plane a posteriori at 76 FPS on KITTI images to enable real-time applications. As our portmanteau suggests, SteReFo interrelates stereo-based depth estimation and refocusing efficiently. In contrast to other approaches, our pipeline is simultaneously fully differentiable, physically motivated, and agnostic to scene content. It also enables computational video focus tracking for moving objects in addition to refocusing of static images. We evaluate our approach on the publicly available datasets SceneFlow, KITTI, CityScapes and quantify the quality of architectural changes.

count=1
* Disentangling Pose from Appearance in Monochrome Hand Images
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/HANDS/LI_Disentangling_Pose_from_Appearance_in_Monochrome_Hand_Images_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/HANDS/LI_Disentangling_Pose_from_Appearance_in_Monochrome_Hand_Images_ICCVW_2019_paper.pdf)]
    * Title: Disentangling Pose from Appearance in Monochrome Hand Images
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Yikang LI, Chris Twigg, Yuting Ye, Lingling Tao, Xiaogang Wang
    * Abstract: Hand pose estimation from the monocular 2D image is challenging due to the variation in lighting, appearance, and background. While some success has been achieved using deep neural networks, they typically require collecting a large dataset that adequately samples all the axes of variation of hand images. It would, therefore, be useful to find a representation of hand pose which is independent of the image appearance(like hand texture, lighting, background), so that we can synthesize unseen images by mixing pose-appearance combinations. In this paper, we present a novel technique that disentangles the representation of pose from a complementary appearance factor in 2D monochrome images. We supervise this disentanglement process using a network that learns to generate images of hand using specified pose+ appearance features. Unlike previous work, we do not require image pairs with a matching pose; instead, we use the pose annotations already available and introduce a novel use of cycle consistency to ensure orthogonality between the factors. Experimental results show that our self-disentanglement scheme successfully decomposes the hand image into the pose and its complementary appearance features of comparable quality as the method using paired data. Additionally, training the model with extra synthesized images with unseen hand-appearance combinations by re-mixing pose and appearance factors from different images can improve the 2D pose estimation performance.

count=1
* Learning From Synthetic Photorealistic Raindrop for Single Image Raindrop Removal
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/PBDL/Hao_Learning_From_Synthetic_Photorealistic_Raindrop_for_Single_Image_Raindrop_Removal_ICCVW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/PBDL/Hao_Learning_From_Synthetic_Photorealistic_Raindrop_for_Single_Image_Raindrop_Removal_ICCVW_2019_paper.pdf)]
    * Title: Learning From Synthetic Photorealistic Raindrop for Single Image Raindrop Removal
    * Publisher: ICCV
    * Publication Date: `2019`
    * Authors: Zhixiang Hao, Shaodi You, Yu Li, Kunming Li, Feng Lu
    * Abstract: Raindrops adhered to camera lens or windshield are inevitable in rainy scenes and can become an issue for many computer vision systems such as autonomous driving. Because raindrop appearance is affected by too many parameters, therefore it is unlikely to find an effective model based solution. Learning based methods are also problematic, because traditional learning method cannot properly model the complex appearance. Whereas deep learning method lacks sufficiently large and realistic training data. To solve it, in our work, we propose the first photo-realistic dataset of synthetic adherent raindrops for training. The rendering is physics based with consideration of the water dynamic, geometric and photometry. The dataset contains various types of rainy scenes and particularly the rainy driving scenes. Based on the modeling of raindrop imagery, we introduce a detection network which has the awareness of the raindrop refraction as well as its blurring. Based on that, we propose the removal network that can well recover the image structure. Rigorous experiments demonstrate the state-of-the-art performance of our proposed framework.

count=1
* SymGAN: Orientation Estimation without Annotation for Symmetric Objects
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Ammirato_SymGAN_Orientation_Estimation_without_Annotation_for_Symmetric_Objects_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Ammirato_SymGAN_Orientation_Estimation_without_Annotation_for_Symmetric_Objects_WACV_2020_paper.pdf)]
    * Title: SymGAN: Orientation Estimation without Annotation for Symmetric Objects
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Phil Ammirato,  Jonathan Tremblay,  Ming-Yu Liu,  Alexander Berg,  Dieter Fox
    * Abstract: Training a computer vision system to predict an object's pose is crucial to improving robotic manipulation, where robots can easily locate and then grasp objects. Some of the key challenges in pose estimation lie in obtaining labeled data and handling objects with symmetries. We explore both these problems of viewpoint estimation (object 3D orientation) by proposing a novel unsupervised training paradigm that only requires a 3D model of the object of interest. We show that we can successfully train an orientation detector, which simply consumes an RGB image, in an adversarial training framework, where the discriminator learns to provide a learning signal to retrieve the object orientation using a black-box non differentiable renderer. In order to overcome this non differentiability, we introduce a randomized sampling method to obtain training gradients. To our knowledge this is the first time an adversarial framework is employed to successfully train a viewpoint detector that can handle symmetric objects.Using this training framework we show state of the art results on 3D orientation prediction on T-LESS, a challenging dataset for texture-less and symmetric objects.

count=1
* Kornia: an Open Source Differentiable Computer Vision Library for PyTorch
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Riba_Kornia_an_Open_Source_Differentiable_Computer_Vision_Library_for_PyTorch_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Riba_Kornia_an_Open_Source_Differentiable_Computer_Vision_Library_for_PyTorch_WACV_2020_paper.pdf)]
    * Title: Kornia: an Open Source Differentiable Computer Vision Library for PyTorch
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Edgar Riba,  Dmytro Mishkin,  Daniel Ponsa,  Ethan Rublee,  Gary Bradski
    * Abstract: This work presents Kornia -- an open source computer vision library which consists of a set of differentiable routines and modules to solve generic computer vision problems. At its core, the package uses PyTorch as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions. Inspired by OpenCV, Kornia is composed of a set of modules containing operators that can be inserted inside neural networks to train models to perform image transformations, camera calibration, epipolar geometry, and low level image processing techniques such as filtering and edge detection that operate directly on high dimensional tensor representations. Examples of classical vision problems implemented using our framework are also provided including a benchmark comparing to existing vision libraries.

count=1
* EDEN: Multimodal Synthetic Dataset of Enclosed GarDEN Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Le_EDEN_Multimodal_Synthetic_Dataset_of_Enclosed_GarDEN_Scenes_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Le_EDEN_Multimodal_Synthetic_Dataset_of_Enclosed_GarDEN_Scenes_WACV_2021_paper.pdf)]
    * Title: EDEN: Multimodal Synthetic Dataset of Enclosed GarDEN Scenes
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Hoang-An Le, Thomas Mensink, Partha Das, Sezer Karaoglu, Theo Gevers
    * Abstract: Multimodal large-scale datasets for outdoor scenes are mostly designed for urban driving problems. The scenes are highly structured and semantically different from scenarios seen in nature-centered scenes such as gardens or parks. To promote machine learning methods for nature-oriented applications, such as agriculture and gardening, we propose the multimodal synthetic dataset for Enclosed garDEN scenes (EDEN). The dataset features more than 300K images captured from more than 100 garden models. Each image is annotated with various low/high-level vision modalities, including semantic segmentation, depth, surface normals, intrinsic colors, and optical flow. Experimental results on the state-of-the-art methods for semantic segmentation and monocular depth prediction, two important tasks in computer vision, show positive impact of pre-training deep networks on our dataset for unstructured natural scenes. The dataset and related materials will be available at https://lhoangan.github.io/eden.

count=1
* Adaptiope: A Modern Benchmark for Unsupervised Domain Adaptation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Ringwald_Adaptiope_A_Modern_Benchmark_for_Unsupervised_Domain_Adaptation_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Ringwald_Adaptiope_A_Modern_Benchmark_for_Unsupervised_Domain_Adaptation_WACV_2021_paper.pdf)]
    * Title: Adaptiope: A Modern Benchmark for Unsupervised Domain Adaptation
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Tobias Ringwald, Rainer Stiefelhagen
    * Abstract: Unsupervised domain adaptation (UDA) deals with the adaptation process of a given source domain with labeled training data to a target domain for which only unannotated data is available. This is a challenging task as the domain shift leads to degraded performance on the target domain data if not addressed. In this paper, we analyze commonly used UDA classification datasets and discover systematic problems with regard to dataset setup, ground truth ambiguity and annotation quality. We manually clean the most popular UDA dataset in the research area (Office-31) and quantify the negative effects of inaccurate annotations through thorough experiments. Based on these insights, we collect the Adaptiope dataset - a large scale, diverse UDA dataset with synthetic, product and real world data - and show that its transfer tasks provide a challenge even when considering recent UDA algorithms. Our datasets are available at https://gitlab.com/tringwald/adaptiope.

count=1
* DeepCSR: A 3D Deep Learning Approach for Cortical Surface Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Santa_Cruz_DeepCSR_A_3D_Deep_Learning_Approach_for_Cortical_Surface_Reconstruction_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Santa_Cruz_DeepCSR_A_3D_Deep_Learning_Approach_for_Cortical_Surface_Reconstruction_WACV_2021_paper.pdf)]
    * Title: DeepCSR: A 3D Deep Learning Approach for Cortical Surface Reconstruction
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Rodrigo Santa Cruz, Leo Lebrat, Pierrick Bourgeat, Clinton Fookes, Jurgen Fripp, Olivier Salvado
    * Abstract: The study of neurodegenerative diseases relies on the reconstruction and analysis of the brain cortex from magnetic resonance imaging (MRI). Traditional frameworks for this task like FreeSurfer demand lengthy runtimes, while its accelerated variant FastSurfer still relies on a voxel-wise segmentation which is limited by its resolution to capture narrow continuous objects as cortical surfaces. Having these limitations in mind, we propose DeepCSR, a 3D deep learning framework for cortical surface reconstruction from MRI. Towards this end, we train a neural network model with hypercolumn features to predict implicit surface representations for points in a brain template space. After training, the cortical surface at a desired level of detail is obtained by evaluating surface representations at specific coordinates, and subsequently applying a topology correction algorithm and an isosurface extraction method. Thanks to the continuous nature of this approach and the efficacy of its hypercolumn features scheme, DeepCSR efficiently reconstructs cortical surfaces at high resolution capturing fine details in the cortical folding. Moreover, DeepCSR is as accurate, more precise, and faster than the widely used FreeSurfer toolbox and its deep learning powered variant FastSurfer on reconstructing cortical surfaces from MRI which should facilitate large-scale medical studies and new healthcare applications.

count=1
* Fast and Explicit Neural View Synthesis
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Guo_Fast_and_Explicit_Neural_View_Synthesis_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Guo_Fast_and_Explicit_Neural_View_Synthesis_WACV_2022_paper.pdf)]
    * Title: Fast and Explicit Neural View Synthesis
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Pengsheng Guo, Miguel Angel Bautista, Alex Colburn, Liang Yang, Daniel Ulbricht, Joshua M. Susskind, Qi Shan
    * Abstract: We study the problem of novel view synthesis from sparse source observations of a scene comprised of 3D objects. We propose a simple yet effective approach that is neither continuous nor implicit, challenging recent trends on view synthesis. Our approach explicitly encodes observations into a volumetric representation that enables amortized rendering. We demonstrate that although continuous radiance field representations have gained a lot of attention due to their expressive power, our simple approach obtains comparable or even better novel view reconstruction quality comparing with state-of-the-art baselines while increasing rendering speed by over 400x. Our model is trained in a category-agnostic manner and does not require scene-specific optimization. Therefore, it is able to generalize novel view synthesis to object categories not seen during training. In addition, we show that with our simple formulation, we can use view synthesis as a self-supervision signal for efficient learning of 3D geometry without explicit 3D supervision.

count=1
* SBEVNet: End-to-End Deep Stereo Layout Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Gupta_SBEVNet_End-to-End_Deep_Stereo_Layout_Estimation_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Gupta_SBEVNet_End-to-End_Deep_Stereo_Layout_Estimation_WACV_2022_paper.pdf)]
    * Title: SBEVNet: End-to-End Deep Stereo Layout Estimation
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Divam Gupta, Wei Pu, Trenton Tabor, Jeff Schneider
    * Abstract: Accurate layout estimation is crucial for planning and navigation in robotics applications, such as self-driving. In this paper, we introduce the Stereo Bird's Eye ViewNetwork (SBEVNet), a novel supervised end-to-end framework for estimation of bird's eye view layout from a pair of stereo images. Although our network reuses some of the building blocks from the state-of-the-art deep learning networks for disparity estimation, we show that explicit depth estimation is neither sufficient nor necessary. Instead, the learning of a good internal bird's eye view feature representation is effective for layout estimation. Specifically, we first generate a disparity feature volume using the features of the stereo images and then project it to the bird's eye view coordinates. This gives us coarse-grained information about the scene structure. We also apply inverse perspective mapping (IPM) to map the input images and their features to the bird's eye view. This gives us fine-grained texture information. Concatenating IPM features with the projected feature volume creates a rich bird's eye view representation which is useful for spatial reasoning. We use this representation to estimate the BEV semantic map. Additionally, we show that using the IPM features as a supervisory signal for stereo features can give an improvement in performance. We demonstrate our approach on two datasets:the KITTI dataset and a synthetically generated dataset from the CARLA simulator. For both of these datasets, we establish state-of-the-art performance compared to baseline techniques.

count=1
* Neural Radiance Fields Approach to Deep Multi-View Photometric Stereo
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Kaya_Neural_Radiance_Fields_Approach_to_Deep_Multi-View_Photometric_Stereo_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Kaya_Neural_Radiance_Fields_Approach_to_Deep_Multi-View_Photometric_Stereo_WACV_2022_paper.pdf)]
    * Title: Neural Radiance Fields Approach to Deep Multi-View Photometric Stereo
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Berk Kaya, Suryansh Kumar, Francesco Sarno, Vittorio Ferrari, Luc Van Gool
    * Abstract: We present a modern solution to the multi-view photometric stereo problem (MVPS). Our work suitably exploits the image formation model in a MVPS experimental setup to recover the dense 3D reconstruction of an object from images. We procure the surface orientation using a photometric stereo (PS) image formation model and blend it with a multi-view neural radiance field representation to recover the object's surface geometry. Contrary to the previous multi-staged framework to MVPS, where the position, iso-depth contours, or orientation measurements are estimated independently and then fused later, our method is simple to implement and realize. Our method performs neural rendering of multi-view images while utilizing surface normals estimated by a deep photometric stereo network. We render the MVPS images by considering the object's surface normals for each 3D sample point along the viewing direction rather than explicitly using the density gradient in the volume space via 3D occupancy information. We optimize the proposed neural radiance field representation for the MVPS setup efficiently using a fully connected deep network to recover the 3D geometry of an object. Extensive evaluation on the DiLiGenT-MV benchmark dataset shows that our method performs better than the approaches that perform only PS or only multi-view stereo (MVS) and provides comparable results against the state-of-the-art multi-stage fusion methods.

count=1
* Style Agnostic 3D Reconstruction via Adversarial Style Transfer
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Petersen_Style_Agnostic_3D_Reconstruction_via_Adversarial_Style_Transfer_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Petersen_Style_Agnostic_3D_Reconstruction_via_Adversarial_Style_Transfer_WACV_2022_paper.pdf)]
    * Title: Style Agnostic 3D Reconstruction via Adversarial Style Transfer
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Felix Petersen, Bastian Goldluecke, Oliver Deussen, Hilde Kuehne
    * Abstract: Reconstructing the 3D geometry of an object from an image is a major challenge in computer vision. Recently introduced differentiable renderers can be leveraged to learn the 3D geometry of objects from 2D images, but those approaches require additional supervision to enable the renderer to produce an output that can be compared to the input image. This can be scene information or constraints such as object silhouettes, uniform backgrounds, material, texture, and lighting. In this paper, we propose an approach that enables a differentiable rendering-based learning of 3D objects from images with backgrounds without the need for silhouette supervision. Instead of trying to render an image close to the input, we propose an adversarial style-transfer and domain adaptation pipeline that allows to translate the input image domain to the rendered image domain. This allows us to directly compare between a translated image and the differentiable rendering of a 3D object reconstruction in order to train the 3D object reconstruction network. We show that the approach learns 3D geometry from images with backgrounds and provides a better performance than constrained methods for single-view 3D object reconstruction on this task.

count=1
* SIRA: Relightable Avatars From a Single Image
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Caselles_SIRA_Relightable_Avatars_From_a_Single_Image_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Caselles_SIRA_Relightable_Avatars_From_a_Single_Image_WACV_2023_paper.pdf)]
    * Title: SIRA: Relightable Avatars From a Single Image
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Pol Caselles, Eduard Ramon, Jaime Garcia, Xavier Giro-i-Nieto, Francesc Moreno-Noguer, Gil Triginer
    * Abstract: Recovering the geometry of a human head from a single image, while factorizing the materials and illumination is a severely ill-posed problem that requires prior information to be solved. Methods based on 3D Morphable Models (3DMM), and their combination with differentiable renderers, have shown promising results. However, the expressiveness of 3DMMs is limited, and they typically yield over-smoothed and identity-agnostic 3D shapes limited to the face region. Highly accurate full head reconstructions have recently been obtained with neural fields that parameterize the geometry using multilayer perceptrons. The versatility of these representations has also proved effective for disentangling geometry, materials and lighting. However, these methods require several tens of input images. In this paper, we introduce SIRA, a method which, from a single image, reconstructs human head avatars with high fidelity geometry and factorized lights and surface materials. Our key ingredients are two data-driven statistical models based on neural fields that resolve the ambiguities of single-view 3D surface reconstruction and appearance factorization. Experiments show that SIRA obtains state of the art results in 3D head reconstruction while at the same time it successfully disentangles the global illumination, and the diffuse and specular albedos. Furthermore, our reconstructions are amenable to physically-based appearance editing and head model relighting.

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
* Fast Differentiable Transient Rendering for Non-Line-of-Sight Reconstruction
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Plack_Fast_Differentiable_Transient_Rendering_for_Non-Line-of-Sight_Reconstruction_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Plack_Fast_Differentiable_Transient_Rendering_for_Non-Line-of-Sight_Reconstruction_WACV_2023_paper.pdf)]
    * Title: Fast Differentiable Transient Rendering for Non-Line-of-Sight Reconstruction
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Markus Plack, Clara Callenberg, Monika Schneider, Matthias B. Hullin
    * Abstract: Research into non-line-of-sight imaging problems has gained momentum in recent years motivated by intriguing prospective applications in e.g. medicine and autonomous driving. While transient image formation is well understood and there exist various reconstruction approaches for non-line-of-sight scenes that combine efficient forward renderers with optimization schemes, those approaches suffer from runtimes in the order of hours even for moderately sized scenes. Furthermore, the ill-posedness of the inverse problem often leads to instabilities in the optimization. Inspired by the latest advances in direct-line-of-sight inverse rendering that have led to stunning results for reconstructing scene geometry and appearance, we present a fast differentiable transient renderer that accelerates the inverse rendering runtime to minutes on consumer hardware, making it possible to apply inverse transient imaging on a wider range of tasks and in more time-critical scenarios. We demonstrate its effectiveness on a series of applications using various datasets and show that it can be used for self-supervised learning.

count=1
* Adversarial Robustness in Discontinuous Spaces via Alternating Sampling & Descent
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Venkatesh_Adversarial_Robustness_in_Discontinuous_Spaces_via_Alternating_Sampling__Descent_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Venkatesh_Adversarial_Robustness_in_Discontinuous_Spaces_via_Alternating_Sampling__Descent_WACV_2023_paper.pdf)]
    * Title: Adversarial Robustness in Discontinuous Spaces via Alternating Sampling & Descent
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Rahul Venkatesh, Eric Wong, Zico Kolter
    * Abstract: Several works have shown that deep learning models are vulnerable to adversarial attacks where seemingly simple label-preserving changes to the input image lead to incorrect predictions. To combat this, gradient based adversarial training is generally employed as a standard defense mechanism. However, in cases where the loss landscape is discontinuous with respect to a given perturbation set, first order methods get stuck in local optima, and fail to defend against threat. This is often a problem for many physically realizable perturbation sets such as 2D affine transformations and 3D scene parameters. To work in such settings, we introduce a new optimization framework that alternates between global zeroth order sampling and local gradient updates to compute strong adversaries that can be used to harden the model against attack. Further, we design a powerful optimization algorithm using this framework, called Alternating Evolutionary Sampling and Descent (ASD), which combines an evolutionary search strategy (viz. covariance matrix adaptation) with gradient descent. We consider two settings with discontinuous/discrete and non-convex loss landscapes to evaluate ASD: a) 3D scene parameters and b) 2D patch attacks, and find that it achieves state-of-the-art results on adversarial robustness.

count=1
* Learning-Based Spotlight Position Optimization for Non-Line-of-Sight Human Localization and Posture Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Chandran_Learning-Based_Spotlight_Position_Optimization_for_Non-Line-of-Sight_Human_Localization_and_Posture_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Chandran_Learning-Based_Spotlight_Position_Optimization_for_Non-Line-of-Sight_Human_Localization_and_Posture_WACV_2024_paper.pdf)]
    * Title: Learning-Based Spotlight Position Optimization for Non-Line-of-Sight Human Localization and Posture Classification
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Sreenithy Chandran, Tatsuya Yatagawa, Hiroyuki Kubo, Suren Jayasuriya
    * Abstract: Non-line-of-sight imaging (NLOS) is the process of estimating information about a scene that is hidden from the direct line of sight of the camera. NLOS imaging typically requires time-resolved detectors and a laser source for illumination, which are both expensive and computationally intensive to handle. In this paper, we propose an NLOS-based localization and posture classification technique that works on a system of an off-the-shelf projector and camera. We leverage a message-passing neural network to learn a scene geometry and predict the best position to be spotlighted by the projector that can maximize the NLOS signal. The training of the neural network is performed in an end-to-end manner. Therefore, the ground truth spotlighted position is unnecessary during the training, and the network parameters are optimized to maximize the NLOS performance. Unlike prior deep-learning-based NLOS techniques that assume planar relay walls, our system allows us to handle line-of-sight scenes where scene geometries are more arbitrary. Our method demonstrates state-of-the-art performance in object localization and position classification using both synthetic and real scenes.

count=1
* Ray Deformation Networks for Novel View Synthesis of Refractive Objects
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Deng_Ray_Deformation_Networks_for_Novel_View_Synthesis_of_Refractive_Objects_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Deng_Ray_Deformation_Networks_for_Novel_View_Synthesis_of_Refractive_Objects_WACV_2024_paper.pdf)]
    * Title: Ray Deformation Networks for Novel View Synthesis of Refractive Objects
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Weijian Deng, Dylan Campbell, Chunyi Sun, Shubham Kanitkar, Matthew Shaffer, Stephen Gould
    * Abstract: Neural Radiance Fields (NeRF) have demonstrated exceptional capabilities in creating photorealistic novel views using volume rendering on a radiance field. However, the intrinsic assumption of straight light rays within NeRF becomes a limitation when dealing with transparent or translucent objects that exhibit refraction, and therefore have curved light paths. This hampers the ability of these approaches to accurately model the appearance of refractive objects, resulting in suboptimal novel view synthesis and geometry estimates. To address this issue, we propose an innovative solution using deformable networks to learn a tailored deformation field for refractive objects. Our approach predicts position and direction offsets, allowing NeRF to model the curved light paths caused by refraction and therefore the complex and highly view-dependent appearances of refractive objects. We also introduce a regularization strategy that encourages piece-wise linear light paths, since most physical systems can be approximated with a piece-wise constant index of refraction. By seamlessly integrating our deformation networks into the NeRF framework, our method achieves significant improvements in rendering refractive objects from novel views.

count=1
* Specular Object Reconstruction Behind Frosted Glass by Differentiable Rendering
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Iwaguchi_Specular_Object_Reconstruction_Behind_Frosted_Glass_by_Differentiable_Rendering_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Iwaguchi_Specular_Object_Reconstruction_Behind_Frosted_Glass_by_Differentiable_Rendering_WACV_2024_paper.pdf)]
    * Title: Specular Object Reconstruction Behind Frosted Glass by Differentiable Rendering
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Takafumi Iwaguchi, Hiroyuki Kubo, Hiroshi Kawasaki
    * Abstract: This paper addresses the problem of reconstructing scenes behind optical diffusers, which is common in applications such as imaging through frosted glass. We propose a new approach that exploits specular reflection to capture sharp light distributions with a point light source, which can be used to detect reflections in low signal-to-noise scenarios. In this paper, we propose a rasterizer-based differentiable renderer to solve this problem by minimizing the difference between the captured and rendered images. Because our method can simultaneously optimize multiple observations for different light source positions, it is confirmed that ambiguities of the scene are efficiently eliminated by increasing the number of observations. Experiments show that the proposed method can reconstruct a scene with several mirror-like objects behind the diffuser in both simulated and real environments.

count=1
* Intrinsic Hand Avatar: Illumination-Aware Hand Appearance and Shape Reconstruction From Monocular RGB Video
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Kalshetti_Intrinsic_Hand_Avatar_Illumination-Aware_Hand_Appearance_and_Shape_Reconstruction_From_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kalshetti_Intrinsic_Hand_Avatar_Illumination-Aware_Hand_Appearance_and_Shape_Reconstruction_From_WACV_2024_paper.pdf)]
    * Title: Intrinsic Hand Avatar: Illumination-Aware Hand Appearance and Shape Reconstruction From Monocular RGB Video
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Pratik Kalshetti, Parag Chaudhuri
    * Abstract: Reconstructing a user-specific hand avatar is essential for a personalized experience in augmented and virtual reality systems. Current state-of-the-art avatar reconstruction methods use implicit representations to capture detailed geometry and appearance combined with neural rendering. However, these methods rely on a complicated multi-view setup, do not explicitly handle environment lighting leading to baked-in illumination and self-shadows, and require long hours for training. We present a method to reconstruct a hand avatar from a monocular RGB video of a user's hand in arbitrary hand poses captured under real-world environment lighting. Specifically, our method jointly optimizes shape, appearance, and lighting parameters using a realistic shading model in a differentiable rendering framework incorporating Monte Carlo path tracing. Despite relying on physically-based rendering, our method can complete the reconstruction within minutes. In contrast to existing work, our method disentangles intrinsic properties of the underlying appearance and environment lighting, leading to realistic self-shadows. We compare our method with state-of-the-art hand avatar reconstruction methods and observe that it outperforms them on all commonly used metrics. We also evaluate our method on our captured dataset to emphasize its generalization capability. Finally, we demonstrate applications of our intrinsic hand avatar on novel pose synthesis and relighting. We plan to release our code to aid further research.

count=1
* RGB-D Mapping and Tracking in a Plenoxel Radiance Field
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Teigen_RGB-D_Mapping_and_Tracking_in_a_Plenoxel_Radiance_Field_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Teigen_RGB-D_Mapping_and_Tracking_in_a_Plenoxel_Radiance_Field_WACV_2024_paper.pdf)]
    * Title: RGB-D Mapping and Tracking in a Plenoxel Radiance Field
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Andreas L. Teigen, Yeonsoo Park, Annette Stahl, Rudolf Mester
    * Abstract: The widespread adoption of Neural Radiance Fields (NeRFs) have ensured significant advances in the domain of novel view synthesis in recent years. These models capture a volumetric radiance field of a scene, creating highly convincing, dense, photorealistic models through the use of simple, differentiable rendering equations. Despite their popularity, these algorithms suffer from severe ambiguities in visual data inherent to the RGB sensor, which means that although images generated with view synthesis can visually appear very believable, the underlying 3D model will often be wrong. This considerably limits the usefulness of these models in practical applications like Robotics and Extended Reality (XR), where an accurate dense 3D reconstruction otherwise would be of significant value. In this paper, we present the vital differences between view synthesis models and 3D reconstruction models. We also comment on why a depth sensor is essential for modeling accurate geometry in general outward-facing scenes using the current paradigm of novel view synthesis methods. Focusing on the structure-from-motion task, we practically demonstrate this need by extending the Plenoxel radiance field model: Presenting an analytical differential approach for dense mapping and tracking with radiance fields based on RGB-D data without a neural network. Our method achieves state-of-the-art results in both mapping and tracking tasks, while also being faster than competing neural network-based approaches. The code is available at: https://github.com/ysus33/RGB-D_Plenoxel_Mapping_Tracking.git

count=1
* Hyb-NeRF: A Multiresolution Hybrid Encoding for Neural Radiance Fields
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Wang_Hyb-NeRF_A_Multiresolution_Hybrid_Encoding_for_Neural_Radiance_Fields_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Wang_Hyb-NeRF_A_Multiresolution_Hybrid_Encoding_for_Neural_Radiance_Fields_WACV_2024_paper.pdf)]
    * Title: Hyb-NeRF: A Multiresolution Hybrid Encoding for Neural Radiance Fields
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Yifan Wang, Yi Gong, Yuan Zeng
    * Abstract: Recent advances in Neural radiance fields (NeRF) have enabled high-fidelity scene reconstruction for novel view synthesis. However, NeRF requires hundreds of network evaluations per pixel to approximate a volume rendering integral, making it slow to train. Caching NeRFs into explicit data structures can effectively enhance rendering speed but at the cost of higher memory usage. To address these issues, we present Hyb-NeRF, a novel neural radiance field with a multi-resolution hybrid encoding that achieves efficient neural modeling and fast rendering, which also allows for high-quality novel view synthesis. The key idea of Hyb-NeRF is to represent the scene using different encoding strategies from coarse-to-fine resolution levels. Hyb-NeRF exploits coherence and compact memory of learnable positional features at coarse resolutions and the fast optimization speed and local details of hash-based feature grids at fine resolutions. In addition, to further boost performance, we embed cone tracing-based Fourier features in our learnable positional encoding that eliminates encoding ambiguity and reduces aliasing artifacts. Extensive experiments on both synthetic and real-world datasets show that Hyb-NeRF achieves faster rendering speed with better rending quality and even a lower memory footprint in comparison to previous state-of-the-art methods.

count=1
* FIRe: Fast Inverse Rendering Using Directional and Signed Distance Functions
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Yenamandra_FIRe_Fast_Inverse_Rendering_Using_Directional_and_Signed_Distance_Functions_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Yenamandra_FIRe_Fast_Inverse_Rendering_Using_Directional_and_Signed_Distance_Functions_WACV_2024_paper.pdf)]
    * Title: FIRe: Fast Inverse Rendering Using Directional and Signed Distance Functions
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Tarun Yenamandra, Ayush Tewari, Nan Yang, Florian Bernard, Christian Theobalt, Daniel Cremers
    * Abstract: Neural 3D implicit representations learn priors that are useful for diverse applications, such as single- or multiple-view 3D reconstruction. A major downside of existing approaches while rendering an image is that they require evaluating the network multiple times per camera ray so that the high computational time forms a bottleneck for downstream applications. We address this problem by introducing a novel neural scene representation that we call the directional distance function (DDF). To this end, we learn a signed distance function (SDF) along with our DDF model to represent a class of shapes. Specifically, our DDF is defined on the unit sphere and predicts the distance to the surface along any given direction. Therefore, our DDF allows rendering images with just a single network evaluation per camera ray. Based on our DDF, we present a novel fast algorithm (FIRe) to reconstruct 3D shapes given a posed depth map. We evaluate our proposed method on 3D reconstruction from single-view depth images, where we empirically show that our algorithm reconstructs 3D shapes more accurately and it is more than 15 times faster (per iteration) than competing methods.

count=1
* Infinite Factorial Dynamical Model
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2015/hash/0768281a05da9f27df178b5c39a51263-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2015/file/0768281a05da9f27df178b5c39a51263-Paper.pdf)]
    * Title: Infinite Factorial Dynamical Model
    * Publisher: NeurIPS
    * Publication Date: `2015`
    * Authors: Isabel Valera, Francisco Ruiz, Lennart Svensson, Fernando Perez-Cruz
    * Abstract: We propose the infinite factorial dynamic model (iFDM), a general Bayesian nonparametric model for source separation. Our model builds on the Markov Indian buffet process to consider a potentially unbounded number of hidden Markov chains (sources) that evolve independently according to some dynamics, in which the state space can be either discrete or continuous. For posterior inference, we develop an algorithm based on particle Gibbs with ancestor sampling that can be efficiently applied to a wide range of source separation problems. We evaluate the performance of our iFDM on four well-known applications: multitarget tracking, cocktail party, power disaggregation, and multiuser detection. Our experimental results show that our approach for source separation does not only outperform previous approaches, but it can also handle problems that were computationally intractable for existing approaches.

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
* Write, Execute, Assess: Program Synthesis with a REPL
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/50d2d2262762648589b1943078712aa6-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/50d2d2262762648589b1943078712aa6-Paper.pdf)]
    * Title: Write, Execute, Assess: Program Synthesis with a REPL
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Kevin Ellis, Maxwell Nye, Yewen Pu, Felix Sosa, Josh Tenenbaum, Armando Solar-Lezama
    * Abstract: We present a neural program synthesis approach integrating components which write, execute, and assess code to navigate the search space of possible programs. We equip the search process with an interpreter or a read-eval-print-loop (REPL), which immediately executes partially written programs, exposing their semantics. The REPL addresses a basic challenge of program synthesis: tiny changes in syntax can lead to huge changes in semantics. We train a pair of models, a policy that proposes the new piece of code to write, and a value function that assesses the prospects of the code written so-far. At test time we can combine these models with a Sequential Monte Carlo algorithm. We apply our approach to two domains: synthesizing text editing programs and inferring 2D and 3D graphics programs.

count=1
* Residual Flows for Invertible Generative Modeling
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/5d0d5594d24f0f955548f0fc0ff83d10-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/5d0d5594d24f0f955548f0fc0ff83d10-Paper.pdf)]
    * Title: Residual Flows for Invertible Generative Modeling
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Ricky T. Q. Chen, Jens Behrmann, David K. Duvenaud, Joern-Henrik Jacobsen
    * Abstract: Flow-based generative models parameterize probability distributions through an invertible transformation and can be trained by maximum likelihood. Invertible residual networks provide a flexible family of transformations where only Lipschitz conditions rather than strict architectural constraints are needed for enforcing invertibility. However, prior work trained invertible residual networks for density estimation by relying on biased log-density estimates whose bias increased with the network's expressiveness. We give a tractable unbiased estimate of the log density, and reduce the memory required during training by a factor of ten. Furthermore, we improve invertible residual blocks by proposing the use of activation functions that avoid gradient saturation and generalizing the Lipschitz condition to induced mixed norms. The resulting approach, called Residual Flows, achieves state-of-the-art performance on density estimation amongst flow-based models, and outperforms networks that use coupling blocks at joint generative and discriminative modeling.

count=1
* Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/b5dc4e5d9b495d0196f61d45b26ef33e-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/b5dc4e5d9b495d0196f61d45b26ef33e-Paper.pdf)]
    * Title: Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Vincent Sitzmann, Michael Zollhoefer, Gordon Wetzstein
    * Abstract: Unsupervised learning with generative models has the potential of discovering rich representations of 3D scenes. While geometric deep learning has explored 3D-structure-aware representations of scene geometry, these models typically require explicit 3D supervision. Emerging neural scene representations can be trained only with posed 2D images, but existing methods ignore the three-dimensional structure of scenes. We propose Scene Representation Networks (SRNs), a continuous, 3D-structure-aware scene representation that encodes both geometry and appearance. SRNs represent scenes as continuous functions that map world coordinates to a feature representation of local scene properties. By formulating the image formation as a differentiable ray-marching algorithm, SRNs can be trained end-to-end from only 2D images and their camera poses, without access to depth or shape. This formulation naturally generalizes across scenes, learning powerful geometry and appearance priors in the process. We demonstrate the potential of SRNs by evaluating them for novel view synthesis, few-shot reconstruction, joint shape and appearance interpolation, and unsupervised discovery of a non-rigid face model.

count=1
* Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic  Flows
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/1349b36b01e0e804a6c2909a6d0ec72a-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/1349b36b01e0e804a6c2909a6d0ec72a-Paper.pdf)]
    * Title: Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic  Flows
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Kunal Gupta, Manmohan Chandraker
    * Abstract: Meshes are important representations of physical 3D entities in the virtual world. Applications like rendering, simulations and 3D printing require meshes to be manifold so that they can interact with the world like the real objects they represent. Prior methods generate meshes with great geometric accuracy but poor manifoldness. In this work, we propose NeuralMeshFlow (NMF) to generate two-manifold meshes for genus-0 shapes. Specifically, NMF is a shape auto-encoder consisting of several Neural Ordinary Differential Equation (NODE)(1) blocks that learn accurate mesh geometry by progressively deforming a spherical mesh. Training NMF is simpler compared to state-of-the-art methods since it does not require any explicit mesh-based regularization. Our experiments demonstrate that NMF facilitates several applications such as single-view mesh reconstruction, global shape parameterization, texture mapping, shape deformation and correspondence. Importantly, we demonstrate that manifold meshes generated using NMF are better-suited for physically-based rendering and simulation compared to prior works.

count=1
* Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf)]
    * Title: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, Ren Ng
    * Abstract: We show that passing input points through a simple Fourier feature mapping enables a multilayer perceptron (MLP) to learn high-frequency functions in low-dimensional problem domains. These results shed light on recent advances in computer vision and graphics that achieve state-of-the-art results by using MLPs to represent complex 3D objects and scenes. Using tools from the neural tangent kernel (NTK) literature, we show that a standard MLP has impractically slow convergence to high frequency signal components. To overcome this spectral bias, we use a Fourier feature mapping to transform the effective NTK into a stationary kernel with a tunable bandwidth. We suggest an approach for selecting problem-specific Fourier features that greatly improves the performance of MLPs for low-dimensional regression tasks relevant to the computer vision and graphics communities.

count=1
* MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/6e01383fd96a17ae51cc3e15447e7533-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/6e01383fd96a17ae51cc3e15447e7533-Paper.pdf)]
    * Title: MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Saim Wani, Shivansh Patel, Unnat Jain, Angel Chang, Manolis Savva
    * Abstract: Navigation tasks in photorealistic 3D environments are challenging because they require perception and effective planning under partial observability. Recent work shows that map-like memory is useful for long-horizon navigation tasks. However, a focused investigation of the impact of maps on navigation tasks of varying complexity has not yet been performed. We propose the multiON task, which requires navigation to an episode-specific sequence of objects in a realistic environment. MultiON generalizes the ObjectGoal navigation task and explicitly tests the ability of navigation agents to locate previously observed goal objects. We perform a set of multiON experiments to examine how a variety of agent models perform across a spectrum of navigation task complexities. Our experiments show that: i) navigation performance degrades dramatically with escalating task complexity; ii) a simple semantic map agent performs surprisingly well relative to more complex neural image feature map agents; and iii) even oracle map agents achieve relatively low performance, indicating the potential for future work in training embodied navigation agents using maps.

count=1
* 3D Shape Reconstruction from Vision and Touch
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/a3842ed7b3d0fe3ac263bcabd2999790-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/a3842ed7b3d0fe3ac263bcabd2999790-Paper.pdf)]
    * Title: 3D Shape Reconstruction from Vision and Touch
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Edward Smith, Roberto Calandra, Adriana Romero, Georgia Gkioxari, David Meger, Jitendra Malik, Michal Drozdzal
    * Abstract: When a toddler is presented a new toy, their instinctual behaviour is to pick it up and inspect it with their hand and eyes in tandem, clearly searching over its surface to properly understand what they are playing with. At any instance here, touch provides high fidelity localized information while vision provides complementary global context. However, in 3D shape reconstruction, the complementary fusion of visual and haptic modalities remains largely unexplored. In this paper, we study this problem and present an effective chart-based approach to multi-modal shape understanding which encourages a similar fusion vision and touch information. To do so, we introduce a dataset of simulated touch and vision signals from the interaction between a robotic hand and a large array of 3D objects. Our results show that (1) leveraging both vision and touch signals consistently improves single- modality baselines; (2) our approach outperforms alternative modality fusion methods and strongly benefits from the proposed chart-based structure; (3) the reconstruction quality increases with the number of grasps provided; and (4) the touch information not only enhances the reconstruction at the touch site but also extrapolates to its local neighborhood.

count=1
* GRAF
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/e92e1b476bb5262d793fd40931e0ed53-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/e92e1b476bb5262d793fd40931e0ed53-Paper.pdf)]
    * Title: GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Katja Schwarz, Yiyi Liao, Michael Niemeyer, Andreas Geiger
    * Abstract: While 2D generative adversarial networks have enabled high-resolution image synthesis, they largely lack an understanding of the 3D world and the image formation process. Thus, they do not provide precise control over camera viewpoint or object pose. To address this problem, several recent approaches leverage intermediate voxel-based representations in combination with differentiable rendering. However, existing methods either produce low image resolution or fall short in disentangling camera and scene properties, e.g., the object identity may vary with the viewpoint. In this paper, we propose a generative model for radiance fields which have recently proven successful for novel view synthesis of a single scene. In contrast to voxel-based representations, radiance fields are not confined to a coarse discretization of the 3D space, yet allow for disentangling camera and scene properties while degrading gracefully in the presence of reconstruction ambiguity. By introducing a multi-scale patch-based discriminator, we demonstrate synthesis of high-resolution images while training our model from unposed 2D images alone. We systematically analyze our approach on several challenging synthetic and real-world datasets. Our experiments reveal that radiance fields are a powerful representation for generative image synthesis, leading to 3D consistent models that render with high fidelity.

count=1
* BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf)]
    * Title: BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G. Wilson, Eytan Bakshy
    * Abstract: Bayesian optimization provides sample-efficient global optimization for a broad range of applications, including automatic machine learning, engineering, physics, and experimental design. We introduce BoTorch, a modern programming framework for Bayesian optimization that combines Monte-Carlo (MC) acquisition functions, a novel sample average approximation optimization approach, auto-differentiation, and variance reduction techniques. BoTorch's modular design facilitates flexible specification and optimization of probabilistic models written in PyTorch, simplifying implementation of new acquisition functions. Our approach is backed by novel theoretical convergence results and made practical by a distinctive algorithmic foundation that leverages fast predictive distributions, hardware acceleration, and deterministic optimization. We also propose a novel "one-shot" formulation of the Knowledge Gradient, enabled by a combination of our theoretical and software contributions. In experiments, we demonstrate the improved sample efficiency of BoTorch relative to other popular libraries.

count=1
* A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/65fc9fb4897a89789352e211ca2d398f-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/65fc9fb4897a89789352e211ca2d398f-Paper.pdf)]
    * Title: A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Shih-Yang Su, Frank Yu, Michael Zollhoefer, Helge Rhodin
    * Abstract: While deep learning reshaped the classical motion capture pipeline with feed-forward networks, generative models are required to recover fine alignment via iterative refinement. Unfortunately, the existing models are usually hand-crafted or learned in controlled conditions, only applicable to limited domains. We propose a method to learn a generative neural body model from unlabelled monocular videos by extending Neural Radiance Fields (NeRFs). We equip them with a skeleton to apply to time-varying and articulated motion. A key insight is that implicit models require the inverse of the forward kinematics used in explicit surface models. Our reparameterization defines spatial latent variables relative to the pose of body parts and thereby overcomes ill-posed inverse operations with an overparameterization. This enables learning volumetric body shape and appearance from scratch while jointly refining the articulated pose; all without ground truth labels for appearance, pose, or 3D shape on the input videos. When used for novel-view-synthesis and motion capture, our neural model improves accuracy on diverse datasets.

count=1
* H-NeRF: Neural Radiance Fields for Rendering and Temporal Reconstruction of Humans in Motion
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/7d62a275027741d98073d42b8f735c68-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/7d62a275027741d98073d42b8f735c68-Paper.pdf)]
    * Title: H-NeRF: Neural Radiance Fields for Rendering and Temporal Reconstruction of Humans in Motion
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Hongyi Xu, Thiemo Alldieck, Cristian Sminchisescu
    * Abstract: We present neural radiance fields for rendering and temporal (4D) reconstruction of humans in motion (H-NeRF), as captured by a sparse set of cameras or even from a monocular video. Our approach combines ideas from neural scene representation, novel-view synthesis, and implicit statistical geometric human representations, coupled using novel loss functions. Instead of learning a radiance field with a uniform occupancy prior, we constrain it by a structured implicit human body model, represented using signed distance functions. This allows us to robustly fuse information from sparse views and generalize well beyond the poses or views observed in training. Moreover, we apply geometric constraints to co-learn the structure of the observed subject -- including both body and clothing -- and to regularize the radiance field to geometrically plausible solutions. Extensive experiments on multiple datasets demonstrate the robustness and the accuracy of our approach, its generalization capabilities significantly outside a small training set of poses and views, and statistical extrapolation beyond the observed shape.

count=1
* Differentiable Spline Approximations
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/a952ddeda0b7e2c20744e52e728e5594-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/a952ddeda0b7e2c20744e52e728e5594-Paper.pdf)]
    * Title: Differentiable Spline Approximations
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Minsu Cho, Aditya Balu, Ameya Joshi, Anjana Deva Prasad, Biswajit Khara, Soumik Sarkar, Baskar Ganapathysubramanian, Adarsh Krishnamurthy, Chinmay Hegde
    * Abstract: The paradigm of differentiable programming has significantly enhanced the scope of machine learning via the judicious use of gradient-based optimization. However, standard differentiable programming methods (such as autodiff) typically require that the machine learning models be differentiable, limiting their applicability. Our goal in this paper is to use a new, principled approach to extend gradient-based optimization to functions well modeled by splines, which encompass a large family of piecewise polynomial models. We derive the form of the (weak) Jacobian of such functions and show that it exhibits a block-sparse structure that can be computed implicitly and efficiently. Overall, we show that leveraging this redesigned Jacobian in the form of a differentiable "layer'' in predictive models leads to improved performance in diverse applications such as image segmentation, 3D point cloud reconstruction, and finite element analysis. We also open-source the code at \url{https://github.com/idealab-isu/DSA}.

count=1
* Generative Occupancy Fields for 3D Surface-Aware Image Synthesis
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/acab0116c354964a558e65bdd07ff047-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/acab0116c354964a558e65bdd07ff047-Paper.pdf)]
    * Title: Generative Occupancy Fields for 3D Surface-Aware Image Synthesis
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Xudong XU, Xingang Pan, Dahua Lin, Bo Dai
    * Abstract: The advent of generative radiance fields has significantly promoted the development of 3D-aware image synthesis. The cumulative rendering process in radiance fields makes training these generative models much easier since gradients are distributed over the entire volume, but leads to diffused object surfaces. In the meantime, compared to radiance fields occupancy representations could inherently ensure deterministic surfaces. However, if we directly apply occupancy representations to generative models, during training they will only receive sparse gradients located on object surfaces and eventually suffer from the convergence problem. In this paper, we propose Generative Occupancy Fields (GOF), a novel model based on generative radiance fields that can learn compact object surfaces without impeding its training convergence. The key insight of GOF is a dedicated transition from the cumulative rendering in radiance fields to rendering with only the surface points as the learned surface gets more and more accurate. In this way, GOF combines the merits of two representations in a unified framework. In practice, the training-time transition of start from radiance fields and march to occupancy representations is achieved in GOF by gradually shrinking the sampling region in its rendering process from the entire volume to a minimal neighboring region around the surface. Through comprehensive experiments on multiple datasets, we demonstrate that GOF can synthesize high-quality images with 3D consistency and simultaneously learn compact and smooth object surfaces. Our code is available at https://github.com/SheldonTsui/GOF_NeurIPS2021.

count=1
* DeepInteraction: 3D Object Detection via Modality Interaction
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/0d18ab3b5fabfa6fe47c62e711af02f0-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/0d18ab3b5fabfa6fe47c62e711af02f0-Paper-Conference.pdf)]
    * Title: DeepInteraction: 3D Object Detection via Modality Interaction
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Zeyu Yang, Jiaqi Chen, Zhenwei Miao, Wei Li, Xiatian Zhu, Li Zhang
    * Abstract: Existing top-performance 3D object detectors typically rely on the multi-modal fusion strategy. This design is however fundamentally restricted due to overlooking the modality-specific useful information and finally hampering the model performance. To address this limitation, in this work we introduce a novel modality interaction strategy where individual per-modality representations are learned and maintained throughout for enabling their unique characteristics to be exploited during object detection. To realize this proposed strategy, we design a DeepInteraction architecture characterized by a multi-modal representational interaction encoder and a multi-modal predictive interaction decoder. Experiments on the large-scale nuScenes dataset show that our proposed method surpasses all prior arts often by a large margin. Crucially, our method is ranked at the first position at the highly competitive nuScenes object detection leaderboard.

count=1
* VectorAdam for Rotation Equivariant Geometry Optimization
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/1a774f3555593986d7d95e4780d9e4f4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/1a774f3555593986d7d95e4780d9e4f4-Paper-Conference.pdf)]
    * Title: VectorAdam for Rotation Equivariant Geometry Optimization
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Selena Zihan Ling, Nicholas Sharp, Alec Jacobson
    * Abstract: The Adam optimization algorithm has proven remarkably effective for optimization problems across machine learning and even traditional tasks in geometry processing. At the same time, the development of equivariant methods, which preserve their output under the action of rotation or some other transformation, has proven to be important for geometry problems across these domains. In this work, we observe that Adam — when treated as a function that maps initial conditions to optimized results — is not rotation equivariant for vector-valued parameters due to per-coordinate moment updates. This leads to significant artifacts and biases in practice. We propose to resolve this deficiency with VectorAdam, a simple modification which makes Adam rotation-equivariant by accounting for the vector structure of optimization variables. We demonstrate this approach on problems in machine learning and traditional geometric optimization, showing that equivariant VectorAdam resolves the artifacts and biases of traditional Adam when applied to vector-valued data, with equivalent or even improved rates of convergence.

count=1
* 3DB: A Framework for Debugging Computer Vision Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/3848bc3112429079af85dedb7d369ef4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/3848bc3112429079af85dedb7d369ef4-Paper-Conference.pdf)]
    * Title: 3DB: A Framework for Debugging Computer Vision Models
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Guillaume Leclerc, Hadi Salman, Andrew Ilyas, Sai Vemprala, Logan Engstrom, Vibhav Vineet, Kai Xiao, Pengchuan Zhang, Shibani Santurkar, Greg Yang, Ashish Kapoor, Aleksander Madry
    * Abstract: We introduce 3DB: an extendable, unified framework for testing and debugging vision models using photorealistic simulation. We demonstrate, through a wide range of use cases, that 3DB allows users to discover vulnerabilities in computer vision systems and gain insights into how models make decisions. 3DB captures and generalizes many robustness analyses from prior work, and enables one to study their interplay. Finally, we find that the insights generated by the system transfer to the physical world. 3DB will be released as a library alongside a set of examples and documentation. We attach 3DB to the submission.

count=1
* SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/3a48b0eaba26ba862220a307a9edb0bb-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/3a48b0eaba26ba862220a307a9edb0bb-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Changan Chen, Carl Schissler, Sanchit Garg, Philip Kobernik, Alexander Clegg, Paul Calamia, Dhruv Batra, Philip Robinson, Kristen Grauman
    * Abstract: We introduce SoundSpaces 2.0, a platform for on-the-fly geometry-based audio rendering for 3D environments. Given a 3D mesh of a real-world environment, SoundSpaces can generate highly realistic acoustics for arbitrary sounds captured from arbitrary microphone locations. Together with existing 3D visual assets, it supports an array of audio-visual research tasks, such as audio-visual navigation, mapping, source localization and separation, and acoustic matching. Compared to existing resources, SoundSpaces 2.0 has the advantages of allowing continuous spatial sampling, generalization to novel environments, and configurable microphone and material properties. To our knowledge, this is the first geometry-based acoustic simulation that offers high fidelity and realism while also being fast enough to use for embodied learning. We showcase the simulator's properties and benchmark its performance against real-world audio measurements. In addition, we demonstrate two downstream tasks---embodied navigation and far-field automatic speech recognition---and highlight sim2real performance for the latter. SoundSpaces 2.0 is publicly available to facilitate wider research for perceptual systems that can both see and hear.

count=1
* Object Scene Representation Transformer
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/3dc83fcfa4d13e30070bd4b230c38cfe-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/3dc83fcfa4d13e30070bd4b230c38cfe-Paper-Conference.pdf)]
    * Title: Object Scene Representation Transformer
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Mehdi S. M. Sajjadi, Daniel Duckworth, Aravindh Mahendran, Sjoerd van Steenkiste, Filip Pavetic, Mario Lucic, Leonidas J. Guibas, Klaus Greff, Thomas Kipf
    * Abstract: A compositional understanding of the world in terms of objects and their geometry in 3D space is considered a cornerstone of human cognition. Facilitating the learning of such a representation in neural networks holds promise for substantially improving labeled data efficiency. As a key step in this direction, we make progress on the problem of learning 3D-consistent decompositions of complex scenes into individual objects in an unsupervised fashion. We introduce Object Scene Representation Transformer (OSRT), a 3D-centric model in which individual object representations naturally emerge through novel view synthesis. OSRT scales to significantly more complex scenes with larger diversity of objects and backgrounds than existing methods. At the same time, it is multiple orders of magnitude faster at compositional rendering thanks to its light field parametrization and the novel Slot Mixer decoder. We believe this work will not only accelerate future architecture exploration and scaling efforts, but it will also serve as a useful tool for both object-centric as well as neural scene representation learning communities.

count=1
* Learning Dense Object Descriptors from Multiple Views for Low-shot Category Generalization
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/517a0884c56008f8bf9d5912ca771d71-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/517a0884c56008f8bf9d5912ca771d71-Paper-Conference.pdf)]
    * Title: Learning Dense Object Descriptors from Multiple Views for Low-shot Category Generalization
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Stefan Stojanov, Anh Thai, Zixuan Huang, James M. Rehg
    * Abstract: A hallmark of the deep learning era for computer vision is the successful use of large-scale labeled datasets to train feature representations. This has been done for tasks ranging from object recognition and semantic segmentation to optical flow estimation and novel view synthesis of 3D scenes. In this work, we aim to learn dense discriminative object representations for low-shot category recognition without requiring any category labels. To this end, we propose Deep Object Patch Encodings (DOPE), which can be trained from multiple views of object instances without any category or semantic object part labels. To train DOPE, we assume access to sparse depths, foreground masks and known cameras, to obtain pixel-level correspondences between views of an object, and use this to formulate a self-supervised learning task to learn discriminative object patches. We find that DOPE can directly be used for low-shot classification of novel categories using local-part matching, and is competitive with and outperforms supervised and self-supervised learning baselines.

count=1
* Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/531230cfac80c65017ad0f85d3031edc-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/531230cfac80c65017ad0f85d3031edc-Paper-Conference.pdf)]
    * Title: Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Samuel Daulton, Xingchen Wan, David Eriksson, Maximilian Balandat, Michael A Osborne, Eytan Bakshy
    * Abstract: Optimizing expensive-to-evaluate black-box functions of discrete (and potentially continuous) design parameters is a ubiquitous problem in scientific and engineering applications. Bayesian optimization (BO) is a popular, sample-efficient method that leverages a probabilistic surrogate model and an acquisition function (AF) to select promising designs to evaluate. However, maximizing the AF over mixed or high-cardinality discrete search spaces is challenging standard gradient-based methods cannot be used directly or evaluating the AF at every point in the search space would be computationally prohibitive. To address this issue, we propose using probabilistic reparameterization (PR). Instead of directly optimizing the AF over the search space containing discrete parameters, we instead maximize the expectation of the AF over a probability distribution defined by continuous parameters. We prove that under suitable reparameterizations, the BO policy that maximizes the probabilistic objective is the same as that which maximizes the AF, and therefore, PR enjoys the same regret bounds as the original BO policy using the underlying AF. Moreover, our approach provably converges to a stationary point of the probabilistic objective under gradient ascent using scalable, unbiased estimators of both the probabilistic objective and its gradient. Therefore, as the number of starting points and gradient steps increase, our approach will recover of a maximizer of the AF (an often-neglected requisite for commonly used BO regret bounds). We validate our approach empirically and demonstrate state-of-the-art optimization performance on a wide range of real-world applications. PR is complementary to (and benefits) recent work and naturally generalizes to settings with multiple objectives and black-box constraints.

count=1
* Streaming Radiance Fields for 3D Video Synthesis
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/57c2cc952f388f6185db98f441351c96-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/57c2cc952f388f6185db98f441351c96-Paper-Conference.pdf)]
    * Title: Streaming Radiance Fields for 3D Video Synthesis
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Lingzhi LI, Zhen Shen, Zhongshu Wang, Li Shen, Ping Tan
    * Abstract: We present an explicit-grid based method for efficiently reconstructing streaming radiance fields for novel view synthesis of real world dynamic scenes. Instead of training a single model that combines all the frames, we formulate the dynamic modeling problem with an incremental learning paradigm in which per-frame model difference is trained to complement the adaption of a base model on the current frame. By exploiting the simple yet effective tuning strategy with narrow bands, the proposed method realizes a feasible framework for handling video sequences on-the-fly with high training efficiency. The storage overhead induced by using explicit grid representations can be significantly reduced through the use of model difference based compression. We also introduce an efficient strategy to further accelerate model optimization for each frame. Experiments on challenging video sequences demonstrate that our approach is capable of achieving a training speed of 15 seconds per-frame with competitive rendering quality, which attains $1000 \times$ speedup over the state-of-the-art implicit methods.

count=1
* Physically-Based Face Rendering for NIR-VIS Face Recognition
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/8f182e220092f7f1fc44f3313023f5a0-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/8f182e220092f7f1fc44f3313023f5a0-Paper-Conference.pdf)]
    * Title: Physically-Based Face Rendering for NIR-VIS Face Recognition
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Yunqi Miao, Alexandros Lattas, Jiankang Deng, Jungong Han, Stefanos Zafeiriou
    * Abstract: Near infrared (NIR) to Visible (VIS) face matching is challenging due to the significant domain gaps as well as a lack of sufficient data for cross-modality model training. To overcome this problem, we propose a novel method for paired NIR-VIS facial image generation. Specifically, we reconstruct 3D face shape and reflectance from a large 2D facial dataset and introduce a novel method of transforming the VIS reflectance to NIR reflectance. We then use a physically-based renderer to generate a vast, high-resolution and photorealistic dataset consisting of various poses and identities in the NIR and VIS spectra. Moreover, to facilitate the identity feature learning, we propose an IDentity-based Maximum Mean Discrepancy (ID-MMD) loss, which not only reduces the modality gap between NIR and VIS images at the domain level but encourages the network to focus on the identity features instead of facial details, such as poses and accessories. Extensive experiments conducted on four challenging NIR-VIS face recognition benchmarks demonstrate that the proposed method can achieve comparable performance with the state-of-the-art (SOTA) methods without requiring any existing NIR-VIS face recognition datasets. With slightly fine-tuning on the target NIR-VIS face recognition datasets, our method can significantly surpass the SOTA performance. Code and pretrained models are released under the insightface GitHub.

count=1
* TANGO: Text-driven Photorealistic and Robust 3D Stylization via Lighting Decomposition
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/c7b925e600ae4880f5c5d7557f70a72b-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/c7b925e600ae4880f5c5d7557f70a72b-Paper-Conference.pdf)]
    * Title: TANGO: Text-driven Photorealistic and Robust 3D Stylization via Lighting Decomposition
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Yongwei Chen, Rui Chen, Jiabao Lei, Yabin Zhang, Kui Jia
    * Abstract: Creation of 3D content by stylization is a promising yet challenging problem in computer vision and graphics research. In this work, we focus on stylizing photorealistic appearance renderings of a given surface mesh of arbitrary topology. Motivated by the recent surge of cross-modal supervision of the Contrastive Language-Image Pre-training (CLIP) model, we propose TANGO, which transfers the appearance style of a given 3D shape according to a text prompt in a photorealistic manner. Technically, we propose to disentangle the appearance style as the spatially varying bidirectional reflectance distribution function, the local geometric variation, and the lighting condition, which are jointly optimized, via supervision of the CLIP loss, by a spherical Gaussians based differentiable renderer. As such, TANGO enables photorealistic 3D style transfer by automatically predicting reflectance effects even for bare, low-quality meshes, without training on a task-specific dataset. Extensive experiments show that TANGO outperforms existing methods of text-driven 3D style transfer in terms of photorealistic quality, consistency of 3D geometry, and robustness when stylizing low-quality meshes. Our codes and results are available at our project webpage https://cyw-3d.github.io/tango/.

count=1
* Mip-Grid: Anti-aliased Grid Representations for Neural Radiance Fields
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/082d3d795520c43214da5123e56a3a34-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/082d3d795520c43214da5123e56a3a34-Paper-Conference.pdf)]
    * Title: Mip-Grid: Anti-aliased Grid Representations for Neural Radiance Fields
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Seungtae Nam, Daniel Rho, Jong Hwan Ko, Eunbyung Park
    * Abstract: Despite the remarkable achievements of neural radiance fields (NeRF) in representing 3D scenes and generating novel view images, the aliasing issue, rendering 'jaggies' or 'blurry' images at varying camera distances, remains unresolved in most existing approaches. The recently proposed mip-NeRF has effectively addressed this challenge by introducing integrated positional encodings (IPE). However, it relies on MLP architecture to represent the radiance fields, missing out on the fast training speed offered by the latest grid-based methods. In this work, we present mip-Grid, a novel approach that integrates anti-aliasing techniques into grid-based representations for radiance fields, mitigating the aliasing artifacts while enjoying fast training time. Notably, the proposed method uses a single-scale shared grid representation and a single-sampling approach, which only introduces minimal additions to the model parameters and computational costs. To handle scale ambiguity, mip-Grid generates multiple grids by applying simple convolution operations over the shared grid and uses the scale-aware coordinate to retrieve the appropriate features from the generated multiple grids. To test the effectiveness, we incorporated the proposed approach into the two recent representative grid-based methods, TensoRF and K-Planes. The experimental results demonstrated that mip-Grid greatly improved the rendering performance of both methods and showed comparable performance to mip-NeRF on multi-scale datasets while achieving significantly faster training time.

count=1
* Datasets and Benchmarks for Nanophotonic Structure and Parametric Design Simulations
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/0f12c9975ff4f2e44a5a26ef01b0b249-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/0f12c9975ff4f2e44a5a26ef01b0b249-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Datasets and Benchmarks for Nanophotonic Structure and Parametric Design Simulations
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Jungtaek Kim, Mingxuan Li, Oliver Hinder, Paul Leu
    * Abstract: Nanophotonic structures have versatile applications including solar cells, anti-reflective coatings, electromagnetic interference shielding, optical filters, and light emitting diodes. To design and understand these nanophotonic structures, electrodynamic simulations are essential. These simulations enable us to model electromagnetic fields over time and calculate optical properties. In this work, we introduce frameworks and benchmarks to evaluate nanophotonic structures in the context of parametric structure design problems. The benchmarks are instrumental in assessing the performance of optimization algorithms and identifying an optimal structure based on target optical properties. Moreover, we explore the impact of varying grid sizes in electrodynamic simulations, shedding light on how evaluation fidelity can be strategically leveraged in enhancing structure designs.

count=None
