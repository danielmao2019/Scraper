mapping_cnns = {
    "Deep Residual Learning for Image Recognition": "ResNet-v1",
    "Identity Mappings in Deep Residual Networks": "ResNet-v2",
    "Perturbative Neural Networks": "Perturbative Neural Networks (PNNs)",
    "FOSNet: An End-to-End Trainable Deep Neural Network for Scene Recognition": "FOS, SCL",
}

mapping_detection = {
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks": "Faster R-CNN, Region Proposal Network (RPN)",
    "End-to-End Object Detection with Transformers": "DEtection TRansformer (DETR)",
}

mapping_semantic_segmentation = {
    # DeepLab Family
    "Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs": "DeepLab-v1",
    "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs": "DeepLab-v2",
    "Rethinking Atrous Convolution for Semantic Image Segmentation": "DeepLab-v3",
    "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation": "DeepLab-v3+",
    # Transformer-Based
    "FlowFormer: A Transformer Architecture for Optical Flow": "FlowFormer",
    "MPViT: Multi-Path Vision Transformer for Dense Prediction": "MPViT",
    "PCT: Point cloud transformer": "PCT",
    "Semantic Image Segmentation via Deep Parsing Network": "Deep Parsing Network (DPN)",
    "ACNet: Attention Based Network to Exploit Complementary Features for RGBD Semantic Segmentation": "Attention Complementary Network (ACNet)",
    "Multi Receptive Field Network for Semantic Segmentation": "Multi Receptive Field Network (MRFN)",
    "Spatial Information Guided Convolution for Real-Time RGBD Semantic Segmentation": "Spatial Information Guided Convolution (S-Conv)",
    "Depth-aware CNN for RGB-D Segmentation": "Depth-aware CNN",
    # Diffusion-Based Approaches
    "DifFSS: Diffusion Model for Few-Shot Semantic Segmentation": "DifFSS",
    "DFormer: Diffusion-guided Transformer for Universal Image Segmentation": "DFormer",
    "SegDiff: Image Segmentation with Diffusion Probabilistic Models": "SegDiff",
}

mapping_instance_segmentation = {
    # Two-stage
    "Learning to Segment Object Candidates": "DeepMask",
    "Learning to Refine Object Segments": "SharpMask",
    # Single-stage
    "Explicit Shape Encoding for Real-Time Instance Segmentation": "Explicit Shape Encoding (ESE-Seg)",
}

mapping_transformer = {
    "AutoFocusFormer: Image Segmentation off the Grid": "AutoFocusFormer (AFF)",
}

mapping_mtl = {
    # gradient manipulation
    "Multi-Task Learning as Multi-Objective Optimization": "Multiple Gradient Descent Algorithm (MGDA)",
    "Gradient Surgery for Multi-Task Learning": "Projecting Conflicting Gradients (PCGrad)",
    "Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models": "Gradient Vaccine (GradVac)",
    "Towards Impartial Multi-task Learning": "Impartial Multi-Task Learning (IMTL)",
    "Conflict-Averse Gradient Descent for Multi-task learning": "Conflict-Averse Gradient Descent (CAGrad)",
    "Measuring and Harnessing Transference in Multi-Task Learning": "Transference, Increased Transfer MTL (IT-MTL)",
    # gradient balancing
    "Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning": "Random Weighting (RW)",
    "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks": "GradNorm",
    "FAMO: Fast Adaptive Multitask Optimization": "Fast Adaptive Multitask Optimization (FAMO)",
    "Self-Paced Multi-Task Learning": "Self-Paced Multi-Task Learning (SPMTL)",
    # architectural solutions
    ## architecture design
    "HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition": "HyperFace",
    "MT-ORL: Multi-Task Occlusion Relationship Learning": "Occlusion-shared and Path-separated Network (OPNet)",
    ## feature fusion
    "Cross-stitch Networks for Multi-task Learning": "Cross-stitch Networks",
    "Latent Multi-task Architecture Learning": "Sluice Networks",
    "NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction": "Neural Discriminative Dimensionality Reduction (NDDR)",
    "CrossInfoNet: Multi-Task Information Sharing Based Hand Pose Estimation": "CrossInfoNet",
    ## self-attention
    ## cross-attention
    "Exploring Relational Context for Multi-Task Dense Prediction": "Adaptive Task-Relational Context (ATRC)",
    "Cross-Task Attention Mechanism for Dense Multi-Task Learning": "Cross-Task Attention Mechanism (xTAM), Multi-Task Exchange Block (mTEB)",
    ## others
    "Deep Cross Residual Learning for Multitask Visual Recognition": "Cross-Residual Learning (CRL)",
    "Integrated perception with recurrent multi-task neural networks": "MultiNet",
    "Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning": "Recon",
    ## task information distillation
    "PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing": "Prediction-and-Distillation (PAD) Networks",
    "Joint Task-Recursive Learning for Semantic Segmentation and Depth Estimation": "Task-Recursive Learning (TRL)",
    "Pattern-Affinitive Propagation across Depth, Surface Normal and Semantic Segmentation": "Pattern-Affinitive Propagation (PAP)",
    "MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning": "Multi-Scale Task Interaction (MTI) Networks",
    ## knowledge distillation
    "TaskExpert: Dynamically Assembling Multi-Task Representations with Memorial Mixture-of-Experts": "Memorial Mixture-of-Experts (MMoE)",
    ## others
    "Stochastic Filter Groups for Multi-Task CNNs: Learning Specialist and Generalist Convolution Kernels": "Stochastic Filter Groups (SFG)",
    "Mitigating Task Interference in Multi-Task Learning via Explicit Task Routing with Non-Learnable Primitives": "ETR-NLP",
    "Many Task Learning with Task Routing": "Many Task Learning (MaTL), Task Routing Layer (TRL)",
    "Mitigating Task Interference in Multi-Task Learning via Explicit Task Routing with Non-Learnable Primitives": "Explicit Task Routing (ETR), Non-Learnable Primitives (NLPs)",
    # architecture search
    "GNAS: A Greedy Neural Architecture Search Method for Multi-Attribute Learning": "Greedy Neural Architecture Search (GNAS)",
    # others
    "Facial Landmark Detection by Deep Multi-task Learning": "Tasks-Constrained Learning",
    "Branched Multi-Task Networks: Deciding What Layers To Share": "Branched Multi-Task Networks",
    "Routing Networks: Adaptive Selection of Non-linear Functions for Multi-Task Learning": "Routing Networks",
    "PathNet: Evolution Channels Gradient Descent in Super Neural Networks": "PathNet",
    "Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout": "Gradient Sign Dropout (GradDrop)",
    "Gradient Surgery for Multi-Task Learning": "PCGrad",
    "Conflict-Averse Gradient Descent for Multi-task Learning": "CAGrad",
    "UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory": "UberNet",
    "Multi-Task Learning as a Bargaining Game": "Nash-MTL",
    "Distral: Robust Multitask Reinforcement Learning": "Distral",
    "Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning": "Actor-Mimic",
    "End-to-End Multi-Task Learning with Attention": "Multi-Task Attention Network (MTAN)",
    "Independent Component Alignment for Multi-Task Learning": "Aligned-MTL",
    "Learning Multiple Tasks with Multilinear Relationship Networks": "Multilinear Relationship Networks (MRN)",
    "Decoupled Multi-task Learning with Cyclical Self-Regulation for Face Parsing": "DML-CSR",
    # continual learning
    "Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference": "Meta-Experience Replay (MER)",
    "Batch Model Consolidation: A Multi-Task Model Consolidation Framework": "Batch Model Consolidation (BMC)",
    # federated learning
    "Towards Hetero-Client Federated Multi-Task Learning": "Hetero-Client Federated Multi-Task Learning (HC-FMTL), Hyper Conflict-Averse, Hyper Cross Attention (HCA$^2$)",
}

mapping_gan_2D = {
    # GAN Inversion
    "Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?": "Image2StyleGAN",
    "Image2StyleGAN++: How to Edit the Embedded Images?": "Image2StyleGAN++",
    "GAN Inversion for Out-of-Range Images with Geometric Transformations": "BDInvert",
    "From Continuity to Editability: Inverting GANs with Consecutive Images": "From Continuity to Editability",
    "Barbershop: GAN-based Image Compositing using Segmentation Masks": "Barbershop",
    "ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement": "ReStyle",
    "Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation": "pSp",
    "Designing an Encoder for StyleGAN Image Manipulation": "E4E",
    "E2Style: Improve the Efficiency and Effectiveness of StyleGAN Inversion": "E2Style",
    "High-Fidelity GAN Inversion for Image Attribute Editing": "Adaptive Distortion Alignment",
    "Feature-Style Encoder for Style-Based GAN Inversion": "Feature-Style Encoder",
    # others
    "A Style-Based Generator Architecture for Generative Adversarial Networks": "StyleGAN-v1",
    "Analyzing and Improving the Image Quality of StyleGAN": "StyleGAN-v2",
    "Alias-Free Generative Adversarial Networks": "StyleGAN-v3",
    "QC-StyleGAN -- Quality Controllable Image Generation and Manipulation": "QC-StyleGAN",
    "Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer": "DualStyleGAN",
    "Training Generative Adversarial Networks with Limited Data": "Adaptive Discriminator Augmentation",
    "Alias-Free Generative Adversarial Networks": "Alias-Free GAN",
    "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets": "InfoGAN",
    "FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery": "FineGAN",
    "AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks": "AttnGAN",
    "StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks": "StackGAN",
    "Large Scale GAN Training for High Fidelity Natural Image Synthesis": "BigGAN",
    "Demystifying MMD GANs": "Maximum Mean Discrepancy (MMD), Kernel Inception Distance (KID)",
    "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium": "Two Time-Scale Update Rule (TTUR), Frechet Inception Distance (FID)",
    "Classification Accuracy Score for Conditional Generative Models": "Classification Accuracy Score (CAS)",
}

mapping_gan_3D = {
    "3D-aware Image Synthesis via Learning Structural and Textural Representations": "VolumeGAN",
}

mapping_diffusion_2D = {
    # Denoising Diffusion Probabilistic Models (DDPMs)
    "Denoising Diffusion Probabilistic Models": "DDPM",
    "Denoising Diffusion Implicit Models": "Denoising Diffusion Implicit Models (DDIM)",
    "gDDIM: Generalized denoising diffusion implicit models": "gDDIM",
    "ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models": "ILVR",
    "Improved Denoising Diffusion Probabilistic Models": "Improved DDPM",
    "Pyramidal Denoising Diffusion Probabilistic Models": "Pyramidal DDPM",
    "AT-DDPM: Restoring Faces degraded by Atmospheric Turbulence using Denoising Diffusion Probabilistic Models": "AT-DDPM",
    "T2V-DDPM: Thermal to Visible Face Translation using Denoising Diffusion Probabilistic Models": "T2V-DDPM",
    "High-Resolution Image Synthesis with Latent Diffusion Models": "Stable Diffusion",
    "Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality": "Differentiable Diffusion Sampler Search",
    # conditional generation
    "Diffusion Models Beat GANs on Image Synthesis": "Classifier Guidance",
    "Classifier-Free Diffusion Guidance": "Classifier-Free Guidance",
    "BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models": "Brownian Bridge Diffusion Model (BBDM)",
    # discrete latent space
    "Neural Discrete Representation Learning": "Vector Quantised-Variational AutoEncoder (VQ-VAE)",
    # others
    "Hierarchical Text-Conditional Image Generation with CLIP Latents": "DALLE2",
    "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding": "Imagen",
    "Adding Conditional Control to Text-to-Image Diffusion Models": "ControlNet",
    "UniTune: Text-Driven Image Editing by Fine Tuning an Image Generation Model on a Single Image": "UniTune",
    "DiffEdit: Diffusion-based semantic image editing with mask guidance": "DiffEdit",
    "RePaint: Inpainting using Denoising Diffusion Probabilistic Models": "RePaint",
    "DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation": "DiffusionCLIP",
    "Imagic: Text-Based Real Image Editing with Diffusion Models": "Imagic",
    "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations": "SDEdit",
    "Multimodal Explanations: Justifying Decisions and Pointing to the Evidence": "Multimodal Explanations",
    "DeepVoting: A Robust and Explainable Deep Network for Semantic Part Detection under Partial Occlusion": "DeepVoting",
    "Diffusion Models already have a Semantic Latent Space": "ASYRP",
    "UnitBox: An Advanced Object Detection Network": "UnitBox",
    "Attentional Bottleneck: Towards an Interpretable Deep Driving Network": "Attentional Bottleneck",
    "VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection": "VoxelNet",
    "PointPillars: Fast Encoders for Object Detection from Point Clouds": "PointPillars",
    "Deep Variational Information Bottleneck": "Deep VIB",
    "InfoBot: Transfer and Exploration via the Information Bottleneck": "InfoBot",
    "Sharp U-Net: Depthwise Convolutional Network for Biomedical Image Segmentation": "Sharp U-Net",
    "SMU-Net: Style matching U-Net for brain tumor segmentation with missing modalities": "SMU-Net",
    "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion": "An Image is Worth One Word",
    "DiffusionRig: Learning Personalized Priors for Facial Appearance Editing": "DiffusionRig",
    "DCFace: Synthetic Face Generation with Dual Condition Diffusion Model": "DCFace",
    "LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation": "LayoutDiffusion",
    "DiffusioNeRF: Regularizing Neural Radiance Fields with Denoising Diffusion Models": "DiffusioNeRF",
    "NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors": "NeRDi",
    "RenderDiffusion: Image Diffusion for 3D Reconstruction, Inpainting and Generation": "RenderDiffusion",
    "DiffRF: Rendering-Guided 3D Radiance Field Diffusion": "DiffRF",
    "NeuralField-LDM: Scene Generation with Hierarchical Latent Diffusion Models": "NeuralField-LDM",
    "Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion": "Trace and Pace",
    "Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding": "Diffusion Video Autoencoders",
    "MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation": "MM-Diffusion",
    "VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation": "VideoFusion",
    "Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction": "Come-Closer-Diffuse-Faster (CCDF)",
    "Diffusion Autoencoders: Toward a Meaningful and Decodable Representation": "Diffusion Autoencoders",
    "Score-based Generative Modeling in Latent Space": "Latent Score-based Generative Model (LSGM)",
    "Implicit Diffusion Models for Continuous Super-Resolution": "Implicit Diffusion Model (IDM)",
    "All are Worth Words: A ViT Backbone for Diffusion Models": "U-ViT",
    "Scalable Diffusion Models with Transformers": "Diffusion Transformers (DiTs)",
    "Class-Balancing Diffusion Models": "Class-Balancing Diffusion Models (CBDM)",
}

mapping_diffusion_3D = {
    "Unconstrained Scene Generation with Locally Conditioned Radiance Fields": "Generative Scene Networks (GSN)",
    "3D Shape Generation and Completion through Point-Voxel Diffusion": "Point-Voxel Diffusion (PVD)",
}

mapping_nerf = {
    # NeRF
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis": "NeRF",
    "Stable View Synthesis": "Stable View Synthesis (SVS)",
    ## Dynamic Scene
    "Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Dynamic Scene From Monocular Video": "NR-NeRF",
    "NeX: Real-time View Synthesis with Neural Basis Expansion": "NeX",
    "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections": "NeRF in the Wild",
    "GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis": "GRAF",
    "Neural Volumes: Learning Dynamic Renderable Volumes from Images": "Neural Volumes",
    "AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation": "AtlasNet",
    "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation": "DeepSDF",
    "NeRF--: Neural Radiance Fields Without Known Camera Parameters": "NeRF--",
    "TensoRF: Tensorial Radiance Fields": "TensoRF",
    "Neural Sparse Voxel Fields": "NSVF",
    # generalizability
    "Is Attention All That NeRF Needs?": "Generalizable NeRF Transformer (GNT)",
    # bundle adjustment & camera pose prior
    "SiNeRF: Sinusoidal Neural Radiance Fields for Joint Pose Estimation and Scene Reconstruction": "Sinusoidal Neural Radiance Fields (SiNeRF)",
    # NeRF - Fundamentals
    "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields": "Mip-NeRF",
    "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields": "Mip-NeRF 360",
    # NeRF - Performance
    "Point-NeRF: Point-based Neural Radiance Fields": "Point-NeRF",
    "Plenoxels: Radiance Fields without Neural Networks": "Plenoxels",
    "PlenOctrees for Real-time Rendering of Neural Radiance Fields": "PlenOctrees",
    "Fourier PlenOctrees for Dynamic Radiance Field Rendering in Real-time": "Fourier PlenOctrees",
    "EfficientNeRF: Efficient Neural Radiance Fields": "EfficientNeRF",
    "Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction": "DVGO",
    "FastNeRF: High-Fidelity Neural Rendering at 200FPS": "FastNeRF",
    # NeRF - Others
    "MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures": "MobileNeRF",
    "NeRF++: Analyzing and Improving Neural Radiance Fields": "NeRF++",
    "GeoNeRF: Generalizing NeRF with Geometry Priors": "GeoNeRF",
    "NAN: Noise-Aware NeRFs for Burst-Denoising": "NAN",
    "NeRFusion: Fusing Radiance Fields for Large-Scale Scene Reconstruction": "NeRFusion",
    "Depth-supervised NeRF: Fewer Views and Faster Training for Free": "Depth-supervised NeRF",
    "InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering": "InfoNeRF",
    "Ray Priors through Reprojection: Improving Neural Radiance Fields for Novel View Extrapolation": "RapNeRF",
    "RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs": "RegNeRF",
    "Aug-NeRF: Training Stronger Neural Radiance Fields with Triple-Level Physically-Grounded Augmentations": "Aug-NeRF",
    "Deblur-NeRF: Neural Radiance Fields from Blurry Images": "Deblur-NeRF",
    "DIVeR: Real-time and Accurate Neural Radiance Fields with Deterministic Integration for Volume Rendering": "DIVeR",
    "Hallucinated Neural Radiance Fields in the Wild": "Ha-NeRF",
    "HDR-NeRF: High Dynamic Range Neural Radiance Fields": "HDR-NeRF",
    "NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images": "NeRF in the Dark",
    "NeRFReN: Neural Radiance Fields with Reflections": "NeRFReN",
    "Neural Rays for Occlusion-aware Image-based Rendering": "Neural Ray (NeuRay)",
    "Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields": "Ref-NeRF",
    "Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations": "SRT",
    "Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs": "Mega-NeRF",
    "Block-NeRF: Scalable Large Scene Neural View Synthesis": "Block-NeRF",
    "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding": "Instant NGP",
}

mapping_representation_3d = {
    "Approximate Differentiable Rendering with Algebraic Surfaces": "Fuzzy Metaballs",
}

mapping_3D = {
    "Differentiable Blocks World: Qualitative 3D Decomposition by Rendering Primitives": "Differentiable Blocks World (DBW)",
}

mapping_geometric = {
    "Learning Deformable Tetrahedral Meshes for 3D Reconstruction": "Deformable Tetrahedral Meshes (DefTet)",
    "Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis": "Deep Marching Tetrahedra (DMTet)",
}

mapping_representation_image = {
    "Masked Autoencoders Are Scalable Vision Learners": "Masked Autoencoders (MAE)",
}

mapping_adversarial = {
    "Explaining and Harnessing Adversarial Examples": "Fast Gradient Sign Method (FGSM)",
}

mapping = {
    # Convolutional Neural Networks
    "InceptionNeXt: When Inception Meets ConvNeXt": "InceptionNeXt",
    "HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions": "HorNet",
    "More ConvNets in the 2020s: Scaling up Kernels Beyond 51x51 using Sparsity": "SLKNet",
    "Focal Modulation Networks": "FocalNets",
    # Representation Learning
    "Representation Learning with Contrastive Predictive Coding": "Contrastive Predictive Coding (CPC)",
    # Transformers
    "Slide-Transformer: Hierarchical Vision Transformer with Local Self-Attention": "Slide-Transformer",
    # Image Retrieval
    "MultiGrain: a unified image embedding for classes and instances": "MultiGrain",
    "A Self-Supervised Descriptor for Image Copy Detection": "SSCD",
    "Particular object retrieval with integral max-pooling of CNN activations": "R-MAC",
    "Fine-tuning CNN Image Retrieval with No Human Annotation": "Generalized-Mean (GeM) pooling",
    "Deep Image Retrieval: Learning global representations for image search": "Triplet Ranking Loss",
    # Image Captioning
    "Injecting Semantic Concepts into End-to-End Image Captioning": "ViTCAP",
    "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention": "Show, Attend and Tell",
    "SmallCap: Lightweight Image Captioning Prompted with Retrieval Augmentation": "SmallCap",
    "Recurrent Image Captioner: Describing Images with Spatial-Invariant Transformation and Attention Filtering": "Recurrent Image Captioner",
    "DenseCap: Fully Convolutional Localization Networks for Dense Captioning": "DenseCap",
    # Image Quality Assessment
    "DeepSim: Deep similarity for image quality assessment": "DeepSim",
    # Transformers
    "Emerging Properties in Self-Supervised Vision Transformers": "DINOv1",
    "DINOv2: Learning Robust Visual Features without Supervision": "DINOv2",
    "Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation": "Mask DINO",
    # Backpropagation
    "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps": "Backpropagation",
    "Striving for Simplicity: The All Convolutional Net": "Guided Backpropagation",
    # Class Activation Maps (CAMs)
    "Learning Deep Features for Discriminative Localization": "CAM",
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization": "Grad-CAM",
    "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks": "Grad-CAM++",
    "Eigen-CAM: Class Activation Map using Principal Components": "Eigen-CAM",
    "Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks": "HiResCAM",
    "Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs": "Axiom-based Grad-CAM",
    "SmoothGrad: removing noise by adding noise": "SmoothGrad",
    "Attention Branch Network: Learning of Attention Mechanism for Visual Explanation": "Attention Branch Network",
    "VisualBackProp: efficient visualization of CNNs": "VisualBackProp",
    # Neural Networks Optimization
    "ADADELTA: An Adaptive Learning Rate Method": "ADADELTA",
    "Adam: A Method for Stochastic Optimization": "Adam",
}

mapping.update(mapping_cnns)
mapping.update(mapping_detection)
mapping.update(mapping_semantic_segmentation)
mapping.update(mapping_instance_segmentation)
mapping.update(mapping_transformer)
mapping.update(mapping_mtl)
mapping.update(mapping_gan_2D)
mapping.update(mapping_gan_3D)
mapping.update(mapping_diffusion_2D)
mapping.update(mapping_diffusion_3D)
mapping.update(mapping_nerf)
mapping.update(mapping_geometric)
mapping.update(mapping_3D)
mapping.update(mapping_representation_image)
mapping.update(mapping_representation_3d)
mapping.update(mapping_adversarial)
