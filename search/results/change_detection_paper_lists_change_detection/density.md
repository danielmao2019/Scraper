count=109
* Adapting to Continuous Covariate Shift via Online Density Ratio Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/5cad96c4433955a2e76749ee74a424f5-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/5cad96c4433955a2e76749ee74a424f5-Paper-Conference.pdf)]
    * Title: Adapting to Continuous Covariate Shift via Online Density Ratio Estimation
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Yu-Jie Zhang, Zhen-Yu Zhang, Peng Zhao, Masashi Sugiyama
    * Abstract: Dealing with distribution shifts is one of the central challenges for modern machine learning. One fundamental situation is the covariate shift, where the input distributions of data change from the training to testing stages while the input-conditional output distribution remains unchanged. In this paper, we initiate the study of a more challenging scenario --- continuous covariate shift --- in which the test data appear sequentially, and their distributions can shift continuously. Our goal is to adaptively train the predictor such that its prediction risk accumulated over time can be minimized. Starting with the importance-weighted learning, we theoretically show the method works effectively if the time-varying density ratios of test and train inputs can be accurately estimated. However, existing density ratio estimation methods would fail due to data scarcity at each time step. To this end, we propose an online density ratio estimation method that can appropriately reuse historical information. Our method is proven to perform well by enjoying a dynamic regret bound, which finally leads to an excess risk guarantee for the predictor. Empirical results also validate the effectiveness.

count=70
* Trimmed Density Ratio Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/ea204361fe7f024b130143eb3e189a18-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2017/file/ea204361fe7f024b130143eb3e189a18-Paper.pdf)]
    * Title: Trimmed Density Ratio Estimation
    * Publisher: NeurIPS
    * Publication Date: `2017`
    * Authors: Song Liu, Akiko Takeda, Taiji Suzuki, Kenji Fukumizu
    * Abstract: Density ratio estimation is a vital tool in both machine learning and statistical community. However, due to the unbounded nature of density ratio, the estimation proceudre can be vulnerable to corrupted data points, which often pushes the estimated ratio toward infinity. In this paper, we present a robust estimator which automatically identifies and trims outliers. The proposed estimator has a convex formulation, and the global optimum can be obtained via subgradient descent. We analyze the parameter estimation error of this estimator under high-dimensional settings. Experiments are conducted to verify the effectiveness of the estimator.

count=56
* Density-Difference Estimation
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/f2fc990265c712c49d51a18a32b39f0c-Paper.pdf)]
    * Title: Density-Difference Estimation
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Masashi Sugiyama, Takafumi Kanamori, Taiji Suzuki, Marthinus Plessis, Song Liu, Ichiro Takeuchi
    * Abstract: We address the problem of estimating the difference between two probability densities. A naive approach is a two-step procedure of first estimating two densities separately and then computing their difference. However, such a two-step procedure does not necessarily work well because the first step is performed without regard to the second step and thus a small estimation error incurred in the first stage can cause a big error in the second stage. In this paper, we propose a single-shot procedure for directly estimating the density difference without separately estimating two densities. We derive a non-parametric finite-sample error bound for the proposed single-shot density-difference estimator and show that it achieves the optimal convergence rate. We then show how the proposed density-difference estimator can be utilized in L2-distance approximation. Finally, we experimentally demonstrate the usefulness of the proposed method in robust distribution comparison such as class-prior estimation and change-point detection.

count=33
* Learning High-Density Regions for a Generalized Kolmogorov-Smirnov Test in High-Dimensional Data
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/6855456e2fe46a9d49d3d3af4f57443d-Paper.pdf)]
    * Title: Learning High-Density Regions for a Generalized Kolmogorov-Smirnov Test in High-Dimensional Data
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Assaf Glazer, Michael Lindenbaum, Shaul Markovitch
    * Abstract: We propose an efficient, generalized, nonparametric, statistical Kolmogorov-Smirnov test for detecting distributional change in high-dimensional data. To implement the test, we introduce a novel, hierarchical, minimum-volume sets estimator to represent the distributions to be tested. Our work is motivated by the need to detect changes in data streams, and the test is especially efficient in this context. We provide the theoretical foundations of our test and show its superiority over existing methods.

count=29
* Crowd Activity Change Point Detection in Videos via Graph Stream Mining
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w6/html/Yang_Crowd_Activity_Change_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Yang_Crowd_Activity_Change_CVPR_2018_paper.pdf)]
    * Title: Crowd Activity Change Point Detection in Videos via Graph Stream Mining
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Meng Yang, Lida Rashidi, Sutharshan Rajasegarar, Christopher Leckie, Aravinda S. Rao, Marimuthu Palaniswami
    * Abstract: In recent years, there has been a growing interest in detecting anomalous behavioral patterns in video. In this work, we address this task by proposing a novel activity change point detection method to identify crowd movement anomalies for video surveillance. In our proposed novel framework, a hyperspherical clustering algorithm is utilized for the automatic identification of interesting regions, then the density of pedestrian flows between every pair of interesting regions over consecutive time intervals is monitored and represented as a sequence of adjacency matrices where the direction and density of flows are captured through a directed graph. Finally, we use graph edit distance as well as a cumulative sum test to detect change points in the graph sequence. We conduct experiments on four real-world video datasets: Dublin, New Orleans, Abbey Road and MCG Datasets. We observe that our proposed approach achieves a high F-measure, i.e., in the range [0.7, 1], for these datasets. The evaluation reveals that our proposed method can successfully detect the change points in all datasets at both global and local levels. Our results also demonstrate the efficiency and effectiveness of our proposed algorithm for change point detection and segmentation tasks.

count=29
* On the Exploration of Local Significant Differences For Two-Sample Test
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/10fc83943b4540a9524af6fc67a23fef-Paper-Conference.pdf)]
    * Title: On the Exploration of Local Significant Differences For Two-Sample Test
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Zhijian Zhou, Jie Ni, Jia-He Yao, Wei Gao
    * Abstract: Recent years have witnessed increasing attentions on two-sample test with diverse real applications, while this work takes one more step on the exploration of local significant differences for two-sample test. We propose the ME$_\text{MaBiD}$, an effective test for two-sample testing, and the basic idea is to exploit local information by multiple Mahalanobis kernels and introduce bi-directional hypothesis for testing. On the exploration of local significant differences, we first partition the embedding space into several rectangle regions via a new splitting criterion, which is relevant to test power and data correlation. We then explore local significant differences based on our bi-directional masked $p$-value together with the ME$_\text{MaBiD}$ test. Theoretically, we present the asymptotic distribution and lower bounds of test power for our ME$_\text{MaBiD}$ test, and control the familywise error rate on the exploration of local significant differences. We finally conduct extensive experiments to validate the effectiveness of our proposed methods on two-sample test and the exploration of local significant differences.

count=16
* Change point detection and inference in multivariate non-parametric models under mixing conditions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/42a0de6b8a1809ceba8fdad1661be06c-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/42a0de6b8a1809ceba8fdad1661be06c-Paper-Conference.pdf)]
    * Title: Change point detection and inference in multivariate non-parametric models under mixing conditions
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, OSCAR HERNAN MADRID PADILLA, Yi Yu
    * Abstract: This paper addresses the problem of localizing and inferring multiple change points, in non-parametric multivariate time series settings. Specifically, we consider a multivariate time series with potentially short-range dependence, whose underlying distributions have Hölder smooth densities and can change over time in a piecewise-constant manner. The change points, which correspond to the times when the distribution changes, are unknown. We present the limiting distributions of the change point estimators under the scenarios where the minimal jump size vanishes or remains constant. Such results have not been revealed in the literature in non-parametric change point settings. As byproducts, we develop a sharp estimator that can accurately localize the change points in multivariate non-parametric time series, and a consistent block-type long-run variance estimator. Numerical studies are provided to complement our theoretical findings.

count=15
* Tracking Most Significant Shifts in Nonparametric Contextual Bandits
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/13b501c58ae3bfe9635a259f4414e943-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/13b501c58ae3bfe9635a259f4414e943-Paper-Conference.pdf)]
    * Title: Tracking Most Significant Shifts in Nonparametric Contextual Bandits
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Joe Suk, Samory Kpotufe
    * Abstract: We study nonparametric contextual bandits where Lipschitz mean reward functions may change over time.We first establish the minimax dynamic regret rate in this less understood setting in terms of number of changes $L$ and total-variation $V$, both capturing all changes in distribution over context space, and argue that state-of-the-art procedures are suboptimal in this setting.Next, we tend to the question of an _adaptivity_ for this setting, i.e. achieving the minimax rate without knowledge of $L$ or $V$. Quite importantly, we posit that the bandit problem, viewed locally at a given context $X_t$, should not be affected by reward changes in other parts of context space $\cal X$. We therefore propose a notion of _change_, which we term _experienced significant shifts_, that better accounts for locality, and thus counts considerably less changes than $L$ and $V$. Furthermore, similar to recent work on non-stationary MAB (Suk & Kpotufe, 2022), _experienced significant shifts_ only count the most _significant_ changes in mean rewards, e.g., severe best-arm changes relevant to observed contexts.Our main result is to show that this more tolerant notion of change can in fact be adapted to.

count=15
* Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/7abbcb05a5d55157ede410bb718e32d7-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/7abbcb05a5d55157ede410bb718e32d7-Paper-Conference.pdf)]
    * Title: Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Models
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Yule Wang, Zijing Wu, Chengrui Li, Anqi Wu
    * Abstract: In the field of behavior-related brain computation, it is necessary to align raw neural signals against the drastic domain shift among them. A foundational framework within neuroscience research posits that trial-based neural population activities rely on low-dimensional latent dynamics, thus focusing on the latter greatly facilitates the alignment procedure. Despite this field's progress, existing methods ignore the intrinsic spatio-temporal structure during the alignment phase. Hence, their solutions usually lead to poor quality in latent dynamics structures and overall performance. To tackle this problem, we propose an alignment method ERDiff, which leverages the expressivity of the diffusion model to preserve the spatio-temporal structure of latent dynamics. Specifically, the latent dynamics structures of the source domain are first extracted by a diffusion model. Then, under the guidance of this diffusion model, such structures are well-recovered through a maximum likelihood alignment procedure in the target domain. We first demonstrate the effectiveness of our proposed method on a synthetic dataset. Then, when applied to neural recordings from the non-human primate motor cortex, under both cross-day and inter-subject settings, our method consistently manifests its capability of preserving the spatio-temporal structure of latent dynamics and outperforms existing approaches in alignment goodness-of-fit and neural decoding performance.

count=14
* Controlled Recognition Bounds for Visual Learning and Exploration
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2012/hash/2a50e9c2d6b89b95bcb416d6857f8b45-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2012/file/2a50e9c2d6b89b95bcb416d6857f8b45-Paper.pdf)]
    * Title: Controlled Recognition Bounds for Visual Learning and Exploration
    * Publisher: NeurIPS
    * Publication Date: `2012`
    * Authors: Vasiliy Karasev, Alessandro Chiuso, Stefano Soatto
    * Abstract: We describe the tradeoff between the performance in a visual recognition problem and the control authority that the agent can exercise on the sensing process. We focus on the problem of “visual search” of an object in an otherwise known and static scene, propose a measure of control authority, and relate it to the expected risk and its proxy (conditional entropy of the posterior density). We show this analytically, as well as empirically by simulation using the simplest known model that captures the phenomenology of image formation, including scaling and occlusions. We show that a “passive” agent given a training set can provide no guarantees on performance beyond what is afforded by the priors, and that an “omnipotent” agent, capable of infinite control authority, can achieve arbitrarily good performance (asymptotically).

count=13
* Space-Time Localization and Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Lee_Space-Time_Localization_and_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Space-Time_Localization_and_ICCV_2017_paper.pdf)]
    * Title: Space-Time Localization and Mapping
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Minhaeng Lee, Charless C. Fowlkes
    * Abstract: This paper addresses the problem of building a spatio-temporal model of the world from a stream of time-stamped data. Unlike traditional models for simultaneous localization and mapping (SLAM) and structure-from-motion (SfM) which focus on recovering a single rigid 3D model, we tackle the problem of mapping scenes in which dynamic components appear, move and disappear independently of each other over time. We introduce a simple generative probabilistic model of 4D structure which specifies location, spatial and temporal extent of rigid surface patches by local Gaussian mixtures. We fit this model to a time-stamped stream of input data using expectation-maximization to estimate the model structure parameters (mapping) and the alignment of the input data to the model (localization). By explicitly representing the temporal extent and observability of surfaces in a scene, our method yields superior localization and reconstruction relative to baselines that assume a static 3D scene. We carry out experiments on both synthetic RGB-D data streams as well as challenging real-world datasets, tracking scene dynamics in a human workspace over the course of several weeks.

count=12
* Robust Real-Time Tracking of Multiple Objects by Volumetric Mass Densities
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Possegger_Robust_Real-Time_Tracking_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Possegger_Robust_Real-Time_Tracking_2013_CVPR_paper.pdf)]
    * Title: Robust Real-Time Tracking of Multiple Objects by Volumetric Mass Densities
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Horst Possegger, Sabine Sternig, Thomas Mauthner, Peter M. Roth, Horst Bischof
    * Abstract: Combining foreground images from multiple views by projecting them onto a common ground-plane has been recently applied within many multi-object tracking approaches. These planar projections introduce severe artifacts and constrain most approaches to objects moving on a common 2D ground-plane. To overcome these limitations, we introduce the concept of an occupancy volume exploiting the full geometry and the objects' center of mass and develop an efficient algorithm for 3D object tracking. Individual objects are tracked using the local mass density scores within a particle filter based approach, constrained by a Voronoi partitioning between nearby trackers. Our method benefits from the geometric knowledge given by the occupancy volume to robustly extract features and train classifiers on-demand, when volumetric information becomes unreliable. We evaluate our approach on several challenging real-world scenarios including the public APIDIS dataset. Experimental evaluations demonstrate significant improvements compared to state-of-theart methods, while achieving real-time performance.

count=11
* Pix2map: Cross-Modal Retrieval for Inferring Street Maps From Images
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Pix2map_Cross-Modal_Retrieval_for_Inferring_Street_Maps_From_Images_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Pix2map_Cross-Modal_Retrieval_for_Inferring_Street_Maps_From_Images_CVPR_2023_paper.pdf)]
    * Title: Pix2map: Cross-Modal Retrieval for Inferring Street Maps From Images
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Xindi Wu, KwunFung Lau, Francesco Ferroni, Aljoša Ošep, Deva Ramanan
    * Abstract: Self-driving vehicles rely on urban street maps for autonomous navigation. In this paper, we introduce Pix2Map, a method for inferring urban street map topology directly from ego-view images, as needed to continually update and expand existing maps. This is a challenging task, as we need to infer a complex urban road topology directly from raw image data. The main insight of this paper is that this problem can be posed as cross-modal retrieval by learning a joint, cross-modal embedding space for images and existing maps, represented as discrete graphs that encode the topological layout of the visual surroundings. We conduct our experimental evaluation using the Argoverse dataset and show that it is indeed possible to accurately retrieve street maps corresponding to both seen and unseen roads solely from image data. Moreover, we show that our retrieved maps can be used to update or expand existing maps and even show proof-of-concept results for visual localization and image retrieval from spatial graphs.

count=11
* SIXO: Smoothing Inference with Twisted Objectives
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/fddc79681b2df2734c01444f9bc2a17e-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/fddc79681b2df2734c01444f9bc2a17e-Paper-Conference.pdf)]
    * Title: SIXO: Smoothing Inference with Twisted Objectives
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Dieterich Lawson, Allan Raventós, Andrew Warrington, Scott Linderman
    * Abstract: Sequential Monte Carlo (SMC) is an inference algorithm for state space models that approximates the posterior by sampling from a sequence of target distributions. The target distributions are often chosen to be the filtering distributions, but these ignore information from future observations, leading to practical and theoretical limitations in inference and model learning. We introduce SIXO, a method that instead learns target distributions that approximate the smoothing distributions, incorporating information from all observations. The key idea is to use density ratio estimation to fit functions that warp the filtering distributions into the smoothing distributions. We then use SMC with these learned targets to define a variational objective for model and proposal learning. SIXO yields provably tighter log marginal lower bounds and offers more accurate posterior inferences and parameter estimates in a variety of domains.

count=10
* Detecting Changes in 3D Structure of a Scene from Multi-view Images Captured by a Vehicle-Mounted Camera
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Sakurada_Detecting_Changes_in_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Sakurada_Detecting_Changes_in_2013_CVPR_paper.pdf)]
    * Title: Detecting Changes in 3D Structure of a Scene from Multi-view Images Captured by a Vehicle-Mounted Camera
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Ken Sakurada, Takayuki Okatani, Koichiro Deguchi
    * Abstract: This paper proposes a method for detecting temporal changes of the three-dimensional structure of an outdoor scene from its multi-view images captured at two separate times. For the images, we consider those captured by a camera mounted on a vehicle running in a city street. The method estimates scene structures probabilistically, not deterministically, and based on their estimates, it evaluates the probability of structural changes in the scene, where the inputs are the similarity of the local image patches among the multi-view images. The aim of the probabilistic treatment is to maximize the accuracy of change detection, behind which there is our conjecture that although it is difficult to estimate the scene structures deterministically, it should be easier to detect their changes. The proposed method is compared with the methods that use multi-view stereo (MVS) to reconstruct the scene structures of the two time points and then differentiate them to detect changes. The experimental results show that the proposed method outperforms such MVS-based methods.

count=9
* Density Invariant Contrast Maximization for Neuromorphic Earth Observations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/html/Arja_Density_Invariant_Contrast_Maximization_for_Neuromorphic_Earth_Observations_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Arja_Density_Invariant_Contrast_Maximization_for_Neuromorphic_Earth_Observations_CVPRW_2023_paper.pdf)]
    * Title: Density Invariant Contrast Maximization for Neuromorphic Earth Observations
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Sami Arja, Alexandre Marcireau, Richard L. Balthazor, Matthew G. McHarg, Saeed Afshar, Gregory Cohen
    * Abstract: Contrast maximization (CMax) techniques are widely used in event-based vision systems to estimate the motion parameters of the camera and generate high-contrast images. However, these techniques are noise-intolerance and suffer from the multiple extrema problem which arises when the scene contains more noisy events than structure, causing the contrast to be higher at multiple locations. This makes the task of estimating the camera motion extremely challenging, which is a problem for neuromorphic earth observation, because, without a proper estimation of the motion parameters, it is not possible to generate a map with high contrast, causing important details to be lost. Similar methods that use CMax addressed this problem by changing or augmenting the objective function to enable it to converge to the correct motion parameters. Our proposed solution overcomes the multiple extrema and noise-intolerance problems by correcting the warped event before calculating the contrast and offers the following advantages: it does not depend on the event data, it does not require a prior about the camera motion and keeps the rest of the CMax pipeline unchanged. This is to ensure that the contrast is only high around the correct motion parameters. Our approach enables the creation of better motion-compensated maps through an analytical compensation technique using a novel dataset from the International Space Station (ISS). Code is available at https://github.com/neuromorphicsystems/event_warping

count=9
* D^2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/d2cc447db9e56c13b993c11b45956281-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/d2cc447db9e56c13b993c11b45956281-Paper-Conference.pdf)]
    * Title: D^2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Tianhao Wu, Fangcheng Zhong, Andrea Tagliasacchi, Forrester Cole, Cengiz Oztireli
    * Abstract: Given a monocular video, segmenting and decoupling dynamic objects while recovering the static environment is a widely studied problem in machine intelligence. Existing solutions usually approach this problem in the image domain, limiting their performance and understanding of the environment. We introduce Decoupled Dynamic Neural Radiance Field (D^2NeRF), a self-supervised approach that takes a monocular video and learns a 3D scene representation which decouples moving objects, including their shadows, from the static background. Our method represents the moving objects and the static background by two separate neural radiance fields with only one allowing for temporal changes. A naive implementation of this approach leads to the dynamic component taking over the static one as the representation of the former is inherently more general and prone to overfitting. To this end, we propose a novel loss to promote correct separation of phenomena. We further propose a shadow field network to detect and decouple dynamically moving shadows. We introduce a new dataset containing various dynamic objects and shadows and demonstrate that our method can achieve better performance than state-of-the-art approaches in decoupling dynamic and static 3D objects, occlusion and shadow removal, and image segmentation for moving objects. Project page: https://d2nerf.github.io/

count=9
* Non-Stationary Bandits with Auto-Regressive Temporal Dependency
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/186a213d720568b31f9b59c085a23e5a-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/186a213d720568b31f9b59c085a23e5a-Paper-Conference.pdf)]
    * Title: Non-Stationary Bandits with Auto-Regressive Temporal Dependency
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Qinyi Chen, Negin Golrezaei, Djallel Bouneffouf
    * Abstract: Traditional multi-armed bandit (MAB) frameworks, predominantly examined under stochastic or adversarial settings, often overlook the temporal dynamics inherent in many real-world applications such as recommendation systems and online advertising. This paper introduces a novel non-stationary MAB framework that captures the temporal structure of these real-world dynamics through an auto-regressive (AR) reward structure. We propose an algorithm that integrates two key mechanisms: (i) an alternation mechanism adept at leveraging temporal dependencies to dynamically balance exploration and exploitation, and (ii) a restarting mechanism designed to discard out-of-date information. Our algorithm achieves a regret upper bound that nearly matches the lower bound, with regret measured against a robust dynamic benchmark. Finally, via a real-world case study on tourism demand prediction, we demonstrate both the efficacy of our algorithm and the broader applicability of our techniques to more complex, rapidly evolving time series.

count=8
* The Multi-Temporal Urban Development SpaceNet Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.pdf)]
    * Title: The Multi-Temporal Urban Development SpaceNet Dataset
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Adam Van Etten, Daniel Hogan, Jesus Martinez Manso, Jacob Shermeyer, Nicholas Weir, Ryan Lewis
    * Abstract: Satellite imagery analytics have numerous human development and disaster response applications, particularly when time series methods are involved. For example, quantifying population statistics is fundamental to 67 of the 231 United Nations Sustainable Development Goals Indicators, but the World Bank estimates that over 100 countries currently lack effective Civil Registration systems. To help address this deficit and develop novel computer vision methods for time series data, we present the Multi-Temporal Urban Development SpaceNet (MUDS, also known as SpaceNet 7) dataset. This open source dataset consists of medium resolution (4.0m) satellite imagery mosaics, which includes 24 images (one per month) covering >100 unique geographies, and comprises >40,000 km2 of imagery and exhaustive polygon labels of building footprints therein, totaling over 11M individual annotations. Each building is assigned a unique identifier (i.e. address), which permits tracking of individual objects over time. Label fidelity exceeds image resolution; this "omniscient labeling" is a unique feature of the dataset, and enables surprisingly precise algorithmic models to be crafted. We demonstrate methods to track building footprint construction (or demolition) over time, thereby directly assessing urbanization. Performance is measured with the newly developed SpaceNet Change and Object Tracking (SCOT) metric, which quantifies both object tracking as well as change detection. We demonstrate that despite the moderate resolution of the data, we are able to track individual building identifiers over time.

count=8
* Shadow Neural Radiance Fields for Multi-View Satellite Photogrammetry
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Derksen_Shadow_Neural_Radiance_Fields_for_Multi-View_Satellite_Photogrammetry_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Derksen_Shadow_Neural_Radiance_Fields_for_Multi-View_Satellite_Photogrammetry_CVPRW_2021_paper.pdf)]
    * Title: Shadow Neural Radiance Fields for Multi-View Satellite Photogrammetry
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Dawa Derksen, Dario Izzo
    * Abstract: We present a new generic method for shadow-aware multi-view satellite photogrammetry of Earth Observation scenes. Our proposed method, the Shadow Neural Radiance Field (S-NeRF) follows recent advances in implicit volumetric representation learning. For each scene, we train S-NeRF using very high spatial resolution optical images taken from known viewing angles. The learning requires no labels or shape priors: it is self-supervised by an image reconstruction loss. To accommodate for changing light source conditions both from a directional light source (the Sun) and a diffuse light source (the sky), we extend the NeRF approach in two ways. First, direct illumination from the Sun is modeled via a local light source visibility field. Second, indirect illumination from a diffuse light source is learned as a non-local color field as a function of the position of the Sun. Quantitatively, the combination of these factors reduces the altitude and color errors in shaded areas, compared to NeRF. The S-NeRF methodology not only performs novel view synthesis and full 3D shape estimation, it also enables shadow detection, albedo synthesis, and transient object filtering, without any explicit shape supervision.

count=7
* Modeling sRGB Camera Noise With Normalizing Flows
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Kousha_Modeling_sRGB_Camera_Noise_With_Normalizing_Flows_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Kousha_Modeling_sRGB_Camera_Noise_With_Normalizing_Flows_CVPR_2022_paper.pdf)]
    * Title: Modeling sRGB Camera Noise With Normalizing Flows
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Shayan Kousha, Ali Maleky, Michael S. Brown, Marcus A. Brubaker
    * Abstract: Noise modeling and reduction are fundamental tasks in low-level computer vision. They are particularly important for smartphone cameras relying on small sensors that exhibit visually noticeable noise. There has recently been renewed interest in using data-driven approaches to improve camera noise models via neural networks. These data-driven approaches target noise present in the raw-sensor image before it has been processed by the camera's image signal processor (ISP). Modeling noise in the RAW-rgb domain is useful for improving and testing the in-camera denoising algorithm; however, there are situations where the camera's ISP does not apply denoising or additional denoising is desired when the RAW-rgb domain image is no longer available. In such cases, the sensor noise propagates through the ISP to the final rendered image encoded in standard RGB (sRGB). The nonlinear steps on the ISP culminate in a significantly more complex noise distribution in the sRGB domain and existing raw-domain noise models are unable to capture the sRGB noise distribution. We propose a new sRGB-domain noise model based on normalizing flows that is capable of learning the complex noise distribution found in sRGB images under various ISO levels. Our normalizing flows-based approach outperforms other models by a large margin in noise modeling and synthesis tasks. We also show that image denoisers trained on noisy images synthesized with our noise model outperforms those trained with noise from baselines models.

count=7
* Ev-NeRF: Event Based Neural Radiance Field
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Hwang_Ev-NeRF_Event_Based_Neural_Radiance_Field_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Hwang_Ev-NeRF_Event_Based_Neural_Radiance_Field_WACV_2023_paper.pdf)]
    * Title: Ev-NeRF: Event Based Neural Radiance Field
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Inwoo Hwang, Junho Kim, Young Min Kim
    * Abstract: We present Ev-NeRF, a Neural Radiance Field derived from event data. While event cameras can measure subtle brightness changes in high frame rates, the measurements in low lighting or extreme motion suffer from significant domain discrepancy with complex noise. As a result, the performance of event-based vision tasks does not transfer to challenging environments, where the event cameras are expected to thrive over normal cameras. We find that the multi-view consistency of NeRF provides a powerful self-supervision signal for eliminating spurious measurements and extracting the consistent underlying structure despite highly noisy input. Instead of posed images of the original NeRF, the input to Ev-NeRF is the event measurements accompanied by the movements of the sensors. Using the loss function that reflects the measurement model of the sensor, Ev-NeRF creates an integrated neural volume that summarizes the unstructured and sparse data points captured for about 2-4 seconds. The generated neural volume can also produce intensity images from novel views with reasonable depth estimates, which can serve as a high-quality input to various vision-based tasks. Our results show that Ev-NeRF achieves competitive performance for intensity image reconstruction under extreme noise conditions and high-dynamic-range imaging.

count=7
* Spatio-Temporal Filter Analysis Improves 3D-CNN for Action Classification
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Kobayashi_Spatio-Temporal_Filter_Analysis_Improves_3D-CNN_for_Action_Classification_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kobayashi_Spatio-Temporal_Filter_Analysis_Improves_3D-CNN_for_Action_Classification_WACV_2024_paper.pdf)]
    * Title: Spatio-Temporal Filter Analysis Improves 3D-CNN for Action Classification
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Takumi Kobayashi, Jiaxing Ye
    * Abstract: As 2D-CNNs are growing in image recognition literature, 3D-CNNs are enthusiastically applied to video action recognition. While spatio-temporal (3D) convolution successfully stems from spatial (2D) convolution, it is still unclear how the convolution works for encoding temporal motion patterns in 3D-CNNs. In this paper, we shed light on the mechanism of feature extraction through analyzing the spatio-temporal filters from a temporal viewpoint. The analysis not only describes characteristics of the two action datasets, Something-Something-v2 (SSv2) and Kinetics-400, but also reveals how temporal dynamics are characterized through stacked spatio-temporal convolutions. Based on the analysis, we propose methods to improve temporal feature extraction, covering temporal filter representation and temporal data augmentation. The proposed method contributes to enlarging temporal receptive field of 3D-CNN without touching its fundamental architecture, thus keeping the computation cost. In the experiments on action classification using SSv2 and Kinetics-400, it produces favorable performance improvement of 3D-CNNs.

count=7
* M-Statistic for Kernel Change-Point Detection
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2015/hash/eb1e78328c46506b46a4ac4a1e378b91-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2015/file/eb1e78328c46506b46a4ac4a1e378b91-Paper.pdf)]
    * Title: M-Statistic for Kernel Change-Point Detection
    * Publisher: NeurIPS
    * Publication Date: `2015`
    * Authors: Shuang Li, Yao Xie, Hanjun Dai, Le Song
    * Abstract: Detecting the emergence of an abrupt change-point is a classic problem in statistics and machine learning. Kernel-based nonparametric statistics have been proposed for this task which make fewer assumptions on the distributions than traditional parametric approach. However, none of the existing kernel statistics has provided a computationally efficient way to characterize the extremal behavior of the statistic. Such characterization is crucial for setting the detection threshold, to control the significance level in the offline case as well as the average run length in the online case. In this paper we propose two related computationally efficient M-statistics for kernel-based change-point detection when the amount of background data is large. A novel theoretical result of the paper is the characterization of the tail probability of these statistics using a new technique based on change-of-measure. Such characterization provides us accurate detection thresholds for both offline and online cases in computationally efficient manner, without the need to resort to the more expensive simulations such as bootstrapping. We show that our methods perform well in both synthetic and real world data.

count=6
* A Temporal Scheme for Fast Learning of Image-Patch Correspondences in Realistic Multi-camera Setups
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W17/html/Eisenbach_A_Temporal_Scheme_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W17/papers/Eisenbach_A_Temporal_Scheme_2013_CVPR_paper.pdf)]
    * Title: A Temporal Scheme for Fast Learning of Image-Patch Correspondences in Realistic Multi-camera Setups
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Jens Eisenbach, Christian Conrad, Rudolf Mester
    * Abstract: This paper addresses the problem of finding corresponding image patches in multi-camera video streams by means of an unsupervised learning method. We determine patchto-patch correspondence relations ('correspondence priors') merely using information from a temporal change detection. Correspondence priors are essential for geometric multi-camera calibration, but are useful also for further vision tasks such as object tracking and recognition. Since any change detection method with reasonably performance can be applied, our method can be used as an encapsulated processing module and be integrated into existing systems without major structural changes. The only assumption that is made is that relative orientation of pairs of cameras may be arbitrary, but fixed, and that the observed scene shows visual activity. Experimental results show the applicability of the presented approach in real world scenarios where the camera views show large differences in orientation and position. Furthermore we show that a classic spatial matching pipeline, e.g., based on SIFT will typically fail to determine correspondences in these kinds of scenarios.

count=6
* Robust Kronecker-Decomposable Component Analysis for Low-Rank Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Bahri_Robust_Kronecker-Decomposable_Component_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Bahri_Robust_Kronecker-Decomposable_Component_ICCV_2017_paper.pdf)]
    * Title: Robust Kronecker-Decomposable Component Analysis for Low-Rank Modeling
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Mehdi Bahri, Yannis Panagakis, Stefanos Zafeiriou
    * Abstract: Dictionary learning and component analysis are part of one of the most well-studied and active research fields, at the intersection of signal and image processing, computer vision, and statistical machine learning. In dictionary learning, the current methods of choice are arguably K-SVD and its variants, which learn a dictionary (i.e., a decomposition) for sparse coding via Singular Value Decomposition. In robust component analysis, leading methods derive from Principal Component Pursuit (PCP), which recovers a low-rank matrix from sparse corruptions of unknown magnitude and support. However, K-SVD is sensitive to the presence of noise and outliers in the training set. Additionally, PCP does not provide a dictionary that respects the structure of the data (e.g., images), and requires expensive SVD computations when solved by convex relaxation. In this paper, we introduce a new robust decomposition of images by combining ideas from sparse dictionary learning and PCP. We propose a novel Kronecker-decomposable component analysis which is robust to gross corruption, can be used for low-rank modeling, and leverages separability to solve significantly smaller problems. We design an efficient learning algorithm by drawing links with a restricted form of tensor factorization. The effectiveness of the proposed approach is demonstrated on real-world applications, namely background subtraction and image denoising, by performing a thorough comparison with the current state of the art.

count=6
* A Unified Model for Near and Remote Sensing
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Workman_A_Unified_Model_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Workman_A_Unified_Model_ICCV_2017_paper.pdf)]
    * Title: A Unified Model for Near and Remote Sensing
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Scott Workman, Menghua Zhai, David J. Crandall, Nathan Jacobs
    * Abstract: We propose a novel convolutional neural network architecture for estimating geospatial functions such as population density, land cover, or land use. In our approach, we combine overhead and ground-level images in an end-to-end trainable neural network, which uses kernel regression and density estimation to convert features extracted from the ground-level images into a dense feature map. The output of this network is a dense estimate of the geospatial function in the form of a pixel-level labeling of the overhead image. To evaluate our approach, we created a large dataset of overhead and ground-level images from a major urban area with three sets of labels: land use, building function, and building age. We find that our approach is more accurate for all tasks, in some cases dramatically so.

count=6
* AGEM: Solving Linear Inverse Problems via Deep Priors and Sampling
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/49182f81e6a13cf5eaa496d51fea6406-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/49182f81e6a13cf5eaa496d51fea6406-Paper.pdf)]
    * Title: AGEM: Solving Linear Inverse Problems via Deep Priors and Sampling
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Bichuan Guo, Yuxing Han, Jiangtao Wen
    * Abstract: In this paper we propose to use a denoising autoencoder (DAE) prior to simultaneously solve a linear inverse problem and estimate its noise parameter. Existing DAE-based methods estimate the noise parameter empirically or treat it as a tunable hyper-parameter. We instead propose autoencoder guided EM, a probabilistically sound framework that performs Bayesian inference with intractable deep priors. We show that efficient posterior sampling from the DAE can be achieved via Metropolis-Hastings, which allows the Monte Carlo EM algorithm to be used. We demonstrate competitive results for signal denoising, image deblurring and image devignetting. Our method is an example of combining the representation power of deep learning with uncertainty quantification from Bayesian statistics.

count=6
* Continuous Meta-Learning without Tasks
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/cc3f5463bc4d26bc38eadc8bcffbc654-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf)]
    * Title: Continuous Meta-Learning without Tasks
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: James Harrison, Apoorva Sharma, Chelsea Finn, Marco Pavone
    * Abstract: Meta-learning is a promising strategy for learning to efficiently learn using data gathered from a distribution of tasks. However, the meta-learning literature thus far has focused on the task segmented setting, where at train-time, offline data is assumed to be split according to the underlying task, and at test-time, the algorithms are optimized to learn in a single task. In this work, we enable the application of generic meta-learning algorithms to settings where this task segmentation is unavailable, such as continual online learning with unsegmented time series data. We present meta-learning via online changepoint analysis (MOCA), an approach which augments a meta-learning algorithm with a differentiable Bayesian changepoint detection scheme. The framework allows both training and testing directly on time series data without segmenting it into discrete tasks. We demonstrate the utility of this approach on three nonlinear meta-regression benchmarks as well as two meta-image-classification benchmarks.

count=5
* Learning to Detect Carried Objects with Minimal Supervision
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/html/Dondera_Learning_to_Detect_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W22/papers/Dondera_Learning_to_Detect_2013_CVPR_paper.pdf)]
    * Title: Learning to Detect Carried Objects with Minimal Supervision
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Radu Dondera, Vlad Morariu, Larry Davis
    * Abstract: We propose a learning-based method for detecting carried objects that generates candidate image regions from protrusion, color contrast and occlusion boundary cues, and uses a classifier to filter out the regions unlikely to be carried objects. The method achieves higher accuracy than state of the art, which can only detect protrusions from the human shape, and the discriminative model it builds for the silhouette context-based region features generalizes well. To reduce annotation effort, we investigate training the model in a Multiple Instance Learning framework where the only available supervision is "walk" and "carry" labels associated with intervals of human tracks, i.e., the spatial extent of carried objects is not annotated. We present an extension to the miSVM algorithm that uses knowledge of the fraction of positive instances in positive bags and that scales to training sets of hundreds of thousands of instances.

count=5
* Can We Speed up 3D Scanning? A Cognitive and Geometric Analysis
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w40/html/Vaiapury_Can_We_Speed_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w40/Vaiapury_Can_We_Speed_ICCV_2017_paper.pdf)]
    * Title: Can We Speed up 3D Scanning? A Cognitive and Geometric Analysis
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Karthikeyan Vaiapury, Balamuralidhar Purushothaman, Arpan Pal, Swapna Agarwal, Brojeshwar Bhowmick
    * Abstract: The paper propose a cognitive inspired change detection method for the detection and localization of shape variations on point clouds. A well defined pipeline is introduced by proposing a coarse to fine approach: i) shape segmentation, ii) fine segment registration using attention blocks. Shape segmentation is obtained using covariance based method and fine segment registration is carried out using gravitational registration algorithm. In particular the introduction of this partition-based approach using visual attention mechanism improves the speed of deformation detection and localization. Some results are shown on synthetic data of house and aircraft models. Experimental results shows that this simple yet effective approach designed with an eye to scalability can detect and localize the deformation in a faster manner. A real world car usecase is also presented with some preliminary promising results useful for auditing and insurance claim tasks.

count=5
* Tracing the Influence of Predecessors on Trajectory Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023W/ROAD%2B%2B/html/Liu_Tracing_the_Influence_of_Predecessors_on_Trajectory_Prediction_ICCVW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023W/ROAD++/papers/Liu_Tracing_the_Influence_of_Predecessors_on_Trajectory_Prediction_ICCVW_2023_paper.pdf)]
    * Title: Tracing the Influence of Predecessors on Trajectory Prediction
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Mengmeng Liu, Hao Cheng, Michael Ying Yang
    * Abstract: In real-world traffic scenarios, agents such as pedestrians and car drivers often observe neighboring agents who exhibit similar behavior as examples and then mimic their actions to some extent in their own behavior. This information can serve as prior knowledge for trajectory prediction, which is unfortunately largely overlooked in current trajectory prediction models. This paper introduces a novel Predecessor-and-Successor (PnS) method that incorporates a predecessor tracing module to model the influence of predecessors (identified from concurrent neighboring agents) on the successor (target agent) within the same scene. The method utilizes the moving patterns of these predecessors to guide the predictor in trajectory prediction. PnS effectively aligns the motion encodings of the successor with multiple potential predecessors in a probabilistic manner, facilitating the decoding process. We demonstrate the effectiveness of PnS by integrating it into a graph-based predictor for pedestrian trajectory prediction on the ETH/UCY datasets, resulting in a new state-of-the-art performance. Furthermore, we replace the HD map-based scene-context module with our PnS method in a transformer-based predictor for vehicle trajectory prediction on the nuScenes dataset, showing that the predictor maintains good prediction performance even without relying on any map information.

count=5
* Infinite-Horizon Gaussian Processes
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2018/hash/b865367fc4c0845c0682bd466e6ebf4c-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2018/file/b865367fc4c0845c0682bd466e6ebf4c-Paper.pdf)]
    * Title: Infinite-Horizon Gaussian Processes
    * Publisher: NeurIPS
    * Publication Date: `2018`
    * Authors: Arno Solin, James Hensman, Richard E. Turner
    * Abstract: Gaussian processes provide a flexible framework for forecasting, removing noise, and interpreting long temporal datasets. State space modelling (Kalman filtering) enables these non-parametric models to be deployed on long datasets by reducing the complexity to linear in the number of data points. The complexity is still cubic in the state dimension m which is an impediment to practical application. In certain special cases (Gaussian likelihood, regular spacing) the GP posterior will reach a steady posterior state when the data are very long. We leverage this and formulate an inference scheme for GPs with general likelihoods, where inference is based on single-sweep EP (assumed density filtering). The infinite-horizon model tackles the cubic cost in the state dimensionality and reduces the cost in the state dimension m to O(m^2) per data point. The model is extended to online-learning of hyperparameters. We show examples for large finite-length modelling problems, and present how the method runs in real-time on a smartphone on a continuous data stream updated at 100 Hz.

count=4
* Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Teutsch_Robust_Detection_of_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Teutsch_Robust_Detection_of_CVPR_2016_paper.pdf)]
    * Title: Robust Detection of Moving Vehicles in Wide Area Motion Imagery
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Michael Teutsch, Michael Grinberg
    * Abstract: Multiple object tracking in Wide Area Motion Imagery (WAMI) data is usually based on initial detections coming from background subtraction or frame differencing. However, these methods are prone to produce split and merged detections. Appearance based vehicle detection can be an alternative but is not well-suited for WAMI data since classifier models are of weak discriminative power for vehicles in top view at low resolution. We introduce a moving vehicle detection algorithm that combines 2-frame differencing with a vehicle appearance model to improve object detection. Our main contributions are (1) integration of robust vehicle detection with split/merge handling and (2) estimation of assignment likelihoods between object hypotheses in consecutive frames using an appearance based similarity measure. Without using any prior knowledge, we achieve state-of-the-art detection rates and produce tracklets that considerably simplify the data association problem for multiple object tracking.

count=4
* Spatio-Temporal Self-Organizing Map Deep Network for Dynamic Object Detection From Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Du_Spatio-Temporal_Self-Organizing_Map_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Du_Spatio-Temporal_Self-Organizing_Map_CVPR_2017_paper.pdf)]
    * Title: Spatio-Temporal Self-Organizing Map Deep Network for Dynamic Object Detection From Videos
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Yang Du, Chunfeng Yuan, Bing Li, Weiming Hu, Stephen Maybank
    * Abstract: In dynamic object detection, it is challenging to construct an effective model to sufficiently characterize the spatial-temporal properties of the background. This paper proposes a new Spatio-Temporal Self-Organizing Map (STSOM) deep network to detect dynamic objects in complex scenarios. The proposed approach has several contributions: First, a novel STSOM shared by all pixels in a video frame is presented to efficiently model complex background. We exploit the fact that the motions of complex background have the global variation in the space and the local variation in the time, to train STSOM using the whole frames and the sequence of a pixel over time to tackle the variance of complex background. Second, a Bayesian parameter estimation based method is presented to learn thresholds automatically for all pixels to filter out the background. Last, in order to model the complex background more accurately, we extend the single-layer STSOM to the deep network. Then the background is filtered out layer by layer. Experimental results on CDnet 2014 dataset demonstrate that the proposed STSOM deep network outperforms numerous recently proposed methods in the overall performance and in most categories of scenarios.

count=4
* A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Lin_A_Prior-Less_Method_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lin_A_Prior-Less_Method_CVPR_2018_paper.pdf)]
    * Title: A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Chung-Ching Lin, Ying Hung
    * Abstract: This paper presents a prior-less method for tracking and clustering an unknown number of human faces and maintaining their individual identities in unconstrained videos. The key challenge is to accurately track faces with partial occlusion and drastic appearance changes in multiple shots resulting from significant variations of makeup, facial expression, head pose and illumination. To address this challenge, we propose a new multi-face tracking and re-identification algorithm, which provides high accuracy in face association in the entire video with automatic cluster number generation, and is robust to outliers. We develop a co-occurrence model of multiple body parts to seamlessly create face tracklets, and recursively link tracklets to construct a graph for extracting clusters. A Gaussian Process model is introduced to compensate the deep feature insufficiency, and is further used to refine the linking results. The advantages of the proposed algorithm are demonstrated using a variety of challenging music videos and newly introduced body-worn camera videos. The proposed method obtains significant improvements over the state of the art [51], while relying less on handling video-specific prior information to achieve high performance.

count=4
* Now You Shake Me: Towards Automatic 4D Cinema
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_Now_You_Shake_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Now_You_Shake_CVPR_2018_paper.pdf)]
    * Title: Now You Shake Me: Towards Automatic 4D Cinema
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Yuhao Zhou, Makarand Tapaswi, Sanja Fidler
    * Abstract: We are interested in enabling automatic 4D cinema by parsing physical and special effects from untrimmed movies. These include effects such as physical interactions, water splashing, light, and shaking, and are grounded to either a character in the scene or the camera. We collect a new dataset referred to as the Movie4D dataset which annotates over 9K effects in 63 movies. We propose a Conditional Random Field model atop a neural network that brings together visual and audio information, as well as semantics in the form of person tracks. Our model further exploits correlations of effects between different characters in the clip as well as across movie threads. We propose effect detection and classification as two tasks, and present results along with ablation studies on our dataset, paving the way towards 4D cinema in everyone’s homes.

count=4
* SyntheWorld: A Large-Scale Synthetic Dataset for Land Cover Mapping and Building Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Song_SyntheWorld_A_Large-Scale_Synthetic_Dataset_for_Land_Cover_Mapping_and_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Song_SyntheWorld_A_Large-Scale_Synthetic_Dataset_for_Land_Cover_Mapping_and_WACV_2024_paper.pdf)]
    * Title: SyntheWorld: A Large-Scale Synthetic Dataset for Land Cover Mapping and Building Change Detection
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Jian Song, Hongruixuan Chen, Naoto Yokoya
    * Abstract: Synthetic datasets, recognized for their cost effectiveness, play a pivotal role in advancing computer vision tasks and techniques. However, when it comes to remote sensing image processing, the creation of synthetic datasets becomes challenging due to the demand for larger-scale and more diverse 3D models. This complexity is compounded by the difficulties associated with real remote sensing datasets, including limited data acquisition and high annotation costs, which amplifies the need for high-quality synthetic alternatives. To address this, we present SyntheWorld, a synthetic dataset unparalleled in quality, diversity, and scale. It includes 40,000 images with submeter-level pixels and fine-grained land cover annotations of eight categories, and it also provides 40,000 pairs of bitemporal image pairs with building change annotations for building change detection. We conduct experiments on multiple benchmark remote sensing datasets to verify the effectiveness of SyntheWorld and to investigate the conditions under which our synthetic data yield advantages. The dataset is available at https://github.com/JTRNEO/SyntheWorld.

count=4
* Detection and Localization of Changes in Conditional Distributions
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/eb189151ced0ff808abafd16a51fec92-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/eb189151ced0ff808abafd16a51fec92-Paper-Conference.pdf)]
    * Title: Detection and Localization of Changes in Conditional Distributions
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Lizhen Nie, Dan Nicolae
    * Abstract: We study the change point problem that considers alterations in the conditional distribution of an inferential target on a set of covariates. This paired data scenario is in contrast to the standard setting where a sequentially observed variable is analyzed for potential changes in the marginal distribution. We propose new methodology for solving this problem, by starting from a simpler task that analyzes changes in conditional expectation, and generalizing the tools developed for that task to conditional distributions. Large sample properties of the proposed statistics are derived. In empirical studies, we illustrate the performance of the proposed method against baselines adapted from existing tools. Two real data applications are presented to demonstrate its potential.

count=3
* Underwater Moving Object Detection Using an End-to-End Encoder-Decoder Architecture and GraphSage With Aggregator and Refactoring
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/WiCV/html/Kapoor_Underwater_Moving_Object_Detection_Using_an_End-to-End_Encoder-Decoder_Architecture_and_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/WiCV/papers/Kapoor_Underwater_Moving_Object_Detection_Using_an_End-to-End_Encoder-Decoder_Architecture_and_CVPRW_2023_paper.pdf)]
    * Title: Underwater Moving Object Detection Using an End-to-End Encoder-Decoder Architecture and GraphSage With Aggregator and Refactoring
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Meghna Kapoor, Suvam Patra, Badri Narayan Subudhi, Vinit Jakhetiya, Ankur Bansal
    * Abstract: Underwater environments are greatly affected by several factors, including low visibility, high turbidity, back-scattering, dynamic background, etc., and hence pose challenges in object detection. Several algorithms consider convolutional neural networks to extract deep features and then object detection using the same. However, the dependency on the kernel's size and the network's depth results in fading relationships of latent space features and also are unable to characterize the spatial-contextual bonding of the pixels. Hence, they are unable to procure satisfactory results in complex underwater scenarios. To re-establish this relationship, we propose a unique architecture for underwater object detection where U-Net architecture is considered with the ResNet-50 backbone. Further, the latent space features from the encoder are fed to the decoder through a GraphSage model. GraphSage-based model is explored to reweight the node relationship in non-euclidean space using different aggregator functions and hence characterize the spatio-contextual bonding among the pixels. Further, we explored the dependency on different aggregator functions: mean, max, and LSTM, to evaluate the model's performance. We evaluated the proposed model on two underwater benchmark databases: F4Knowledge and underwater change detection. The performance of the proposed model is evaluated against eleven state-of-the-art techniques in terms of both visual and quantitative evaluation measures.

count=3
* iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.pdf)]
    * Title: iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Syed Waqas Zamir,  Aditya Arora,  Akshita Gupta,  Salman  Khan,  Guolei Sun,  Fahad Shahbaz Khan,  Fan Zhu,  Ling Shao,  Gui-Song Xia,  Xiang Bai
    * Abstract: Existing Earth Vision datasets are either suitable for semantic segmentation or object detection. In this work, we introduce the first benchmark dataset for instance segmentation in aerial imagery that combines instance-level object detection and pixel-level segmentation tasks. In comparison to instance segmentation in natural scenes, aerial images present unique challenges e.g., huge number of instances per image, large object-scale variations and abundant tiny objects. Our large-scale and densely annotated Instance Segmentation in Aerial Images Dataset (IS-AID) comes with 655,451 object instances for 15 categories across 2,806 high-resolution images. Such precise per-pixel annotations for each instance ensure accurate localization that is essential for detailed scene analysis. Compared to existing small-scale aerial image based instance segmentation datasets, IS-AID contains 15x the number of object categories and 5x the number of instances. We benchmark our dataset using two popular instance segmentation approaches for natural images, namely Mask R-CNN and PANet. In our experiments we show that direct application of off-the-shelf Mask R-CNN and PANet on aerial images provide sub-optimal instance segmentation results, thus requiring specialized solutions from the research community.

count=3
* Road Segmentation Using Multipass Single-Pol Synthetic Aperture Radar Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/Koch_Road_Segmentation_Using_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/Koch_Road_Segmentation_Using_2015_CVPR_paper.pdf)]
    * Title: Road Segmentation Using Multipass Single-Pol Synthetic Aperture Radar Imagery
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Mark W. Koch, Mary M. Moya, James G. Chow, Jeremy Goold, Rebecca Malinas
    * Abstract: Synthetic aperture radar (SAR) is a remote sensing technology that can truly operate 24/7. It's an all-weather system that can operate at any time except in the most extreme conditions. By making multiple passes over a wide area, a SAR can provide surveillance over a long time period. For high level processing it is convenient to segment and classify the SAR images into objects that identify various terrains and man-made structures that we call "static features." In this paper we concentrate on automatic road segmentation. This not only serves as a surrogate for finding other static features, but road detection in of itself is important for aligning SAR images with other data sources. In this paper we introduce a novel SAR image product that captures how different regions decorrelate at different rates. We also show how a modified Kolmogorov-Smirnov test can be used to model the static features even when the independent observation assumption is violated.

count=3
* Deep Occlusion Reasoning for Multi-Camera Multi-Target Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Baque_Deep_Occlusion_Reasoning_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Baque_Deep_Occlusion_Reasoning_ICCV_2017_paper.pdf)]
    * Title: Deep Occlusion Reasoning for Multi-Camera Multi-Target Detection
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Pierre Baque, Francois Fleuret, Pascal Fua
    * Abstract: People detection in 2D images has improved greatly in recent years. However, comparatively little of this progress has percolated into multi-camera multi-people tracking algorithms, whose performance still degrades severely when scenes become very crowded. In this work, we introduce a new architecture that combines Convolutional Neural Nets and Conditional Random Fields to explicitly resolve ambiguities. One of its key ingredients are high-order CRF terms that model potential occlusions and give our approach its robustness even when many people are present. Our model is trained end-to-end and we show that it outperforms several state-of-the-art algorithms on challenging scenes.

count=3
* JanusNet: Detection of Moving Objects From UAV Platforms
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/html/Zhao_JanusNet_Detection_of_Moving_Objects_From_UAV_Platforms_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/papers/Zhao_JanusNet_Detection_of_Moving_Objects_From_UAV_Platforms_ICCVW_2021_paper.pdf)]
    * Title: JanusNet: Detection of Moving Objects From UAV Platforms
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yuxiang Zhao, Khurram Shafique, Zeeshan Rasheed, Maoxu Li
    * Abstract: In this paper, we present JanusNet, an efficient CNN model that can perform online background subtraction and robustly detect moving targets using resource-constrained computational hardware on-board unmanned aerial vehicles (UAVs). Most of the existing work on background subtraction either assume that the camera is stationary or make limiting assumptions about the motion of the camera, the structure of the scene under observation, or the apparent motion of the background in video. JanusNet does not have these limitations and therefore, is applicable to a variety of UAV applications. JanusNet learns to extract and combine motion and appearance features to separate background and foreground to generate accurate pixel-wise masks of the moving objects. The network is trained using a simulated video dataset (generated using Unreal Engine 4) with ground-truth labels. Results on UCF Aerial and Kaggle Drone videos datasets show that the learned model transfers well to real UAV videos and can robustly detect moving targets in a wide variety of scenarios. Moreover, experiments on CDNet dataset demonstrate that even without explicitly assuming that the camera is stationary, the performance of JanusNet is comparable to traditional background subtraction methods.

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
* Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/html/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/papers/Beleznai_Reliable_Left_Luggage_2013_ICCV_paper.pdf)]
    * Title: Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Csaba Beleznai, Peter Gemeiner, Christian Zinner
    * Abstract: Reliable and timely detection of abandoned items in public places still represents an unsolved problem for automated visual surveillance. Typical surveilled scenarios are associated with high visual ambiguity such as shadows, occlusions, illumination changes and substantial clutter consisting of a mixture of dynamic and stationary objects. Motivated by these challenges we propose a reliable left item detection approach based on the combination of intensity and depth data from a passive stereo setup. The employed in-house developed stereo system consists of low-cost sensors and it is capable to perform detection in environments of up to 10m x 10m in size. The proposed algorithm is tested on a set of indoor sequences and compared to manually annotated ground truth data. Obtained results show that many failure modes of intensity-based approaches are absent and even small-sized objects such as a handbag can be reliably detected when left behind in a scene. The presented results display a very promising approach, which can robustly detect left luggage in dynamic environments at a close to real-time computational speed.

count=3
* Single Satellite Optical Imagery Dehazing using SAR Image Prior Based on conditional Generative Adversarial Networks
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.pdf)]
    * Title: Single Satellite Optical Imagery Dehazing using SAR Image Prior Based on conditional Generative Adversarial Networks
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Binghui Huang,  Li Zhi,  Chao Yang,  Fuchun Sun,  Yixu Song
    * Abstract: Satellite image dehazing aims at precisely retrieving the real situations of the obscured parts from the hazy remote sensing (RS) images, which is a challenging task since the hazy regions contain both ground features and haze components. Many approaches of removing haze focus on processing multi-spectral or RGB images, whereas few of them utilize multi-sensor data. The multi-sensor data fusion is significant to provide auxiliary information since RGB images are sensitive to atmospheric conditions. In this paper, a dataset called SateHaze1k is established and composed of 1200 pairs clear Synthetic Aperture Radar (SAR), hazy RGB, and corresponding ground truth images, which are divided into three degrees of the haze, i.e. thin, moderate, and thick fog. Moreover, we propose a novel fusion dehazing method to directly restore the haze-free RS images by using an end-to-end conditional generative adversarial network(cGAN). The proposed network combines the information of both RGB and SAR images to eliminate the image blurring. Besides, the dilated residual blocks of the generator can also sufficiently improve the dehazing effects. Our experiments demonstrate that the proposed method, which fuses the information of different sensors applied to the cloudy conditions, can achieve more precise results than other baseline models.

count=3
* BSUV-Net: A Fully-Convolutional Neural Network for Background Subtraction of Unseen Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_WACV_2020/html/Tezcan_BSUV-Net_A_Fully-Convolutional_Neural_Network_for_Background_Subtraction_of_Unseen_WACV_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_WACV_2020/papers/Tezcan_BSUV-Net_A_Fully-Convolutional_Neural_Network_for_Background_Subtraction_of_Unseen_WACV_2020_paper.pdf)]
    * Title: BSUV-Net: A Fully-Convolutional Neural Network for Background Subtraction of Unseen Videos
    * Publisher: WACV
    * Publication Date: `2020`
    * Authors: Ozan Tezcan,  Prakash Ishwar,  Janusz Konrad
    * Abstract: Background subtraction is a basic task in computer vision and video processing often applied as a pre-processing step for object tracking, people recognition, etc. Recently, a number of successful background-subtraction algorithms have been proposed, however nearly all of the top-performing ones are supervised. Crucially, their success relies upon the availability of some annotated frames of the test video during training. Consequently, their performance on completely "unseen" videos is undocumented in the literature. In this work, we propose a new, supervised, background-subtraction algorithm for unseen videos (BSUV-Net) based on a fully-convolutional neural network. The input to our network consists of the current frame and two background frames captured at different time scales along with their semantic segmentation maps. In order to reduce the chance of overfitting, we also introduce a new data-augmentation technique which mitigates the impact of illumination difference between the background frames and the current frame. On the CDNet-2014 dataset, BSUV-Net outperforms state-of-the-art algorithms evaluated on unseen videos in terms of several metrics including F-measure, recall and precision.

count=3
* Autoencoder-Based Background Reconstruction and Foreground Segmentation With Background Noise Estimation
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2023/html/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2023/papers/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.pdf)]
    * Title: Autoencoder-Based Background Reconstruction and Foreground Segmentation With Background Noise Estimation
    * Publisher: WACV
    * Publication Date: `2023`
    * Authors: Bruno Sauvalle, Arnaud de La Fortelle
    * Abstract: Even after decades of research, dynamic scene background reconstruction and foreground object segmentation are still considered as open problems due to various challenges such as illumination changes, camera movements, or background noise caused by air turbulence or moving trees. We propose in this paper to model the background of a frame sequence as a low dimensional manifold using an autoencoder and compare the reconstructed background provided by this autoencoder with the original image to compute the foreground/background segmentation masks. The main novelty of the proposed model is that the autoencoder is also trained to predict the background noise, which allows to compute for each frame a pixel-dependent threshold to perform the foreground segmentation. Although the proposed model does not use any temporal or motion information, it exceeds the state of the art for unsupervised background subtraction on the CDnet 2014 and LASIESTA datasets, with a significant improvement on videos where the camera is moving. It is also able to perform background reconstruction on some non-video image datasets.

count=3
* Locally private online change point detection
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/1c1d4df596d01da60385f0bb17a4a9e0-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/1c1d4df596d01da60385f0bb17a4a9e0-Paper.pdf)]
    * Title: Locally private online change point detection
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Tom Berrett, Yi Yu
    * Abstract: We study online change point detection problems under the constraint of local differential privacy (LDP) where, in particular, the statistician does not have access to the raw data. As a concrete problem, we study a multivariate nonparametric regression problem. At each time point $t$, the raw data are assumed to be of the form $(X_t, Y_t)$, where $X_t$ is a $d$-dimensional feature vector and $Y_t$ is a response variable. Our primary aim is to detect changes in the regression function $m_t(x)=\mathbb{E}(Y_t |X_t=x)$ as soon as the change occurs. We provide algorithms which respect the LDP constraint, which control the false alarm probability, and which detect changes with a minimal (minimax rate-optimal) delay. To quantify the cost of privacy, we also present the optimal rate in the benchmark, non-private setting. These non-private results are also new to the literature and thus are interesting \emph{per se}. In addition, we study the univariate mean online change point detection problem, under privacy constraints. This serves as the blueprint of studying more complicated private change point detection problems.

count=3
* Neural Tangent Kernel Maximum Mean Discrepancy
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/348a38cd25abeab0e440f37510e9b1fa-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/348a38cd25abeab0e440f37510e9b1fa-Paper.pdf)]
    * Title: Neural Tangent Kernel Maximum Mean Discrepancy
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Xiuyuan Cheng, Yao Xie
    * Abstract: We present a novel neural network Maximum Mean Discrepancy (MMD) statistic by identifying a new connection between neural tangent kernel (NTK) and MMD. This connection enables us to develop a computationally efficient and memory-efficient approach to compute the MMD statistic and perform NTK based two-sample tests towards addressing the long-standing challenge of memory and computational complexity of the MMD statistic, which is essential for online implementation to assimilating new samples. Theoretically, such a connection allows us to understand the NTK test statistic properties, such as the Type-I error and testing power for performing the two-sample test, by adapting existing theories for kernel MMD. Numerical experiments on synthetic and real-world datasets validate the theory and demonstrate the effectiveness of the proposed NTK-MMD statistic.

count=3
* Perception Test: A Diagnostic Benchmark for Multimodal Video Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/8540fba4abdc7f9f7a7b1cc6cd60e409-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: Perception Test: A Diagnostic Benchmark for Multimodal Video Models
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, joseph heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fréchette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, Joao Carreira
    * Abstract: We propose a novel multimodal video benchmark - the Perception Test - to evaluate the perception and reasoning skills of pre-trained multimodal models (e.g. Flamingo, BEiT-3, or GPT-4). Compared to existing benchmarks that focus on computational tasks (e.g. classification, detection or tracking), the Perception Test focuses on skills (Memory, Abstraction, Physics, Semantics) and types of reasoning (descriptive, explanatory, predictive, counterfactual) across video, audio, and text modalities, to provide a comprehensive and efficient evaluation tool. The benchmark probes pre-trained models for their transfer capabilities, in a zero-shot / few-shot or limited finetuning regime. For these purposes, the Perception Test introduces 11.6k real-world videos, 23s average length, designed to show perceptually interesting situations, filmed by around 100 participants worldwide. The videos are densely annotated with six types of labels (multiple-choice and grounded video question-answers, object and point tracks, temporal action and sound segments), enabling both language and non-language evaluations. The fine-tuning and validation splits of the benchmark are publicly available (CC-BY license), in addition to a challenge server with a held-out test split. Human baseline results compared to state-of-the-art video QA models show a significant gap in performance (91.4% vs 45.8%), suggesting that there is significant room for improvement in multimodal video understanding.Dataset, baselines code, and challenge server are available at https://github.com/deepmind/perception_test

count=2
* Ensemble Video Object Cut in Highly Dynamic Scenes
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Ren_Ensemble_Video_Object_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ren_Ensemble_Video_Object_2013_CVPR_paper.pdf)]
    * Title: Ensemble Video Object Cut in Highly Dynamic Scenes
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Xiaobo Ren, Tony X. Han, Zhihai He
    * Abstract: We consider video object cut as an ensemble of framelevel background-foreground object classifiers which fuses information across frames and refine their segmentation results in a collaborative and iterative manner. Our approach addresses the challenging issues of modeling of background with dynamic textures and segmentation of foreground objects from cluttered scenes. We construct patch-level bagof-words background models to effectively capture the background motion and texture dynamics. We propose a foreground salience graph (FSG) to characterize the similarity of an image patch to the bag-of-words background models in the temporal domain and to neighboring image patches in the spatial domain. We incorporate this similarity information into a graph-cut energy minimization framework for foreground object segmentation. The background-foreground classification results at neighboring frames are fused together to construct a foreground probability map to update the graph weights. The resulting object shapes at neighboring frames are also used as constraints to guide the energy minimization process during graph cut. Our extensive experimental results and performance comparisons over a diverse set of challenging videos with dynamic scenes, including the new Change Detection Challenge Dataset, demonstrate that the proposed ensemble video object cut method outperforms various state-ofthe-art algorithms.

count=2
* Online Dominant and Anomalous Behavior Detection in Videos
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Roshtkhari_Online_Dominant_and_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Roshtkhari_Online_Dominant_and_2013_CVPR_paper.pdf)]
    * Title: Online Dominant and Anomalous Behavior Detection in Videos
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Mehrsan Javan Roshtkhari, Martin D. Levine
    * Abstract: We present a novel approach for video parsing and simultaneous online learning of dominant and anomalous behaviors in surveillance videos. Dominant behaviors are those occurring frequently in videos and hence, usually do not attract much attention. They can be characterized by different complexities in space and time, ranging from a scene background to human activities. In contrast, an anomalous behavior is defined as having a low likelihood of occurrence. We do not employ any models of the entities in the scene in order to detect these two kinds of behaviors. In this paper, video events are learnt at each pixel without supervision using densely constructed spatio-temporal video volumes. Furthermore, the volumes are organized into large contextual graphs. These compositions are employed to construct a hierarchical codebook model for the dominant behaviors. By decomposing spatio-temporal contextual information into unique spatial and temporal contexts, the proposed framework learns the models of the dominant spatial and temporal events. Thus, it is ultimately capable of simultaneously modeling high-level behaviors as well as low-level spatial, temporal and spatio-temporal pixel level changes.

count=2
* Rolling Shutter Motion Deblurring
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Su_Rolling_Shutter_Motion_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Su_Rolling_Shutter_Motion_2015_CVPR_paper.pdf)]
    * Title: Rolling Shutter Motion Deblurring
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Shuochen Su, Wolfgang Heidrich
    * Abstract: Although motion blur and rolling shutter deformations are closely coupled artifacts in images taken with CMOS image sensors, the two phenomena have so far mostly been treated separately, with deblurring algorithms being unable to handle rolling shutter wobble, and rolling shutter algorithms being incapable of dealing with motion blur. We propose an approach that delivers sharp and undistorted output given a single rolling shutter motion blurred image. The key to achieving this is a global modeling of the camera motion trajectory, which enables each scanline of the image to be deblurred with the corresponding motion segment. We show the results of the proposed framework through experiments on synthetic and real data.

count=2
* Background Subtraction Using Local SVD Binary Pattern
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/html/Guo_Background_Subtraction_Using_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/Guo_Background_Subtraction_Using_CVPR_2016_paper.pdf)]
    * Title: Background Subtraction Using Local SVD Binary Pattern
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Lili Guo, Dan Xu, Zhenping Qiang
    * Abstract: Background subtraction is a basic problem for change detection in videos and also the first step of high-level computer vision applications. Most background subtraction methods rely on color and texture feature. However, due to illuminations changes in different scenes and affections of noise pixels, those methods often resulted in high false positives in a complex environment. To solve this problem, we propose an adaptive background subtraction model which uses a novel Local SVD Binary Pattern (named LSBP) feature instead of simply depending on color intensity. This feature can describe the potential structure of the local regions in a given image, thus, it can enhance the robustness to illumination variation, noise, and shadows. We use a sample consensus model which is well suited for our LSBP feature. Experimental results on CDnet 2012 dataset demonstrate that our background subtraction method using LSBP feature is more effective than many state-of-the-art methods.

count=2
* Joint Learning From Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Audebert_Joint_Learning_From_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/papers/Audebert_Joint_Learning_From_CVPR_2017_paper.pdf)]
    * Title: Joint Learning From Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Nicolas Audebert, Bertrand Le Saux, Sebastien Lefevre
    * Abstract: We investigate the use of OSM data for semantic labeling of EO images. Deep neural networks have been used in the past for remote sensing data classification from various sensors, including multispectral, hyperspectral, Radar and Lidar data. However, OSM is an abundant data source that has already been used as ground truth data, but rarely exploited as an input information layer. We study different use cases and deep network architectures to leverage this OSM data for semantic labeling of aerial and satellite images. Especially, we look into fusion based architectures and coarse-to-fine segmentation to include the OSM layer into multispectral-based deep fully convolutional networks. We illustrate how these methods can be used successfully on two public datasets: the ISPRS Potsdam and the DFC2017. We show that OSM data can efficiently be integrated into the vision-based deep learning models and that it significantly improves both the accuracy performance and the convergence.

count=2
* Learning Shape Trends: Parameter Estimation in Diffusions on Shape Manifolds
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w7/html/Staneva_Learning_Shape_Trends_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w7/papers/Staneva_Learning_Shape_Trends_CVPR_2017_paper.pdf)]
    * Title: Learning Shape Trends: Parameter Estimation in Diffusions on Shape Manifolds
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Valentina Staneva, Laurent Younes
    * Abstract: Learning the dynamics of shape is at the heart of many computer vision problems: object tracking, change detection, longitudinal shape analysis, trajectory classification, etc. In this work we address the problem of statistical inference of diffusion processes of shapes. We formulate a general It\^o diffusion on the manifold of deformable landmarks and propose several drift models for the evolution of shapes. We derive explicit formulas for the maximum likelihood estimators of the unknown parameters in these models, and demonstrate their convergence properties on simulated sequences when true parameters are known. We further discuss how these models can be extended to a more general non-parametric approach to shape estimation.

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
* TPNet: Trajectory Proposal Network for Motion Prediction
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_TPNet_Trajectory_Proposal_Network_for_Motion_Prediction_CVPR_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_TPNet_Trajectory_Proposal_Network_for_Motion_Prediction_CVPR_2020_paper.pdf)]
    * Title: TPNet: Trajectory Proposal Network for Motion Prediction
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Liangji Fang,  Qinhong Jiang,  Jianping Shi,  Bolei Zhou
    * Abstract: Making accurate motion prediction of the surrounding traffic agents such as pedestrians, vehicles, and cyclists is crucial for autonomous driving. Recent data-driven motion prediction methods have attempted to learn to directly regress the exact future position or its distribution from massive amount of trajectory data. However, it remains difficult for these methods to provide multimodal predictions as well as integrate physical constraints such as traffic rules and movable areas. In this work we propose a novel two-stage motion prediction framework, Trajectory Proposal Network (TPNet). TPNet first generates a candidate set of future trajectories as hypothesis proposals, then makes the final predictions by classifying and refining the proposals which meets the physical constraints. By steering the proposal generation process, safe and multimodal predictions are realized. Thus this framework effectively mitigates the complexity of motion prediction problem while ensuring the multimodal output. Experiments on four large-scale trajectory prediction datasets, i.e. the ETH, UCY, Apollo and Argoverse datasets, show that TPNet achieves the state-of-the-art results both quantitatively and qualitatively.

count=2
* DyStaB: Unsupervised Object Segmentation via Dynamic-Static Bootstrapping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_DyStaB_Unsupervised_Object_Segmentation_via_Dynamic-Static_Bootstrapping_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_DyStaB_Unsupervised_Object_Segmentation_via_Dynamic-Static_Bootstrapping_CVPR_2021_paper.pdf)]
    * Title: DyStaB: Unsupervised Object Segmentation via Dynamic-Static Bootstrapping
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Yanchao Yang, Brian Lai, Stefano Soatto
    * Abstract: We describe an unsupervised method to detect and segment portions of images of live scenes that, at some point in time, are seen moving as a coherent whole, which we refer to as objects. Our method first partitions the motion field by minimizing the mutual information between segments. Then, it uses the segments to learn object models that can be used for detection in a static image. Static and dynamic models are represented by deep neural networks trained jointly in a bootstrapping strategy, which enables extrapolation to previously unseen objects. While the training process requires motion, the resulting object segmentation network can be used on either static images or videos at inference time. As the volume of seen videos grows, more and more objects are seen moving, priming their detection, which then serves as a regularizer for new objects, turning our method into unsupervised continual learning to segment objects. Our models are compared to the state of the art in both video object segmentation and salient object detection. In the six benchmark datasets tested, our models compare favorably even to those using pixel-level supervision, despite requiring no manual annotation.

count=2
* Segmenting Across Places: The Need for Fair Transfer Learning With Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/html/Zhang_Segmenting_Across_Places_The_Need_for_Fair_Transfer_Learning_With_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Zhang_Segmenting_Across_Places_The_Need_for_Fair_Transfer_Learning_With_CVPRW_2022_paper.pdf)]
    * Title: Segmenting Across Places: The Need for Fair Transfer Learning With Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Miao Zhang, Harvineet Singh, Lazarus Chok, Rumi Chunara
    * Abstract: The increasing availability of high-resolution satellite imagery has enabled the use of machine learning to support land-cover measurement and inform policy-making. However, labelling satellite images is expensive and is available for only some locations. This prompts the use of transfer learning to adapt models from data-rich locations to others. Given the potential for high-impact applications of satellite imagery across geographies, a systematic assessment of transfer learning implications is warranted. In this work, we consider the task of land-cover segmentation and study the fairness implications of transferring models across locations. We leverage a large satellite image segmentation benchmark with 5987 images from 18 districts (9 urban and 9 rural). Via fairness metrics we quantify disparities in model performance along two axes -- across urban-rural locations and across land-cover classes. Findings show that state-of-the-art models have better overall accuracy in rural areas compared to urban areas, through unsupervised domain adaptation methods transfer learning better to urban versus rural areas and enlarge fairness gaps. In analysis of reasons for these findings, we show that raw satellite images are overall more dissimilar between source and target districts for rural than for urban locations. This work highlights the need to conduct fairness analysis for satellite imagery segmentation models and motivates the development of methods for fair transfer learning in order not to introduce disparities between places, particularly urban and rural locations.

count=2
* ZBS: Zero-Shot Background Subtraction via Instance-Level Background Modeling and Foreground Selection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/An_ZBS_Zero-Shot_Background_Subtraction_via_Instance-Level_Background_Modeling_and_Foreground_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/An_ZBS_Zero-Shot_Background_Subtraction_via_Instance-Level_Background_Modeling_and_Foreground_CVPR_2023_paper.pdf)]
    * Title: ZBS: Zero-Shot Background Subtraction via Instance-Level Background Modeling and Foreground Selection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yongqi An, Xu Zhao, Tao Yu, Haiyun Guo, Chaoyang Zhao, Ming Tang, Jinqiao Wang
    * Abstract: Background subtraction (BGS) aims to extract all moving objects in the video frames to obtain binary foreground segmentation masks. Deep learning has been widely used in this field. Compared with supervised-based BGS methods, unsupervised methods have better generalization. However, previous unsupervised deep learning BGS algorithms perform poorly in sophisticated scenarios such as shadows or night lights, and they cannot detect objects outside the pre-defined categories. In this work, we propose an unsupervised BGS algorithm based on zero-shot object detection called Zero-shot Background Subtraction ZBS. The proposed method fully utilizes the advantages of zero-shot object detection to build the open-vocabulary instance-level background model. Based on it, the foreground can be effectively extracted by comparing the detection results of new frames with the background model. ZBS performs well for sophisticated scenarios, and it has rich and extensible categories. Furthermore, our method can easily generalize to other tasks, such as abandoned object detection in unseen environments. We experimentally show that ZBS surpasses state-of-the-art unsupervised BGS methods by 4.70% F-Measure on the CDnet 2014 dataset. The code is released at https://github.com/CASIA-IVA-Lab/ZBS.

count=2
* AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning With Masked Autoencoders
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.pdf)]
    * Title: AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning With Masked Autoencoders
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Wele Gedara Chaminda Bandara, Naman Patel, Ali Gholami, Mehdi Nikkhah, Motilal Agrawal, Vishal M. Patel
    * Abstract: Masked Autoencoders (MAEs) learn generalizable representations for image, text, audio, video, etc., by reconstructing masked input data from tokens of the visible data. Current MAE approaches for videos rely on random patch, tube, or frame based masking strategies to select these tokens. This paper proposes AdaMAE, an adaptive masking strategy for MAEs that is end-to-end trainable. Our adaptive masking strategy samples visible tokens based on the semantic context using an auxiliary sampling network. This network estimates a categorical distribution over spacetime-patch tokens. The tokens that increase the expected reconstruction error are rewarded and selected as visible tokens, motivated by the policy gradient algorithm in reinforcement learning. We show that AdaMAE samples more tokens from the high spatiotemporal information regions, thereby allowing us to mask 95% of tokens, resulting in lower memory requirements and faster pre-training. We conduct ablation studies on the Something-Something v2 (SSv2) dataset to demonstrate the efficacy of our adaptive sampling approach and report state-of-the-art results of 70.0% and 81.7% in top-1 accuracy on SSv2 and Kinetics-400 action classification datasets with a ViT-Base backbone and 800 pre-training epochs. Code and pre-trained models are available at: https://github.com/wgcban/adamae.git

count=2
* Probability-Based Global Cross-Modal Upsampling for Pansharpening
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Probability-Based_Global_Cross-Modal_Upsampling_for_Pansharpening_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Probability-Based_Global_Cross-Modal_Upsampling_for_Pansharpening_CVPR_2023_paper.pdf)]
    * Title: Probability-Based Global Cross-Modal Upsampling for Pansharpening
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Zeyu Zhu, Xiangyong Cao, Man Zhou, Junhao Huang, Deyu Meng
    * Abstract: Pansharpening is an essential preprocessing step for remote sensing image processing. Although deep learning (DL) approaches performed well on this task, current upsampling methods used in these approaches only utilize the local information of each pixel in the low-resolution multispectral (LRMS) image while neglecting to exploit its global information as well as the cross-modal information of the guiding panchromatic (PAN) image, which limits their performance improvement. To address this issue, this paper develops a novel probability-based global cross-modal upsampling (PGCU) method for pan-sharpening. Precisely, we first formulate the PGCU method from a probabilistic perspective and then design an efficient network module to implement it by fully utilizing the information mentioned above while simultaneously considering the channel specificity. The PGCU module consists of three blocks, i.e., information extraction (IE), distribution and expectation estimation (DEE), and fine adjustment (FA). Extensive experiments verify the superiority of the PGCU method compared with other popular upsampling methods. Additionally, experiments also show that the PGCU module can help improve the performance of existing SOTA deep learning pansharpening methods. The codes are available at https://github.com/Zeyu-Zhu/PGCU.

count=2
* Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.pdf)]
    * Title: Scan2LoD3: Reconstructing Semantic 3D Building Models at LoD3 Using Ray Casting and Bayesian Networks
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Olaf Wysocki, Yan Xia, Magdalena Wysocki, Eleonora Grilli, Ludwig Hoegner, Daniel Cremers, Uwe Stilla
    * Abstract: Reconstructing semantic 3D building models at the level of detail (LoD) 3 is a long-standing challenge. Unlike mesh-based models, they require watertight geometry and object-wise semantics at the facade level. The principal challenge of such demanding semantic 3D reconstruction is reliable facade-level semantic segmentation of 3D input data. We present a novel method, called Scan2LoD3, that accurately reconstructs semantic LoD3 building models by improving facade-level semantic 3D segmentation. To this end, we leverage laser physics and 3D building model priors to probabilistically identify model conflicts. These probabilistic physical conflicts propose locations of model openings: Their final semantics and shapes are inferred in a Bayesian network fusing multimodal probabilistic maps of conflicts, 3D point clouds, and 2D images. To fulfill demanding LoD3 requirements, we use the estimated shapes to cut openings in 3D building priors and fit semantic 3D objects from a library of facade objects. Extensive experiments on the TUM city campus datasets demonstrate the superior performance of the proposed Scan2LoD3 over the state-of-the-art methods in facade-level detection, semantic segmentation, and LoD3 building model reconstruction. We believe our method can foster the development of probability-driven semantic 3D reconstruction at LoD3 since not only the high-definition reconstruction but also reconstruction confidence becomes pivotal for various applications such as autonomous driving and urban simulations.

count=2
* PLGSLAM: Progressive Neural Scene Represenation with Local to Global Bundle Adjustment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Deng_PLGSLAM_Progressive_Neural_Scene_Represenation_with_Local_to_Global_Bundle_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_PLGSLAM_Progressive_Neural_Scene_Represenation_with_Local_to_Global_Bundle_CVPR_2024_paper.pdf)]
    * Title: PLGSLAM: Progressive Neural Scene Represenation with Local to Global Bundle Adjustment
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tianchen Deng, Guole Shen, Tong Qin, Jianyu Wang, Wentao Zhao, Jingchuan Wang, Danwei Wang, Weidong Chen
    * Abstract: Neural implicit scene representations have recently shown encouraging results in dense visual SLAM. However existing methods produce low-quality scene reconstruction and low-accuracy localization performance when scaling up to large indoor scenes and long sequences. These limitations are mainly due to their single global radiance field with finite capacity which does not adapt to large scenarios. Their end-to-end pose networks are also not robust enough with the growth of cumulative errors in large scenes. To this end we introduce PLGSLAM a neural visual SLAM system capable of high-fidelity surface reconstruction and robust camera tracking in real-time. To handle large-scale indoor scenes PLGSLAM proposes a progressive scene representation method which dynamically allocates new local scene representation trained with frames within a local sliding window. This allows us to scale up to larger indoor scenes and improves robustness (even under pose drifts). In local scene representation PLGSLAM utilizes tri-planes for local high-frequency features with multi-layer perceptron (MLP) networks for the low-frequency feature achieving smoothness and scene completion in unobserved areas. Moreover we propose local-to-global bundle adjustment method with a global keyframe database to address the increased pose drifts on long sequences. Experimental results demonstrate that PLGSLAM achieves state-of-the-art scene reconstruction results and tracking performance across various datasets and scenarios (both in small and large-scale indoor environments).

count=2
* A Category Agnostic Model for Visual Rearrangment
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_A_Category_Agnostic_Model_for_Visual_Rearrangment_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_A_Category_Agnostic_Model_for_Visual_Rearrangment_CVPR_2024_paper.pdf)]
    * Title: A Category Agnostic Model for Visual Rearrangment
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Yuyi Liu, Xinhang Song, Weijie Li, Xiaohan Wang, Shuqiang Jiang
    * Abstract: This paper presents a novel category agnostic model for visual rearrangement task which can help an embodied agent to physically recover the shuffled scene configuration without any category concepts to the goal configuration. Previous methods usually follow a similar architecture completing the rearrangement task by aligning the scene changes of the goal and shuffled configuration according to the semantic scene graphs. However constructing scene graphs requires the inference of category labels which not only causes the accuracy drop of the entire task but also limits the application in real world scenario. In this paper we delve deep into the essence of visual rearrangement task and focus on the two most essential issues scene change detection and scene change matching. We utilize the movement and the protrusion of point cloud to accurately identify the scene changes and match these changes depending on the similarity of category agnostic appearance feature. Moreover to assist the agent to explore the environment more efficiently and comprehensively we propose a closer-aligned-retrace exploration policy aiming to observe more details of the scene at a closer distance. We conduct extensive experiments on AI2THOR Rearrangement Challenge based on RoomR dataset and a new multi-room multi-instance dataset MrMiR collected by us. The experimental results demonstrate the effectiveness of our proposed method.

count=2
* Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.pdf)]
    * Title: Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Ke Li, Di Wang, Zhangyuan Hu, Wenxuan Zhu, Shaofeng Li, Quan Wang
    * Abstract: Deep Convolutional Neural Networks (DCNNs) have achieved remarkable performance in synthetic aperture radar (SAR) object detection but this comes at the cost of tremendous computational resources partly due to extracting redundant features within a single convolutional layer. Recent works either delve into model compression methods or focus on the carefully-designed lightweight models both of which result in performance degradation. In this paper we propose an efficient convolution module for SAR object detection called SFS-Conv which increases feature diversity within each convolutional layer through a shunt-perceive-select strategy. Specifically we shunt input feature maps into space and frequency aspects. The former perceives the context of various objects by dynamically adjusting receptive field while the latter captures abundant frequency variations and textural features via fractional Gabor transformer. To adaptively fuse features from space and frequency aspects a parameter-free feature selection module is proposed to ensure that the most representative and distinctive information are preserved. With SFS-Conv we build a lightweight SAR object detection network called SFS-CNet. Experimental results show that SFS-CNet outperforms state-of-the-art (SoTA) models on a series of SAR object detection benchmarks while simultaneously reducing both the model size and computational cost.

count=2
* Detecting Out-Of-Distribution Earth Observation Images with Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Le_Bellier_Detecting_Out-Of-Distribution_Earth_Observation_Images_with_Diffusion_Models_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Le_Bellier_Detecting_Out-Of-Distribution_Earth_Observation_Images_with_Diffusion_Models_CVPRW_2024_paper.pdf)]
    * Title: Detecting Out-Of-Distribution Earth Observation Images with Diffusion Models
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Georges Le Bellier, Nicolas Audebert
    * Abstract: Earth Observation imagery can capture rare and unusual events such as disasters and major landscape changes whose visual appearance contrasts with the usual observations. Deep models trained on common remote sensing data will output drastically different features for these out-of-distribution samples compared to those closer to their training dataset. Detecting them could therefore help anticipate changes in the observations either geographical or environmental. In this work we show that the reconstruction error of diffusion models can effectively serve as unsupervised out-of-distribution detectors for remote sensing images using them as a plausibility score. Moreover we introduce ODEED a novel reconstruction-based scorer using the probability-flow ODE of diffusion models. We validate it experimentally on SpaceNet 8 with various scenarios such as classical OOD detection with geographical shift and near-OOD setups: pre/post-flood and non-flooded/flooded image recognition. We show that our ODEED scorer significantly outperforms other diffusion-based and discriminative baselines on the more challenging near-OOD scenarios of flood image detection where OOD images are close to the distribution tail. We aim to pave the way towards better use of generative models for anomaly detection in remote sensing.

count=2
* A Multi-sensor Fusion Framework in 3-D
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/html/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W13/papers/Jain_A_Multi-sensor_Fusion_2013_CVPR_paper.pdf)]
    * Title: A Multi-sensor Fusion Framework in 3-D
    * Publisher: CVPR
    * Publication Date: `2013`
    * Authors: Vishal Jain, Andrew C. Miller, Joseph L. Mundy
    * Abstract: The majority of existing image fusion techniques operate in the 2-d image domain which perform well for imagery of planar regions but fails in presence of any 3-d relief and provides inaccurate alignment of imagery from different sensors. A framework for multi-sensor image fusion in 3-d is proposed in this paper. The imagery from different sensors, specifically EO and IR, are fused in a common 3-d reference coordinate frame. A dense probabilistic and volumetric 3-d model is reconstructed from each of the sensors. The imagery is registered by aligning the 3-d models as the underlying 3-d structure in the images is the true invariant information. The image intensities are back-projected onto a 3-d model and every discretized location (voxel) of the 3-d model stores an array of intensities from different modalities. This 3-d model is forward-projected to produce a fused image of EO and IR from any viewpoint.

count=2
* Spectral-360: A Physics-Based Technique for Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Sedky_Spectral-360_A_Physics-Based_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Sedky_Spectral-360_A_Physics-Based_2014_CVPR_paper.pdf)]
    * Title: Spectral-360: A Physics-Based Technique for Change Detection
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Mohamed Sedky, Mansour Moniri, Claude C. Chibelushi
    * Abstract: This paper presents and assesses a novel physics-based change detection technique, Spectral-360, which is based on the dichromatic color reflectance model. This approach, uses image formation models to computationally estimate, from the camera output, a consistent physics-based color descriptor of the spectral reflectance of surfaces visible in the image, and then to measure the similarity between the full-spectrum reflectance of the background and foreground pixels to segment the foreground from a static background. This method represents a new approach to change detection, using explicit hypotheses about the physics that create images. The assumptions which have been made are that diffuse-only-reflection is applicable, and the existence of a dominant illuminant. The objective evaluation performed using the 'changedetection.net 2014' dataset shows that our Spectral-360 method outperforms most state-of-the-art methods.

count=2
* CDnet 2014: An Expanded Change Detection Benchmark Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Wang_CDnet_2014_An_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Wang_CDnet_2014_An_2014_CVPR_paper.pdf)]
    * Title: CDnet 2014: An Expanded Change Detection Benchmark Dataset
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Yi Wang, Pierre-Marc Jodoin, Fatih Porikli, Janusz Konrad, Yannick Benezeth, Prakash Ishwar
    * Abstract: Change detection is one of the most important low-level tasks in video analytics. In 2012, we introduced the changedetection.net (CDnet) benchmark, a video dataset devoted to the evaluation of change and motion detection approaches. Here, we present the latest release of the CDnet dataset, which includes 22 additional videos (~70,000 pixel-wise annotated frames) spanning 5 new categories that incorporate challenges encountered in many surveillance settings. We describe these categories in detail and provide an overview of the results of more than a dozen methods submitted to the IEEE Change Detection Workshop 2014. We highlight strengths and weaknesses of these methods and identify remaining issues in change detection.

count=2
* Going Unconstrained With Rolling Shutter Deblurring
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/R._Going_Unconstrained_With_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/R._Going_Unconstrained_With_ICCV_2017_paper.pdf)]
    * Title: Going Unconstrained With Rolling Shutter Deblurring
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Mahesh Mohan M. R., A. N. Rajagopalan, Gunasekaran Seetharaman
    * Abstract: Most present-day imaging devices are equipped with CMOS sensors. Motion blur is a common artifact in hand-held cameras. Because CMOS sensors mostly employ a rolling shutter (RS), the motion deblurring problem takes on a new dimension. Although few works have recently addressed this problem, they suffer from many constraints including heavy computational cost, need for precise sensor information, and inability to deal with wide-angle systems (which most cell-phone and drone cameras are) and irregular camera trajectory. In this work, we propose a model for RS blind motion deblurring that mitigates these issues significantly. Comprehensive comparisons with state-of-the-art methods reveal that our approach not only exhibits significant computational gains and unconstrained functionality but also leads to improved deblurring performance.

count=2
* Geography-Aware Self-Supervised Learning
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.pdf)]
    * Title: Geography-Aware Self-Supervised Learning
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell, Stefano Ermon
    * Abstract: Contrastive learning methods have significantly narrowed the gap between supervised and unsupervised learning on computer vision tasks. In this paper, we explore their application to geo-located datasets, e.g. remote sensing, where unlabeled data is often abundant but labeled data is scarce. We first show that due to their different characteristics, a non-trivial gap persists between contrastive and supervised learning on standard benchmarks. To close the gap, we propose novel training methods that exploit the spatio-temporal structure of remote sensing data. We leverage spatially aligned images over time to construct temporal positive pairs in contrastive learning and geo-location to design pre-text tasks. Our experiments show that our proposed method closes the gap between contrastive and supervised learning on image classification, object detection and semantic segmentation for remote sensing. Moreover, we demonstrate that the proposed method can also be applied to geo-tagged ImageNet images, improving downstream performance on various tasks.

count=2
* Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/papers/Giraldo_Graph_CNN_for_Moving_Object_Detection_in_Complex_Environments_From_ICCVW_2021_paper.pdf)]
    * Title: Graph CNN for Moving Object Detection in Complex Environments From Unseen Videos
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Jhony H. Giraldo, Sajid Javed, Naoufel Werghi, Thierry Bouwmans
    * Abstract: Moving Object Detection (MOD) is a fundamental step for many computer vision applications. MOD becomes very challenging when a video sequence captured from a static or moving camera suffers from the challenges: camouflage, shadow, dynamic backgrounds, and lighting variations, to name a few. Deep learning methods have been successfully applied to address MOD with competitive performance. However, in order to handle the overfitting problem, deep learning methods require a large amount of labeled data which is a laborious task as exhaustive annotations are always not available. Moreover, some MOD deep learning methods show performance degradation in the presence of unseen video sequences because the testing and training splits of the same sequences are involved during the network learning process. In this work, we pose the problem of MOD as a node classification problem using Graph Convolutional Neural Networks (GCNNs). Our algorithm, dubbed as GraphMOD-Net, encompasses instance segmentation, background initialization, feature extraction, and graph construction. GraphMOD-Net is tested on unseen videos and outperforms state-of-the-art methods in unsupervised, semi-supervised, and supervised learning in several challenges of the Change Detection 2014 (CDNet2014) and UCSD background subtraction datasets.

count=2
* Automatic Open-World Reliability Assessment
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2021/html/Jafarzadeh_Automatic_Open-World_Reliability_Assessment_WACV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2021/papers/Jafarzadeh_Automatic_Open-World_Reliability_Assessment_WACV_2021_paper.pdf)]
    * Title: Automatic Open-World Reliability Assessment
    * Publisher: WACV
    * Publication Date: `2021`
    * Authors: Mohsen Jafarzadeh, Touqeer Ahmad, Akshay Raj Dhamija, Chunchun Li, Steve Cruz, Terrance E. Boult
    * Abstract: Image classification in the open-world must handle out-of-distribution (OOD) images. Systems should ideally reject OOD images, or they will map atop of known classes and reduce reliability. Using open-set classifiers that can reject OOD inputs can help. However, optimal accuracy of open-set classifiers depend on the frequency of OOD data. Thus, for either standard or open-set classifiers, it is important to be able to determine when the world changes and increasing OOD inputs will result in reduced system reliability. However, during operations, we cannot directly assess accuracy as there are no labels. Thus, the reliability assessment of these classifiers must be done by human operators, made more complex because networks are not 100% accurate, so some failures are to be expected. To automate this process, herein, we formalize the open-world recognition reliability problem and propose multiple automatic reliability assessment policies to address this new problem using only the distribution of reported scores/probability data. The distributional algorithms can be applied to both classic classifiers with SoftMax as well as the open-world Extreme Value Machine (EVM) to provide automated reliability assessment. We show that all of the new algorithms significantly outperform detection using the mean of SoftMax.

count=2
* Implicit Neural Representation for Change Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Naylor_Implicit_Neural_Representation_for_Change_Detection_WACV_2024_paper.pdf)]
    * Title: Implicit Neural Representation for Change Detection
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Peter Naylor, Diego Di Carlo, Arianna Traviglia, Makoto Yamada, Marco Fiorucci
    * Abstract: Identifying changes in a pair of 3D aerial LiDAR point clouds, obtained during two distinct time periods over the same geographic region presents a significant challenge due to the disparities in spatial coverage and the presence of noise in the acquisition system. The most commonly used approaches to detecting changes in point clouds are based on supervised methods which necessitate extensive labelled data often unavailable in real-world applications. To address these issues, we propose an unsupervised approach that comprises two components: Implicit Neural Representation (INR) for continuous shape reconstruction and a Gaussian Mixture Model for categorising changes. INR offers a grid-agnostic representation for encoding bi-temporal point clouds, with unmatched spatial support that can be regularised to enhance high-frequency details and reduce noise. The reconstructions at each timestamp are compared at arbitrary spatial scales, leading to a significant increase in detection capabilities. We apply our method to a benchmark dataset comprising simulated LiDAR point clouds for urban sprawling. This dataset encompasses diverse challenging scenarios, varying in resolutions, input modalities and noise levels. This enables a comprehensive multi-scenario evaluation, comparing our method with the current state-of-the-art approach. We outperform the previous methods by a margin of 10% in the intersection over union metric. In addition, we put our techniques to practical use by applying them in a real-world scenario to identify instances of illicit excavation of archaeological sites and validate our results by comparing them with findings from field experts.

count=2
* Structure Learning with Side Information: Sample Complexity
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2019/hash/e025b6279c1b88d3ec0eca6fcb6e6280-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2019/file/e025b6279c1b88d3ec0eca6fcb6e6280-Paper.pdf)]
    * Title: Structure Learning with Side Information: Sample Complexity
    * Publisher: NeurIPS
    * Publication Date: `2019`
    * Authors: Saurabh Sihag, Ali Tajer
    * Abstract: Graphical models encode the stochastic dependencies among random variables (RVs). The vertices represent the RVs, and the edges signify the conditional dependencies among the RVs. Structure learning is the process of inferring the edges by observing realizations of the RVs, and it has applications in a wide range of technological, social, and biological networks. Learning the structure of graphs when the vertices are treated in isolation from inferential information known about them is well-investigated. In a wide range of domains, however, often there exist additional inferred knowledge about the structure, which can serve as valuable side information. For instance, the gene networks that represent different subtypes of the same cancer share similar edges across all subtypes and also have exclusive edges corresponding to each subtype, rendering partially similar graphical models for gene expression in different cancer subtypes. Hence, an inferential decision regarding a gene network can serve as side information for inferring other related gene networks. When such side information is leveraged judiciously, it can translate to significant improvement in structure learning. Leveraging such side information can be abstracted as inferring structures of distinct graphical models that are {\sl partially} similar. This paper focuses on Ising graphical models, and considers the problem of simultaneously learning the structures of two {\sl partially} similar graphs, where any inference about the structure of one graph offers side information for the other graph. The bounded edge subclass of Ising models is considered, and necessary conditions (information-theoretic ), as well as sufficient conditions (algorithmic) for the sample complexity for achieving a bounded probability of error, are established. Furthermore, specific regimes are identified in which the necessary and sufficient conditions coincide, rendering the optimal sample complexity.

count=2
* Bandit Quickest Changepoint Detection
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2021/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2021/file/f3a4ff4839c56a5f460c88cce3666a2b-Paper.pdf)]
    * Title: Bandit Quickest Changepoint Detection
    * Publisher: NeurIPS
    * Publication Date: `2021`
    * Authors: Aditya Gopalan, Braghadeesh Lakshminarayanan, Venkatesh Saligrama
    * Abstract: Many industrial and security applications employ a suite of sensors for detecting abrupt changes in temporal behavior patterns. These abrupt changes typically manifest locally, rendering only a small subset of sensors informative. Continuous monitoring of every sensor can be expensive due to resource constraints, and serves as a motivation for the bandit quickest changepoint detection problem, where sensing actions (or sensors) are sequentially chosen, and only measurements corresponding to chosen actions are observed. We derive an information-theoretic lower bound on the detection delay for a general class of finitely parameterized probability distributions. We then propose a computationally efficient online sensing scheme, which seamlessly balances the need for exploration of different sensing options with exploitation of querying informative actions. We derive expected delay bounds for the proposed scheme and show that these bounds match our information-theoretic lower bounds at low false alarm rates, establishing optimality of the proposed method. We then perform a number of experiments on synthetic and real datasets demonstrating the effectiveness of our proposed method.

count=2
* Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/2ab3163ee384cd46baa7f1abb2b1bf19-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/2ab3163ee384cd46baa7f1abb2b1bf19-Paper-Conference.pdf)]
    * Title: Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Katie Luo, Zhenzhen Liu, Xiangyu Chen, Yurong You, Sagie Benaim, Cheng Perng Phoo, Mark Campbell, Wen Sun, Bharath Hariharan, Kilian Q. Weinberger
    * Abstract: Recent advances in machine learning have shown that Reinforcement Learning from Human Feedback (RLHF) can improve machine learning models and align them with human preferences. Although very successful for Large Language Models (LLMs), these advancements have not had a comparable impact in research for autonomous vehicles—where alignment with human expectations can be imperative. In this paper, we propose to adapt similar RL-based methods to unsupervised object discovery, i.e. learning to detect objects from LiDAR points without any training labels. Instead of labels, we use simple heuristics to mimic human feedback. More explicitly, we combine multiple heuristics into a simple reward function that positively correlates its score with bounding box accuracy, i.e., boxes containing objects are scored higher than those without. We start from the detector’s own predictions to explore the space and reinforce boxes with high rewards through gradient updates. Empirically, we demonstrate that our approach is not only more accurate, but also orders of magnitudes faster to train compared to prior works on object discovery. Code is available at https://github.com/katieluo88/DRIFT.

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
* Protecting Against Screenshots: An Image Processing Approach
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Chia_Protecting_Against_Screenshots_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Chia_Protecting_Against_Screenshots_2015_CVPR_paper.pdf)]
    * Title: Protecting Against Screenshots: An Image Processing Approach
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Alex Yong-Sang Chia, Udana Bandara, Xiangyu Wang, Hiromi Hirano
    * Abstract: Motivated by reasons related to data security and privacy, we propose a method to limit meaningful visual contents of a display from being captured by screenshots. Traditional methods take a system architectural approach to protect against screenshots. We depart from this framework, and instead exploit image processing techniques to distort visual data of a display and present the distorted data to the viewer. Given that a screenshot captures distorted visual contents, it yields limited useful data. We exploit the human visual system to empower viewers to automatically and mentally recover the distorted contents into a meaningful form in real-time. Towards this end, we leverage on findings from psychological studies which show that blending of visual information from recent and current fixations enables human to form meaningful representation of a scene. We model this blending of information by an additive process, and exploit this to design a visual contents distortion algorithm that supports real-time contents recovery by the human visual system. Our experiments and user study demonstrate the feasibility of our method to allow viewers to readily interpret visual contents of a display, while limiting meaningful contents from being captured by screenshots.

count=1
* Statistical Inference Models for Image Datasets With Systematic Variations
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Kim_Statistical_Inference_Models_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Kim_Statistical_Inference_Models_2015_CVPR_paper.pdf)]
    * Title: Statistical Inference Models for Image Datasets With Systematic Variations
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Won Hwa Kim, Barbara B. Bendlin, Moo K. Chung, Sterling C. Johnson, Vikas Singh
    * Abstract: Statistical analysis of longitudinal or cross sectionalbrain imaging data to identify effects of neurodegenerative diseases is a fundamental task in various studies in neuroscience. However, when there are systematic variations in the images due to parameters changes such as changes in the scanner protocol, hardware changes, or when combining data from multi-site studies, the statistical analysis becomes problematic. Motivated by this scenario, the goal of this paper is to develop a unified statistical solution to the problem of systematic variations in statistical image analysis. Based in part on recent literature in harmonic analysis on diffusion maps, we propose an algorithm which compares operators that are resilient to the systematic variations described above. These operators are derived from the empirical measurements of the image data and provide an efficient surrogate to capturing the actual changes across images. We also establish a connection between our method to the design of Wavelets in non-Euclidean space. To evaluate the proposed ideas, we present various experimental results on detecting changes in simulations as well as show how the method offers improved statistical power in the analysis of longitudinal real PIB-PET imaging data acquired from participants at risk for Alzheimer's disease(AD).

count=1
* Geospatial Correspondences for Multimodal Registration
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Marcos_Geospatial_Correspondences_for_CVPR_2016_paper.pdf)]
    * Title: Geospatial Correspondences for Multimodal Registration
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Diego Marcos, Raffay Hamid, Devis Tuia
    * Abstract: The growing availability of very high resolution (<1 m/pixel) satellite and aerial images has opened up unprecedented opportunities to monitor and analyze the evolution of land-cover and land-use across the world. To do so, images of the same geographical areas acquired at different times and, potentially, with different sensors must be efficiently parsed to update maps and detect land-cover changes. However, a naive transfer of ground truth labels from one location in the source image to the corresponding location in the target image is not generally feasible, as these images are often only loosely registered (with up to +- 50m of non-uniform errors). Furthermore, land-cover changes in an area over time must be taken into account for an accurate ground truth transfer. To tackle these challenges, we propose a mid-level sensor-invariant representation that encodes image regions in terms of the spatial distribution of their spectral neighbors. We incorporate this representation in a Markov Random Field to simultaneously account for nonlinear mis-registrations and enforce locality priors to find matches between multi-sensor images. We show how our approach can be used to assist in several multimodal land-cover update and change detection problems.

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
* Detecting Anomalous Objects on Mobile Platforms
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/html/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w29/papers/Lawson_Detecting_Anomalous_Objects_CVPR_2016_paper.pdf)]
    * Title: Detecting Anomalous Objects on Mobile Platforms
    * Publisher: CVPR
    * Publication Date: `2016`
    * Authors: Wallace Lawson, Laura Hiatt, Keith Sullivan
    * Abstract: We present an approach where a robot patrols a fixed path through an environment, autonomously locating suspicious or anomalous objects. To learn, the robot patrols this environment building a dictionary describing what is present. The dictionary is built by clustering features from a deep neural network. The objects present vary depending on the scene, which means that an object that is anomalous in one scene may be completely normal in another. To reason about this, the robot uses a computational cognitive model to learn the dictionary elements that are typically found in each scene. Once the dictionary and model has been built, the robot can patrol the environment matching objects against the dictionary, and querying the model to find the most likely objects present and to determine which objects (if any) are anomalous. We demonstrate our approach by patrolling two indoor and one outdoor environments.

count=1
* Minimum Delay Moving Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Lao_Minimum_Delay_Moving_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lao_Minimum_Delay_Moving_CVPR_2017_paper.pdf)]
    * Title: Minimum Delay Moving Object Detection
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Dong Lao, Ganesh Sundaramoorthi
    * Abstract: We present a general framework and method for detection of an object in a video based on apparent motion. The object moves relative to background motion at some unknown time in the video, and the goal is to detect and segment the object as soon it moves in an online manner. Due to unreliability of motion between frames, more than two frames are needed to reliably detect the object. Our method is designed to detect the object(s) with minimum delay, i.e., frames after the object moves, constraining the false alarms. Experiments on a new extensive dataset for moving object detection show that our method achieves less delay for all false alarm constraints than existing state-of-the-art.

count=1
* Dynamic Time-Of-Flight
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Schober_Dynamic_Time-Of-Flight_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Schober_Dynamic_Time-Of-Flight_CVPR_2017_paper.pdf)]
    * Title: Dynamic Time-Of-Flight
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Michael Schober, Amit Adam, Omer Yair, Shai Mazor, Sebastian Nowozin
    * Abstract: Time-of-flight (TOF) depth cameras provide robust depth inference at low power requirements in a wide variety of consumer and industrial applications. These cameras reconstruct a single depth frame from a given set of infrared (IR) frames captured over a very short exposure period. Operating in this mode the camera essentially forgets all information previously captured - and performs depth inference from scratch for every frame. We challenge this practice and propose using previously captured information when inferring depth. An inherent problem we have to address is camera motion over this longer period of collecting observations. We derive a probabilistic framework combining a simple but robust model of camera and object motion, together with an observation model. This combination allows us to integrate information over multiple frames while remaining robust to rapid changes. Operating the camera in this manner has implications in terms of both computational efficiency and how information should be captured. We address these two issues and demonstrate a realtime TOF system with robust temporal integration that improves depth accuracy over strong baseline methods including adaptive spatio-temporal filters.

count=1
* Unsupervised Human Action Detection by Action Matching
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/html/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/papers/Fernando_Unsupervised_Human_Action_CVPR_2017_paper.pdf)]
    * Title: Unsupervised Human Action Detection by Action Matching
    * Publisher: CVPR
    * Publication Date: `2017`
    * Authors: Basura Fernando, Sareh Shirazi, Stephen Gould
    * Abstract: We propose a new task of unsupervised action detection by action matching. Given two long videos, the objective is to temporally detect all pairs of matching video segments. A pair of video segments are matched if they share the same human action. The task is category independent---it does not matter what action is being performed---and no supervision is used to discover such video segments. Unsupervised action detection by action matching allows us to align videos in a meaningful manner. As such, it can be used to discover new action categories or as an action proposal technique within, say, an action detection pipeline. We solve this new task using an effective and efficient method. We use an unsupervised temporal encoding method and exploit the temporal consistency in human actions to obtain candidate action segments. We evaluate our method on this challenging task using three activity recognition benchmarks.

count=1
* WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.pdf)]
    * Title: WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Tatjana Chavdarova, Pierre Baqué, Stéphane Bouquet, Andrii Maksai, Cijo Jose, Timur Bagautdinov, Louis Lettry, Pascal Fua, Luc Van Gool, François Fleuret
    * Abstract: People detection methods are highly sensitive to occlusions between pedestrians, which are extremely frequent in many situations where cameras have to be mounted at a limited height. The reduction of camera prices allows for the generalization of static multi-camera set-ups. Using joint visual information from multiple synchronized cameras gives the opportunity to improve detection performance. In this paper, we present a new large-scale and high-resolution dataset. It has been captured with seven static cameras in a public open area, and unscripted dense groups of pedestrians standing and walking. Together with the camera frames, we provide an accurate joint (extrinsic and intrinsic) calibration, as well as 7 series of 400 annotated frames for detection at a rate of 2 frames per second. This results in over 40,000 bounding boxes delimiting every person present in the area of interest, for a total of more than 300 individuals. We provide a series of benchmark results using baseline algorithms published over the recent months for multi-view detection with deep neural networks, and trajectory estimation using a non-Markovian model.

count=1
* Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Qu_Unsupervised_Sparse_Dirichlet-Net_CVPR_2018_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qu_Unsupervised_Sparse_Dirichlet-Net_CVPR_2018_paper.pdf)]
    * Title: Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution
    * Publisher: CVPR
    * Publication Date: `2018`
    * Authors: Ying Qu, Hairong Qi, Chiman Kwan
    * Abstract: In many computer vision applications, obtaining images of high resolution in both the spatial and spectral domains are equally important. However, due to hardware limitations, one can only expect to acquire images of high resolution in either the spatial or spectral domains. This paper focuses on hyperspectral image super-resolution (HSI-SR), where a hyperspectral image (HSI) with low spatial resolution (LR) but high spectral resolution is fused with a multispectral image (MSI) with high spatial resolution (HR) but low spectral resolution to obtain HR HSI. Existing deep learning-based solutions are all supervised that would need a large training set and the availability of HR HSI, which is unrealistic. Here, we make the first attempt to solving the HSI-SR problem using an unsupervised encoder-decoder architecture that carries the following uniquenesses. First, it is composed of two encoder-decoder networks, coupled through a shared decoder, in order to preserve the rich spectral information from the HSI network. Second, the network encourages the representations from both modalities to follow a sparse Dirichlet distribution which naturally incorporates the two physical constraints of HSI and MSI. Third, the angular difference between representations are minimized in order to reduce the spectral distortion. We refer to the proposed architecture as unsupervised Sparse Dirichlet-Net, or uSDN. Extensive experimental results demonstrate the superior performance of uSDN as compared to the state-of-the-art.

count=1
* Speed Invariant Time Surface for Learning to Detect Corner Points With Event-Based Cameras
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Manderscheid_Speed_Invariant_Time_Surface_for_Learning_to_Detect_Corner_Points_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Manderscheid_Speed_Invariant_Time_Surface_for_Learning_to_Detect_Corner_Points_CVPR_2019_paper.pdf)]
    * Title: Speed Invariant Time Surface for Learning to Detect Corner Points With Event-Based Cameras
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Jacques Manderscheid,  Amos Sironi,  Nicolas Bourdis,  Davide Migliore,  Vincent Lepetit
    * Abstract: We propose a learning approach to corner detection for event-based cameras that is stable even under fast and abrupt motions. Event-based cameras offer high temporal resolution, power efficiency, and high dynamic range. However, the properties of event-based data are very different compared to standard intensity images, and simple extensions of corner detection methods designed for these images do not perform well on event-based data. We first introduce an efficient way to compute a time surface that is invariant to the speed of the objects. We then show that we can train a Random Forest to recognize events generated by a moving corner from our time surface. Random Forests are also extremely efficient, and therefore a good choice to deal with the high capture frequency of event-based cameras ---our implementation processes up to 1.6Mev/s on a single CPU. Thanks to our time surface formulation and this learning approach, our method is significantly more robust to abrupt changes of direction of the corners compared to previous ones. Our method also naturally assigns a confidence score for the corners, which can be useful for postprocessing. Moreover, we introduce a high-resolution dataset suitable for quantitative evaluation and comparison of corner detection methods for event-based cameras. We call our approach SILC, for Speed Invariant Learned Corners, and compare it to the state-of-the-art with extensive experiments, showing better performance.

count=1
* Moving Object Detection Under Discontinuous Change in Illumination Using Tensor Low-Rank and Invariant Sparse Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Shakeri_Moving_Object_Detection_Under_Discontinuous_Change_in_Illumination_Using_Tensor_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shakeri_Moving_Object_Detection_Under_Discontinuous_Change_in_Illumination_Using_Tensor_CVPR_2019_paper.pdf)]
    * Title: Moving Object Detection Under Discontinuous Change in Illumination Using Tensor Low-Rank and Invariant Sparse Decomposition
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Moein Shakeri,  Hong Zhang
    * Abstract: Although low-rank and sparse decomposition based methods have been successfully applied to the problem of moving object detection using structured sparsity-inducing norms, they are still vulnerable to significant illumination changes that arise in certain applications. We are interested in moving object detection in applications involving time-lapse image sequences for which current methods mistakenly group moving objects and illumination changes into foreground. Our method relies on the multilinear (tensor) data low-rank and sparse decomposition framework to address the weaknesses of existing methods. The key to our proposed method is to create first a set of prior maps that can characterize the changes in the image sequence due to illumination. We show that they can be detected by a k-support norm. To deal with concurrent, two types of changes, we employ two regularization terms, one for detecting moving objects and the other for accounting for illumination changes, in the tensor low-rank and sparse decomposition formulation. Through comprehensive experiments using challenging datasets, we show that our method demonstrates a remarkable ability to detect moving objects under discontinuous change in illumination, and outperforms the state-of-the-art solutions to this challenging problem.

count=1
* Privacy Protection in Street-View Panoramas Using Depth and Multi-View Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Uittenbogaard_Privacy_Protection_in_Street-View_Panoramas_Using_Depth_and_Multi-View_Imagery_CVPR_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Uittenbogaard_Privacy_Protection_in_Street-View_Panoramas_Using_Depth_and_Multi-View_Imagery_CVPR_2019_paper.pdf)]
    * Title: Privacy Protection in Street-View Panoramas Using Depth and Multi-View Imagery
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Ries Uittenbogaard,  Clint Sebastian,  Julien Vijverberg,  Bas Boom,  Dariu M. Gavrila,  Peter H.N. de With
    * Abstract: The current paradigm in privacy protection in street-view images is to detect and blur sensitive information. In this paper, we propose a framework that is an alternative to blurring, which automatically removes and inpaints moving objects (e.g. pedestrians, vehicles) in street-view imagery. We propose a novel moving object segmentation algorithm exploiting consistencies in depth across multiple street-view images that are later combined with the results of a segmentation network. The detected moving objects are removed and inpainted with information from other views, to obtain a realistic output image such that the moving object is not visible anymore. We evaluate our results on a dataset of 1000 images to obtain a peak noise-to-signal ratio (PSNR) and L 1 loss of 27.2 dB and 2.5%, respectively. To assess overall quality, we also report the results of a survey conducted on 35 professionals, asked to visually inspect the images whether object removal and inpainting had taken place. The inpainting dataset will be made publicly available for scientific benchmarking purposes at https://research.cyclomedia.com/.

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
* Learning Compositional Representation for 4D Captures With Neural ODE
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiang_Learning_Compositional_Representation_for_4D_Captures_With_Neural_ODE_CVPR_2021_paper.pdf)]
    * Title: Learning Compositional Representation for 4D Captures With Neural ODE
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Boyan Jiang, Yinda Zhang, Xingkui Wei, Xiangyang Xue, Yanwei Fu
    * Abstract: Learning based representation has become the key to the success of many computer vision systems. While many 3D representations have been proposed, it is still an unaddressed problem how to represent a dynamically changing 3D object. In this paper, we introduce a compositional representation for 4D captures, i.e. a deforming 3D object over a temporal span, that disentangles shape, initial state, and motion respectively. Each component is represented by a latent code via a trained encoder. To model the motion, a neural Ordinary Differential Equation (ODE) is trained to update the initial state conditioned on the learned motion code, and a decoder takes the shape code and the updated state code to reconstruct the 3D model at each time stamp. To this end, we propose an Identity Exchange Training (IET) strategy to encourage the network to learn effectively decoupling each component. Extensive experiments demonstrate that the proposed method outperforms existing state-of-the-art deep learning based methods on 4D reconstruction, and significantly improves on various tasks, including motion transfer and completion.

count=1
* An Efficient Approach for Anomaly Detection in Traffic Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AICity/html/Doshi_An_Efficient_Approach_for_Anomaly_Detection_in_Traffic_Videos_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/AICity/papers/Doshi_An_Efficient_Approach_for_Anomaly_Detection_in_Traffic_Videos_CVPRW_2021_paper.pdf)]
    * Title: An Efficient Approach for Anomaly Detection in Traffic Videos
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Keval Doshi, Yasin Yilmaz
    * Abstract: Due to its relevance in intelligent transportation systems, anomaly detection in traffic videos has recently received much interest. It remains a difficult problem due to a variety of factors influencing the video quality of a real-time traffic feed, such as temperature, perspective, lighting conditions, and so on. Even though state-of-the-art methods perform well on the available benchmark datasets, they need a large amount of external training data as well as substantial computational resources. In this paper, we propose an efficient approach for a video anomaly detection system which is capable of running at the edge devices, e.g., on a roadside camera. The proposed approach comprises a pre-processing module that detects changes in the scene and removes the corrupted frames, a two-stage background modelling module and a two-stage object detector. Finally, a backtracking anomaly detection algorithm computes a similarity statistic and decides on the onset time of the anomaly. We also propose a sequential change detection algorithm that can quickly adapt to a new scene and detect changes in the similarity statistic. Experimental results on the Track 4 test set of the 2021 AI City Challenge show the efficacy of the proposed framework as we achieve an F1-score of 0.9157 along with 8.4027 root mean square error (RMSE) and are ranked fourth in the competition.

count=1
* QFabric: Multi-Task Change Detection Dataset
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.pdf)]
    * Title: QFabric: Multi-Task Change Detection Dataset
    * Publisher: CVPR
    * Publication Date: `2021`
    * Authors: Sagar Verma, Akash Panigrahi, Siddharth Gupta
    * Abstract: Detecting change through multi-image, multi-date remote sensing is essential to developing an understanding of global conditions. Despite recent advancements in remote sensing realized through deep learning, novel methods for accurate multi-image change detection remain unrealized. Recently, several promising methods have been proposed to address this topic, but a paucity of publicly available data limits the methods that can be assessed. In particular, there exists limited work on categorizing the nature and status of change across an observation period. This paper introduces the first labeled dataset available for such a task. We present an open-source change detection dataset, termed QFabric, with 450,000 change polygons annotated across 504 locations in 100 different cities covering a wide range of geographies and urban fabrics. QFabric is a temporal multi-task dataset with 6 change types and 9 change status classes. The geography and environment metadata around each polygon provides context that can be leveraged to build robust deep neural networks. We apply multiple benchmarks on our dataset for change detection, change type and status classification tasks. Project page: https://sagarverma.github.io/qfabric

count=1
* ISDNet: Integrating Shallow and Deep Networks for Efficient Ultra-High Resolution Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_ISDNet_Integrating_Shallow_and_Deep_Networks_for_Efficient_Ultra-High_Resolution_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_ISDNet_Integrating_Shallow_and_Deep_Networks_for_Efficient_Ultra-High_Resolution_CVPR_2022_paper.pdf)]
    * Title: ISDNet: Integrating Shallow and Deep Networks for Efficient Ultra-High Resolution Segmentation
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Shaohua Guo, Liang Liu, Zhenye Gan, Yabiao Wang, Wuhao Zhang, Chengjie Wang, Guannan Jiang, Wei Zhang, Ran Yi, Lizhuang Ma, Ke Xu
    * Abstract: The huge burden of computation and memory are two obstacles in ultra-high resolution image segmentation. To tackle these issues, most of the previous works follow the global-local refinement pipeline, which pays more attention to the memory consumption but neglects the inference speed. In comparison to the pipeline that partitions the large image into small local regions, we focus on inferring the whole image directly. In this paper, we propose ISDNet, a novel ultra-high resolution segmentation framework that integrates the shallow and deep networks in a new manner, which significantly accelerates the inference speed while achieving accurate segmentation. To further exploit the relationship between the shallow and deep features, we propose a novel Relational-Aware feature Fusion module, which ensures high performance and robustness of our framework. Extensive experiments on Deepglobe, Inria Aerial, and Cityscapes datasets demonstrate our performance is consistently superior to state-of-the-arts. Specifically, it achieves 73.30 mIoU with a speed of 27.70 FPS on Deepglobe, which is more accurate and 172 x faster than the recent competitor. Code available at https://github.com/cedricgsh/ISDNet.

count=1
* DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.pdf)]
    * Title: DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Mathias Parger, Chengcheng Tang, Christopher D. Twigg, Cem Keskin, Robert Wang, Markus Steinberger
    * Abstract: Convolutional neural network inference on video data requires powerful hardware for real-time processing. Given the inherent coherence across consecutive frames, large parts of a video typically change little. By skipping identical image regions and truncating insignificant pixel updates, computational redundancy can in theory be reduced significantly. However, these theoretical savings have been difficult to translate into practice, as sparse updates hamper computational consistency and memory access coherence; which are key for efficiency on real hardware. With DeltaCNN, we present a sparse convolutional neural network framework that enables sparse frame-by-frame updates to accelerate video inference in practice. We provide sparse implementations for all typical CNN layers and propagate sparse feature updates end-to-end - without accumulating errors over time. DeltaCNN is applicable to all convolutional neural networks without retraining. To the best of our knowledge, we are the first to significantly outperform the dense reference, cuDNN, in practical settings, achieving speedups of up to 7x with only marginal differences in accuracy.

count=1
* Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.pdf)]
    * Title: Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Nicolae-Cătălin Ristea, Neelu Madan, Radu Tudor Ionescu, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B. Moeslund, Mubarak Shah
    * Abstract: Anomaly detection is commonly pursued as a one-class classification problem, where models can only learn from normal training samples, while being evaluated on both normal and abnormal test samples. Among the successful approaches for anomaly detection, a distinguished category of methods relies on predicting masked information (e.g. patches, future frames, etc.) and leveraging the reconstruction error with respect to the masked information as an abnormality score. Different from related methods, we propose to integrate the reconstruction-based functionality into a novel self-supervised predictive architectural building block. The proposed self-supervised block is generic and can easily be incorporated into various state-of-the-art anomaly detection methods. Our block starts with a convolutional layer with dilated filters, where the center area of the receptive field is masked. The resulting activation maps are passed through a channel attention module. Our block is equipped with a loss that minimizes the reconstruction error with respect to the masked area in the receptive field. We demonstrate the generality of our block by integrating it into several state-of-the-art frameworks for anomaly detection on image and video, providing empirical evidence that shows considerable performance improvements on MVTec AD, Avenue, and ShanghaiTech. We release our code as open source at: https://github.com/ristea/sspcab.

count=1
* Ambiguous Medical Image Segmentation Using Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Rahman_Ambiguous_Medical_Image_Segmentation_Using_Diffusion_Models_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Rahman_Ambiguous_Medical_Image_Segmentation_Using_Diffusion_Models_CVPR_2023_paper.pdf)]
    * Title: Ambiguous Medical Image Segmentation Using Diffusion Models
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Aimon Rahman, Jeya Maria Jose Valanarasu, Ilker Hacihaliloglu, Vishal M. Patel
    * Abstract: Collective insights from a group of experts have always proven to outperform an individual's best diagnostic for clinical tasks. For the task of medical image segmentation, existing research on AI-based alternatives focuses more on developing models that can imitate the best individual rather than harnessing the power of expert groups. In this paper, we introduce a single diffusion model-based approach that produces multiple plausible outputs by learning a distribution over group insights. Our proposed model generates a distribution of segmentation masks by leveraging the inherent stochastic sampling process of diffusion using only minimal additional learning. We demonstrate on three different medical image modalities- CT, ultrasound, and MRI that our model is capable of producing several possible variants while capturing the frequencies of their occurrences. Comprehensive results show that our proposed approach outperforms existing state-of-the-art ambiguous segmentation networks in terms of accuracy while preserving naturally occurring variation. We also propose a new metric to evaluate the diversity as well as the accuracy of segmentation predictions that aligns with the interest of clinical practice of collective insights. Implementation code will be released publicly after the review process.

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
* Multi-Modal Multi-Objective Contrastive Learning for Sentinel-1/2 Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.pdf)]
    * Title: Multi-Modal Multi-Objective Contrastive Learning for Sentinel-1/2 Imagery
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Jonathan Prexl, Michael Schmitt
    * Abstract: The field of spaceborne Earth observation offers, due to constant monitoring of the Earth's surface, a huge amount of unlabeled data. At the same time, for many applications, there still exists a shortage of high-quality labelled datasets. This is one of the major bottlenecks for progress in developing globally applicable deep learning models for analysing the dynamics of our planet from space. In recent years self-supervised representation learning revealed itself to state a very powerful way of incorporating unlabeled data into the typical supervised machine learning workflow. Still, many questions on how to adapt commonly used approaches to domain-specific properties of Earth observation data remain. In this work, we introduce and study approaches to incorporate multi-modal Earth observation data into a contrastive self-supervised learning framework by forcing inter- and intra-modality similarity in the loss function. Further, we introduce a batch-sampling strategy that leverages the geo-coding of the imagery in order to obtain harder negative pairs for the contrastive learning problem. We show through extensive experiments that various domain-specific downstream problems are benefitting from the above-mentioned contributions.

count=1
* Image Vegetation Index Through a Cycle Generative Adversarial Network
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/html/PBVS/Suarez_Image_Vegetation_Index_Through_a_Cycle_Generative_Adversarial_Network_CVPRW_2019_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Suarez_Image_Vegetation_Index_Through_a_Cycle_Generative_Adversarial_Network_CVPRW_2019_paper.pdf)]
    * Title: Image Vegetation Index Through a Cycle Generative Adversarial Network
    * Publisher: CVPR
    * Publication Date: `2019`
    * Authors: Patricia L. Suarez,  Angel D. Sappa,  Boris X. Vintimilla,  Riad I. Hammoud
    * Abstract: This paper proposes a novel approach to estimate the Normalized Difference Vegetation Index (NDVI) just from an RGB image. The NDVI values are obtained by using images from the visible spectral band together with a synthetic near infrared image obtained by a cycled GAN. The cycled GAN network is able to obtain a NIR image from a given gray scale image. It is trained by using unpaired set of gray scale and NIR images by using a U-net architecture and a multiple loss function (gray scale images are obtained from the provided RGB images). Then, the NIR image estimated with the proposed cycle generative adversarial network is used to compute the NDVI index. Experimental results are provided showing the validity of the proposed approach. Additionally, comparisons with previous approaches are also provided.

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
* Intelligent Scene Caching to Improve Accuracy for Energy-Constrained Embedded Vision
    [[abs-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Simpson_Intelligent_Scene_Caching_to_Improve_Accuracy_for_Energy-Constrained_Embedded_Vision_CVPRW_2020_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Simpson_Intelligent_Scene_Caching_to_Improve_Accuracy_for_Energy-Constrained_Embedded_Vision_CVPRW_2020_paper.pdf)]
    * Title: Intelligent Scene Caching to Improve Accuracy for Energy-Constrained Embedded Vision
    * Publisher: CVPR
    * Publication Date: `2020`
    * Authors: Benjamin Simpson, Ekdeep Lubana, Yuchen Liu, Robert Dick
    * Abstract: We describe an efficient method of improving the performance of vision algorithms operating on video streams by reducing the amount of data captured and transferred from image sensors to analysis servers in a data-aware manner. The key concept is to combine guided, highly heterogeneous sampling with an intelligent Scene Cache. This enables the system to adapt to spatial and temporal patterns in the scene, thus reducing redundant data capture and processing. A software prototype of our framework running on a general-purpose embedded processor enables superior object detection accuracy (by 56%) at similar energy consumption (slight improvement of 4%) compared to an H.264 hardware accelerator.

count=1
* Flexible Background Subtraction With Self-Balanced Local Sensitivity
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf)]
    * Title: Flexible Background Subtraction With Self-Balanced Local Sensitivity
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, Robert Bergevin
    * Abstract: Most background subtraction approaches offer decent results in baseline scenarios, but adaptive and flexible solutions are still uncommon as many require scenario-specific parameter tuning to achieve optimal performance. In this paper, we introduce a new strategy to tackle this problem that focuses on balancing the inner workings of a non-parametric model based on pixel-level feedback loops. Pixels are modeled using a spatiotemporal feature descriptor for increased sensitivity. Using the video sequences and ground truth annotations of the 2012 and 2014 CVPR Change Detection Workshops, we demonstrate that our approach outperforms all previously ranked methods in the original dataset while achieving good results in the most recent one.

count=1
* Static and Moving Object Detection Using Flux Tensor with Split Gaussian Models
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/html/Wang_Static_and_Moving_2014_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Wang_Static_and_Moving_2014_CVPR_paper.pdf)]
    * Title: Static and Moving Object Detection Using Flux Tensor with Split Gaussian Models
    * Publisher: CVPR
    * Publication Date: `2014`
    * Authors: Rui Wang, Filiz Bunyak, Guna Seetharaman, Kannappan Palaniappan
    * Abstract: In this paper, we present a moving object detection system named Flux Tensor with Split Gaussian models (FTSG) that exploits the benefits of fusing a motion computation method based on spatio-temporal tensor formulation, a novel foreground and background modeling scheme, and a multi-cue appearance comparison. This hybrid system can handle challenges such as shadows, illumination changes, dynamic background, stopped and removed objects. Extensive testing performed on the CVPR 2014 Change Detection benchmark dataset shows that FTSG outperforms state-ofthe-art methods.

count=1
* A Model-Based Approach to Finding Tracks in SAR CCD Images
    [[abs-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/html/Quach_A_Model-Based_Approach_2015_CVPR_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W05/papers/Quach_A_Model-Based_Approach_2015_CVPR_paper.pdf)]
    * Title: A Model-Based Approach to Finding Tracks in SAR CCD Images
    * Publisher: CVPR
    * Publication Date: `2015`
    * Authors: Tu-Thach Quach, Rebecca Malinas, Mark W. Koch
    * Abstract: Combining multiple synthetic aperture radar (SAR) images taken at different times of the same scene produces coherent change detection (CCD) images that can detect small surface changes such as tire tracks. The resulting CCD images can be used in an automated approach to identify and label tracks. Existing techniques have limited success due to the noisy nature of these CCD images. In particular, existing techniques require some user cues and can only trace a single track. This paper presents an approach to automatically identify and label multiple tracks in CCD images. We use an explicit objective function that utilizes the Bayesian information criterion to find the simplest set of curves that explains the observed data. Experimental results show that it is capable of identifying tracks under various scenes and can correctly declare when no tracks are present.

count=1
* Dynamic Probabilistic Volumetric Models
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Ulusoy_Dynamic_Probabilistic_Volumetric_2013_ICCV_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2013/papers/Ulusoy_Dynamic_Probabilistic_Volumetric_2013_ICCV_paper.pdf)]
    * Title: Dynamic Probabilistic Volumetric Models
    * Publisher: ICCV
    * Publication Date: `2013`
    * Authors: Ali Osman Ulusoy, Octavian Biris, Joseph L. Mundy
    * Abstract: This paper presents a probabilistic volumetric framework for image based modeling of general dynamic 3-d scenes. The framework is targeted towards high quality modeling of complex scenes evolving over thousands of frames. Extensive storage and computational resources are required in processing large scale space-time (4-d) data. Existing methods typically store separate 3-d models at each time step and do not address such limitations. A novel 4-d representation is proposed that adaptively subdivides in space and time to explain the appearance of 3-d dynamic surfaces. This representation is shown to achieve compression of 4-d data and provide efficient spatio-temporal processing. The advances of the proposed framework is demonstrated on standard datasets using free-viewpoint video and 3-d tracking applications.

count=1
* A Scalable Architecture for Operational FMV Exploitation
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w1/html/Thissell_A_Scalable_Architecture_ICCV_2015_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_iccv_2015_workshops/w1/papers/Thissell_A_Scalable_Architecture_ICCV_2015_paper.pdf)]
    * Title: A Scalable Architecture for Operational FMV Exploitation
    * Publisher: ICCV
    * Publication Date: `2015`
    * Authors: William R. Thissell, Robert Czajkowski, Francis Schrenk, Timothy Selway, Anthony J. Ries, Shamoli Patel, Patricia L. McDermott, Rod Moten, Ron Rudnicki, Guna Seetharaman, Ilker Ersoy, Kannappan Palaniappan
    * Abstract: A scalable open systems and standards derived software ecosystem is described for computer vision analytics (CVA) assisted exploitation of full motion video (FMV). The ecosystem, referred to as the Advanced Video Activity Analytics (AVAA), has two instantiations, one for size, weight, and power (SWAP) constrained conditions, and the other for large to massive cloud based configurations. The architecture is designed to meet operational analyst requirements to increase their productivity and accuracy for exploiting FMV using local cluster or scalable cloud-based computing resources. CVAs are encapsulated within a software plug-in architecture and FMV processing pipelines are constructed by combining these plug-ins to accomplish analytical tasks and manage provenance of processing history. An example pipeline for real-time motion detection and moving object characterization using the flux tensor approach is presented. An example video ingest experiment is described. Quantitative and qualitative methods for human factors engineering (HFE) assessment to evaluate cognitive loads for alternative work flow design choices are discussed. This HFE process is used for validating that an AVAA system instantiation with candidate workflow pipelines meets CVA assisted FMV exploitation operational goals for specific analyst workflows. AVAA offers a new framework for video understanding at scale for large enterprise applications in the government and commercial sectors.

count=1
* Moving Object Detection in Time-Lapse or Motion Trigger Image Sequences Using Low-Rank and Invariant Sparse Decomposition
    [[abs-CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Shakeri_Moving_Object_Detection_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Shakeri_Moving_Object_Detection_ICCV_2017_paper.pdf)]
    * Title: Moving Object Detection in Time-Lapse or Motion Trigger Image Sequences Using Low-Rank and Invariant Sparse Decomposition
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Moein Shakeri, Hong Zhang
    * Abstract: Low-rank and sparse representation based methods have attracted wide attention in background subtraction and moving object detection, where moving objects in the scene are modeled as pixel-wise sparse outliers. Since in real scenarios moving objects are also structurally sparse, recently researchers have attempted to extract moving objects using structured sparse outliers. Although existing methods with structured sparsity-inducing norms produce promising results, they are still vulnerable to various illumination changes that frequently occur in real environments, specifically for time-lapse image sequences where assumptions about sparsity between images such as group sparsity are not valid. In this paper, we first introduce a prior map obtained by illumination invariant representation of images. Next, we propose a low-rank and invariant sparse decomposition using the prior map to detect moving objects under significant illumination changes. Experiments on challenging benchmark datasets demonstrate the superior performance of our proposed method under complex illumination changes.

count=1
* Panning and Jitter Invariant Incremental Principal Component Pursuit for Video Background Modeling
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w25/html/Chau_Panning_and_Jitter_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w25/Chau_Panning_and_Jitter_ICCV_2017_paper.pdf)]
    * Title: Panning and Jitter Invariant Incremental Principal Component Pursuit for Video Background Modeling
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Gustavo Chau, Paul Rodriguez
    * Abstract: Video background modeling is an important preprocessing stage for various applications and principal component pursuit (PCP) is among the state-of-the-art algorithms for this task. One of the main drawbacks of PCP is its sensitivity to jitter and camera movement. This problem has only been partially solved by a few methods devised for jitter or small transformations. However, such methods cannot handle the case of moving or panning cameras. We present a novel, fully incremental PCP algorithm, named incPCP-PTI, that is able to cope with panning scenarios and jitter by continuously aligning the low-rank component to the current reference frame of the camera. To the best of our knowledge, incPCP-PTI is the first low rank plus additive incremental matrix method capable of handling these scenarios. Results on synthetic videos and CDNET2014 videos show that incPCP-PTI is able to maintain a good performance in the detection of moving objects even when panning and jitter are present in a video

count=1
* Compressed Singular Value Decomposition for Image and Video Processing
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w25/html/Erichson_Compressed_Singular_Value_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w25/Erichson_Compressed_Singular_Value_ICCV_2017_paper.pdf)]
    * Title: Compressed Singular Value Decomposition for Image and Video Processing
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: N. Benjamin Erichson, Steven L. Brunton, J. Nathan Kutz
    * Abstract: We demonstrate a heuristic algorithm to compute the approximate low-rank singular value decomposition. The algorithm is inspired by ideas from compressed sensing and, in particular, is suitable for image and video processing applications. Specifically, our compressed singular value decomposition (cSVD) algorithm employs aggressive random test matrices to efficiently sketch the row space of the input matrix. The resulting compressed representation of the data enables the computation of an accurate approximation of the dominant high-dimensional left and right singular vectors. We benchmark cSVD against the current state-of-the-art randomized SVD and show a performance boost while attaining near similar relative errors. The cSVD is simple to implement as well as embarrassingly parallel, i.e, ideally suited for GPU computations and mobile platforms.

count=1
* The Visual Object Tracking VOT2017 Challenge Results
    [[abs-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w28/html/Kristan_The_Visual_Object_ICCV_2017_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Kristan_The_Visual_Object_ICCV_2017_paper.pdf)]
    * Title: The Visual Object Tracking VOT2017 Challenge Results
    * Publisher: ICCV
    * Publication Date: `2017`
    * Authors: Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pflugfelder, Luka Cehovin Zajc, Tomas Vojir, Gustav Hager, Alan Lukezic, Abdelrahman Eldesokey, Gustavo Fernandez
    * Abstract: The Visual Object Tracking challenge VOT2017 is the fifth annual tracker benchmarking activity organized by the VOT initiative. Results of 51 trackers are presented; many are state-of-the-art published at major computer vision conferences or journals in recent years. The evaluation included the standard VOT and other popular methodologies and a new "real-time" experiment simulating a situation where a tracker processes images as if provided by a continuously running sensor. Performance of the tested trackers typically by far exceeds standard baselines. The source code for most of the trackers is publicly available from the VOT page. The VOT2017 goes beyond its predecessors by (i) improving the VOT public dataset and introducing a separate VOT2017 sequestered dataset, (ii) introducing a real-time tracking experiment and (iii) releasing a redesigned toolkit that supports complex experiments. The dataset, the evaluation kit and the results are publicly available at the challenge w ....

count=1
* Sparse-to-Dense Feature Matching: Intra and Inter Domain Cross-Modal Learning in Domain Adaptation for 3D Semantic Segmentation
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Sparse-to-Dense_Feature_Matching_Intra_and_Inter_Domain_Cross-Modal_Learning_in_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Peng_Sparse-to-Dense_Feature_Matching_Intra_and_Inter_Domain_Cross-Modal_Learning_in_ICCV_2021_paper.pdf)]
    * Title: Sparse-to-Dense Feature Matching: Intra and Inter Domain Cross-Modal Learning in Domain Adaptation for 3D Semantic Segmentation
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Duo Peng, Yinjie Lei, Wen Li, Pingping Zhang, Yulan Guo
    * Abstract: Domain adaptation is critical for success when confronting with the lack of annotations in a new domain. As the huge time consumption of labeling process on 3D point cloud, domain adaptation for 3D semantic segmentation is of great expectation. With the rise of multi-modal datasets, large amount of 2D images are accessible besides 3D point clouds. In light of this, we propose to further leverage 2D data for 3D domain adaptation by intra and inter domain cross modal learning. As for intra-domain cross modal learning, most existing works sample the dense 2D pixel-wise features into the same size with sparse 3D point-wise features, resulting in the abandon of numerous useful 2D features. To address this problem, we propose Dynamic sparse-to-dense Cross Modal Learning (DsCML) to increase the sufficiency of multi-modality information interaction for domain adaptation. For inter-domain cross modal learning, we further advance Cross Modal Adversarial Learning (CMAL) on 2D and 3D data which contains different semantic content aiming to promote high-level modal complementarity. We evaluate our model under various multi-modality domain adaptation settings including day-to-night, country-to-country and dataset-to-dataset, brings large improvements over both uni-modal and multi-modal domain adaptation methods on all settings.

count=1
* Describing and Localizing Multiple Changes With Transformers
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Qiu_Describing_and_Localizing_Multiple_Changes_With_Transformers_ICCV_2021_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiu_Describing_and_Localizing_Multiple_Changes_With_Transformers_ICCV_2021_paper.pdf)]
    * Title: Describing and Localizing Multiple Changes With Transformers
    * Publisher: ICCV
    * Publication Date: `2021`
    * Authors: Yue Qiu, Shintaro Yamamoto, Kodai Nakashima, Ryota Suzuki, Kenji Iwata, Hirokatsu Kataoka, Yutaka Satoh
    * Abstract: Existing change captioning studies have mainly focused on a single change. However, detecting and describing multiple changed parts in image pairs is essential for enhancing adaptability to complex scenarios. We solve the above issues from three aspects: (i) We propose a simulation-based multi-change captioning dataset; (ii) We benchmark existing state-of-the-art methods of single change captioning on multi-change captioning; (iii) We further propose Multi-Change Captioning transformers (MCCFormers) that identify change regions by densely correlating different regions in image pairs and dynamically determines the related change regions with words in sentences. The proposed method obtained the highest scores on four conventional change captioning evaluation metrics for multi-change captioning. Additionally, our proposed method can separate attention maps for each change and performs well with respect to change localization. Moreover, the proposed framework outperformed the previous state-of-the-art methods on an existing change captioning benchmark, CLEVR-Change, by a large margin (+6.1 on BLEU-4 and +9.7 on CIDEr scores), indicating its general ability in change captioning tasks. The code and dataset are available at the project page.

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
* Self-Supervised Object Detection from Egocentric Videos
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Akiva_Self-Supervised_Object_Detection_from_Egocentric_Videos_ICCV_2023_paper.pdf)]
    * Title: Self-Supervised Object Detection from Egocentric Videos
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Peri Akiva, Jing Huang, Kevin J Liang, Rama Kovvuri, Xingyu Chen, Matt Feiszli, Kristin Dana, Tal Hassner
    * Abstract: Understanding the visual world from the perspective of humans (egocentric) has been a long-standing challenge in computer vision. Egocentric videos exhibit high scene complexity and irregular motion flows compared to typical video understanding tasks. With the egocentric domain in mind, we address the problem of self-supervised, class-agnostic object detection, which aims to locate all objects in a given view, regardless of category, without any annotations or pre-training weights. Our method, self-supervised object Detection from Egocentric VIdeos (DEVI), generalizes appearance-based methods to learn features that are category-specific and invariant to viewing angles and illumination conditions from highly ambiguous environments in an end-to-end manner. Our approach leverages typical human behavior and its egocentric perception to sample diverse views of the same objects for our multi-view and scale-regression loss functions. With our learned cluster residual module, we are able to effectively describe multi-category patches for better complex scene understanding. DEVI provides a boost in performance on recent egocentric datasets, with performance gains up to 4.11% AP50, 0.11% AR1, 1.32% AR10, and 5.03% AR100, while significantly reducing model complexity. We also demonstrate competitive performance on out-of-domain datasets without additional training or fine-tuning.

count=1
* Active Learning for Improved Semi-Supervised Semantic Segmentation in Satellite Images
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2022/html/Desai_Active_Learning_for_Improved_Semi-Supervised_Semantic_Segmentation_in_Satellite_Images_WACV_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Desai_Active_Learning_for_Improved_Semi-Supervised_Semantic_Segmentation_in_Satellite_Images_WACV_2022_paper.pdf)]
    * Title: Active Learning for Improved Semi-Supervised Semantic Segmentation in Satellite Images
    * Publisher: WACV
    * Publication Date: `2022`
    * Authors: Shasvat Desai, Debasmita Ghose
    * Abstract: Remote sensing data is crucial for applications ranging from monitoring forest fires and deforestation to tracking urbanization. Most of these tasks require dense pixel-level annotations for the model to parse visual information from limited labeled data available for these satellite images. Due to the dearth of high-quality labeled training data in this domain, there is a need to focus on semi-supervised techniques. These techniques generate pseudo-labels from a small set of labeled examples which are used to augment the labeled training set. This makes it necessary to have a highly representative and diverse labeled training set. Therefore, we propose to use an active learning-based sampling strategy to select a highly representative set of labeled training data. We demonstrate our proposed method's effectiveness on two existing semantic segmentation datasets containing satellite images: UC Merced Land Use Classification Dataset and DeepGlobe Land Cover Classification Dataset. We report a 27% improvement in mIoU with as little as 2% labeled data using active learning sampling strategies over randomly sampling the small set of labeled training data.

count=1
* Multi-View 3D Object Reconstruction and Uncertainty Modelling With Neural Shape Prior
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Liao_Multi-View_3D_Object_Reconstruction_and_Uncertainty_Modelling_With_Neural_Shape_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Liao_Multi-View_3D_Object_Reconstruction_and_Uncertainty_Modelling_With_Neural_Shape_WACV_2024_paper.pdf)]
    * Title: Multi-View 3D Object Reconstruction and Uncertainty Modelling With Neural Shape Prior
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Ziwei Liao, Steven L. Waslander
    * Abstract: 3D object reconstruction is important for semantic scene understanding. It is challenging to reconstruct detailed 3D shapes from monocular images directly due to a lack of depth information, occlusion and noise. Most current methods generate deterministic object models without any awareness of the uncertainty of the reconstruction. We tackle this problem by leveraging a neural object representation which learns an object shape distribution from large dataset of 3d object models and maps it into a latent space. We propose a method to model uncertainty as part of the representation and define an uncertainty-aware encoder which generates latent codes with uncertainty directly from individual input images. Further, we propose a method to propagate the uncertainty in the latent code to SDF values and generate a 3d object mesh with local uncertainty for each mesh component. Finally, we propose an incremental fusion method under a Bayesian framework to fuse the latent codes from multi-view observations. We evaluate the system in both synthetic and real datasets to demonstrate the effectiveness of uncertainty-based fusion to improve 3D object reconstruction accuracy.

count=1
* Limits on Testing Structural Changes in Ising Models
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2020/hash/70431e77d378d760c3c5456519f06efe-Abstract.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2020/file/70431e77d378d760c3c5456519f06efe-Paper.pdf)]
    * Title: Limits on Testing Structural Changes in Ising Models
    * Publisher: NeurIPS
    * Publication Date: `2020`
    * Authors: Aditya Gangrade, Bobak Nazer, Venkatesh Saligrama
    * Abstract: We present novel information-theoretic limits on detecting sparse changes in Isingmodels, a problem that arises in many applications where network changes canoccur due to some external stimuli. We show that the sample complexity fordetecting sparse changes, in a minimax sense, is no better than learning the entiremodel even in settings with local sparsity. This is a surprising fact in light of priorwork rooted in sparse recovery methods, which suggest that sample complexityin this context scales only with the number of network changes. To shed light onwhen change detection is easier than structured learning, we consider testing ofedge deletion in forest-structured graphs, and high-temperature ferromagnets ascase studies. We show for these that testing of small changes is similarly hard, buttesting oflargechanges is well-separated from structure learning. These resultsimply that testing of graphical models may not be amenable to concepts such asrestricted strong convexity leveraged for sparsity pattern recovery, and algorithmdevelopment instead should be directed towards detection of large changes.

count=1
* Learning the Structure of Large Networked Systems Obeying Conservation Laws
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2022/hash/5e0347e19c51cfd0f6fe52f371004dfc-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2022/file/5e0347e19c51cfd0f6fe52f371004dfc-Paper-Conference.pdf)]
    * Title: Learning the Structure of Large Networked Systems Obeying Conservation Laws
    * Publisher: NeurIPS
    * Publication Date: `2022`
    * Authors: Anirudh Rayas, Rajasekhar Anguluri, Gautam Dasarathy
    * Abstract: Many networked systems such as electric networks, the brain, and social networks of opinion dynamics are known to obey conservation laws. Examples of this phenomenon include the Kirchoff laws in electric networks and opinion consensus in social networks. Conservation laws in networked systems are modeled as balance equations of the form $X = B^\ast Y$, where the sparsity pattern of $B^\ast \in \mathbb{R}^{p\times p}$ captures the connectivity of the network on $p$ nodes, and $Y, X \in \mathbb{R}^p$ are vectors of ''potentials'' and ''injected flows'' at the nodes respectively. The node potentials $Y$ cause flows across edges which aim to balance out the potential difference, and the flows $X$ injected at the nodes are extraneous to the network dynamics. In several practical systems, the network structure is often unknown and needs to be estimated from data to facilitate modeling, management, and control. To this end, one has access to samples of the node potentials $Y$, but only the statistics of the node injections $X$. Motivated by this important problem, we study the estimation of the sparsity structure of the matrix $B^\ast$ from $n$ samples of $Y$ under the assumption that the node injections $X$ follow a Gaussian distribution with a known covariance $\Sigma_X$. We propose a new $\ell_{1}$-regularized maximum likelihood estimator for tackling this problem in the high-dimensional regime where the size of the network may be vastly larger than the number of samples $n$. We show that this optimization problem is convex in the objective and admits a unique solution. Under a new mutual incoherence condition, we establish sufficient conditions on the triple $(n,p,d)$ for which exact sparsity recovery of $B^\ast$ is possible with high probability; $d$ is the degree of the underlying graph. We also establish guarantees for the recovery of $B^\ast$ in the element-wise maximum, Frobenius, and operator norms. Finally, we complement these theoretical results with experimental validation of the performance of the proposed estimator on synthetic and real-world data.

count=1
* Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/6b8c6f846c3575e1d1ad496abea28826-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/6b8c6f846c3575e1d1ad496abea28826-Paper-Conference.pdf)]
    * Title: Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Congyue Deng, Jiahui Lei, William B Shen, Kostas Daniilidis, Leonidas J. Guibas
    * Abstract: Equivariance has gained strong interest as a desirable network property that inherently ensures robust generalization. However, when dealing with complex systems such as articulated objects or multi-object scenes, effectively capturing inter-part transformations poses a challenge, as it becomes entangled with the overall structure and local transformations. The interdependence of part assignment and per-part group action necessitates a novel equivariance formulation that allows for their co-evolution. In this paper, we present Banana, a Banach fixed-point network for equivariant segmentation with inter-part equivariance by construction. Our key insight is to iteratively solve a fixed-point problem, where point-part assignment labels and per-part SE(3)-equivariance co-evolve simultaneously. We provide theoretical derivations of both per-step equivariance and global convergence, which induces an equivariant final convergent state. Our formulation naturally provides a strict definition of inter-part equivariance that generalizes to unseen inter-part configurations. Through experiments conducted on both articulated objects and multi-object scans, we demonstrate the efficacy of our approach in achieving strong generalization under inter-part transformations, even when confronted with substantial changes in pointcloud geometry and topology.

count=1
* CityRefer: Geography-aware 3D Visual Grounding Dataset on  City-scale Point Cloud Data
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/f4cef76305dcad4efd3537da087ff520-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/f4cef76305dcad4efd3537da087ff520-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: CityRefer: Geography-aware 3D Visual Grounding Dataset on  City-scale Point Cloud Data
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Taiki Miyanishi, Fumiya Kitamori, Shuhei Kurita, Jungdae Lee, Motoaki Kawanabe, Nakamasa Inoue
    * Abstract: City-scale 3D point cloud is a promising way to express detailed and complicated outdoor structures. It encompasses both the appearance and geometry features of segmented city components, including cars, streets, and buildings that can be utilized for attractive applications such as user-interactive navigation of autonomous vehicles and drones. However, compared to the extensive text annotations available for images and indoor scenes, the scarcity of text annotations for outdoor scenes poses a significant challenge for achieving these applications. To tackle this problem, we introduce the CityRefer dataset for city-level visual grounding. The dataset consists of 35k natural language descriptions of 3D objects appearing in SensatUrban city scenes and 5k landmarks labels synchronizing with OpenStreetMap. To ensure the quality and accuracy of the dataset, all descriptions and labels in the CityRefer dataset are manually verified. We also have developed a baseline system that can learn encoded language descriptions, 3D object instances, and geographical information about the city's landmarks to perform visual grounding on the CityRefer dataset. To the best of our knowledge, the CityRefer dataset is the largest city-level visual grounding dataset for localizing specific 3D objects.

