count=19
* Towards Geospatial Foundation Models via Continual Pretraining
    [[abs-CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.pdf)]
    * Title: Towards Geospatial Foundation Models via Continual Pretraining
    * Publisher: ICCV
    * Publication Date: `2023`
    * Authors: Matías Mendieta, Boran Han, Xingjian Shi, Yi Zhu, Chen Chen
    * Abstract: Geospatial technologies are becoming increasingly essential in our world for a wide range of applications, including agriculture, urban planning, and disaster response. To help improve the applicability and performance of deep learning models on these geospatial tasks, various works have begun investigating foundation models for this domain. Researchers have explored two prominent approaches for introducing such models in geospatial applications, but both have drawbacks in terms of limited performance benefit or prohibitive training cost. Therefore, in this work, we propose a novel paradigm for building highly effective geospatial foundation models with minimal resource cost and carbon impact. We first construct a compact yet diverse dataset from multiple sources to promote feature diversity, which we term GeoPile. Then, we investigate the potential of continual pretraining from large-scale ImageNet-22k models and propose a multi-objective continual pretraining paradigm, which leverages the strong representations of ImageNet while simultaneously providing the freedom to learn valuable in-domain features. Our approach outperforms previous state-of-the-art geospatial pretraining methods in an extensive evaluation on seven downstream datasets covering various tasks such as change detection, classification, multi-label classification, semantic segmentation, and super-resolution. Code is available at https://github.com/mmendiet/GFM.

count=17
* Long-Tailed Anomaly Detection with Learnable Class Names
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ho_Long-Tailed_Anomaly_Detection_with_Learnable_Class_Names_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ho_Long-Tailed_Anomaly_Detection_with_Learnable_Class_Names_CVPR_2024_paper.pdf)]
    * Title: Long-Tailed Anomaly Detection with Learnable Class Names
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chih-Hui Ho, Kuan-Chuan Peng, Nuno Vasconcelos
    * Abstract: Anomaly detection (AD) aims to identify defective images and localize their defects (if any). Ideally AD models should be able to detect defects over many image classes; without relying on hard-coded class names that can be uninformative or inconsistent across datasets; learn without anomaly supervision; and be robust to the long-tailed distributions of real-world applications. To address these challenges we formulate the problem of long-tailed AD by introducing several datasets with different levels of class imbalance and metrics for performance evaluation. We then propose a novel method LTAD to detect defects from multiple and long-tailed classes without relying on dataset class names. LTAD combines AD by reconstruction and semantic AD modules. AD by reconstruction is implemented with a transformer-based reconstruction module. Semantic AD is implemented with a binary classifier which relies on learned pseudo class names and a pretrained foundation model. These modules are learned over two phases. Phase 1 learns the pseudo-class names and a variational autoencoder (VAE) for feature synthesis that augments the training data to combat long-tails. Phase 2 then learns the parameters of the reconstruction and classification modules of LTAD. Extensive experiments using the proposed long-tailed datasets show that LTAD substantially outperforms the state-of-the-art methods for most forms of dataset imbalance. The long-tailed dataset split is available at https://zenodo.org/records/10854201

count=16
* SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.pdf)]
    * Title: SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xin Guo, Jiangwei Lao, Bo Dang, Yingying Zhang, Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang, Kang Wu, Dingxiang Hu, Huimei He, Jian Wang, Jingdong Chen, Ming Yang, Yongjun Zhang, Yansheng Li
    * Abstract: Prior studies on Remote Sensing Foundation Model (RSFM) reveal immense potential towards a generic model for Earth Observation. Nevertheless these works primarily focus on a single modality without temporal and geo-context modeling hampering their capabilities for diverse tasks. In this study we present SkySense a generic billion-scale model pre-trained on a curated multi-modal Remote Sensing Imagery (RSI) dataset with 21.5 million temporal sequences. SkySense incorporates a factorized multi-modal spatiotemporal encoder taking temporal sequences of optical and Synthetic Aperture Radar (SAR) data as input. This encoder is pre-trained by our proposed Multi-Granularity Contrastive Learning to learn representations across different modal and spatial granularities. To further enhance the RSI representations by the geo-context clue we introduce Geo-Context Prototype Learning to learn region-aware prototypes upon RSI's multi-modal spatiotemporal features. To our best knowledge SkySense is the largest Multi-Modal RSFM to date whose modules can be flexibly combined or used individually to accommodate various tasks. It demonstrates remarkable generalization capabilities on a thorough evaluation encompassing 16 datasets over 7 tasks from single- to multi-modal static to temporal and classification to localization. SkySense surpasses 18 recent RSFMs in all test scenarios. Specifically it outperforms the latest models such as GFM SatLas and Scale-MAE by a large margin i.e. 2.76% 3.67% and 3.61% on average respectively. We will release the pre-trained weights to facilitate future research and Earth Observation applications.

count=15
* S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.pdf)]
    * Title: S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Xuyang Li, Danfeng Hong, Jocelyn Chanussot
    * Abstract: In the expansive domain of computer vision a myriad of pre-trained models are at our disposal. However most of these models are designed for natural RGB images and prove inadequate for spectral remote sensing (RS) images. Spectral RS images have two main traits: (1) multiple bands capturing diverse feature information (2) spatial alignment and consistent spectral sequencing within the spatial-spectral dimension. In this paper we introduce Spatial-SpectralMAE (S2MAE) a specialized pre-trained architecture for spectral RS imagery. S2MAE employs a 3D transformer for masked autoencoder modeling integrating learnable spectral-spatial embeddings with a 90% masking ratio. The model efficiently captures local spectral consistency and spatial invariance using compact cube tokens demonstrating versatility to diverse input characteristics. This adaptability facilitates progressive pretraining on extensive spectral datasets. The effectiveness of S2MAE is validated through continuous pretraining on two sizable datasets totaling over a million training images. The pre-trained model is subsequently applied to three distinct downstream tasks with in-depth ablation studies conducted to emphasize its efficacy.

count=14
* GEO-Bench: Toward Foundation Models for Earth Monitoring
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/a0644215d9cff6646fa334dfa5d29c5a-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: GEO-Bench: Toward Foundation Models for Earth Monitoring
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Björn Lütjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, Mehmet Gunturkun, Gabriel Huang, David Vazquez, Dava Newman, Yoshua Bengio, Stefano Ermon, Xiaoxiang Zhu
    * Abstract: Recent progress in self-supervision has shown that pre-training large neural networks on vast amounts of unsupervised data can lead to substantial increases in generalization to downstream tasks. Such models, recently coined foundation models, have been transformational to the field of natural language processing.Variants have also been proposed for image data, but their applicability to remote sensing tasks is limited.To stimulate the development of foundation models for Earth monitoring, we propose a benchmark comprised of six classification and six segmentation tasks, which were carefully curated and adapted to be both relevant to the field and well-suited for model evaluation. We accompany this benchmark with a robust methodology for evaluating models and reporting aggregated results to enable a reliable assessment of progress. Finally, we report results for 20 baselines to gain information about the performance of existing models.We believe that this benchmark will be a driver of progress across a variety of Earth monitoring tasks.

count=10
* CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf)]
    * Title: CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Anthony Fuller, Koreen Millard, James Green
    * Abstract: A vital and rapidly growing application, remote sensing offers vast yet sparsely labeled, spatially aligned multimodal data; this makes self-supervised learning algorithms invaluable. We present CROMA: a framework that combines contrastive and reconstruction self-supervised objectives to learn rich unimodal and multimodal representations. Our method separately encodes masked-out multispectral optical and synthetic aperture radar samples—aligned in space and time—and performs cross-modal contrastive learning. Another encoder fuses these sensors, producing joint multimodal encodings that are used to predict the masked patches via a lightweight decoder. We show that these objectives are complementary when leveraged on spatially aligned multimodal data. We also introduce X- and 2D-ALiBi, which spatially biases our cross- and self-attention matrices. These strategies improve representations and allow our models to effectively extrapolate to images up to $17.6\times$ larger at test-time. CROMA outperforms the current SoTA multispectral model, evaluated on: four classification benchmarks—finetuning (avg.$\uparrow$ 1.8%), linear (avg.$\uparrow$ 2.4%) and nonlinear (avg.$\uparrow$ 1.4%) probing, $k$NN classification (avg.$\uparrow$ 3.5%), and $K$-means clustering (avg.$\uparrow$ 8.4%); and three segmentation benchmarks (avg.$\uparrow$ 6.4%). CROMA’s rich, optionally multimodal representations can be widely leveraged across remote sensing applications.

count=8
* SSL4EO-L: Datasets and Foundation Models for Landsat Imagery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/bbf7ee04e2aefec136ecf60e346c2e61-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/bbf7ee04e2aefec136ecf60e346c2e61-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: SSL4EO-L: Datasets and Foundation Models for Landsat Imagery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Adam Stewart, Nils Lehmann, Isaac Corley, Yi Wang, Yi-Chia Chang, Nassim Ait Ait Ali Braham, Shradha Sehgal, Caleb Robinson, Arindam Banerjee
    * Abstract: The Landsat program is the longest-running Earth observation program in history, with 50+ years of data acquisition by 8 satellites. The multispectral imagery captured by sensors onboard these satellites is critical for a wide range of scientific fields. Despite the increasing popularity of deep learning and remote sensing, the majority of researchers still use decision trees and random forests for Landsat image analysis due to the prevalence of small labeled datasets and lack of foundation models. In this paper, we introduce SSL4EO-L, the first ever dataset designed for Self-Supervised Learning for Earth Observation for the Landsat family of satellites (including 3 sensors and 2 product levels) and the largest Landsat dataset in history (5M image patches). Additionally, we modernize and re-release the L7 Irish and L8 Biome cloud detection datasets, and introduce the first ML benchmark datasets for Landsats 4–5 TM and Landsat 7 ETM+ SR. Finally, we pre-train the first foundation models for Landsat imagery using SSL4EO-L and evaluate their performance on multiple semantic segmentation tasks. All datasets and model weights are available via the TorchGeo library, making reproducibility and experimentation easy, and enabling scientific advancements in the burgeoning field of remote sensing for a multitude of downstream applications.

count=7
* Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Koch_Open3DSG_Open-Vocabulary_3D_Scene_Graphs_from_Point_Clouds_with_Queryable_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Koch_Open3DSG_Open-Vocabulary_3D_Scene_Graphs_from_Point_Clouds_with_Queryable_CVPR_2024_paper.pdf)]
    * Title: Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Sebastian Koch, Narunas Vaskevicius, Mirco Colosi, Pedro Hermosilla, Timo Ropinski
    * Abstract: Current approaches for 3D scene graph prediction rely on labeled datasets to train models for a fixed set of known object classes and relationship categories. We present Open3DSG an alternative approach to learn 3D scene graph prediction in an open world without requiring labeled scene graph data. We co-embed the features from a 3D scene graph prediction backbone with the feature space of powerful open world 2D vision language foundation models. This enables us to predict 3D scene graphs from 3D point clouds in a zero-shot manner by querying object classes from an open vocabulary and predicting the inter-object relationships from a grounded LLM with scene graph features and queried object classes as context. Open3DSG is the first 3D point cloud method to predict not only explicit open-vocabulary object classes but also open-set relationships that are not limited to a predefined label set making it possible to express rare as well as specific objects and relationships in the predicted 3D scene graph. Our experiments show that Open3DSG is effective at predicting arbitrary object classes as well as their complex inter-object relationships describing spatial supportive semantic and comparative relationships.

count=7
* Adapting the Segment Anything Model During Usage in Novel Situations
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Schon_Adapting_the_Segment_Anything_Model_During_Usage_in_Novel_Situations_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Schon_Adapting_the_Segment_Anything_Model_During_Usage_in_Novel_Situations_CVPRW_2024_paper.pdf)]
    * Title: Adapting the Segment Anything Model During Usage in Novel Situations
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Robin Schön, Julian Lorenz, Katja Ludwig, Rainer Lienhart
    * Abstract: The interactive segmentation task consists in the creation of object segmentation masks based on user interactions. The most common way to guide a model towards producing a correct segmentation consists in clicks on the object and background. The recently published Segment Anything Model (SAM) supports a generalized version of the interactive segmentation problem and has been trained on an object segmentation dataset which contains 1.1B masks. Though being trained extensively and with the explicit purpose of serving as a foundation model we show significant limitations of SAM when being applied for interactive segmentation on novel domains or object types. On the used datasets SAM displays a failure rate FR30@90 of up to 72.6 %. Since we still want such foundation models to be immediately applicable we present a framework that can adapt SAM during immediate usage. For this we will leverage the user interactions and masks which are constructed during the interactive segmentation process. We use this information to generate pseudo-labels which we use to compute a loss function and optimize a part of the SAM model. The presented method causes a relative reduction of up to 48.1 % in the FR20@85 and 46.6 % in the FR30@90 metrics.

count=6
* APPLeNet: Visual Attention Parameterized Prompt Learning for Few-Shot Remote Sensing Image Generalization Using CLIP
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Jha_APPLeNet_Visual_Attention_Parameterized_Prompt_Learning_for_Few-Shot_Remote_Sensing_CVPRW_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Jha_APPLeNet_Visual_Attention_Parameterized_Prompt_Learning_for_Few-Shot_Remote_Sensing_CVPRW_2023_paper.pdf)]
    * Title: APPLeNet: Visual Attention Parameterized Prompt Learning for Few-Shot Remote Sensing Image Generalization Using CLIP
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Mainak Singha, Ankit Jha, Bhupendra Solanki, Shirsha Bose, Biplab Banerjee
    * Abstract: In recent years, the success of large-scale vision-language models (VLMs) such as CLIP has led to their increased usage in various computer vision tasks. These models enable zero-shot inference through carefully crafted instructional text prompts without task-specific supervision. However, the potential of VLMs for generalization tasks in remote sensing (RS) has not been fully realized. To address this research gap, we propose a novel image-conditioned prompt learning strategy called the Visual Attention Parameterized Prompts Learning Network (APPLeNet). APPLeNet emphasizes the importance of multi-scale feature learning in RS scene classification and disentangles visual style and content primitives for domain generalization tasks. To achieve this, APPLeNet combines visual content features obtained from different layers of the vision encoder and style properties obtained from feature statistics of domain-specific batches. An attention-driven injection module is further introduced to generate visual tokens from this information. We also introduce an anti-correlation regularizer to ensure discrimination among the token embeddings, as this visual information is combined with the textual tokens. To validate APPLeNet, we curated four available RS benchmarks and introduced experimental protocols and datasets for three domain generalization tasks. Our results consistently outperform the relevant literature.

count=6
* Revisiting Pre-trained Remote Sensing Model Benchmarks: Resizing and Normalization Matters
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Corley_Revisiting_Pre-trained_Remote_Sensing_Model_Benchmarks_Resizing_and_Normalization_Matters_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Corley_Revisiting_Pre-trained_Remote_Sensing_Model_Benchmarks_Resizing_and_Normalization_Matters_CVPRW_2024_paper.pdf)]
    * Title: Revisiting Pre-trained Remote Sensing Model Benchmarks: Resizing and Normalization Matters
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Isaac Corley, Caleb Robinson, Rahul Dodhia, Juan M. Lavista Ferres, Peyman Najafirad
    * Abstract: Research in self-supervised learning (SSL) with natural images has progressed rapidly in recent years and is now increasingly being applied to and benchmarked with datasets containing remotely sensed imagery. A common benchmark case is to evaluate SSL pre-trained model embeddings on datasets of remotely sensed imagery with small patch sizes e.g. 32 x 32 pixels whereas standard SSL pre-training takes place with larger patch sizes e.g. 224 x 224. Furthermore pre-training methods tend to use different image normalization preprocessing steps depending on the dataset. In this paper we show across seven satellite and aerial imagery datasets of varying resolution that by simply following the preprocessing steps used in pre-training (precisely image sizing and normalization methods) one can achieve significant performance improvements when evaluating the extracted features on downstream tasks -- an important detail overlooked in previous work in this space. We show that by following these steps ImageNet pre-training remains a competitive baseline for satellite imagery based transfer learning tasks -- for example we find that these steps give +32.28 to overall accuracy on the So2Sat random split dataset and +11.16 on the EuroSAT dataset. Finally we report comprehensive benchmark results with a variety of simple baseline methods for each of the seven datasets forming an initial benchmark suite for remote sensing imagery.

count=5
* GeoLLM-Engine: A Realistic Environment for Building Geospatial Copilots
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Singh_GeoLLM-Engine_A_Realistic_Environment_for_Building_Geospatial_Copilots_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Singh_GeoLLM-Engine_A_Realistic_Environment_for_Building_Geospatial_Copilots_CVPRW_2024_paper.pdf)]
    * Title: GeoLLM-Engine: A Realistic Environment for Building Geospatial Copilots
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Simranjit Singh, Michael Fore, Dimitrios Stamoulis
    * Abstract: Geospatial Copilots unlock unprecedented potential for performing Earth Observation (EO) applications through natural language instructions. However existing agents rely on overly simplified single tasks and template-based prompts creating a disconnect with real-world scenarios. In this work we present GeoLLM-Engine an environment for tool-augmented agents with intricate tasks routinely executed by analysts on remote sensing platforms. We enrich our environment with geospatial API tools dynamic maps/UIs and external multimodal knowledge bases to properly gauge an agent's proficiency in interpreting realistic high-level natural language commands and its functional correctness in task completions. By alleviating overheads typically associated with human-in-the-loop benchmark curation we harness our massively parallel engine across 100 GPT-4-Turbo nodes scaling to over half a million diverse multi-tool tasks and across 1.1 million satellite images. By moving beyond traditional single-task image-caption paradigms we investigate state-of-the-art agents and prompting techniques against long-horizon prompts.

count=5
* Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/2ab3163ee384cd46baa7f1abb2b1bf19-Abstract-Conference.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/2ab3163ee384cd46baa7f1abb2b1bf19-Paper-Conference.pdf)]
    * Title: Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Katie Luo, Zhenzhen Liu, Xiangyu Chen, Yurong You, Sagie Benaim, Cheng Perng Phoo, Mark Campbell, Wen Sun, Bharath Hariharan, Kilian Q. Weinberger
    * Abstract: Recent advances in machine learning have shown that Reinforcement Learning from Human Feedback (RLHF) can improve machine learning models and align them with human preferences. Although very successful for Large Language Models (LLMs), these advancements have not had a comparable impact in research for autonomous vehicles—where alignment with human expectations can be imperative. In this paper, we propose to adapt similar RL-based methods to unsupervised object discovery, i.e. learning to detect objects from LiDAR points without any training labels. Instead of labels, we use simple heuristics to mimic human feedback. More explicitly, we combine multiple heuristics into a simple reward function that positively correlates its score with bounding box accuracy, i.e., boxes containing objects are scored higher than those without. We start from the detector’s own predictions to explore the space and reinforce boxes with high rewards through gradient updates. Empirically, we demonstrate that our approach is not only more accurate, but also orders of magnitudes faster to train compared to prior works on object discovery. Code is available at https://github.com/katieluo88/DRIFT.

count=4
* Prompt-RSVQA: Prompting Visual Context to a Language Model for Remote Sensing Visual Question Answering
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Chappuis_Prompt-RSVQA_Prompting_Visual_Context_to_a_Language_Model_for_Remote_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Chappuis_Prompt-RSVQA_Prompting_Visual_Context_to_a_Language_Model_for_Remote_CVPRW_2022_paper.pdf)]
    * Title: Prompt-RSVQA: Prompting Visual Context to a Language Model for Remote Sensing Visual Question Answering
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Christel Chappuis, Valérie Zermatten, Sylvain Lobry, Bertrand Le Saux, Devis Tuia
    * Abstract: Remote sensing visual question answering (RSVQA) was recently proposed with the aim of interfacing natural language and vision to ease the access of information contained in Earth Observation data for a wide audience, which is granted by simple questions in natural language. The traditional vision/language interface is an embedding obtained by fusing features from two deep models, one processing the image and another the question. Despite the success of early VQA models, it remains difficult to control the adequacy of the visual information extracted by its deep model, which should act as a context regularizing the work of the language model. We propose to extract this context information with a visual model, convert it to text and inject it, i.e. prompt it, into a language model. The language model is therefore responsible to process the question with the visual context, and extract features, which are useful to find the answer. We study the effect of prompting with respect to a black-box visual extractor and discuss the importance of training a visual model producing accurate context.

count=3
* Good at Captioning Bad at Counting: Benchmarking GPT-4V on Earth Observation Data
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Zhang_Good_at_Captioning_Bad_at_Counting_Benchmarking_GPT-4V_on_Earth_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Zhang_Good_at_Captioning_Bad_at_Counting_Benchmarking_GPT-4V_on_Earth_CVPRW_2024_paper.pdf)]
    * Title: Good at Captioning Bad at Counting: Benchmarking GPT-4V on Earth Observation Data
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Chenhui Zhang,Sherrie Wang
    * Abstract: Large Vision-Language Models (VLMs) have demonstrated impressive performance on complex tasks involving visual input with natural language instructions. However it remains unclear to what extent capabilities on natural images transfer to Earth observation (EO) data which are predominantly satellite and aerial images less common in VLM training data. In this work we propose a comprehensive benchmark to gauge the progress of VLMs toward being useful tools for EO data by assessing their abilities on scene understanding localization and counting and change detection. Motivated by real-world applications our benchmark includes scenarios like urban monitoring disaster relief land use and conservation. We discover that although state-of-the-art VLMs like GPT-4V possess world knowledge that leads to strong performance on location understanding and image captioning their poor spatial reasoning limits usefulness on object localization and counting. Our benchmark leaderboard and evaluation suite are available at https://vleo.danielz.ch/. A full version of this paper is available at https://arxiv.org/abs/2401.17600.

count=2
* Decoupled DETR For Few-shot Object Detection
    [[abs-CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Shangguan_Decoupled_DETR_For_Few-shot_Object_Detection_ACCV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Shangguan_Decoupled_DETR_For_Few-shot_Object_Detection_ACCV_2024_paper.pdf)]
    * Title: Decoupled DETR For Few-shot Object Detection
    * Publisher: ACCV
    * Publication Date: `2024`
    * Authors: Zeyu Shangguan, Lian Huai, Tong Liu, Yuyu Liu, Xingqun Jiang
    * Abstract: The efficient technique for dealing with severe data-hungry issues in object detection, known as Few-shot object detection (FSOD), has been widely explored. However, FSOD encounters some notable challenges such as the model's natural bias towards pre-training data and the inherent defects present in the existing models. In this paper, we introduce improved methods for the FSOD problem based on DETR structures: (i) To reduce bias from pre-training classes (i.e. many-shot base classes), we investigate the impact of decoupling the parameters of pre-training classes and fine-tuning classes (i.e. few-shot novel classes) in various ways. As a result, we propose a "base-novel categories decoupled DETR (DeDETR)" network for FSOD. (ii) To further improve the efficiency of the DETR's skip connection structure, we explore varied skip connection types in the DETR's encoder and decoder. Subsequently, we introduce a unified decoder module that dynamically blends decoder layers to generate the output feature. Our model's effectiveness is evaluated using PASCAL VOC and MSCOCO datasets. Our results indicate that our proposed module consistently improves performance by 5% to 10% in both fine-tuning and meta-learning frameworks and has surpassed the top scores achieved in recent studies.

count=2
* Hephaestus: A Large Scale Multitask Dataset Towards InSAR Understanding
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Bountos_Hephaestus_A_Large_Scale_Multitask_Dataset_Towards_InSAR_Understanding_CVPRW_2022_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Bountos_Hephaestus_A_Large_Scale_Multitask_Dataset_Towards_InSAR_Understanding_CVPRW_2022_paper.pdf)]
    * Title: Hephaestus: A Large Scale Multitask Dataset Towards InSAR Understanding
    * Publisher: CVPR
    * Publication Date: `2022`
    * Authors: Nikolaos Ioannis Bountos, Ioannis Papoutsis, Dimitrios Michail, Andreas Karavias, Panagiotis Elias, Isaak Parcharidis
    * Abstract: Synthetic Aperture Radar (SAR) data and Interferometric SAR (InSAR) products in particular, are one of the largest sources of Earth Observation data. InSAR provides unique information on diverse geophysical processes and geology, and on the geotechnical properties of man-made structures. However, there are only a limited number of applications that exploit the abundance of InSAR data and deep learning methods to extract such knowledge. The main barrier has been the lack of a large curated and annotated InSAR dataset, which would be costly to create and would require an interdisciplinary team of experts experienced on InSAR data interpretation. In this work, we put the effort to create and make available the first of its kind, manually annotated dataset that consists of 19,919 individual Sentinel-1 interferograms acquired over 44 different volcanoes globally, which are split into 216,106 InSAR patches. The annotated dataset is designed to address different computer vision problems, including volcano state classification, semantic segmentation of ground deformation, detection and classification of atmospheric signals in InSAR imagery, interferogram captioning, text to InSAR generation, and InSAR image quality assessment.

count=2
* Token Boosting for Robust Self-Supervised Visual Transformer Pre-Training
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Token_Boosting_for_Robust_Self-Supervised_Visual_Transformer_Pre-Training_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Token_Boosting_for_Robust_Self-Supervised_Visual_Transformer_Pre-Training_CVPR_2023_paper.pdf)]
    * Title: Token Boosting for Robust Self-Supervised Visual Transformer Pre-Training
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Tianjiao Li, Lin Geng Foo, Ping Hu, Xindi Shang, Hossein Rahmani, Zehuan Yuan, Jun Liu
    * Abstract: Learning with large-scale unlabeled data has become a powerful tool for pre-training Visual Transformers (VTs). However, prior works tend to overlook that, in real-world scenarios, the input data may be corrupted and unreliable. Pre-training VTs on such corrupted data can be challenging, especially when we pre-train via the masked autoencoding approach, where both the inputs and masked "ground truth" targets can potentially be unreliable in this case. To address this limitation, we introduce the Token Boosting Module (TBM) as a plug-and-play component for VTs that effectively allows the VT to learn to extract clean and robust features during masked autoencoding pre-training. We provide theoretical analysis to show how TBM improves model pre-training with more robust and generalizable representations, thus benefiting downstream tasks. We conduct extensive experiments to analyze TBM's effectiveness, and results on four corrupted datasets demonstrate that TBM consistently improves performance on downstream tasks.

count=2
* Step Differences in Instructional Video
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Nagarajan_Step_Differences_in_Instructional_Video_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Nagarajan_Step_Differences_in_Instructional_Video_CVPR_2024_paper.pdf)]
    * Title: Step Differences in Instructional Video
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Tushar Nagarajan, Lorenzo Torresani
    * Abstract: Comparing a user video to a reference how-to video is a key requirement for AR/VR technology delivering personalized assistance tailored to the user's progress. However current approaches for language-based assistance can only answer questions about a single video. We propose an approach that first automatically generates large amounts of visual instruction tuning data involving pairs of videos from HowTo100M by leveraging existing step annotations and accompanying narrations and then trains a video-conditioned language model to jointly reason across multiple raw videos. Our model achieves state-of-the-art performance at identifying differences between video pairs and ranking videos based on the severity of these differences and shows promising ability to perform general reasoning over multiple videos.

count=2
* ENIGMA-51: Towards a Fine-Grained Understanding of Human Behavior in Industrial Scenarios
    [[abs-CVF](https://openaccess.thecvf.com/content/WACV2024/html/Ragusa_ENIGMA-51_Towards_a_Fine-Grained_Understanding_of_Human_Behavior_in_Industrial_WACV_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Ragusa_ENIGMA-51_Towards_a_Fine-Grained_Understanding_of_Human_Behavior_in_Industrial_WACV_2024_paper.pdf)]
    * Title: ENIGMA-51: Towards a Fine-Grained Understanding of Human Behavior in Industrial Scenarios
    * Publisher: WACV
    * Publication Date: `2024`
    * Authors: Francesco Ragusa, Rosario Leonardi, Michele Mazzamuto, Claudia Bonanno, Rosario Scavo, Antonino Furnari, Giovanni Maria Farinella
    * Abstract: ENIGMA-51 is a new egocentric dataset acquired in an industrial scenario by 19 subjects who followed instructions to complete the repair of electrical boards using industrial tools (e.g., electric screwdriver) and equipments (e.g., oscilloscope). The 51 egocentric video sequences are densely annotated with a rich set of labels that enable the systematic study of human behavior in the industrial domain. We provide benchmarks on four tasks related to human behavior: 1) untrimmed temporal detection of human-object interactions, 2) egocentric human-object interaction detection, 3) short-term object interaction anticipation and 4) natural language understanding of intents and entities. Baseline results show that the ENIGMA-51 dataset poses a challenging benchmark to study human behavior in industrial scenarios. We publicly release the dataset at https://iplab.dmi.unict.it/ENIGMA-51.

count=1
* ZBS: Zero-Shot Background Subtraction via Instance-Level Background Modeling and Foreground Selection
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/An_ZBS_Zero-Shot_Background_Subtraction_via_Instance-Level_Background_Modeling_and_Foreground_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/An_ZBS_Zero-Shot_Background_Subtraction_via_Instance-Level_Background_Modeling_and_Foreground_CVPR_2023_paper.pdf)]
    * Title: ZBS: Zero-Shot Background Subtraction via Instance-Level Background Modeling and Foreground Selection
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Yongqi An, Xu Zhao, Tao Yu, Haiyun Guo, Chaoyang Zhao, Ming Tang, Jinqiao Wang
    * Abstract: Background subtraction (BGS) aims to extract all moving objects in the video frames to obtain binary foreground segmentation masks. Deep learning has been widely used in this field. Compared with supervised-based BGS methods, unsupervised methods have better generalization. However, previous unsupervised deep learning BGS algorithms perform poorly in sophisticated scenarios such as shadows or night lights, and they cannot detect objects outside the pre-defined categories. In this work, we propose an unsupervised BGS algorithm based on zero-shot object detection called Zero-shot Background Subtraction ZBS. The proposed method fully utilizes the advantages of zero-shot object detection to build the open-vocabulary instance-level background model. Based on it, the foreground can be effectively extracted by comparing the detection results of new frames with the background model. ZBS performs well for sophisticated scenarios, and it has rich and extensible categories. Furthermore, our method can easily generalize to other tasks, such as abandoned object detection in unseen environments. We experimentally show that ZBS surpasses state-of-the-art unsupervised BGS methods by 4.70% F-Measure on the CDnet 2014 dataset. The code is released at https://github.com/CASIA-IVA-Lab/ZBS.

count=1
* Learning To Exploit Temporal Structure for Biomedical Vision-Language Processing
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.pdf)]
    * Title: Learning To Exploit Temporal Structure for Biomedical Vision-Language Processing
    * Publisher: CVPR
    * Publication Date: `2023`
    * Authors: Shruthi Bannur, Stephanie Hyland, Qianchu Liu, Fernando Pérez-García, Maximilian Ilse, Daniel C. Castro, Benedikt Boecking, Harshita Sharma, Kenza Bouzid, Anja Thieme, Anton Schwaighofer, Maria Wetscherek, Matthew P. Lungren, Aditya Nori, Javier Alvarez-Valle, Ozan Oktay
    * Abstract: Self-supervised learning in vision--language processing (VLP) exploits semantic alignment between imaging and text modalities. Prior work in biomedical VLP has mostly relied on the alignment of single image and report pairs even though clinical notes commonly refer to prior images. This does not only introduce poor alignment between the modalities but also a missed opportunity to exploit rich self-supervision through existing temporal content in the data. In this work, we explicitly account for prior images and reports when available during both training and fine-tuning. Our approach, named BioViL-T, uses a CNN--Transformer hybrid multi-image encoder trained jointly with a text model. It is designed to be versatile to arising challenges such as pose variations and missing input images across time. The resulting model excels on downstream tasks both in single- and multi-image setups, achieving state-of-the-art (SOTA) performance on (I) progression classification, (II) phrase grounding, and (III) report generation, whilst offering consistent improvements on disease classification and sentence-similarity tasks. We release a novel multi-modal temporal benchmark dataset, CXR-T, to quantify the quality of vision--language representations in terms of temporal semantics. Our experimental results show the significant advantages of incorporating prior images and reports to make most use of the data.

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
* Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Noman_Rethinking_Transformers_Pre-training_for_Multi-Spectral_Satellite_Imagery_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Noman_Rethinking_Transformers_Pre-training_for_Multi-Spectral_Satellite_Imagery_CVPR_2024_paper.pdf)]
    * Title: Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Mubashir Noman, Muzammal Naseer, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Fahad Shahbaz Khan
    * Abstract: Recent advances in unsupervised learning have demonstrated the ability of large vision models to achieve promising results on downstream tasks by pre-training on large amount of unlabelled data. Such pre-training techniques have also been explored recently in the remote sensing domain due to the availability of large amount of unlabelled data. Different from standard natural image datasets remote sensing data is acquired from various sensor technologies and exhibit diverse range of scale variations as well as modalities. Existing satellite image pre-training methods either ignore the scale information present in the remote sensing imagery or restrict themselves to use only a single type of data modality. In this paper we re-visit transformers pre-training and leverage multi-scale information that is effectively utilized with multiple modalities. Our proposed approach named SatMAE++ performs multi-scale pre-training and utilizes convolution based upsampling blocks to reconstruct the image at higher scales making it extensible to include more scales. Compared to existing works the proposed SatMAE++ with multi-scale pre-training is equally effective for both optical as well as multi-spectral imagery. Extensive experiments on six datasets reveal the merits of proposed contributions leading to state-of-the-art performance on all datasets. SatMAE++ achieves mean average precision (mAP) gain of 2.5% for multi-label classification task on BigEarthNet dataset.

count=1
* Fantastic Animals and Where to Find Them: Segment Any Marine Animal with Dual SAM
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Fantastic_Animals_and_Where_to_Find_Them_Segment_Any_Marine_CVPR_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Fantastic_Animals_and_Where_to_Find_Them_Segment_Any_Marine_CVPR_2024_paper.pdf)]
    * Title: Fantastic Animals and Where to Find Them: Segment Any Marine Animal with Dual SAM
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Pingping Zhang, Tianyu Yan, Yang Liu, Huchuan Lu
    * Abstract: As an important pillar of underwater intelligence Marine Animal Segmentation (MAS) involves segmenting animals within marine environments. Previous methods don't excel in extracting long-range contextual features and overlook the connectivity between discrete pixels. Recently Segment Anything Model (SAM) offers a universal framework for general segmentation tasks. Unfortunately trained with natural images SAM does not obtain the prior knowledge from marine images. In addition the single-position prompt of SAM is very insufficient for prior guidance. To address these issues we propose a novel feature learning framework named Dual-SAM for high-performance MAS. To this end we first introduce a dual structure with SAM's paradigm to enhance feature learning of marine images. Then we propose a Multi-level Coupled Prompt (MCP) strategy to instruct comprehensive underwater prior information and enhance the multi-level features of SAM's encoder with adapters. Subsequently we design a Dilated Fusion Attention Module (DFAM) to progressively integrate multi-level features from SAM's encoder. Finally instead of directly predicting the masks of marine animals we propose a Criss-Cross Connectivity Prediction (C3P) paradigm to capture the inter-connectivity between discrete pixels. With dual decoders it generates pseudo-labels and achieves mutual supervision for complementary feature representations resulting in considerable improvements over previous techniques. Extensive experiments verify that our proposed method achieves state-of-the-art performances on five widely-used MAS datasets. The code is available at https://github.com/Drchip61/Dual SAM.

count=1
* DeepLocalization: Using Change Point Detection for Temporal Action Localization
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Rahman_DeepLocalization_Using_Change_Point_Detection_for_Temporal_Action_Localization_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Rahman_DeepLocalization_Using_Change_Point_Detection_for_Temporal_Action_Localization_CVPRW_2024_paper.pdf)]
    * Title: DeepLocalization: Using Change Point Detection for Temporal Action Localization
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Mohammed Shaiqur Rahman, Ibne Farabi Shihab, Lynna Chu, Anuj Sharma
    * Abstract: In this study we introduce DeepLocalization an innovative framework devised for the real-time localization of actions tailored explicitly for monitoring driver behavior. Utilizing the power of advanced deep learning methodologies our objective is to tackle the critical issue of distracted driving--a significant factor contributing to road accidents. Our strategy employs a dual approach: leveraging Graph-Based Change-Point Detection for pinpointing actions in time alongside a Video Large Language Model (Video-LLM) for precisely categorizing activities. Through careful prompt engineering we customize the Video-LLM to adeptly handle driving activities' nuances ensuring its classification efficacy even with sparse data. Engineered to be lightweight our framework is optimized for consumer-grade GPUs making it vastly applicable in practical scenarios. We subjected our method to rigorous testing on the SynDD2 dataset a complex benchmark for distracted driving behaviors where it demonstrated commendable performance--achieving 57.5% accuracy in event classification and 51% in event detection. These outcomes underscore the substantial promise of DeepLocalization in accurately identifying diverse driver behaviors and their temporal occurrences all within the bounds of limited computational resources.

count=1
* Detecting Out-Of-Distribution Earth Observation Images with Diffusion Models
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Le_Bellier_Detecting_Out-Of-Distribution_Earth_Observation_Images_with_Diffusion_Models_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Le_Bellier_Detecting_Out-Of-Distribution_Earth_Observation_Images_with_Diffusion_Models_CVPRW_2024_paper.pdf)]
    * Title: Detecting Out-Of-Distribution Earth Observation Images with Diffusion Models
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Georges Le Bellier, Nicolas Audebert
    * Abstract: Earth Observation imagery can capture rare and unusual events such as disasters and major landscape changes whose visual appearance contrasts with the usual observations. Deep models trained on common remote sensing data will output drastically different features for these out-of-distribution samples compared to those closer to their training dataset. Detecting them could therefore help anticipate changes in the observations either geographical or environmental. In this work we show that the reconstruction error of diffusion models can effectively serve as unsupervised out-of-distribution detectors for remote sensing images using them as a plausibility score. Moreover we introduce ODEED a novel reconstruction-based scorer using the probability-flow ODE of diffusion models. We validate it experimentally on SpaceNet 8 with various scenarios such as classical OOD detection with geographical shift and near-OOD setups: pre/post-flood and non-flooded/flooded image recognition. We show that our ODEED scorer significantly outperforms other diffusion-based and discriminative baselines on the more challenging near-OOD scenarios of flood image detection where OOD images are close to the distribution tail. We aim to pave the way towards better use of generative models for anomaly detection in remote sensing.

count=1
* Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Okabayashi_Cross-sensor_super-resolution_of_irregularly_sampled_Sentinel-2_time_series_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Okabayashi_Cross-sensor_super-resolution_of_irregularly_sampled_Sentinel-2_time_series_CVPRW_2024_paper.pdf)]
    * Title: Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Aimi Okabayashi, Nicolas Audebert, Simon Donike, Charlotte Pelletier
    * Abstract: Satellite imaging generally presents a trade-off between the frequency of acquisitions and the spatial resolution of the images. Super-resolution is often advanced as a way to get the best of both worlds. In this work we investigate multi-image super-resolution of satellite image time series i.e. how multiple images of the same area acquired at different dates can help reconstruct a higher resolution observation. In particular we extend state-of-the-art deep single and multi-image super-resolution algorithms such as SRDiff and HighRes-net to deal with irregularly sampled Sentinel-2 time series. We introduce BreizhSR a new dataset for 4x super-resolution of Sentinel-2 time series using very high-resolution SPOT-6 imagery of Brittany a French region. We show that using multiple images significantly improves super-resolution performance and that a well-designed temporal positional encoding allows us to perform super-resolution for different times of the series. In addition we observe a trade-off between spectral fidelity and perceptual quality of the reconstructed HR images questioning future directions for super-resolution of Earth Observation data. The source code is available at https://github.com/aimiokab/MISR-S2.

count=1
* UrbanSARFloods: Sentinel-1 SLC-Based Benchmark Dataset for Urban and Open-Area Flood Mapping
    [[abs-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Zhao_UrbanSARFloods_Sentinel-1_SLC-Based_Benchmark_Dataset_for_Urban_and_Open-Area_Flood_CVPRW_2024_paper.html)]
    [[pdf-CVF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Zhao_UrbanSARFloods_Sentinel-1_SLC-Based_Benchmark_Dataset_for_Urban_and_Open-Area_Flood_CVPRW_2024_paper.pdf)]
    * Title: UrbanSARFloods: Sentinel-1 SLC-Based Benchmark Dataset for Urban and Open-Area Flood Mapping
    * Publisher: CVPR
    * Publication Date: `2024`
    * Authors: Jie Zhao, Zhitong Xiong, Xiao Xiang Zhu
    * Abstract: Due to its cloud-penetrating capability and independence from solar illumination satellite Synthetic Aperture Radar (SAR) is the preferred data source for large-scale flood mapping providing global coverage and including various land cover classes. However most studies on large-scale SAR-derived flood mapping using deep learning algorithms have primarily focused on flooded open areas utilizing available open-access datasets (e.g. Sen1Floods11) and with limited attention to urban floods. To address this gap we introduce UrbanSARFloods a floodwater dataset featuring pre-processed Sentinel-1 intensity data and interferometric coherence imagery acquired before and during flood events. It contains 8879 512 x 512 chips covering 807500 km2 across 20 land cover classes and 5 continents spanning 18 flood events. We used UrbanSARFloods to benchmark existing state-of-the-art convolutional neural networks (CNNs) for segmenting open and urban flood areas. Our findings indicate that prevalent approaches including the Weighted Cross-Entropy (WCE) loss and the application of transfer learning with pretrained models fall short in overcoming the obstacles posed by imbalanced data and the constraints of a small training dataset. Urban flood detection remains challenging. Future research should explore strategies for addressing imbalanced data challenges and investigate transfer learning's potential for SAR-based large-scale flood mapping. Besides expanding this dataset to include additional flood events holds promise for enhancing its utility and contributing to advancements in flood mapping techniques. The UrbanSARFloods dataset including training validation data and raw data can be found at https://github.com/jie666-6/UrbanSARFloods.

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
* VidChapters-7M: Video Chapters at Scale
    [[abs-NIPS](https://papers.nips.cc/paper_files/paper/2023/hash/9b5c3e00d6ed30aad7adac9e7a664de1-Abstract-Datasets_and_Benchmarks.html)]
    [[pdf-NIPS](https://papers.nips.cc/paper_files/paper/2023/file/9b5c3e00d6ed30aad7adac9e7a664de1-Paper-Datasets_and_Benchmarks.pdf)]
    * Title: VidChapters-7M: Video Chapters at Scale
    * Publisher: NeurIPS
    * Publication Date: `2023`
    * Authors: Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid
    * Abstract: Segmenting untrimmed videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters-7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title. We benchmark both simple baselines as well as state-of-the-art video-language models on these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset.

