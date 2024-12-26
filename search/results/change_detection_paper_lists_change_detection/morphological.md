* Rolling Shutter Motion Deblurring
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/html/Su_Rolling_Shutter_Motion_2015_CVPR_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2015/papers/Su_Rolling_Shutter_Motion_2015_CVPR_paper.pdf)]
    * Title: Rolling Shutter Motion Deblurring
    * Year: `2015`
    * Authors: Shuochen Su, Wolfgang Heidrich
    * Abstract: Although motion blur and rolling shutter deformations are closely coupled artifacts in images taken with CMOS image sensors, the two phenomena have so far mostly been treated separately, with deblurring algorithms being unable to handle rolling shutter wobble, and rolling shutter algorithms being incapable of dealing with motion blur. We propose an approach that delivers sharp and undistorted output given a single rolling shutter motion blurred image. The key to achieving this is a global modeling of the camera motion trajectory, which enables each scanline of the image to be deblurred with the corresponding motion segment. We show the results of the proposed framework through experiments on synthetic and real data.

    Count: 1* Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    [[abs-CVPR](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Video_Rain_Streak_CVPR_2018_paper.html)]
    [[pdf-CVPR](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Video_Rain_Streak_CVPR_2018_paper.pdf)]
    * Title: Video Rain Streak Removal by Multiscale Convolutional Sparse Coding
    * Year: `2018`
    * Authors: Minghan Li, Qi Xie, Qian Zhao, Wei Wei, Shuhang Gu, Jing Tao, Deyu Meng
    * Abstract: Videos captured by outdoor surveillance equipments sometimes contain unexpected rain streaks, which brings difficulty in subsequent video processing tasks. Rain streak removal from a video is thus an important topic in recent computer vision research. In this paper, we raise two intrinsic characteristics specifically possessed by rain streaks. Firstly, the rain streaks in a video contain repetitive local patterns sparsely scattered over different positions of the video. Secondly, the rain streaks are with multiscale configurations due to their occurrence on positions with different distances to the cameras. Based on such understanding, we specifically formulate both characteristics into a multiscale convolutional sparse coding (MS-CSC) model for the video rain streak removal task. Specifically, we use multiple convolutional filters convolved on the sparse feature maps to deliver the former characteristic, and further use multiscale filters to represent different scales of rain streaks. Such a new encoding manner makes the proposed method capable of properly extracting rain streaks from videos, thus getting fine video deraining effects. Experiments implemented on synthetic and real videos verify the superiority of the proposed method, as compared with the state-of-the-art ones along this research line, both visually and quantitatively.

    Count: 1* Simultaneous Foreground Detection and Classification With Hybrid Features
    [[abs-ICCV](https://openaccess.thecvf.com/content_iccv_2015/html/Kim_Simultaneous_Foreground_Detection_ICCV_2015_paper.html)]
    [[pdf-ICCV](https://openaccess.thecvf.com/content_iccv_2015/papers/Kim_Simultaneous_Foreground_Detection_ICCV_2015_paper.pdf)]
    * Title: Simultaneous Foreground Detection and Classification With Hybrid Features
    * Year: `2015`
    * Authors: Jaemyun Kim, Adin Ramirez Rivera, Byungyong Ryu, Oksam Chae
    * Abstract: In this paper, we propose a hybrid background model that relies on edge and non-edge features of the image to produce the model. We encode these features into a coding scheme, that we called Local Hybrid Pattern (LHP), that selectively models edges and non-edges features of each pixel. Furthermore, we model each pixel with an adaptive code dictionary to represent the background dynamism, and update it by adding stable codes and discarding unstable ones. We weight each code in the dictionary to enhance its description of the pixel it models. The foreground is detected as the incoming codes that deviate from the dictionary. We can detect (as foreground or background) and classify (as edge or inner region) each pixel simultaneously. We tested our proposed method in existing databases with promising results.

    Count: 1* Should We Encode Rain Streaks in Video as Deterministic or Stochastic?
    [[abs-ICCV](https://openaccess.thecvf.com/content_iccv_2017/html/Wei_Should_We_Encode_ICCV_2017_paper.html)]
    [[pdf-ICCV](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wei_Should_We_Encode_ICCV_2017_paper.pdf)]
    * Title: Should We Encode Rain Streaks in Video as Deterministic or Stochastic?
    * Year: `2017`
    * Authors: Wei Wei, Lixuan Yi, Qi Xie, Qian Zhao, Deyu Meng, Zongben Xu
    * Abstract: Videos taken in the wild sometimes contain unexpected rain streaks, which brings difficulty in subsequent video processing tasks. Rain streak removal in a video (RSRV) is thus an important issue and has been attracting much attention in computer vision. Different from previous RSRV methods formulating rain streaks as a deterministic message, this work first encodes the rains in a stochastic manner, i.e., a patch-based mixture of Gaussians. Such modification makes the proposed model capable of finely adapting a wider range of rain variations instead of certain types of rain configurations as traditional. By integrating with the spatiotemporal smoothness configuration of moving objects and low-rank structure of background scene, we propose a concise model for RSRV, containing one likelihood term imposed on the rain streak layer and two prior terms on the moving object and background scene layers of the video. Experiments implemented on videos with synthetic and real rains verify the superiority of the proposed method, as com- pared with the state-of-the-art methods, both visually and quantitatively in various performance metrics.

    Count: 1* Spot the Difference: Detection of Topological Changes via Geometric Alignment
    [[abs-NeurIPS](https://papers.nips.cc/paper_files/paper/2021/hash/7867d6557b82ed3b5d61e6591a2a2fd3-Abstract.html)]
    [[pdf-NeurIPS](https://papers.nips.cc/paper_files/paper/2021/file/7867d6557b82ed3b5d61e6591a2a2fd3-Paper.pdf)]
    * Title: Spot the Difference: Detection of Topological Changes via Geometric Alignment
    * Year: `2021`
    * Authors: Per Steffen Czolbe, Aasa Feragen, Oswin Krause
    * Abstract: Geometric alignment appears in a variety of applications, ranging from domain adaptation, optimal transport, and normalizing flows in machine learning; optical flow and learned augmentation in computer vision and deformable registration within biomedical imaging. A recurring challenge is the alignment of domains whose topology is not the same; a problem that is routinely ignored, potentially introducing bias in downstream analysis. As a first step towards solving such alignment problems, we propose an unsupervised algorithm for the detection of changes in image topology. The model is based on a conditional variational auto-encoder and detects topological changes between two images during the registration step. We account for both topological changes in the image under spatial variation and unexpected transformations. Our approach is validated on two tasks and datasets: detection of topological changes in microscopy images of cells, and unsupervised anomaly detection brain imaging.

    Count: 1