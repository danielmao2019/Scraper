* Active Learning by Learning
    [[abs-AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9597)]
    [[pdf-AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9597/9456)]
    * Title: Active Learning by Learning
    * Year: `2015`/02/21
    * Authors: Wei-Ning Hsu, Hsuan-Tien Lin
    * Abstract: Pool-based active learning is an important technique that helps reduce labeling efforts within a pool of unlabeled instances. Currently, most pool-based active learning strategies are constructed based on some human-designed philosophy; that is, they reflect what human beings assume to be “good labeling questions.” However, while such human-designed philosophies can be useful on specific data sets, it is often difficult to establish the theoretical connection of those philosophies to the true learning performance of interest. In addition, given that a single human-designed philosophy is unlikely to work on all scenarios, choosing and blending those strategies under different scenarios is an important but challenging practical task. This paper tackles this task by letting the machines adaptively “learn” from the performance of a set of given strategies on a particular data set. More specifically, we design a learning algorithm that connects active learning with the well-known multi-armed bandit problem. Further, we postulate that, given an appropriate choice for the multi-armed bandit learner, it is possible to estimate the performance of different strategies on the fly. Extensive empirical studies of the resulting ALBL algorithm confirm that it performs better than state-of-the-art strategies and a leading blending algorithm for active learning, all of which are based on human-designed philosophy.