# InstructSeg: Unifying Instructed Visual Segmentation with Multi-modal Large Language Models

> #### Cong Wei, Yujie Zhong<sup>&dagger;</sup>, Haoxian Tan, Yingsen Zeng, Yong Liu, Zheng Zhao, and Yujiu Yang<sup>&dagger;</sup>
>
> <sup>&dagger;</sup>Correspondence


<p align="center">
 <img src="imgs/intro.jpg" width="100%">
</p>


## News
- [x] Release InstructSeg project page.
- [x] Our model and code is coming soon.



## Abstract
Boosted by Multi-modal Large Language Models (MLLMs), text-guided universal segmentation models for the image and video domains have made rapid progress recently. However, these methods are often developed separately for specific domains, overlooking the similarities in task settings and solutions across these two domains. In this paper, we define the union of referring segmentation and reasoning segmentation at both the image and video levels as Instructed Visual Segmentation (IVS). Correspondingly, we propose InstructSeg, an end-to-end segmentation pipeline equipped with MLLMs for IVS. Specifically, we employ an object-aware video perceiver to extract temporal and object information from reference frames, facilitating comprehensive video understanding. Additionally, we introduce vision-guided multi-granularity text fusion to better integrate global and detailed text information with fine-grained visual guidance. By leveraging multi-task and end-to-end training, InstructSeg demonstrates superior performance across diverse image and video segmentation tasks, surpassing both segmentation specialists and MLLM-based methods with a single model.




## Pipeline
<p align="center">
 <img src="imgs/model.jpg" width="100%">
</p>


## Results
### RES & ReasonSeg
<p align="center">
 <img src="imgs/exp_res.jpg" width="90%">
</p>

### Reasoning Video Object Segmentation
<p align="center">
 <img src="imgs/exp_revos.jpg" width="90%">
</p>

### Referring Video Object Segmentation
<p align="center">
 <img src="imgs/exp_rvos.jpg" width="90%">
</p>

### Multi-modal Question Answering Benchmarks.
<p align="center">
 <img src="imgs/exp_mmbench.jpg" width="75%">
</p>


