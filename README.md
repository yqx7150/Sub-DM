# Sub-DM
**Paper**: Sub-DM: Subspace Diffusion Model with Orthogonal Decomposition for MRI Reconstruction

**Authors**: Yu Guan, Qinrong Cai, Wei Li, Qiuyun Fan, Dong Liang, Qiegen Liu *

https://arxiv.org/abs/2411.03758

Date : November-6-2024  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Mathematics and Computer Sciences, Nanchang University. 


Diffusion model-based approaches recently achieved remarkable success in MRI reconstruction, but inte-gration into clinical routine remains challenging due to its time-consuming convergence. This phenomenon is particularly notable when directly apply conventional diffusion process to k-space data without considering the inherent properties of k-space sampling, limiting k-space learning efficiency and image reconstruction quality. To tackle these challenges, we introduce subspace diffusion model with orthogonal decomposition, a method (referred to as Sub-DM) that restrict the diffusion process via projections onto subspace as the k-space data distribution evolves toward noise. Particularly, the orthogonal decomposition strategy con-structs a low-rank subspace, which is formed by wavelet components and structured through tensor stacking. The low-rank property of this subspace ensures that the diffusion process requires only a few simple iterations to produce accurate prior information. Moreover, when the diffusion process is trans-ferred to this subspace, the focus shifts to learning the low-dimensional intrinsic features of the data, thereby enhancing the generalization ability of the diffusion model. Considering the strategy is approximately re-versible and incurs no information loss, it allows the diffusion process in different spaces to refine models through a mutual feedback mechanism, thereby enriching the prior information learning from multiple dimensions. Comprehensive experiments on different datasets clearly demonstrate that Sub-DM achieves faster convergence speed and exhibits more robust generalization ability.

## Requirements and Dependencies
    python==3.7.11
    Pytorch==1.7.0
    tensorflow==2.4.0
    torchvision==0.8.0
    tensorboard==2.7.0
    scipy==1.7.3
    numpy==1.19.5
    ninja==1.10.2
    matplotlib==3.5.1
    jax==0.2.26

## Training Demo
``` bash
python main.py --config=configs/subvp/cifar10_ncsnpp_continuous.py --workdir=exp --mode=train --eval_folder=result

```
## Test Demo
``` bash
python PCsampling_demo_parallel_svd_dwt_2model2.py
```

## Graphical representation
### The whole pipeline of GLDM is illustrated in Fig1
<div align="center"><img src="https://github.com/yqx7150/GLDM/blob/main/Fig1.png" >  </div>
The schematic of the proposed GLDM algorithm. Red and blue parts represent the training stage that fully encoded full-resolution reference data is constructed through a time-interleaved acquisition scheme. Red part merges all time frames to train the global model (GM) while the blue part merges local time frames to train the local model (LM). Green part represents the reconstruction stage which the structure of the reconstruction model exists in a cascade form and the under-sampled k-space data (16 frames) are sequentially input into the network. At the same time, optimization unit (OU) containing a LR operator and a DC term is introduced to better remove aliasing and restore details

### Time-interleaved acquisition scheme is visualized in Fig2.
<div align="center"><img src="https://github.com/yqx7150/GLDM/blob/main/Fig2.png" >  </div>
The core of the approach is to construct a complete k-space dataset by merging any number of adjacent time frames. In the above example, two different under-sampled patterns (uniform and random) at 5-fold acceleration are acquired via a time-interleaved acquisition scheme.

### The time-interleaved acquisition scheme of 4 frames of dynamic MRI is visualized in Fig3.
<div align="center"><img src="https://github.com/yqx7150/GLDM/blob/main/Fig3.png" >  </div>
The ACS of each frame remains unaltered, while the remainder of the area is filled with data from adjacent frames. The distinct colors rep-resent data contributions from different frames

###  Convergence curves of PSNR and MSE of GLDM and the number of iterations
<div align="center"><img src="https://github.com/yqx7150/GLDM/blob/main/Fig4.png" >  </div>
Convergence curves of PSNR and MSE of GLDM and the number of iterations

## Other Related Projects    
  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9435335)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP) [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  

* One-shot Generative Prior in Hankel-k-space for Parallel Imaging Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2208.07181)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HKGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)
