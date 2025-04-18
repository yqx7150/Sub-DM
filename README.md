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
### The stacked formulation in subspace in Fig.1
<div align="center"><img src="https://github.com/yqx7150/Sub-DM/blob/main/Fig.1.png" >  </div>
Low-rank property of wavelet components and the corresponding stacked formulation in subspace.

### An overview of Sub-DM based on subspace low-rank learning. in Fig2.
<div align="center"><img src="https://github.com/yqx7150/Sub-DM/blob/main/Fig.2.png" >  </div>
An overview of Sub-DM based on subspace low-rank learning. In the training phase, k-space data undergoes diffusion migration across two distinct spaces. The original k-space data diffuses in the full space, while the orthogonally decomposed k-space data components diffuse in the subspace. During the reconstruction phase, the dimension of the under-sampled k-space data is dynamically changed by orthogonal wavelet decomposition and iteratively reconstructed in various diffusion spaces. Upon completion of the iterations, an optimization module is integrated to enhance the sampling quality.

### Convergence analysis of different models in Fig3.
<div align="center"><img src="https://github.com/yqx7150/Sub-DM/blob/main/Fig.3.png" >  </div>
Convergence analysis of HKGM, WKGM, HFS-SDE, CSGM-MRI, Score-MRI and Sub-DM in terms of PSNR versus the iteration steps for brain image reconstruction at R=8 under Poisson sampling.

###  Experimental results of knee in Fig4.
<div align="center"><img src="https://github.com/yqx7150/Sub-DM/blob/main/Fig.4.png" >  </div>
Experimental results of different methods in terms of PSNR and SSIM in out-of-distribution reconstruction tasks with 2D Random and Poisson mask.

