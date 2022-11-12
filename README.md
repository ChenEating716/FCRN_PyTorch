# FCRN_PyTorch
PyTorch Implementation of paper "Deeper Depth Prediction with Fully Convolutional Residual Networks"



## Introduction

This is an original implementation of [Deeper Depth Prediction with Fully Convolutional Residual Networks](http://ieeexplore.ieee.org/document/7785097/) based on PyTorch (1.9.1). This paper addresses the problem of estimating the depth map of a scene given a single RGB image using a fully convolutional architecture.



## Tested Environment

- Ubuntu 20.04
- torch 1.9.1+cu111
- numpy 1.21.2



## Results

After 15 epochs of training:

<img src="pics/rgb_2.png" alt="rgb_2" style="zoom:65%;" /> <img src="pics/gt_2.png" alt="pred_2" style="zoom:65%;" /><img src="pics/pred_2.png" alt="gt_2" style="zoom:65%;" />

------



<img src="pics/rgb_0.png" alt="rgb_0" style="zoom:65%;" /><img src="pics/gt_0.png" alt="gt_0" style="zoom:65%;" /><img src="pics/pred_0.png" alt="pred_0" style="zoom:65%;" />

------



<img src="pics/rgb_1.png" alt="rgb_1" style="zoom:65%;" /><img src="pics/gt_1.png" alt="pred_1" style="zoom:65%;" /><img src="pics/pred_1.png" alt="pred_1" style="zoom:65%;" />

------

**Complete Code will be uploaded in the near future.** 





