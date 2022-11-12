# FCRN_PyTorch
PyTorch Implementation of paper "Deeper Depth Prediction with Fully Convolutional Residual Networks"



## Introduction

This is an original implementation of [Deeper Depth Prediction with Fully Convolutional Residual Networks](http://ieeexplore.ieee.org/document/7785097/) based on PyTorch (1.12.0). This paper addresses the problem of estimating the depth map of a scene given a single RGB image using a fully convolutional architecture.



## Tested Environment

- Ubuntu 20.04
- torch 1.9.1+cu111
- numpy 1.21.2



## Results

After 15 epochs of training:

![rgb_2](pics/rgb_2.png) ![pred_2](pics/gt_2.png)![gt_2](pics/pred_2.png)

------



![rgb_0](pics/rgb_0.png)![gt_0](pics/gt_0.png)![pred_0](pics/pred_0.png)

------



![rgb_1](pics/rgb_1.png)![pred_1](pics/gt_1.png)![pred_1](pics/pred_1.png)

------

**Complete Code will be uploaded in the near future.** 





