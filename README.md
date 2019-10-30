# Deep Learning Reading List

This repository is a compilation of all the important readings related to Deep Learning.

# Philosophy
- It won't be an exhaustive list
- This list would focus on the principles rather than the catching up with latest models. 


## Image Classification

Initially, this subfield is focusing on improving the accuracy on the imagenet-1000 challenge. 
Recently, the papers here focuses on two things:
- These models as a feature extractor to other area such as object detection, segmentation, GAN, etc.
- Making the models smaller without harming the accuracy

Gradient-Based Learning Applied to Document Recognition, IEEE 1998 [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

ImageNet Classification with Deep Convolutional Neural Networks, NIPS 2012 [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Very Deep Convolutional Neural Networks for Large-Scale Image Recognition, ICLR 2015 [paper](https://arxiv.org/pdf/1409.1556.pdf)   
Key Insight:  It's all about deep neural network architecture. This is one of the classic image classification architecture wherein many of our recent algorithm uses it as a backbone.

Deep Residual Learning for Image Recognition, CVPR 2016 [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)   
Key Insight: Going deep requires us to consider the vanishing gradient problem. One way to alleviate it is to use a residual block that harnesses skip connection to propagate the gradients to the lower layers.

Densely Connected Convolutional Network, CVPR 2017 [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, Arxiv 2017 [paper](https://arxiv.org/pdf/1704.04861.pdf)

Squeeze and Excitation Network, CVPR 2018 [web](http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)

FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction, NIPS 2018 [web](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction) [paper](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf)

## Neural Network Modules

Rectifier Nonlinearities Improve Neural Network Acoustic Models, ICML 2013 [paper](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)

Dropout: A Simple Way to Prevent Neural Networks from Overfitting, JMLR 2014 [paper](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)   
Key Insight:  It's all about the concept of regularization. On how we can use it to make our ML algorithm be more robust on generalizing on our test set.

Adam: A Method for Stochastic Optimization, ICLR 2015 [paper](https://arxiv.org/pdf/1412.6980.pdf)   
Key Insight:  It's about understanding the behaviours on how we optimize the weights of the network. We would understand the direction of the research on this field of optimization.


Batch Normalization, ICML 2015 [web](http://proceedings.mlr.press/v37/ioffe15.html) [paper](http://proceedings.mlr.press/v37/ioffe15.pdf)

Spatial Transformer Network, NIPS 2015 [web](https://papers.nips.cc/paper/5854-spatial-transformer-networks) [paper](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)

Deformable Convolutional Networks, ICCV 2017 [website](http://openaccess.thecvf.com/content_iccv_2017/html/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.html) [paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf)

Group Normalization, ECCV 2018 [paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf)

Deformable ConvNets v2: More Deformable, Better Results, CVPR 2019 [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.pdf)

## Object Detection

Region-based convolutional networks for accurate object detection and segmentation, TPAMI 2015

Spatial pyramid pooling in deep convolutional networks for visual recognition, TPAMI 2015 [paper](https://arxiv.org/pdf/1406.4729.pdf)

Fast rcnn, ICCV 2016 [paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

Faster r-cnn: Towards real-time object detection with region proposal networks, NIPS 2016 [paper](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)

SSD: single shot multibox detector, ECCV 2016 [paper](https://arxiv.org/pdf/1512.02325.pdf)

Speed Accuracy trade-offs for modern convolutional object detection network, CVPR 2017 [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_SpeedAccuracy_Trade-Offs_for_CVPR_2017_paper.pdf) 

Feature Pyramid Networks for Object Detection CVPR 2017 [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)   
Key Insight: Applying the principle of U-net for object detection

Focal Loss for Dense Object Detection, ICCV 2017 [paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

Cascade R-CNN: Delving Into High Quality Object Detection, ECCV 2018 [web](http://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html) [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf)

## Multi-Object Tracker

Tracking without bells and whistles, ICCV 2019 [paper](https://arxiv.org/pdf/1903.05625.pdf)

## Instance Segmentation

U-net: Convolutional networks for biomedical image segmentation,  International Conference on Medical image computing and computer-assisted intervention 2015 [paper](https://arxiv.org/pdf/1505.04597.pdf) [web](https://arxiv.org/abs/1611.09326)

Hybrid Task Cascade for instance segmentation, CVPR 2019 [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Hybrid_Task_Cascade_for_Instance_Segmentation_CVPR_2019_paper.pdf)

## Layer Visualization
Visualizing and Understanding Convolutional Networks, ECCV 2014 [paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)


## Sequence Models

On the Properties of Neural Machine Translation: Encoder–Decoder Approaches, Arxiv 2014 [paper](https://arxiv.org/pdf/1409.1259.pdf)

Empirical evaluation of gated recurrent neural networks on sequence modeling, Arxiv 2014 [paper](https://arxiv.org/pdf/1412.3555.pdf)

Sequence to Sequence Learning with Neural Networks, NIPS 2014 [website](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks) [paper](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

Neural Machine Translation by Jointly Learning to Align and Translate, ICLR 2015 [paper](https://arxiv.org/pdf/1409.0473.pdf) [slide](https://iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2015:bahdanau-iclr2015.pdf)

An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition, ICDAR 2015 [paper](https://arxiv.org/pdf/1507.05717.pdf)

## Generative Advserial Network
Generative Adversarial Nets, NIPS 2014 [paper](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

Unsupervised Representational Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016 [paper](https://arxiv.org/pdf/1511.06434.pdf)

Unpaired Image-To-Image Translation Using Cycle-Consistent Adversarial Networks, ICCV 2017 [web](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html) [paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)

Image-to-Image Translation with Conditional Adversarial Networks, CVPR 2017 [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)

High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs [website](https://tcwang0509.github.io/pix2pixHD/) [paper](https://arxiv.org/pdf/1711.11585.pdf)

## Motion Transfer
Everybody Dance Now, ICCV 2019 [website](https://carolineec.github.io/everybody_dance_now/) [paper](https://arxiv.org/pdf/1808.07371.pdf)
