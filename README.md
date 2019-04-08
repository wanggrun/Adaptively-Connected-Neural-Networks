# Adaptively Connected Neural Networks
A re-implementation of our CVPR 2019 paper "Adaptively Connected Neural Networks"

[Guangrun Wang](https://wanggrun.github.io/) 

Sun Yat-sen University (SYSU)



![intro](https://github.com/wanggrun/Adaptively-Connected-Neural-Networks/blob/master/intro.jpg)




# Table of Contents
0. [Introduction](#introduction)
0. [ImageNet](#imagenet)
0. [Cora](#cora)
0. [Citation](#citation)

# Introduction

This repository contains the training & testing code on [ImageNet](http://image-net.org/challenges/LSVRC/2015/) and [Cora](http://linqs.cs.umd.edu/projects/projects/lbc/) by Adaptively Connected Neural Networks (ACNet). 


# ImageNet

+ Training and testing curve on ImageNet：



   ![curves](https://github.com/wanggrun/Adaptively-Connected-Neural-Networks/blob/master/error.jpg)
	   
	   
   

+ ImageNet accuracy and pretrained model:

| Model            | Top 5 Error | Top 1 Error | Download                                                                          |
|:-----------------|:------------|:-----------:|:---------------------------------------------------------------------------------:|
| ResNet50         | 6.9%       | 23.6%      | [:arrow_down:](http://models.tensorpack.com/ResNet/ImageNet-ResNet50.npz)         |
| ResNet50-ACNet   | 6.4%       | 22.5%      | [:arrow_down:](http://models.tensorpack.com/ResNet/ImageNet-ResNet50-SE.npz)      |
| ResNet50-ACNet-pixel-aware| 6.4%       | 22.5%      | [:arrow_down:](http://models.tensorpack.com/ResNet/ImageNet-ResNet101.npz)        |

### ImageNet

+ Training script:
```
cd pyramid/ImageNet/
python imagenet-resnet.py   --gpu 0,1,2,3,4,5,6,7   --data_format NHWC  -d 101  --mode resnet --data  [ROOT-OF-IMAGENET-DATASET]
```

+ Testing script:
```
cd pyramid/ImageNet/
python imagenet-resnet.py   --gpu 0,1,2,3,4,5,6,7  --load [ROOT-TO-LOAD-MODEL]  --data_format NHWC  -d 101  --mode resnet --data  [ROOT-OF-IMAGENET-DATASET] --eval
```


### Citation

If you use these models in your research, please cite:

	@inproceedings{yang2017learning,
            title={Learning feature pyramids for human pose estimation},
            author={Yang, Wei and Li, Shuang and Ouyang, Wanli and Li, Hongsheng and Wang, Xiaogang},
            booktitle={The IEEE International Conference on Computer Vision (ICCV)},
            volume={2},
            year={2017}
        }

### Dependencies
+ Python 2.7 or 3
+ TensorFlow >= 1.3.0
+ [Tensorpack](https://github.com/ppwwyyxx/tensorpack)
   The code depends on Yuxin Wu's Tensorpack. For convenience, we provide a stable version 'tensorpack-installed' in this repository. 
   ```
   # install tensorpack locally:
   cd tensorpack-installed
   python setup.py install --user
   ```
