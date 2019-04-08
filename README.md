# Adaptively Connected Neural Networks
A re-implementation of our CVPR 2019 paper "Adaptively Connected Neural Networks"

[Guangrun Wang](https://wanggrun.github.io/) 

Sun Yat-sen University (SYSU)



![intro](https://github.com/wanggrun/Adaptively-Connected-Neural-Networks/blob/master/intro.jpg)




### Table of Contents
0. [Introduction](#introduction)
0. [ImageNet](#imagenet)
0. [Cora](#cora)
0. [Citation](#citation)

### Introduction

This repository contains the training & testing code on [ImageNet](http://image-net.org/challenges/LSVRC/2015/) and [Cora](http://linqs.cs.umd.edu/projects/projects/lbc/) by Adaptively Connected Neural Networks (ACNet). 


### Results

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

+ Trained Models:
   
   ResNet101:

   [Baidu Pan](https://pan.baidu.com/s/1SKEmrjcYA-NR9oFBOD7Y2w), code: 269o

   [Google Drive](https://drive.google.com/drive/folders/1pVSCQ6gap0b73FFr8bF-5p5Am6e2rXRr?usp=sharing)

   ResNet50:

   [Baidu Pan](https://pan.baidu.com/s/1ADYUt0QL1Vq42uqz75-W0A), code: zvgd

   [Google Drive](https://drive.google.com/drive/folders/1zcwLZVFdm8PONL_R6_8TNSLvb1vs6Lh7?usp=sharing)


### PASCAL VOC2012

+ Training script:
```
# Use the ImageNet classification model as pretrained model.
# Because ImageNet has 1,000 categories while voc only has 21 categories, 
# we must first fix all the parameters except the last layer including 21 channels. We only train the last layer for adaption
# by adding: "with freeze_variables(stop_gradient=True, skip_collection=True): " in Line 206 of resnet_model_voc_aspp.py
# Then we finetune all the parameters.
# For evaluation on voc val set, the model is first trained on COCO, then on train_aug of voc. 
# For evaluation on voc leaderboard (test set), the above model is further trained on voc val.
# it achieves 81.0% on voc leaderboard.
# a training script example is as follows.
cd pyramid/VOC/
python resnet-msc-voc-aspp.py   --gpu 0,1,2,3,4,5,6,7  --load [ROOT-TO-LOAD-MODEL]  --data_format NHWC  -d 101  --mode resnet --log_dir [ROOT-TO-SAVE-MODEL]  --data [ROOT-OF-TRAINING-DATA]
```

+ Testing script:
```
cd pyramid/VOC/
python gr_test_pad_crf_msc_flip.py 
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
