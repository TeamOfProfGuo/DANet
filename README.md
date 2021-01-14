# (Linux-Codebase)

## Setup
load the hpc modules
```
module load anaconda3
module load cuda/10.0
module load gcc/7.3 
```
create a new environment named `dl` and install `pytorch`
```
conda create -n dl python=3.6
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```
clone this branch and redirect into the folder
```
git clone -b linux-codebase https://github.com/TeamOfProfGuo/DANet.git
cd DANet
```
There are a few unfixed errors when installing pytorch-encoding, so we directly copy the folder into the python directory. Note that if you make any change in the encoding folder, please recopy it to the python library. Replace the `[YOUR_NETID]` with your netID.
```
cp encoding -r /gpfsnyu/home/[YOUR_NETID]/.conda/envs/dl/lib/python3.6/site-packages/encoding/
```
Test with the below code and there should not be any error:
```
python
>>> import encoding
```
Install other packages from `requirements.txt`
```
pip install -r requirements.txt
```
Download PASCAL VOC data
- Download and extract 
[PASCAL VOC training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 
(2GB tar file), specifying the location with the `./datasets/VOCdevkit`.  
- Download and extract 
[augmented segmentation data](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) 
(Thanks to DrSleep), specifying the location with `./datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/`).  
## Introduction

We propose a Dual Attention Network (DANet) to adaptively integrate local features with their global dependencies based on the self-attention mechanism. And we achieve new state-of-the-art segmentation performance on three challenging scene segmentation datasets, i.e., Cityscapes, PASCAL Context and COCO Stuff-10k dataset.

![image](img/overview.png)

## Cityscapes testing set result

We train our DANet-101 with only fine annotated data and submit our test results to the official evaluation server.

![image](img/tab3.png)

## Updates

<font color="#dd0000">**2020/9**：</font>**Renew the code**, which supports **Pytorch 1.4.0** or later!

2020/8：The new TNNLS version DRANet achieves [**82.9%**](https://www.cityscapes-dataset.com/method-details/?submissionID=4792) on Cityscapes test set (submit the result on August, 2019), which is a new state-of-the-arts performance with only using fine annotated dataset and Resnet-101. The code will be released in [DRANet](<https://github.com/junfu1115/DRAN>).

2020/7：DANet is supported on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/danet), in which DANet achieves **80.47%** with single scale testing and **82.02%** with multi-scale testing on Cityscapes val set.

2018/9：DANet released. The trained model with ResNet101 achieves 81.5% on Cityscapes test set.

## Usage

1. Install pytorch 

   - The code is tested on python3.6 and torch 1.4.0.
   - The code is modified from [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). 

2. Clone the resposity

   ```shell
   git clone https://github.com/junfu1115/DANet.git 
   cd DANet 
   python setup.py install
   ```

3. Dataset
   - Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py). 
   - Please put dataset in folder `./datasets`

4. Evaluation for DANet

   - Download trained model [DANet101](https://drive.google.com/open?id=1XmpFEF-tbPH0Rmv4eKRxYJngr3pTbj6p) and put it in folder `./experiments/segmentation/models/`

   - `cd ./experiments/segmentation/`

   - For single scale testing, please run:

   - ```shell
     CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset citys --model danet --backbone resnet101 --resume  models/DANet101.pth.tar --eval --base-size 2048 --crop-size 768 --workers 1 --multi-grid --multi-dilation 4 8 16 --os 8 --aux --no-deepstem
     ```

   - Evaluation Result

     The expected scores will show as follows: DANet101 on cityscapes val set (mIoU/pAcc): **79.93/95.97**(ss) 

5. Evaluation for DRANet

   - Download trained model [DRANet101](https://drive.google.com/file/d/1xCl2N0b0rVFH4y30HCGfy7RY3-ars7Ce/view?usp=sharing) and put it in folder `./experiments/segmentation/models/`

   - Evaluation code is in folder `./experiments/segmentation/`

   - `cd ./experiments/segmentation/`

   - For single scale testing, please run:

   - ```shell
     CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset citys --model dran --backbone resnet101 --resume  models/dran101.pth.tar --eval --base-size 2048 --crop-size 768 --workers 1 --multi-grid --multi-dilation 4 8 16 --os 8 --aux
     ```

   - Evaluation Result

     The expected scores will show as follows: DRANet101 on cityscapes val set (mIoU/pAcc): **81.63/96.62** (ss) 

## Citation

if you find DANet and DRANet useful in your research, please consider citing:

```
@article{fu2020scene,
  title={Scene Segmentation With Dual Relation-Aware Attention Network},
  author={Fu, Jun and Liu, Jing and Jiang, Jie and Li, Yong and Bao, Yongjun and Lu, Hanqing},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  publisher={IEEE}
}
```

```
@inproceedings{fu2019dual,
  title={Dual attention network for scene segmentation},
  author={Fu, Jun and Liu, Jing and Tian, Haijie and Li, Yong and Bao, Yongjun and Fang, Zhiwei and Lu, Hanqing},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3146--3154},
  year={2019}
}
```



## Acknowledgement

Thanks [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), especially the Synchronized BN!
