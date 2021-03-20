# (Linux-Codebase)

## Apply early fusion
Early fusion allows you to feed 4-channel images as input, whereas you'll feed rgb images as input when dep_dim is set to be **False**. To apply early fusion, please modify ./results/danet_resnet50/config.yaml and set:
```
dep_dim = True
```

## Switching config files
train_danet.py switches its config file on different platforms.

If you test your code on Mac (i.e. Darwin platform), then please modify this line of code: 
```
CONFIG_PATH_MAC = './results/danet_resnet50/config_mac.yaml'
```
Otherwise, you can directly pass the config path as a cmd argument while running on the HPC (i.e. Linux platform).

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

## Run on HPC
```
sbatch train_xxx.sh [YOUR_NETID] [ENV_NAME]
```
## Possible Problems
1. Did not deactivate the env before `sbatch`. Possible error:
```
Traceback (most recent call last):
  File "experiments/danet/train_danet.py", line 14, in <module>
    from addict import Dict
ModuleNotFoundError: No module named 'addict'
```