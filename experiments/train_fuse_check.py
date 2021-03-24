###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import copy
import yaml
import logging
import argparse
import numpy as np
from tqdm import tqdm
from addict import Dict

import torch
from torch.utils import data
from tensorboardX import SummaryWriter
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model
CONFIG_PATH = 'results/fusenet/config.yaml'

from torchviz import make_dot

def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(description='semantic segmentation using PASCAL VOC')
    parser.add_argument('--config_path', type=str, help='path of a config file')
    return parser.parse_args(['--config_path', CONFIG_PATH])
    #return parser.parse_args()




config = get_arguments()
# configuration
args = Dict(yaml.safe_load(open(config.config_path)))
args.cuda = (args.use_cuda and torch.cuda.is_available())
torch.manual_seed(args.seed)


args.batch_size = 2

# data transforms
input_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize([.485, .456, .406], [.229, .224, .225])])   # mean and std based on imageNet
dep_transform= transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=(0.2798,), std=(0.1387, ) )
]) # mean and std for depth

# dataset
data_kwargs = {'transform': input_transform,  'dep_transform': dep_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
trainset = get_dataset(args.dataset, split=args.train_split, mode='train',  **data_kwargs)
testset = get_dataset(args.dataset, split='val', mode='val', **data_kwargs)
# dataloader
args.batch_size = 2
kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                   drop_last=True, shuffle=True, **kwargs)
valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                 drop_last=False, shuffle=False, **kwargs)
nclass = trainset.num_class
# model
model = get_segmentation_model(args.model, dataset=args.dataset)
print(model)
# optimizer using different LR
params_list = [{'params': model.parameters(), 'lr': args.lr}, ]

optimizer = torch.optim.SGD(params_list, lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
# criterions
criterion = SegmentationLosses(se_loss=args.se_loss,
                                    aux=args.aux,
                                    nclass=nclass,
                                    se_weight=args.se_weight,
                                    aux_weight=args.aux_weight)

# clear start epoch if fine-tuning
if args.ft:
    args.start_epoch = 0
# lr scheduler
scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr,
                                         args.epochs, len(trainloader))
best_pred = 0.0


train_loss = 0.0
model.train()
for i, (image, dep, target) in enumerate(trainloader):
    print(i)
    break

# visialize the graph

yhat = model(image, dep)
from torchviz import make_dot

make_dot(yhat, params=dict(list(model.named_parameters()))).render("fuseNet_torchviz", format="png")


i, epoch = 0, 1
scheduler(optimizer, i, epoch, best_pred)
optimizer.zero_grad()
outputs = model(image, dep)

loss = criterion(*outputs, target)

loss.backward()
optimizer.step()
