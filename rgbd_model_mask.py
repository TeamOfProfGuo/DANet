
import os
import copy
import yaml
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from addict import Dict
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transform

from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model

GPUS = [0, 1]
MODEL_PATH = './psp_ef.pth'
CONFIG_PATH = './results/psp_resnet50/config.yaml'
OUTPUT_PATH = './pred_masks/'

class Mask():
    def __init__(self, args):
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),  # convert RGB [0,255] to FloatTensor in range [0, 1]
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])  # mean and std based on imageNet
        dep_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.2798], std=[0.1387])  # mean and std for depth
        ])
        # dataset
        data_kwargs = {'transform': input_transform, 'dep_transform': dep_transform,
                       'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        testset = get_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} if (args.cuda and torch.cuda.is_available()) else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class

        self.use_cuda = False # args.cuda and torch.cuda.is_available()

        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux=args.aux,
                                       se_loss=args.se_loss,  # norm_layer=SyncBatchNorm,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       dep_dim=args.dep_dim)
        
        model.load_state_dict(torch.load(MODEL_PATH), strict=False)

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        # using cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(),
                      "GPUs!")  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                model = nn.DataParallel(model, device_ids=GPUS)
        self.model = model.to(self.device) if self.use_cuda else model
        self.colors = self.generate_colors()

    def generate_colors(self):
        colors = []
        t = 255 * 0.2
        for i in range(1, 5):
            for j in range(1, 5):
                for k in range(1, 5):
                    colors.append(np.array([t * i, t * j, t * k], dtype=np.uint8))
        while len(colors) <= 256:
            colors.append(np.array([0, 0, 0], dtype=np.uint8))
        return colors

    def mask_to_rgb(self, t):
        assert len(t.shape) == 2
        t = t.numpy().astype(np.uint8)
        rgb = np.zeros((t.shape[0], t.shape[1], 3), dtype=np.uint8)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                rgb[i, j, :] = self.colors[t[i, j]]
        return rgb # Image.fromarray(rgb)
    
    def denormalize(self, input_image, mean, std, imtype=np.uint8):
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor): # if it's torch.Tensor, then convert
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            for i in range(len(mean)): # denormalize
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            image_numpy = image_numpy * 255 # [0,1] to [0,255]
            image_numpy = np.transpose(image_numpy, (1, 2, 0))  # chw to hwc
        else:
            image_numpy = input_image
        return image_numpy.astype(imtype)

    def generate_masks(self, dataset = 'val'):
        print('=' * 20, 'Generating Masks for RGB-D', '=' * 20)
        self.model.eval()
        total_img_count = 0
        for i, (image, dep, target) in enumerate(self.valloader):
            image_with_dep = torch.cat((image, dep), 1)
            if self.use_cuda:
                image_with_dep, target = image_with_dep.to(self.device), target.to(self.device)
            pred = torch.argmax(self.model(image_with_dep)[0], dim = 1)
            # pickle.dump({'img': image, 'dep': dep, 'target': target, 'pred': pred}, open('./batch1.dat', 'wb'))
            for j in range(self.args.batch_size):
                img = self.denormalize(image[j], mean=[.485, .456, .406], std=[.229, .224, .225])
                dep_size = dep[j].size()
                depth = self.denormalize(dep[j].expand(3, dep_size[1], dep_size[2]), mean=[0.2798], std=[0.1387])
                part1 = np.concatenate((img, depth), axis = 1)

                mask_gt = self.mask_to_rgb(target[j])
                mask_pred = self.mask_to_rgb(pred[j])
                part2 = np.concatenate((mask_gt, mask_pred), axis = 1)

                res = np.concatenate((part1, part2), axis = 0)
                res_img = Image.fromarray(res)
                res_img.save(OUTPUT_PATH + str(total_img_count) + '.png')

                # mask_gt.save(OUTPUT_PATH + str(total_img_count) + '_mask_gt.png')
                # mask_pred.save(OUTPUT_PATH + str(total_img_count) + '_mask_pred.png')
                total_img_count += 1

            print('Batch %02d done' % (i+1))
            if i == 2:
                break
            # if (i+1) % 5 == 0:
            #     print('Batch %02d done' % (i+1))

        print('=' * 68)

if __name__ == "__main__":

    args = Dict(yaml.safe_load(open(CONFIG_PATH)))
    args.cuda = (args.use_cuda and torch.cuda.is_available())
    args.resume = None if args.resume=='None' else args.resume
    torch.manual_seed(args.seed)

    gen_masks = Mask(args)
    gen_masks.generate_masks()