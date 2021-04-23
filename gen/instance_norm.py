import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transform

NUM = 1449
DEP_PATH = './../dataset/NYUD_v2/nyu_depths/'
OUT_PATH = './../dataset/NYUD_v2/nyu_depths_norm/'

def instance_normalize_img(idx = 0, gl_norm = False):
    img = Image.open(DEP_PATH + ('/%s.png' % idx))
    # img.show()
    dep = transform.ToTensor()(img)
    std, mean = torch.std_mean(dep)
    dep_norm = transform.Normalize(mean=[mean], std=[std])(dep)
    dep_norm_img = transform.ToPILImage()(dep_norm)
    dep_norm_img.save(OUT_PATH + ('norm_%s.png' % idx)) 
    # print(std, mean)
    # print(dep, dep_norm, sep = '\n')
    if gl_norm:
        dep_gl_norm = transform.Normalize(mean=[0.2798], std=[0.1387])(dep)
        dep_gl_norm_img = transform.ToPILImage()(dep_norm)
        dep_gl_norm_img.save(OUT_PATH + ('gl_norm_%s.png' % idx)) 
        

if __name__ == '__main__':
    # print(os.listdir(DEP_PATH))
    if not os.path.isdir(DEP_PATH):
        print('Invalid Dep Data Path.')
        exit(0)
    if not os.path.isdir(OUT_PATH):
        os.makedirs(OUT_PATH)
    for i in range(NUM):
        instance_normalize_img(idx = i)
        if (i + 1) % 100 == 0:
            print('(%4d/%4d) done.' % ((i + 1), NUM))
    
    