import os
import random
import scipy.io
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from .base import BaseDataset


class NYUD_in(BaseDataset):
    classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
            'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes',
            'ceiling','books','refridgerator','television','paper','towel','shower curtain','box','whiteboard','person',
            'night stand','toilet','sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

    NUM_CLASS = 40
    TRAIN_BASE_DIR = '../dataset/NYUD_v2/'

    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, dep_transform=None, target_transform=None, **kwargs):
        self.dep_transform = dep_transform
        print('==check dep_transform {}'.format(dep_transform))
        super(NYUD_in, self).__init__(root, split, mode, transform, target_transform, **kwargs)

        # train/val/test splits are pre-cut
        _nyu_root = os.path.abspath(self.TRAIN_BASE_DIR)
        _mask_dir = os.path.join(_nyu_root, 'nyu_labels40')
        _image_dir = os.path.join(_nyu_root, 'nyu_images')
        _depth_dir = os.path.join(_nyu_root, 'nyu_depths_norm')
        if self.mode == 'train':
            _split_f = os.path.join(_nyu_root, 'splits/train.txt')
        else:
            _split_f = os.path.join(_nyu_root, 'splits/test.txt')
        self.images = []  # list of file names
        self.depths = []  # list of depth image
        self.masks = []   # list of file names
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                line = int(line.rstrip('\n')) - 1
                _image = os.path.join(_image_dir, str(line) + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                _depth = os.path.join(_depth_dir, 'norm_%s.png' % line)
                assert os.path.isfile(_depth)
                self.depths.append(_depth)
                _mask = os.path.join(_mask_dir, str(line) + ".png")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _dep = Image.open(self.depths[index])  # depth image with shape [h ,w]
        if self.mode == 'test':   # return image(tensor), depth(tensor) and (fileName)
            if self.transform is not None:
                _img = self.transform(_img)
            if self.dep_transform is not None:
                _dep = self.dep_transform(_dep)
            return _img, _dep, os.path.basename(self.images[index])

        _target = Image.open(self.masks[index])  # image with shape [h, w]
        # synchrosized transform
        if self.mode == 'train':
            # return _img (Image), _dep (Image), _target (2D tensor)
            _img, _dep, _target = self._sync_transform(_img, _target, depth=_dep, IGNORE_LABEL=0)
        elif self.mode == 'val':
            _img, _dep, _target = self._val_sync_transform(_img, _target, depth = _dep)

        _target -= 1  # since 0 represent the boundary
        # general resize, normalize and toTensor
        if self.transform is not None:
            _img = self.transform(_img)  #_img to tensor, normalize
        if self.dep_transform is not None:
            _dep = self.dep_transform(_dep)  # depth to tensor, normalize
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _dep, _target  # all tensors

    def _load_mat(self, filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True,
                               struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.images)

    def make_pred(self, x):
        return x


