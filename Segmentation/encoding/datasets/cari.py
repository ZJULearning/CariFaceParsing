import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

import torch
from .base import BaseDataset



def swap_N(T, m, n): #Distinguish left & right
    T = np.asarray(T)
    #A = np.zeros(T.shape)
    A = T.copy()
    A[T==n] = m
    A[T==m] = n
    return Image.fromarray(np.uint8(A))

class CariSegmentation(BaseDataset):
    NUM_CLASS = 11
    def __init__(self, root='dataset/cari/', split='train', mode=None, transform=None,
                 target_transform=None):
        super(CariSegmentation, self).__init__(root, split, mode, transform, target_transform,base_size=256, crop_size=256)
        _mask_dir = os.path.join(root, 'label')
        _image_dir = os.path.join(root, 'image')
        # train/val/test splits are pre-cut
        #_splits_dir = os.path.join(_cityscapes_root, 'ImageSets/Segmentation')
        if self.mode == 'train':
            _split_f = os.path.join(root, 'train.txt')
        elif self.mode == 'val':
            _split_f = os.path.join(root, 'val.txt')
        elif self.mode == 'testval':
            _split_f = os.path.join(root, 'val.txt')
        elif self.mode == 'test':
            _split_f = os.path.join(root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        self.names = []
        self.crop_size_h = self.crop_size
        self.crop_size_w = self.crop_size
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, self.split + '/' + line.rstrip('\n'))
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.names.append(line.rstrip('\n'))
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, self.split + '/' + line.rstrip('\n')[:-3] + "png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def _val_sync_transform(self, img, mask):

        w, h = img.size
        oh = self.crop_size_h
        ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size_w) / 2.))
        y1 = int(round((h - self.crop_size_h) / 2.))
        img = img.crop((x1, y1, x1+self.crop_size_w, y1+self.crop_size_h))
        mask = mask.crop((x1, y1, x1+self.crop_size_w, y1+self.crop_size_h))
        # final transform
        return img, self._mask_transform(mask)


    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask = swap_N(mask, 2, 3)
            mask = swap_N(mask, 4, 5)
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = img.size
        oh = short_size
        ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if oh < self.crop_size_h:
            padh = self.crop_size_h - oh if oh < self.crop_size_h else 0
            padw = self.crop_size_w - ow if ow < self.crop_size_w else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size_w)
        y1 = random.randint(0, h - self.crop_size_h)
        img = img.crop((x1, y1, x1+self.crop_size_w, y1+self.crop_size_h))
        mask = mask.crop((x1, y1, x1+self.crop_size_w, y1+self.crop_size_h))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        return img, self._mask_transform(mask)


    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        img = img.resize((self.crop_size_w, self.crop_size_h), Image.BILINEAR)
        target = target.resize((self.crop_size_w, self.crop_size_h), Image.NEAREST)
        #if self.mode != 'testval':
        #    target = target.resize((self.crop_size_w, self.crop_size_h), Image.NEAREST)
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform( img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform( img, target)
        else:
            assert self.mode == 'testval'
            target = self._mask_transform(target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            #print("transform for input")
            img = self.transform(img)
        if self.target_transform is not None:
            #print("transform for label")
            target = self.target_transform(target)
        if self.mode == 'testval':
            return img, target, self.names[index]
        return img, target

    def __len__(self):
        return len(self.images)
