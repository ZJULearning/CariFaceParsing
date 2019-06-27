import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import json

import torch
from .base import BaseDataset

class CocostuffSegmentation(BaseDataset):
    NUM_CLASS = 80 + 36 + 17
    BASE_DIR = 'COCO'
    def __init__(self, root=os.path.expanduser('/tmp5/yunshen_data'), split='train', mode=None, transform=None,
                 target_transform=None):
        super(CocostuffSegmentation, self).__init__(root, split, mode, transform, target_transform)
        _cocostuff_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_cocostuff_root, 'annotations')
        _image_dir = os.path.join(_cocostuff_root, 'images')
        if self.mode == 'train':
            _split_f = os.path.join(_mask_dir, 'panoptic_train2017.json')
        elif self.mode == 'val':
            _split_f = os.path.join(_mask_dir, 'panoptic_val2017.json')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(_split_f, "r") as f:
            gt_json = json.load(f)
        file_names = [el['file_name'] for el in gt_json['images']]
        if self.mode == 'train':
            self.images = [_image_dir+"/train2017/"+el['file_name'] for el in gt_json['images'][:2003]]
            self.masks = [_mask_dir+"/panoptic_train_semantic_stuff2017/"+el['file_name'][:-3]+"png" for el in gt_json['images'][:2003]]
        elif self.mode == 'val':
            self.images = [_image_dir+"/val2017/"+el['file_name'] for el in gt_json['images'][:2003]]
            self.masks = [_mask_dir+"/panoptic_val_semantic_stuff2017/"+el['file_name'][:-3]+"png" for el in gt_json['images'][:2003]]
        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        #img = img.resize((self.crop_size_w, self.crop_size_h), Image.BILINEAR)
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
        return img, target

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')

        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)
