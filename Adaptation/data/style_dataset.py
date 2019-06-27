import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np

class StyleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.style_type = opt.num_style
        self.dir_content = os.path.join(opt.dataroot, opt.phase + '_content')
        self.dir_style = os.path.join(opt.dataroot, opt.phase + '_style')
        self.dir_val = os.path.join(opt.dataroot, 'val')

        self.content_paths = make_dataset(self.dir_content)
        self.style_paths = make_dataset(self.dir_style)
        self.val_paths = make_dataset(self.dir_val)



        self.content_paths = sorted(self.content_paths)
        self.style_paths = sorted(self.style_paths)
        self.val_paths = sorted(self.val_paths)

        self.content_size = len(self.content_paths)
        self.style_size = len(self.style_paths)
        self.val_size = len(self.val_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):

        if self.opt.serial_batches:
            index_content = index % self.content_size
        else:
            index_content = random.randint(0, self.content_size - 1)
        content_path = self.content_paths[index_content % self.content_size]


        tt = random.randint(1, self.style_type)
        style_path = self.style_paths[tt-1]
        style_label = tt - 1


        if self.opt.serial_batches:
            index_val = index % self.val_size
        else:
            index_val = random.randint(0, self.val_size - 1)
        val_path = self.val_paths[index_val]

        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')

        val_img = Image.open(val_path).convert('RGB')

        content_img = self.transform(content_img)
        style_img = self.transform(style_img)

        val = self.transform(val_img)

        return {'content': content_img, 'style': style_img, 'style_label': style_label,'val': val, 'content_path': content_path}

    def __len__(self):
        return max(self.content_size, self.style_size)

    def name(self):
        return 'StyleDataset'
