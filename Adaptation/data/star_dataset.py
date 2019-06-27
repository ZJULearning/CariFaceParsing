import os.path
from data.base_dataset import BaseDataset, get_transform, get_target_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np
import pdb
import torch.nn.functional as F


class StarDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.all_attr_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        self.attr2idx = {}
        self.idx2attr = {}
        for i, attr_name in enumerate(self.all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        self.paths = {}
        self.sizes = {}
        self.dir_val = os.path.join(opt.dataroot, 'val')
        self.val_path = make_dataset(self.dir_val)
        self.val_paths = sorted(self.val_path)
        self.val_size = len(self.val_path)
        for i, attr_name in enumerate(self.all_attr_names):
            self.paths[attr_name] = make_dataset(os.path.join(opt.dataroot, opt.phase + attr_name))
            self.paths[attr_name] = sorted(self.paths[attr_name])
            self.sizes[attr_name] = len(self.paths[attr_name])

        self.transform = get_transform(opt)
        self.target_transform = get_target_transform(opt)

    def __getitem__(self, index):
        path = {}
        face_path = {}
        for i, attr_name in enumerate(self.all_attr_names):
            if self.opt.serial_batches:
                attr_index = index % self.sizes[attr_name]
            else:
                attr_index = random.randint(0, self.sizes[attr_name] - 1)

            path[attr_name] = self.paths[attr_name][attr_index]
            face_path[attr_name] = path[attr_name].replace('npy', 'jpg')
            #face_path[attr_name] = face_path[attr_name].replace('landmark_webcaricature', 'face_webcaricature')
            face_path[attr_name] = face_path[attr_name].replace('landmark', 'face')


        if self.opt.serial_batches:
            index_val = index % self.val_size
        else:
            index_val = random.randint(0, self.val_size - 1)
        val_path = self.val_paths[index_val]
        val_path_face = val_path.replace('npy', 'jpg')
        #val_path_face = val_path_face.replace('landmark_webcaricature', 'face_webcaricature')
        val_path_face = val_path_face.replace('landmark', 'face')

        ret_index = random.randint(1,len(self.all_attr_names))
        attr_name = self.all_attr_names[ret_index-1]
        A_label = ret_index-1
        A_path = path[attr_name]
        A_path_face = face_path[attr_name]

        A_img_face = Image.open(A_path_face).convert('RGB')
        val_img_face = Image.open(val_path_face).convert('RGB')

        A = np.load(A_path)
        A = torch.from_numpy(A).float()
        A_face = self.transform(A_img_face)

        val = np.load(val_path)
        val = torch.from_numpy(val).float()
        val_face = self.transform(val_img_face)
        val_label = 0

        return {'A': A, 'A_face': A_face, 'A_label': A_label, 'val': val, 'val_face': val_face, 'val_label': val_label, 'A_path': A_path, 'val_path': val_path}

    def __len__(self):
        max_size = 0
        for i, attr_name in enumerate(self.all_attr_names):
            max_size = max(self.sizes[attr_name], max_size)
        return max_size

    def name(self):
        return 'StarDataset'
