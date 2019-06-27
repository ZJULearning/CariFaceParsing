import os.path
from data.base_dataset import BaseDataset, get_transform, get_target_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np
import pdb
import torch.nn.functional as F
import torchvision.transforms as transforms

class HelenDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.path = make_dataset(os.path.join(opt.dataroot))
        self.path = sorted(self.path)
        self.size = len(self.path)
        self.transform = get_transform(opt)
        self.target_transform = get_target_transform(opt)

    def channel_1toN(self, img, num_channel):
        transform1 = transforms.Compose([transforms.ToTensor(), ])
        img = (transform1(img) * 255.0).long()
        T = torch.LongTensor(num_channel, img.size(1), img.size(2)).zero_()
        mask = torch.LongTensor(img.size(1), img.size(2)).zero_()
        for i in range(num_channel):
            T[i] = T[i] + i
            layer = T[i] - img
            T[i] = torch.from_numpy(np.logical_not(np.logical_xor(layer.numpy(), mask.numpy())).astype(int))
        return T.float()

    def __getitem__(self, index):
        A_label = 0
        A_path = self.path[index]
        A_path_face = A_path.replace('landmark', 'images')
        A_path_face = A_path_face.replace('npy', 'jpg')

        A_path_parsing = A_path.replace('landmark', 'labels')
        A_path_parsing = A_path_parsing.replace('npy', 'png')

        A_img_face = Image.open(A_path_face).convert('RGB')
        A_img_parsing = Image.open(A_path_parsing)

        A = np.load(A_path)
        A = torch.from_numpy(A).float()
        A_face = self.transform(A_img_face)
        A_parsing = self.channel_1toN(A_img_parsing, 11)

        return {'A': A, 'A_face': A_face, 'A_label': A_label, 'A_parsing': A_parsing, 'A_path': A_path}

    def __len__(self):
        max_size = 0
        max_size = self.size
        return max_size

    def name(self):
        return 'HelenDataset'
