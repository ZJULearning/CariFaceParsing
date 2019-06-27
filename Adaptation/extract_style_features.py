import torch
import itertools
import numpy as np
import torch.nn.functional as F
import random
import pdb
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import copy
import torch.optim as optim
# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

class BNMatching(nn.Module):
    # A style loss by aligning the BN statistics (mean and standard deviation)
    # of two feature maps between two images. Details can be found in
    # https://arxiv.org/abs/1701.01036
    def __init__(self, target, name):
        super(BNMatching, self).__init__()
        self.mu_target = self.FeatureMean(target).detach()
        self.std_target = self.FeatureStd(target).detach()
        self.name = name

    def FeatureMean(self,input):
        b,c,h,w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        return torch.mean(f,dim=2)
    def FeatureStd(self,input):
        b,c,h,w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        return torch.std(f, dim=2)
    def forward(self,input):
        # input: 1 x c x H x W
        mu_input = self.FeatureMean(input)
        std_input = self.FeatureStd(input)
        self.loss = F.mse_loss(mu_input,self.mu_target) + F.mse_loss(std_input,self.std_target)
        return input

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature, name):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.name = name

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def BNFeature(input):
    # A style loss by aligning the BN statistics (mean and standard deviation)
    # of two feature maps between two images. Details can be found in
    # https://arxiv.org/abs/1701.01036
    b, c, h, w = input.size()
    f = input.view(b, c, h * w)  # bxcx(hxw)
    input_mean = torch.mean(f, dim=2)
    input_std = torch.std(f, dim=2)
    return input_mean, input_std

cnn = models.vgg19(pretrained=True).features.cuda().eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).cuda()
normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).cuda()
f = open('datasets/crop_faces_align_frontal/caricature.list')
for image in f.readlines():
    image = image.strip()
    print(image)
    image_path = 'datasets/crop_faces_align_frontal/caricature/' + image
    style = Image.open(image_path)
    style = transforms.ToTensor()(style)
    style = style.unsqueeze(0)
    style = style.cuda()
    content_losses = []
    style_losses = []
    style_features1 = np.array([])
    style_features2 = np.array([])
    model = nn.Sequential(normalization)
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name in style_layers_default:
            # add style loss:
            target_feature = model((style + 1) / 2.0).detach()
            style_feature1, style_feature2 = BNFeature(target_feature)
            style_features1 = np.concatenate((style_features1, style_feature1[0].cpu().numpy()), axis=0)
            style_features2 = np.concatenate((style_features2, style_feature2[0].cpu().numpy()), axis=0)
    np.save('stylefeature1/' + image[:-3] + 'npy', np.array(style_features1))
    np.save('stylefeature2/' + image[:-3] + 'npy', np.array(style_features2))



