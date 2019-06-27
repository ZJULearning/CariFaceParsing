import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import DAENet
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import random
from torch.nn import CrossEntropyLoss
import pdb
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import copy
import torch.optim as optim
import pdb
import sys
import pdb
# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
style_weight=500
content_weight=100

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




def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


class StyleAdaptationModel(BaseModel):
    def name(self):
        return 'StyleAdaptationModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default DeformingGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # Model configurations.
        self.c_dim = opt.num_style
        self.image_size = 256
        self.g_conv_dim = 256
        self.d_conv_dim = 32
        self.g_repeat_num = 5
        self.d_repeat_num = 5
        self.lambda_cls = 1
        self.lambda_rec = 10
        self.lambda_gp = 10

        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.n_critic = 1
        self.CLSloss = CrossEntropyLoss()
        self.BNMatching = True
        #self.BNMatching = False

        # specify the training losses you want to print out. The program will call base_model.get_current_losses

        self.loss_names = ['content', 'style']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['content', 'style', 'translated_content']
        visual_names_val = ['val', 'val_translated']
        visual_names_test = ['val']
        for i in range(opt.num_style):
            visual_names_test.append('val_translated_' + str(i))

        visual_names_val = ['val', 'val_translated']
        if opt.phase == 'train':
            self.visual_names = visual_names_A + visual_names_val
        else:
            self.visual_names = []
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G']

        # load/define networks
        #self.netG = networks.Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_6blocks_unit', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.num_style)

        # initialize optimizers

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.netG.to(self.device)
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
            self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
            self.normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(self.device)




    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

    def set_input(self, input, epoch):
        self.image_paths = input['content_path']
        #self.scheduler(self.optimizer_G, 0, epoch, 0)
        #self.scheduler(self.optimizer_D, 0, epoch, 0)

        #if epoch == 2:
        #    pdb.set_trace()
        content = input['content']
        content = content.to(self.device)  # Input landmarks.
        self.content = content

        style = input['style']
        label_trg = input['style_label']
        c_trg = self.label2onehot(label_trg, self.c_dim)
        style = style.to(self.device)  # Input landmarks.
        self.c_trg = c_trg.to(self.device)  # Target domain labels.
        self.label_trg = label_trg.to(self.device)  # Labels for computing classification loss.
        self.style = style

        val = input['val']
        self.val = val.to(self.device)



    def optimize_parameters(self, epoch):
        self.optimizer_G.zero_grad()
        self.netG.train()
        self.translated_content = self.netG(self.content, self.c_trg)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []
        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(self.normalization)
        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers_default:
                # add content loss:
                target = model((self.content+1)/2.0).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers_default:
                # add style loss:
                target_feature = model((self.style+1)/2.0).detach()
                if self.BNMatching:
                    style_loss = BNMatching(target_feature, "style_loss_{}".format(i))
                else:
                    style_loss = StyleLoss(target_feature, "style_loss_{}".format(i))
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], BNMatching):
                break

        model = model[:(i + 1)]

        # correct the values of updated input image
        #self.translated_content.data.clamp_(0, 1)
        model((self.translated_content+1)/2.0)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        self.loss_content = content_score
        self.loss_style = style_score
        loss = style_score + content_score
        loss.backward()




        self.optimizer_G.step()
        self.netG.eval()
        with torch.no_grad():
            if epoch == 800:
                #pdb.set_trace()
                #state_dict = torch.load('checkpoints/face_style_gan/latest_net_G.pth')
                #self.netG.load_state_dict(state_dict)
                transform_list = []
                osize = [256, 256]
                transform_list.append(transforms.Resize(osize, Image.BICUBIC))
                transform_list.append(transforms.RandomCrop(256))
                transform_list += [transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5))]
                val_transform = transforms.Compose(transform_list)
                print('start testing!!!')
                val_list = open('datasets/style_faces/sub_val.list')
                val_path = 'datasets/style_faces/val/'
                for val_name in val_list.readlines():
                    val_name = val_name.strip()
                    val_path_face = val_path + val_name
                    val_img_face = Image.open(val_path_face).convert('RGB')
                    val_face = val_transform(val_img_face)
                    val_face = val_face.unsqueeze(0)
                    val_face = val_face.to(self.device)
                    m = nn.ReflectionPad2d(32)
                    val_face = m(val_face)

                    val_translated_A = self.netG(val_face, self.c_trg_1)
                    val_translated_A = val_translated_A[:,:,32:32+256,32:32+256]
                    image_path = 'results/face_style_gan_new/' + val_name[:-4] + '_' + 'translated_A.png'
                    image_numpy = val_translated_A.data[0].cpu().float().numpy()
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                    image_numpy = image_numpy.astype(np.uint8)
                    image_pil = Image.fromarray(image_numpy)
                    image_pil.save(image_path)

                    val_translated_B = self.netG(val_face, self.c_trg_2)
                    val_translated_B = val_translated_B[:,:,32:32+256,32:32+256]
                    image_path = 'results/face_style_gan_new/' + val_name[:-4] + '_' + 'translated_B.png'
                    image_numpy = val_translated_B.data[0].cpu().float().numpy()
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                    image_numpy = image_numpy.astype(np.uint8)
                    image_pil = Image.fromarray(image_numpy)
                    image_pil.save(image_path)

                    val_translated_C = self.netG(val_face, self.c_trg_3)
                    val_translated_C = val_translated_C[:,:,32:32+256,32:32+256]
                    image_path = 'results/face_style_gan_new/' + val_name[:-4] + '_' + 'translated_C.png'
                    image_numpy = val_translated_C.data[0].cpu().float().numpy()
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                    image_numpy = image_numpy.astype(np.uint8)
                    image_pil = Image.fromarray(image_numpy)
                    image_pil.save(image_path)

                    val_translated_D = self.netG(val_face, self.c_trg_4)
                    val_translated_D = val_translated_D[:,:,32:32+256,32:32+256]
                    image_path = 'results/face_style_gan_new/' + val_name[:-4] + '_' + 'translated_D.png'
                    image_numpy = val_translated_D.data[0].cpu().float().numpy()
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                    image_numpy = image_numpy.astype(np.uint8)
                    image_pil = Image.fromarray(image_numpy)
                    image_pil.save(image_path)

                    val_translated_E = self.netG(val_face, self.c_trg_5)
                    val_translated_E = val_translated_E[:,:,32:32+256,32:32+256]
                    image_path = 'results/face_style_gan_new/' + val_name[:-4] + '_' + 'translated_E.png'
                    image_numpy = val_translated_E.data[0].cpu().float().numpy()
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                    image_numpy = image_numpy.astype(np.uint8)
                    image_pil = Image.fromarray(image_numpy)
                    image_pil.save(image_path)

                    val_translated_F = self.netG(val_face, self.c_trg_6)
                    val_translated_F = val_translated_F[:,:,32:32+256,32:32+256]
                    image_path = 'results/face_style_gan_new/' + val_name[:-4] + '_' + 'translated_F.png'
                    image_numpy = val_translated_F.data[0].cpu().float().numpy()
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                    image_numpy = image_numpy.astype(np.uint8)
                    image_pil = Image.fromarray(image_numpy)
                    image_pil.save(image_path)

                    val_translated_G = self.netG(val_face, self.c_trg_7)
                    val_translated_G = val_translated_G[:,:,32:32+256,32:32+256]
                    image_path = 'results/face_style_gan_new/' + val_name[:-4] + '_' + 'translated_G.png'
                    image_numpy = val_translated_G.data[0].cpu().float().numpy()
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                    image_numpy = image_numpy.astype(np.uint8)
                    image_pil = Image.fromarray(image_numpy)
                    image_pil.save(image_path)

                    val_translated_H = self.netG(val_face, self.c_trg_8)
                    val_translated_H = val_translated_H[:,:,32:32+256,32:32+256]
                    image_path = 'results/face_style_gan_new/' + val_name[:-4] + '_' + 'translated_H.png'
                    image_numpy = val_translated_H.data[0].cpu().float().numpy()
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                    image_numpy = image_numpy.astype(np.uint8)
                    image_pil = Image.fromarray(image_numpy)
                    image_pil.save(image_path)
                torch.save(self.netG.cpu().state_dict(), 'latest_net_G.pth')
                sys.exit()

            else:
                m = nn.ReflectionPad2d(32)
                self.val = m(self.val)
                self.val_translated = self.netG(self.val, self.c_trg)
                self.val_translated = self.val_translated[:,:,32:32+256,32:32+256]




    def forward(self):
        self.netG.eval()
        random_style = random.randint(1, self.c_dim) - 1
        m = nn.ReflectionPad2d(32)
        out = (torch.zeros(1) + random_style).long()
        c_trg = self.label2onehot(out, self.c_dim)
        c_trg = c_trg.to(self.device)
        self.content = m(self.content)
        translated_content = self.netG(self.content, c_trg)
        translated_content = translated_content[:, :, 32:32 + 256, 32:32 + 256]
        image_name = self.image_paths[0].split('/')[-1]
        image_name = 'datasets/helen_style_adaptation/images/' + image_name
        image_numpy = translated_content.data[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = image_numpy.astype(np.uint8)
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_name)


