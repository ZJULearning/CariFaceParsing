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
import torchvision.transforms as transforms
import sys

class ShapeAdaptationModel(BaseModel):
    def name(self):
        return 'ShapeAdaptationModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default DeformingGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0,
                                help='weight for cycle loss (B -> A -> B)')
            #parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_identity', type=float, default=0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.palette = np.array([0, 0, 0, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 255, 0, 0, 250, 170, 30,
                   0, 0, 230, 0, 80, 100, 152, 251, 152, 0, 255, 255, 0, 0, 142, 119, 11, 32])
        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette = np.append(self.palette, 0)
        self.palette = self.palette.reshape((256, 3))

        # Model configurations.
        self.c_dim = 9
        self.all_attr_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        self.image_size = 64
        self.g_conv_dim = 64
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
        self.CLSloss = CrossEntropyLoss()

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['d_real', 'd_fake', 'd_cls', 'd_gp', 'g_fake', 'g_rec', 'g_cls', 'tvw_real_A', 'tvw_rec_A', 'br_real_A', 'br_rec_A']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A_color', 'real_A_color_grid', 'transformed_real_A_color', 'rec_A_color_grid', 'rec_A_color']
        visual_names_val_faces = ['val_face']
        for i, attr_name in enumerate(self.all_attr_names):
            visual_names_val_faces.append('val_face_' + attr_name)
        if opt.phase == 'train':
            self.visual_names = visual_names_A + visual_names_val_faces
        else:
            self.visual_names = visual_names_val_faces
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        self.netG = networks.define_S(opt.input_nc, opt.output_nc, opt.ngf, 'star_gan', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD = networks.Discriminator(opt.output_nc, self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.zeroWarp = torch.cuda.FloatTensor(1, 2, 64, 64).fill_(0).to(self.device)
        self.baseg = networks.getBaseGrid(N=64, getbatch=True, batchSize=opt.batch_size).to(self.device)


        # criteria/loss
        self.criterionTVWarp = DAENet.TotalVaryLoss(opt)
        self.criterionBiasReduce = DAENet.BiasReduceLoss(opt)

        # initialize optimizers

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.netG.to(self.device)
        self.netD.to(self.device)

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

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




    def set_input_helen(self, input, epoch):
        self.image_paths = input['A_path']

        x_real = input['A']
        label_org = input['A_label']
        x_real_face = input['A_face']
        x_real_parsing = input['A_parsing']

        # Generate target domain labels randomly.
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]

        c_org = self.label2onehot(label_org, self.c_dim)
        c_trg = self.label2onehot(label_trg, self.c_dim)

        self.real_A = x_real.to(self.device)  # Input landmarks.
        self.real_A_face = x_real_face.to(self.device) # Input faces.
        self.real_A_parsing = x_real_parsing.to(self.device) # Input faces.
        self.c_org = c_org.to(self.device)  # Original domain labels.
        self.c_trg = c_trg.to(self.device)  # Target domain labels.
        self.label_org = label_org.to(self.device)  # Labels for computing classification loss.
        self.label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

        self.c_trgs = {}
        self.label_trgs = {}

        for i, attr_name in enumerate(self.all_attr_names):
            attr_label_trg = (torch.zeros(label_org.size()) + i).long()
            attr_c_trg = self.label2onehot(attr_label_trg, self.c_dim)
            self.c_trgs[attr_name] = attr_c_trg.to(self.device)
            self.label_trgs[attr_name] = attr_label_trg.to(self.device)  # Labels for computing classification loss.




    def set_input(self, input, epoch):
        self.image_paths = input['val_path']

        x_real = input['A']
        label_org = input['A_label']
        x_real_face = input['A_face']

        # Generate target domain labels randomly.
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]

        c_org = self.label2onehot(label_org, self.c_dim)
        c_trg = self.label2onehot(label_trg, self.c_dim)

        self.real_A = x_real.to(self.device)  # Input landmarks.
        self.real_A_face = x_real_face.to(self.device) # Input faces.
        self.c_org = c_org.to(self.device)  # Original domain labels.
        self.c_trg = c_trg.to(self.device)  # Target domain labels.
        self.label_org = label_org.to(self.device)  # Labels for computing classification loss.
        self.label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

        self.c_trgs = {}
        self.label_trgs = {}

        for i, attr_name in enumerate(self.all_attr_names):
            attr_label_trg = (torch.zeros(label_org.size()) + i).long()
            attr_c_trg = self.label2onehot(attr_label_trg, self.c_dim)
            self.c_trgs[attr_name] = attr_c_trg.to(self.device)
            self.label_trgs[attr_name] = attr_label_trg.to(self.device)  # Labels for computing classification loss.

        val_real = input['val']
        val_real_face = input['val_face']
        self.real_val_A = val_real.to(self.device)
        self.val_face = val_real_face.to(self.device)

        #  visualize the real_A
        real_A_color = input['A'][0].numpy().copy()
        real_A_color = real_A_color.transpose(1, 2, 0)
        real_A_color = np.asarray(np.argmax(real_A_color, axis=2), dtype=np.uint8)
        real_A_color_numpy = np.zeros((real_A_color.shape[0], real_A_color.shape[1], 3))
        for i in range(13):
            real_A_color_numpy[real_A_color == i] = self.palette[i]
        real_A_color = real_A_color_numpy.astype(np.uint8)
        real_A_color = real_A_color.transpose(2, 0, 1)
        real_A_color = real_A_color[np.newaxis, :]
        self.real_A_color = (torch.from_numpy(real_A_color).float()/255.0*2-1).to(self.device)

    def backward_D(self):
        # Compute loss with real images.
        out_src, out_cls = self.netD(self.real_A)
        d_loss_real = - torch.mean(out_src)
        self.label_org = self.label_org.view(self.label_org.size(0), 1, 1)
        d_loss_cls = F.cross_entropy(out_cls, self.label_org.repeat(1, out_cls.size(2), out_cls.size(3)))

        # Compute loss with fake images.

        transformed_real_A_Wact = self.netG(self.real_A, self.c_trg)
        x_fake = F.grid_sample(self.real_A, transformed_real_A_Wact, padding_mode='border')

        out_src, out_cls = self.netD(x_fake.detach())
        d_loss_fake = torch.mean(out_src)

        # Compute loss for gradient penalty.
        alpha = torch.rand(self.real_A.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * self.real_A.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = self.netD(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)

        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
        self.reset_grad()
        d_loss.backward()
        self.optimizer_D.step()
        # Logging.
        self.loss_d_real = d_loss_real
        self.loss_d_fake = d_loss_fake
        self.loss_d_cls = d_loss_cls
        self.loss_d_gp = d_loss_gp


    def backward_G(self):
        transformed_real_A_Wact = {}
        transformed_real_A = {}
        transformed_real_A_face = {}
        out_src = {}
        out_cls = {}
        all_g_loss_fake = {}
        all_g_loss_cls = {}
        rec_A_Wact = {}
        rec_A = {}
        all_g_loss_rec = {}
        rec_A_face = {}
        batch_size = self.real_A.size(0)

        for i, attr_name in enumerate(self.all_attr_names):
            # Original-to-target domain.
            transformed_real_A_Wact[attr_name] = self.netG(self.real_A, self.c_trgs[attr_name])
            transformed_real_A[attr_name] = F.grid_sample(self.real_A, transformed_real_A_Wact[attr_name], padding_mode='border')
            A_face_Wact = F.upsample(transformed_real_A_Wact[attr_name].permute(0, 3, 1, 2), size=(256, 256),
                                              mode='bilinear')
            transformed_real_A_face[attr_name] = F.grid_sample(self.real_A_face, A_face_Wact.permute(0, 2, 3, 1))
            out_src[attr_name], out_cls[attr_name] = self.netD(transformed_real_A[attr_name])
            all_g_loss_fake[attr_name] = - torch.mean(out_src[attr_name])
            label_trg = self.label_trgs[attr_name].view(batch_size, 1, 1)
            all_g_loss_cls[attr_name] = F.cross_entropy(out_cls[attr_name], label_trg.repeat(1, out_cls[attr_name].size(2), out_cls[attr_name].size(3)))

            # Target-to-original domain.
            rec_A_Wact[attr_name] = self.netG(transformed_real_A[attr_name], self.c_org)
            rec_A[attr_name] = F.grid_sample(transformed_real_A[attr_name], rec_A_Wact[attr_name], padding_mode='border')
            all_g_loss_rec[attr_name] = torch.mean(torch.abs(self.real_A - rec_A[attr_name]))
            rec_A_face_Wact = F.upsample(rec_A_Wact[attr_name].permute(0, 3, 1, 2), size=(256, 256), mode='bilinear')
            rec_A_face[attr_name] = F.grid_sample(transformed_real_A_face[attr_name], rec_A_face_Wact.permute(0, 2, 3, 1))

        tvw_weight = 1e-3
        all_loss_tvw_real_A = {}
        all_loss_tvw_rec_A = {}
        all_loss_br_real_A = {}
        all_loss_br_rec_A = {}
        for i, attr_name in enumerate(self.all_attr_names):
            all_loss_tvw_real_A[attr_name] = self.criterionTVWarp(transformed_real_A_Wact[attr_name].permute(0, 3, 1, 2) - self.baseg[:batch_size], weight=tvw_weight)
            all_loss_tvw_rec_A[attr_name] = self.criterionTVWarp(rec_A_Wact[attr_name].permute(0, 3, 1, 2) - self.baseg[:batch_size], weight=tvw_weight)
            all_loss_br_real_A[attr_name] = self.criterionBiasReduce(transformed_real_A_Wact[attr_name].permute(0,3,1,2) - self.baseg[:batch_size], self.zeroWarp[:batch_size], weight=1)
            all_loss_br_rec_A[attr_name] = self.criterionBiasReduce(rec_A_Wact[attr_name].permute(0,3,1,2) - self.baseg[:batch_size], self.zeroWarp[:batch_size], weight=1)

        # all loss functions

        # Backward and optimize.
        self.g_loss_cls = 0
        self.g_loss_fake = 0
        self.g_loss_rec = 0
        self.loss_tvw_real_A = 0
        self.loss_tvw_rec_A = 0
        self.loss_br_real_A = 0
        self.loss_br_rec_A = 0
        is_random = False
        if is_random == True:
            random_number = random.randint(1,9)
            attr_name = self.all_attr_names[random_number-1]
            self.g_loss_cls = all_g_loss_cls[attr_name]
            self.g_loss_fake = all_g_loss_fake[attr_name]
            self.g_loss_rec = all_g_loss_rec[attr_name]
            self.loss_tvw_real_A = all_loss_tvw_real_A[attr_name]
            self.loss_tvw_rec_A = all_loss_tvw_rec_A[attr_name]
            self.loss_br_real_A = all_loss_br_real_A[attr_name]
            self.loss_br_rec_A = all_loss_br_rec_A[attr_name]
            g_loss = self.g_loss_fake + self.lambda_rec * self.g_loss_rec + self.lambda_cls * self.g_loss_cls\
                     + self.loss_tvw_real_A + self.loss_tvw_rec_A + self.loss_br_real_A + self.loss_br_rec_A

        else:
            for i, attr_name in enumerate(self.all_attr_names):
                self.g_loss_cls = self.g_loss_cls + all_g_loss_cls[attr_name]
                self.g_loss_fake = self.g_loss_fake + all_g_loss_fake[attr_name]
                self.g_loss_rec = self.g_loss_rec + all_g_loss_rec[attr_name]
                self.loss_tvw_real_A = self.loss_tvw_real_A + all_loss_tvw_real_A[attr_name]
                self.loss_tvw_rec_A = self.loss_tvw_rec_A + all_loss_tvw_rec_A[attr_name]
                self.loss_br_real_A = self.loss_br_real_A + all_loss_br_real_A[attr_name]
                self.loss_br_rec_A = self.loss_br_rec_A + all_loss_br_rec_A[attr_name]

            self.g_loss_fake *= 0.2
            self.g_loss_rec *= 1
            self.g_loss_cls *= 0.1
            self.loss_tvw_real_A *= 0.1
            self.loss_tvw_rec_A *= 0.01
            self.loss_br_real_A *= 0.1
            self.loss_br_rec_A *= 0.01
            g_loss = self.g_loss_fake + self.lambda_rec * self.g_loss_rec + self.lambda_cls * self.g_loss_cls\
                     + self.loss_tvw_real_A + self.loss_tvw_rec_A + self.loss_br_real_A + self.loss_br_rec_A

        self.reset_grad()
        g_loss.backward()
        self.optimizer_G.step()
        # Logging.
        self.loss_g_fake = self.g_loss_fake
        self.loss_g_rec = self.g_loss_rec
        self.loss_g_cls = self.g_loss_cls

        random_number = random.randint(1, 9)
        attr_name = self.all_attr_names[random_number-1]
        source_control_points = transformed_real_A_Wact[attr_name][0].cpu().float().detach()
        real_A_color = self.real_A_color[0].cpu().float().numpy()
        real_A_color = (real_A_color + 1.0) / 2 * 255
        real_A_color = Image.fromarray(real_A_color.transpose(1, 2, 0).astype(np.uint8)).convert('RGB').resize((128, 128))
        canvas = Image.new(mode='RGB', size=(64 * 4, 64 * 4), color=(128, 128, 128))
        canvas.paste(real_A_color, (64, 64))
        source_points = (source_control_points + 1) / 2 * 128 + 64
        draw = ImageDraw.Draw(canvas)

        grid_size = 8
        grid_step = int(64 / 8)
        for j in range(grid_size):
            for k in range(grid_size):
                x, y = source_points[j*grid_step + 4][k*grid_step + 4]
                draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))

        source_points = source_points.view(64, 64, 2)
        for j in range(grid_size):
            for k in range(grid_size):
                x1, y1 = source_points[j*grid_step + 4, k*grid_step + 4]
                if j > 0:  # connect to left
                    x2, y2 = source_points[(j - 1)*grid_step + 4, k*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
                if k > 0:  # connect to up
                    x2, y2 = source_points[j*grid_step + 4, (k - 1)*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
        real_A_color = np.asarray(canvas, np.uint8)
        real_A_color_grid = real_A_color.transpose(2, 0, 1)
        real_A_color_grid = real_A_color_grid[np.newaxis, :]
        self.real_A_color_grid = (torch.from_numpy(real_A_color_grid).float()/255*2-1).to(self.device)


        # visualize the fake_B
        transformed_A_color = transformed_real_A[attr_name].data[0].cpu().numpy()
        transformed_A_color = transformed_A_color.transpose(1,2,0)
        transformed_A_color = np.asarray(np.argmax(transformed_A_color, axis=2), dtype=np.uint8)
        transformed_A_color_numpy = np.zeros((transformed_A_color.shape[0], transformed_A_color.shape[1],3))
        for i in range(13):
            transformed_A_color_numpy[transformed_A_color==i] = self.palette[i]
        transformed_A_color = transformed_A_color_numpy.astype(np.uint8)
        transformed_A_color = transformed_A_color.transpose(2, 0, 1)
        transformed_A_color = transformed_A_color[np.newaxis, :]
        self.transformed_real_A_color = (torch.from_numpy(transformed_A_color).float()/255*2-1).to(self.device)

        # visualize the rec_A
        source_control_points = rec_A_Wact[attr_name][0].cpu().float().detach()
        real_A_color = self.transformed_real_A_color[0].cpu().float().numpy()
        real_A_color = (real_A_color + 1.0) / 2 * 255
        real_A_color = Image.fromarray(real_A_color.transpose(1, 2, 0).astype(np.uint8)).convert('RGB').resize((128, 128))
        canvas = Image.new(mode='RGB', size=(64 * 4, 64 * 4), color=(128, 128, 128))
        canvas.paste(real_A_color, (64, 64))
        source_points = (source_control_points + 1) / 2 * 128 + 64
        draw = ImageDraw.Draw(canvas)

        for j in range(grid_size):
            for k in range(grid_size):
                x, y = source_points[j*grid_step + 4][k*grid_step + 4]
                draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))

        source_points = source_points.view(64, 64, 2)
        for j in range(grid_size):
            for k in range(grid_size):
                x1, y1 = source_points[j*grid_step + 4, k*grid_step + 4]
                if j > 0:  # connect to left
                    x2, y2 = source_points[(j - 1)*grid_step + 4, k*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
                if k > 0:  # connect to up
                    x2, y2 = source_points[j*grid_step + 4, (k - 1)*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
        real_A_color = np.asarray(canvas, np.uint8)
        real_A_color_grid = real_A_color.transpose(2, 0, 1)
        real_A_color_grid = real_A_color_grid[np.newaxis, :]
        self.rec_A_color_grid = (torch.from_numpy(real_A_color_grid).float()/255*2-1).to(self.device)



        # visualize the rec
        rec_A_color = rec_A[attr_name].data[0].cpu().numpy()
        rec_A_color = rec_A_color.transpose(1,2,0)
        rec_A_color = np.asarray(np.argmax(rec_A_color, axis=2), dtype=np.uint8)
        rec_A_color_numpy = np.zeros((rec_A_color.shape[0], rec_A_color.shape[1],3))
        for i in range(13):
            rec_A_color_numpy[rec_A_color==i] = self.palette[i]
        rec_A_color = rec_A_color_numpy.astype(np.uint8)
        rec_A_color = rec_A_color.transpose(2, 0, 1)
        rec_A_color = rec_A_color[np.newaxis, :]
        self.rec_A_color = (torch.from_numpy(rec_A_color).float()/255*2-1).to(self.device)


    def optimize_parameters(self, epoch):
        self.backward_D()
        self.backward_G()
        self.netG.eval()
        for i, attr_name in enumerate(self.all_attr_names):
            val_Wact = self.netG(self.real_val_A, self.c_trgs[attr_name])
            val_face_Wact = F.upsample(val_Wact.permute(0, 3, 1, 2), size=(256, 256), mode='bilinear')
            setattr(self, 'val_face_' + attr_name, F.grid_sample(self.val_face, val_face_Wact.permute(0, 2, 3, 1)))
        self.netG.train()



    def forward(self):
        self.netG.eval()
        transform_list = []
        osize = [256, 256]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(256))
        transform_list += [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
        val_transform = transforms.Compose(transform_list)

        for i, attr_name in enumerate(self.all_attr_names):
            if i==0:
                continue
            val_Wact = self.netG(self.real_A, self.c_trgs[attr_name])
            val_face_Wact = F.upsample(val_Wact.permute(0, 3, 1, 2), size=(256, 256), mode='bilinear')

            val_face = F.grid_sample(self.real_A_face, val_face_Wact.permute(0,2,3,1))
            image_name = self.image_paths[0].split('/')[-1]
            image_name = 'datasets/helen_shape_adaptation/images/' + image_name[:-4] + '_' + attr_name + '.jpg'
            image_numpy = val_face.data[0].cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            image_numpy = image_numpy.astype(np.uint8)
            image_pil = Image.fromarray(image_numpy)
            image_pil.save(image_name)

            val_parsing = F.grid_sample(self.real_A_parsing, val_face_Wact.permute(0,2,3,1))
            val_parsing = val_parsing.data[0].cpu().float().numpy()
            val_parsing = np.argmax(np.transpose(val_parsing, (1,2,0)), axis=2)
            val_parsing = Image.fromarray(val_parsing.astype('uint8'))
            image_name = self.image_paths[0].split('/')[-1]
            image_name = 'datasets/helen_shape_adaptation/labels/' + image_name[:-4] + '_' + attr_name + '.png'
            val_parsing.save(image_name)

