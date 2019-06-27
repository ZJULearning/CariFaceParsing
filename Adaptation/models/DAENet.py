
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import numpy as np
import pdb

###################################
######### loss functions ##########
###################################



class BiasReduceLoss(nn.Module):
    def __init__(self,opt):
        super(BiasReduceLoss, self).__init__()
        self.opt = opt
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
    def forward(self, x, y, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        #if self.opt.cuda:
        w.cuda()
        w = Variable(w, requires_grad=False)
        self.avg = torch.mean(x,0).unsqueeze(0)
        self.loss = w*self.criterion(self.avg, y)
        return self.loss

class TotalVaryLoss(nn.Module):
    def __init__(self,opt):
        super(TotalVaryLoss, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        #if self.opt.cuda:
        w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w * (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        return self.loss


###################################
#########  basic blocks  ##########
###################################
# a mixer (linear layer)
class waspMixer(nn.Module):
    def __init__(self, ngpu=1, nin=128, nout=128):
        super(waspMixer, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # simply a linear layer
            nn.Linear(nin, nout),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# warp image according to the grid
class waspWarper(nn.Module):
    def __init__(self, imgSize = 256, batchSize = 1):
        super(waspWarper, self).__init__()
        self.batchSize = batchSize
        self.imgSize = imgSize

    def forward(self, input_img, input_grid):
        print(input_grid.size())
        self.warp = input_grid.permute(0,2,3,1)
        self.output = F.grid_sample(input_img, self.warp)
        return self.output

# integrate over the predicted grid offset to get the grid(deformation field)
class waspGridSpatialIntegral(nn.Module):
    def __init__(self, imgSize = 256, cuda = True):
        super(waspGridSpatialIntegral, self).__init__()
        self.w = imgSize
        self.filterx = torch.cuda.FloatTensor(1,1,1,self.w).fill_(1)
        self.filtery = torch.cuda.FloatTensor(1,1,self.w,1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if cuda:
            self.filterx.cuda()
            self.filtery.cuda()
    def forward(self, input_diffgrid):
        #print(input_diffgrid.size())
        fullx = F.conv_transpose2d(input_diffgrid[:,0,:,:].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:,1,:,:].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:,:,0:self.w,0:self.w], fully[:,:,0:self.w,0:self.w]),1)
        return output_grid

# integrate over the predicted grid offset to get the grid(deformation field) from right
class waspGridSpatialIntegral2(nn.Module):
    def __init__(self, imgSize=256, cuda=True):
        super(waspGridSpatialIntegral2, self).__init__()
        self.w = imgSize
        self.filterx = torch.cuda.FloatTensor(1, 1, 1, self.w).fill_(1)
        self.filtery = torch.cuda.FloatTensor(1, 1, self.w, 1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if cuda:
            self.filterx.cuda()
            self.filtery.cuda()

    def forward(self, input_diffgrid):
        # print(input_diffgrid.size())
        fullx = F.conv_transpose2d(input_diffgrid[:, 0, :, :].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:, 1, :, :].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:, :, 0:self.w, -self.w:], fully[:, :, -self.w:, 0:self.w]), 1)
        return output_grid

# an encoder architecture
class waspEncoder(nn.Module):
    def __init__(self, ngpu=1, nc=1, ndf = 32, ndim = 128, norm_layer=nn.BatchNorm2d):
        super(waspEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            norm_layer(ndf * 2),
            #nn.InstanceNorm2d(ndf * 2, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            norm_layer(ndf * 4),
            #nn.InstanceNorm2d(ndf * 4, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            norm_layer(ndf * 8),
            #nn.InstanceNorm2d(ndf * 8, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        #print(output.size())
        return output

# a decoder architecture
class waspDecoder(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            #nn.InstanceNorm2d(ngf * 8, affine=True, track_running_stats=True),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            #nn.InstanceNorm2d(ngf * 4, affine=True, track_running_stats=True),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            #nn.InstanceNorm2d(ngf * 2, affine=True, track_running_stats=True),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            #nn.InstanceNorm2d(ngf, affine=True, track_running_stats=True),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            #nn.InstanceNorm2d(ngf, affine=True, track_running_stats=True),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoderTanh(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderTanh, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            #nn.InstanceNorm2d(ngf * 8, affine=True, track_running_stats=True),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            #nn.InstanceNorm2d(ngf * 4, affine=True, track_running_stats=True),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            #nn.InstanceNorm2d(ngf * 2, affine=True, track_running_stats=True),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            #nn.InstanceNorm2d(ngf, affine=True, track_running_stats=True),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            #nn.InstanceNorm2d(ngf, affine=True, track_running_stats=True),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            #nn.Hardtanh(lb,ub),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

###################################
########  densenet blocks #########
###################################

class DenseBlockEncoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseBlockEncoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers     = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    norm_layer(n_channels),
                    activation(*args),
                    nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseBlockDecoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseBlockDecoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    norm_layer(n_channels),
                    activation(*args),
                    nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseTransitionBlockEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp             = mp
        self.main           = nn.Sequential(
                norm_layer(n_channels_in),
                activation(*args),
                nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(mp),
        )
    def forward(self, inputs):
        return self.main(inputs)


class DenseTransitionBlockDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.main           = nn.Sequential(
                norm_layer(n_channels_in),
                activation(*args),
                nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False),
        )
    def forward(self, inputs):
        return self.main(inputs)




class waspDenseEncoder(nn.Module):
    def __init__(self, ngpu=1, nc=1, ndf = 32, ndim = 128, activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Sigmoid, f_args=[], norm_layer=nn.BatchNorm2d):
        super(waspDenseEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim

        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                #nn.BatchNorm2d(nc),
                #nn.InstanceNorm2d(nc, affine=True, track_running_stats=True),
                #norm_layer(nc),
                #nn.ReLU(True),
                nn.Conv2d(nc, ndf, 4, stride=2, padding=1),

                # state size. (ndf) x 32 x 32
                DenseBlockEncoder(ndf, 6, norm_layer=norm_layer),
                DenseTransitionBlockEncoder(ndf, ndf*2, 2, activation=activation, args=args, norm_layer=norm_layer),

                # state size. (ndf*2) x 16 x 16
                DenseBlockEncoder(ndf*2, 12, norm_layer=norm_layer),
                DenseTransitionBlockEncoder(ndf*2, ndf*4, 2, activation=activation, args=args, norm_layer=norm_layer),

                # state size. (ndf*4) x 8 x 8
                DenseBlockEncoder(ndf*4, 24, norm_layer=norm_layer),
                DenseTransitionBlockEncoder(ndf*4, ndf*8, 2, activation=activation, args=args, norm_layer=norm_layer),

                # state size. (ndf*8) x 4 x 4
                DenseBlockEncoder(ndf*8, 16, norm_layer=norm_layer),
                DenseTransitionBlockEncoder(ndf*8, ndim, 4, activation=activation, args=args, norm_layer=norm_layer),
                f_activation(*f_args),
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        return output


class waspDenseDecoder(nn.Module):
    def __init__(self, ngpu=1, nz=128, nc=1, ngf=32, lb=0, ub=1, activation=nn.ReLU, args=[False], f_activation=nn.Hardtanh, f_args=[0,1], norm_layer=nn.BatchNorm2d):
        super(waspDenseDecoder, self).__init__()
        self.ngpu   = ngpu
        self.main   = nn.Sequential(
            # input is Z, going into convolution
            #nn.BatchNorm2d(nz),
            #nn.InstanceNorm2d(nz, affine=True, track_running_stats=True),
            #norm_layer(nz),

            #activation(*args),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

            # state size. (ngf*8) x 4 x 4
            DenseBlockDecoder(ngf*8, 16, norm_layer=norm_layer),
            DenseTransitionBlockDecoder(ngf*8, ngf*4, norm_layer=norm_layer),

            # state size. (ngf*4) x 8 x 8
            DenseBlockDecoder(ngf*4, 24, norm_layer=norm_layer),
            DenseTransitionBlockDecoder(ngf*4, ngf*2, norm_layer=norm_layer),

            # state size. (ngf*2) x 16 x 16
            DenseBlockDecoder(ngf*2, 12, norm_layer=norm_layer),
            DenseTransitionBlockDecoder(ngf*2, ngf, norm_layer=norm_layer),

            # state size. (ngf) x 32 x 32
            DenseBlockDecoder(ngf, 6, norm_layer=norm_layer),
            DenseTransitionBlockDecoder(ngf, ngf, norm_layer=norm_layer),

            # state size (ngf) x 64 x 64
            norm_layer(ngf),
            activation(*args),
            nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False),
            f_activation(*f_args),
        )
    def forward(self, inputs):
        return self.main(inputs)

###################################
###### encoders and decoders ######
###################################

#### The encoders ####

# encoders of DAE
class Encoders(nn.Module):
    def __init__(self, opt):
        super(Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp


# encoders of DAE, using DenseNet architecture
class Dense_Encoders(nn.Module):
    def __init__(self, ngpu=1, nc=3, ngf=32, ndf=32, idim = 16, wdim = 128, zdim = 128, norm_layer=nn.BatchNorm2d):
        super(Dense_Encoders, self).__init__()
        self.ngpu = ngpu
        self.encoder = waspDenseEncoder(ngpu=1, nc=nc, ndf = ndf, ndim = zdim, norm_layer=norm_layer)
        self.zImixer = waspMixer(ngpu=1, nin = zdim, nout = idim)
        self.zWmixer = waspMixer(ngpu=1, nin = zdim, nout = wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp


#### The decoders ####

# decoders of DAE
class DecodersIntegralWarper2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2, self).__init__()
        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.warper   = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*(5.0/self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping





# decoders of DAE, using DenseNet architecture
class Dense_DecodersIntegralWarper2(nn.Module):
    def __init__(self, ngpu=1, nc=3, ngf=32, ndf=32, wdim = 128, imgSize=256, batch_size=1, norm_layer=nn.BatchNorm2d):
        super(Dense_DecodersIntegralWarper2, self).__init__()
        self.imagedimension = imgSize
        self.ngpu = ngpu
        self.wdim = wdim
        self.decoderW_left = waspDenseDecoder(ngpu=self.ngpu, nz=wdim, nc=2, ngf=ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[], norm_layer=norm_layer)
        self.decoderW_right = waspDenseDecoder(ngpu=self.ngpu, nz=wdim, nc=2, ngf=ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[], norm_layer=norm_layer)

        self.warper   = waspWarper(imgSize, batch_size)
        self.integrator = waspGridSpatialIntegral(imgSize=imgSize)
        self.integrator2 = waspGridSpatialIntegral2(imgSize=imgSize)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zW):

        self.diffentialWarping_left = (self.decoderW_left(zW.view(-1, self.wdim, 1, 1)) - 0.5) * (4.0 / self.imagedimension) + 2.0 / self.imagedimension
        self.diffentialWarping_right = (self.decoderW_right(zW.view(-1, self.wdim, 1, 1)) - 0.5) * (4.0 / self.imagedimension) + 2.0 / self.imagedimension

        self.warping_left = self.integrator(self.diffentialWarping_left)-1.0
        self.warping_right = 1.0 - self.integrator2(self.diffentialWarping_right)

        self.warping_left = self.cutter(self.warping_left)
        self.warping_right = self.cutter(self.warping_right)
        self.warping = (self.warping_left + self.warping_right) / 2.0 / 63.0 * 64.0

        return self.warping


class StarGenerator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ngf=32, ndf=32, idim = 9, wdim = 128, zdim = 128, imgSize = 256, batch_size=1, norm_layer=nn.BatchNorm2d):
        super(StarGenerator, self).__init__()
        self.encoders = waspDenseEncoder(ngpu=ngpu, nc=nc, ndf=ndf, ndim=zdim-9, norm_layer=norm_layer)
        self.decoders = Dense_DecodersIntegralWarper2(ngpu, nc, ngf, ndf, zdim, imgSize, batch_size, norm_layer=norm_layer)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        input_c = c.view(c.size(0), c.size(1), 1, 1)
        input_c = input_c.repeat(1, 1, x.size(2), x.size(3))
        dp0_img = torch.cat([x, input_c], dim=1)

        dp0_z= self.encoders(dp0_img)

        dp0_zW = torch.cat([dp0_z, c], dim=1)
        dp0_Wact = self.decoders(dp0_zW)
        dp0_Wact = dp0_Wact.permute(0, 2, 3, 1)
        return dp0_Wact


