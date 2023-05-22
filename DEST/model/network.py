from .base_function import *
from .external_function import SpectralNorm
import torch.nn.functional as F
import torch
from torchvision import models
import functools
import os
import torchvision.transforms as transforms


##############################################################################################################
# Network function
##############################################################################################################
def define_es(init_type='normal', gpu_ids=[]):

    net = Encoder_S()

    return init_net(net, init_type, gpu_ids)

def define_et(init_type='normal', gpu_ids=[]):

    net = Encoder_T()

    return init_net(net, init_type, gpu_ids)


def define_de(init_type='orthogonal', gpu_ids=[]):

    net = Decoder()

    return init_net(net, init_type, gpu_ids)


def define_dis_g(init_type='orthogonal', gpu_ids=[]):

    net = GlobalDiscriminator()

    return init_net(net, init_type, gpu_ids)

def define_attn(init_type='orthogonal', gpu_ids=[]):

    net = Cross_Attention()

    return init_net(net, init_type, gpu_ids)

def define_fuse_s(init_type='orthogonal', gpu_ids=[]):

    net = Trans_conv_s()

    return init_net(net, init_type, gpu_ids)

def define_fuse_t(init_type='orthogonal', gpu_ids=[]):

    net = Trans_conv_t()

    return init_net(net, init_type, gpu_ids)

def define_G2(init_type='orthogonal', gpu_ids=[]):

    net = refine_G2()

    return init_net(net, init_type, gpu_ids)


#############################################################################################################
# Network structure
#############################################################################################################
class Encoder_S(nn.Module):
    def __init__(self):
        super(Encoder_S, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        kwargs_gate = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.gate_nonlinearity = nn.Sigmoid()

        # encoder0
        self.conv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.bypass1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs_short))
        self.model1 = nn.Sequential(self.conv1, self.nonlinearity, self.conv2, self.pool)
        self.shortcut1 = nn.Sequential(self.pool, self.bypass1)
        self.gateconv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs_gate))
        self.gate1 = nn.Sequential(self.gateconv1, self.gate_nonlinearity)

        # encoder1
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs))
        self.shortcut2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model2 = nn.Sequential(self.nonlinearity, self.conv3, self.nonlinearity, self.conv4)
        self.gateconv2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_gate))
        self.gate2 = nn.Sequential(self.gateconv2, self.gate_nonlinearity)

        # encoder2
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 128, **kwargs))
        self.shortcut3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.nonlinearity, self.conv5, self.nonlinearity, self.conv6)
        self.gateconv3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_gate))
        self.gate3 = nn.Sequential(self.gateconv3, self.gate_nonlinearity)

        # encoder3
        self.conv7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(128, 256, **kwargs))
        self.shortcut4 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_short))
        self.model4 = nn.Sequential(self.nonlinearity, self.conv7, self.nonlinearity, self.conv8)
        self.gateconv4 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_gate))
        self.gate4 = nn.Sequential(self.gateconv4, self.gate_nonlinearity)

        # encoder4
        self.conv9 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(256, 512, **kwargs))
        self.shortcut5 = SpectralNorm(nn.Conv2d(256, 512, **kwargs_short))
        self.model5 = nn.Sequential(self.nonlinearity, self.conv9, self.nonlinearity, self.conv10)
        self.gateconv5 = SpectralNorm(nn.Conv2d(256, 512, **kwargs_gate))
        self.gate5 = nn.Sequential(self.gateconv5, self.gate_nonlinearity)

        # encoder5
        self.prior_conv1 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.prior_conv2 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.prior_shortcut = SpectralNorm(nn.Conv2d(512, 512, **kwargs_short))
        self.prior_model = nn.Sequential(self.nonlinearity, self.prior_conv1, self.nonlinearity, self.prior_conv2)
        self.gateconv6 = SpectralNorm(nn.Conv2d(512, 512, **kwargs_gate))
        self.gate6 = nn.Sequential(self.gateconv6, self.gate_nonlinearity)

    def forward(self, x):
        feature = []
        distribution = []
        x = self.nonlinearity(self.model1(x) + self.shortcut1(x)) * self.gate1(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.model2(x)) + self.pool(self.shortcut2(x))) * self.gate2(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.model3(x)) + self.pool(self.shortcut3(x))) * self.gate3(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.model4(x)) + self.pool(self.shortcut4(x))) * self.gate4(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.model5(x)) + self.pool(self.shortcut5(x))) * self.gate5(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.prior_model(x)) + self.pool(self.prior_shortcut(x))) * self.gate6(x)
        feature.append(x)

        q_mu, q_std = torch.split(x, 256, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])
        return distribution, feature


class Encoder_T(nn.Module):
    def __init__(self):
        super(Encoder_T, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        kwargs_gate = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.gate_nonlinearity = nn.Sigmoid()

        # encoder0
        self.conv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.bypass1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs_short))
        self.model1 = nn.Sequential(self.conv1, self.nonlinearity, self.conv2, self.pool)
        self.shortcut1 = nn.Sequential(self.pool, self.bypass1)
        self.gateconv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs_gate))
        self.gate1 = nn.Sequential(self.gateconv1, self.gate_nonlinearity)

        # encoder1
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs))
        self.shortcut2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model2 = nn.Sequential(self.nonlinearity, self.conv3, self.nonlinearity, self.conv4)
        self.gateconv2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_gate))
        self.gate2 = nn.Sequential(self.gateconv2, self.gate_nonlinearity)

        # encoder2
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 128, **kwargs))
        self.shortcut3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.nonlinearity, self.conv5, self.nonlinearity, self.conv6)
        self.gateconv3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_gate))
        self.gate3 = nn.Sequential(self.gateconv3, self.gate_nonlinearity)

        # encoder3
        self.conv7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(128, 256, **kwargs))
        self.shortcut4 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_short))
        self.model4 = nn.Sequential(self.nonlinearity, self.conv7, self.nonlinearity, self.conv8)
        self.gateconv4 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_gate))
        self.gate4 = nn.Sequential(self.gateconv4, self.gate_nonlinearity)

        # encoder4
        self.conv9 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(256, 512, **kwargs))
        self.shortcut5 = SpectralNorm(nn.Conv2d(256, 512, **kwargs_short))
        self.model5 = nn.Sequential(self.nonlinearity, self.conv9, self.nonlinearity, self.conv10)
        self.gateconv5 = SpectralNorm(nn.Conv2d(256, 512, **kwargs_gate))
        self.gate5 = nn.Sequential(self.gateconv5, self.gate_nonlinearity)

        # encoder5
        self.prior_conv1 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.prior_conv2 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.prior_shortcut = SpectralNorm(nn.Conv2d(512, 512, **kwargs_short))
        self.prior_model = nn.Sequential(self.nonlinearity, self.prior_conv1, self.nonlinearity, self.prior_conv2)
        self.gateconv6 = SpectralNorm(nn.Conv2d(512, 512, **kwargs_gate))
        self.gate6 = nn.Sequential(self.gateconv6, self.gate_nonlinearity)

    def forward(self, x):
        feature = []
        distribution = []
        x = self.nonlinearity(self.model1(x) + self.shortcut1(x)) * self.gate1(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.model2(x)) + self.pool(self.shortcut2(x))) * self.gate2(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.model3(x)) + self.pool(self.shortcut3(x))) * self.gate3(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.model4(x)) + self.pool(self.shortcut4(x))) * self.gate4(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.model5(x)) + self.pool(self.shortcut5(x))) * self.gate5(x)
        feature.append(x)
        x = self.nonlinearity(self.pool(self.prior_model(x)) + self.pool(self.prior_shortcut(x))) * self.gate6(x)
        feature.append(x)

        q_mu, q_std = torch.split(x, 256, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])
        return distribution, feature



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        kwargs_out = {'kernel_size': 3, 'padding': 0, 'bias': True}
        kwargs_fuse_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)
        self.gate_nonlinearity = nn.Sigmoid()


        #decoder1
        self.conv1 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.shortcut1 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.model1 = nn.Sequential(self.nonlinearity, self.conv1, self.nonlinearity, self.conv2)
        self.gateconv1 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.gate1 = nn.Sequential(self.gateconv1, self.gate_nonlinearity)

        #decoder2
        self.conv3 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.shortcut2 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.model2 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv3, self.norm(512), self.nonlinearity, self.conv4)
        self.gateconv2 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.gate2 = nn.Sequential(self.gateconv2, self.gate_nonlinearity)

        #decoder3
        self.conv5 = SpectralNorm(nn.Conv2d(512, 256, **kwargs))
        self.conv6 = SpectralNorm(nn.ConvTranspose2d(256, 128, **kwargs_short))
        self.shortcut3 = SpectralNorm(nn.ConvTranspose2d(512, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv5, self.norm(256), self.nonlinearity, self.conv6)
        self.gateconv3 = SpectralNorm(nn.ConvTranspose2d(512, 128, **kwargs_short))
        self.gate3 = nn.Sequential(self.gateconv3, self.gate_nonlinearity)

        #out1
        self.conv_out1 = SpectralNorm(nn.Conv2d(128, 3, **kwargs_out))
        self.model_out1 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out1, nn.Tanh())


        #decoder4
        self.conv7 = SpectralNorm(nn.Conv2d(259, 64, **kwargs))
        self.conv8 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_short))
        self.shortcut4 = SpectralNorm(nn.ConvTranspose2d(259, 64, **kwargs_short))
        self.model4 = nn.Sequential(self.norm(259), self.nonlinearity, self.conv7, self.norm(64), self.nonlinearity, self.conv8)
        self.gateconv4 = SpectralNorm(nn.ConvTranspose2d(259, 64, **kwargs_short))
        self.gate4 = nn.Sequential(self.gateconv4, self.gate_nonlinearity)

        #out2
        self.conv_out2 = SpectralNorm(nn.Conv2d(64, 3, **kwargs_out))
        self.model_out2 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out2, nn.Tanh())

        #decoder5
        self.conv9 = SpectralNorm(nn.Conv2d(131, 32, **kwargs))
        self.conv10 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_short))
        self.shortcut5 = SpectralNorm(nn.ConvTranspose2d(131, 32, **kwargs_short))
        self.model5 = nn.Sequential(self.norm(131), self.nonlinearity, self.conv9, self.norm(32), self.nonlinearity, self.conv10)
        self.gateconv5 = SpectralNorm(nn.ConvTranspose2d(131, 32, **kwargs_short))
        self.gate5 = nn.Sequential(self.gateconv5, self.gate_nonlinearity)

        #out3
        self.conv_out3 = SpectralNorm(nn.Conv2d(32, 3, **kwargs_out))
        self.model_out3 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out3, nn.Tanh())

        #decoder6
        self.conv11 = SpectralNorm(nn.Conv2d(67, 16, **kwargs))
        self.conv12 = SpectralNorm(nn.ConvTranspose2d(16, 16, **kwargs_short))
        self.shortcut6 = SpectralNorm(nn.ConvTranspose2d(67, 16, **kwargs_short))
        self.model6 = nn.Sequential(self.norm(67), self.nonlinearity, self.conv11, self.norm(16), self.nonlinearity, self.conv12)
        self.gateconv6 = SpectralNorm(nn.ConvTranspose2d(67, 16, **kwargs_short))
        self.gate6 = nn.Sequential(self.gateconv6, self.gate_nonlinearity)

        #out4
        self.conv_out4 = SpectralNorm(nn.Conv2d(16, 3, **kwargs_out))
        self.model_out4 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out4, nn.Tanh())

        # decoder7
        self.conv13 = SpectralNorm(nn.Conv2d(19, 16, **kwargs))
        self.conv14 = SpectralNorm(nn.Conv2d(16, 16, **kwargs))
        self.shortcut7 = SpectralNorm(nn.ConvTranspose2d(19, 16, **kwargs))
        self.model7 = nn.Sequential(self.norm(19), self.nonlinearity, self.conv13, self.norm(16), self.nonlinearity, self.conv14)
        self.gateconv7 = SpectralNorm(nn.ConvTranspose2d(19, 16, **kwargs))
        self.gate7 = nn.Sequential(self.gateconv7, self.gate_nonlinearity)

        #out5
        self.conv_out5 = SpectralNorm(nn.Conv2d(16, 3, **kwargs_out))
        self.model_out5 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out5, nn.Tanh())

        #diffusion
        self.fuse_conv1 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_conv2 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_conv3 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_down1 = nn.Sequential(self.fuse_conv1, self.nonlinearity, self.fuse_conv2, self.nonlinearity, self.fuse_conv3, self.nonlinearity)

        self.fuse_conv4 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_conv4_1 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_down2 = nn.Sequential(self.fuse_conv4, self.nonlinearity, self.fuse_conv4_1, self.nonlinearity)

        self.fuse_conv5 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_down3 = nn.Sequential(self.fuse_conv5, self.nonlinearity)

        self.fuse_conv6 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.fuse_up1 = nn.Sequential(self.fuse_conv6, self.norm(128), self.nonlinearity)

        self.fuse_conv8 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.fuse_conv10 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.fuse_conv11 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_short))
        self.fuse_up2 = nn.Sequential(self.fuse_conv8, self.norm(128), self.nonlinearity, self.fuse_conv10, self.norm(64), self.nonlinearity,
                                      self.fuse_conv11, self.norm(64), self.nonlinearity)

        self.fuse_conv12 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.fuse_conv13 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.fuse_conv14 = SpectralNorm(nn.Conv2d(64, 32, **kwargs))
        self.fuse_conv15 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_short))
        self.fuse_conv16 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.fuse_conv17 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_short))
        self.fuse_up3 = nn.Sequential(self.fuse_conv12, self.norm(128), self.nonlinearity, self.fuse_conv13, self.norm(64), self.nonlinearity,
                                      self.fuse_conv14, self.norm(32), self.nonlinearity, self.fuse_conv15, self.norm(32), self.nonlinearity,
                                      self.fuse_conv16, self.norm(32), self.nonlinearity, self.fuse_conv17, self.norm(32), self.nonlinearity,)


    def forward(self, x, fuse_s, fuse_t):
        results = []
        s_1 = self.fuse_down1(fuse_s)
        s_2 = self.fuse_down2(fuse_s)
        s_3 = self.fuse_down3(fuse_s)

        t_1 = self.fuse_up1(fuse_t)
        t_2 = self.fuse_up2(fuse_t)
        t_3 = self.fuse_up3(fuse_t)

        # out = f_m + x
        out = x
        out = torch.cat([out, s_1], dim=1)
        out = self.nonlinearity(self.model1(out) + self.shortcut1(out)) * self.gate1(out)
        out = torch.cat([out, s_2], 1)
        out = self.nonlinearity(self.model2(out) + self.shortcut2(out)) * self.gate2(out)
        out = torch.cat([out, s_3], 1)
        out = self.nonlinearity(self.model3(out) + self.shortcut3(out)) * self.gate3(out)
        output = self.model_out1(out)
        results.append(output)
        out = torch.cat([out, output], dim=1)
        out = torch.cat([out, t_1], 1)
        out = self.nonlinearity(self.model4(out) + self.shortcut4(out)) * self.gate4(out)
        output = self.model_out2(out)
        results.append(output)
        out = torch.cat([out, output], dim=1)

        out = torch.cat([out, t_2], 1)
        out = self.nonlinearity(self.model5(out) + self.shortcut5(out)) * self.gate5(out)
        output = self.model_out3(out)
        results.append(output)
        out = torch.cat([out, output], dim=1)
        out = torch.cat([out, t_3], 1)
        out = self.nonlinearity(self.model6(out) + self.shortcut6(out)) * self.gate6(out)
        output = self.model_out4(out)
        results.append(output)
        out = torch.cat([out, output], dim=1)
        out = self.nonlinearity(self.model7(out) + self.shortcut7(out)) * self.gate7(out)
        output = self.model_out5(out)
        results.append(output)
        return results


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # encoder0
        self.conv1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.bypass1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs_short))
        self.model1 = nn.Sequential(self.conv1, self.nonlinearity, self.conv2, self.pool)
        self.shortcut1 = nn.Sequential(self.pool, self.bypass1)

        # encoder1
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs))
        self.bypass2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model2 = nn.Sequential(self.nonlinearity, self.conv3, self.nonlinearity, self.conv4)
        self.shortcut2 = nn.Sequential(self.bypass2)

        # encoder2
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 128, **kwargs))
        self.bypass3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.nonlinearity, self.conv5, self.nonlinearity, self.conv6)
        self.shortcut3 = nn.Sequential(self.bypass3)

        # encoder3
        self.conv7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass4 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model4 = nn.Sequential(self.nonlinearity, self.conv7, self.nonlinearity, self.conv8)
        self.shortcut4 = nn.Sequential(self.bypass4)

        # encoder4
        self.conv9 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model5 = nn.Sequential(self.nonlinearity, self.conv9, self.nonlinearity, self.conv10)
        self.shortcut5 = nn.Sequential(self.bypass5)

        # encoder5
        self.conv11 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv12 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass6 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model6 = nn.Sequential(self.nonlinearity, self.conv11, self.nonlinearity, self.conv12)
        self.shortcut6 = nn.Sequential(self.bypass6)

        # concat
        self.concat = SpectralNorm(nn.Conv2d(128, 1, 3))

    def forward(self, x):
        x = self.model1(x) + self.shortcut1(x)
        x = self.pool(self.model2(x)) + self.pool(self.shortcut2(x))
        x = self.pool(self.model3(x)) + self.pool(self.shortcut3(x))
        out = self.pool(self.model4(x)) + self.pool(self.shortcut4(x))
        out = self.pool(self.model5(out)) + self.pool(self.shortcut5(out))
        out = self.pool(self.model6(out)) + self.pool(self.shortcut6(out))
        out = self.concat(self.nonlinearity(out))

        return out


class Cross_Attention(nn.Module):
    def __init__(self):
        super(Cross_Attention, self).__init__()

        self.query_conv = nn.Conv2d(512 , 64, kernel_size= (1,1))
        self.key_conv = nn.Conv2d(512 , 64, kernel_size= (1,1))
        self.value_conv = nn.Conv2d(512 , 512 , kernel_size= (1,1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x_s, x_t):

        B, C, W ,H = x_t.size()
        proj_query  = self.query_conv(x_t).view(B, -1, W * H).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x_s).view(B, -1, W * H) # B X C x (*W*H)
        proj_value = self.value_conv(x_s).view(B, -1, W * H)
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(B, C, W, H)
        out = self.gamma*out + x_s


        return out



class Trans_conv_s(nn.Module):
    def __init__(self):
        super(Trans_conv_s, self).__init__()
        kwargs_short = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)


        self.conv1 = SpectralNorm(nn.Conv2d(32, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.model1 = nn.Sequential(self.conv1, self.norm(128), self.nonlinearity, self.conv2, self.norm(256), self.nonlinearity)

        self.conv3 = SpectralNorm(nn.Conv2d(64, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.model2 = nn.Sequential(self.conv3, self.norm(256), self.nonlinearity)

        self.conv4 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.model3 = nn.Sequential(self.conv4, self.norm(256), self.nonlinearity)

        self.conv5 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv6 = SpectralNorm(nn.ConvTranspose2d(256, 256,**kwargs_short))
        self.model4 = nn.Sequential(self.conv5, self.norm(256), self.nonlinearity, self.conv6, self.norm(256), self.nonlinearity)

        self.conv7 = SpectralNorm(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv8 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv9 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv10 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.model5 = nn.Sequential(self.conv7, self.norm(256), self.nonlinearity, self.conv8, self.norm(256),self.nonlinearity,
                                    self.conv9, self.norm(256), self.nonlinearity, self.conv10, self.norm(256),self.nonlinearity)

        self.conv11 = SpectralNorm(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv12 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv13 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv14 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv15 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv16 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.model6 = nn.Sequential(self.conv11, self.norm(256), self.nonlinearity, self.conv12, self.norm(256),self.nonlinearity,
                                    self.conv13, self.norm(256), self.nonlinearity, self.conv14, self.norm(256),self.nonlinearity,
                                    self.conv15, self.norm(256), self.nonlinearity, self.conv16, self.norm(256),self.nonlinearity)

        self.conv17 = SpectralNorm(nn.Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1)))
        self.down = nn.Sequential(self.conv17, self.norm(256), self.nonlinearity)
    def forward(self, feature1):
        x1 = self.model1(feature1[0])
        x2 = self.model2(feature1[1])
        x3 = self.model3(feature1[2])
        x4 = self.model4(feature1[3])
        x5 = self.model5(feature1[4])
        x6 = self.model6(feature1[5])
        x_fuse = torch.cat([x1, x2, x3, x4, x5, x6], 1)
        x = self.down(x_fuse)

        return x

class Trans_conv_t(nn.Module):
    def __init__(self):
        super(Trans_conv_t, self).__init__()
        kwargs_short = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)


        self.conv1 = SpectralNorm(nn.Conv2d(32, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.model1 = nn.Sequential(self.conv1, self.norm(128), self.nonlinearity, self.conv2, self.norm(256), self.nonlinearity)

        self.conv3 = SpectralNorm(nn.Conv2d(64, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.model2 = nn.Sequential(self.conv3, self.norm(256), self.nonlinearity)

        self.conv4 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.model3 = nn.Sequential(self.conv4, self.norm(256), self.nonlinearity)

        self.conv5 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv6 = SpectralNorm(nn.ConvTranspose2d(256, 256,**kwargs_short))
        self.model4 = nn.Sequential(self.conv5, self.norm(256), self.nonlinearity, self.conv6, self.norm(256), self.nonlinearity)

        self.conv7 = SpectralNorm(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv8 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv9 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv10 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.model5 = nn.Sequential(self.conv7, self.norm(256), self.nonlinearity, self.conv8, self.norm(256),self.nonlinearity,
                                    self.conv9, self.norm(256), self.nonlinearity, self.conv10, self.norm(256),self.nonlinearity)

        self.conv11 = SpectralNorm(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv12 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv13 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv14 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv15 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv16 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.model6 = nn.Sequential(self.conv11, self.norm(256), self.nonlinearity, self.conv12, self.norm(256),self.nonlinearity,
                                    self.conv13, self.norm(256), self.nonlinearity, self.conv14, self.norm(256),self.nonlinearity,
                                    self.conv15, self.norm(256), self.nonlinearity, self.conv16, self.norm(256),self.nonlinearity)

        self.conv17 = SpectralNorm(nn.Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1)))
        self.down = nn.Sequential(self.conv17, self.norm(256), self.nonlinearity)
    def forward(self, feature1):
        x1 = self.model1(feature1[0])
        x2 = self.model2(feature1[1])
        x3 = self.model3(feature1[2])
        x4 = self.model4(feature1[3])
        x5 = self.model5(feature1[4])
        x6 = self.model6(feature1[5])
        x_fuse = torch.cat([x1, x2, x3, x4, x5, x6], 1)
        x = self.down(x_fuse)

        return x


class refine_G2(nn.Module):
    def __init__(self):
        super(refine_G2, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_3 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_5 = {'kernel_size': 5, 'stride': 1, 'padding': 2}
        kwargs_7 = {'kernel_size': 7, 'stride': 1, 'padding': 3}
        kwargs_up = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.gate_nonlinearity = nn.Sigmoid()
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)

        # encoder1
        self.conv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.shortcut1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.model1 = nn.Sequential(self.conv1, self.norm(32), self.nonlinearity, self.conv2)
        self.gateconv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.gate1 = nn.Sequential(self.gateconv1, self.gate_nonlinearity)

        # encoder2
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_down))
        self.shortcut2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_down))
        self.model2 = nn.Sequential(self.norm(32), self.nonlinearity, self.conv3, self.norm(32), self.nonlinearity, self.conv4)
        self.gateconv2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_down))
        self.gate2 = nn.Sequential(self.gateconv2, self.gate_nonlinearity)

        # encoder3
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.shortcut3 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.model3 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv5, self.norm(64), self.nonlinearity, self.conv6)
        self.gateconv3 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.gate3 = nn.Sequential(self.gateconv3, self.gate_nonlinearity)

        # encoder4
        self.conv7 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_down))
        self.shortcut4 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_down))
        self.model4 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv7, self.norm(64), self.nonlinearity, self.conv8)
        self.gateconv4 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_down))
        self.gate4 = nn.Sequential(self.gateconv4, self.gate_nonlinearity)

        # encoder5
        self.conv9 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.model5 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv9, self.norm(128), self.nonlinearity, self.conv10)
        self.gateconv5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.gate5 = nn.Sequential(self.gateconv5, self.gate_nonlinearity)

        #Multi_conv
        self.conv_3 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_3))
        self.multi_3 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv_3)
        self.conv_5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_5))
        self.multi_5 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv_5)
        self.conv_7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_7))
        self.multi_7 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv_7)

        # decoder1
        self.de_conv1 = SpectralNorm(nn.Conv2d(384, 384, **kwargs))
        self.de_conv2 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.de_shortcut1 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.de_model1 = nn.Sequential(self.norm(384), self.nonlinearity, self.de_conv1, self.norm(384), self.nonlinearity, self.de_conv2)
        self.de_gateconv1 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.de_gate1 = nn.Sequential(self.de_gateconv1, self.gate_nonlinearity)

        # decoder2
        self.de_conv3 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.de_conv4 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs))
        self.de_shortcut2 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs))
        self.de_model2 = nn.Sequential(self.norm(128), self.nonlinearity, self.de_conv3, self.norm(128), self.nonlinearity,
                                    self.de_conv4)
        self.de_gateconv2 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs))
        self.de_gate2 = nn.Sequential(self.de_gateconv2, self.gate_nonlinearity)

        self.attn1 = Auto_Attn(128)

        # decoder3
        self.de_conv5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.de_conv6 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_up))
        self.de_shortcut3 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_up))
        self.de_model3 = nn.Sequential(self.norm(128), self.nonlinearity, self.de_conv5, self.norm(128), self.nonlinearity,
                                    self.de_conv6)
        self.de_gateconv3 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_up))
        self.de_gate3 = nn.Sequential(self.de_gateconv3, self.gate_nonlinearity)

        # decoder4
        self.de_conv7 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.de_conv8 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs))
        self.de_shortcut4 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs))
        self.de_model4 = nn.Sequential(self.norm(64), self.nonlinearity, self.de_conv7, self.norm(64), self.nonlinearity,
                                    self.de_conv8)
        self.de_gateconv4 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs))
        self.de_gate4 = nn.Sequential(self.de_gateconv4, self.gate_nonlinearity)

        self.attn2 = Auto_Attn(64)

        # decoder5
        self.de_conv9 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.de_conv10 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_up))
        self.de_shortcut5 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_up))
        self.de_model5 = nn.Sequential(self.norm(64), self.nonlinearity, self.de_conv9, self.norm(64), self.nonlinearity,
                                    self.de_conv10)
        self.de_gateconv5 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_up))
        self.de_gate5 = nn.Sequential(self.de_gateconv5, self.gate_nonlinearity)

        self.out = SpectralNorm(nn.Conv2d(32, 3, **kwargs))
        self.model_out = nn.Sequential(self.nonlinearity, self.out, nn.Tanh())

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        feature = []
        x = self.nonlinearity(self.model1(x) + self.shortcut1(x)) * self.gate1(x)
        feature.append(x)
        x = self.nonlinearity(self.model2(x) + self.shortcut2(x)) * self.gate2(x)
        feature.append(x)
        x = self.nonlinearity(self.model3(x) + self.shortcut3(x)) * self.gate3(x)
        feature.append(x)
        x = self.nonlinearity(self.model4(x) + self.shortcut4(x)) * self.gate4(x)
        feature.append(x)
        x = self.nonlinearity(self.model5(x) + self.shortcut5(x)) * self.gate5(x)
        feature.append(x)
        multi1 = self.multi_3(x)
        multi2 = self.multi_5(x)
        multi3 = self.multi_7(x)
        fuse = torch.cat([multi1, multi2, multi3], 1)
        x = self.nonlinearity(self.de_model1(fuse) + self.de_shortcut1(fuse)) * self.de_gate1(fuse)
        x = self.nonlinearity(self.de_model2(x) + self.de_shortcut2(x)) * self.de_gate2(x)
        x = self.attn1(x, feature[3], mask)
        x = self.nonlinearity(self.de_model3(x) + self.de_shortcut3(x)) * self.de_gate3(x)
        x = self.nonlinearity(self.de_model4(x) + self.de_shortcut4(x)) * self.de_gate4(x)
        x = self.attn2(x, feature[1], mask)
        x = self.nonlinearity(self.de_model5(x) + self.de_shortcut5(x)) * self.de_gate5(x)
        x = self.model_out(x)


        return x


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]