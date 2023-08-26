import torch
from torch import nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from ..utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
class US(nn.Module):
    """Up-sampling block
    """

    def __init__(self, num_feat, scale):
        super(US, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 1)
        # plugin pixel attention
        self.pa_conv = nn.Conv2d(num_feat, num_feat, 1)
        self.pa_sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x_ = self.conv1(F.interpolate(x, scale_factor=self.scale, mode='nearest'))
        x_ = self.lrelu(x_)
        z = self.pa_conv(x_)
        z = self.pa_sigmoid(z)
        z = torch.mul(x_, z) + x_
        z = self.conv2(z)
        out = self.lrelu(z)
        return out


class RPA(nn.Module):
    """Residual pixel-attention block
    """

    def __init__(self, num_feat):
        super(RPA, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 1)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, 1)
        self.conv3 = nn.Conv2d(num_feat * 4, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialize layer weights
        for layer in [self.conv1, self.conv2, self.conv3, self.conv3]:
            nn.init.kaiming_normal_(layer.weight)
            # scale factor emperically set to 0.1
            layer.weight.data *= 0.1

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        z = self.conv3(z)
        z = self.sigmoid(z)
        z = x * z + x
        z = self.conv4(z)
        out = self.lrelu(z)
        return out


@ARCH_REGISTRY.register()
class Generator_RPA(nn.Module):
    """The generator of A-ESRGAN is comprised of residual pixel-attention(PA) blocks
     and consequent up-sampling blocks.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=20):
        super(Generator_RPA, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # residual pixel-attention blocks
        self.rpa = nn.Sequential(
            OrderedDict(
                [("rpa{}".format(i), RPA(num_feat=num_feat)) for i in range(num_block)]))
        # up-sampling blocks with pixel-attention
        num_usblock = ceil(log2(scale))
        self.us = nn.Sequential(
            OrderedDict(
                [("us{}".format(i), US(num_feat=num_feat, scale=2)) for i in range(num_usblock)]))
        self.conv2 = nn.Conv2d(num_feat, num_feat // 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat // 2, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z_ = self.rpa(z)
        z = z + z_
        z = self.us(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        out = self.conv3(z)
        return out


class Generator_RRDB(nn.Module):
    """The generator of A-ESRGAN is comprised of Residual in Residual Dense Blocks(RRDBs) as
    ESRGAN. And we employ pixel unshuffle to input feature before the network.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(Generator_RRDB, self).__init__()
        self.scale = scale
        num_in_ch *= 16 // (scale)**2
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # embed rrdb network here
        self.rrdb = nn.Sequential(
            OrderedDict(
                [("rrdb{}".format(i), RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch)) for i in range(num_block)]))
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # conv3 & conv4 are for up-sampling
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv6 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        z = pixel_unshuffle(x, scale=4 // self.scale)
        z = self.conv1(z)
        z_ = self.conv2(self.rrdb(z))
        z = z + z_
        z = self.lrelu(self.conv3(F.interpolate(z, scale_factor=2, mode='nearest')))
        z = self.lrelu(self.conv4(F.interpolate(z, scale_factor=2, mode='nearest')))
        z = self.conv6(self.lrelu(self.conv5(z)))
        return z