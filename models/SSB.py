"""
Copyright 2025, Yuan Chen, BHU.
Multilevel Reduced Semantic Segmentation Block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as enn

# from .SeS_seg import *
from models.network import *


class SSB(nn.Module):  # nn.Module
    """Lite R-ASPP style segmentation network."""

    def __init__(self, config, aspp='aspp', num_filters=128, num_class=8, use_softmax=True):
        """Initialize a new segmentation model.

        Keyword arguments:
        num_classes -- number of output classes (e.g., 19 for Cityscapes)
        trunk -- the name of the trunk to use ('mobilenetv3_large', 'mobilenetv3_small')
        num_filters -- the number of filters in the segmentation head
        """
        super(SSB, self).__init__()
        # mobilenetv3_large
        s2_ch = 16
        s4_ch = 24
        high_level_ch = 960

        self.aspp = aspp
        self.use_softmax = use_softmax
        self.use_e2 = config['network']['aspp']['use_e2']
        self.num_class = num_class
        if self.use_e2:
            self.feat_dim = config['network']['e2cnn']['feat_dim']
            self.gspace = gspaces.Rot2dOnR2(N=config['network']['e2cnn']['nbr_rotations'])
        # Reduced atrous spatial pyramid pooling
        if (self.aspp == 'asppv3'):
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=12, padding=12),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv3 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=36, padding=36),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            aspp_out_ch = num_filters * 4
        elif (self.aspp == 'asppv3+'):
            if self.use_e2:
                # triv_in_type = enn.FieldType(self.gspace,self.feat_dim[2]* [self.gspace.trivial_repr])
                self.in_type = enn.FieldType(self.gspace, self.feat_dim[2] * [self.gspace.regular_repr])
                self.out_type = enn.FieldType(self.gspace, self.feat_dim[2] * [self.gspace.regular_repr])
                self.aspp_conv1 = R2conv_bn(1, self.in_type, self.out_type, stride=1, dilation=1, mode='zero',
                                            padding=0)
                self.aspp_conv2 = R2conv_bn(3, self.in_type, self.out_type, stride=1, dilation=6, mode='zero',
                                            padding=6)
                self.aspp_conv3 = R2conv_bn(3, self.in_type, self.out_type, stride=1, dilation=12, mode='zero',
                                            padding=12)
                self.aspp_conv4 = R2conv_bn(3, self.in_type, self.out_type, stride=1, dilation=18, mode='zero',
                                            padding=18)
                self.aspp_pool = nn.Sequential(enn.PointwiseAdaptiveAvgPool(self.in_type, 1),
                                               R2conv_bn(1, self.in_type, self.out_type, stride=1, mode='zero',
                                                         padding=0))
                # self._in_type =enn.FieldType(self.gspace, self.feat_dim[2] *5* [self.gspace.regular_repr])
                conv_triv_repr = enn.FieldType(self.out_type.gspace,
                                               self.num_class * [self.out_type.gspace.trivial_repr])
                self.conv_up1 = enn.R2Conv(self.in_type, conv_triv_repr, kernel_size=1, padding=0, stride=1, bias=False)
            else:
                self.aspp_conv1 = nn.Sequential(
                    nn.Conv2d(config['network']['e2cnn']['out_dim'][2], num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(inplace=True),
                )
                self.aspp_conv2 = nn.Sequential(
                    nn.Conv2d(config['network']['e2cnn']['out_dim'][2], num_filters, 3, dilation=6, padding=6,
                              bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(inplace=True),
                )
                self.aspp_conv3 = nn.Sequential(
                    nn.Conv2d(config['network']['e2cnn']['out_dim'][2], num_filters, 3, dilation=12, padding=12,
                              bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(inplace=True),
                )
                self.aspp_conv4 = nn.Sequential(
                    nn.Conv2d(config['network']['e2cnn']['out_dim'][2], num_filters, 3, dilation=18, padding=18,
                              bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(inplace=True),
                )
                self.aspp_pool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(config['network']['e2cnn']['out_dim'][2], num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(inplace=True),
                )
                aspp_out_ch = num_filters * 5
                self.conv_up1 = nn.Conv2d(aspp_out_ch, self.num_class, kernel_size=1)


        elif ('mr' in self.aspp):
            if self.use_e2:
                self.in_type = enn.FieldType(self.gspace, self.feat_dim[2] * [self.gspace.regular_repr])
                self.out_type = self.in_type
                self.aspp_conv1 = R2conv_bn(1, self.in_type, self.out_type, stride=1, dilation=1, mode='zero',
                                            padding=0)
                self.aspp_conv2 = nn.Sequential(
                    enn.PointwiseAdaptiveAvgPool(self.in_type, 1),  # pytorch
                    enn.R2Conv(self.in_type, self.out_type, kernel_size=1, padding=0, stride=1, bias=False),
                    enn.PointwiseNonLinearity(self.in_type, function='p_sigmoid'))  # sigmoid
                self.in_type = enn.FieldType(self.gspace, self.feat_dim[0] * [self.gspace.regular_repr])
                self.convs2 = enn.R2Conv(self.in_type, self.in_type, kernel_size=1, padding=0, stride=1, bias=False)  #
                self.in_type = enn.FieldType(self.gspace, self.feat_dim[1] * [self.gspace.regular_repr])
                self.convs4 = enn.R2Conv(self.in_type, self.in_type, kernel_size=1, padding=0, stride=1, bias=False)
                self.in_type1 = self.out_type
                self.conv_up1 = enn.R2Conv(self.in_type1, self.in_type1, kernel_size=1, padding=0, stride=1, bias=False)
                self.in_type2 = enn.FieldType(self.gspace,
                                              (self.feat_dim[2] + self.feat_dim[1]) * [self.gspace.regular_repr])
                self.out_type = enn.FieldType(self.gspace, self.feat_dim[2] * [self.gspace.regular_repr])
                self.conv_up2 = R2conv_bn(1, self.in_type2, self.out_type, stride=1, dilation=1, mode='zero', padding=0)
                self.in_type3 = enn.FieldType(self.gspace,
                                              (self.feat_dim[2] + self.feat_dim[0]) * [self.gspace.regular_repr])
                self.out_type = enn.FieldType(self.gspace, self.feat_dim[2] * [self.gspace.regular_repr])
                self.conv_up3 = R2conv_bn(1, self.in_type3, self.out_type, stride=1, dilation=1, mode='zero', padding=0)
                conv_triv_repr = enn.FieldType(self.out_type.gspace, self.num_class * [
                    self.out_type.gspace.trivial_repr])
                self.last = enn.R2Conv(self.out_type, conv_triv_repr, kernel_size=1, padding=0, stride=1, bias=False)
            else:
                self.aspp_conv1 = nn.Sequential(
                    nn.Conv2d(config['network']['e2cnn']['out_dim'][2], num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(inplace=True),
                )
                self.aspp_conv2 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(config['network']['e2cnn']['out_dim'][2], num_filters, 1, bias=False),
                    nn.Sigmoid(),
                )
                aspp_out_ch = num_filters
                self.convs2 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
                self.convs4 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
                self.conv_up1 = nn.Conv2d(aspp_out_ch, num_filters, kernel_size=1)
                self.conv_up2 = ConvBnRelu(num_filters + 64, num_filters, kernel_size=1)
                self.conv_up3 = ConvBnRelu(num_filters + 32, num_filters, kernel_size=1)
                self.last = nn.Conv2d(num_filters, self.num_class, kernel_size=1)

    def forward(self, x):
        s2, s4, final = x
        if self.aspp == 'asppv3+':
            if self.use_e2:
                self.interpolate = enn.R2Upsampling(self.in_type, size=final.shape[2:])

                aspp = self.aspp_conv1(final) + self.aspp_conv2(final) + self.aspp_conv3(final) + self.aspp_conv4(
                    final) + self.interpolate(self.aspp_pool(final))  # cat 改成+有问题，伪影严重
                y = self.conv_up1(aspp)
            else:
                aspp = torch.cat([
                    self.aspp_conv1(final),
                    self.aspp_conv2(final),
                    self.aspp_conv3(final),
                    self.aspp_conv4(final),
                    F.interpolate(self.aspp_pool(final), size=final.shape[2:]),
                ], 1)
                y = self.conv_up1(aspp)
        else:
            if self.aspp == 'asppv3':
                aspp = torch.cat([
                    self.aspp_conv1(final),
                    self.aspp_conv2(final),
                    self.aspp_conv3(final),
                    F.interpolate(self.aspp_pool(final), size=final.shape[2:]),
                ], 1)
            elif 'mr' in self.aspp:
                if self.use_e2:
                    # downsample to 120
                    aspp = enn.GeometricTensor(self.aspp_conv1(final).tensor * self.aspp_conv2(final).tensor,
                                               final.type)
                    y = self.conv_up1(aspp)
                    self.interpolate = enn.R2Upsampling(self.in_type, size=y.shape[2:])
                    y = enn.tensor_directsum([y, self.convs4(self.interpolate(s4))])
                    y = self.conv_up2(y)
                    self.interpolate = enn.R2Upsampling(
                        enn.FieldType(self.gspace, self.feat_dim[0] * [self.gspace.regular_repr]), size=y.shape[2:])
                    y = enn.tensor_directsum([y, self.convs2(self.interpolate(s2))])
                    y = self.conv_up3(y)
                    y = self.last(y)
                    return y
                else:
                    aspp = self.aspp_conv1(final) * F.interpolate(
                        self.aspp_conv2(final),
                        final.shape[2:],
                        mode='bilinear',
                        align_corners=True
                    )
            y = self.conv_up1(aspp)
            # y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)  # 256*270*480
            s4 = F.interpolate(s4, size=y.shape[2:], mode='bilinear', align_corners=False)  # 256*270*480

            y = torch.cat([y, self.convs4(s4)], 1)  # 320*270*480
            y = self.conv_up2(y)  # 256*270*480
            s2 = F.interpolate(s2, size=y.shape[2:], mode='bilinear', align_corners=False)  # 256*540*960

            y = torch.cat([y, self.convs2(s2)], 1)  # 288*540*960
            y = self.conv_up3(y)  # 256*540*960
            y = self.last(y)  # 19*540*960
            # y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)#19*1080*1920
        if self.use_softmax:
            y = nn.functional.softmax(y, dim=1)
        return y


class MobileV3Large(SSB):
    """MobileNetV3-Large segmentation network."""
    model_name = 'mobilev3large-lraspp'

    def __init__(self, num_classes, **kwargs):
        super(MobileV3Large, self).__init__(
            num_classes,
            trunk='mobilenetv3_large',
            **kwargs
        )


class MobileV3Small(SSB):
    """MobileNetV3-Small segmentation network."""
    model_name = 'mobilev3small-lraspp'

    def __init__(self, num_classes, **kwargs):
        super(MobileV3Small, self).__init__(
            num_classes,
            trunk='mobilenetv3_small',
            **kwargs
        )


class ConvBnRelu(nn.Module):
    """Convenience layer combining a Conv2d, BatchNorm2d, and a ReLU activation.

    Original source of this code comes from
    https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 norm_layer=nn.BatchNorm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
