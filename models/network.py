import collections
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import nn as enn
from itertools import repeat
from typing import Tuple

from utils.utils import gather_nd, torch_gather_nd


class conv_bn(nn.Module):
    """
    (convolution => [BN] )
    """

    def __init__(self, kernel_size, in_channels, out_channels, stride, bias=False, relu=True):
        super().__init__()
        self.relu = relu
        if stride == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', stride=stride,
                                  bias=bias)
        elif stride > 1:
            self.conv = Conv2dSame(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, affine=False, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class deform_conv(nn.Module):
    """
    (convolution => [BN] )
    """

    def __init__(self, kernel_size, in_channels, out_channels, stride, dilation_rate=1, deform_type='u', modulated=True,
                 bias=False, relu=True, bn=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.modulated = modulated
        self.offset_num = kernel_size ** 2  # offset neigborhood size
        self.relu = relu
        self.bn = bn
        if deform_type == 'a':
            print('coming soon')
        elif deform_type == 'h':
            print('coming soon')
        elif deform_type == 'u':
            self.offset = nn.Conv2d(in_channels, self.offset_num * 2, kernel_size=kernel_size, padding='valid',
                                    stride=stride, bias=True)

        if modulated:
            self.amplitude = nn.Conv2d(in_channels, self.offset_num, kernel_size=kernel_size, padding='valid',
                                       stride=stride, bias=True)
            self.sigmoid = nn.Sigmoid()

        self.conv3d = nn.Conv3d(in_channels, out_channels, (1, 1, kernel_size ** 2), stride=1, padding='valid',
                                bias=bias)
        # self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='valid',stride=1,bias=bias)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels, affine=False, momentum=0.01)

            # weight init
        self._init_weight()

    def _pad_intput(self, input):  # only seen dilation=1 so far?
        dilated_filter_size = self.kernel_size + (self.kernel_size - 1) * (self.dilation_rate - 1)
        padding_list = []
        for i in range(2):
            same_output = (input.shape[i + 2] + self.stride - 1) // self.stride
            valid_output = (input.shape[i + 2] - dilated_filter_size + self.stride) // self.stride
            p = dilated_filter_size - 1
            p_0 = p // 2
            if same_output == valid_output:
                padding_list = padding_list + [0, 0]
            else:
                padding_list = padding_list + [p_0, p - p_0]
        input = F.pad(input, (padding_list[2], padding_list[3], padding_list[0], padding_list[1], 0, 0, 0, 0))
        return input

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map

        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [i for i in feature_map_size[0: 2]]

        y, x = torch.meshgrid(torch.as_tensor(range(feat_h)), torch.as_tensor(range(feat_w)))
        x, y = x.reshape(1, 1, feat_h, feat_w), y.reshape(1, 1, feat_h, feat_w)  # shape [1, 1, h, w]
        x, y = [nn.Unfold(self.kernel_size, self.dilation_rate, 0, stride=self.stride)(m.float()) for m in
                [x, y]]  # extract conv patches shape [1,c*filter_h * filter_w,blocks ]
        x, y = x.transpose(1, 2).reshape(1, feat_h - 2, feat_w - 2, -1), y.transpose(1, 2).reshape(1, feat_h - 2,
                                                                                                   feat_w - 2, -1)
        # shape [1, out_h, out_w, c*filter_h * filter_w]
        y, x = y.to(self.device), x.to(self.device)
        return y, x

    def _init_weight(self):  # zero init
        for m in self.modules():  # 继承nn.Module的方法
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        self.device = x.device
        pad = self._pad_intput(x)  # padding =same
        in_h, in_w = pad.shape[-2], pad.shape[-1]
        offset = self.offset(pad)
        if self.modulated:
            amplitude = self.amplitude(pad)
            amplitude = self.sigmoid(amplitude)

        # get x, y axis offset
        offset = offset.reshape(-1, self.offset_num, 2, offset.shape[-2], offset.shape[-1])
        offset = offset.permute(0, 3, 4, 1, 2)
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]
        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])  # PS pooling

        # add offset
        y, x = y + y_off, x + x_off
        y = torch.clip(y, 0, in_h - 1)
        x = torch.clip(x, 0, in_w - 1)
        # get four coordinates of points around (x, y)
        y0, x0 = torch.floor(y), torch.floor(x)
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [torch.clip(m, 0, in_h - 1) for m in [y0, y1]]
        x0, x1 = [torch.clip(m, 0, in_w - 1) for m in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [gather_nd(pad.permute(0, 2, 3, 1), torch.stack(i, dim=-1), batch_dims=1) for i in indices]
        # p0, p1, p2, p3 = [torch_gather_nd(pad.permute(0, 2, 3, 1), torch.stack(i, dim=-1), 1) for i in indices]#train dcn
        # x0, x1, y0, y1 = [i.float() for i in [x0, x1, y0, y1]]
        # weights (modulated dcn)
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [i.unsqueeze(-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3

        if self.modulated:
            pixels = pixels.permute(0, 3, 1, 2, 4) * amplitude.unsqueeze(-1)  # 2*120*120*9*128
        pixels = pixels.permute(0, 4, 2, 3, 1)  # batch*channel*depth*h*w
        out = self.conv3d(pixels).squeeze(-1)

        if self.relu:
            out = self.relu(out)
        if self.bn:
            out = self.bn(out)

        return out


class R2conv_bn(enn.EquivariantModule):
    def __init__(self, kernel_size, in_type, out_type, stride, mode='same', padding=1, dilation=1, bias=False):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        if mode == 'same':
            self.conv = R2Conv2dSame(in_type=in_type, out_type=out_type, kernel_size=kernel_size, stride=stride,
                                     dilation=dilation, bias=bias)
        else:
            self.conv = enn.R2Conv(in_type=in_type, out_type=out_type, kernel_size=kernel_size, padding=padding,
                                   stride=stride, dilation=dilation, bias=bias)
        self.bn = enn.InnerBatchNorm(out_type)
        self.relu = enn.ReLU(out_type, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        b, out_type_size, ho, wo = self.conv.evaluate_output_shape(input_shape)
        return b, out_type_size, ho, wo


class R2conv_bn_act(R2conv_bn):
    def __init__(self, kernel_size, in_type, out_type, stride, mode='same', padding=1, bias=False):
        super().__init__(kernel_size, in_type, out_type, stride, mode='same', padding=1, bias=False)
        self.stride = stride
        self.kernel_size = kernel_size
        if mode == 'same':
            self.conv = R2Conv2dSame(in_type=in_type, out_type=out_type, kernel_size=kernel_size, stride=stride,
                                     bias=bias)
        else:
            self.conv = enn.R2Conv(in_type=in_type, out_type=out_type, kernel_size=kernel_size, padding=padding,
                                   stride=stride, bias=bias)
        self.bn = enn.InnerBatchNorm(out_type)
        self.act = enn.PointwiseNonLinearity(out_type, 'p_elu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def _ntuple(n):
    """Copied from PyTorch since it's not importable as an internal function

    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/utils.py#L6
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class Conv2dSame(nn.Module):
    """Manual convolution with same padding (can use stride>1

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(stride, stride),
            dilation=dilation,
            **kwargs)  # padding=0
        self.dilation = dilation
        self.stride = stride
        # Setup internal representations
        self.kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        # self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        # for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
        #     total_padding = d * (k - 1)
        #     left_pad = total_padding // 2
        #     self._reversed_padding_repeated_twice[2 * i] = left_pad
        #     self._reversed_padding_repeated_twice[2 * i + 1] = (
        #             total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        H, W = imgs.shape[-2:]
        # effective_filter_size_H = (self.kernel_size_[0] - 1) * self.dilation + 1
        out_H = (H + self.stride - 1) // self.stride
        out_W = (W + self.stride - 1) // self.stride
        # padding_needed = max(0, (out_H - 1) * stride[0] + effective_filter_size_H -input_rows)
        padding_H = max(0, (out_H - 1) * self.stride +
                        (self.kernel_size_[0] - 1) * self.dilation + 1 - H)
        H_odd = (padding_H % 2 != 0)
        padding_W = max(0, (out_W - 1) * self.stride +
                        (self.kernel_size_[1] - 1) * self.dilation + 1 - W)
        W_odd = (padding_W % 2 != 0)
        # if rows_odd or cols_odd:
        padded = F.pad(imgs, [padding_W // 2, padding_W // 2 + int(W_odd), padding_H // 2, padding_H // 2 + int(H_odd)])
        return self.conv(padded)


class FConv2dSame(nn.Module):
    """Manual convolution with same padding (can use stride>1
    """

    def __init__(self, input, weight, bias=None, stride=1, dilation=1, groups=1, **kwargs):
        super().__init__()
        self.input = input,
        self.weight = weight,
        self.stride = stride,
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        # Setup internal representations
        self.kernel_size_ = weight.shape[-2:]

    def forward(self, imgs):
        H, W = imgs.shape[-2:]
        out_H = (H + self.stride - 1) // self.stride
        out_W = (W + self.stride - 1) // self.stride
        padding_H = max(0, (out_H - 1) * self.stride +
                        (self.kernel_size_[0] - 1) * self.dilation + 1 - H)
        H_odd = (padding_H % 2 != 0)
        padding_W = max(0, (out_W - 1) * self.stride +
                        (self.kernel_size_[1] - 1) * self.dilation + 1 - W)
        W_odd = (padding_W % 2 != 0)
        # if rows_odd or cols_odd:
        padded = F.pad(imgs, [padding_W // 2, padding_W // 2 + int(W_odd), padding_H // 2, padding_H // 2 + int(H_odd)])
        return F.conv2d(input=padded, weight=self.weight, stride=self.stride, groups=self.groups, bias=self.bias)


class MaxPool2dSame(nn.Module):
    """
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

        # Setup internal representations
        kernel_size_ = _pair(self.kernel_size)
        dilation_ = _pair(1)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, x):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(x, self._reversed_padding_repeated_twice)
        x = self.pool(padded)
        return x


def avg_pool2d(input, kernel_size, padding=0, dilation=1, stride=1):
    def _output_size(i):
        out_i = (i + 2 * padding - kernel_size - (kernel_size - 1) * (dilation - 1)) / stride + 1
        return int(out_i)

    B, C, H, W = input.shape
    all_pat = nn.Unfold(kernel_size, dilation, padding, stride=stride)(input)
    out_H, out_W = _output_size(H), _output_size(W)
    all_pat = all_pat.reshape(B, C, kernel_size ** 2, out_H, out_W)
    avgpool = torch.mean(all_pat, dim=2)
    return avgpool


class R2Conv2dSame(enn.EquivariantModule):
    """Manual E2 convolution with same padding (can use stride>1
    """

    def __init__(self, in_type, out_type, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = enn.R2Conv(in_type,
                               out_type,
                               kernel_size=kernel_size,
                               stride=stride,
                               dilation=dilation,
                               **kwargs)  # padding=0
        self.dilation = dilation
        self.stride = stride
        # Setup internal representations
        self.kernel_size_ = _pair(kernel_size)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned
        All shapes (input/output) are ``(N, C, W, H)`` convention
        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        H, W = imgs.shape[-2:]
        out_H = (H + self.stride - 1) // self.stride
        out_W = (W + self.stride - 1) // self.stride
        padding_H = max(0, (out_H - 1) * self.stride +
                        (self.kernel_size_[0] - 1) * self.dilation + 1 - H)
        H_odd = (padding_H % 2 != 0)
        padding_W = max(0, (out_W - 1) * self.stride +
                        (self.kernel_size_[1] - 1) * self.dilation + 1 - W)
        W_odd = (padding_W % 2 != 0)
        self.padding_W = [padding_W // 2, padding_W // 2 + int(W_odd)]
        self.padding_H = [padding_H // 2, padding_H // 2 + int(H_odd)]
        pad_tensor = F.pad(imgs.tensor, [self.padding_W[0], self.padding_W[1], self.padding_H[0], self.padding_H[1]])
        imgs_pad = imgs
        imgs_pad.tensor = pad_tensor
        return self.conv(imgs_pad)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size

        b, c, hi, wi = input_shape

        ho = math.floor(
            (hi + self.padding_H[0] + self.padding_H[1] - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        wo = math.floor(
            (wi + self.padding_W[0] + self.padding_W[1] - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

        return b, self.out_type.size, ho, wo
