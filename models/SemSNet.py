"""
Copyright 2025, Yuan Chen, BHU.
"""
from __future__ import print_function

import cv2
import numpy as np
import torch
from e2cnn import gspaces
from e2cnn import nn as enn

from models.network import *
from utils.ts_geom import interpolate
from utils.utils import *
from .SSB import SSB


class BaseNet(nn.Module):
    def __init__(self, config, dropout_rate=0, seed=None, epsilon=1e-5):
        super(BaseNet, self).__init__()
        self.det_config = config['network']['det']
        self.is_training = config['is_training']
        self.interpolate = interpolate
        # Dropout rate
        self.dropout_rate = dropout_rate
        # Seed for randomness
        self.seed = seed
        # The epsilon paramater in BN layer.
        self.bn_epsilon = epsilon
        # output
        self.endpoints = []
        self.layers = []
        # setting
        self.nbr_rotations = config['network']['e2cnn']['nbr_rotations']
        # dim_reduction=nbr_rotations #2->* nbr_rotations
        self.feat_dim = config['network']['e2cnn']['feat_dim']
        self.out_dim = config['network']['e2cnn']['out_dim']
        self.grouppool = config['network']['e2cnn']['group_pool']

        self.kernel_size = 3
        # field type
        self.gspace = gspaces.Rot2dOnR2(N=self.nbr_rotations)
        self.triv_in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        self.in_type = enn.FieldType(self.gspace,
                                     (self.feat_dim[0]) * [self.gspace.regular_repr])
        self._in_type = self.in_type

        # net construct
        self.out_type = enn.FieldType(self._in_type.gspace, self.feat_dim[0] * [self._in_type.gspace.regular_repr])
        self.conv_bn0 = R2conv_bn(self.kernel_size, self.triv_in_type, self.in_type, stride=1, mode='zero',
                                  padding=1)  # conv0 dim0
        self.conv1 = enn.R2Conv(self.in_type, self.out_type, kernel_size=self.kernel_size, padding=1, stride=1,
                                bias=False)  # conv1 dim0
        self._in_type = self.out_type
        self.bn1 = enn.InnerBatchNorm(self._in_type, affine=False, momentum=0.01)  # no gamma & beta
        self.relu1 = enn.ReLU(self._in_type, inplace=True)
        # trivial representation
        conv1_triv_repr = enn.FieldType(self.gspace, self.out_dim[0] * [self.gspace.trivial_repr])
        self.conv1_out = enn.R2Conv(self.in_type, conv1_triv_repr, kernel_size=self.kernel_size, padding=1, stride=1,
                                    bias=False)

        self.out_type = enn.FieldType(self.gspace, self.feat_dim[1] * [self.gspace.regular_repr])
        self.conv_bn2 = R2conv_bn(self.kernel_size, self._in_type, self.out_type, stride=2, mode='zero',
                                  padding=1)  # conv2 dim1
        self._in_type = self.out_type

        conv3_triv_repr = enn.FieldType(self.out_type.gspace, self.out_dim[1] * [self.out_type.gspace.trivial_repr])
        self.conv3_out = enn.R2Conv(self.out_type, conv3_triv_repr, kernel_size=self.kernel_size, padding=1, stride=1,
                                    bias=False)
        self.out_type = enn.FieldType(self.gspace, self.feat_dim[1] * [self.gspace.regular_repr])
        self.conv3 = enn.R2Conv(self._in_type, self.out_type, kernel_size=self.kernel_size, padding=1, stride=1,
                                bias=False)  # conv3 dim1
        self._in_type = self.out_type
        self.bn3 = enn.InnerBatchNorm(self._in_type, affine=False, momentum=0.01)
        self.relu3 = enn.ReLU(self._in_type, inplace=True)
        self.out_type = enn.FieldType(self.gspace, self.feat_dim[2] * [self.gspace.regular_repr])
        self.conv_bn4 = R2conv_bn(self.kernel_size, self._in_type, self.out_type, 2, mode='zero',
                                  padding=1)  # conv4 dim2
        self._in_type = self.out_type

        self.conv_bn5 = R2conv_bn(self.kernel_size, self._in_type, self._in_type, 1, mode='zero',
                                  padding=1)  # conv5 dim2

        self.conv_bn6_0 = R2conv_bn(self.kernel_size, self._in_type, self._in_type, 1, mode='zero',
                                    padding=1)  # conv6 dim2
        self.conv_bn6_1 = R2conv_bn(self.kernel_size, self._in_type, self._in_type, 1, mode='zero',
                                    padding=1)  # conv7 dim2
        self.out_type = self._in_type
        conv6_triv_repr = enn.FieldType(self.out_type.gspace, self.out_dim[2] * [self.out_type.gspace.trivial_repr])
        self.conv6 = enn.R2Conv(self.out_type, conv6_triv_repr, kernel_size=self.kernel_size, padding=1, stride=1,
                                bias=False)  # conv8 dim2

        self.moving_instance_max = {}
        self.moving_instance_max['conv1'] = torch.tensor([1.], device='cuda')  # moving avg
        self.moving_instance_max['conv3'] = torch.tensor([1.], device='cuda')
        self.moving_instance_max['conv6'] = torch.tensor([1.], device='cuda')

    def _init_weight(self):  # zero init
        for m in self.modules():  # 继承nn.Module的方法
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)  # xavier_uniform_
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def network(self, x):
        # net forward
        x = self.conv_bn0(x)
        x = self.conv1(x)
        conv1 = self.conv1_out(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv_bn2(x)
        x = self.conv3(x)
        conv3 = self.conv3_out(x)

        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv_bn4(x)

        conv5 = self.conv_bn5(x)
        x = self.conv_bn6_0(conv5)
        x = self.conv_bn6_1(x)

        conv6 = self.conv6(x)  # 128*120*120
        return conv1, conv3, conv6

    def forward(self, input):
        if isinstance(input, dict):
            x = input['data']
        else:
            x = input
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        ori_h = x.shape[-2]
        ori_w = x.shape[-1]

        x = enn.GeometricTensor(x, self.triv_in_type)
        # net forward
        conv1, conv3, conv6 = self.network(x)

        if self.det_config['weight'] > 0:
            dense_feat_map = conv6.tensor  # to tensor
            kpt_n = self.det_config['kpt_n']

            if self.det_config['multi_level']:
                comb_names = ['conv1', 'conv3', 'conv6']
                ksize = [3, 2, 1]
                comb_weights = torch.tensor([1., 2., 3.])
                comb_weights = comb_weights / torch.sum(comb_weights)
            else:
                comb_names = ['conv6']
                ksize = [1]

            sum_det_score_map = None
            det_score_maps = []
            for idx, tmp_name in enumerate(comb_names):
                tmp_feat_map = locals()[tmp_name].tensor  # to tensor
                alpha, beta = self.peakiness_score(tmp_feat_map, ksize=3,
                                                   dilation=ksize[idx], name=tmp_name, det_config=self.det_config)
                score_vol = alpha * beta
                det_score_map = torch.max(score_vol, dim=1, keepdims=True)[0]  # channel
                det_score_map = F.interpolate(det_score_map, (ori_h, ori_w), mode='bilinear',
                                              align_corners=False)  # resize
                det_score_map = comb_weights[idx] * det_score_map

                if idx == 0:
                    sum_det_score_map = det_score_map
                else:
                    sum_det_score_map = sum_det_score_map + det_score_map  # multi-level score map add
                det_score_maps.append(det_score_map.permute(0, 2, 3, 1))

            det_score_map = sum_det_score_map

            # tf style
            det_score_map = det_score_map.permute(0, 2, 3, 1)
            det_kpt_inds, det_kpt_scores, drop_list = self.extract_kpts(
                det_score_maps, k=kpt_n,
                score_thld=self.det_config['score_thld'], edge_thld=self.det_config['edge_thld'],
                nms_size=self.det_config['nms_size'], eof_size=self.det_config['eof_mask'])

            if self.det_config['kpt_refinement']:
                offsets = self.kpt_refinement(det_score_map).squeeze(-2)  # x.tensor.permute(0,2,3,1)
                offsets = gather_nd(offsets, det_kpt_inds, batch_dims=1)
                det_kpt_inds = det_kpt_inds + offsets
                det_kpt_inds[..., 0] = torch.clip(det_kpt_inds[..., 0], 0, det_score_map.shape[1] - 1)
                det_kpt_inds[..., 1] = torch.clip(det_kpt_inds[..., 1], 0, det_score_map.shape[2] - 1)
            else:
                det_kpt_inds = det_kpt_inds.float()

            det_kpt_coords = torch.stack([det_kpt_inds[:, :, 1], det_kpt_inds[:, :, 0]], dim=-1)
            det_kpt_ncoords = torch.stack([(det_kpt_coords[:, :, 0] - ori_w / 2) / (ori_w / 2),
                                           (det_kpt_coords[:, :, 1] - ori_h / 2) / (ori_h / 2)], dim=-1)

            # torch style
            if self.is_training:
                """training mode"""
                det_score_map = det_score_map.permute(0, 3, 1, 2)
                self.endpoints = [dense_feat_map, det_score_map]

            else:
                """inference mode"""

                descs = F.normalize(interpolate(det_kpt_inds / 4, dense_feat_map), p=self.det_config['norm_type'],
                                    dim=1)
                if self.det_config['norm_type'] == 1:
                    descs = torch.sqrt(descs)
                self.endpoints = [det_kpt_coords, descs, det_kpt_scores]  # return [kpts descs scores]
                return self.endpoints
            kpt_feat = self.interpolate(det_kpt_inds / 4, dense_feat_map)
        else:
            # use original kpt position
            kpt_coord = input['kpt_coord']
            kpt_inds = torch.stack([kpt_coord[:, :, 1], kpt_coord[:, :, 0]], dim=-1)
            hs = conv6.shape[-2:].to(kpt_inds.device).reshape(1, 1, 2) / 2
            kpt_inds = kpt_inds * hs + hs
            kpt_feat = self.interpolate(kpt_inds, conv6)

        return det_kpt_coords, kpt_feat, self.endpoints

    def peakiness_score(self, inputs, ksize=3, dilation=1, name='conv', det_config=None):
        # torch style tensor
        decay = 0.99

        if self.is_training:
            instance_max = torch.max(inputs)
            self.moving_instance_max[name] = decay * self.moving_instance_max[name] + (
                    1 - decay) * instance_max  # self.moving_instance_max-(1 - decay) * (self.moving_instance_max - instance_max)

            self.moving_instance_max[name] = self.moving_instance_max[name].detach()

            inputs = inputs / self.moving_instance_max[name]
        else:
            inputs = inputs / self.moving_instance_max[name].detach()

        pad_size = ksize // 2 + (dilation - 1)
        pad_inputs = F.pad(inputs, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        avg_spatial_inputs = avg_pool2d(pad_inputs, ksize, 0, dilation)

        if (det_config != None) & ('func' in det_config):
            if 'max' in det_config['func']:
                avg_channel_inputs = torch.max(inputs, dim=1, keepdims=True)[0]
            else:
                avg_channel_inputs = torch.mean(inputs, dim=1, keepdims=True)
            if 'sqrt' in det_config['func']:
                if 'b' in det_config['func']:
                    b = float(det_config['func'].split('b')[-1])
                else:
                    b = 1.9218120556728057
                alpha = squareplus(inputs - avg_spatial_inputs, b)
                beta = squareplus(inputs - avg_channel_inputs, b)  # 1.9218120556728057
            else:
                alpha = F.softplus(inputs - avg_spatial_inputs)
                beta = F.softplus(inputs - avg_channel_inputs)
        else:
            avg_channel_inputs = torch.mean(inputs, dim=1, keepdims=True)
            alpha = F.softplus(inputs - avg_spatial_inputs)
            beta = F.softplus(inputs - avg_channel_inputs)

        return alpha, beta

    def extract_kpts(self, score_map, k=256, score_thld=0, edge_thld=0, nms_size=3, eof_size=5):
        # tf style
        # bs,h, w =  score_map.shape[0],score_map.shape[1], score_map.shape[2]
        bs, h, w = score_map[0].shape[0], score_map[0].shape[1], score_map[0].shape[2]
        # score_map1=score_map[0]
        score_map = score_map[0] + score_map[1] + score_map[2]

        mask = score_map > score_thld
        if nms_size > 0:
            maxpool = MaxPool2dSame(nms_size, stride=1)
            nms_mask = maxpool(score_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # tf -> torch -> tf
            nms_mask = score_map == nms_mask
            mask = torch.logical_and(nms_mask, mask)
        if eof_size > 0:
            eof_mask = torch.ones(1, h - 2 * eof_size, w - 2 * eof_size, 1, device=mask.device)
            eof_mask = F.pad(eof_mask, (0, 0, eof_size, eof_size, eof_size, eof_size, 0, 0))
            eof_mask = eof_mask.bool()
            mask = torch.logical_and(eof_mask, mask)
        if edge_thld > 0:
            non_edge_mask = self.edge_mask(score_map, 1, dilation=3, edge_thld=edge_thld)
            mask = torch.logical_and(non_edge_mask, mask)

        # bs = score_map.shape[0]
        indices = []
        scores = []
        drop_i = []
        for i in range(bs):
            tmp_mask = mask[i].reshape(h, w)
            tmp_score_map = score_map[i].reshape(h, w)
            tmp_indices = torch.stack(torch.where(tmp_mask), dim=1)
            tmp_scores = gather_nd(tmp_score_map, tmp_indices)
            tmp_sample = torch.argsort(tmp_scores, descending=True)[0:k]  # select k points
            tmp_indices = torch.index_select(tmp_indices, 0, tmp_sample)
            tmp_scores = torch.index_select(tmp_scores, 0, tmp_sample)
            indices.append(tmp_indices)
            scores.append(tmp_scores)
        if bs > 1:  # avoid different target
            k = min(indices[0].shape[0], indices[1].shape[0])  # avoid different length
            indices[0] = indices[0][:k]
            indices[1] = indices[1][:k]
            scores[0] = scores[0][:k]
            scores[1] = scores[1][:k]
        indices = torch.stack(indices, dim=0)
        scores = torch.stack(scores, dim=0)
        return indices, scores, drop_i

    def kpt_refinement(self, inputs):
        # tf->torch style->tf
        h, w, n_channel = inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]
        inputs = inputs.permute(0, 3, 1, 2)
        di_filter = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], device=inputs.device).reshape(1, 1, 3, 3)
        dj_filter = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], device=inputs.device).reshape(1, 1, 3, 3)
        dii_filter = torch.tensor([[0, 1., 0], [0, -2., 0], [0, 1., 0]], device=inputs.device).reshape(1, 1, 3, 3)
        dij_filter = 0.25 * torch.tensor([[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]], device=inputs.device).reshape(1, 1,
                                                                                                                 3, 3)
        djj_filter = torch.tensor([[0, 0, 0], [1., -2., 1.], [0, 0, 0]], device=inputs.device).reshape(1, 1, 3, 3)

        dii_filter = torch.tile(dii_filter, (n_channel, 1, 1, 1))  # out_channel in_channel/groups H W
        dii = F.conv2d(inputs, dii_filter, None, stride=1, padding='same', groups=n_channel)

        dij_filter = torch.tile(dij_filter, (n_channel, 1, 1, 1))  # out_channel in_channel/groups H W
        dij = F.conv2d(inputs, dij_filter, None, stride=1, padding='same', groups=n_channel)

        djj_filter = torch.tile(djj_filter, (n_channel, 1, 1, 1))  # out_channel in_channel/groups H W
        djj = F.conv2d(inputs, djj_filter, None, stride=1, padding='same', groups=n_channel)

        det = dii * djj - dij * dij

        inv_hess_00 = torch.div(djj, det)
        inv_hess_01 = torch.div(-dij, det)
        inv_hess_11 = torch.div(dii, det)
        inv_hess_00[torch.isinf(inv_hess_00) | torch.isnan(inv_hess_00)] = 0.
        inv_hess_01[torch.isinf(inv_hess_01) | torch.isnan(inv_hess_01)] = 0.
        inv_hess_11[torch.isinf(inv_hess_11) | torch.isnan(inv_hess_11)] = 0.

        di_filter = torch.tile(di_filter, (n_channel, 1, 1, 1))
        di = F.conv2d(inputs, di_filter, None, stride=1, padding='same', groups=n_channel)
        dj_filter = torch.tile(dj_filter, (n_channel, 1, 1, 1))
        dj = F.conv2d(inputs, dj_filter, None, stride=1, padding='same', groups=n_channel)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)

        offset = torch.stack([step_i, step_j], dim=-1)
        offset1 = torch.clip(offset, -0.5, 0.5)

        return offset1.permute(0, 2, 3, 1, 4)

    def edge_mask(self, inputs, n_channel, dilation=1, edge_thld=5):
        # tf->torch style->tf
        # non-edge
        inputs = inputs.permute(0, 3, 1, 2)
        dii_filter = torch.tensor([[0, 1., 0], [0, -2., 0], [0, 1., 0]], device=inputs.device).reshape(1, 1, 3, 3)
        dij_filter = 0.25 * torch.tensor([[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]], device=inputs.device).reshape(1, 1,
                                                                                                                 3, 3)
        djj_filter = torch.tensor([[0, 0, 0], [1., -2., 1.], [0, 0, 0]], device=inputs.device).reshape(1, 1, 3, 3)

        pad_inputs = F.pad(inputs, (dilation, dilation, dilation, dilation), mode='constant', value=0)

        dii_filter = torch.tile(dii_filter, (1, n_channel, 1, 1))
        dii = F.conv2d(pad_inputs, dii_filter, bias=None, stride=1, padding='valid', dilation=dilation)

        dij_filter = torch.tile(dij_filter, (1, n_channel, 1, 1))
        dij = F.conv2d(pad_inputs, dij_filter, bias=None, stride=1, padding='valid', dilation=dilation)

        djj_filter = torch.tile(djj_filter, (1, n_channel, 1, 1))
        djj = F.conv2d(pad_inputs, djj_filter, bias=None, stride=1, padding='valid', dilation=dilation)

        det = dii * djj - dij * dij
        tr = dii + djj
        thld = (edge_thld + 1) ** 2 / edge_thld
        is_not_edge = torch.logical_and(tr * tr / det <= thld, det > 0)
        is_not_edge = is_not_edge.permute(0, 2, 3, 1)
        return is_not_edge

    def reload_config(self, config):
        self.det_config = config['network']['det']
        self.is_training = config['is_training']


class SemSNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.aspp_config = config['network']['aspp']
        # sem segmentation head
        if self.aspp_config['use_aspp']:
            self.gspace = gspaces.Rot2dOnR2(N=config['network']['e2cnn']['nbr_rotations'])
            self.in_type = enn.FieldType(self.gspace, self.feat_dim[2] * [self.gspace.regular_repr])
            self.out_type = enn.FieldType(self.gspace, int(self.feat_dim[2] / 4) * [self.gspace.regular_repr])
            if 'res' in self.aspp_config['type']:
                self.conv_bn_aspp6_0 = R2conv_bn(1, self.in_type, self.out_type, stride=1, mode='zero',
                                                 padding=0)
                self.conv_bn_aspp6_1 = R2conv_bn(1, self.out_type, self.out_type, stride=1, mode='zero', padding=0)
                self.conv_bn_aspp6_2 = R2conv_bn(1, self.out_type, self.in_type, stride=1, mode='zero',
                                                 padding=0)
            else:
                self.conv_bn_aspp6 = R2conv_bn(3, self.in_type, self.in_type, stride=1, mode='zero',
                                               padding=1)
                self.conv_bn_aspp7 = R2conv_bn(3, self.in_type, self.in_type, stride=1, mode='zero', padding=1)
            self.conv_bn_aspp8 = nn.Conv2d(config['network']['aspp']['num_class'], self.out_dim[2], kernel_size=3,
                                           padding='same', stride=1, bias=False)
            self.use_softmax = False if config['is_training'] else True
            self.aspp = SSB(config, self.aspp_config['type'], self.out_dim[2],
                            num_class=config['network']['aspp']['num_class'], use_softmax=False)  #
            self.bn1_1 = nn.BatchNorm2d(self.out_dim[0], affine=False, momentum=0.01)
            self.bn3_1 = nn.BatchNorm2d(self.out_dim[1], affine=False, momentum=0.01)

    def network(self, x):
        # net forward
        x = self.conv_bn0(x)
        x = self.conv1(x)
        conv1 = self.conv1_out(x)
        x = self.bn1(x)
        conv1_bn = self.relu1(x)
        x = self.conv_bn2(conv1_bn)
        x = self.conv3(x)
        conv3 = self.conv3_out(x)
        x = self.bn3(x)
        conv3_bn = self.relu3(x)
        x = self.conv_bn4(conv3_bn)
        conv5 = self.conv_bn5(x)
        x = self.conv_bn6_0(conv5)
        x = self.conv_bn6_1(x)
        conv6 = self.conv6(x)  # 128*120*120
        return conv1, conv3, conv1_bn, conv3_bn, conv5, conv6

    def forward(self, input):
        if isinstance(input, dict):
            x = input['data']
        else:
            x = input
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        ori_h = x.shape[-2]
        ori_w = x.shape[-1]

        x = enn.GeometricTensor(x, self.triv_in_type)
        # net forward
        conv1, conv3, conv1_bn, conv3_bn, conv5, conv6 = self.network(x)
        conv1, conv3, conv6 = conv1.tensor, conv3.tensor, conv6.tensor

        if (self.aspp_config['use_aspp']):
            x = self.conv_bn_aspp6_0(conv5)
            x = self.conv_bn_aspp6_1(x)
            x = self.conv_bn_aspp6_2(x)
            x = x + conv5
            seg_feat = self.aspp([conv1_bn, conv3_bn, conv5])
            seg_feat = seg_feat.tensor
            seg_feat = F.interpolate(seg_feat, (conv6.shape[-2], conv6.shape[-1]), mode='bilinear', align_corners=False)
            x = self.conv_bn_aspp8(seg_feat)
            conv6 = x + conv6  #

        down_ratio = ori_w / conv6.shape[-1]

        if self.det_config['weight'] > 0:
            dense_feat_map = conv6  # to tensor
            kpt_n = self.det_config['kpt_n']

            if self.det_config['multi_level']:
                comb_names = ['conv1', 'conv3', 'conv6']
                ksize = [3, 2, 1]
                comb_weights = torch.tensor([1., 2., 3.])
                comb_weights = comb_weights / torch.sum(comb_weights)
            else:
                comb_names = ['conv6']
                ksize = [1]

            sum_det_score_map = None
            det_score_maps = []
            for idx, tmp_name in enumerate(comb_names):
                tmp_feat_map = locals()[tmp_name]  # e2cnn to tensor
                alpha, beta = self.peakiness_score(tmp_feat_map, ksize=3,
                                                   dilation=ksize[idx], name=tmp_name, det_config=self.det_config)

                score_vol = alpha * beta
                det_score_map = torch.max(score_vol, dim=1, keepdims=True)[0]  # channel
                det_score_map = F.interpolate(det_score_map, (ori_h, ori_w), mode='bilinear',
                                              align_corners=False)  # resize
                det_score_map = comb_weights[idx] * det_score_map

                if idx == 0:
                    sum_det_score_map = det_score_map
                else:
                    sum_det_score_map = sum_det_score_map + det_score_map  # multi-level score map add
                det_score_maps.append(det_score_map.permute(0, 2, 3, 1))

            det_score_map = sum_det_score_map

            # tf style
            det_score_map = det_score_map.permute(0, 2, 3, 1)
            det_kpt_inds, det_kpt_scores, drop_list = self.extract_kpts(
                det_score_maps, k=kpt_n,
                score_thld=self.det_config['score_thld'], edge_thld=self.det_config['edge_thld'],
                nms_size=self.det_config['nms_size'], eof_size=self.det_config['eof_mask'])

            if self.det_config['kpt_refinement']:
                offsets = self.kpt_refinement(det_score_map).squeeze(-2)  # x.tensor.permute(0,2,3,1)
                offsets = gather_nd(offsets, det_kpt_inds, batch_dims=1)
                det_kpt_inds = det_kpt_inds + offsets
                det_kpt_inds[..., 0] = torch.clip(det_kpt_inds[..., 0], 0, det_score_map.shape[1] - 1)
                det_kpt_inds[..., 1] = torch.clip(det_kpt_inds[..., 1], 0, det_score_map.shape[2] - 1)
            else:
                det_kpt_inds = det_kpt_inds.float()

            det_kpt_coords = torch.stack([det_kpt_inds[:, :, 1], det_kpt_inds[:, :, 0]], dim=-1)  #
            det_kpt_ncoords = torch.stack([(det_kpt_coords[:, :, 0] - ori_w / 2) / (ori_w / 2),
                                           (det_kpt_coords[:, :, 1] - ori_h / 2) / (ori_h / 2)], dim=-1)

            # torch style
            if self.is_training:
                """training mode"""
                det_score_map = det_score_map.permute(0, 3, 1, 2)
                self.endpoints = [dense_feat_map, det_score_map, seg_feat, drop_list]

            else:
                """inference mode"""
                descs = F.normalize(interpolate(det_kpt_inds / down_ratio, dense_feat_map),
                                    p=self.det_config['norm_type'], dim=1)
                if self.det_config['norm_type'] == 1:
                    descs = torch.sqrt(descs)
                if self.aspp_config['use_aspp']:
                    self.endpoints = [det_kpt_coords, descs, det_kpt_scores, seg_feat]  # det_kpt_scores,
                else:
                    self.endpoints = [det_kpt_coords, descs, det_kpt_scores]  # return [kpts descs scores]
                return self.endpoints

            kpt_feat = self.interpolate(det_kpt_inds / down_ratio, dense_feat_map)
        else:
            # use original kpt position
            kpt_coord = input['kpt_coord']
            kpt_inds = torch.stack([kpt_coord[:, :, 1], kpt_coord[:, :, 0]], dim=-1)
            hs = conv6.shape[-2:].to(kpt_inds.device).reshape(1, 1, 2) / 2
            kpt_inds = kpt_inds * hs + hs
            kpt_feat = self.interpolate(kpt_inds, conv6)

        return det_kpt_coords, kpt_feat, self.endpoints


class SemSNet_fuse(SemSNet):
    def __init__(self, config):
        super().__init__(config)
        self.module_config = config['network']['e2cnn']['module']
        if 'rgb' in config['network']['e2cnn']['module']:
            self.triv_in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        self.in_type_temp = enn.FieldType(self.gspace,
                                          (self.feat_dim[0]) * [self.gspace.regular_repr])
        self.conv_bn0 = R2conv_bn(self.kernel_size, self.triv_in_type, self.in_type_temp, stride=1, mode='zero',
                                  padding=1)
        self.conv_out = nn.Conv2d(config['network']['aspp']['num_class'] + self.out_dim[2], self.out_dim[2],
                                  kernel_size=1, padding='same',
                                  stride=1, bias=False)

    def forward(self, input):
        if isinstance(input, dict):
            x = input['data']
        else:
            x = input
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        ori_h = x.shape[-2]
        ori_w = x.shape[-1]

        x = enn.GeometricTensor(x, self.triv_in_type)
        # net forward
        conv1, conv3, conv1_bn, conv3_bn, conv5, conv6 = self.network(x)
        conv1, conv3, conv6 = conv1.tensor, conv3.tensor, conv6.tensor
        if (self.aspp_config['use_aspp']):
            x = self.conv_bn_aspp6_0(conv5)
            x = self.conv_bn_aspp6_1(x)
            x = self.conv_bn_aspp6_2(x)
            x = x + conv5
            seg_feat = self.aspp([conv1_bn, conv3_bn, x])
            seg_feat = seg_feat.tensor
            x = seg_feat  # for cat
            x = torch.cat((conv6, x), dim=1)
            conv6 = self.conv_out(x)

        down_ratio = ori_w / conv6.shape[-1]
        if self.det_config['weight'] > 0:
            dense_feat_map = conv6  # to tensor
            kpt_n = self.det_config['kpt_n']

            if self.det_config['multi_level']:
                comb_names = ['conv1', 'conv3', 'conv6']
                ksize = [3, 2, 1]
                comb_weights = torch.tensor([1., 2., 3.])
                comb_weights = comb_weights / torch.sum(comb_weights)
            else:
                comb_names = ['conv6']
                ksize = [1]

            sum_det_score_map = None
            det_score_maps = []
            for idx, tmp_name in enumerate(comb_names):
                tmp_feat_map = locals()[tmp_name]  # e2cnn to tensor
                alpha, beta = self.peakiness_score(tmp_feat_map, ksize=3,
                                                   dilation=ksize[idx], name=tmp_name, det_config=self.det_config)

                score_vol = alpha * beta
                det_score_map = torch.max(score_vol, dim=1, keepdims=True)[0]  # channel
                det_score_map = F.interpolate(det_score_map, (ori_h, ori_w), mode='bilinear',
                                              align_corners=False)  # resize
                det_score_map = comb_weights[idx] * det_score_map

                if idx == 0:
                    sum_det_score_map = det_score_map
                else:
                    sum_det_score_map = sum_det_score_map + det_score_map  # multi-level score map add
                det_score_maps.append(det_score_map.permute(0, 2, 3, 1))

            det_score_map = sum_det_score_map

            # tf style
            det_score_map = det_score_map.permute(0, 2, 3, 1)
            det_kpt_inds, det_kpt_scores, drop_list = self.extract_kpts(
                det_score_maps, k=kpt_n,
                score_thld=self.det_config['score_thld'], edge_thld=self.det_config['edge_thld'],
                nms_size=self.det_config['nms_size'], eof_size=self.det_config['eof_mask'])

            if self.det_config['kpt_refinement']:
                offsets = self.kpt_refinement(det_score_map).squeeze(-2)  # x.tensor.permute(0,2,3,1)
                offsets = gather_nd(offsets, det_kpt_inds, batch_dims=1)
                det_kpt_inds = det_kpt_inds + offsets
                det_kpt_inds[..., 0] = torch.clip(det_kpt_inds[..., 0], 0, det_score_map.shape[1] - 1)
                det_kpt_inds[..., 1] = torch.clip(det_kpt_inds[..., 1], 0, det_score_map.shape[2] - 1)
            else:
                det_kpt_inds = det_kpt_inds.float()

            det_kpt_coords = torch.stack([det_kpt_inds[:, :, 1], det_kpt_inds[:, :, 0]], dim=-1)  #
            det_kpt_ncoords = torch.stack([(det_kpt_coords[:, :, 0] - ori_w / 2) / (ori_w / 2),
                                           (det_kpt_coords[:, :, 1] - ori_h / 2) / (ori_h / 2)], dim=-1)

            # torch style
            if self.is_training:
                """training mode"""
                det_score_map = det_score_map.permute(0, 3, 1, 2)
                self.endpoints = [dense_feat_map, det_score_map, seg_feat, drop_list]
            else:
                """inference mode"""
                descs = F.normalize(interpolate(det_kpt_inds / down_ratio, dense_feat_map),
                                    p=self.det_config['norm_type'], dim=1)
                if self.det_config['norm_type'] == 1:
                    descs = torch.sqrt(descs)
                if self.aspp_config['use_aspp']:
                    self.endpoints = [det_kpt_coords, descs, det_kpt_scores, seg_feat]  # det_kpt_scores,
                else:
                    self.endpoints = [det_kpt_coords, descs, det_kpt_scores]  # return [kpts descs scores]
                return self.endpoints
            kpt_feat = self.interpolate(det_kpt_inds / down_ratio, dense_feat_map)
        else:
            # use original kpt position
            kpt_coord = input['kpt_coord']
            kpt_inds = torch.stack([kpt_coord[:, :, 1], kpt_coord[:, :, 0]], dim=-1)
            hs = conv6.shape[-2:].to(kpt_inds.device).reshape(1, 1, 2) / 2
            kpt_inds = kpt_inds * hs + hs
            kpt_feat = self.interpolate(kpt_inds, conv6)

        return det_kpt_coords, kpt_feat, self.endpoints
