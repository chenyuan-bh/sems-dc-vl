#!/usr/bin/env python
"""
Adapted from SuperPoint:
https://github.com/rpautrat/SuperPoint
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus(x, a=2):
    e = torch.log(torch.exp(a * x) + 1)
    return e


def squareplus(x, b=4):
    x2b = torch.pow(x, 2) + b
    sp = 1 / 2 * (x + torch.sqrt(x2b))
    return sp


def apply_coord_pert(trans, pert_homo, batch_size, num_corr):
    tmp_ones = torch.ones((batch_size, num_corr, 1))
    homo_coord = torch.concat((trans, tmp_ones), dim=-1)
    pert_coord = torch.matmul(homo_coord, pert_homo.transpose())
    homo_scale = torch.unsqueeze(pert_coord[:, :, 2], dim=-1)
    pert_coord = pert_coord[:, :, 0:2]
    pert_coord = pert_coord / homo_scale
    return pert_coord


def apply_patch_pert(kpt_param, pert_affine, batch_size, num_corr, adjust_ratio=1):
    """
    Args:
        kpt_param: 6-d keypoint parameterization
        pert_mat: 3x3 perturbation matrix.
    Returns:
        pert_theta: perturbed affine transformations.
        trans: translation vectors, i.e., keypoint coordinates.
        pert_mat: perturbation matrix.
    """
    kpt_affine = kpt_param.reshape(batch_size, num_corr, 2, 3)
    rot = kpt_affine[:, :, :, 0:2]
    trans = kpt_affine[:, :, :, 2]
    # adjust the translation as input images are padded.
    trans_with_pad = torch.unsqueeze(trans * adjust_ratio, dim=-1)
    kpt_affine_with_pad = torch.cat((rot, trans_with_pad), dim=-1)
    pert_affine = torch.as_tensor(pert_affine, device=kpt_affine_with_pad.device)
    pert_kpt_affine = torch.matmul(kpt_affine_with_pad, pert_affine)
    return pert_kpt_affine, trans


def torch_gather_nd(params: torch.Tensor, indices: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    """
    torch_gather_nd implements tf.gather_nd in PyTorch.

    This supports multiple batch dimensions as well as multiple channel dimensions.
    """
    index_shape = indices.shape[:-1]
    num_dim = indices.size(-1)
    tail_sizes = params.shape[batch_dim + num_dim:]

    # flatten extra dimensions
    for s in tail_sizes:
        row_indices = torch.arange(s, device=params.device)
        indices = indices.unsqueeze(-2)
        indices = indices.repeat(*[1 for _ in range(indices.dim() - 2)], s, 1)
        row_indices = row_indices.expand(*indices.shape[:-2], -1).unsqueeze(-1)
        indices = torch.cat((indices, row_indices), dim=-1)
        num_dim += 1

    # flatten indices and params to batch specific ones instead of channel specific
    for i in range(num_dim):
        size = math.prod(params.shape[batch_dim + i + 1:batch_dim + num_dim])
        indices[..., i] *= size

    indices = indices.sum(dim=-1)
    indices = indices.long()
    params = params.flatten(batch_dim, -1)
    indices = indices.flatten(batch_dim, -1)

    out = torch.gather(params, dim=batch_dim, index=indices)
    return out.reshape(*index_shape, *tail_sizes)


def gather_nd(params, indices, batch_dims=0):
    """ The same as tf.gather_nd.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:]

    """
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    else:
        if not isinstance(indices, np.ndarray):
            raise ValueError(f'indices must be `torch.Tensor` or `numpy.array`. Got {type(indices)}')
    if batch_dims == 0:
        orig_shape = list(indices.shape)
        num_samples = int(np.prod(orig_shape[:-1]))
        m = orig_shape[-1]
        n = len(params.shape)

        if m <= n:
            out_shape = orig_shape[:-1] + list(params.shape[m:])
        else:
            raise ValueError(
                f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
            )
        # indices[indices==-9223372036854775808]=0
        indices = indices.reshape((num_samples, m)).transpose().tolist()
        output = params[indices]  # (num_samples, ...)
        return output.reshape(out_shape).contiguous()
    else:
        batch_shape = params.shape[:batch_dims]
        orig_indices_shape = list(indices.shape)
        orig_params_shape = list(params.shape)
        assert (
                batch_shape == indices.shape[:batch_dims]
        ), f'if batch_dims is not 0, then both "params" and "indices" have batch_dims leading batch dimensions that exactly match.'
        mbs = np.prod(batch_shape)
        if batch_dims != 1:
            params = params.reshape(mbs, *(params.shape[batch_dims:]))
            indices = indices.reshape(mbs, *(indices.shape[batch_dims:]))
        output = []
        for i in range(mbs):
            output.append(gather_nd(params[i], indices[i], batch_dims=0))
        output = torch.stack(output, dim=0)
        output_shape = orig_indices_shape[:-1] + list(orig_params_shape[orig_indices_shape[-1] + batch_dims:])
        return output.reshape(*output_shape).contiguous()


def get_neighbour(feat_map, pos, win_size):
    """the x, y coordinates in the window when a filter sliding on the feature map

    :param feature_map_size:
    :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
    """
    b, c, feat_h, feat_w = feat_map.shape
    win_r = int(win_size // 2)
    pos[..., 0] = torch.clip(pos[..., 0], win_r, feat_h - win_r - 1)  # avoid padding edge   inplace
    pos[..., 1] = torch.clip(pos[..., 1], win_r, feat_w - win_r - 1)
    neighbor_map = nn.Unfold(win_size, 1, padding=win_r, stride=1)(feat_map)
    # out_h,out_w=#feat_h-win_r*2,feat_w-win_r*2
    neighbor_map = neighbor_map.view(b, c * win_size * win_size, feat_h, feat_w)
    local_maps = gather_nd(neighbor_map.permute(0, 2, 3, 1), pos, batch_dims=1)
    local_maps = local_maps.view(b, -1, c, win_size * win_size)
    return local_maps


def get_neighbour_rd(feat_map, pos,
                     win_size):  # pos at left-top corner,extract right-bottom neighborhood. win_size isn't radius
    """the x, y coordinates in the window when a filter sliding on the feature map

    :param feature_map_size:
    :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
    """
    b, c, feat_h, feat_w = feat_map.shape
    pos[..., 0] = torch.clip(pos[..., 0], 0, feat_h - win_size - 1)  # avoid padding edge   inplace
    pos[..., 1] = torch.clip(pos[..., 1], 0, feat_w - win_size - 1)
    neighbor_map = nn.Unfold(win_size, 1, padding=0, stride=1)(feat_map)
    neighbor_map = neighbor_map.view(b, c * win_size * win_size, feat_h - win_size + 1, feat_w - win_size + 1)
    local_maps = gather_nd(neighbor_map.permute(0, 2, 3, 1), pos, batch_dims=1)
    local_maps = local_maps.view(b, -1, c, win_size * win_size)
    return local_maps


def kpt_refinement(score_map, pos):
    # torch style->tf
    n_channel, h, w = score_map.shape[-3], score_map.shape[-2], score_map.shape[-1]
    # inputs = inputs.permute(0, 3, 1, 2)
    di_filter = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], device=score_map.device).reshape(1, 1, 3, 3)
    dj_filter = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], device=score_map.device).reshape(1, 1, 3, 3)
    dii_filter = torch.tensor([[0, 1., 0], [0, -2., 0], [0, 1., 0]], device=score_map.device).reshape(1, 1, 3, 3)
    dij_filter = 0.25 * torch.tensor([[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]], device=score_map.device).reshape(1, 1, 3,
                                                                                                                3)
    djj_filter = torch.tensor([[0, 0, 0], [1., -2., 1.], [0, 0, 0]], device=score_map.device).reshape(1, 1, 3, 3)

    dii_filter = torch.tile(dii_filter, (n_channel, 1, 1, 1))  # out_channel in_channel/groups H W
    dii = F.conv2d(score_map, dii_filter, None, stride=1, padding='same', groups=n_channel)

    dij_filter = torch.tile(dij_filter, (n_channel, 1, 1, 1))  # out_channel in_channel/groups H W
    dij = F.conv2d(score_map, dij_filter, None, stride=1, padding='same', groups=n_channel)

    djj_filter = torch.tile(djj_filter, (n_channel, 1, 1, 1))  # out_channel in_channel/groups H W
    djj = F.conv2d(score_map, djj_filter, None, stride=1, padding='same', groups=n_channel)

    det = dii * djj - dij * dij
    epsilon = lambda x: (x < 0) * (-1e-6) + ~(x < 0) * 1e-6
    det = det + epsilon(det)
    dii = dii + epsilon(dii)

    inv_hess_00 = torch.nan_to_num(torch.div(djj, det))
    inv_hess_01 = torch.nan_to_num(torch.div(-dij, det))
    inv_hess_11 = torch.nan_to_num(torch.div(dii, det))

    di_filter = torch.tile(di_filter, (n_channel, 1, 1, 1))
    di = F.conv2d(score_map, di_filter, None, stride=1, padding='same', groups=n_channel)
    dj_filter = torch.tile(dj_filter, (n_channel, 1, 1, 1))
    dj = F.conv2d(score_map, dj_filter, None, stride=1, padding='same', groups=n_channel)

    step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
    step_j = -(inv_hess_01 * di + inv_hess_11 * dj)

    offsets = torch.stack([step_i, step_j], dim=-1)  #
    offsets = torch.clip(offsets, -0.5, 0.5).permute(0, 2, 3, 1, 4).squeeze(-2)
    offset = gather_nd(offsets, pos, batch_dims=1)
    pos_refine = pos + offset

    return pos_refine
