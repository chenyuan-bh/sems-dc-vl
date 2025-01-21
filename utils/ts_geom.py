"""
Heavily adapted from D2-Net:
https://github.com/mihaidusmanu/d2-net
"""
import numpy as np
import torch

from utils.utils import gather_nd


# np.random.seed(0)

def get_pose_inv(rel_pose):
    b = rel_pose.shape[0]
    rel_pose_inv = torch.zeros(rel_pose.shape, device=rel_pose.device)
    for i in range(b):
        rel_pose_inv[i, :, :3] = rel_pose[i, :, :3].transpose(0, 1)
        rel_pose_inv[i, :, 3] = -torch.matmul(rel_pose_inv[i, :, :3], rel_pose[i, :, 3])
    return rel_pose_inv


def validate_and_interpolate(pos, inputs, valid_corner=True, validate_val=None, return_corners=False):
    device = pos.device

    ids = torch.arange(0, pos.size(0), device=device)

    h, w = inputs.shape

    i = pos[:, 0].float()
    j = pos[:, 1].float()

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    if valid_corner:
        valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)
    # else:
    #     i_top_left=torch.clip(i_top_left,0,h-1)
    #     j_top_left = torch.clip(j_top_left, 0, w - 1)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    if valid_corner:
        valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)
    # else:
    #     i_top_right=torch.clip(i_top_right,0,h-1)
    #     j_top_right = torch.clip(j_top_right, 0, w - 1)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    if valid_corner:
        valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)
    # else:
    #     i_bottom_left=torch.clip(i_bottom_left,0,h-1)
    #     j_bottom_left = torch.clip(j_bottom_left, 0, w - 1)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    if valid_corner:
        valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)
    # else:
    #     i_bottom_right=torch.clip(i_bottom_right,0,h-1)
    #     j_bottom_right = torch.clip(j_bottom_right, 0, w - 1)

    if valid_corner:
        valid_corners = torch.min(
            torch.min(valid_top_left, valid_top_right),
            torch.min(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corners]
        j_top_left = j_top_left[valid_corners]

        i_top_right = i_top_right[valid_corners]
        j_top_right = j_top_right[valid_corners]

        i_bottom_left = i_bottom_left[valid_corners]
        j_bottom_left = j_bottom_left[valid_corners]

        i_bottom_right = i_bottom_right[valid_corners]
        j_bottom_right = j_bottom_right[valid_corners]

        ids = ids[valid_corners]

        # Interpolation
        # i = i[ids]
        # j = j[ids]

    if validate_val is not None:
        # Valid depth
        valid_depth = torch.logical_and(
            torch.logical_and(inputs[..., i_top_left, j_top_left] > 0, inputs[..., i_top_right, j_top_right] > 0),
            torch.logical_and(inputs[..., i_bottom_left, j_bottom_left] > 0,
                              inputs[..., i_bottom_right, j_bottom_right] > 0)
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]

        ids = ids[valid_depth]

    i = i[ids]
    j = j[ids]

    # Interpolation
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    descriptors = (
            w_top_left * inputs[..., i_top_left, j_top_left] +
            w_top_right * inputs[..., i_top_right, j_top_right] +
            w_bottom_left * inputs[..., i_bottom_left, j_bottom_left] +
            w_bottom_right * inputs[..., i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(-1, 1), j.view(-1, 1)], dim=1)

    if not return_corners:
        return [descriptors, pos, ids]
    else:
        corners = torch.stack([
            torch.stack([i_top_left, j_top_left], dim=0),
            torch.stack([i_top_right, j_top_right], dim=0),
            torch.stack([i_bottom_left, j_bottom_left], dim=0),
            torch.stack([i_bottom_right, j_bottom_right], dim=0)
        ], dim=0)
        return [descriptors, pos, ids, corners]


def get_warp(pos0, rel_pose, depth0, K0, depth1, K1, bs, valid_depth=True):
    def swap_axis(data):
        return torch.stack([data[:, 1], data[:, 0]], dim=-1)

    device = pos0.device
    all_pos0 = []
    all_pos1 = []
    all_ids = []
    # pos0=pos0+torch.randn(pos0.shape,device=device)*4#debug
    for i in range(bs):
        z0, new_pos0, ids = validate_and_interpolate(pos0[i], depth0[i], validate_val=0)  # depth>0

        uv0_homo = torch.cat((swap_axis(new_pos0), torch.ones(new_pos0.shape[0], 1, device=device)), dim=-1)
        xy0_homo = torch.matmul(torch.linalg.inv(K0[i]), uv0_homo.transpose(0, 1))
        xyz0_homo = torch.cat((z0.unsqueeze(0) * xy0_homo,
                               torch.ones((1, new_pos0.shape[0]), device=device)), dim=0)  # 归一化相机坐标系点

        xyz1 = torch.matmul(rel_pose[i], xyz0_homo)  # [R t]
        xy1_homo = xyz1 / (xyz1[-1, :].unsqueeze(0) + 1e-6)  # 另一个相机 归一化相机坐标系点
        uv1 = torch.matmul(K1[i], xy1_homo).transpose(0, 1)[:, 0:2]  # 另一图像点

        new_pos1 = swap_axis(uv1)
        annotated_depth, new_pos1, new_ids = validate_and_interpolate(
            new_pos1, depth1[i], validate_val=0)  # return depth>0 ,depth map interpolate

        ids = torch.index_select(ids, 0, new_ids)
        new_pos0 = torch.index_select(new_pos0, 0, new_ids)
        estimated_depth = torch.index_select(xyz1.transpose(0, 1), 0, new_ids)[:, -1]  # z1
        if valid_depth:
            inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05  # 选择 计算的深度一致的

            all_ids.append(ids[inlier_mask])
            all_pos0.append(new_pos0[inlier_mask])
            all_pos1.append(new_pos1[inlier_mask])
        else:
            all_ids.append(ids)
            all_pos0.append(new_pos0)
            all_pos1.append(new_pos1)
    return all_pos0, all_pos1, all_ids


def get_pos_warp(pos0, pos1, theta, H, W):
    def swap_axis(data):
        return torch.stack([data[:, 1], data[:, 0]], dim=-1)

    device = theta.device
    all_pos0 = []
    all_pos1 = []
    for i in range(theta.shape[0]):
        # pos0
        pos0[i][:, 0] = (pos0[i][:, 0] - H / 2) / H * 2
        pos0[i][:, 1] = (pos0[i][:, 1] - W / 2) / W * 2
        uv0_homo = torch.cat((swap_axis(pos0[i]), torch.ones(pos0[i].shape[0], 1, device=device)), dim=-1)
        # uv0_homo = torch.cat((pos0[i], torch.ones(pos0[i].shape[0], 1, device=device)), dim=-1)
        xy0_homo = torch.matmul(theta[i, 0], uv0_homo.transpose(0, 1)).transpose(0, 1)
        new_pos0 = swap_axis(xy0_homo) / xy0_homo[:, -1].unsqueeze(-1)
        # new_pos0=xy0_homo/xy0_homo[:,-1].unsqueeze(-1)
        new_pos0[:, 0] = (new_pos0[:, 0] + 1) * H / 2
        new_pos0[:, 1] = (new_pos0[:, 1] + 1) * W / 2
        valid0 = torch.logical_and(torch.logical_and(new_pos0[:, 0] > 0, new_pos0[:, 0] < H),
                                   torch.logical_and(new_pos0[:, 1] > 0, new_pos0[:, 1] < W))
        # pos1
        pos1[i][:, 0] = (pos1[i][:, 0] - H / 2) / H * 2
        pos1[i][:, 1] = (pos1[i][:, 1] - W / 2) / W * 2
        uv1_homo = torch.cat((swap_axis(pos1[i]), torch.ones(pos1[i].shape[0], 1, device=device)), dim=-1)
        # uv1_homo = torch.cat((pos1[i], torch.ones(pos1[i].shape[0], 1, device=device)), dim=-1)
        xy1_homo = torch.matmul(theta[i, 1], uv1_homo.transpose(0, 1)).transpose(0, 1)
        new_pos1 = swap_axis(xy1_homo) / xy1_homo[:, -1].unsqueeze(-1)
        # new_pos1=xy1_homo/xy1_homo[:,-1].unsqueeze(-1)
        new_pos1[:, 0] = (new_pos1[:, 0] + 1) * H / 2
        new_pos1[:, 1] = (new_pos1[:, 1] + 1) * W / 2
        valid1 = torch.logical_and(torch.logical_and(new_pos1[:, 0] > 0, new_pos1[:, 0] < H),
                                   torch.logical_and(new_pos1[:, 1] > 0, new_pos1[:, 1] < W))
        valid = torch.logical_and(valid0, valid1)
        new_pos0 = new_pos0[valid][:, :2]
        new_pos1 = new_pos1[valid][:, :2]
        all_pos0.append(new_pos0)
        all_pos1.append(new_pos1)
    # all_pos0=torch.stack(all_pos0,dim=0)
    # all_pos1 = torch.stack(all_pos1, dim=0)
    return all_pos0, all_pos1


def rnd_sample(inputs, n_sample, seed=None):
    # import random
    cur_size = inputs[0].shape[0]
    if seed != None: np.random.seed(seed)
    rnd_idx = np.array(range(cur_size))
    rnd_idx = np.random.permutation(rnd_idx)
    rnd_idx = rnd_idx[0:n_sample]
    outputs = [torch.index_select(i, 0, torch.tensor(rnd_idx, device=i.device).int()) for i in inputs]
    return outputs


def rnd_pts(size, n_pts, step=1, device='cuda', seed=None):
    b, h, w = size
    h = int(h / step)
    w = int(w / step)
    # x=np.random.randint(0,h,size=(b,n_pts))
    # y=np.random.randint(0,w,size=(b,n_pts))
    if seed != None: np.random.seed(seed)
    pts = []
    for i in range(b):
        id = np.random.choice(np.arange(h * w), n_pts)
        x = id // h
        y = id % h
        pts.append(np.stack((x, y), axis=-1))
    pts = np.stack(pts, axis=0) * step
    return torch.from_numpy(pts).to(device)


def get_dist_mat(feat1, feat2, dist_type):
    # feat shape (num,channel) ->(num,num)
    eps = 1e-6
    cos_dist_mat = torch.matmul(feat1, feat2.transpose(-2, -1))
    if dist_type == 'cosine_dist':
        dist_mat = torch.clip(cos_dist_mat, -1, 1)
    elif dist_type == 'euclidean_dist':
        dist_mat = torch.sqrt(torch.maximum(2 - 2 * cos_dist_mat, torch.tensor(eps)))
    elif dist_type == 'euclidean_dist_no_norm':
        norm1 = torch.sum(feat1 * feat1, dim=-1, keepdims=True)
        norm2 = torch.sum(feat2 * feat2, dim=-1, keepdims=True)
        dist_mat = torch.sqrt(torch.maximum(torch.tensor(0.),
                                            norm1 - 2 * cos_dist_mat + norm2.transpose(0, 1)) + eps)
    else:
        raise NotImplementedError()
    return dist_mat


def interpolate(pos, inputs, batched=True, nd=True):
    # torch style
    if not batched:
        pos = pos.unsqueeze(0)
        inputs = inputs.unsqueeze(0)

    h, w = inputs.shape[-2:]

    i = pos[:, :, 0].float()
    j = pos[:, :, 1].float()

    i_top_left = torch.clip(torch.floor(i).long(), 0, h - 1)
    j_top_left = torch.clip(torch.floor(j).long(), 0, w - 1)

    i_top_right = torch.clip(torch.floor(i).long(), 0, h - 1)
    j_top_right = torch.clip(torch.ceil(j).long(), 0, w - 1)

    i_bottom_left = torch.clip(torch.ceil(i).long(), 0, h - 1)
    j_bottom_left = torch.clip(torch.floor(j).long(), 0, w - 1)

    i_bottom_right = torch.clip(torch.ceil(i).long(), 0, h - 1)
    j_bottom_right = torch.clip(torch.ceil(j).long(), 0, w - 1)

    dist_i_top_left = i - i_top_left
    dist_j_top_left = j - j_top_left
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    if nd:
        w_top_left = w_top_left[..., None]
        w_top_right = w_top_right[..., None]
        w_bottom_left = w_bottom_left[..., None]
        w_bottom_right = w_bottom_right[..., None]

    # inputs=inputs.permute(0,2,3,1)
    inputs = inputs.transpose(-1, -3).transpose(-2, -3)  # tf
    interpolated_val = (
            w_top_left * gather_nd(inputs, torch.stack([i_top_left, j_top_left], dim=-1), batch_dims=1) +
            w_top_right * gather_nd(inputs, torch.stack([i_top_right, j_top_right], dim=-1), batch_dims=1) +
            w_bottom_left * gather_nd(inputs, torch.stack([i_bottom_left, j_bottom_left], dim=-1), batch_dims=1) +
            w_bottom_right * gather_nd(inputs, torch.stack([i_bottom_right, j_bottom_right], dim=-1), batch_dims=1)
    )
    interpolated_val = interpolated_val.permute(0, 2, 1)

    if not batched:
        interpolated_val = interpolated_val.squeeze(0)
    return interpolated_val


def interpolate_n(pos, inputs, batched=True):
    # torch style
    if not batched:
        pos = pos.unsqueeze(0)
        inputs = inputs.unsqueeze(0)

    n, c, h, w = inputs.shape[-4:]

    i = pos[:, :, 0]
    j = pos[:, :, 1]

    i_top_left = torch.clip(torch.floor(i).long(), 0, h - 1)
    j_top_left = torch.clip(torch.floor(j).long(), 0, w - 1)

    i_top_right = torch.clip(torch.floor(i).long(), 0, h - 1)
    j_top_right = torch.clip(torch.ceil(j).long(), 0, w - 1)

    i_bottom_left = torch.clip(torch.ceil(i).long(), 0, h - 1)
    j_bottom_left = torch.clip(torch.floor(j).long(), 0, w - 1)

    i_bottom_right = torch.clip(torch.ceil(i).long(), 0, h - 1)
    j_bottom_right = torch.clip(torch.ceil(j).long(), 0, w - 1)

    dist_i_top_left = i - i_top_left
    dist_j_top_left = j - j_top_left
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    # inputs=inputs.permute(0,2,3,1)
    inputs = inputs.transpose(-1, -3).transpose(-2, -3)  # tf
    inputs = inputs.view(n, h * w, c)
    # w_top_left=w_top_left[...,-2,:]*h+w_top_left[...,-1]
    # w_top_right = w_top_right[..., -2, :] * h + w_top_right[..., -1]
    # w_bottom_left = w_top_right[..., -2, :] * h + w_bottom_left[..., -1]
    # w_bottom_right = w_top_right[..., -2, :] * h + w_bottom_right[..., -1]
    # interpolated_val = (
    #     w_top_left * gather_nd(inputs, torch.stack([i_top_left, j_top_left], dim=-1), batch_dims=1) +
    #     w_top_right * gather_nd(inputs, torch.stack([i_top_right, j_top_right], dim=-1), batch_dims=1) +
    #     w_bottom_left * gather_nd(inputs, torch.stack([i_bottom_left, j_bottom_left], dim=-1), batch_dims=1) +
    #     w_bottom_right *gather_nd(inputs, torch.stack([i_bottom_right, j_bottom_right], dim=-1), batch_dims=1)
    # )
    interpolated_val = (
            w_top_left.transpose(0, 1) * torch.gather(inputs, 1,
                                                      (i_top_left * h + j_top_left).repeat(1, 32, 1).permute(2, 0,
                                                                                                             1)).squeeze() +
            w_top_right.transpose(0, 1) * torch.gather(inputs, 1,
                                                       (i_top_right * h + j_top_right).repeat(1, 32, 1).permute(2, 0,
                                                                                                                1)).squeeze() +
            w_bottom_left.transpose(0, 1) * torch.gather(inputs, 1,
                                                         (i_bottom_left * h + j_bottom_left).repeat(1, 32, 1).permute(2,
                                                                                                                      0,
                                                                                                                      1)).squeeze() +
            w_bottom_right.transpose(0, 1) * torch.gather(inputs, 1, (i_bottom_right * h + j_bottom_right).repeat(1, 32,
                                                                                                                  1).permute(
        2, 0, 1)).squeeze()
    )
    # interpolated_val=interpolated_val.permute(0,2,1)

    if not batched:
        interpolated_val = interpolated_val.squeeze(0)
    else:
        interpolated_val = interpolated_val.unsqueeze(0)
    return interpolated_val
