#!/usr/bin/env python3
"""
Copyright 2025, Yuan Chen, BHU.
"""
import argparse
import cv2
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from utils.cv_utils import MatcherWrapper
from models.ASLFeat import *
from models.SemSNet import *
from utils.clustering import cluster


def get_args():
    parser = argparse.ArgumentParser(description='global matching',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='matching_eval.yaml',
                        metavar='FILE', help="Specify the config file")

    return parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_img_list(img_list_path):
    with open(img_list_path, "r") as f:
        data = f.read().split('\n')
    img_list = []
    for m in data:
        if m != '':
            img_list.append(m.split('\t')[0])
    return img_list


def load_imgs(img_dir, img_list, ext='jpg'):
    rgb_list = []
    gray_list = []
    for img_path in img_list:
        img = cv2.imread(os.path.join(img_dir, img_path) + '.' + ext)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        rgb_list.append(img)
        gray_list.append(gray)
    return rgb_list, gray_list


def load_maps(img_path, max_cols, max_rows):
    rgb_list = []
    gray_list = []
    corner_list = []
    img = cv2.imdecode(np.fromfile(os.path.join(img_path), dtype=np.uint8), -1)  # cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    img = img[..., ::-1]
    if (img.shape[0] > max_rows) | (img.shape[1] > max_cols):  # cut
        for i in range(img.shape[0] // max_rows + np.sign(img.shape[0] % max_rows)):
            for j in range(img.shape[1] // max_cols + np.sign(img.shape[1] % max_cols)):
                corner = [min(i * max_rows, img.shape[0] - 386), min(j * max_cols, img.shape[1] - 386)]
                img_cut = img[corner[0]:corner[0] + max_rows, corner[1]:corner[1] + max_cols]
                gray_cut = gray[corner[0]:corner[0] + max_rows, corner[1]:corner[1] + max_cols]
                rgb_list.append(img_cut)
                gray_list.append(gray_cut)
                corner_list.append([i * max_rows, j * max_cols])
    else:
        rgb_list.append(img)  #
        gray_list.append(gray)
        corner_list.append([0, 0])

    return rgb_list, gray_list, corner_list


def extract_local_features(gray_list, net, config):
    descriptors = []
    keypoints = []
    for imi, gray_img in enumerate(gray_list):
        assert len(gray_img.shape) == 3
        H, W, _ = gray_img.shape
        max_dim = max(H, W)
        if max_dim > config['network']['max_dim']:
            downsample_ratio = config['network']['max_dim'] / float(max_dim)
            gray_img = cv2.resize(gray_img, (0, 0), fx=downsample_ratio, fy=downsample_ratio)
            if len(gray_img.shape) < 3:
                gray_img = gray_img[..., np.newaxis]
        if config['network']['det']['multi_scale']:
            scale_f = 1 / (2 ** 0.50)
            min_scale = max(0.3, 128 / max(H, W))
            n_scale = math.floor(max(math.log(min_scale) / math.log(scale_f), 1))
            sigma = 0.8
        else:
            n_scale = 1
        descs, kpts, scores = [], [], []
        scales = []
        for i in range(n_scale):
            if i > 0:
                gray_img = cv2.GaussianBlur(gray_img, None, sigma / scale_f)
                gray_img = cv2.resize(gray_img, dsize=None, fx=scale_f, fy=scale_f)
                if len(gray_img.shape) < 3:
                    gray_img = gray_img[..., np.newaxis]

            data = torch.as_tensor(gray_img.copy(), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            mean = torch.mean(data, (-2, -1), keepdim=True)  # normalization
            std = torch.std(data, (-2, -1), keepdim=True)
            data = transforms.Normalize(mean, std)(data)
            # forward
            with torch.no_grad():
                if config['network']['aspp']['use_aspp']:
                    kpt, desc, score, seg = net(data)
                else:
                    kpt, desc, score = net(data)
                # jit_model = torch.jit.trace(net.eval(), data)
            if config['disp'] and config['network']['aspp']['use_aspp']:
                from utils.seg_label import draw_seg
                segdisp = draw_seg(seg)
                cv2.imwrite(f'../outputs/seg/seg_{imi}_s{i}.jpg', segdisp[..., ::-1])
            descs.append(desc.squeeze().transpose(0, 1))
            kpts.append(kpt.squeeze() * torch.tensor([W / data.shape[-1], H / data.shape[-2]], device=device))
            scores.append(score.squeeze())
            scales.append(torch.ones(score.shape[1]) * i)
        descs = torch.cat(descs, dim=0)
        kpts = torch.cat(kpts, dim=0)
        scores = torch.cat(scores, dim=0)
        scales = torch.cat(scales, dim=0)

        ids = torch.argsort(scores, descending=True)[0:config['network']['det']['kpt_n']]
        descs = descs[ids]
        kpts = kpts[ids]
        descs = descs.detach().cpu().numpy()
        kpts = kpts.detach().cpu().numpy()
        descriptors.append(descs)
        keypoints.append(kpts)
    return descriptors, keypoints


def match(config):  # pylint: disable=unused-argument
    cols = config['params']['cols']  #
    rows = config['params']['rows']
    # load testing images.
    img_list = load_img_list(config['img_list'])
    uav_rgb_list, uav_gray_list = load_imgs(config['img0_paths'], img_list, config['params']['extension'])
    map_rgb = cv2.imread(config['img1_paths'])
    map_rgb = map_rgb[..., ::-1]
    map_rgb_list, map_gray_list, corner_list = load_maps(config['img1_paths'], config['params']['cols'],
                                                         config['params'][
                                                             'rows'])  # config['network']['max_dim'] diff from cut
    # load model
    if config['network']['aspp']['use_aspp']:
        net = SemSNet_fuse(config).to(device)
    else:
        net = BaseNet(train_config)
    net.to(device).train()

    def count_param(model):
        param_count = 0
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
        return param_count

    print('model params num:', count_param(net))

    # load
    loadpath = config['model_path']
    load_model = torch.load(loadpath, map_location=device)
    if 'model' in load_model.keys():
        net.load_state_dict(load_model['model'])
        if 'moving_instance_max' in load_model.keys():
            net.moving_instance_max = load_model['moving_instance_max']  #
        print('Pre-trained model loaded from %s' % loadpath)
    else:
        net.load_state_dict(load_model)
    # extract regional features.
    net = net.eval()

    if 'rgb' in config['network']['e2cnn']['module']:
        descs1, kpts1 = extract_local_features(map_rgb_list, net, config)  # kpt [x-horiz,y-vert]
    else:
        descs1, kpts1 = extract_local_features(map_gray_list, net, config)  # kpt [x-horiz,y-vert]
    for i in range(len(map_gray_list)):
        kpts1[i][:, 0] = kpts1[i][:, 0] + corner_list[i][1]
        kpts1[i][:, 1] = kpts1[i][:, 1] + corner_list[i][0]
    kpts1 = np.concatenate(kpts1, axis=0)
    descs1 = np.concatenate(descs1, axis=0)
    # match
    submap_locs, kpt0_match, kpt1_match = [], [], []
    if 'rgb' in config['network']['e2cnn']['module']:
        uav_list = uav_rgb_list
    else:
        uav_list = uav_gray_list
    with tqdm(total=len(uav_list), desc=f'Start', unit='img', initial=0, ncols=80) as pbar:
        for i, uav_img in enumerate(uav_list):
            desc0, kpt0 = extract_local_features([uav_img], net, config)

            # feature matching and draw matches.
            matcher = MatcherWrapper()  # config['network']['det']['norm_type']
            match, mask = matcher.get_matches(
                desc0[0], descs1, kpt0[0], kpts1,
                ratio=config['match']['ratio_test'], cross_check=config['match']['cross_check'],
                err_thld=config['match']['err_thld'], ransac=config['match']['ransac'], info='SeSnet', display=False)
            if match == None or len(kpts1) == 0:
                # zero-size array to reduction operation minimum which has no identity
                left_top = [0, 0, 2000, 2000]
                left_top.insert(0, img_list[i])
                submap_locs.append(left_top)
                pbar.update(1)
                continue

            ## draw matches
            # disp = matcher.draw_matches(uav_rgb_list[i],kpt0[0], map_rgb, kpts1, match, mask)

            if 'cluster' in config['match'] and 'db' in config['match']['cluster']:
                left_top = get_map_area(kpts1, match, mask=mask, size=[cols, rows], cluster_method=2)
            else:
                left_top = get_map_area(kpts1, match, mask=mask, size=[cols, rows])
            left_top = [max(int(left_top[0]), 0), max(int(left_top[1]), 0)]  #
            left_top = [min(int(left_top[0]), map_rgb.shape[1] - cols),
                        min(int(left_top[1]), map_rgb.shape[0] - rows)]  #
            submap = map_rgb[int(left_top[1]):int(left_top[1]) + rows, int(left_top[0]):int(left_top[0]) + cols]
            print(img_list[i], 'loc in reference image:', left_top)
            pbar.set_postfix(**{f'{img_list[i]} loc': left_top})
            pbar.update(1)
            # plt.imshow(submap)
            # plt.show()

            left_top.extend([min(left_top[0] + cols, map_rgb.shape[1]) - left_top[0],  #
                             min(left_top[1] + rows, map_rgb.shape[0]) - left_top[1]])  # submap size
            left_top.insert(0, img_list[i])
            submap_locs.append(left_top)

            # output
            if not os.path.exists('../outputs'):
                os.mkdir('../outputs')
            output_name = '../outputs/' + 'semsnet_mydisp.jpg'
            # print('image save to', output_name)
            # disp = matcher.draw_matches(uav_rgb_list[i], kpt0[0], map_rgb, kpts1, match, mask)
            disp = matcher.draw_matches_overlay(uav_rgb_list[i], kpt0[0], map_rgb, kpts1, match, mask)
            plt.imsave(output_name, disp)
            plt.imshow(disp)
            plt.show()

    if not os.path.exists(os.path.split(config['output_file'])[0]):
        os.mkdir(os.path.split(config['output_file'])[0])
    np.savetxt(config['output_file'], np.array(submap_locs), fmt='%s')

    return kpt0_match, kpt1_match, submap_locs


def get_map_area(kpts1, match, mask=None, size=[1024, 1024],
                 cluster_method=0):  # cols rows cluster:0->3sigma 1->dbscan 2->dbscan3sigma
    kpts1_inlier = []
    if isinstance(mask, np.ndarray) and np.sum(mask) != 0:
        for i in range(len(match)):
            if mask[i]:
                kpts1_inlier.append(np.expand_dims(kpts1[match[i].trainIdx], axis=0))
    else:
        for i in range(len(match)):
            kpts1_inlier.append(np.expand_dims(kpts1[match[i].trainIdx], axis=0))
    kpts1_inlier = np.concatenate(kpts1_inlier, axis=0)
    kpts1_inlier = np.unique(kpts1_inlier, axis=0)  # remove repeated points for mean
    ##clustering
    if cluster_method == 2:
        kpts_inlier = cluster(kpts1_inlier, minpts=2, max_sigma=max(size) / 3, max_dist=max(size))
    elif cluster_method == 1:
        kpts_inlier = cluster(kpts1_inlier, minpts=2, max_sigma=max(size) / 3, max_dist=max(size), dbscan=True)
    elif cluster_method == 0:
        kpts_inlier = three_sigma(kpts1_inlier, n=1, max_sigma=max(size) / 3)
    left_top = [np.min(kpts_inlier[:, 0]), np.min(kpts_inlier[:, 1])]
    right_bottom = [np.max(kpts_inlier[:, 0]), np.max(kpts_inlier[:, 1])]
    center = [(left_top[0] + right_bottom[0]) / 2, (left_top[1] + right_bottom[1]) / 2]
    left_top_new = [center[0] - size[0] / 2, center[1] - size[1] / 2]
    return left_top_new


def three_sigma(dataset, n=3, max_sigma=200):
    new_data = dataset
    sigma = max_sigma + 1
    while (np.max(sigma) > max_sigma):
        mean = np.mean(new_data, axis=0)
        sigma = np.std(new_data, axis=0)
        remove_idx = np.where(abs(new_data - mean) > n * sigma)
        remove_idx = np.unique(remove_idx[0])
        if (len(remove_idx) == 0) | (len(new_data) == 1) | (np.max(sigma) <= max_sigma) & (len(new_data) < 10):  #
            break
        if len(remove_idx) == len(new_data):  #
            idx = np.where(np.min(abs(new_data - mean)))
            new_data = new_data[idx]
            break
        new_data = np.delete(new_data, remove_idx, axis=0)
        if len(new_data) == 0:
            idx = np.where(np.min(abs(new_data - mean)))
            new_data = new_data[idx]
    return new_data


def save_match(kpt0, kpt1, match, mask, left_top):
    matches = [[m.queryIdx, m.trainIdx] for m in match]
    matches = np.array(matches).transpose()
    in_matches = matches[:, np.where(mask == 1)[0]]
    in_kpt0, in_kpt1 = [], []
    for m in in_matches.transpose():
        in_kpt0.append(kpt0[m[0]])
        in_kpt1.append(kpt1[m[1]] - left_top)
    return in_kpt0, in_kpt1


if __name__ == '__main__':
    args = get_args()  # input params
    # parse input
    with open(os.path.join(args.config), 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    match(config)
