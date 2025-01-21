#!/usr/bin/env python3
import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import yaml

from models.SeS_seg import *
from utils.cv_utils import MatcherWrapper


def get_args():
    parser = argparse.ArgumentParser(description='image matching',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='matching_eval.yaml',
                        metavar='FILE', help="Specify the config file")

    return parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_imgs(img_paths, max_dim):
    rgb_list = []
    gray_list = []
    for img_path in img_paths:
        img = cv2.imdecode(np.fromfile(os.path.join(img_path), dtype=np.uint8), -1)  # cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        if max(img.shape) > max_dim:
            scale = max_dim / max(img.shape)
            img = cv2.resize(img, None, None, scale, scale, cv2.INTER_AREA)
            gray = cv2.resize(gray, None, None, scale, scale, cv2.INTER_AREA)[..., np.newaxis]
        rgb_list.append(img)
        gray_list.append(gray)
    return rgb_list, gray_list


def extract_local_features(gray_list, net, config):
    descriptors = []
    keypoints = []
    for gray_img in gray_list:
        assert len(gray_img.shape) == 3
        H, W, _ = gray_img.shape
        max_dim = max(H, W)
        if max_dim > config['network']['max_dim']:
            downsample_ratio = config['network']['max_dim'] / float(max_dim)
            gray_img = cv2.resize(gray_img, (0, 0), fx=downsample_ratio, fy=downsample_ratio)
            gray_img = gray_img[..., np.newaxis]
        if config['network']['det']['multi_scale']:
            scale_f = 1 / (2 ** 0.50)
            min_scale = max(0.3, 128 / max(H, W))
            n_scale = math.floor(max(math.log(min_scale) / math.log(scale_f), 1))
            sigma = 0.8
        else:
            n_scale = 1
        descs, kpts, scores = [], [], []
        for i in range(n_scale):
            if i > 0:
                gray_img = cv2.GaussianBlur(gray_img, None, sigma / scale_f)
                if 'rgb' in config['network']['e2cnn']['module']:
                    gray_img = cv2.resize(gray_img, dsize=None, fx=scale_f, fy=scale_f)
                else:
                    gray_img = cv2.resize(gray_img, dsize=None, fx=scale_f, fy=scale_f)[..., np.newaxis]

            data = torch.as_tensor(gray_img.copy(), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(
                0)  # rgb .copy()?
            mean = torch.mean(data, (-2, -1), keepdim=True)  # normalization
            std = torch.std(data, (-2, -1), keepdim=True)
            data = transforms.Normalize(mean, std)(data)
            # forward
            with torch.no_grad():
                try:
                    kpt, desc, score = net(data)
                except:
                    kpt, desc, score, _ = net(data)
                # jit_model = torch.jit.trace(net.eval(), data)
            descs.append(desc.squeeze().transpose(0, 1))
            kpts.append(kpt.squeeze() * torch.tensor([W / data.shape[-1], H / data.shape[-2]], device=device))
            scores.append(score.squeeze())
        descs = torch.cat(descs, dim=0)
        kpts = torch.cat(kpts, dim=0)
        scores = torch.cat(scores, dim=0)

        ids = torch.argsort(scores, descending=True)[0:config['network']['det']['kpt_n']]
        descs = descs[ids]
        kpts = kpts[ids]
        scores = scores[ids]

        print('feature_num', kpts.shape[0])
        descs = descs.detach().cpu().numpy()
        kpts = kpts.detach().cpu().numpy()
        descriptors.append(descs)
        keypoints.append(kpts)
    return descriptors, keypoints


def match(config):
    # load testing images.
    rgb_list, gray_list = load_imgs(config['img_paths'], config['network']['max_dim'])
    # load model
    if config['network']['aspp']['use_aspp']:
        net = SeSNet_seg_e2_cat(config).to(device)
    else:
        net = SeSNet(train_config)
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
            net.moving_instance_max = load_model[
                'moving_instance_max']
        print('Pre-trained model loaded from %s' % loadpath)
    else:
        net.load_state_dict(load_model)
    # extract regional features.
    net = net.eval()
    if 'rgb' in config['network']['e2cnn']['module']:
        descs, kpts = extract_local_features(rgb_list, net, config)
    else:
        descs, kpts = extract_local_features(gray_list, net, config)
    # feature matching and draw matches.
    matcher = MatcherWrapper()
    match, mask = matcher.get_matches(
        descs[0], descs[1], kpts[0], kpts[1],
        ratio=config['match']['ratio_test'], cross_check=config['match']['cross_check'],
        err_thld=config['match']['err_thld'], ransac=config['match']['ransac'], info='ASLFeat')
    # draw matches
    disp = matcher.draw_matches(rgb_list[0], kpts[0], rgb_list[1], kpts[1], match, mask)

    # output
    if not os.path.exists('../outputs'):
        os.mkdir('../outputs')
    output_name = '../outputs/' + 'image_matching_mydisp.jpg'
    print('image save to', output_name)
    plt.imsave(output_name, disp)

    # save as iobinary
    keydescpath = config['output_path'] + 'keypt_desc.bin'
    if os.path.exists(keydescpath):
        os.remove(keydescpath)
    in_kpt0, in_kpt1, in_desc0, in_desc1 = [], [], [], []
    in_matches = match[:, np.where(mask == 1)[0]]
    for m in in_matches.transpose():
        in_kpt0.append(kpts[0][m[0]])
        in_kpt1.append(kpts[1][m[1]])
        in_desc0.append(descs[0][m[0]])
        in_desc1.append(descs[1][m[1]])
    with open(keydescpath, 'w') as fout:
        fout.write(str(len(in_kpt0)) + ' ')  # num
        fout.write(str(descs[0].shape[1]) + '\n')  # dim
        for i in range(len(in_kpt0)):
            _ = [fout.write(str(k) + ' ') for k in in_kpt0[i]]
            _ = [fout.write(str(k) + ' ') for k in in_desc0[i]]
            fout.write('\n')
        fout.write(str(len(in_kpt1)) + ' ')  # num
        fout.write(str(descs[1].shape[1]) + '\n')  # dim
        for i in range(len(in_kpt1)):
            _ = [fout.write(str(k) + ' ') for k in in_kpt1[i]]
            _ = [fout.write(str(k) + ' ') for k in in_desc1[i]]
            fout.write('\n')


if __name__ == '__main__':
    args = get_args()  # input params
    # parse input
    with open(os.path.join(args.config), 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    match(config)
