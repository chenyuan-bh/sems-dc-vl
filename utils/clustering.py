"""
Copyright 2025, Yuan Chen, BHU.
MGMDBC
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster._dbscan_inner import dbscan_inner
from sklearn.neighbors import NearestNeighbors


def cluster(dataset, minpts=2, max_sigma=200, max_dist=2000, dbscan=False):
    new_data = dataset
    if dbscan:
        cls = DBSCAN(eps=max_sigma, min_samples=minpts).fit_predict(new_data)  # dbscan as initial
    else:
        cls = dbscan_core(new_data, eps=max_sigma, minpts=minpts)  # neglect non-core

    cls_pts, centers = [], []
    neigh = NearestNeighbors(radius=max_sigma * minpts)
    neigh.fit(new_data)
    for i in range(max(cls + 1)):
        ptsi = new_data[np.where(cls == i)]
        mean = np.mean(ptsi, axis=0)
        sigma = np.sqrt(np.mean(np.sum((ptsi - mean) ** 2, axis=1)))  # np.linalg.norm(np.std(ptsi, axis=0)) 2dnorm
        rng = neigh.radius_neighbors(mean.reshape(-1, 2), sigma * 3)
        ptsi_new = new_data[rng[1][0]]
        cls_pts.append(ptsi_new)
        centers.append(np.mean(ptsi_new, axis=0))
    ###determine inlier clusters
    cls_num = [len(c) for c in cls_pts]
    if len(cls_pts) == 1:
        kpts_inlier = cls_pts[0]
    elif len(cls_pts) == 2:
        dist = np.linalg.norm(centers[0] - centers[1])
        if dist >= max_dist:
            kpts_inlier = cls_pts[np.argmax(cls_num)]
        else:
            kpts_inlier = np.concatenate(cls_pts, axis=0)
    elif len(cls_pts) > 2:
        neigh = NearestNeighbors(radius=max_dist * 4)  # try to include all
        neigh.fit(np.stack(centers, axis=0))
        rng = neigh.radius_neighbors()
        dists = rng[0]
        gt_max_num = np.array([np.sum(dist > max_dist) for dist in dists])
        dist_sum = np.array([np.sum(dist) for dist in dists])
        remain_cls = np.ones(len(cls_pts))
        # remove out of maxdist >2
        remove_out2 = np.where(gt_max_num >= 2)[0]
        remain_cls[remove_out2] = 0
        # remove remain pts out of maxdist ==1
        for d in dists: d[remove_out2 - 1] = 0  # remove removed pts dist
        dists[remove_out2 - 1] = np.array([0])
        gt_max_num = np.array([np.sum(dist > max_dist) for dist in dists])
        gt_sing_ind = np.where(gt_max_num == 1)[0]
        if len(gt_sing_ind) > 0:
            remove_cls_sing = gt_sing_ind[
                np.argmax(dist_sum[gt_sing_ind])]  # remove max dist sum (most likely to be outlier)
            remain_cls[remove_cls_sing] = 0
        remain_cls_pts = [cls_pts[i] for i in range(len(remain_cls)) if remain_cls[i] == 1]
        if np.sum(remain_cls) == 0:
            kpts_inlier = cls_pts[np.argmin(dist_sum)]  # at least one cls
        else:
            kpts_inlier = np.concatenate(remain_cls_pts, axis=0)
    kpts_inlier = np.unique(kpts_inlier, axis=0)
    return kpts_inlier


def dbscan_core(X, eps=1, minpts=2, max_dist=2000):
    neigh = NearestNeighbors(radius=eps)
    neigh.fit(X)
    neighborhoods = neigh.radius_neighbors(X, return_distance=False)
    n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
    core_samples = np.asarray(n_neighbors >= minpts, dtype=np.int)
    core = X[np.where(core_samples)]  # remove non-core
    if len(core) == 0:
        dists = mutal_dist(X)
        min_dist = np.min(dists + 10000 * np.eye(len(dists)), axis=0)
        core_ind = np.where(min_dist == np.min(min_dist))[0]  # at least 2
        core = X[core_ind].reshape(-1, 2)
        core_samples[core_ind] = 1
    neigh.fit(core)
    neighborhoods1 = neigh.radius_neighbors(core, return_distance=False)
    labels = np.full(core.shape[0], -1, dtype=np.intp)
    core_samples_new = np.ones(len(core), dtype=np.uint8)
    dbscan_inner(core_samples_new, neighborhoods1, labels)  # cluster core
    new_labels = core_samples - 1
    new_labels[np.where(core_samples)] += labels
    return new_labels


def mutal_dist(pts):  # dists between all pts -->symmetric matrix
    pts = np.array(pts)
    diff_square = (pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2
    distances_squared = np.sum(diff_square, axis=-1)
    distances = np.sqrt(distances_squared)
    return distances
