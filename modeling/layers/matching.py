import torch
import numpy as np
import custom_ext as _C


def nn_search(query, ref):
    """Nearest neighbor search"""
    idx, dist = _C.nn_search(query, ref)
    N = query.size(0)

    # TODO: post-processing time?
    corres = torch.empty(N, 3)
    corres[:, 0] = torch.arange(N)
    corres[:, 1] = idx
    corres[:, 2] = dist

    return corres


def icp_pt(src, dst, max_iter, threshold):
    prev_dist = 0
    N = src.size(0)

    for i in range(max_iter):
        # 1. Find Nearest Neighbor
        idx, dist = _C.nn_search(src, dst)  # TODO: to device
        dst_temp = dst[idx]

        # 2. Compute H matrix
        src_center = src.mean(dim=0)
        dst_temp_center = dst_temp.mean(dim=0)
        src_norm = src - src_center
        dst_temp_norm = dst_temp - dst_temp_center
        h_matrix = torch.mm(src_norm.T, dst_temp_norm)

        # 3. SVD
        U, S, V = torch.svd(h_matrix)

        # 4. Rotation matrix and translation vector
        R = torch.mm(U, V.T)
        t = dst_temp_center - torch.mm(R, src_center.unsqueeze(1)).squeeze()

        # 5. Transform
        src = torch.mm(src, R) + t.unsqueeze(0)
        mean_dist = dist.mean()
        if torch.abs(mean_dist - prev_dist).item() < threshold:
            break
        prev_dist = mean_dist

    corres = torch.empty(N, 3)
    corres[:, 0] = torch.arange(N)
    corres[:, 1] = idx
    corres[:, 2] = dist

    return corres


# TODO batch matching
def matching(query, ref, eve_cfg, cur_label, prev_label):
    if eve_cfg.MATCH_ALGO == 'NN':
        corres = nn_search(query, ref)
    elif eve_cfg.MATCH_ALGO == 'ICP':
        corres = icp_pt(query, ref, eve_cfg.MATCH_ITER, eve_cfg.MATCH_THRESH)
    else:
        raise NotImplementedError

    N = query.size(0)

    # FIXME: alternatives?
    # skip gt for test (when label is all 0)
    if eve_cfg.MATCH_GT and cur_label.sum() != 0:
        correct_idx = (cur_label[corres[:, 0].long()] == \
            prev_label[corres[:, 1].long()]) \
            & (cur_label != 255)
        corres[correct_idx, 2] = 0  # set gt distance to 0
    _, mink = torch.topk(-corres[:, 2], min(N, corres.size(0)))

    corres = corres[mink, :2].long()
    correct = (cur_label[corres[:, 0]] == prev_label[corres[:, 1]]) \
        & (cur_label != 255)
    match_acc = correct.sum().item() / corres.size(0)

    return corres[:, 0], corres[:, 1], match_acc
