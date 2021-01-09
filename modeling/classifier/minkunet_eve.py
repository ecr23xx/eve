import time
import logging
import numpy as np

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from ..layers.matching import matching
from .minkunet import MinkUNet, ResidualBlock
from utils.collation import sparse_collate


class MinkUNetEve(nn.Module):
    def __init__(self, cfg):
        super(MinkUNetEve, self).__init__()

        self.fem = MinkUNet(cfg)
        self.rrm = MinkUNet(cfg)

        self.num_frames = cfg.INPUT.VIDEO.NUM_FRAMES
        self.voxel_size = cfg.INPUT.VOXEL_SIZE
        self.dim_out = cfg.MODEL.NUM_CLASSES
        self.num_classes = cfg.MODEL.NUM_CLASSES

        self.eve_cfg = cfg.MODEL.EVE.clone()
        if self.eve_cfg.FUSION:
            self.fem.classifier = nn.Identity()
            self.rrm.classifier = nn.Identity()
            self.dim_out = self.eve_cfg.DIM_OUT
            self.fusion = nn.ReLU(inplace=True)
            self.classifier = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.dim_out, self.num_classes,
                    kernel_size=1, stride=1, dimension=3))

    def forward(self, x, y):
        # FIXME: batch size = 1
        outs, coords, labels, logits = [], [], [], []
        match_accs, match_times = [], []
        device = x.F.device

        prev_coord = None
        for time_idx in range(self.num_frames):
            cur_coord = x.C[x.C[:, -2] == time_idx][:, :4]
            cur_coord[:, -1] = 0  # change time idx to batch idx
            cur_feat = x.F[x.C[:, -2] == time_idx]
            cur_label = y[x.C[:, -2] == time_idx]

            if time_idx == 0:
                # FIXME: check time
                cur_sptensor = ME.SparseTensor(cur_feat, cur_coord).to(device)
                cur_out = self.fem(cur_sptensor, mink=True)
                if self.eve_cfg.FUSION:
                    outs.append(cur_out.F)
                    logits.append(self.classifier(cur_out).F)
                else:
                    outs.append(cur_out.F)
                    logits.append(cur_out.F)
            else:
                start_time = time.time()
                # TODO: batch matching
                # TODO: part matching
                # NOTE: the cur_match and prev_match is sorted, where
                # the pair with lower distance is in the front.
                # For most circustances, cur_match contains all point pair.
                cur_match, prev_match, match_acc = matching(
                    cur_coord[:, :3].to(device).float(),
                    coords[-1][:, :3].to(device).float(),
                    eve_cfg=self.eve_cfg, cur_label=cur_label, prev_label=labels[-1])
                match_accs.append(match_acc)
                match_times.append(time.time() - start_time)

                # 2. copy from last frame
                cur_num = cur_feat.size(0)
                residual_num = int(cur_num * (1 - self.eve_cfg.MATCH_RATIO))
                cur_out = torch.zeros(cur_num, self.dim_out).to(device)
                cur_out[cur_match] = outs[time_idx - 1][prev_match]

                # 3. compute residual feature
                if residual_num == 0:
                    if self.eve_cfg.FUSION:
                        outs.append(cur_out.F)
                        logits.append(self.classifier(cur_out).F)
                    else:
                        outs.append(cur_out)
                        logits.append(cur_out)
                else:
                    left_behind_idx = cur_match[-residual_num:]
                    tr = self.rrm(ME.SparseTensor(
                            cur_feat[left_behind_idx],
                            cur_coord[left_behind_idx]).to(device))
                    if self.eve_cfg.FUSION:
                        # When fusion,
                        cur_out[left_behind_idx] = self.fusion(
                            tr + cur_out[left_behind_idx])
                        cur_out = ME.SparseTensor(cur_out, cur_coord).to(device)
                        outs.append(cur_out.F)
                        logits.append(self.classifier(cur_out).F)
                    else:
                        cur_out[left_behind_idx] = tr
                        outs.append(cur_out)
                        logits.append(cur_out)

            coords.append(cur_coord)
            labels.append(cur_label)

        out_logits = torch.cat(logits, 0)

        if self.training:
            match_acc = torch.tensor(match_accs).to(device).mean()
            match_time = torch.tensor(match_times).to(device).mean()
            return out_logits, (match_acc, match_time)
        else:
            return out_logits
