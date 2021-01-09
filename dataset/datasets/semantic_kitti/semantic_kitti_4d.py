import os
import sys
import json
import h5py
import os.path
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from config import cfg
from utils.voxelization import sparse_quantize
from utils.collation import sparse_collate
from .semantic_kitti import SemanticKittiDataset
from .utils import parse_calib, parse_pose


class SemanticKittiDataset4d(SemanticKittiDataset):
    def __init__(self, root, split, voxel_size, num_points, trainval,
                 num_frames, align):
        self.num_frames = num_frames
        self.align = align
        if split == 'train':
            self.sample_rate = 1
        else:
            self.sample_rate = num_frames

        super().__init__(root, split, voxel_size, num_points, align, trainval)

    def _prepare_files(self):
        self.all_files = []
        self.files = []
        self.label_files = []

        for seq in self.seqs:
            unique_files = set()
            seq_files = sorted(os.listdir(os.path.join(
                self.sequences, seq, 'velodyne')))
            seq_files_with_prefix = [os.path.join(
                self.sequences, seq, 'velodyne', x) for x in seq_files]
            self.all_files.extend(seq_files_with_prefix)
            total_num = len(seq_files_with_prefix)

            for i in range(0, total_num - self.num_frames + 1, self.sample_rate):
                clip = seq_files_with_prefix[i:i+self.num_frames]
                clip_labels = [filename.replace('velodyne', 'labels').replace(
                    '.bin', '.label') for filename in clip]
                self.files.append(clip)
                self.label_files.append(clip_labels)
                unique_files.update(clip)

            # check every file is in a clip
            if len(unique_files) < total_num:
                remainder_frames = set(seq_files_with_prefix) - \
                    unique_files
                for f in remainder_frames:
                    f_idx = seq_files_with_prefix.index(f)
                    clip = seq_files_with_prefix[f_idx -
                                                 self.num_frames+1:f_idx+1]
                    clip_labels = [filename.replace('velodyne', 'labels').replace(
                        '.bin', '.label') for filename in clip]
                    self.files.append(clip)
                    self.label_files.append(clip_labels)
                    unique_files.update(clip)
            assert len(unique_files) == total_num

    def __getitem__(self, index):
        pc_clip = []
        feat_clip = []
        labels_clip = []
        metadata_clip = []

        for time_idx, file_path in enumerate(self.files[index]):
            # 1. prepare point clouds
            with open(file_path, 'rb') as b:
                block = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
            if self.align and time_idx != 0:
                block = self._data_align(
                    block, file_path, self.files[index][0])
            # NOTE: use the same augumentation in one clip
            block = self._data_augment(block, new=(time_idx == 0))
            pc = np.round(block[:, :3] / self.voxel_size)
            # NOTE: normalize will decrease icp match accuracy ~50%
            # pc -= pc.min(0, keepdims=1)
            feat = block[:, :3]

            # prepare labels ~ 0s
            if self.split == 'test':
                all_labels = np.zeros((pc.shape[0])).astype(np.int32)
            else:
                with open(self.label_files[index][time_idx], 'rb') as a:
                    all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
            all_labels = all_labels & 0xFFFF
            labels = self.label_map[all_labels]

            # sparse quantize ~ 0.015s
            inds, _, inverse_map = sparse_quantize(
                pc, feat, labels, return_index=True)

            if self.split == 'train':
                if len(inds) > self.num_points:
                    inds = np.random.choice(
                        inds, self.num_points, replace=False)

            time_idxs = np.array([time_idx] * len(inds)).reshape(-1, 1)

            pc_clip.append(np.concatenate((pc[inds], time_idxs), axis=1))
            feat_clip.append(feat[inds])
            labels_clip.append(labels[inds])

            metadata = {}
            metadata['index'] = index
            metadata['time_idx'] = time_idx
            metadata['inverse_map'] = torch.tensor(inverse_map)
            metadata['file_path'] = file_path
            metadata_clip.append(metadata)

        pc_clip = np.concatenate(pc_clip, axis=0)
        feat_clip = np.concatenate(feat_clip, axis=0)
        labels_clip = np.concatenate(labels_clip, axis=0)

        return pc_clip, feat_clip, labels_clip, metadata_clip

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate_fn(tbl):
        locs, feats, labels, metadata = zip(*tbl)
        return sparse_collate(locs, feats, labels) + (metadata,)
