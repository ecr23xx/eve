import os
import sys
import json
import h5py
import yaml
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
from utils.voxelization import sparse_quantize
from utils.collation import sparse_collate
from .utils import parse_calib, parse_pose


class SemanticKittiDataset(data.Dataset):
    def __init__(self, root, split, voxel_size, num_points, align, trainval):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.align = align

        self.sequences = os.path.join(self.root, 'sequences')
        self.metafile = os.path.join(self.root, 'semantic-kitti.yaml')
        with open(self.metafile, 'r') as f:
            self.meta = yaml.safe_load(f)

        self.seqs = self.meta['split'][self.split]
        if trainval and self.split is 'train':
            self.seqs.extend(self.meta['split']['val'])
        self.num_classes = len(self.meta['learning_ignore']) - 1
        self.ignore_label = 255
        self.label_map = np.zeros(260, dtype=np.int32) + 255
        self.label_map_inv = np.zeros(260, dtype=np.int32) + 255
        for k, v in self.meta['learning_map'].items():
            self.label_map[k] = v if v == 255 else v - 1
        for k, v in self.meta['learning_map_inv'].items():
            if k == 255:
                self.label_map_inv[k] = v
            else:
                self.label_map_inv[k-1] = self.meta['learning_map_inv'][k]

        self._prepare_files()
        # self._prepare_calib()
        self.angle = 0.0

    def _prepare_calib(self):
        self.calibs = {}
        self.poses = {}

        for seq in self.seqs:
            # calibration file
            self.calibs[seq] = parse_calib(
                os.path.join(self.sequences, seq, 'calib.txt'))
            self.poses[seq] = parse_pose(os.path.join(
                self.sequences, seq, 'poses.txt'), self.calibs[seq])

    def _prepare_files(self):
        self.files = []
        for seq in self.seqs:
            seq_files = sorted(os.listdir(os.path.join(
                self.sequences, seq, 'velodyne')))
            seq_files_with_prefix = [os.path.join(
                self.sequences, seq, 'velodyne', x) for x in seq_files]
            self.files.extend(seq_files_with_prefix)
        self.all_files = self.files
        self.label_files = [filename.replace('velodyne', 'labels').replace(
            '.bin', '.label') for filename in self.files]

    def _data_align(self, block_, cur_filename, last_filename):
        """Align each frame to the last frame in a clip"""
        def parse(name):
            seq = name.split('/')[-3]
            index = int(name.split('/')[-1].split('.')[0])
            return seq, index
        cur_seq, cur_index = parse(cur_filename)
        last_seq, last_index = parse(last_filename)

        cur_pose = self.poses[cur_seq][cur_index]
        last_pose = self.poses[last_seq][last_index]
        diff = np.linalg.inv(last_pose) @ cur_pose
        return np.dot(block_, diff.T)

    def _data_augment(self, block_, new=True):
        block = np.zeros_like(block_)
        if new or not hasattr(self, 'transform_mat'):
            if self.split == 'train':
                theta = np.random.uniform(0, 2 * np.pi)
                scale_mat = np.eye(3) + np.random.randn(3, 3) * 0.1
                rot_mat = np.array([
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
                self.transform_mat = np.dot(scale_mat, rot_mat)
            else:
                theta = self.angle
                self.transform_mat = np.array([
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

        block[:, :3] = np.dot(block_[:, :3], self.transform_mat)
        # block[:, 3:] = block_[:, 3:] + np.random.randn(3) * 0.1
        return block

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 1. prepare point clouds
        with open(self.files[index], 'rb') as b:
            block = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        if self.align and index != 0:
            block = self._data_align(
                block, self.files[index], self.files[index-1])
        block = self._data_augment(block)
        pc = np.round(block[:, :3] / self.voxel_size)
        # pc -= pc.min(0, keepdims=1)
        feat = block[:, :3]

        # 2. prepare labels
        if self.split == 'test':
            all_labels = np.zeros((pc.shape[0])).astype(np.int32)
        else:
            with open(self.label_files[index], 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        all_labels = all_labels & 0xFFFF
        labels = self.label_map[all_labels]

        # 3. sparse quantize
        inds, _, inverse_map = sparse_quantize(
            pc, feat, labels, return_index=True)

        # 4. sample voxels
        if self.split == 'train':
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)
        pc = pc[inds]
        feat = feat[inds]
        labels = labels[inds]

        # 5. metadata
        metadata = {}
        if self.split != 'train':
            metadata['index'] = index
            metadata['inverse_map'] = torch.tensor(inverse_map)
            metadata['file_path'] = self.files[index]

        return pc, feat, labels, metadata

    @staticmethod
    def collate_fn(tbl):
        locs, feats, labels, metadata = zip(*tbl)
        return sparse_collate(locs, feats, labels) + (metadata,)
