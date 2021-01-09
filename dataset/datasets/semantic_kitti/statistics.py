import torch
import numpy as np

from .semantic_kitti_4d import SemanticKittiDataset4d

class SK(SemanticKittiDataset4d):
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

            time_idxs = np.array([time_idx] * len(block)).reshape(-1, 1)

            pc_clip.append(np.concatenate((pc, time_idxs), axis=1))
            feat_clip.append(feat)
            labels_clip.append(labels)

            metadata = {}
            metadata['index'] = index
            metadata['time_idx'] = time_idx
            metadata['file_path'] = file_path
            metadata_clip.append(metadata)

        pc_clip = np.concatenate(pc_clip, axis=0)
        feat_clip = np.concatenate(feat_clip, axis=0)
        labels_clip = np.concatenate(labels_clip, axis=0)

        return pc_clip, feat_clip, labels_clip, metadata_clip
