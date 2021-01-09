import os


class DatasetCatalog(object):
    DATA_DIR = 'data'

    @staticmethod
    def get(name):
        if 'semantic_kitti_4d' in name:
            data_dir = DatasetCatalog.DATA_DIR
            args = dict(
                root=os.path.join(data_dir, 'semanticKITTI'),
            )
            return dict(
                factory='SemanticKittiDataset4d',
                args=args
            )
        elif 'semantic_kitti_eve' in name:
            data_dir = DatasetCatalog.DATA_DIR
            args = dict(
                root=os.path.join(data_dir, 'semanticKITTI'),
            )
            return dict(
                factory='SemanticKittiDatasetEve',
                args=args
            )
        elif 'semantic_kitti' in name:
            data_dir = DatasetCatalog.DATA_DIR
            args = dict(
                root=os.path.join(data_dir, 'semanticKITTI'),
            )
            return dict(
                factory='SemanticKittiDataset',
                args=args
            )
        else:
            raise RuntimeError("Dataset not available: {}".format(name))
