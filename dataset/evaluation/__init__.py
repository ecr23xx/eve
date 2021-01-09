from dataset import datasets
from .semantic_kitti import do_sk_evaluation


def evaluate(dataset, outputs, targets, file_paths):
    if isinstance(dataset, datasets.SemanticKittiDataset):
        return do_sk_evaluation(dataset, outputs, targets, file_paths)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError(
            "Unsupported dataset type {}.".format(dataset_name))
