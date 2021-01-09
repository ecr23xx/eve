import torch.utils.data

from . import samplers
from . import datasets as D
from utils.comm import get_world_size
from utils.imports import import_file


def build_dataset(cfg, dataset_name, dataset_catalog, stage='train'):
    data = dataset_catalog.get(dataset_name.lower())
    factory = getattr(D, data['factory'])
    args = data['args']
    
    if data['factory'] == 'SemanticKittiDataset':
        args['align'] = cfg.INPUT.ALIGN

    elif data['factory'] == 'SemanticKittiDataset4d':
        args['num_frames'] = cfg.INPUT.VIDEO.NUM_FRAMES
        args['align'] = cfg.INPUT.ALIGN

    else:
        raise NotImplementedError

    args['num_points'] = cfg.INPUT.NUM_POINTS
    args['voxel_size'] = cfg.INPUT.VOXEL_SIZE
    args['trainval'] = cfg.DATASETS.TRAINVAL
    args['split'] = stage
    dataset = factory(**args)
    return dataset


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, pcs_per_batch, num_iters=None, start_iter=0, drop_last=False):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, pcs_per_batch, drop_last=drop_last
    )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, stage='train', is_distributed=False, start_iter=0):
    assert stage in ['train', 'val', 'test']
    num_gpus = get_world_size()
    is_train = (stage == 'train')

    if is_train:
        pcs_per_gpu = cfg.SOLVER.PCS_PER_GPU_TRAIN
        shuffle = True
        drop_last = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        pcs_per_gpu = cfg.SOLVER.PCS_PER_GPU_VAL
        shuffle = False
        drop_last = False
        num_iters = None
        start_iter = 0

    paths_catalog = import_file(
        'config.paths_catalog', 'config/path_catalog.py', True)

    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_name = cfg.DATASETS[stage.upper()]
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = build_dataset(cfg, dataset_name, DatasetCatalog, stage)
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, pcs_per_gpu, num_iters, start_iter, drop_last)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    return data_loader
