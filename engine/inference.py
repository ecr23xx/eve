import os
import time
import logging
import datetime
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
import MinkowskiEngine as ME

from config import cfg
from dataset.evaluation import evaluate
from utils.comm import get_rank, is_main_process, scatter_gather, synchronize, get_world_size


def compute_one_frame(outputs, targets, locs_frame, inv_map):
    cpu_device = torch.device('cpu')
    outputs_logits = F.softmax(outputs[locs_frame], dim=1)
    one_output = outputs_logits.argmax(1)[inv_map].to(cpu_device)
    one_target = targets[locs_frame][inv_map].to(cpu_device)
    return one_output, one_target


def inference(model, criterion, data_loader, dataset_name, save_result=False):
    logger = logging.getLogger('eve.' + __name__)

    device = torch.device('cuda')
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset ({} point clouds).".format(
        dataset_name, len(dataset)))

    if get_world_size() == 1:
        extra_args = {}
    else:
        rank = get_rank()
        extra_args = dict(desc="rank {}".format(rank))

    start_time = time.time()

    model.eval()
    outputs_per_gpu = {}
    targets_per_gpu = {}
    file_path_per_gpu = {}

    times = []

    with torch.no_grad():
        for batch in tqdm(data_loader, **extra_args):
            locs, feats, targets, metadata = batch
            inputs = ME.SparseTensor(feats, coords=locs).to(device)
            targets = targets.to(device, non_blocking=True).long()

            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(inputs, y=targets)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

            arch = cfg.MODEL.ARCHITECTURE
            if arch == 'minkunet4d' or arch == 'minkunet_eve':
                for batch_idx in range(len(metadata)):
                    for time_idx in range(cfg.INPUT.VIDEO.NUM_FRAMES):
                        inv_map = metadata[batch_idx][time_idx]['inverse_map']
                        file_path = metadata[batch_idx][time_idx]['file_path']

                        locs_frame = (locs[:, -1] == batch_idx) & \
                            (locs[:, -2] == time_idx)
                        one_output, one_target = compute_one_frame(
                            outputs, targets, locs_frame, inv_map)

                        outputs_per_gpu[file_path] = one_output
                        targets_per_gpu[file_path] = one_target
                        file_path_per_gpu[file_path] = file_path
            else:  # other minknet
                for batch_idx in range(len(metadata)):
                    inv_map = metadata[batch_idx]['inverse_map']
                    file_path = metadata[batch_idx]['file_path']

                    # From MinkowskiEngine v0.3, batch index is on the first column
                    locs_frame = locs[:, -1] == batch_idx
                    one_output, one_target = compute_one_frame(
                        outputs, targets, locs_frame, inv_map)

                    outputs_per_gpu[file_path] = one_output
                    targets_per_gpu[file_path] = one_target
                    file_path_per_gpu[file_path] = file_path

    synchronize()

    logger.info("Total inference time: {}".format(np.sum(times)))

    # NOTE: `all_gather` will lead to CUDA out of memory
    # We use `scatter_gather` to save result of each process
    # in LOGS.DIR/tmp and will be cleared after gathering.
    outputs = scatter_gather(outputs_per_gpu)
    targets = scatter_gather(targets_per_gpu)
    file_paths = scatter_gather(file_path_per_gpu)
    if not is_main_process():
        return None

    all_outputs = {k: v.numpy() for o in outputs for k, v in o.items()}
    all_targets = {k: v.numpy() for t in targets for k, v in t.items()}
    all_file_paths = {k: v for f in file_paths for k, v in f.items()}

    assert len(all_outputs) == len(dataset.all_files), \
        '%d vs %d' % (len(all_outputs), len(dataset.all_files))

    if cfg.LOGS.SAVE_RESULT is False:
        all_file_paths = None
    metrics = evaluate(dataset, all_outputs, all_targets, all_file_paths)

    return metrics
