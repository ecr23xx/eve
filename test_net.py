import os
import random
import argparse
import numpy as np

import torch
import MinkowskiEngine as ME
import torch.backends.cudnn as cudnn

from config import cfg
from dataset import make_data_loader
from engine.inference import inference
from modeling.classifier import build_model
from utils.checkpoint import Checkpointer
from utils.comm import synchronize, get_rank, is_main_process
from utils.logger import setup_logger

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


def test(cfg, local_rank, distributed, logger=None):
    device = torch.device('cuda')
    cpu_device = torch.device('cpu')

    # create model
    logger.info("Creating model \"{}\"".format(cfg.MODEL.ARCHITECTURE))
    model = build_model(cfg).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,)

    # checkpoint
    checkpointer = Checkpointer(model, save_dir=cfg.LOGS.DIR, logger=logger)
    _ = checkpointer.load(f=cfg.MODEL.WEIGHT)

    # data_loader
    logger.info('Loading dataset "{}"'.format(cfg.DATASETS.TEST))
    stage = cfg.DATASETS.TEST.split('_')[-1]
    data_loader = make_data_loader(cfg, stage, distributed)
    dataset_name = cfg.DATASETS.TEST

    metrics = inference(model, criterion, data_loader, dataset_name, True)

    if is_main_process():
        logger.info("Metrics:")
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,)

    args = parser.parse_args()

    # distributed setting
    if 'WORLD_SIZE' in os.environ:
        num_gpus = int(os.environ['WORLD_SIZE'])
    else:
        num_gpus = 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # setup logger for master process
    if not os.path.exists(cfg.LOGS.DIR):
        os.makedirs(cfg.LOGS.DIR, exist_ok=True)

    logger = setup_logger('eve', cfg.LOGS.DIR, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("MinkowskiEngine version: {}".format(ME.__version__))
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Merged configuration \n{}\n".format(cfg))

    # set devices
    cudnn.benchmark = False
    device = torch.device('cuda')

    # set random seed
    random.seed(cfg.SOLVER.RANDOM_SEED)
    np.random.seed(cfg.SOLVER.RANDOM_SEED)
    torch.manual_seed(cfg.SOLVER.RANDOM_SEED)
    if device == 'cuda':
        cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SOLVER.RANDOM_SEED)

    test(cfg, args.local_rank, args.distributed, logger=logger)
