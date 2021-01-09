import os
import random
import argparse
import numpy as np

import torch
import MinkowskiEngine as ME
import torch.backends.cudnn as cudnn

from config import cfg
from dataset import make_data_loader
from modeling.classifier import build_model
from utils.logger import setup_logger
from utils.profile import profiler


def profile(cfg, logger=None):
    device = torch.device('cuda')

    # create model
    logger.info("Creating model \"{}\"".format(cfg.MODEL.ARCHITECTURE))
    model = build_model(cfg).to(device)
    model.eval()

    # data_loader
    logger.info("Loading dataset \"{}\"".format(cfg.DATASETS.TRAIN))
    data_loader = make_data_loader(cfg, 'train', False)

    # profile
    locs, feats, targets, metadata = next(iter(data_loader))
    inputs = ME.SparseTensor(feats, coords=locs).to(device)
    targets = targets.to(device, non_blocking=True).long()
    return profiler(model, inputs={'x': inputs, 'y': targets})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # setup logger for master process
    if not os.path.exists(cfg.LOGS.DIR):
        os.makedirs(cfg.LOGS.DIR, exist_ok=True)

    logger = setup_logger('eve', cfg.LOGS.DIR, 0)
    logger.info(args)
    logger.info("MinkowskiEngine version: {}".format(ME.__version__))
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Merged configuration {}".format(cfg))

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

    macs, params = profile(cfg, logger=logger)
    logger.info("MACs: {}G".format(macs / 1e9))
    logger.info("Params: {}M".format(params / 1e6))
