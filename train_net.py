import os
import random
import argparse
import numpy as np

import torch
import MinkowskiEngine as ME
import torch.backends.cudnn as cudnn

from config import cfg
from dataset import make_data_loader
from engine.trainer import do_train, val_in_train
from modeling.classifier import build_model
from solver import make_optimizer, make_lr_scheduler
from utils.checkpoint import Checkpointer
from utils.comm import synchronize, get_rank
from utils.logger import setup_logger, setup_tblogger

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


def train(cfg, local_rank, distributed, logger=None, tblogger=None,
          transfer_weight=False, change_lr=False):
    device = torch.device('cuda')

    # create model
    logger.info('Creating model "{}"'.format(cfg.MODEL.ARCHITECTURE))
    model = build_model(cfg).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
    optimizer = make_optimizer(cfg, model)
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level='O2')
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        # model = apex.parallel.DistributedDataParallel(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,)

    save_to_disk = get_rank() == 0

    # checkpoint
    arguments = {}
    arguments['iteration'] = 0
    arguments['best_iou'] = 0
    checkpointer = Checkpointer(
        model, optimizer, scheduler, cfg.LOGS.DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load(
        f=cfg.MODEL.WEIGHT,
        model_weight_only=transfer_weight,
        change_scheduler=change_lr)
    arguments.update(extra_checkpoint_data)

    # data_loader
    logger.info('Loading dataset "{}"'.format(cfg.DATASETS.TRAIN))
    data_loader = make_data_loader(cfg, 'train', distributed)
    data_loader_val = make_data_loader(cfg, 'val', distributed)

    do_train(
        cfg,
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        checkpointer=checkpointer,
        device=device,
        arguments=arguments,
        tblogger=tblogger,
        data_loader_val=data_loader_val,
        distributed=distributed
    )


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
        "--transfer",
        dest="transfer_weight",
        help="Transfer weight from a pretrained model",
        action="store_true")
    parser.add_argument(
        "--change-lr",
        dest="change_lr",
        help="Change learning rate during the training process",
        action="store_true")
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
    tblogger = setup_tblogger(cfg.LOGS.DIR, get_rank())

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

    train(cfg, args.local_rank, args.distributed,
          logger=logger, tblogger=tblogger,
          transfer_weight=args.transfer_weight,
          change_lr=args.change_lr)
