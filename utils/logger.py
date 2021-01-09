import os
import sys
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        filename = os.path.join(save_dir, '{}.log'.format(datetime.now()))
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def setup_tblogger(save_dir, distributed_rank):
    if distributed_rank > 0:
        return None
    tbdir = os.path.join(save_dir)
    os.makedirs(tbdir, exist_ok=True)
    tblogger = SummaryWriter(tbdir)
    return tblogger
