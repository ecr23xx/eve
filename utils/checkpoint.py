import os
import logging
from collections import OrderedDict

import torch

from config import cfg
from utils.imports import import_file
from utils.comm import get_rank


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.

    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    logger = logging.getLogger('eve.' + __name__)
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key)
                           for key in loaded_keys]) if loaded_keys else 1
    skip_prefixes = []
    
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if key.split('.')[0] in skip_prefixes or key.split('.')[1] in skip_prefixes:
            continue
        if model_state_dict[key].size() == loaded_state_dict[key_old].size():
            model_state_dict[key] = loaded_state_dict[key_old]
            # if cfg.DEBUG and get_rank() == 0:
            #     logger.info("{} -> {}".format(key_old, key))
        else:
            # TODO: WTF?
            logger.info("Size mismatch: {} -> {}".format(key_old, key))
            prefix = key.split('.')[0]
            if prefix == 'module':
                prefix = key.split('.')[1]
            logger.info("Will skip prefix with {}".format(prefix))
            skip_prefixes.append(prefix)


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


class Checkpointer(object):
    def __init__(self, model, optimizer=None, scheduler=None,
                 save_dir='', save_to_disk=None, logger=None,):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger("eve." + __name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data['model'] = self.model.state_dict()
        if self.optimizer is not None and not isinstance(self.optimizer, list):
            data['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None and not isinstance(self.scheduler, list):
            data['scheduler'] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None, model_weight_only=False, change_scheduler=False):
        if not f:
            # no checkpoint could be found
            self.logger.info(
                "No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from \"{}\"".format(f))
        checkpoint = self._load_file(f)

        # load model
        loaded_state_dict = checkpoint.pop('model')
        model_state_dict = self.model.state_dict()
        loaded_state_dict = strip_prefix_if_present(
            loaded_state_dict, prefix="module.")
        if len(model_state_dict.keys()) == len(loaded_state_dict.keys()):
            self.logger.info("Loaded state_dict is the same as model state_dict")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        self.model.load_state_dict(model_state_dict)

        # load optimizer
        if 'optimizer' in checkpoint and self.optimizer:
            if model_weight_only:
                del checkpoint['optimizer']
                self.logger.info("Initializing optimizer from scratch.")
            else:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        else:
            self.logger.info("Initializing optimizer from scratch.")

        # load scheduler
        if 'scheduler' in checkpoint and self.scheduler:
            if model_weight_only:
                del checkpoint['scheduler']
                self.logger.info("Initializing scheduler from scratch.")
            elif change_scheduler:
                last_epoch = checkpoint.pop('scheduler')['last_epoch']
                self.logger.info(
                    "Change scheduler from iteration {}".format(last_epoch))
                self.scheduler.step(last_epoch)
            else:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        else:
            self.logger.info("Initializing scheduler from scratch.")

        if model_weight_only:
            checkpoint['iteration'] = 0
            checkpoint['best_iou'] = 0

        # return any further checkpoint data
        return checkpoint

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))
