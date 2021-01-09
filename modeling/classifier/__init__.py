import sys
from .minkunet import MinkUNet, MinkUNet14A, MinkUNet18A
from .minkunet4d import MinkUNet4d
from .minkunet_eve import MinkUNetEve

_ARCHITECTURES = {
    'minkunet': MinkUNet,
    'minkunet4d': MinkUNet4d,
    'minkunet_eve': MinkUNetEve,
    'minknet14a': MinkUNet14A,
    'minknet18a': MinkUNet18A,
}


def build_model(cfg):
    model_arch = _ARCHITECTURES[cfg.MODEL.ARCHITECTURE]
    return model_arch(cfg)
