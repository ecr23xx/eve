import os
from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.WEIGHT = ""
_C.MODEL.NUM_CLASSES = 13
_C.MODEL.DIM_IN = 3

_C.MODEL.ARCHITECTURE = "minkunet"
_C.MODEL.DIM_IN = 3  # feature dim in
_C.MODEL.FREEZE_PREFIX = ()

# used for EVE
_C.MODEL.EVE = CN()
_C.MODEL.EVE.MATCH_ALGO = 'NN'       # matching algorithm name
_C.MODEL.EVE.MATCH_GT = False        # only match correct pair (use gt label)
_C.MODEL.EVE.MATCH_RATIO = 0.5       # matching ratio
_C.MODEL.EVE.MATCH_ITER = 15         # max iteration for icp matching
_C.MODEL.EVE.MATCH_THRESH = 5e-3     # thershold for icp matching
_C.MODEL.EVE.FUSION = False          # fusion or not
_C.MODEL.EVE.DIM_OUT = 96            # eve out feature channel. used for fusion

# ---------------------------------------------------------------------------- #
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ""
_C.DATASETS.VAL = ""
_C.DATASETS.TEST = ""
_C.DATASETS.TRAINVAL = False

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.ALIGN = False       # align each frame

_C.INPUT.VOXEL_SIZE = 0.05
_C.INPUT.NUM_POINTS = 80000

_C.INPUT.VIDEO = CN()
_C.INPUT.VIDEO.NUM_FRAMES = 5

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 400000

# scheuler: cosine_annealing, warmup_multi_step
_C.SOLVER.BASE_LR = 0.1
_C.SOLVER.SCHEDULE = "cosine_annealing"
_C.SOLVER.WEIGHT_DECAY = 0.001
_C.SOLVER.STEPS = (150000, 300000)
# warmup: linear, constant
_C.SOLVER.WARMUP_ON = False
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.LOG_PERIOD = 10000
_C.SOLVER.EVAL_PERIOD = 20000
_C.SOLVER.CHECKPOINT_PERIOD = 10000
# total batch size will be PCS_PER_GPU * NUM_GPUS
_C.SOLVER.PCS_PER_GPU_TRAIN = 16
_C.SOLVER.PCS_PER_GPU_VAL = 1

# number of views in evaluation. used for rotation augument.
_C.SOLVER.NUM_VIEWS = 1

_C.SOLVER.RANDOM_SEED = 0

# ---------------------------------------------------------------------------- #
# Logs
# ---------------------------------------------------------------------------- #
_C.LOGS = CN()
_C.LOGS.DIR = "."
_C.LOGS.SAVE_RESULT = False

# ---------------------------------------------------------------------------- #
# Other
# ---------------------------------------------------------------------------- #
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
_C.DEBUG = True
