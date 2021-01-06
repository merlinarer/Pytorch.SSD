from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.NAME = ""
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = "0,1,2,3"

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "SSDVGG"
_C.MODEL.BACKBONE.DEPTH = 16
_C.MODEL.BACKBONE.PRETRAIN_PATH = ""
_C.MODEL.BACKBONE.OUT_INDICES = (22, 34)

# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "EmbeddingHead"
_C.MODEL.HEADS.NUM_CLASSES = 0




# ---------------------------------------------------------------------------- #
# LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = "CrossEntropyLoss"

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 300
# Size of the image during test
_C.INPUT.SIZE_TEST = 300


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = 'VOC'
_C.DATA.ROOT = ''
_C.DATA.VALROOT = '/home/workspace/merlin/data_dirs/VOCdevkit/VOCdevkit/'

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# CHECKPOINT_PERIOD and EVAL_PERIOD
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.EVAL_PERIOD = 1

# Optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [30, 55]
_C.SOLVER.CHECKPOINT_PERIOD = 20
_C.SOLVER.BATCH_SIZE = 64

_C.SOLVER.WARMUP = False
_C.SOLVER.WARMUP_ITERS = 0


# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()

# Number of images per batch in one process.
_C.TEST.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.LOAD_FROM = ""
_C.RESUME = False
_C.VARIRANCE = [1, 1]
