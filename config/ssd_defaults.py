from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

Cfg = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
Cfg.MODEL = CN()

Cfg.MODEL.NAME = ""
Cfg.MODEL.DEVICE = "cuda"
Cfg.MODEL.DEVICE_ID = "0,1,2,3"

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
Cfg.MODEL.BACKBONE = CN()

Cfg.MODEL.BACKBONE.NAME = "SSDVGG"
Cfg.MODEL.BACKBONE.DEPTH = 16
Cfg.MODEL.BACKBONE.PRETRAIN_PATH = ""
Cfg.MODEL.BACKBONE.OUT_INDICES = (22, 34)

# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
Cfg.MODEL.HEADS = CN()
Cfg.MODEL.HEADS.NAME = "EmbeddingHead"
Cfg.MODEL.HEADS.NUM_CLASSES = 0

# ---------------------------------------------------------------------------- #
# LOSSES options
# ---------------------------------------------------------------------------- #
Cfg.MODEL.LOSSES = CN()
Cfg.MODEL.LOSSES.NAME = "CrossEntropyLoss"

# Values to be used for image normalization
Cfg.MODEL.PIXEL_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
# Values to be used for image normalization
Cfg.MODEL.PIXEL_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
Cfg.INPUT = CN()
# Size of the image during training
Cfg.INPUT.SIZE_TRAIN = 300
# Size of the image during test
Cfg.INPUT.SIZE_TEST = 300

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
Cfg.DATA = CN()
Cfg.DATA.NAME = 'VOC'
Cfg.DATA.ROOT = ''
Cfg.DATA.VALROOT = ''

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
Cfg.SOLVER = CN()

# CHECKPOINT_PERIOD and EVAL_PERIOD
Cfg.SOLVER.CHECKPOINT_PERIOD = 1
Cfg.SOLVER.EVAL_PERIOD = 1

# Optimizer
Cfg.SOLVER.OPTIMIZER_NAME = "Adam"
Cfg.SOLVER.MAX_EPOCHS = 100
Cfg.SOLVER.BASE_LR = 3e-4
Cfg.SOLVER.MOMENTUM = 0.9
Cfg.SOLVER.WEIGHT_DECAY = 0.0005
Cfg.SOLVER.GAMMA = 0.1
Cfg.SOLVER.STEPS = [30, 55]
Cfg.SOLVER.CHECKPOINT_PERIOD = 20
Cfg.SOLVER.BATCH_SIZE = 64

Cfg.SOLVER.WARMUP = False
Cfg.SOLVER.WARMUP_ITERS = 0

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
Cfg.TEST = CN()

# Number of images per batch in one process.
Cfg.TEST.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
Cfg.OUTPUT_DIR = ""
Cfg.LOAD_FROM = ""
Cfg.RESUME = False
Cfg.VARIRANCE = [1, 1]
