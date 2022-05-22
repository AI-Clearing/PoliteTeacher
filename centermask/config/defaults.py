# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = "giou"


# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.VOVNET = CN()

_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256
_C.MODEL.VOVNET.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.VOVNET.WITH_MODULATED_DCN = False
_C.MODEL.VOVNET.DEFORMABLE_GROUPS = 1


# ---------------------------------------------------------------------------- #
# CenterMask
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD.ASSIGN_CRITERION = "area"
_C.MODEL.MASKIOU_ON = False
_C.MODEL.MASKIOU_LOSS_WEIGHT = 1.0

_C.MODEL.ROI_MASKIOU_HEAD = CN()
_C.MODEL.ROI_MASKIOU_HEAD.NAME = "MaskIoUHead"
_C.MODEL.ROI_MASKIOU_HEAD.CONV_DIM = 256
_C.MODEL.ROI_MASKIOU_HEAD.NUM_CONV = 4


# ---------------------------------------------------------------------------- #
# Keypoint Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_KEYPOINT_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.ROI_KEYPOINT_HEAD.ASSIGN_CRITERION = "ratio"

# ---------------------------------------------------------------------------- #
# Dataloader (from Unbiased Teacher)
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.SUP_PERCENT = 100.0
_C.DATALOADER.RANDOM_DATA_SEED = 1
_C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.REPEAT_THRESHOLD = 0.0
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

# ---------------------------------------------------------------------------- #
# Semi-supervised learning
# ---------------------------------------------------------------------------- #
_C.SEMISUPNET = CN()
_C.SEMISUPNET.Trainer = "DefaultTrainer"
_C.SEMISUPNET.BBOX_THRESHOLD = 0.7
_C.SEMISUPNET.MASK_THRESHOLD = 0.5
_C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
_C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
_C.SEMISUPNET.BURN_UP_STEP = 12000
_C.SEMISUPNET.EMA_KEEP_RATE = 0.0
_C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 1.0
_C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
_C.SEMISUPNET.MASK_LOSS = False
_C.SEMISUPNET.NORM_LOSS = False
_C.SEMISUPNET.NORM_LOSS_KEEP_RATE = 0.9997



# ---------------------------------------------------------------------------- #
# Solver (from Unbiased Teacher)
# ---------------------------------------------------------------------------- #

_C.SOLVER.IMG_PER_BATCH_LABEL = 1
_C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
_C.SOLVER.FACTOR_LIST = (1,)
_C.SOLVER.CHECKPOINT_PERIOD = 100000

# ---------------------------------------------------------------------------- #
# Datasets (from Unbiased Teacher)
# ---------------------------------------------------------------------------- #
_C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
_C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
_C.DATASETS.CROSS_DATASET = False


# ---------------------------------------------------------------------------- #
# CLEARML
# ---------------------------------------------------------------------------- #

_C.CLEARML = CN()
_C.CLEARML.ON = True


# ---------------------------------------------------------------------------- #
# DEBUGGING IDEAS
# ---------------------------------------------------------------------------- #

_C.DEBUG_OPT  = CN()
_C.DEBUG_OPT.BOX_THRESHOLD = 0.7
_C.DEBUG_OPT.FILTER_PSEUDO_INST = False
_C.DEBUG_OPT.GRAD_CLIPPING = False
_C.DEBUG_OPT.LOG_GRADIENT = False