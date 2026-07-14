import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .config import Config, make_config
from .model import MbEdNet
from .loss import MbEdLoss
from .data import OAIZIBDataset, build_loaders
from .metrics import SurfaceMetrics, postprocess_segmentation, largest_cc_all, dice_score, macro_dice
from .trainer import MbEdTrainer, run_experiment

__all__ = [
    "Config", "make_config", "MbEdNet", "MbEdLoss", "OAIZIBDataset", "build_loaders",
    "SurfaceMetrics", "postprocess_segmentation", "largest_cc_all", "dice_score",
    "macro_dice", "MbEdTrainer", "run_experiment",
]
