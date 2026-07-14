import os
from dataclasses import dataclass, field
from typing import List, Tuple

DEFAULT_DATA_PATH = os.environ.get("OAIZIB_DATA_PATH", "data/OAI-ZIB-CM")
KL4_SAMPLING_WEIGHTS = {0: 0.3, 1: 0.5, 2: 0.8, 3: 2.0, 4: 4.0}


@dataclass
class Config:
    data_path: str = DEFAULT_DATA_PATH
    checkpoint_dir: str = "checkpoints"
    num_classes: int = 6
    in_channels: int = 1
    features: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    elrs_rank: int = 16

    batch_size: int = 4
    grad_accum_steps: int = 2
    num_workers: int = 16
    prefetch_factor: int = 2
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_amp: bool = True
    device: str = "cuda"

    patch_size: Tuple[int, int, int] = (64, 192, 192)
    target_spacing: Tuple[float, float, float] = (0.7, 0.365, 0.365)

    edge_lambda: float = 0.1

    val_interval: int = 10
    early_stopping_patience: int = 60

    cv_folds: int = 5
    cv_fold: int = 0
    cv_seed: int = 35
    cv_inner_val_frac: float = 0.15

    def __post_init__(self):
        self.class_names = [
            "Background", "Femoral_Bone", "Femoral_Cartilage",
            "Tibial_Bone", "Medial_Tibial_Cartilage", "Lateral_Tibial_Cartilage",
        ]
        os.makedirs(self.checkpoint_dir, exist_ok=True)


def make_config(checkpoint_dir, fold, epochs=200, data_path=DEFAULT_DATA_PATH,
                num_workers=16, **overrides):
    cfg = Config(
        data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        num_workers=num_workers,
        epochs=epochs,
        cv_folds=5,
        cv_fold=fold,
        cv_seed=35,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
