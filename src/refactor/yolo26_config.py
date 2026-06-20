"""Configuration for the YOLO26 training notebook conversion."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Yolo26Config:
    src_folder: Path = Path("/kaggle/input/vietnamese-traffic-signs/archive")
    dst_folder: Path = Path("/kaggle/working/dataset_yolov11")
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    img_size: int = 640
    seed: int = 42
    pretrained_weights: str = "yolo26m.pt"
    tuning_trials: int = 8
    tuning_epochs: int = 10
    batch: int = 16
    device: int = 0
    workers: int = 0
    optuna_project: str = "optuna_tuning"
    best_params_path: Path = Path("/kaggle/working/best_hyperparameters.pt")

    @property
    def data_yaml_path(self) -> Path:
        return self.dst_folder / "data.yaml"

