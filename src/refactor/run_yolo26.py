"""Run the converted YOLO26 notebook in order."""

from .yolo26_config import Yolo26Config
from .yolo26_dataset import prepare_yolo_dataset
from .yolo26_train import tune_yolo


def main() -> None:
    config = Yolo26Config()
    prepare_yolo_dataset(config)
    tune_yolo(config)


if __name__ == "__main__":
    main()

