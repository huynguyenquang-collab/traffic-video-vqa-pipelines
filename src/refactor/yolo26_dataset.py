"""Dataset preparation from `notebooks/yolo26.ipynb`.

Converts the Vietnamese traffic-sign dataset to YOLO format, applies the same
training augmentations as the notebook, and writes both `dataset.yaml` and
`data.yaml`.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import albumentations as A
import cv2
import yaml
from tqdm import tqdm

from .yolo26_config import Yolo26Config


def count_source_images(config: Yolo26Config) -> int:
    image_dir = config.src_folder / "images"
    return len([p for p in image_dir.iterdir() if p.suffix.lower() == ".jpg"])


def create_augmentations(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0),
            A.RandomRotate90(p=0),
            A.Rotate(limit=15, p=0.5),
            A.Affine(shear=5, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.2),
            A.GaussNoise(p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(format="yolo"),
    )


def load_classes(src_folder: Path) -> list[str]:
    with (src_folder / "classes_vie.txt").open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_images(config: Yolo26Config) -> tuple[list[Path], list[Path], list[Path]]:
    images = [
        p
        for p in (config.src_folder / "images").iterdir()
        if p.suffix.lower() == ".jpg"
    ]
    random.seed(config.seed)
    random.shuffle(images)

    train_size = int(len(images) * config.train_ratio)
    valid_size = int(len(images) * config.valid_ratio)
    return (
        images[:train_size],
        images[train_size : train_size + valid_size],
        images[train_size + valid_size :],
    )


def _create_output_dirs(dst_folder: Path) -> None:
    for split in ["train", "valid", "test"]:
        (dst_folder / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_folder / split / "labels").mkdir(parents=True, exist_ok=True)


def _read_yolo_labels(label_path: Path) -> list[list[float]]:
    with label_path.open("r", encoding="utf-8") as f:
        return [list(map(float, line.split())) for line in f if line.strip()]


def _valid_yolo_bbox(bbox: list[float]) -> bool:
    x_center, y_center, bbox_w, bbox_h, _ = bbox
    return (
        0 <= x_center <= 1
        and 0 <= y_center <= 1
        and 0 <= bbox_w <= 1
        and 0 <= bbox_h <= 1
    )


def _write_yolo_labels(label_path: Path, bboxes: list[list[float]]) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for bbox in bboxes:
            cls_id = int(bbox[-1])
            bbox_coords = bbox[:4]
            f.write(f"{cls_id} {' '.join(map(str, bbox_coords))}\n")


def process_and_save(
    images: list[Path],
    split: str,
    config: Yolo26Config,
    augmentations: A.Compose,
) -> None:
    for img_path in tqdm(images, desc=f"Processing {split}"):
        filename = img_path.name
        label_filename = img_path.with_suffix(".txt").name

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skipping {filename}: could not read image.")
            continue

        labels = _read_yolo_labels(config.src_folder / "labels" / label_filename)
        bboxes = [label[1:] + [int(label[0])] for label in labels]
        if any(not _valid_yolo_bbox(bbox) for bbox in bboxes):
            print(f"Skipping {filename} due to invalid bbox values.")
            continue

        try:
            if split == "train":
                original_img_resized = cv2.resize(image, (config.img_size, config.img_size))
                cv2.imwrite(
                    str(config.dst_folder / split / "images" / f"original_{filename}"),
                    original_img_resized,
                )
                _write_yolo_labels(
                    config.dst_folder / split / "labels" / f"original_{label_filename}",
                    bboxes,
                )
                augmented = augmentations(image=image, bboxes=bboxes)
                img_processed = augmented["image"]
                bboxes_processed = augmented["bboxes"]
            else:
                img_processed = cv2.resize(image, (config.img_size, config.img_size))
                bboxes_processed = bboxes
        except ValueError as exc:
            print(f"Skipping {filename} due to augmentation error: {exc}")
            continue

        cv2.imwrite(str(config.dst_folder / split / "images" / filename), img_processed)
        _write_yolo_labels(config.dst_folder / split / "labels" / label_filename, bboxes_processed)


def write_dataset_yaml(config: Yolo26Config, classes: list[str]) -> None:
    dataset_yaml = {
        "path": os.path.abspath(config.dst_folder),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": classes,
    }
    with (config.dst_folder / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True)


def write_data_yaml(config: Yolo26Config, classes: list[str]) -> None:
    data_yaml = {
        "train": str(config.dst_folder / "train/images"),
        "val": str(config.dst_folder / "valid/images"),
        "test": str(config.dst_folder / "test/images"),
        "nc": len(classes),
        "names": classes,
    }
    with config.data_yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=None)


def prepare_yolo_dataset(config: Yolo26Config = Yolo26Config()) -> list[str]:
    _create_output_dirs(config.dst_folder)
    classes = load_classes(config.src_folder)
    train_images, valid_images, test_images = split_images(config)
    augmentations = create_augmentations(config.img_size)

    process_and_save(train_images, "train", config, augmentations)
    process_and_save(valid_images, "valid", config, augmentations)
    process_and_save(test_images, "test", config, augmentations)

    write_dataset_yaml(config, classes)
    write_data_yaml(config, classes)
    return classes


def main() -> None:
    config = Yolo26Config()
    print(f"Source images: {count_source_images(config)}")
    classes = prepare_yolo_dataset(config)
    print(f"Prepared {len(classes)} classes at {config.dst_folder}")


if __name__ == "__main__":
    main()

