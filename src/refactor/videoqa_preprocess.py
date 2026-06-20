"""Video-QA preprocessing from the original Kaggle notebook."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

from .translation_utils import batch_translate, save_translation_cache

INPUT_BASE = Path("/kaggle/input/datasets/huyqn12/cropped-zalo")


def split_train_test(
    train_json_path: Path = Path("/kaggle/input/train3/train/train/train.json"),
    input_base: Path = INPUT_BASE,
    work_dir: Path = Path("/kaggle/working"),
) -> tuple[Path, Path]:
    cached_train = input_base / "train_split.json"
    cached_test = input_base / "test_split.json"
    if cached_train.exists() and cached_test.exists():
        print(f"Using cached splits: {cached_train}, {cached_test}")
        return cached_train, cached_test

    with train_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    video_dict = defaultdict(list)
    for item in data["data"]:
        video_dict[item["video_path"]].append(item)

    video_paths = list(video_dict.keys())
    random.seed(42)
    random.shuffle(video_paths)
    split_idx = int(0.8 * len(video_paths))

    train_items = [item for v in video_paths[:split_idx] for item in video_dict[v]]
    test_items = [item for v in video_paths[split_idx:] for item in video_dict[v]]

    train_split_path = work_dir / "train_split.json"
    test_split_path = work_dir / "test_split.json"
    with train_split_path.open("w", encoding="utf-8") as f:
        json.dump({"__count__": len(train_items), "data": train_items}, f, indent=2, ensure_ascii=False)
    with test_split_path.open("w", encoding="utf-8") as f:
        json.dump({"__count__": len(test_items), "data": test_items}, f, indent=2, ensure_ascii=False)
    return train_split_path, test_split_path


def prefix_train_paths(
    train_split_path: Path,
    input_base: Path = INPUT_BASE,
    prefix: str = "/kaggle/input/train3/train/",
    work_dir: Path = Path("/kaggle/working"),
) -> Path:
    cached_prefixed = input_base / "train_split_prefixed.json"
    if cached_prefixed.exists():
        print(f"Using cached train split: {cached_prefixed}")
        return cached_prefixed

    with train_split_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data["data"]:
        item["video_path"] = prefix + item["video_path"]

    output_path = work_dir / "train_split_prefixed.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return output_path


def extract_sample_frames(video_path: str, save_dir: Path, id_prefix: str, max_frames: int = 5, max_size: int = 2000) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    save_dir.mkdir(parents=True, exist_ok=True)
    idxs = [int(i * total / max_frames) for i in range(max_frames)]
    saved_paths = []
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        path = save_dir / f"{id_prefix}_{i:04d}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        saved_paths.append(str(path))
    cap.release()
    return saved_paths


def convert_train_to_english(
    train_split_prefixed_path: Path,
    input_base: Path = INPUT_BASE,
    work_dir: Path = Path("/kaggle/working"),
) -> tuple[Path, Path]:
    cached_conv = input_base / "converted_train_en.json"
    cached_frames = input_base / "train_frames"
    working_conv = work_dir / "converted_train_en.json"
    if cached_conv.exists():
        frames_dir = cached_frames if cached_frames.is_dir() else work_dir / "train_frames"
        return cached_conv, frames_dir
    if working_conv.exists():
        return working_conv, work_dir / "train_frames"

    frames_dir = work_dir / "train_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with train_split_prefixed_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_questions = [item["question"] for item in data["data"]]
    all_answers = [item.get("answer", "") for item in data["data"]]
    all_choices_flat, choice_map = [], []
    for i, item in enumerate(data["data"]):
        for j, choice in enumerate(item.get("choices", [])):
            all_choices_flat.append(choice)
            choice_map.append((i, j))

    translated_questions = batch_translate(all_questions)
    translated_answers = batch_translate(all_answers)
    translated_choices = batch_translate(all_choices_flat)
    save_translation_cache()

    translated_choices_per_item = [[] for _ in data["data"]]
    for flat_idx, (item_idx, _) in enumerate(choice_map):
        translated_choices_per_item[item_idx].append(translated_choices[flat_idx])

    converted_dataset = []
    for idx, item in enumerate(tqdm(data["data"], desc="Converting training data")):
        video_path = item["video_path"]
        id_prefix = os.path.splitext(os.path.basename(video_path))[0]
        frame_paths = extract_sample_frames(video_path, frames_dir, id_prefix, max_frames=4)
        if not frame_paths:
            continue
        choices_text = "\n".join(translated_choices_per_item[idx])
        user_text = f"{translated_questions[idx]}\n{choices_text}" if choices_text else translated_questions[idx]
        converted_dataset.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": path} for path in frame_paths],
                            {"type": "text", "text": user_text},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": translated_answers[idx]}]},
                ]
            }
        )

    with working_conv.open("w", encoding="utf-8") as f:
        json.dump(converted_dataset, f, indent=2, ensure_ascii=False)
    return working_conv, frames_dir


def load_converted_train(converted_json_path: Path, input_base: Path = INPUT_BASE) -> tuple[list[dict], list[dict]]:
    with converted_json_path.open("r", encoding="utf-8") as f:
        converted_dataset = json.load(f)
    cached_frames = input_base / "train_frames"
    if converted_json_path == input_base / "converted_train_en.json" and cached_frames.is_dir():
        old_prefix = "/kaggle/working/train_frames"
        for entry in converted_dataset:
            for msg in entry.get("messages", []):
                for part in msg.get("content", []):
                    if part.get("type") == "image" and part["image"].startswith(old_prefix):
                        part["image"] = part["image"].replace(old_prefix, str(cached_frames))

    random.seed(42)
    indices = list(range(len(converted_dataset)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    return [converted_dataset[i] for i in indices[:split]], [converted_dataset[i] for i in indices[split:]]


def prepare_test_english(
    test_split_path: Path,
    input_base: Path = INPUT_BASE,
    work_dir: Path = Path("/kaggle/working"),
) -> Path:
    cached_test_en = input_base / "test_en.json"
    working_test_en = work_dir / "test_en.json"
    cached_test_prefixed = input_base / "test_prefixed.json"
    if cached_test_en.exists():
        return cached_test_en
    if working_test_en.exists():
        return working_test_en

    if cached_test_prefixed.exists():
        src_path = cached_test_prefixed
    else:
        with test_split_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data["data"]:
            item["video_path"] = "/kaggle/input/train3/train/" + item["video_path"]
        src_path = work_dir / "test_prefixed.json"
        with src_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    all_q = [item["question"] for item in data["data"]]
    all_a = [item.get("answer", "") for item in data["data"]]
    all_c_flat, c_map = [], []
    for i, item in enumerate(data["data"]):
        for j, choice in enumerate(item.get("choices", [])):
            all_c_flat.append(choice)
            c_map.append((i, j))

    trans_q = batch_translate(all_q)
    trans_a = batch_translate(all_a)
    trans_c = batch_translate(all_c_flat)
    save_translation_cache()

    trans_choices_per = [[] for _ in data["data"]]
    for flat_idx, (item_idx, _) in enumerate(c_map):
        trans_choices_per[item_idx].append(trans_c[flat_idx])
    for idx, item in enumerate(data["data"]):
        item["question_vi"] = item["question"]
        item["question"] = trans_q[idx]
        item["answer_vi"] = item.get("answer", "")
        item["answer"] = trans_a[idx]
        item["choices_vi"] = item.get("choices", [])
        item["choices"] = trans_choices_per[idx]

    with working_test_en.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return working_test_en


def load_test_data(test_prefixed_path: Path) -> list[dict]:
    with test_prefixed_path.open("r", encoding="utf-8") as f:
        return json.load(f)["data"]

