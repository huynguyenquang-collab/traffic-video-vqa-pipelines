from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .data import (
    iter_jsonl,
    normalize_message_content,
    read_json,
    rebase_qwen_message_images,
    resolve_annotation_video_paths,
    split_by_video,
    write_json,
    write_jsonl,
)
from .translation import TranslationCache
from .video import extract_sample_frames


def create_splits(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_json = paths["train_annotations"]
    payload = resolve_annotation_video_paths(
        read_json(train_json),
        train_json,
        video_root=paths["video_root"],
        data_root=paths["data_root"],
        keep_original=data_cfg.get("keep_original_paths", True),
    )
    train, eval_ = split_by_video(
        payload,
        split_ratio=float(data_cfg["split_ratio"]),
        seed=int(cfg["seed"]),
        key=data_cfg.get("split_by", "video_path"),
    )
    write_json(train, paths["train_split"])
    write_json(eval_, paths["eval_split"])
    return train, eval_


def convert_video_qa_to_qwen_dataset(cfg: dict[str, Any], split_path: str | Path | None = None) -> list[dict[str, Any]]:
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    split_path = split_path or paths["train_split"]
    payload = read_json(split_path)
    translator = TranslationCache(paths["translation_cache"])

    items = payload["data"]
    questions = [x["question"] for x in items]
    answers = [x.get("answer", "") for x in items]
    choices_flat: list[str] = []
    choice_map: list[tuple[int, int]] = []
    for i, item in enumerate(items):
        for j, choice in enumerate(item.get("choices", [])):
            choices_flat.append(choice)
            choice_map.append((i, j))

    questions_en = translator.batch(questions)
    answers_en = translator.batch(answers)
    choices_en_flat = translator.batch(choices_flat)
    choices_en: list[list[str]] = [[] for _ in items]
    for flat_idx, (item_idx, _) in enumerate(choice_map):
        choices_en[item_idx].append(choices_en_flat[flat_idx])

    converted: list[dict[str, Any]] = []
    for idx, item in enumerate(tqdm(items, desc="Converting video QA")):
        video_path = item["video_path"]
        video_id = Path(video_path).stem
        frame_paths = extract_sample_frames(
            video_path,
            paths["frames_dir"],
            video_id,
            max_frames=int(data_cfg["train_frame_count"]),
            max_size=int(data_cfg["frame_max_size"]),
        )
        if not frame_paths:
            continue
        user_text = questions_en[idx]
        if choices_en[idx]:
            user_text += "\n" + "\n".join(choices_en[idx])
        converted.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": p} for p in frame_paths],
                            {"type": "text", "text": user_text},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": answers_en[idx]}]},
                ]
            }
        )

    write_json(converted, paths["converted_train"])
    return converted


def translate_traffic_sign_vqa(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sign_cfg = cfg["traffic_sign_vqa"]
    translator = TranslationCache(cfg["paths"]["translation_cache"])
    source = Path(sign_cfg["jsonl"])
    translated_path = Path(sign_cfg["translated_jsonl"])
    if translated_path.exists():
        return list(iter_jsonl(translated_path))

    entries = list(iter_jsonl(source))
    entries = rebase_qwen_message_images(
        entries,
        old_prefix=sign_cfg.get("image_old_prefix", ""),
        new_root=sign_cfg["image_new_root"],
    )

    texts: list[str] = []
    locations: list[tuple[int, int, int]] = []
    for entry_idx, entry in enumerate(entries):
        for msg_idx, message in enumerate(entry.get("messages", [])):
            normalize_message_content(message)
            for part_idx, part in enumerate(message.get("content", [])):
                if part.get("type") == "text":
                    texts.append(part["text"])
                    locations.append((entry_idx, msg_idx, part_idx))

    translated = translator.batch(texts)
    for text_idx, (entry_idx, msg_idx, part_idx) in enumerate(locations):
        entries[entry_idx]["messages"][msg_idx]["content"][part_idx]["text"] = translated[text_idx]

    write_jsonl(entries, translated_path)
    return entries


def build_mixed_training_dataset(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    import random

    paths = cfg["paths"]
    converted = read_json(paths["converted_train"])
    mixed = list(converted)
    if cfg.get("traffic_sign_vqa", {}).get("enabled", False):
        mixed.extend(translate_traffic_sign_vqa(cfg))
    random.Random(int(cfg["seed"])).shuffle(mixed)
    write_json(mixed, paths["mixed_train"])
    return mixed


def prepare_test_annotations(cfg: dict[str, Any], source_path: str | Path | None = None) -> dict[str, Any]:
    paths = cfg["paths"]
    source_path = source_path or paths["public_test_annotations"]
    payload = resolve_annotation_video_paths(
        read_json(source_path),
        source_path,
        video_root=paths["video_root"],
        data_root=paths["data_root"],
        keep_original=True,
    )
    translator = TranslationCache(paths["translation_cache"])

    items = payload["data"]
    questions = [x["question"] for x in items]
    answers = [x.get("answer", "") for x in items]
    choices_flat: list[str] = []
    choice_map: list[tuple[int, int]] = []
    for i, item in enumerate(items):
        for j, choice in enumerate(item.get("choices", [])):
            choices_flat.append(choice)
            choice_map.append((i, j))

    questions_en = translator.batch(questions)
    answers_en = translator.batch(answers)
    choices_en_flat = translator.batch(choices_flat)

    translated_choices: list[list[str]] = [[] for _ in items]
    for flat_idx, (item_idx, _) in enumerate(choice_map):
        translated_choices[item_idx].append(choices_en_flat[flat_idx])

    for idx, item in enumerate(items):
        item["question_vi"] = item["question"]
        item["question"] = questions_en[idx]
        if "answer" in item:
            item["answer_vi"] = item["answer"]
            item["answer"] = answers_en[idx]
        item["choices_vi"] = item.get("choices", [])
        item["choices"] = translated_choices[idx]

    write_json(payload, paths["test_en"])
    return payload


def maybe_repin_loaded_numpy() -> None:
    """Kaggle helper for Unsloth notebooks where pip metadata drifts from loaded numpy."""
    import subprocess
    import sys

    import numpy as np

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", f"numpy=={np.__version__}"],
        check=True,
    )
