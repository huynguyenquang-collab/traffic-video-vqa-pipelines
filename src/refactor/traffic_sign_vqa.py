"""Traffic-sign VQA JSONL conversion and splitting."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .translation_utils import batch_translate, save_translation_cache

INPUT_BASE = Path("/kaggle/input/datasets/huyqn12/cropped-zalo")


def _normalize_content(msg: dict) -> None:
    content = msg.get("content", [])
    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        msg["content"] = [
            {"type": "text", "text": item} if isinstance(item, str) else item
            for item in content
        ]


def prepare_traffic_vqa_english(
    input_base: Path = INPUT_BASE,
    work_dir: Path = Path("/kaggle/working"),
) -> Path:
    cached_vqa_en = input_base / "traffic_vqa_en.jsonl"
    working_vqa_en = work_dir / "traffic_vqa_en.jsonl"
    if cached_vqa_en.exists():
        return cached_vqa_en
    if working_vqa_en.exists():
        return working_vqa_en

    cached_vqa = input_base / "traffic_vqa_qwen2vl_fixed.jsonl"
    input_jsonl = cached_vqa if cached_vqa.exists() else Path("/kaggle/input/vqa-sign/traffic_vqa_qwen2vl (1).jsonl")
    old_prefix = "/kaggle/input/traffic_sign_images/"
    new_prefix = "/kaggle/input/traffic-sign-images/traffic_sign_images/"

    raw_entries = []
    with input_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            raw_entries.append(json.loads(line))

    texts_to_translate, text_locations = [], []
    for entry_idx, entry in enumerate(raw_entries):
        for msg_idx, msg in enumerate(entry.get("messages", [])):
            _normalize_content(msg)
            for content_idx, content in enumerate(msg.get("content", [])):
                if content.get("type") == "image":
                    img_path = content["image"]
                    if img_path.startswith(old_prefix):
                        content["image"] = img_path.replace(old_prefix, new_prefix)
                elif content.get("type") == "text":
                    texts_to_translate.append(content["text"])
                    text_locations.append((entry_idx, msg_idx, content_idx))

    translated = batch_translate(texts_to_translate)
    save_translation_cache()
    for idx, (entry_idx, msg_idx, content_idx) in enumerate(text_locations):
        raw_entries[entry_idx]["messages"][msg_idx]["content"][content_idx]["text"] = translated[idx]

    with working_vqa_en.open("w", encoding="utf-8") as fout:
        for entry in raw_entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return working_vqa_en


def load_sign_vqa_split(traffic_vqa_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    sign_data = []
    with traffic_vqa_path.open("r", encoding="utf-8") as f:
        for line in f:
            sign_data.append(json.loads(line))
    random.seed(42)
    indices = list(range(len(sign_data)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    sign_train = [sign_data[i] for i in indices[:split]]
    sign_eval = [sign_data[i] for i in indices[split:]]
    return sign_data, sign_train, sign_eval

