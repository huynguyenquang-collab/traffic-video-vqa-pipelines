"""Mix video QA and traffic-sign QA datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path

INPUT_BASE = Path("/kaggle/input/datasets/huyqn12/cropped-zalo")


def mix_datasets(
    converted_dataset: list[dict],
    sign_data: list[dict],
    input_base: Path = INPUT_BASE,
    work_dir: Path = Path("/kaggle/working"),
) -> tuple[list[dict], list[dict], list[dict]]:
    cache_input = input_base / "mixed_train_en.json"
    cache_working = work_dir / "mixed_train_en.json"

    if cache_input.exists():
        with cache_input.open("r", encoding="utf-8") as f:
            mixed_dataset = json.load(f)
        print(f"Loaded mixed dataset from input ({len(mixed_dataset)} entries)")
    elif cache_working.exists():
        with cache_working.open("r", encoding="utf-8") as f:
            mixed_dataset = json.load(f)
        print(f"Loaded mixed dataset from working ({len(mixed_dataset)} entries)")
    else:
        mixed_dataset = converted_dataset + sign_data
        random.seed(42)
        random.shuffle(mixed_dataset)
        with cache_working.open("w", encoding="utf-8") as f:
            json.dump(mixed_dataset, f, ensure_ascii=False)
        print(f"Mixed {len(converted_dataset)} video QA + {len(sign_data)} sign QA.")

    random.seed(42)
    indices = list(range(len(mixed_dataset)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    return mixed_dataset, [mixed_dataset[i] for i in indices[:split]], [mixed_dataset[i] for i in indices[split:]]

