from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import maybe_relative, resolve_path


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def annotation_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "data" not in payload or not isinstance(payload["data"], list):
        raise ValueError("Annotation JSON must contain a top-level list at key 'data'.")
    return payload["data"]


def resolve_video_path(value: str, *, annotation_dir: Path, video_root: Path, data_root: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    for base in (video_root, data_root, annotation_dir):
        candidate = resolve_path(raw, base)
        if candidate and candidate.exists():
            return candidate
    return resolve_path(raw, video_root)  # type: ignore[return-value]


def resolve_annotation_video_paths(
    payload: dict[str, Any],
    annotation_path: str | Path,
    *,
    video_root: str | Path,
    data_root: str | Path,
    keep_original: bool = True,
) -> dict[str, Any]:
    """Return a copy with video_path values made absolute for local execution."""
    data = json.loads(json.dumps(payload, ensure_ascii=False))
    annotation_dir = Path(annotation_path).resolve().parent
    for item in annotation_items(data):
        original = item.get("video_path")
        if not original:
            continue
        if keep_original:
            item.setdefault("video_path_original", original)
        item["video_path"] = str(
            resolve_video_path(
                original,
                annotation_dir=annotation_dir,
                video_root=Path(video_root),
                data_root=Path(data_root),
            )
        )
    return data


def relativize_annotation_video_paths(
    payload: dict[str, Any],
    *,
    base: str | Path,
    keep_absolute_key: str | None = "video_path_abs",
) -> dict[str, Any]:
    """Return a copy with video_path values relative to base when possible."""
    data = json.loads(json.dumps(payload, ensure_ascii=False))
    for item in annotation_items(data):
        path = item.get("video_path")
        if not path:
            continue
        if keep_absolute_key:
            item[keep_absolute_key] = str(Path(path).resolve())
        item["video_path"] = maybe_relative(path, base)
    return data


def split_by_video(
    payload: dict[str, Any],
    *,
    split_ratio: float = 0.8,
    seed: int = 42,
    key: str = "video_path",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split samples by video id/path so the same video does not leak across splits."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotation_items(payload):
        grouped[str(item[key])].append(item)

    videos = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(videos)
    cut = int(split_ratio * len(videos))

    train_items = [item for video in videos[:cut] for item in grouped[video]]
    eval_items = [item for video in videos[cut:] for item in grouped[video]]
    return {"__count__": len(train_items), "data": train_items}, {
        "__count__": len(eval_items),
        "data": eval_items,
    }


def normalize_message_content(message: dict[str, Any]) -> None:
    content = message.get("content", [])
    if isinstance(content, str):
        message["content"] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        message["content"] = [{"type": "text", "text": x} if isinstance(x, str) else x for x in content]


def rebase_qwen_message_images(
    entries: list[dict[str, Any]],
    *,
    old_prefix: str,
    new_root: str | Path,
) -> list[dict[str, Any]]:
    """Patch image paths inside Qwen-style message datasets."""
    patched = json.loads(json.dumps(entries, ensure_ascii=False))
    new_root = str(new_root)
    for entry in patched:
        for message in entry.get("messages", []):
            normalize_message_content(message)
            for part in message.get("content", []):
                if part.get("type") == "image" and str(part.get("image", "")).startswith(old_prefix):
                    tail = str(part["image"])[len(old_prefix) :].lstrip("/")
                    part["image"] = str(Path(new_root) / tail)
    return patched


def split_list(rows: list[Any], *, eval_ratio: float = 0.1, seed: int = 42) -> tuple[list[Any], list[Any]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    cut = int((1.0 - eval_ratio) * len(indices))
    return [rows[i] for i in indices[:cut]], [rows[i] for i in indices[cut:]]
