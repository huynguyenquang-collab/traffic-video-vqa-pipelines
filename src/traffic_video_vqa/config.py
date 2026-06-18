from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load default config and optionally merge a user override YAML."""
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    with default_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if path:
        with Path(path).expanduser().open("r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)

    cfg["_project_root"] = str(PROJECT_ROOT)
    resolve_config_paths(cfg)
    return cfg


def resolve_path(value: str | Path | None, base: str | Path | None = None) -> Path | None:
    """Expand env/user vars and resolve relative paths against base or project root."""
    if value in (None, ""):
        return None
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    p = Path(raw)
    if p.is_absolute():
        return p
    root = Path(base) if base else PROJECT_ROOT
    return (root / p).resolve()


def maybe_relative(path: str | Path, base: str | Path | None) -> str:
    """Return path relative to base when possible, otherwise an absolute string."""
    p = Path(path)
    if base is None:
        return str(p)
    try:
        return str(p.resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(p.resolve())


def resolve_config_paths(cfg: dict[str, Any]) -> None:
    """Resolve known path-like config entries in-place."""
    paths = cfg.setdefault("paths", {})
    root = resolve_path(paths.get("data_root"), PROJECT_ROOT)
    paths["data_root"] = str(root)

    for key in (
        "work_dir",
        "cache_dir",
        "output_dir",
        "train_split",
        "eval_split",
        "converted_train",
        "mixed_train",
        "test_en",
        "submission",
        "frames_dir",
        "key_frames_dir",
        "crops_dir",
        "frames_log",
        "translation_cache",
    ):
        if key in paths:
            paths[key] = str(resolve_path(paths[key], PROJECT_ROOT))

    video_root = resolve_path(paths.get("video_root", "."), root)
    paths["video_root"] = str(video_root)
    for key in ("train_annotations", "public_test_annotations", "private_test_annotations"):
        if key in paths:
            paths[key] = str(resolve_path(paths[key], root))

    models = cfg.setdefault("models", {})
    for key in ("base_vlm", "cached_finetuned_vlm", "output_finetuned_vlm", "yolo_weights"):
        if key in models and _looks_like_path(models[key]):
            models[key] = str(resolve_path(models[key], PROJECT_ROOT))

    sign = cfg.setdefault("traffic_sign_vqa", {})
    for key in ("jsonl", "image_new_root", "translated_jsonl"):
        if key in sign:
            sign[key] = str(resolve_path(sign[key], PROJECT_ROOT))

    rag = cfg.setdefault("rag", {})
    for key in ("ref_parquet_glob", "ref_corpus_cache", "sbert_cache", "clip_image_cache"):
        if key in rag:
            rag[key] = str(resolve_path(rag[key], PROJECT_ROOT))


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create output/cache directories referenced by the config."""
    paths = cfg["paths"]
    for key in ("work_dir", "cache_dir", "output_dir", "frames_dir", "key_frames_dir", "crops_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    for key in ("output_finetuned_vlm",):
        value = cfg.get("models", {}).get(key)
        if value:
            Path(value).parent.mkdir(parents=True, exist_ok=True)


def _looks_like_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith((".", "/", "~", "$")):
        return True
    return Path(os.path.expandvars(os.path.expanduser(value))).exists()
