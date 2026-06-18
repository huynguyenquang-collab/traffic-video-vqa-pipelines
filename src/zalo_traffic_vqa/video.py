from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image


def sanitize_filename(value: str) -> str:
    return re.sub(r'[./\\:*?"<>|]', "_", value)


def extract_sample_frames(
    video_path: str | Path,
    save_dir: str | Path,
    id_prefix: str,
    *,
    max_frames: int = 5,
    max_size: int = 2000,
    jpeg_quality: int = 90,
) -> list[str]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / max_frames) for i in range(max_frames)]
    paths: list[str] = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = resize_bgr(frame, max_size=max_size)
        out = save_dir / f"{sanitize_filename(id_prefix)}_{i:04d}.jpg"
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        paths.append(str(out))
    cap.release()
    return paths


def resize_bgr(frame, *, max_size: int):
    h, w = frame.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def select_key_frames(
    video_path: str | Path,
    question_text: str,
    video_id: str,
    *,
    yolo_model,
    class_names_en: dict[int, str],
    score_frame: Callable[[Image.Image, str], float],
    fullframe_save_dir: str | Path,
    crop_save_dir: str | Path,
    frames_log_path: str | Path | None = None,
    num_candidates: int = 20,
    max_key_frames: int = 5,
    max_size: int = 1280,
    yolo_conf: float = 0.55,
    yolo_iou: float = 0.45,
) -> tuple[list[str], list[tuple[str, str]], dict[str, np.ndarray]]:
    """Select CLIP-scored key frames and traffic-sign crops."""
    fullframe_save_dir = Path(fullframe_save_dir)
    crop_save_dir = Path(crop_save_dir)
    fullframe_save_dir.mkdir(parents=True, exist_ok=True)
    crop_save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [], [], {}

    candidates = []
    for idx in np.linspace(0, total - 1, num_candidates, dtype=int).tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            continue
        frame = resize_bgr(frame, max_size=max_size)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        candidates.append((idx, frame, Image.fromarray(rgb)))
    cap.release()
    if not candidates:
        return [], [], {}

    clip_scores: list[float] = []
    yolo_results = []
    for _, frame_bgr, frame_pil in candidates:
        clip_scores.append(score_frame(frame_pil, question_text))
        yolo_results.append(yolo_model(frame_bgr, conf=yolo_conf, iou=yolo_iou, verbose=False))

    selected = _dedupe_temporal_indices(np.argsort(clip_scores)[::-1], candidates, total, max_key_frames)
    key_frame_paths: list[str] = []
    crops_info: list[tuple[str, str]] = []
    unique_crops: dict[str, np.ndarray] = {}
    seen_classes: set[str] = set()

    for si in selected:
        frame_idx, frame_bgr, _ = candidates[si]
        frame_path = fullframe_save_dir / f"{sanitize_filename(video_id)}_key_{frame_idx}.jpg"
        cv2.imwrite(str(frame_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        key_frame_paths.append(str(frame_path))

        boxes = yolo_results[si][0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        h, w = frame_bgr.shape[:2]
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = class_names_en.get(cls_id, f"class_{cls_id}")
            if class_name in seen_classes and sum(1 for _, c in crops_info if c == class_name) >= 2:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            margin = 30
            crop = frame_bgr[max(0, y1 - margin) : min(h, y2 + margin), max(0, x1 - margin) : min(w, x2 + margin)]
            if crop.size == 0:
                continue
            crop_path = crop_save_dir / f"{sanitize_filename(video_id)}_{frame_idx}_{sanitize_filename(class_name)}.jpg"
            cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            crops_info.append((str(crop_path), class_name))
            seen_classes.add(class_name)
            unique_crops.setdefault(class_name, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    if frames_log_path:
        _append_frame_log(frames_log_path, video_path, question_text, key_frame_paths, crops_info)
    return key_frame_paths, crops_info, unique_crops


def _dedupe_temporal_indices(sorted_indices, candidates, total_frames: int, max_key_frames: int) -> list[int]:
    selected: list[int] = []
    selected_frame_indices: list[int] = []
    for si in sorted_indices:
        if len(selected) >= max_key_frames:
            break
        frame_idx = candidates[int(si)][0]
        if any(abs(frame_idx - prev) < max(total_frames // 8, 5) for prev in selected_frame_indices):
            continue
        selected.append(int(si))
        selected_frame_indices.append(frame_idx)
    for si in sorted_indices:
        if len(selected) >= max_key_frames:
            break
        if int(si) not in selected:
            selected.append(int(si))
    selected.sort(key=lambda i: candidates[i][0])
    return selected


def _append_frame_log(
    path: str | Path,
    video_path: str | Path,
    question: str,
    key_frames: list[str],
    crops: list[tuple[str, str]],
) -> None:
    row = {
        "video_path": str(video_path),
        "question": question,
        "key_frames": key_frames,
        "crops": [{"path": p, "class_name": c} for p, c in crops],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
