"""Micro-hint YOLO/RAG inference pipeline from the original notebook."""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from .translation_utils import save_translation_cache, translate_vi_to_en
from .vlm_inference import (
    ITERATIVE_FIRST_PASS_INSTRUCTION,
    extract_label,
    inference_with_logits_gating,
)

MAX_HISTORY = 3
VLM_MAX_SIZE = 1280


def sanitize_filename(value: str) -> str:
    return re.sub(r'[./\\:*?"<>|]', "_", value)


def get_zoom_context_hint(detected_classes: list[str]) -> str:
    if not detected_classes:
        return ""
    strict_keywords = ["km/h", "speed limit", "tan", "ton", "meter", "m", "chieu cao", "tai trong"]
    for sign in detected_classes:
        if any(keyword in sign.lower() for keyword in strict_keywords):
            return f" (Zoom Context: {sign})"
    return ""


class MicroHintPipeline:
    def __init__(
        self,
        yolo_weights: str = "/kaggle/input/besttraffic-real/best (2).pt",
        crop_save_dir: Path = Path("/kaggle/working/cropped_signs"),
        fullframe_save_dir: Path = Path("/kaggle/working/full_frames"),
    ) -> None:
        self.yolo_model = YOLO(yolo_weights)
        self.class_names_en = {
            k: translate_vi_to_en(str(v)) for k, v in self.yolo_model.names.items()
        }
        save_translation_cache()
        self.crop_save_dir = crop_save_dir
        self.fullframe_save_dir = fullframe_save_dir
        self.crop_save_dir.mkdir(parents=True, exist_ok=True)
        self.fullframe_save_dir.mkdir(parents=True, exist_ok=True)

    def select_key_frames(
        self,
        video_path: str,
        question_text: str,
        video_id: str,
        rag_retriever,
        num_candidates: int = 20,
        max_key_frames: int = 5,
        max_size: int = VLM_MAX_SIZE,
    ):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return [], [], {}

        candidate_indices = np.linspace(0, total_frames - 1, num_candidates, dtype=int).tolist()
        candidates = []
        for idx in candidate_indices:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
                continue
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            h, w = frame.shape[:2]
            scale = min(1.0, max_size / max(h, w))
            frame_resized = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            frame_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            candidates.append((idx, frame_resized, frame_pil))
        cap.release()
        if not candidates:
            return [], [], {}

        clip_scores, yolo_results = [], []
        for _, frame_bgr, frame_pil in candidates:
            clip_scores.append(rag_retriever.score_frame_with_clip(frame_pil, question_text))
            yolo_results.append(self.yolo_model(frame_bgr, conf=0.55, iou=0.45, verbose=False))

        sorted_idx = np.argsort(clip_scores)[::-1]
        selected, selected_fi = [], []
        for si in sorted_idx:
            if len(selected) >= max_key_frames:
                break
            frame_idx = candidates[si][0]
            if any(abs(frame_idx - prev) < max(total_frames // 8, 5) for prev in selected_fi):
                continue
            selected.append(si)
            selected_fi.append(frame_idx)
        for si in sorted_idx:
            if si not in selected and len(selected) < max_key_frames:
                selected.append(si)
        selected.sort(key=lambda si: candidates[si][0])

        key_frame_paths, all_crops_info, unique_crops = [], [], {}
        seen_classes = set()
        for si in selected:
            frame_idx, frame_bgr, _ = candidates[si]
            frame_path = self.fullframe_save_dir / f"{video_id}_key_{frame_idx}.jpg"
            cv2.imwrite(str(frame_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            key_frame_paths.append(str(frame_path))

            boxes = yolo_results[si][0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            h_f, w_f = frame_bgr.shape[:2]
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = self.class_names_en.get(cls_id, f"class_{cls_id}")
                if class_name in seen_classes and sum(1 for _, c in all_crops_info if c == class_name) >= 2:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                margin = 30
                crop = frame_bgr[
                    max(0, y1 - margin) : min(h_f, y2 + margin),
                    max(0, x1 - margin) : min(w_f, x2 + margin),
                ]
                if crop.size == 0:
                    continue
                crop_path = self.crop_save_dir / f"{video_id}_{frame_idx}_{sanitize_filename(class_name)}.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                all_crops_info.append((str(crop_path), class_name))
                seen_classes.add(class_name)
                unique_crops.setdefault(class_name, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return key_frame_paths, all_crops_info, unique_crops

    def evaluate(self, model, tokenizer, test_data: list[dict], rag_retriever, submission_path: Path = Path("/kaggle/working/submission.csv")):
        results_a, results_b, results_c = [], [], []
        history_dict = defaultdict(list)
        correct_a = correct_b = correct_c = total = rag_triggered = 0

        for item in tqdm(test_data, desc="Evaluating Micro-Hint RAG"):
            video_path = item["video_path"]
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            question = item["question"]
            choices = item.get("choices", [])
            gt_answer = extract_label(item.get("answer", ""))
            key_frame_paths, crops_info, _ = self.select_key_frames(video_path, question, video_id, rag_retriever)
            if not key_frame_paths:
                continue
            detected_classes = list({class_name for _, class_name in crops_info})
            all_vlm_images = key_frame_paths + [crop_path for crop_path, _ in crops_info]
            history_prompt = ""
            video_history = history_dict.get(video_path, [])[-MAX_HISTORY:]
            if video_history:
                history_prompt = "Previous Q&A from this video:\n" + "\n".join(video_history) + "\n\n"

            try:
                prompt_c = f"{history_prompt}Question: {question}\nChoices:\n" + "\n".join(choices) + "\n\nSelect the correct answer (A, B, C, or D)."
                resp_c, _ = inference_with_logits_gating(model, tokenizer, all_vlm_images, prompt_c)
                pred_c = extract_label(resp_c) or "A"

                zoom_context_hint = get_zoom_context_hint(detected_classes)
                prompt_a = f"{history_prompt}Question: {question}{zoom_context_hint}\nChoices:\n" + "\n".join(choices) + "\n\nSelect the correct answer (A, B, C, or D)."
                resp_a, _ = inference_with_logits_gating(model, tokenizer, all_vlm_images, prompt_a)
                pred_a = extract_label(resp_a) or "A"

                prompt_b1 = f"{history_prompt}Question: {question}\nChoices:\n" + "\n".join(choices) + "\n\nAnswer ONLY with the letter (A/B/C/D) if certain, or 'UNSURE' if not."
                resp_b1, confidence = inference_with_logits_gating(
                    model,
                    tokenizer,
                    all_vlm_images,
                    prompt_b1,
                    instruction=ITERATIVE_FIRST_PASS_INSTRUCTION,
                )
                pred_b = extract_label(resp_b1) or "A"
                if confidence < 0.85 and zoom_context_hint:
                    rag_triggered += 1
                    resp_b2, _ = inference_with_logits_gating(model, tokenizer, all_vlm_images, prompt_a)
                    pred_b = extract_label(resp_b2) or "A"

                correct_c += pred_c == gt_answer
                correct_a += pred_a == gt_answer
                correct_b += pred_b == gt_answer
                total += 1
                results_a.append({"id": item["id"], "answer": pred_a})
                results_b.append({"id": item["id"], "answer": pred_b})
                results_c.append({"id": item["id"], "answer": pred_c})
                history_dict[video_path].append(f"Q: {question} -> A: {resp_c.strip()[:80]}")

                if total % 10 == 0:
                    print(
                        f"[{total}] C(Base): {correct_c / total:.4f} | "
                        f"A(Micro-RAG): {correct_a / total:.4f} | "
                        f"B(Gated Micro): {correct_b / total:.4f} [Triggers: {rag_triggered}]"
                    )
            except Exception as exc:
                print(f"[Error] {video_id}: {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if total == 0:
            raise RuntimeError("No test items were evaluated.")
        print(f"Pipeline C (NO RAG Baseline): {correct_c / total:.4f} ({correct_c}/{total})")
        print(f"Pipeline A (Micro-Hint RAG): {correct_a / total:.4f} ({correct_a}/{total})")
        print(f"Pipeline B (Gated Micro RAG): {correct_b / total:.4f} ({correct_b}/{total}) [Triggers: {rag_triggered}]")

        best_results = results_a if correct_a >= max(correct_c, correct_b) else (results_b if correct_b >= correct_c else results_c)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        with submission_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "answer"])
            writer.writeheader()
            writer.writerows(best_results)
        return submission_path
