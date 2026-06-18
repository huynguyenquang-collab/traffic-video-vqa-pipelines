from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from .data import read_json
from .inference import (
    DEFAULT_INSTRUCTION,
    ITERATIVE_FIRST_PASS_INSTRUCTION,
    extract_label,
    inference_with_logits_gating,
    question_prompt,
)
from .rag import ClipFrameScorer, TrafficSignRAG
from .translation import TranslationCache
from .video import select_key_frames


@dataclass
class ItemPrediction:
    item_id: str
    answer: str
    raw_response: str
    confidence: float | None = None
    pipeline: str = ""


class PipelineRunner:
    def __init__(self, cfg: dict[str, Any], model, tokenizer, *, use_rag: bool = True) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.inf_cfg = cfg["inference"]
        self.translator = TranslationCache(cfg["paths"]["translation_cache"])
        self.yolo_model = YOLO(cfg["models"]["yolo_weights"])
        self.class_names_en = {
            int(k): self.translator.translate(str(v)) for k, v in self.yolo_model.names.items()
        }
        self.rag = TrafficSignRAG(cfg).build() if use_rag else None
        self.frame_scorer = self.rag if self.rag is not None else ClipFrameScorer(cfg)

    def run_file(self, test_json: str | Path, *, pipeline: str, output_csv: str | Path) -> dict[str, Any]:
        items = read_json(test_json)["data"]
        history: dict[str, list[str]] = defaultdict(list)
        rows: list[dict[str, str]] = []
        correct = 0
        total_with_gt = 0
        triggered = 0

        for item in tqdm(items, desc=f"Running {pipeline}"):
            prediction, did_trigger = self.predict_item(item, pipeline=pipeline, history=history)
            rows.append({"id": prediction.item_id, "answer": prediction.answer})
            triggered += int(did_trigger)
            gt = extract_label(item.get("answer"))
            if gt:
                total_with_gt += 1
                correct += int(prediction.answer == gt)
            history[item["video_path"]].append(
                f"Q: {item['question']} -> A: {prediction.raw_response.strip()[:80]}"
            )
            torch.cuda.empty_cache()

        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "answer"])
            writer.writeheader()
            writer.writerows(rows)

        accuracy = correct / total_with_gt if total_with_gt else None
        return {
            "pipeline": pipeline,
            "output_csv": str(output_csv),
            "total": len(rows),
            "total_with_gt": total_with_gt,
            "correct": correct,
            "accuracy": accuracy,
            "rag_triggered": triggered,
        }

    def predict_item(
        self,
        item: dict[str, Any],
        *,
        pipeline: str,
        history: dict[str, list[str]],
    ) -> tuple[ItemPrediction, bool]:
        video_path = item["video_path"]
        video_id = Path(video_path).stem
        question = item["question"]
        choices = item.get("choices", [])
        key_frames, crops_info, unique_crops = self._select_visuals(item, video_id)
        images = key_frames + [path for path, _ in crops_info]
        detected_classes = sorted({name for _, name in crops_info})
        history_prompt = self._history_prompt(video_path, history)
        micro_hint = get_micro_ocr_hint(detected_classes)

        if pipeline in {"no_rag", "no_finetune_prompt"}:
            prompt = history_prompt + question_prompt(
                question,
                choices,
                suffix="Answer ONLY with the letter (A/B/C/D).",
            )
            response, confidence = inference_with_logits_gating(
                self.model,
                self.tokenizer,
                images,
                prompt,
                instruction=ITERATIVE_FIRST_PASS_INSTRUCTION,
            )
            return self._prediction(item, response, confidence, pipeline), False

        if pipeline == "micro_hint_rag":
            prompt = history_prompt + question_prompt(
                question + micro_hint,
                choices,
                suffix="Select the correct answer (A, B, C, or D).",
            )
            response, confidence = inference_with_logits_gating(
                self.model,
                self.tokenizer,
                images,
                prompt,
                instruction=DEFAULT_INSTRUCTION,
            )
            return self._prediction(item, response, confidence, pipeline), False

        if pipeline == "gated_micro_rag":
            first_prompt = history_prompt + question_prompt(
                question,
                choices,
                suffix="Answer ONLY with the letter (A/B/C/D) if certain, or 'UNSURE' if not.",
            )
            response, confidence = inference_with_logits_gating(
                self.model,
                self.tokenizer,
                images,
                first_prompt,
                instruction=ITERATIVE_FIRST_PASS_INSTRUCTION,
            )
            triggered = confidence < float(self.inf_cfg["gated_confidence_threshold"]) and bool(micro_hint)
            if triggered:
                second_prompt = history_prompt + question_prompt(
                    question + micro_hint,
                    choices,
                    suffix="Select the correct answer (A, B, C, or D).",
                )
                response, confidence = inference_with_logits_gating(
                    self.model,
                    self.tokenizer,
                    images,
                    second_prompt,
                    instruction=DEFAULT_INSTRUCTION,
                )
            return self._prediction(item, response, confidence, pipeline), triggered

        if pipeline == "full_rag":
            pil_crops = [Image.fromarray(arr) for arr in unique_crops.values()][:8]
            if self.rag is None:
                raise RuntimeError("full_rag requires PipelineRunner(use_rag=True).")
            rules = self.rag.retrieve(
                pil_crops,
                question,
                choices,
                detected_classes=detected_classes,
                top_k=int(self.cfg["rag"]["top_k"]),
            )
            rag_context = "Reference traffic regulations:\n- " + "\n- ".join(rules) + "\n" if rules else ""
            prompt = history_prompt + rag_context + question_prompt(
                question,
                choices,
                suffix="Select the correct answer (A, B, C, or D).",
            )
            response, confidence = inference_with_logits_gating(
                self.model,
                self.tokenizer,
                images,
                prompt,
                instruction=DEFAULT_INSTRUCTION,
            )
            return self._prediction(item, response, confidence, pipeline), bool(rules)

        raise ValueError(f"Unknown pipeline: {pipeline}")

    def _select_visuals(self, item: dict[str, Any], video_id: str):
        return select_key_frames(
            item["video_path"],
            item["question"],
            video_id,
            yolo_model=self.yolo_model,
            class_names_en=self.class_names_en,
            score_frame=self.frame_scorer.score_frame,
            fullframe_save_dir=self.cfg["paths"]["key_frames_dir"],
            crop_save_dir=self.cfg["paths"]["crops_dir"],
            frames_log_path=self.cfg["paths"]["frames_log"],
            num_candidates=int(self.inf_cfg["num_candidates"]),
            max_key_frames=int(self.inf_cfg["max_key_frames"]),
            max_size=int(self.inf_cfg["vlm_max_size"]),
            yolo_conf=float(self.inf_cfg["yolo_conf"]),
            yolo_iou=float(self.inf_cfg["yolo_iou"]),
        )

    def _history_prompt(self, video_path: str, history: dict[str, list[str]]) -> str:
        previous = history.get(video_path, [])[-int(self.inf_cfg["max_history"]) :]
        if not previous:
            return ""
        return "Previous Q&A from this video:\n" + "\n".join(previous) + "\n\n"

    def _prediction(
        self,
        item: dict[str, Any],
        response: str,
        confidence: float,
        pipeline: str,
    ) -> ItemPrediction:
        return ItemPrediction(
            item_id=str(item["id"]),
            answer=extract_label(response) or "A",
            raw_response=response,
            confidence=confidence,
            pipeline=pipeline,
        )


def get_micro_ocr_hint(detected_classes: list[str]) -> str:
    """Use only sign names that usually contain hard-to-read numeric constraints."""
    strict_keywords = [
        "km/h",
        "speed limit",
        "tấn",
        "ton",
        "meter",
        "m",
        "chiều cao",
        "tải trọng",
    ]
    for sign in detected_classes:
        lowered = sign.lower()
        if any(keyword in lowered for keyword in strict_keywords):
            return f" (Zoom Context: {sign})"
    return ""
