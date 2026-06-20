"""Single-sample pipeline comparison from the original notebook."""

from __future__ import annotations

import os
import random

from PIL import Image

from .rag_retrieval import RAG_TOP_K
from .vlm_inference import extract_label, inference_with_images


def run_pipeline_a(model, tokenizer, item: dict, key_frame_paths: list[str]):
    prompt = (
        f"Question: {item['question']}\n"
        f"Choices:\n" + "\n".join(item.get("choices", [])) + "\n"
        f"\nSelect the correct answer (A, B, C, or D)."
    )
    response = inference_with_images(model, tokenizer, key_frame_paths, prompt, max_new_tokens=128)
    return extract_label(response.strip()), response.strip()


def run_pipeline_b(model, tokenizer, item: dict, key_frame_paths: list[str], unique_crops: dict, rag_retriever):
    pil_crops = [Image.fromarray(array) for array in unique_crops.values()][:8]
    rules = rag_retriever.retrieve(pil_crops, item["question"], item.get("choices", []), top_k=RAG_TOP_K)
    rag_ctx = "Reference traffic regulations:\n- " + "\n- ".join(rules) + "\n" if rules else ""
    prompt = (
        f"{rag_ctx}"
        f"Question: {item['question']}\n"
        f"Choices:\n" + "\n".join(item.get("choices", [])) + "\n"
        f"\nSelect the correct answer (A, B, C, or D)."
    )
    response = inference_with_images(model, tokenizer, key_frame_paths, prompt, max_new_tokens=128)
    return extract_label(response.strip()), response.strip(), rules


def compare_single_sample(model, tokenizer, test_data: list[dict], micro_pipeline, rag_retriever, test_item_id=None):
    if test_item_id is not None:
        test_item = next((x for x in test_data if str(x["id"]) == str(test_item_id)), None)
        if test_item is None:
            raise ValueError(f"TEST_ITEM_ID {test_item_id!r} not found")
    else:
        test_item = random.choice(test_data)

    video_id = os.path.splitext(os.path.basename(test_item["video_path"]))[0]
    key_frame_paths, crops, unique_crops = micro_pipeline.select_key_frames(
        test_item["video_path"],
        test_item["question"],
        video_id + "_cmp",
        rag_retriever,
        num_candidates=20,
        max_key_frames=4,
    )
    pred_a, resp_a = run_pipeline_a(model, tokenizer, test_item, key_frame_paths)
    pred_b, resp_b, rules_b = run_pipeline_b(model, tokenizer, test_item, key_frame_paths, unique_crops, rag_retriever)
    true_ans = extract_label(test_item.get("answer", ""))

    print(f"ITEM ID: {test_item['id']}")
    print(f"TRUE: {true_ans}")
    print(f"Pipeline A: {pred_a} ({pred_a == true_ans}) response={resp_a[:300]}")
    print(f"Pipeline B: {pred_b} ({pred_b == true_ans}) response={resp_b[:300]}")
    for i, rule in enumerate(rules_b, 1):
        print(f"[{i}] {rule[:120]}")

    for path in key_frame_paths:
        if os.path.exists(path):
            os.remove(path)
    for crop_path, _ in crops:
        if os.path.exists(crop_path):
            os.remove(crop_path)
    return {"true": true_ans, "pipeline_a": pred_a, "pipeline_b": pred_b}

