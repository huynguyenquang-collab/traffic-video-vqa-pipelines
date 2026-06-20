"""Shared VLM inference helpers from the original notebook."""

from __future__ import annotations

import re

import torch

DEFAULT_INSTRUCTION = (
    "You are an expert at answering multiple-choice questions about road traffic videos. "
    "You will first see key frames from a video, then cropped images of detected traffic signs (if any). "
    "Based on all the visual content, select the best answer (A, B, C, or D). "
    "Answer with ONLY the letter of the correct choice.\n\n"
)

ITERATIVE_FIRST_PASS_INSTRUCTION = (
    "You are a traffic video analyst. "
    "Examine the key video frames and any detected traffic sign crops provided. "
    "If you are confident in your answer from visual context alone, respond with ONLY the "
    "single answer letter (A, B, C, or D). "
    "If the visual information is insufficient, respond with exactly: UNSURE\n\n"
)


def extract_label(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text[0] in "ABCD":
        return text[0]
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([abcd])\b", text)
    if match:
        return match.group(1).upper()
    return None


def prepare_for_inference(model):
    from unsloth import FastVisionModel

    return FastVisionModel.for_inference(model)


def inference_with_images(
    model,
    tokenizer,
    image_urls: list[str],
    prompt: str,
    max_new_tokens: int = 128,
    instruction: str | None = None,
) -> str:
    instruction = DEFAULT_INSTRUCTION if instruction is None else instruction
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": url} for url in image_urls],
                {"type": "text", "text": instruction + prompt},
            ],
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=None,
    ).to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer_ids = generated_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def inference_with_logits_gating(model, tokenizer, image_urls, prompt, instruction: str | None = None):
    instruction = DEFAULT_INSTRUCTION if instruction is None else instruction
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": url} for url in image_urls],
                {"type": "text", "text": instruction + prompt},
            ],
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            temperature=0.0,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = outputs.sequences[0, inputs["input_ids"].shape[1] :]
    decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    confidence = 1.0
    if hasattr(outputs, "scores") and len(outputs.scores) > 0 and len(gen_tokens) > 0:
        first_token_logits = outputs.scores[0][0]
        probs = torch.nn.functional.softmax(first_token_logits, dim=-1)
        confidence = float(probs[gen_tokens[0]].item())
    return decoded_text, confidence

