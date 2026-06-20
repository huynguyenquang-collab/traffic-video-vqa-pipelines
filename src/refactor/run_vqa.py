"""Run the converted original Kaggle notebook after YOLO26 training."""

from __future__ import annotations

from .mixed_training import mix_datasets
from .micro_hint_pipeline import MicroHintPipeline
from .qwen_model import load_qwen_model, setup_lora_trainer, train_and_save
from .rag_retrieval import EnglishRagRetriever
from .traffic_sign_vqa import load_sign_vqa_split, prepare_traffic_vqa_english
from .translation_utils import load_translation_cache
from .videoqa_preprocess import (
    convert_train_to_english,
    load_converted_train,
    load_test_data,
    prefix_train_paths,
    prepare_test_english,
    split_train_test,
)
from .vlm_inference import prepare_for_inference


def main() -> None:
    load_translation_cache()
    model, tokenizer, skip_training = load_qwen_model()

    train_split_path, test_split_path = split_train_test()
    train_prefixed_path = prefix_train_paths(train_split_path)
    converted_train_path, _ = convert_train_to_english(train_prefixed_path)
    video_train, video_eval = load_converted_train(converted_train_path)
    converted_dataset = video_train + video_eval

    traffic_vqa_path = prepare_traffic_vqa_english()
    sign_data, _, _ = load_sign_vqa_split(traffic_vqa_path)
    _, mixed_train, mixed_eval = mix_datasets(converted_dataset, sign_data)

    model, trainer = setup_lora_trainer(model, tokenizer, mixed_train, mixed_eval, skip_training)
    train_and_save(model, tokenizer, trainer, skip_training)
    model = prepare_for_inference(model)

    test_prefixed_path = prepare_test_english(test_split_path)
    test_data = load_test_data(test_prefixed_path)

    rag_retriever = EnglishRagRetriever()
    micro_pipeline = MicroHintPipeline()
    micro_pipeline.evaluate(model, tokenizer, test_data, rag_retriever)


if __name__ == "__main__":
    main()

