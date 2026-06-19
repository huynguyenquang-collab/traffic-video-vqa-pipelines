from __future__ import annotations

import argparse
from pathlib import Path

from .compat import configure_runtime
from .config import ensure_dirs, load_config
from .preprocess import (
    build_mixed_training_dataset,
    convert_video_qa_to_qwen_dataset,
    create_splits,
    prepare_test_annotations,
)
from .training import load_vlm, set_for_inference, train_lora


configure_runtime()

PIPELINES = ("no_finetune_prompt", "no_rag", "micro_hint_rag", "gated_micro_rag", "full_rag")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic video QA pipelines")
    parser.add_argument("-c", "--config", default=None, help="Path to YAML config override.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("split", help="Split training annotations by video.")
    sub.add_parser("convert-train", help="Translate train split and convert to Qwen message format.")
    sub.add_parser("mix-train", help="Build mixed video-QA + traffic-sign-QA training dataset.")
    sub.add_parser("train", help="Fine-tune the configured VLM with LoRA.")

    prep = sub.add_parser("prepare-test", help="Resolve/translate a test annotation JSON.")
    prep.add_argument("--source", default=None, help="Override test annotation path.")

    infer = sub.add_parser("infer", help="Run one inference pipeline and write a submission CSV.")
    infer.add_argument("--test-json", default=None, help="Prepared English test JSON. Defaults to paths.test_en.")
    infer.add_argument("--pipeline", choices=PIPELINES, default=None)
    infer.add_argument("--output", default=None, help="Submission CSV path.")

    sub.add_parser("run-prep", help="Run split, train conversion, mixed dataset, and test preparation.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.command == "split":
        train, eval_ = create_splits(cfg)
        print(f"train={train['__count__']} eval={eval_['__count__']}")
        return

    if args.command == "convert-train":
        rows = convert_video_qa_to_qwen_dataset(cfg)
        print(f"converted={len(rows)} path={cfg['paths']['converted_train']}")
        return

    if args.command == "mix-train":
        rows = build_mixed_training_dataset(cfg)
        print(f"mixed={len(rows)} path={cfg['paths']['mixed_train']}")
        return

    if args.command == "train":
        train_lora(cfg)
        return

    if args.command == "prepare-test":
        prepared = prepare_test_annotations(cfg, args.source)
        print(f"prepared={len(prepared['data'])} path={cfg['paths']['test_en']}")
        return

    if args.command == "run-prep":
        create_splits(cfg)
        convert_video_qa_to_qwen_dataset(cfg)
        build_mixed_training_dataset(cfg)
        prepare_test_annotations(cfg)
        return

    if args.command == "infer":
        pipeline = args.pipeline or cfg["inference"]["pipeline"]
        prefer_cached = pipeline != "no_finetune_prompt"
        model, tokenizer, _ = load_vlm(cfg, prefer_cached=prefer_cached)
        model = set_for_inference(model)
        from .pipelines import PipelineRunner

        runner = PipelineRunner(cfg, model, tokenizer, use_rag=pipeline not in {"no_rag", "no_finetune_prompt"})
        result = runner.run_file(
            args.test_json or cfg["paths"]["test_en"],
            pipeline=pipeline,
            output_csv=args.output or cfg["paths"]["submission"],
        )
        print(result)
        return

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
