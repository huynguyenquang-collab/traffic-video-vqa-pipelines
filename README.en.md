# Traffic Video VQA Pipelines

This repository contains Python refactors of two Kaggle notebooks:

1. `notebooks/yolo26.ipynb`: trains/tunes the YOLO traffic-sign detector.
2. `notebooks/vqa.ipynb` or `notebooks/original_kaggle_notebook.ipynb`: uses the trained YOLO model inside the video VQA pipeline.

The converted code lives in `src/refactor`. The intended running order is YOLO first, then VQA.

## Main Entry Points

```bash
# 1. Convert dataset, augment it, and run YOLO26 Optuna tuning.
PYTHONPATH=src python3 -m refactor.run_yolo26

# 2. Run the VQA flow after the YOLO model is available.
PYTHONPATH=src python3 -m refactor.run_vqa
```

The scripts preserve the original Kaggle paths by default, so they are mainly intended to run in the same Kaggle environment unless you edit the constants in the modules.

## Refactor Layout

YOLO notebook conversion:

- `src/refactor/run_yolo26.py`: runs the YOLO notebook flow in order.
- `src/refactor/yolo26_config.py`: central YOLO paths and training defaults.
- `src/refactor/yolo26_dataset.py`: converts the Vietnamese traffic-sign dataset into YOLO format and writes `dataset.yaml` / `data.yaml`.
- `src/refactor/yolo26_train.py`: runs Optuna tuning with `YOLO("yolo26m.pt")` and saves best hyperparameters.

VQA notebook conversion:

- `src/refactor/run_vqa.py`: runs the converted VQA notebook flow in order.
- `src/refactor/kaggle_environment.py`: optional notebook-style dependency installation helper.
- `src/refactor/translation_utils.py`: Vietnamese to English translation with disk cache.
- `src/refactor/qwen_model.py`: Qwen/Unsloth loading, LoRA setup, training, and saving.
- `src/refactor/videoqa_preprocess.py`: train/test split, video path prefixing, frame extraction, and Qwen message conversion.
- `src/refactor/traffic_sign_vqa.py`: traffic-sign VQA JSONL normalization and translation.
- `src/refactor/mixed_training.py`: mixes video QA and traffic-sign QA training data.
- `src/refactor/vlm_inference.py`: shared VLM prompting, generation, and answer-label extraction.
- `src/refactor/rag_retrieval.py`: English BM25 + SBERT + CLIP retrieval.
- `src/refactor/micro_hint_pipeline.py`: YOLO crop extraction, key-frame selection, Micro-Hint RAG, and submission writing.
- `src/refactor/pipeline_comparison.py`: single-sample comparison helper for no-RAG vs full-RAG inspection.

## Expected Kaggle Inputs

The refactor keeps the notebook defaults:

- YOLO source dataset: `/kaggle/input/vietnamese-traffic-signs/archive`
- YOLO converted dataset: `/kaggle/working/dataset_yolov11`
- YOLO best hyperparameters: `/kaggle/working/best_hyperparameters.pt`
- VQA train annotations: `/kaggle/input/train3/train/train/train.json`
- Cached/generated VQA assets: `/kaggle/input/datasets/huyqn12/cropped-zalo` and `/kaggle/working`
- Trained traffic-sign detector for VQA: `/kaggle/input/besttraffic-real/best (2).pt`
- Final submission: `/kaggle/working/submission.csv`

## Install

```bash
pip install -r requirements.txt
```

Inside Kaggle, the original notebook also installed some packages at runtime. That logic is available as:

```bash
PYTHONPATH=src python3 -m refactor.kaggle_environment
```

## Notes

- `src/refactor` is a faithful notebook split: it keeps the original Kaggle assumptions and execution order.
- The older `src/traffic_video_vqa` package and YAML configs are still present, but these README files document the newer notebook-to-`src/refactor` conversion.
- If you run locally, update the hard-coded Kaggle paths in the relevant config/constants modules before executing the entry points.

