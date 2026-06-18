# Traffic Video VQA

Modular traffic video question-answering pipelines for the dataset. This repo was extracted from `notebooks/original_kaggle_notebook.ipynb` and reorganized so each module has one clear responsibility.

## What Is Inside

- `configs/default.yaml`: portable config for local paths, caches, models, training, RAG, and inference.
- `configs/kaggle_train3.yaml`: override matching the original Kaggle `train3` dataset layout.
- `src/traffic_video_vqa/data.py`: JSON annotation IO, split-by-video, and path rebasing helpers.
- `src/traffic_video_vqa/preprocess.py`: VI→EN translation, frame extraction, Qwen message dataset conversion, mixed training data creation.
- `src/traffic_video_vqa/training.py`: Qwen-VL LoRA fine-tuning with Unsloth.
- `src/traffic_video_vqa/video.py`: frame sampling, CLIP-selected key frames, YOLO sign crops.
- `src/traffic_video_vqa/rag.py`: BM25 + SBERT + CLIP traffic-sign retrieval.
- `src/traffic_video_vqa/pipelines.py`: no-finetune, no-RAG, Micro-Hint RAG, Gated Micro RAG, and Full RAG inference.
- `notebooks/original_kaggle_notebook.ipynb`: original notebook preserved for auditability.

## Dataset Layout

Expected challenge data:

- Training: about 600 videos and about 1000 QA samples, with `train.json`, videos, answers, and support frames.
- Public test: about 300 videos and about 500 QA samples, with questions and videos.
- Private test: about 300 videos and about 500 QA samples, with questions and videos.

Each annotation item is expected to contain:

- `video_path`
- `question`
- `choices`
- `answer` for training/eval data
- `support_frames` when available

The original notebook used Kaggle path `/kaggle/input/train3/train/train/train.json`. In this repo, that path is represented by config:

```yaml
paths:
  data_root: /kaggle/input/train3/train
  train_annotations: train/train.json
  video_root: .
```

For local use, change only `paths.data_root`, `paths.train_annotations`, and `paths.video_root`.

## Portable Path Handling

The notebook stored absolute paths such as `/kaggle/input/...` inside generated JSON files. That is hard to reproduce outside Kaggle, so this repo centralizes path logic:

- Annotation JSON is loaded through `resolve_annotation_video_paths`.
- Relative `video_path` values are resolved against `video_root`, then `data_root`, then the annotation file folder.
- Generated Qwen message datasets can rebase image paths with `rebase_qwen_message_images`.
- Config controls all generated files: splits, translated JSON, frames, crops, cache files, models, and submission CSV.

Recommended local workflow:

```bash
cp configs/default.yaml configs/local.yaml
# edit configs/local.yaml
pip install -e ".[finetune]"
```

## Pipelines

Current best notebook result on the held-out 326-item split:

- `no_rag`: 0.7393, 241/326.
- `micro_hint_rag`: 0.7485, 244/326.
- `gated_micro_rag`: 0.7485, 244/326, with 33 second-pass triggers.

### `no_finetune_prompt`

Prompt-only baseline using the base VLM from `models.base_vlm`. It does not load the fine-tuned checkpoint and does not use RAG. It still uses CLIP for key-frame selection and YOLO crops, matching the visual input style of the stronger pipelines.

```bash
PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml infer \
  --pipeline no_finetune_prompt
```

### `no_rag`

Fine-tuned model, CLIP-selected key frames, YOLO crops, and the strongest no-RAG prompt format from the notebook. This is the main ablation baseline.

### `micro_hint_rag`

Best practical pipeline in the notebook. It avoids long retrieved rule blocks and injects only a strict micro hint when YOLO detects a sign class with numeric constraints such as speed, tonnage, or height.

### `gated_micro_rag`

Two-pass variant. First pass answers or returns `UNSURE`; second pass is triggered only when confidence is below `inference.gated_confidence_threshold` and a valid micro hint exists.

### `full_rag`

Traditional reference retrieval with BM25, SBERT dense retrieval, CLIP visual retrieval, CLIP text-to-image retrieval, and reciprocal-rank fusion. This is useful for inspection and ablations, but the compact Micro-Hint strategy was stronger in the notebook run.

## Commands

```bash
# Split train/eval by video, avoiding video leakage.
PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml split

# Translate and convert video QA samples into Qwen message format.
PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml convert-train

# Mix video QA with optional traffic-sign VQA data.
PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml mix-train

# Fine-tune Qwen-VL with LoRA.
PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml train

# Prepare public/private test annotations.
PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml prepare-test \
  --source /path/to/test.json

# Run inference and write submission.csv.
PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml infer \
  --pipeline micro_hint_rag
```

## Reproducing The Original Kaggle Notebook

```bash
pip install -r requirements.txt
bash scripts/run_kaggle_train3.sh configs/kaggle_train3.yaml
```

The original notebook is preserved under `notebooks/` and should be treated as a reference, not the main entry point.
