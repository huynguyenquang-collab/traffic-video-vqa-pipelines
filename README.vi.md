# Traffic Video VQA Pipelines

Repo này chứa bản chuyển đổi sang Python module từ hai notebook Kaggle:

1. `notebooks/yolo26.ipynb`: train/tune YOLO để detect biển báo giao thông.
2. `notebooks/vqa.ipynb` hoặc `notebooks/original_kaggle_notebook.ipynb`: dùng YOLO đã train để chạy pipeline Video VQA.

Code sau khi tách notebook nằm trong `src/refactor`. Thứ tự chạy đúng là YOLO trước, VQA sau.

## Entry Point Chính

```bash
# 1. Convert dataset, augment data, và tune YOLO26 bằng Optuna.
PYTHONPATH=src python3 -m refactor.run_yolo26

# 2. Chạy pipeline VQA sau khi đã có YOLO model.
PYTHONPATH=src python3 -m refactor.run_vqa
```

Các script vẫn giữ path mặc định của Kaggle giống notebook gốc, nên phù hợp nhất để chạy trong cùng môi trường Kaggle. Nếu chạy local, bạn cần sửa các constant/path trong module tương ứng.

## Cấu Trúc `src/refactor`

Phần chuyển đổi từ notebook YOLO:

- `src/refactor/run_yolo26.py`: chạy flow YOLO theo đúng thứ tự notebook.
- `src/refactor/yolo26_config.py`: gom path và tham số mặc định của YOLO.
- `src/refactor/yolo26_dataset.py`: convert dataset biển báo Việt Nam sang format YOLO, augment data, ghi `dataset.yaml` và `data.yaml`.
- `src/refactor/yolo26_train.py`: tune YOLO bằng Optuna với `YOLO("yolo26m.pt")`, lưu best hyperparameters.

Phần chuyển đổi từ notebook VQA:

- `src/refactor/run_vqa.py`: chạy flow VQA theo đúng thứ tự notebook.
- `src/refactor/kaggle_environment.py`: helper optional để install package kiểu notebook Kaggle.
- `src/refactor/translation_utils.py`: dịch Việt sang Anh và cache kết quả.
- `src/refactor/qwen_model.py`: load Qwen/Unsloth, setup LoRA, train và save model.
- `src/refactor/videoqa_preprocess.py`: split train/test theo video, thêm prefix path, extract frame, convert sang Qwen message format.
- `src/refactor/traffic_sign_vqa.py`: normalize và dịch traffic-sign VQA JSONL.
- `src/refactor/mixed_training.py`: trộn video QA với traffic-sign QA.
- `src/refactor/vlm_inference.py`: hàm inference VLM, prompt chung, extract nhãn A/B/C/D.
- `src/refactor/rag_retrieval.py`: retrieval tiếng Anh bằng BM25 + SBERT + CLIP.
- `src/refactor/micro_hint_pipeline.py`: chọn key frame, crop biển báo bằng YOLO, Micro-Hint RAG, ghi submission.
- `src/refactor/pipeline_comparison.py`: helper kiểm tra một sample giữa no-RAG và full-RAG.

## Input Kaggle Mặc Định

Bản refactor giữ các path mặc định từ notebook:

- YOLO source dataset: `/kaggle/input/vietnamese-traffic-signs/archive`
- YOLO converted dataset: `/kaggle/working/dataset_yolov11`
- YOLO best hyperparameters: `/kaggle/working/best_hyperparameters.pt`
- VQA train annotations: `/kaggle/input/train3/train/train/train.json`
- Cache/generated VQA assets: `/kaggle/input/datasets/huyqn12/cropped-zalo` và `/kaggle/working`
- YOLO detector dùng trong VQA: `/kaggle/input/besttraffic-real/best (2).pt`
- File submission cuối: `/kaggle/working/submission.csv`

## Cài Đặt

```bash
pip install -r requirements.txt
```

Trong Kaggle, notebook gốc có cell install package lúc runtime. Logic đó được tách ra ở:

```bash
PYTHONPATH=src python3 -m refactor.kaggle_environment
```

## Ghi Chú

- `src/refactor` là bản tách notebook trung thành với flow gốc, vẫn giữ nhiều giả định/path Kaggle.
- Package cũ `src/traffic_video_vqa` và các YAML config vẫn còn trong repo, nhưng README hiện tại mô tả bản refactor mới từ hai notebook.
- Nếu muốn chạy local, hãy sửa các hard-coded Kaggle path trong module config/constants trước khi chạy entry point.

