# Zalo Traffic Video QA

Đây là bản repository hoá và module hoá từ notebook gốc `zaloaichallengfinal-5-5.ipynb`. Mục tiêu là giữ lại các pipeline đang cho kết quả tốt nhất, bỏ hard-code path kiểu Kaggle, và thêm một baseline không fine-tune chỉ dùng prompt.

## Cấu Trúc Repo

- `configs/default.yaml`: config portable cho path local, cache, model, training, RAG và inference.
- `configs/kaggle_train3.yaml`: config override đúng layout Kaggle `train3` trong notebook gốc.
- `src/zalo_traffic_vqa/data.py`: đọc/ghi JSON annotation, split theo video, xử lý và rebase path.
- `src/zalo_traffic_vqa/preprocess.py`: dịch VI→EN, extract frame, convert sang Qwen message dataset, trộn data train.
- `src/zalo_traffic_vqa/training.py`: fine-tune Qwen-VL bằng LoRA/Unsloth.
- `src/zalo_traffic_vqa/video.py`: chọn key frame bằng CLIP, crop biển báo bằng YOLO.
- `src/zalo_traffic_vqa/rag.py`: retrieval biển báo bằng BM25 + SBERT + CLIP + RRF.
- `src/zalo_traffic_vqa/pipelines.py`: các pipeline inference.
- `notebooks/zaloaichallengfinal-5-5.ipynb`: notebook gốc để đối chiếu.

## Dữ Liệu

Dataset challenge thường có:

- Training: khoảng 600 video, khoảng 1000 sample gồm câu hỏi, video, đáp án, support frames.
- Public test: khoảng 300 video, khoảng 500 sample gồm câu hỏi và video.
- Private test: khoảng 300 video, khoảng 500 sample gồm câu hỏi và video.

Mỗi item trong annotation thường có:

- `video_path`
- `question`
- `choices`
- `answer` với train/eval
- `support_frames` nếu có

Notebook gốc dùng path Kaggle:

```text
/kaggle/input/train3/train/train/train.json
```

Trong repo này path đó được đưa vào config:

```yaml
paths:
  data_root: /kaggle/input/train3/train
  train_annotations: train/train.json
  video_root: .
```

Khi chạy local, bạn chỉ cần sửa `paths.data_root`, `paths.train_annotations`, `paths.video_root` trong file config riêng.

## Xử Lý Path Trong JSON

Notebook gốc có nhiều JSON sinh ra chứa path tuyệt đối `/kaggle/input/...`, nên khi chuyển máy sẽ khó reproduce. Repo này xử lý theo hướng:

- Khi load annotation, `video_path` tương đối sẽ được resolve theo `video_root`, sau đó `data_root`, sau đó folder chứa annotation.
- Khi cần lưu split/cache, path output đi qua config.
- Dataset Qwen message có image path bên trong `messages[].content[]`; helper `rebase_qwen_message_images` xử lý việc đổi prefix ảnh.
- Toàn bộ split, translated JSON, frames, crops, embedding cache, model output, submission đều nằm trong `configs/*.yaml`.

Khởi tạo local:

```bash
cp configs/default.yaml configs/local.yaml
# sửa configs/local.yaml cho đúng máy của bạn
pip install -e ".[finetune]"
```

## Các Pipeline

Kết quả notebook hiện tại trên split eval 326 mẫu:

- `no_rag`: 0.7393, 241/326.
- `micro_hint_rag`: 0.7485, 244/326.
- `gated_micro_rag`: 0.7485, 244/326, trigger lượt 2 là 33 lần.

### `no_finetune_prompt`

Baseline mới theo yêu cầu: dùng base VLM từ `models.base_vlm`, không load checkpoint fine-tuned, không dùng RAG. Pipeline vẫn dùng CLIP để chọn key frame và YOLO crop biển báo để input hình ảnh giống style pipeline mạnh hơn.

```bash
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml infer \
  --pipeline no_finetune_prompt
```

### `no_rag`

Dùng model đã fine-tune, key frame chọn bằng CLIP, crop biển báo bằng YOLO, prompt format mạnh nhất của baseline trong notebook. Đây là baseline ablation chính.

### `micro_hint_rag`

Pipeline thực dụng nhất hiện tại. Thay vì nhét một đoạn rule dài từ RAG vào prompt, pipeline chỉ thêm một micro hint rất ngắn khi YOLO phát hiện biển báo có thông tin số khó đọc như tốc độ, tải trọng, chiều cao.

### `gated_micro_rag`

Biến thể hai lượt. Lượt đầu trả lời trực tiếp hoặc `UNSURE`; lượt hai chỉ chạy khi confidence thấp hơn `inference.gated_confidence_threshold` và có micro hint hợp lệ.

### `full_rag`

Pipeline RAG đầy đủ: BM25 + SBERT dense retrieval + CLIP visual retrieval + CLIP text-to-image retrieval + reciprocal-rank fusion. Pipeline này hữu ích để audit/ablation, nhưng trong notebook hiện tại Micro-Hint RAG gọn hơn và cho kết quả tốt hơn.

## Lệnh Chạy

```bash
# Split train/eval theo video, tránh leak cùng video qua hai split.
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml split

# Dịch và convert video QA sang Qwen message format.
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml convert-train

# Trộn video QA với traffic-sign VQA nếu bật trong config.
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml mix-train

# Fine-tune Qwen-VL bằng LoRA.
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml train

# Chuẩn bị public/private test.
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml prepare-test \
  --source /path/to/test.json

# Inference và ghi submission.csv.
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml infer \
  --pipeline micro_hint_rag
```

## Reproduce Trên Kaggle `train3`

```bash
pip install -r requirements.txt
bash scripts/run_kaggle_train3.sh configs/kaggle_train3.yaml
```

Notebook gốc vẫn nằm trong `notebooks/`, nhưng entry point chính của repo là CLI/module trong `src/`.
