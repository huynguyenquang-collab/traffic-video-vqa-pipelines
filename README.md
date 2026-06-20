# Traffic Video VQA Pipelines

This repo contains Python refactors of two Kaggle notebooks:

- `notebooks/yolo26.ipynb`: train/tune the YOLO traffic-sign detector.
- `notebooks/vqa.ipynb` / `notebooks/original_kaggle_notebook.ipynb`: run the video VQA pipeline using the trained YOLO model.

Read the full docs:

- [English README](README.en.md)
- [Vietnamese README](README.vi.md)

Quick start in the original Kaggle-style environment:

```bash
pip install -r requirements.txt

# Run YOLO first.
PYTHONPATH=src python3 -m refactor.run_yolo26

# Then run VQA.
PYTHONPATH=src python3 -m refactor.run_vqa
```

Converted modules are in `src/refactor`. The older `src/traffic_video_vqa` package is still in the repo, but the current README documents the newer notebook split.

