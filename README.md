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

Kaggle stable install:

```bash
bash scripts/bootstrap_kaggle.sh
bash scripts/run_kaggle_train3.sh
```

If Kaggle reports a broken `ensurepip`/`wrapt` during virtualenv creation, the
bootstrap script automatically falls back to `/kaggle/working/tvqa_site` and
`scripts/run_kaggle_train3.sh` will use that fallback through
`/kaggle/working/tvqa_env.sh`.
