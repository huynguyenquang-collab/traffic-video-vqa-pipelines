# Traffic Video VQA

This repository is the modular version of the original `original_kaggle_notebook.ipynb` Kaggle notebook. It keeps the strongest current pipelines from the notebook, moves hard-coded Kaggle paths into YAML config, and adds a no-finetune prompt-only baseline.

Read the full docs:

- [English README](README.en.md)
- [Vietnamese README](README.vi.md)

Quick start:

```bash
pip install -e ".[finetune]"
cp configs/default.yaml configs/local.yaml
# edit configs/local.yaml so paths.data_root points to your local dataset

PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml run-prep
PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml train
PYTHONPATH=src python -m traffic_video_vqa.cli -c configs/local.yaml infer --pipeline micro_hint_rag
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
