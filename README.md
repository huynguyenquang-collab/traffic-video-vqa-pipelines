# Zalo Traffic Video QA

This repository is the modular version of the original `zaloaichallengfinal-5-5.ipynb` Kaggle notebook. It keeps the strongest current pipelines from the notebook, moves hard-coded Kaggle paths into YAML config, and adds a no-finetune prompt-only baseline.

Read the full docs:

- [English README](README.en.md)
- [Vietnamese README](README.vi.md)

Quick start:

```bash
pip install -e ".[finetune]"
cp configs/default.yaml configs/local.yaml
# edit configs/local.yaml so paths.data_root points to your local dataset

PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml run-prep
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml train
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c configs/local.yaml infer --pipeline micro_hint_rag
```
