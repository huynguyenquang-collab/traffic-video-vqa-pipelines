#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/kaggle_train3.yaml}"

PYTHONPATH=src python -m zalo_traffic_vqa.cli -c "$CONFIG" run-prep
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c "$CONFIG" train
PYTHONPATH=src python -m zalo_traffic_vqa.cli -c "$CONFIG" infer --pipeline micro_hint_rag
