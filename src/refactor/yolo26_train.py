"""Optuna tuning and YOLO training from `notebooks/yolo26.ipynb`."""

from __future__ import annotations

import torch
from ultralytics import YOLO
from ultralytics.utils import checks

from .yolo26_config import Yolo26Config


def patch_amp_check() -> None:
    checks.check_amp = lambda *args, **kwargs: True


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def objective_factory(config: Yolo26Config):
    def objective(trial):
        lr0 = trial.suggest_float("lr0", 1e-4, 1e-1, log=True)
        lrf = trial.suggest_float("lrf", 0.01, 1.0)
        momentum = trial.suggest_float("momentum", 0.85, 0.98)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        warmup_epochs = trial.suggest_float("warmup_epochs", 1.0, 5.0)
        warmup_momentum = trial.suggest_float("warmup_momentum", 0.6, 0.95)
        box = trial.suggest_float("box", 3.0, 10.0)
        cls = trial.suggest_float("cls", 0.2, 1.0)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        trial_model = YOLO(config.pretrained_weights)
        results = trial_model.train(
            data=str(config.data_yaml_path),
            epochs=config.tuning_epochs,
            imgsz=config.img_size,
            batch=config.batch,
            workers=config.workers,
            cache=False,
            device=config.device,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            box=box,
            cls=cls,
            dropout=dropout,
            project=config.optuna_project,
            name=f"trial_{trial.number}",
            exist_ok=True,
            verbose=False,
            amp=False,
        )
        return results.results_dict["metrics/mAP50-95(B)"]

    return objective


def tune_yolo(config: Yolo26Config = Yolo26Config()):
    import optuna

    clear_cuda_cache()
    patch_amp_check()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_factory(config), n_trials=config.tuning_trials)

    trial = study.best_trial
    print("Best trial:")
    print(f"Value: {trial.value}")
    print("Best hyperparameters:", trial.params)

    config.best_params_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trial.params, config.best_params_path)
    print(f"Best hyperparameters saved to {config.best_params_path}")
    return study


def main() -> None:
    tune_yolo(Yolo26Config())


if __name__ == "__main__":
    main()

