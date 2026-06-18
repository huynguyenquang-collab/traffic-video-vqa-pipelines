from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import read_json, split_list


def load_vlm(cfg: dict[str, Any], *, prefer_cached: bool = True):
    """Load the cached fine-tuned model if present, otherwise the configured base VLM."""
    from unsloth import FastVisionModel

    models = cfg["models"]
    train_cfg = cfg["training"]
    cached = Path(models["cached_finetuned_vlm"])
    model_path = str(cached if prefer_cached and cached.is_dir() else models["base_vlm"])
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=bool(train_cfg["load_in_4bit"]),
        use_gradient_checkpointing=train_cfg["gradient_checkpointing"],
    )
    return model, tokenizer, cached.is_dir()


def train_lora(cfg: dict[str, Any]) -> None:
    """Fine-tune Qwen-VL with the mixed English video/sign dataset."""
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    model, tokenizer, loaded_cached = load_vlm(cfg, prefer_cached=bool(cfg["training"]["skip_if_cached"]))
    if loaded_cached and cfg["training"]["skip_if_cached"]:
        print("Cached fine-tuned model found; skipping training.")
        return

    train_cfg = cfg["training"]
    dataset = read_json(cfg["paths"]["mixed_train"])
    train_dataset, eval_dataset = split_list(dataset, eval_ratio=cfg["data"]["eval_ratio"], seed=cfg["seed"])

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            per_device_train_batch_size=int(train_cfg["batch_size"]),
            gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
            num_train_epochs=int(train_cfg["epochs"]),
            warmup_ratio=float(train_cfg["warmup_ratio"]),
            learning_rate=float(train_cfg["learning_rate"]),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_8bit",
            weight_decay=float(train_cfg["weight_decay"]),
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=cfg["paths"]["output_dir"],
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=int(train_cfg["max_length"]),
        ),
    )
    trainer.train()
    out = cfg["models"]["output_finetuned_vlm"]
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)


def set_for_inference(model):
    from unsloth import FastVisionModel

    return FastVisionModel.for_inference(model)
