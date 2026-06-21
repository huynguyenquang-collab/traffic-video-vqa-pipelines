"""Qwen/Unsloth model loading and fine-tuning helpers."""

from __future__ import annotations

from pathlib import Path


INPUT_BASE = Path("/kaggle/input/datasets/huyqn12/cropped-zalo")
MIXED_MODEL_DIRNAME = "qwen3-vl-8b-mixed-en"
CACHED_MIXED_MODEL = INPUT_BASE / MIXED_MODEL_DIRNAME
BASE_QWEN_MODEL = "/kaggle/input/qwen-3-vl/transformers/8b-instruct/1"


def load_qwen_model(input_base: Path = INPUT_BASE, base_model: str = BASE_QWEN_MODEL):
    from unsloth import FastVisionModel

    cached_mixed_model = input_base / MIXED_MODEL_DIRNAME
    if cached_mixed_model.is_dir():
        print(f"Loading cached {MIXED_MODEL_DIRNAME}; skipping training.")
        model, tokenizer = FastVisionModel.from_pretrained(
            str(cached_mixed_model),
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer, True

    print("Loading base Qwen3 VL 8B model.")
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer, False


def setup_lora_trainer(model, tokenizer, mixed_train, mixed_eval, skip_training: bool):
    if skip_training:
        print("Skipping LoRA config and SFTTrainer setup; mixed model already cached.")
        return model, None

    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
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
        train_dataset=mixed_train,
        eval_dataset=mixed_eval,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            warmup_ratio=0.05,
            learning_rate=2e-4,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=1024,
        ),
    )
    return model, trainer


def train_and_save(model, tokenizer, trainer, skip_training: bool, output_dir: str = MIXED_MODEL_DIRNAME):
    if skip_training:
        print("Skipping trainer.train() and save; mixed model already cached.")
        return None
    trainer_stats = trainer.train()
    print("Saving English mixed-trained model.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer_stats
