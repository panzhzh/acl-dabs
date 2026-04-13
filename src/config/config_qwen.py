#!/usr/bin/env python3
# src/config/config_qwen.py
# -*- coding: utf-8 -*-

"""
Qwen-specific training profile.

This file is intentionally separate from src/config/config.py so that
default experiments stay unchanged.
"""

from pathlib import Path
import os


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 1
    return (int(a) + int(b) - 1) // int(b)


# ---------- 1. Paths ----------
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------- 2. Dataset ----------
DATASET_CHOICE = _env_int("DATASET_CHOICE", 3)

# ---------- 3. Pretrained Model ----------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
# IMPORTANT: Qwen 7B+ classifiers are too large to keep in fp32 on a 32GB GPU once
# you include additional DABS modules + optimizer states. Load the backbone in
# half precision by default to avoid spilling into shared (system) memory.
BACKBONE_TORCH_DTYPE = os.environ.get("BACKBONE_TORCH_DTYPE", "bfloat16").strip().lower()

# ---------- 4. Training ----------
FINETUNE_MODE = os.environ.get("FINETUNE_MODE", "freeze_backbone").strip().lower()

# Checkpoint policy for long Qwen runs.
# - Default keeps old behavior (no checkpoint files) to save disk.
# - run_qwen_training_modes.py now enables these flags so best checkpoints are persisted.
QWEN_SAVE_CHECKPOINTS = _env_bool("QWEN_SAVE_CHECKPOINTS", False)
QWEN_LOAD_BEST_MODEL = _env_bool("QWEN_LOAD_BEST_MODEL", QWEN_SAVE_CHECKPOINTS)
if QWEN_LOAD_BEST_MODEL and not QWEN_SAVE_CHECKPOINTS:
    QWEN_SAVE_CHECKPOINTS = True

NUM_LABELS = 3
NUM_EPOCHS = _env_int("NUM_EPOCHS", 8)
TRAIN_BATCH_SIZE = max(1, _env_int("TRAIN_BATCH_SIZE", 16))
EVAL_BATCH_SIZE = max(1, _env_int("EVAL_BATCH_SIZE", TRAIN_BATCH_SIZE))
# Keep effective batch close to previous default (32) while allowing larger per-device batch.
TARGET_EFFECTIVE_BATCH = max(1, _env_int("TARGET_EFFECTIVE_BATCH", 32))
_default_grad_acc = max(1, _ceil_div(TARGET_EFFECTIVE_BATCH, TRAIN_BATCH_SIZE))
WARMUP_RATIO = _env_float("WARMUP_RATIO", 0.03)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.01)
LOGGING_STEPS = _env_int("LOGGING_STEPS", 20)
MAX_GRAD_NORM = _env_float("MAX_GRAD_NORM", 0.5)
FP16 = _env_bool("FP16", False)
BF16 = _env_bool("BF16", True)
if FP16 and BF16:
    # Prefer BF16 by default on modern GPUs for better stability.
    FP16 = False

_mode_default_lr = {
    "freeze_backbone": 5e-4,  # head-only updates need larger LR
    "qlora": 2e-4,            # standard QLoRA scale
}
LEARNING_RATE = _env_float("LEARNING_RATE", _mode_default_lr.get(FINETUNE_MODE, 2e-5))
GRAD_ACC_STEPS = max(1, _env_int("GRAD_ACC_STEPS", _default_grad_acc))
EFFECTIVE_BATCH_SIZE = TRAIN_BATCH_SIZE * GRAD_ACC_STEPS

USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None

model_suffix = MODEL_NAME.split("/")[-1] if "/" in MODEL_NAME else MODEL_NAME
_base_output_dir = OUTPUT_DIR / f"results_{model_suffix}"
_base_output_dir.mkdir(parents=True, exist_ok=True)
_base_output_dir = str(_base_output_dir)

TRAINER_ARGS = dict(
    output_dir=_base_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch" if QWEN_SAVE_CHECKPOINTS else "no",
    load_best_model_at_end=QWEN_LOAD_BEST_MODEL,
    metric_for_best_model="eval_mf1",
    greater_is_better=True,
    save_total_limit=1,
    save_only_model=True,
    fp16=FP16,
    bf16=BF16,
    warmup_ratio=WARMUP_RATIO,
    learning_rate=LEARNING_RATE,
    max_grad_norm=MAX_GRAD_NORM,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    logging_steps=LOGGING_STEPS,
    disable_tqdm=False,
    log_level="error",
    logging_strategy="no",
)

# ---------- 5. Misc ----------
LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
USE_WANDB = False

# ---------- 6. Early Stopping ----------
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = _env_int("EARLY_STOPPING_PATIENCE", 5)
EARLY_STOPPING_THRESHOLD = _env_float("EARLY_STOPPING_THRESHOLD", 0.0)

# ---------- 7. Output Control ----------
VERBOSE_TRAINING = True

# ---------- 8. Data Augmentation ----------
USE_DATA_AUG = False
DATA_AUG_SUFFIX = ""
