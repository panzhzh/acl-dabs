#!/usr/bin/env python3
# src/config/config.py
# -*- coding: utf-8 -*-

from importlib import import_module
from pathlib import Path

# ---------- 1. Paths ----------
BASE_DIR   = Path(__file__).parent.parent.parent.absolute()  # Go up 3 levels: config -> src -> project_root
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"      # Unified output root.

# ---------- 2. Dataset ----------
import os


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)


DATASET_CHOICE = int(os.environ.get('DATASET_CHOICE', 3))


def _load_dataset_overrides(dataset_choice: int):
    from .dataset_configs import DATASET_CONFIG_MODULES

    module_path = DATASET_CONFIG_MODULES.get(dataset_choice)
    if not module_path:
        return "default", {}

    dataset_module = import_module(module_path)
    config_name = getattr(dataset_module, "CONFIG_NAME", f"dataset_{dataset_choice}")
    overrides = getattr(dataset_module, "OVERRIDES", {})
    if not isinstance(overrides, dict):
        raise TypeError(
            f"{module_path}.OVERRIDES must be a dict, got {type(overrides).__name__}"
        )
    return config_name, overrides

# ---------- 3. Pretrained Model ----------
MODEL_NAME = os.environ.get('MODEL_NAME', "microsoft/deberta-v3-base")

# ---------- 4. Training ----------
NUM_LABELS        = 3
NUM_EPOCHS           = 30
TRAIN_BATCH_SIZE     = 32
EVAL_BATCH_SIZE   = 64
WARMUP_RATIO         = 0.06
WEIGHT_DECAY      = 0.01
LOGGING_STEPS     = 100
MAX_GRAD_NORM        = 1.0
LEARNING_RATE        = 2e-5
FP16                 = _env_bool("FP16", False)
BF16                 = _env_bool("BF16", True)
if FP16 and BF16:
    FP16 = False
USE_FOCAL_LOSS = False         
FOCAL_GAMMA    = 2.0           # Focusing factor gamma.
FOCAL_ALPHA    = None          # Optional class-wise weighting.

_DATASET_TUNABLE_KEYS = {
    "NUM_EPOCHS",
    "TRAIN_BATCH_SIZE",
    "EVAL_BATCH_SIZE",
    "WARMUP_RATIO",
    "WEIGHT_DECAY",
    "MAX_GRAD_NORM",
    "LEARNING_RATE",
    "EARLY_STOPPING_PATIENCE",
}

ACTIVE_DATASET_CONFIG, _DATASET_OVERRIDES = _load_dataset_overrides(DATASET_CHOICE)
_unknown_override_keys = set(_DATASET_OVERRIDES) - _DATASET_TUNABLE_KEYS
if _unknown_override_keys:
    raise KeyError(
        f"Unsupported dataset override keys for {ACTIVE_DATASET_CONFIG}: "
        f"{sorted(_unknown_override_keys)}"
    )
for _key, _value in _DATASET_OVERRIDES.items():
    globals()[_key] = _value

NUM_EPOCHS = _env_int("NUM_EPOCHS", NUM_EPOCHS)
TRAIN_BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", TRAIN_BATCH_SIZE)
EVAL_BATCH_SIZE = _env_int("EVAL_BATCH_SIZE", EVAL_BATCH_SIZE)
WARMUP_RATIO = _env_float("WARMUP_RATIO", WARMUP_RATIO)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", WEIGHT_DECAY)
MAX_GRAD_NORM = _env_float("MAX_GRAD_NORM", MAX_GRAD_NORM)
LEARNING_RATE = _env_float("LEARNING_RATE", LEARNING_RATE)

# Dynamically build the output directory (set in trainer.py).
# Include model name to avoid conflicts when running multiple models simultaneously
model_suffix = MODEL_NAME.split('/')[-1] if '/' in MODEL_NAME else MODEL_NAME
_base_output_dir = OUTPUT_DIR / f"results_{model_suffix}"

_base_output_dir = str(_base_output_dir)

TRAINER_ARGS = dict(
    output_dir      = _base_output_dir,  # Updated during training to the final metric-tagged path.
    eval_strategy   = "epoch",
    save_strategy         = "epoch",
    load_best_model_at_end= True,
    metric_for_best_model = "eval_mf1",
    greater_is_better     = True,
    save_total_limit      = 1,  # Only keep the best checkpoint to save disk space
    save_only_model       = True,  # Only save model weights, not optimizer/scheduler states
    fp16                  = FP16,
    bf16                  = BF16,
    warmup_ratio          = WARMUP_RATIO,
    learning_rate         = LEARNING_RATE,
    max_grad_norm         = MAX_GRAD_NORM,
    gradient_accumulation_steps = 1,
    # Logging configuration: effectively disable automatic training logs.
    logging_steps         = 999999,  # Set high enough that it rarely triggers.
    disable_tqdm          = False,
    log_level             = "error",  # Only surface error-level logs.
    logging_strategy      = "no",     # Disable automatic logging.
)

# ---------- 5. Misc ----------
LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
CLASS_NAMES   = ["Negative", "Neutral", "Positive"]
USE_WANDB     = False

# ---------- 6. Early Stopping ----------
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 5  # Stop early after 5 epochs without improvement.
EARLY_STOPPING_THRESHOLD = 0.0  # Any improvement counts.

if "EARLY_STOPPING_PATIENCE" in _DATASET_OVERRIDES:
    EARLY_STOPPING_PATIENCE = _DATASET_OVERRIDES["EARLY_STOPPING_PATIENCE"]
EARLY_STOPPING_PATIENCE = _env_int("EARLY_STOPPING_PATIENCE", EARLY_STOPPING_PATIENCE)

# ---------- 7. Output Control ----------
VERBOSE_TRAINING = True  # True = detailed JSON-style output, False = concise output.

# ---------- 8. Data Augmentation ----------
USE_DATA_AUG = False
DATA_AUG_SUFFIX = ""
