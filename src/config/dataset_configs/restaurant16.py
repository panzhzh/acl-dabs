#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset-specific recipe for Restaurant-16.

Initial values intentionally match the current global defaults so existing
results remain unchanged before dataset-wise tuning starts.
"""

CONFIG_NAME = "restaurant16"

OVERRIDES = {
    "NUM_EPOCHS": 30,
    "TRAIN_BATCH_SIZE": 32,
    "EVAL_BATCH_SIZE": 64,
    "WARMUP_RATIO": 0.0,
    "WEIGHT_DECAY": 0.01,
    "MAX_GRAD_NORM": 1.0,
    "LEARNING_RATE": 2e-5,
}
