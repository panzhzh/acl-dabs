#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset-specific training hyperparameter overrides.

These files are intentionally limited to recipe-level settings so that
architecture/method choices remain defined by the default config and the
ablation configuration.
"""

DATASET_CONFIG_MODULES = {
    2: "src.config.dataset_configs.laptop14",
    3: "src.config.dataset_configs.restaurant14",
    4: "src.config.dataset_configs.restaurant15",
    5: "src.config.dataset_configs.restaurant16",
}

DATASET_CONFIG_NAMES = {
    2: "laptop14",
    3: "restaurant14",
    4: "restaurant15",
    5: "restaurant16",
}
