#!/usr/bin/env python3
# src/config/__init__.py
# -*- coding: utf-8 -*-

"""
Configuration management for ABSA project.

This package contains the main configuration file in config/config.py
All configuration variables are imported and exposed at package level.

Usage:
    from src.config import MODEL_NAME, DATA_DIR  # Import specific items
    # or
    from src import config  # Import the module
    # then use config.MODEL_NAME, config.DATA_DIR, etc.
"""

import os

# Select config profile at import time (per-process).
# Default profile keeps original behavior.
_profile = os.environ.get("TRAIN_CONFIG_PROFILE", os.environ.get("CONFIG_PROFILE", "default"))
_profile = (_profile or "default").strip().lower()

if _profile in {"qwen", "qwen7b", "qwen_7b"}:
    from .config_qwen import *  # noqa: F401,F403
    ACTIVE_CONFIG_PROFILE = "qwen"
else:
    from .config import *  # noqa: F401,F403
    ACTIVE_CONFIG_PROFILE = "default"
