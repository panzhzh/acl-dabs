"""Portable checkpoint helpers for the public journal implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(path: str | Path, payload: Any) -> Path:
    """Write a checkpoint atomically without experiment-specific staging."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu"):
    """Load a DABS checkpoint using safe tensor-only deserialization when available."""

    checkpoint = Path(path).expanduser().resolve()
    try:
        return torch.load(checkpoint, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location=map_location)
