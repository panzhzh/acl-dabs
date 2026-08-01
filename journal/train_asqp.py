#!/usr/bin/env python3
"""Train the complete DORA--QCBS model for Rest15/Rest16 ASQP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOURNAL_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from journal.dabs_structured.checkpoint import atomic_torch_save  # noqa: E402
from journal.dabs_structured.model import DABSStructuredModel  # noqa: E402
from journal.dabs_structured.asqp.data import (  # noqa: E402
    DEFAULT_REST_MAX_LENGTH,
    DEFAULT_REST_QUAD_ROOT,
    NUM_REST_CATEGORIES,
    REST_DATASETS,
    RestQuadExample,
    read_rest_quad_split,
)
from journal.dabs_structured.asqp.dataset import (  # noqa: E402
    RestQuadCollator,
    RestQuadHardNegative,
    RestQuadTrainingDataset,
)
from journal.dabs_structured.asqp.decode import (  # noqa: E402
    decode_rest_quad_examples,
    predictions_at_threshold,
    score_rest_quad_predictions,
    score_rest_quad_predictions_official,
)


MODEL_NAME = "microsoft/deberta-v3-base"
TRAINER_VERSION = "dabs-structured-asqp-v1"
HARD_NEGATIVE_FORMAT_VERSION = 1
DEFAULT_MAX_LENGTH = DEFAULT_REST_MAX_LENGTH
DEFAULT_THRESHOLDS = (
    0.02,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    0.995,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    torch.cuda.set_rng_state_all(payload["cuda"])


def require_cuda(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Rest Full training requires CUDA; CPU fallback is disabled")
    if device.index is not None:
        torch.cuda.set_device(device.index)
    torch.set_float32_matmul_precision("high")
    return device


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _dataset(
    examples,
    tokenizer,
    args,
    *,
    hard_negative_pairs_by_example=None,
):
    return RestQuadTrainingDataset(
        examples,
        tokenizer,
        max_length=args.max_length,
        max_pairs=args.max_pairs,
        pair_candidate_max_span_len=args.pair_candidate_max_span_len,
        span_proposal_max_len=args.span_proposal_max_len,
        max_span_proposals=args.max_span_proposals,
        candidate_seed=args.seed,
        hard_negative_pairs_by_example=hard_negative_pairs_by_example,
    )


def examples_digest(examples: Sequence[RestQuadExample]) -> str:
    """Stable identity for a split without serializing implementation details."""

    return digest(
        [
            {
                "text": example.text,
                "line_no": example.line_no,
                "quadruples": [row.as_tuple() for row in example.raw_quads],
            }
            for example in examples
        ]
    )


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_hard_negative_file(
    path: Path | None,
    *,
    dataset: str,
    examples: Sequence[RestQuadExample],
    max_length: int,
    model_name: str,
):
    if path is None:
        return None, {
            "enabled": False,
            "selection": "deterministic random negatives only",
        }
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if int(payload.get("format_version", -1)) != HARD_NEGATIVE_FORMAT_VERSION:
        raise ValueError("Unsupported Rest hard-negative artifact version")
    if payload.get("dataset") != dataset or payload.get("split") != "train":
        raise ValueError(
            "Hard-negative artifact must match the selected dataset train split"
        )
    if payload.get("model_name") != model_name:
        raise ValueError("Hard-negative artifact tokenizer/model mismatch")
    if int(payload.get("max_length", -1)) != int(max_length):
        raise ValueError("Hard-negative artifact max_length mismatch")
    expected_digest = examples_digest(examples)
    if payload.get("examples_sha256") != expected_digest:
        raise ValueError("Hard-negative artifact training examples mismatch")
    example_rows = payload.get("examples")
    if not isinstance(example_rows, list) or len(example_rows) != len(examples):
        raise ValueError(
            "Hard-negative artifact must contain exactly one row per example"
        )

    parsed = []
    selected_pairs = 0
    for index, (example, row) in enumerate(zip(examples, example_rows)):
        if int(row.get("example_index", -1)) != index:
            raise ValueError("Hard-negative example indices are not contiguous")
        text_sha = hashlib.sha256(example.text.encode("utf-8")).hexdigest()
        if row.get("text_sha256") != text_sha:
            raise ValueError(f"Hard-negative text mismatch at example {index}")
        pairs = []
        prior_score = float("inf")
        for pair in row.get("pairs", []):
            pair_span = tuple(int(value) for value in pair["pair_span"])
            if len(pair_span) != 4:
                raise ValueError("Hard-negative pair_span must have four values")
            score = float(pair["score"])
            if score > prior_score + 1e-12:
                raise ValueError(
                    "Hard-negative rows must be sorted by descending score"
                )
            prior_score = score
            pairs.append(RestQuadHardNegative(pair_span, score))
        parsed.append(tuple(pairs))
        selected_pairs += len(pairs)
    return tuple(parsed), {
        "enabled": True,
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "format_version": HARD_NEGATIVE_FORMAT_VERSION,
        "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
        "source_train_file_sha256": payload.get("train_file_sha256"),
        "source_run_config_sha256": payload.get("source_run_config_sha256"),
        "examples_sha256": expected_digest,
        "examples": len(parsed),
        "selected_pairs": selected_pairs,
        "mining": payload.get("mining"),
    }


def _category_pos_weights(dataset, cap: float, mode: str = "balanced"):
    mode = mode.strip().lower()
    if mode not in {"balanced", "sqrt", "none"}:
        raise ValueError(f"Unsupported category pos-weight mode: {mode!r}")
    positives = torch.zeros(NUM_REST_CATEGORIES, dtype=torch.float64)
    valid_pairs = 0
    for feature in dataset.features:
        mask = feature.category_target_mask.bool()
        positives += feature.category_targets[mask].double().sum(dim=0)
        valid_pairs += int(mask.sum())
    balanced = torch.ones_like(positives)
    observed = positives > 0
    negatives = float(valid_pairs) - positives
    balanced[observed] = (negatives[observed] / positives[observed]).clamp(
        min=1.0,
        max=cap,
    )
    if mode == "balanced":
        weights = balanced
    elif mode == "sqrt":
        # Clamp before taking the root so ``cap`` retains its exact meaning.
        weights = balanced.sqrt()
    else:
        weights = torch.ones_like(balanced)
    return weights.float(), {
        "mode": mode,
        "positive_relation_pairs": valid_pairs,
        "positive_assignments": int(positives.sum().item()),
        "labels_observed_in_train": int(observed.sum().item()),
        "labels_absent_from_train": int((~observed).sum().item()),
        "cap": float(cap),
        "balanced_min": float(balanced.min()),
        "balanced_median": float(balanced.median()),
        "balanced_max": float(balanced.max()),
        "min": float(weights.min()),
        "median": float(weights.median()),
        "max": float(weights.max()),
    }


def _model_config(args, category_pos_weight):
    return {
        "model_name": args.model_name,
        "k_value": 12,
        "dropout": args.dropout,
        "bio_loss_weights": (0.2, 1.0, 1.0),
        "span_proposal_loss_weight": 1.0,
        "span_proposal_pos_weights": (20.0, 20.0),
        "span_proposal_ranking_loss_weight": 0.5,
        "span_proposal_ranking_margin": 1.0,
        "span_proposal_ranking_negatives": 16,
        "pair_head_type": "factorized",
        "pair_loss_weights": (1.0, 2.0, 2.0, 2.0),
        "pair_focal_gamma": 2.0,
        "pair_selection_pos_weight": args.relation_positive_weight,
        "pair_relation_loss_weight": 1.0,
        "pair_polarity_loss_weight": 1.0,
        "pair_contrastive_loss_weight": args.pair_contrastive_loss_weight,
        "pair_contrastive_temperature": 0.1,
        "pair_distance_embedding_dim": 64,
        "pair_distance_max": 32,
        "enable_null_aspects": True,
        "num_category_labels": NUM_REST_CATEGORIES,
        "category_loss_weight": 1.0,
        "category_pos_weight": category_pos_weight.tolist(),
    }


def _optimizer_groups(model, args):
    encoder, head, classifier, category_classifier = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            encoder.append(parameter)
        elif name.startswith("category_classifier."):
            category_classifier.append(parameter)
        elif "classifier" in name or "proposal_depth_output" in name:
            classifier.append(parameter)
        else:
            head.append(parameter)
    groups = [
        {"params": encoder, "lr": args.encoder_learning_rate},
        {"params": head, "lr": args.head_learning_rate},
        {"params": classifier, "lr": args.classifier_learning_rate},
        {
            "params": category_classifier,
            "lr": (
                args.category_learning_rate
                if args.category_learning_rate is not None
                else args.classifier_learning_rate
            ),
        },
    ]
    parameter_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("Optimizer parameter groups contain duplicates")
    return groups


def _model_batch(batch, device):
    pair_spans = batch["pair_spans"].to(device, non_blocking=True).clone()
    null_mask = batch["pair_null_mask"].to(device, non_blocking=True)
    if bool(null_mask[..., 1].any()):
        raise RuntimeError("Rest data unexpectedly contains a NULL opinion")
    pair_spans[..., 0:2] = torch.where(
        null_mask[..., 0:1],
        torch.full_like(pair_spans[..., 0:2], -1),
        pair_spans[..., 0:2],
    )
    keys = (
        "input_ids",
        "attention_mask",
        "aspect_bio_labels",
        "opinion_bio_labels",
        "pair_mask",
        "pair_labels",
        "category_targets",
        "category_target_mask",
        "span_proposal_spans",
        "span_aspect_labels",
        "span_opinion_labels",
        "span_proposal_mask",
    )
    output = {
        key: batch[key].to(device, non_blocking=True)
        for key in keys
    }
    output["pair_spans"] = pair_spans
    return output


def _decode_kwargs(args, device):
    return {
        "device": device,
        "max_length": args.max_length,
        "max_pairs": args.decode_max_pairs,
        "max_proposal_span_len": args.span_proposal_max_len,
        "span_proposal_threshold": args.span_proposal_threshold,
        "span_proposal_top_k": args.span_proposal_top_k,
    }


def _aggregate_candidate_diagnostics(rows):
    rows = list(rows)

    def total(key):
        return sum(int(row.get(key, 0)) for row in rows)

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 1.0

    pair_counts_before = [int(row["pair_count_before_cap"]) for row in rows]
    pair_counts_after = [int(row["pair_count"]) for row in rows]
    gold_pairs = total("gold_representable_pairs")
    gold_explicit_pairs = total("gold_explicit_pairs")
    gold_null_pairs = total("gold_null_pairs")
    gold_aspects = total("gold_explicit_aspects")
    gold_opinions = total("gold_opinions")
    coverage_keys = {
        "proposal_only_pair": "covered_gold_pairs_proposal_only",
        "bio_only_pair": "covered_gold_pairs_bio_only",
        "union_pair_before_cap": "covered_gold_pairs_union_before_cap",
        "candidate_pair_before_cap": "covered_gold_pairs_before_cap",
        "candidate_pair_after_cap": "covered_gold_pairs",
    }
    pair_coverage = {
        name: {
            "covered": total(key),
            "gold": gold_pairs,
            "recall": ratio(total(key), gold_pairs),
        }
        for name, key in coverage_keys.items()
    }
    span_coverage = {}
    for role, denominator, keys in (
        (
            "aspect",
            gold_aspects,
            {
                "proposal": "covered_gold_aspects_proposal",
                "bio": "covered_gold_aspects_bio",
                "union": "covered_gold_aspects_union",
            },
        ),
        (
            "opinion",
            gold_opinions,
            {
                "proposal": "covered_gold_opinions_proposal",
                "bio": "covered_gold_opinions_bio",
                "union": "covered_gold_opinions_union",
            },
        ),
    ):
        span_coverage[role] = {
            source: {
                "covered": total(key),
                "gold": denominator,
                "recall": ratio(total(key), denominator),
            }
            for source, key in keys.items()
        }
    cap_sentences = sum(bool(row.get("cap_applied", False)) for row in rows)
    return {
        "sentences": len(rows),
        "pair_count_before_cap": {
            "mean": float(np.mean(pair_counts_before)) if rows else 0.0,
            "p95": float(np.percentile(pair_counts_before, 95)) if rows else 0.0,
        },
        "pair_count_after_cap": {
            "mean": float(np.mean(pair_counts_after)) if rows else 0.0,
            "p95": float(np.percentile(pair_counts_after, 95)) if rows else 0.0,
        },
        "cap": {
            "sentences": cap_sentences,
            "sentence_fraction": ratio(cap_sentences, len(rows)),
            "candidate_pairs_removed": total("pairs_removed_by_cap"),
            "gold_pairs_removed": total("gold_pairs_removed_by_cap"),
        },
        "pair_coverage": pair_coverage,
        "explicit_pair_coverage": {
            "before_cap": {
                "covered": total("covered_gold_explicit_pairs_before_cap"),
                "gold": gold_explicit_pairs,
                "recall": ratio(
                    total("covered_gold_explicit_pairs_before_cap"),
                    gold_explicit_pairs,
                ),
            },
            "after_cap": {
                "covered": total("covered_gold_explicit_pairs"),
                "gold": gold_explicit_pairs,
                "recall": ratio(
                    total("covered_gold_explicit_pairs"), gold_explicit_pairs
                ),
            },
        },
        "null_pair_coverage": {
            "before_cap": {
                "covered": total("covered_gold_null_pairs_before_cap"),
                "gold": gold_null_pairs,
                "recall": ratio(
                    total("covered_gold_null_pairs_before_cap"), gold_null_pairs
                ),
            },
            "after_cap": {
                "covered": total("covered_gold_null_pairs"),
                "gold": gold_null_pairs,
                "recall": ratio(
                    total("covered_gold_null_pairs"), gold_null_pairs
                ),
            },
        },
        "span_coverage": span_coverage,
    }


@torch.no_grad()
def select_on_dev(model, tokenizer, examples, args, device):
    synchronize(device)
    started = time.perf_counter()
    decoded = decode_rest_quad_examples(
        model,
        tokenizer,
        examples,
        precision="bf16",
        quad_threshold=1.1,
        return_score_lattice=True,
        **_decode_kwargs(args, device),
    )
    candidates = []
    for threshold in args.quad_thresholds:
        predictions = predictions_at_threshold(
            examples,
            decoded.diagnostics,
            float(threshold),
        )
        official_metrics = score_rest_quad_predictions_official(predictions, examples)
        candidates.append(
            {
                "threshold": float(threshold),
                "metrics": official_metrics,
                "exact_multiset": score_rest_quad_predictions(
                    predictions, examples
                ),
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            float(row["metrics"]["quadruple"]["f1"]),
            float(row["metrics"]["quadruple"]["precision"]),
            float(row["metrics"]["quadruple"]["recall"]),
            -float(row["threshold"]),
        ),
    )
    synchronize(device)
    return {
        "threshold": best["threshold"],
        "metrics": best["metrics"],
        "candidate_diagnostics": _aggregate_candidate_diagnostics(
            decoded.diagnostics
        ),
        "wall_s": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate_test_once(model, tokenizer, examples, threshold, args, device):
    synchronize(device)
    started = time.perf_counter()
    decoded = decode_rest_quad_examples(
        model,
        tokenizer,
        examples,
        precision="bf16",
        quad_threshold=float(threshold),
        return_score_lattice=False,
        **_decode_kwargs(args, device),
    )
    synchronize(device)
    return {
        "predictions": decoded.predictions,
        "raw_official": score_rest_quad_predictions_official(
            decoded.predictions, examples
        ),
        "exact_multiset": score_rest_quad_predictions(
            decoded.predictions, examples
        ),
        "representable_only": score_rest_quad_predictions(
            decoded.predictions,
            examples,
            representable_only=True,
        ),
        "candidate_diagnostics": _aggregate_candidate_diagnostics(
            decoded.diagnostics
        ),
        "wall_s": time.perf_counter() - started,
    }


def _save_predictions(path: Path, examples, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example, rows in zip(examples, predictions):
            handle.write(
                json.dumps(
                    {
                        "text": example.text,
                        "quadruples": [row.as_tuple() for row in rows],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(path)


def train(args):
    device = require_cuda(args.device)
    if args.model_name != MODEL_NAME:
        raise RuntimeError(f"Rest Full protocol is locked to {MODEL_NAME}")
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    test_access_path = run_dir / "test_access.json"
    if test_access_path.exists() and not args.force:
        raise RuntimeError(
            "This run has already crossed the frozen test-access boundary but "
            "did not produce a complete summary. Refusing an automatic second "
            "test evaluation; inspect the run and use --force only for an "
            "explicitly documented replacement run."
        )

    set_seed(args.seed)
    # ``fix_mistral_regex`` is intentionally false for DeBERTa-v3.  The
    # Transformers compatibility flag targets Mistral tokenizers and turns
    # ordinary spaces into ``[UNK]`` tokens for this SentencePiece model.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        fix_mistral_regex=False,
    )
    train_read = read_rest_quad_split(
        args.dataset, "train", data_root=args.data_root
    )
    dev_read = read_rest_quad_split(args.dataset, "dev", data_root=args.data_root)
    train_examples = list(train_read.examples)
    dev_examples = list(dev_read.examples)
    if args.max_train_examples is not None:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_dev_examples is not None:
        dev_examples = dev_examples[: args.max_dev_examples]
    hard_negative_pairs, hard_negative_report = _load_hard_negative_file(
        args.hard_negative_file,
        dataset=args.dataset,
        examples=train_examples,
        max_length=args.max_length,
        model_name=args.model_name,
    )
    train_dataset = _dataset(
        train_examples,
        tokenizer,
        args,
        hard_negative_pairs_by_example=hard_negative_pairs,
    )
    dev_alignment = _dataset(dev_examples, tokenizer, args)
    alignment = {
        "train": train_dataset.alignment_stats.as_dict(),
        "dev": dev_alignment.alignment_stats.as_dict(),
    }
    write_json(run_dir / "alignment_preselection.json", alignment)

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=RestQuadCollator(),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    category_pos_weight, category_weight_report = _category_pos_weights(
        train_dataset,
        args.category_pos_weight_cap,
        args.category_pos_weight_mode,
    )
    model_config = _model_config(args, category_pos_weight)
    model = DABSStructuredModel(**model_config).to(device=device, dtype=torch.float32)
    model_device = next(model.parameters()).device
    if model_device != device:
        raise RuntimeError(
            f"Rest model device mismatch: expected {device}, found {model_device}"
        )
    optimizer_kwargs = {
        "params": _optimizer_groups(model, args),
        "lr": args.encoder_learning_rate,
        "weight_decay": args.weight_decay,
        "fused": True,
    }
    try:
        optimizer = torch.optim.AdamW(**optimizer_kwargs)
    except (RuntimeError, TypeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(**optimizer_kwargs)
    steps_per_epoch = max(
        1, math.ceil(len(loader) / args.gradient_accumulation_steps)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(steps_per_epoch * args.epochs * args.warmup_ratio),
        num_training_steps=steps_per_epoch * args.epochs,
    )

    run_config = {
        "trainer_version": TRAINER_VERSION,
        "architecture": "Full DORA--QCBS",
        "dataset": args.dataset,
        "seed": args.seed,
        "model_name": args.model_name,
        "tokenizer_fix_mistral_regex": False,
        "precision": "bf16",
        "max_length": args.max_length,
        "max_pairs": args.max_pairs,
        "pair_candidate_max_span_len": args.pair_candidate_max_span_len,
        "span_proposal_max_len": args.span_proposal_max_len,
        "max_span_proposals": args.max_span_proposals,
        "decode_max_pairs": args.decode_max_pairs,
        "span_proposal_threshold": args.span_proposal_threshold,
        "span_proposal_top_k": args.span_proposal_top_k,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "early_stop_patience": args.early_stop_patience,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "category_learning_rate": (
            args.category_learning_rate
            if args.category_learning_rate is not None
            else args.classifier_learning_rate
        ),
        "category_pos_weight_mode": args.category_pos_weight_mode,
        "category_pos_weight_cap": args.category_pos_weight_cap,
        "hard_negatives": hard_negative_report,
        "relation_positive_weight": args.relation_positive_weight,
        "pair_contrastive_loss_weight": args.pair_contrastive_loss_weight,
        "dropout": args.dropout,
        "k_value": model_config["k_value"],
        "proposal_depth_evidence": "span",
        "pair_depth_evidence": "span",
        "quad_thresholds": list(args.quad_thresholds),
        "protocol_sha256": args.protocol_sha256,
        "timer_scope": (
            "CUDA-synchronized wall clock from immediately before epoch 1 through "
            "each development selection; data/model construction excluded"
        ),
        "smoke": bool(args.smoke),
        "dev_only": bool(args.dev_only),
    }
    config_sha = digest(run_config)
    write_json(run_dir / "run_config.json", {**run_config, "sha256": config_sha})

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    if args.force:
        best_path.unlink(missing_ok=True)
        last_path.unlink(missing_ok=True)
    history = []
    best_f1 = -1.0
    best_precision = -1.0
    best_epoch = 0
    best_threshold = None
    best_dev = None
    best_timing = None
    stale = 0
    start_epoch = 1
    resumed = False
    resume_path = None
    if args.resume and not args.force:
        if last_path.exists():
            resume_path = last_path
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["run_config_sha256"] != config_sha:
            raise RuntimeError("Resume checkpoint configuration mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint["history"]
        best_f1 = checkpoint["best_f1"]
        best_precision = checkpoint["best_precision"]
        best_epoch = checkpoint["best_epoch"]
        best_threshold = checkpoint["best_threshold"]
        best_dev = checkpoint["best_dev"]
        best_timing = checkpoint["best_timing"]
        stale = checkpoint["stale"]
        start_epoch = checkpoint["epoch"] + 1
        restore_rng_state(checkpoint["rng_state"])
        resumed = True

    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    prior_cumulative_wall_s = (
        float(history[-1]["cumulative_train_to_selection_wall_s"])
        if history
        else 0.0
    )
    train_to_selection_started = time.perf_counter()
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        synchronize(device)
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        component_loss_sums: dict[str, float] = {}
        batches = 0
        optimizer_steps = 0
        for batch_index, batch in enumerate(loader, start=1):
            model_batch = _model_batch(batch, device)
            if epoch == start_epoch and batch_index == 1:
                misplaced = {
                    key: str(value.device)
                    for key, value in model_batch.items()
                    if torch.is_tensor(value) and value.device != device
                }
                if misplaced:
                    raise RuntimeError(
                        f"Rest first batch contains non-CUDA tensors: {misplaced}"
                    )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**model_batch)
                loss = output["loss"]
            if loss is None or not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite Rest loss at epoch {epoch}: {loss}")
            (loss / args.gradient_accumulation_steps).backward()
            if (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            loss_sum += float(loss.detach().float().cpu())
            for key, value in output.items():
                if key == "loss" or not key.endswith("_loss"):
                    continue
                if not torch.is_tensor(value) or value.numel() != 1:
                    continue
                component_loss_sums[key] = component_loss_sums.get(key, 0.0) + float(
                    value.detach().float().cpu()
                )
            batches += 1
        synchronize(device)
        train_wall_s = time.perf_counter() - epoch_started
        dev = select_on_dev(model, tokenizer, dev_examples, args, device)
        cumulative_wall_s = (
            prior_cumulative_wall_s
            + time.perf_counter()
            - train_to_selection_started
        )
        quad = dev["metrics"]["quadruple"]
        f1 = float(quad["f1"])
        precision = float(quad["precision"])
        improved = (f1, precision) > (best_f1, best_precision)
        prospective_history = history + [
            {
                "train_wall_s": train_wall_s,
                "dev_selection_wall_s": float(dev["wall_s"]),
            }
        ]
        if improved:
            best_f1 = f1
            best_precision = precision
            best_epoch = epoch
            best_threshold = float(dev["threshold"])
            best_dev = dev
            train_wall_to_selected = sum(
                float(row["train_wall_s"]) for row in prospective_history
            )
            examples_to_selected = len(train_examples) * epoch
            best_timing = {
                "selected_epoch": epoch,
                "train_examples_processed": examples_to_selected,
                "train_step_wall_s": train_wall_to_selected,
                "dev_selection_wall_s": sum(
                    float(row["dev_selection_wall_s"])
                    for row in prospective_history
                ),
                "train_to_selected_checkpoint_wall_s": cumulative_wall_s,
                "train_examples_per_s": (
                    examples_to_selected / train_wall_to_selected
                    if train_wall_to_selected
                    else 0.0
                ),
                "selection_inclusive_examples_per_s": (
                    examples_to_selected / cumulative_wall_s
                    if cumulative_wall_s
                    else 0.0
                ),
                "cuda_synchronized": True,
            }
            stale = 0
            atomic_torch_save(
                best_path,
                {
                    "run_config_sha256": config_sha,
                    "run_config": run_config,
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_threshold": best_threshold,
                    "best_dev": best_dev,
                    "best_timing": best_timing,
                },
            )
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, batches),
            "train_loss_components": {
                key: value / max(1, batches)
                for key, value in sorted(component_loss_sums.items())
            },
            "batches": batches,
            "optimizer_steps": optimizer_steps,
            "train_wall_s": train_wall_s,
            "train_examples_per_s": len(train_examples) / train_wall_s,
            "dev_selection_wall_s": float(dev["wall_s"]),
            "cumulative_train_to_selection_wall_s": cumulative_wall_s,
            "dev": dev,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "stale_epochs": stale,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        history.append(row)
        atomic_torch_save(
            last_path,
            {
                "run_config_sha256": config_sha,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_f1": best_f1,
                "best_precision": best_precision,
                "best_epoch": best_epoch,
                "best_threshold": best_threshold,
                "best_dev": best_dev,
                "best_timing": best_timing,
                "stale": stale,
                "rng_state": capture_rng_state(),
            },
        )
        write_json(run_dir / "history.json", history)
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        if (
            args.early_stop_patience > 0
            and epoch >= args.min_epochs
            and stale >= args.early_stop_patience
        ):
            stopped_early = True
            break

    selected_best_path = best_path
    if not selected_best_path.exists():
        raise RuntimeError("No development-selected Rest checkpoint was produced")
    checkpoint = torch.load(
        selected_best_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["run_config_sha256"] != config_sha:
        raise RuntimeError("Selected checkpoint configuration mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = int(checkpoint["best_epoch"])
    best_threshold = float(checkpoint["best_threshold"])
    best_dev = checkpoint["best_dev"]
    best_timing = checkpoint["best_timing"]
    model.to(device)
    def clean_resume_files() -> None:
        last_path.unlink(missing_ok=True)

    if args.dev_only:
        summary = {
            "status": "dev_complete",
            "trainer_version": TRAINER_VERSION,
            "created_utc": utc_now(),
            "architecture": "Full DORA--QCBS",
            "dataset": args.dataset,
            "seed": args.seed,
            "model_name": MODEL_NAME,
            "precision": "bf16",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "protocol_sha256": args.protocol_sha256,
            "run_config_sha256": config_sha,
            "run_config": run_config,
            "read_stats": {
                "train": train_read.stats.as_dict(),
                "dev": dev_read.stats.as_dict(),
            },
            "alignment": alignment,
            "category_pos_weight": category_weight_report,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "best_dev": best_dev,
            "selected_checkpoint_timing": best_timing,
            "test": None,
            "test_access_policy": "test split was not read in dev-only mode",
            "history": history,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "resumed": resumed,
            "resumed_from": str(resume_path) if resume_path is not None else None,
            "full_run_train_wall_s": sum(
                float(row["train_wall_s"]) for row in history
            ),
            "full_run_dev_selection_wall_s": sum(
                float(row["dev_selection_wall_s"]) for row in history
            ),
            "gpu_peak_allocated_bytes": max(
                int(row["gpu_peak_allocated_bytes"]) for row in history
            ),
            "gpu_peak_reserved_bytes": max(
                int(row["gpu_peak_reserved_bytes"]) for row in history
            ),
            "checkpoint": str(best_path),
        }
        write_json(summary_path, summary)
        clean_resume_files()
        return summary

    # Test access begins only after the checkpoint and threshold are frozen.
    test_loaded_utc = utc_now()
    write_json(
        test_access_path,
        {
            "architecture": "Full DORA--QCBS",
            "dataset": args.dataset,
            "seed": args.seed,
            "run_config_sha256": config_sha,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "test_loaded_utc": test_loaded_utc,
            "policy": (
                "frozen audit marker written immediately before the one allowed "
                "test split read"
            ),
        },
    )
    test_read = read_rest_quad_split(args.dataset, "test", data_root=args.data_root)
    test_examples = list(test_read.examples)
    if args.max_test_examples is not None:
        test_examples = test_examples[: args.max_test_examples]
    test_alignment = _dataset(test_examples, tokenizer, args)
    alignment["test"] = test_alignment.alignment_stats.as_dict()
    write_json(run_dir / "alignment.json", alignment)
    test = evaluate_test_once(
        model,
        tokenizer,
        test_examples,
        best_threshold,
        args,
        device,
    )
    predictions = test.pop("predictions")
    prediction_path = run_dir / "test_predictions.jsonl"
    _save_predictions(prediction_path, test_examples, predictions)
    summary = {
        "status": "complete",
        "trainer_version": TRAINER_VERSION,
        "created_utc": utc_now(),
        "architecture": "Full DORA--QCBS",
        "dataset": args.dataset,
        "seed": args.seed,
        "model_name": MODEL_NAME,
        "precision": "bf16",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "protocol_sha256": args.protocol_sha256,
        "run_config_sha256": config_sha,
        "run_config": run_config,
        "read_stats": {
            "train": train_read.stats.as_dict(),
            "dev": dev_read.stats.as_dict(),
            "test": test_read.stats.as_dict(),
        },
        "alignment": alignment,
        "category_pos_weight": category_weight_report,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_dev": best_dev,
        "selected_checkpoint_timing": best_timing,
        "test_loaded_utc": test_loaded_utc,
        "test_access_policy": (
            "test split loaded and evaluated exactly once after dev-only "
            "checkpoint and threshold selection"
        ),
        "test": test,
        "history": history,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "resumed": resumed,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "full_run_train_wall_s": sum(float(row["train_wall_s"]) for row in history),
        "full_run_dev_selection_wall_s": sum(
            float(row["dev_selection_wall_s"]) for row in history
        ),
        "gpu_peak_allocated_bytes": max(
            int(row["gpu_peak_allocated_bytes"]) for row in history
        ),
        "gpu_peak_reserved_bytes": max(
            int(row["gpu_peak_reserved_bytes"]) for row in history
        ),
        "checkpoint": str(best_path),
        "predictions": str(prediction_path),
    }
    write_json(summary_path, summary)
    clean_resume_files()
    return summary


def parse_args():
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=Path, default=None)
    known_config, _ = config_probe.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file containing official ASQP training defaults.",
    )
    parser.add_argument("--dataset", choices=REST_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_REST_QUAD_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=20)
    # The matched DeBERTa ASQP protocol caps training at 50 epochs and stops
    # after five consecutive development selections without improvement.
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    # The released Rest15/Rest16 data fit within the 128-token protocol.  The
    # alignment audit confirms that every surface-representable quad in
    # all six official splits remains token-representable at this length, so
    # padding every training example to 256 only wastes compute and memory.
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--pair-candidate-max-span-len", type=int, default=3)
    parser.add_argument("--span-proposal-max-len", type=int, default=3)
    parser.add_argument("--max-span-proposals", type=int, default=384)
    parser.add_argument("--decode-max-pairs", type=int, default=256)
    parser.add_argument("--span-proposal-threshold", type=float, default=0.5)
    parser.add_argument("--span-proposal-top-k", type=int, default=8)
    parser.add_argument(
        "--quad-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--category-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--category-pos-weight-cap", type=float, default=100.0)
    parser.add_argument(
        "--category-pos-weight-mode",
        choices=("balanced", "sqrt", "none"),
        default="balanced",
    )
    parser.add_argument("--hard-negative-file", type=Path, default=None)
    parser.add_argument("--relation-positive-weight", type=float, default=2.0)
    parser.add_argument("--pair-contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--protocol-sha256", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    if known_config.config is not None:
        config_path = known_config.config.expanduser().resolve()
        config_defaults = json.loads(config_path.read_text(encoding="utf-8"))
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = set(config_defaults) - valid_keys
        if unknown_keys:
            parser.error(
                f"Unknown keys in {config_path}: {sorted(unknown_keys)}"
            )
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    if args.run_dir is None:
        args.run_dir = (
            JOURNAL_ROOT
            / "outputs"
            / "asqp"
            / args.dataset
            / f"seed_{args.seed}"
        )
    if args.smoke:
        args.epochs = 1
        args.min_epochs = 1
        args.early_stop_patience = 0
        args.num_workers = 0
        args.max_train_examples = args.max_train_examples or 8
        args.max_dev_examples = args.max_dev_examples or 4
        args.max_test_examples = args.max_test_examples or 4
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("batch sizes must be positive")
    if args.category_pos_weight_cap < 1.0:
        parser.error("--category-pos-weight-cap must be at least 1")
    if (
        args.category_learning_rate is not None
        and args.category_learning_rate <= 0.0
    ):
        parser.error("--category-learning-rate must be positive")
    return args


def main():
    args = parse_args()
    result = train(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "seed": result["seed"],
                "best_epoch": result["best_epoch"],
                "test": result.get("test"),
                "summary": str(args.run_dir.expanduser().resolve() / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
