#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate the complete DORA--QCBS model with exact-match ASTE decoding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOURNAL_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from journal.dabs_structured.aste.data import (  # noqa: E402
    gold_spans_for_example,
    read_aste_split,
    score_aste_examples,
    score_span_predictions,
)
from journal.dabs_structured.aste.decode import decode_aste_examples_batched  # noqa: E402
from journal.dabs_structured.aste.dataset import load_aste_tokenizer  # noqa: E402
from journal.dabs_structured.checkpoint import load_checkpoint as load_payload  # noqa: E402
from journal.dabs_structured.model import DABSStructuredModel  # noqa: E402


DEFAULT_ASTE_ROOT = JOURNAL_ROOT / "data" / "aste"


def load_model_state(model: DABSStructuredModel, checkpoint):
    model.validate_checkpoint_config(
        checkpoint.get("config") if isinstance(checkpoint, dict) else None
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.validate_checkpoint_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)


def evaluate(args) -> dict[str, object]:
    device = torch.device(args.device)
    examples = read_aste_split(args.root, args.dataset, args.split)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    checkpoint = (
        load_payload(args.checkpoint, map_location="cpu")
        if args.checkpoint is not None
        else None
    )
    checkpoint_config = (
        checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    )
    model_name = args.model_name or checkpoint_config.get(
        "model_name", "microsoft/deberta-v3-base"
    )
    k_value = args.k_value or int(checkpoint_config.get("k_value", 12))
    pair_head_type = args.pair_head_type or checkpoint_config.get(
        "pair_head_type", "joint"
    )
    pair_distance_embedding_dim = (
        args.pair_distance_embedding_dim
        if args.pair_distance_embedding_dim is not None
        else int(checkpoint_config.get("pair_distance_embedding_dim", 64))
    )
    max_length = args.max_length or int(checkpoint_config.get("max_length", 128))
    max_pairs = args.max_pairs or int(checkpoint_config.get("decode_max_pairs", 256))
    max_proposal_span_len = args.max_proposal_span_len or int(
        checkpoint_config.get("span_proposal_max_len", 3)
    )
    span_proposal_threshold = (
        args.span_proposal_threshold
        if args.span_proposal_threshold is not None
        else float(checkpoint_config.get("span_proposal_threshold", 0.7))
    )
    span_proposal_top_k = args.span_proposal_top_k or int(
        checkpoint_config.get("span_proposal_top_k", 4)
    )
    include_bio_proposals = (
        args.include_bio_proposals
        if args.include_bio_proposals is not None
        else bool(checkpoint_config.get("include_bio_proposals", True))
    )
    confidence_mode = args.confidence_mode or checkpoint_config.get(
        "confidence_mode", "joint"
    )

    tokenizer = load_aste_tokenizer(model_name)

    model = DABSStructuredModel(
        model_name=model_name,
        k_value=k_value,
        pair_head_type=pair_head_type,
        pair_distance_embedding_dim=pair_distance_embedding_dim,
        pair_distance_max=args.pair_distance_max,
    )
    if checkpoint is not None:
        load_model_state(model, checkpoint)
    threshold = args.pair_confidence_threshold
    if threshold is None and isinstance(checkpoint, dict):
        threshold = checkpoint.get("best_dev_threshold")
    threshold = 0.0 if threshold is None else float(threshold)
    model.to(device=device, dtype=torch.float32)

    started = time.time()
    decoded = decode_aste_examples_batched(
        model,
        tokenizer,
        examples,
        batch_size=args.decode_batch_size,
        device=device,
        max_length=max_length,
        max_pairs=max_pairs,
        pair_confidence_threshold=threshold,
        proposal_source=args.proposal_source,
        max_proposal_span_len=max_proposal_span_len,
        span_proposal_threshold=span_proposal_threshold,
        span_proposal_top_k=span_proposal_top_k,
        span_proposal_aspect_top_k=args.span_proposal_aspect_top_k,
        span_proposal_opinion_top_k=args.span_proposal_opinion_top_k,
        include_bio_proposals=include_bio_proposals,
        confidence_mode=confidence_mode,
        pair_pruning_mode=args.pair_pruning_mode,
        precision=args.precision,
    )
    elapsed_s = time.time() - started

    triplet_metrics = score_aste_examples(decoded.predictions, examples).as_dict()
    aspect_metrics = score_span_predictions(
        decoded.aspect_spans,
        [gold_spans_for_example(example, "aspect") for example in examples],
    ).as_dict()
    opinion_metrics = score_span_predictions(
        decoded.opinion_spans,
        [gold_spans_for_example(example, "opinion") for example in examples],
    ).as_dict()

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "model_name": model_name,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "examples": len(examples),
        "decode_batch_size": args.decode_batch_size,
        "elapsed_s": elapsed_s,
        "examples_per_second": len(examples) / elapsed_s if elapsed_s > 0 else 0.0,
        "triplet": triplet_metrics,
        "aspect_span": aspect_metrics,
        "opinion_span": opinion_metrics,
        "pair_count_mean": sum(decoded.pair_counts) / len(decoded.pair_counts)
        if decoded.pair_counts
        else 0.0,
        "pair_count_max": max(decoded.pair_counts) if decoded.pair_counts else 0,
        "numerical_fallback_examples": sum(
            bool(diagnostic.get("batched_numerical_fallback"))
            for diagnostic in decoded.diagnostics
        ),
        "pair_confidence_threshold": threshold,
        "proposal_source": args.proposal_source,
        "max_proposal_span_len": max_proposal_span_len,
        "span_proposal_threshold": span_proposal_threshold,
        "span_proposal_top_k": span_proposal_top_k,
        "proposal_depth_evidence": "span",
        "pair_head_type": pair_head_type,
        "pair_depth_evidence": "span",
        "span_proposal_aspect_top_k": args.span_proposal_aspect_top_k,
        "span_proposal_opinion_top_k": args.span_proposal_opinion_top_k,
        "include_bio_proposals": include_bio_proposals,
        "confidence_mode": confidence_mode,
        "pair_pruning_mode": args.pair_pruning_mode,
        "precision": args.precision,
    }
    if args.output_json:
        out_path = args.output_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end DABS ASTE decoding.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ASTE_ROOT)
    parser.add_argument(
        "--dataset",
        default="14lap",
        help="Dataset directory name under --root (for example 14lap, ca, or eu).",
    )
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--decode-batch-size", type=int, default=16)
    parser.add_argument("--proposal-source", choices=["bio", "gold", "enumerated", "span_proposal"], default="span_proposal")
    parser.add_argument("--max-proposal-span-len", type=int, default=None)
    parser.add_argument("--span-proposal-threshold", type=float, default=None)
    parser.add_argument("--span-proposal-top-k", type=int, default=None)
    parser.add_argument("--span-proposal-aspect-top-k", type=int, default=None)
    parser.add_argument("--span-proposal-opinion-top-k", type=int, default=None)
    parser.add_argument(
        "--include-bio-proposals",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--confidence-mode", choices=["pair", "joint"], default=None)
    parser.add_argument("--pair-pruning-mode", choices=["sequential", "proposal_score", "selector"], default="sequential")
    parser.add_argument("--pair-confidence-threshold", type=float, default=None)
    parser.add_argument(
        "--pair-head-type",
        choices=["joint", "factorized"],
        default=None,
    )
    parser.add_argument("--pair-distance-embedding-dim", type=int, default=None)
    parser.add_argument("--pair-distance-max", type=int, default=32)
    parser.add_argument("--k-value", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny model, CPU, and a few examples for a fast decoder check.",
    )
    args = parser.parse_args()

    if args.checkpoint is None and not args.smoke:
        parser.error("--checkpoint is required outside --smoke mode")
    if args.output_json is None and args.checkpoint is not None:
        args.output_json = args.checkpoint.expanduser().resolve().parent / f"{args.split}.json"

    if args.smoke:
        args.model_name = "sshleifer/tiny-distilroberta-base"
        args.max_examples = 4
        args.max_length = min(args.max_length or 96, 96)
        args.max_pairs = min(args.max_pairs or 64, 64)
        args.decode_batch_size = 1
        args.k_value = min(args.k_value or 2, 2)
        args.device = "cpu"
        args.precision = "fp32"

    result = evaluate(args)
    print("DABS ASTE end-to-end decode result")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
