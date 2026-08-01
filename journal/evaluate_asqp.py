#!/usr/bin/env python3
"""Evaluate a development-selected DABS checkpoint on Rest15/Rest16 ASQP."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from journal.dabs_structured.asqp.data import (  # noqa: E402
    DEFAULT_REST_QUAD_ROOT,
    REST_DATASETS,
    read_rest_quad_split,
)
from journal.dabs_structured.asqp.decode import (  # noqa: E402
    decode_rest_quad_examples,
    score_rest_quad_predictions,
    score_rest_quad_predictions_official,
)
from journal.dabs_structured.checkpoint import load_checkpoint  # noqa: E402
from journal.dabs_structured.model import DABSStructuredModel  # noqa: E402


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_predictions(path: Path, examples, predictions) -> None:
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


def evaluate(args) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is not None:
            torch.cuda.set_device(device.index)
        torch.set_float32_matmul_precision("high")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("ASQP checkpoint must contain model_config and model_state_dict")
    raw_model_config = dict(checkpoint.get("model_config") or {})
    run_config = dict(checkpoint.get("run_config") or {})
    allowed = set(inspect.signature(DABSStructuredModel.__init__).parameters)
    model_config = {
        key: value for key, value in raw_model_config.items() if key in allowed
    }
    model = DABSStructuredModel(**model_config)
    model.validate_checkpoint_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("model_name", "microsoft/deberta-v3-base"),
        use_fast=True,
        fix_mistral_regex=False,
    )
    read_result = read_rest_quad_split(
        args.dataset,
        args.split,
        data_root=args.data_root,
    )
    examples = list(read_result.examples)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint["best_threshold"])
    )
    max_length = (
        int(args.max_length)
        if args.max_length is not None
        else int(run_config.get("max_length", 128))
    )
    max_pairs = (
        int(args.max_pairs)
        if args.max_pairs is not None
        else int(run_config.get("decode_max_pairs", 256))
    )
    span_proposal_max_len = (
        int(args.span_proposal_max_len)
        if args.span_proposal_max_len is not None
        else int(run_config.get("span_proposal_max_len", 3))
    )
    span_proposal_threshold = (
        float(args.span_proposal_threshold)
        if args.span_proposal_threshold is not None
        else float(run_config.get("span_proposal_threshold", 0.5))
    )
    span_proposal_top_k = (
        int(args.span_proposal_top_k)
        if args.span_proposal_top_k is not None
        else int(run_config.get("span_proposal_top_k", 8))
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    decoded = decode_rest_quad_examples(
        model,
        tokenizer,
        examples,
        precision=args.precision,
        device=device,
        max_length=max_length,
        max_pairs=max_pairs,
        max_proposal_span_len=span_proposal_max_len,
        span_proposal_threshold=span_proposal_threshold,
        span_proposal_top_k=span_proposal_top_k,
        quad_threshold=threshold,
        return_score_lattice=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "examples": len(examples),
        "threshold": threshold,
        "max_length": max_length,
        "max_pairs": max_pairs,
        "span_proposal_max_len": span_proposal_max_len,
        "span_proposal_threshold": span_proposal_threshold,
        "span_proposal_top_k": span_proposal_top_k,
        "official": score_rest_quad_predictions_official(
            decoded.predictions, examples
        ),
        "exact_multiset": score_rest_quad_predictions(
            decoded.predictions, examples
        ),
        "representable_only": score_rest_quad_predictions(
            decoded.predictions, examples, representable_only=True
        ),
        "elapsed_s": elapsed_s,
        "examples_per_second": len(examples) / elapsed_s if elapsed_s else 0.0,
    }
    _write_predictions(args.predictions, examples, decoded.predictions)
    _write_json(args.output_json, result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Full DABS ASQP model.")
    parser.add_argument("--dataset", choices=REST_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_REST_QUAD_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--span-proposal-max-len", type=int, default=None)
    parser.add_argument("--span-proposal-threshold", type=float, default=None)
    parser.add_argument("--span-proposal-top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.checkpoint.expanduser().resolve().parent
    if args.output_json is None:
        args.output_json = run_dir / f"{args.split}.json"
    if args.predictions is None:
        args.predictions = run_dir / f"{args.split}_predictions.jsonl"
    if args.precision == "bf16" and torch.device(args.device).type != "cuda":
        args.precision = "fp32"
    return args


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
