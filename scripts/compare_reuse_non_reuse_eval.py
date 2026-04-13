#!/usr/bin/env python3
"""
Compare non-reuse vs reuse evaluation for a dual-layer checkpoint.

This script loads one checkpoint, runs the same test set through:
1. Non-reuse evaluation: one full forward per aspect
2. Reuse evaluation: one shared sentence encoding + per-aspect readout

It reports accuracy, macro-F1, wall-clock time, and speedup in English.
"""

import argparse
import json
import os
import sys
import time
from statistics import mean
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config
from src.core.data import (
    preprocess_and_filter,
    read_dataset,
    tokenize_texts,
    tokenize_texts_with_aspect_positions,
    TweetsDataset,
)
from src.core.model import load_tokenizer, load_bert_model, move_model_to_device


def _resolve_checkpoint_state_file(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if path.is_file():
        return path
    for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No model state file found under checkpoint dir: {path}. "
        "Expected one of model.safetensors / pytorch_model.bin / model.safetensors.index.json."
    )


def _load_state_dict(state_file: Path) -> Dict[str, torch.Tensor]:
    suffix = state_file.suffix.lower()
    if state_file.name == "model.safetensors.index.json":
        from safetensors.torch import load_file as load_safetensors_file

        index_data = json.loads(state_file.read_text(encoding="utf-8"))
        weight_map = index_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        merged: Dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            shard_path = state_file.parent / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard file listed in index: {shard_path}")
            merged.update(load_safetensors_file(str(shard_path)))
        return merged
    if suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file

        return load_safetensors_file(str(state_file))
    if suffix in {".bin", ".pt", ".pth"}:
        state_obj = torch.load(str(state_file), map_location="cpu")
        if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
            return state_obj["state_dict"]
        if isinstance(state_obj, dict):
            return state_obj
        raise RuntimeError(f"Unsupported state object type in {state_file}: {type(state_obj)}")
    raise RuntimeError(f"Unsupported checkpoint file format: {state_file}")


def _load_sidecar_eval_metadata(checkpoint_dir: Path) -> Dict[str, Any]:
    metrics_path = checkpoint_dir / "eval_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metadata = data.get("metadata", {})
    metrics = data.get("metrics", {})
    return {"metadata": metadata, "metrics": metrics}


def _infer_architecture_config(state_dict: Dict[str, torch.Tensor], k_value: Optional[int]) -> Dict[str, Any]:
    keys = set(state_dict.keys())

    def has_prefix(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in keys)

    arch = {
        "enable_multi_scale": has_prefix("dora_multi_scale_convs") or has_prefix("dora_scale_fusion"),
        "enable_inter_gru": has_prefix("dora_inter_gru_cell") or has_prefix("dora_inter_lstm_cell"),
        "enable_aspect_attention": has_prefix("acbs_aspect_context_mha"),
        "enable_context_importance": has_prefix("acbs_token_gating_net"),
        "enable_layer_attention": has_prefix("acbs_layer_attention"),
        "disable_gating": not has_prefix("acbs_fusion_gate"),
        "k_value": int(k_value) if k_value is not None else 6,
        "sparse_weight": 1e-3,
    }
    if has_prefix("dora_inter_lstm_cell"):
        arch["inter_layer_fusion"] = "lstm"
    elif has_prefix("dora_inter_gru_cell"):
        arch["inter_layer_fusion"] = "gru"
    return arch


def _build_sentence_groups(df) -> List[List[int]]:
    groups: List[List[int]] = []
    current_group: List[int] = []
    previous_key = None

    for idx, row in df.reset_index(drop=True).iterrows():
        token_field = row["token"]
        token_key = tuple(token_field) if isinstance(token_field, list) else token_field
        current_key = (row["clean_text"], token_key)
        if previous_key is None or current_key != previous_key:
            if current_group:
                groups.append(current_group)
            current_group = []
            previous_key = current_key
        current_group.append(int(idx))

    if current_group:
        groups.append(current_group)
    return groups


def _is_pure_dora(arch_config: Dict[str, Any]) -> bool:
    return (
        (arch_config.get("enable_multi_scale", False) or arch_config.get("enable_inter_gru", False))
        and not (
            arch_config.get("enable_aspect_attention", False)
            or arch_config.get("enable_context_importance", False)
            or arch_config.get("enable_layer_attention", False)
        )
    )


def _build_test_dataset(tokenizer, df_test, arch_config: Dict[str, Any], max_length: int):
    x_test = df_test["clean_text"]
    y_test = df_test["label"]
    if _is_pure_dora(arch_config):
        test_encodings = tokenize_texts(tokenizer, x_test, max_length=max_length)
        return TweetsDataset(test_encodings, y_test.tolist())

    test_encodings, test_aspect_positions = tokenize_texts_with_aspect_positions(
        tokenizer, x_test, df_test, max_length=max_length
    )
    return TweetsDataset(test_encodings, y_test.tolist(), test_aspect_positions)


def _normalize_aspect_positions(aspect_positions):
    if aspect_positions is None:
        return None
    if isinstance(aspect_positions, tuple):
        return [tuple(int(x) for x in aspect_positions)]
    if isinstance(aspect_positions, list):
        if len(aspect_positions) == 0:
            return None
        first = aspect_positions[0]
        if isinstance(first, (tuple, list)):
            return [tuple(int(x) for x in first)]
        if len(aspect_positions) == 2 and all(isinstance(x, (int, np.integer)) for x in aspect_positions):
            return [tuple(int(x) for x in aspect_positions)]
    return aspect_positions


def _build_grouped_eval_items(df, test_dataset) -> List[Dict[str, Any]]:
    items = []
    for group_indices in _build_sentence_groups(df):
        first_sample = test_dataset[group_indices[0]]
        shared_inputs = {
            key: value.unsqueeze(0)
            for key, value in first_sample.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"} and torch.is_tensor(value)
        }

        aspects = []
        for idx in group_indices:
            sample = test_dataset[idx]
            aspects.append(
                {
                    "aspect_positions": _normalize_aspect_positions(sample.get("aspect_token_positions")),
                    "aspect_mask": sample["aspect_mask"].unsqueeze(0),
                    "label": int(sample["labels"].item()),
                    "sample_inputs": {
                        key: value.unsqueeze(0)
                        for key, value in sample.items()
                        if key in {"input_ids", "attention_mask", "token_type_ids"} and torch.is_tensor(value)
                    },
                }
            )

        items.append({"shared_inputs": shared_inputs, "aspects": aspects})
    return items


def _metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


def _select_items_by_aspect_count(items: List[Dict[str, Any]], aspect_count: int) -> List[Dict[str, Any]]:
    return [item for item in items if len(item["aspects"]) == aspect_count]


def _build_subset_result(
    subset_name: str,
    items: List[Dict[str, Any]],
    model,
    timing_repeats: int,
) -> Optional[Dict[str, Any]]:
    if not items:
        return None

    non_reuse = _evaluate_path(model, items, reuse=False, timing_repeats=timing_repeats)
    reuse = _evaluate_path(model, items, reuse=True, timing_repeats=timing_repeats)
    return {
        "subset_name": subset_name,
        "num_sentences": int(len(items)),
        "num_samples": int(reuse["num_samples"]),
        "non_reuse": non_reuse,
        "reuse": reuse,
        "delta": {
            "accuracy": float(reuse["accuracy"] - non_reuse["accuracy"]),
            "macro_f1": float(reuse["macro_f1"] - non_reuse["macro_f1"]),
            "seconds": float(reuse["seconds"] - non_reuse["seconds"]),
            "speedup_x": float(non_reuse["seconds"] / reuse["seconds"]) if reuse["seconds"] > 0 else None,
        },
    }


def _build_aspect_count_breakdown(
    items: List[Dict[str, Any]],
    model,
    timing_repeats: int,
    aspect_counts: List[int],
) -> Dict[str, Any]:
    breakdown: Dict[str, Any] = {}
    for aspect_count in aspect_counts:
        subset_items = _select_items_by_aspect_count(items, aspect_count)
        subset_result = _build_subset_result(f"M_eq_{aspect_count}", subset_items, model, timing_repeats)
        if subset_result is not None:
            breakdown[f"M_eq_{aspect_count}"] = subset_result
    return breakdown


def _move_items_to_device(items: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    moved_items: List[Dict[str, Any]] = []
    for item in items:
        moved_items.append(
            {
                "shared_inputs": {k: v.to(device) for k, v in item["shared_inputs"].items()},
                "aspects": [
                    {
                        "aspect_positions": aspect["aspect_positions"],
                        "aspect_mask": aspect["aspect_mask"].to(device),
                        "label": aspect["label"],
                        "sample_inputs": {k: v.to(device) for k, v in aspect["sample_inputs"].items()},
                    }
                    for aspect in item["aspects"]
                ],
            }
        )
    return moved_items


def _warmup_paths(model, items: List[Dict[str, Any]], warmup_repeats: int) -> None:
    if not items:
        return
    first_inputs = items[0]["shared_inputs"]
    first_aspect = items[0]["aspects"][0]
    for _ in range(max(1, int(warmup_repeats))):
        with torch.no_grad():
            _ = model(
                **first_inputs,
                aspect_token_positions=first_aspect["aspect_positions"],
                aspect_mask=first_aspect["aspect_mask"],
            )
            shared = model.encode_shared(**first_inputs)
            _ = model.forward_aspect_from_shared(
                shared,
                aspect_token_positions=first_aspect["aspect_positions"],
                aspect_mask=first_aspect["aspect_mask"],
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _predict_non_reuse(model, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true: List[int] = []
    y_pred_parts: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for item in items:
            for aspect in item["aspects"]:
                outputs = model(
                    **aspect["sample_inputs"],
                    aspect_token_positions=aspect["aspect_positions"],
                    aspect_mask=aspect["aspect_mask"],
                )
                y_pred_parts.append(outputs["logits"].argmax(dim=-1).detach())
                y_true.append(aspect["label"])

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = torch.cat(y_pred_parts, dim=0).cpu().numpy()
    metrics = _metrics_from_preds(y_true_np, y_pred_np)
    return {**metrics, "num_samples": int(len(y_true_np))}


def _predict_reuse(model, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true: List[int] = []
    y_pred_parts: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for item in items:
            shared_output = model.encode_shared(**item["shared_inputs"])
            for aspect in item["aspects"]:
                outputs = model.forward_aspect_from_shared(
                    shared_output,
                    aspect_token_positions=aspect["aspect_positions"],
                    aspect_mask=aspect["aspect_mask"],
                )
                y_pred_parts.append(outputs["logits"].argmax(dim=-1).detach())
                y_true.append(aspect["label"])

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = torch.cat(y_pred_parts, dim=0).cpu().numpy()
    metrics = _metrics_from_preds(y_true_np, y_pred_np)
    return {**metrics, "num_samples": int(len(y_true_np))}


def _time_non_reuse(model, items: List[Dict[str, Any]], repeats: int) -> float:
    elapsed_times: List[float] = []
    model.eval()
    for _ in range(max(1, int(repeats))):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for item in items:
                for aspect in item["aspects"]:
                    _ = model(
                        **aspect["sample_inputs"],
                        aspect_token_positions=aspect["aspect_positions"],
                        aspect_mask=aspect["aspect_mask"],
                    )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_times.append(time.perf_counter() - start)
    return float(mean(elapsed_times))


def _time_reuse(model, items: List[Dict[str, Any]], repeats: int) -> float:
    elapsed_times: List[float] = []
    model.eval()
    for _ in range(max(1, int(repeats))):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for item in items:
                shared_output = model.encode_shared(**item["shared_inputs"])
                for aspect in item["aspects"]:
                    _ = model.forward_aspect_from_shared(
                        shared_output,
                        aspect_token_positions=aspect["aspect_positions"],
                        aspect_mask=aspect["aspect_mask"],
                    )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_times.append(time.perf_counter() - start)
    return float(mean(elapsed_times))


def _evaluate_path(model, items: List[Dict[str, Any]], reuse: bool, timing_repeats: int) -> Dict[str, Any]:
    metrics = _predict_reuse(model, items) if reuse else _predict_non_reuse(model, items)
    seconds = _time_reuse(model, items, timing_repeats) if reuse else _time_non_reuse(model, items, timing_repeats)
    return {
        **metrics,
        "seconds": float(seconds),
    }


def _pretty_print(result: Dict[str, Any]) -> None:
    print("\nReuse vs Non-Reuse Evaluation")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Dataset: {result['dataset_name']} (choice={result['dataset_choice']})")
    print(f"Device: {result['device']}")
    print(f"Samples: {result['reuse']['num_samples']}")
    print("")
    print(
        f"Overall: non-reuse={result['non_reuse']['seconds']:.4f}s, "
        f"reuse={result['reuse']['seconds']:.4f}s, "
        f"speedup={result['delta']['speedup_x']:.4f}x, "
        f"acc={result['reuse']['accuracy']:.10f}, "
        f"mf1={result['reuse']['macro_f1']:.10f}"
    )

    by_aspect_count = result.get("by_aspect_count", {})
    if by_aspect_count:
        print("")
        print("By Aspect Count")
        for key in sorted(by_aspect_count.keys()):
            subset = by_aspect_count[key]
            label = key.replace("M_eq_", "M=")
            print(
                f"  {label}: sentences={subset['num_sentences']}, samples={subset['num_samples']}, "
                f"non-reuse={subset['non_reuse']['seconds']:.4f}s, "
                f"reuse={subset['reuse']['seconds']:.4f}s, "
                f"speedup={subset['delta']['speedup_x']:.4f}x"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare non-reuse vs reuse evaluation for a dual-layer checkpoint."
    )
    parser.add_argument("checkpoint", help="Checkpoint directory or checkpoint state file.")
    parser.add_argument("--dataset-choice", type=int, default=None, help="Dataset choice override.")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", config.MODEL_NAME),
        help="Backbone model name used to build the model.",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length.")
    parser.add_argument("--k-value", type=int, default=None, help="Override k_value if needed.")
    parser.add_argument("--device", default="", help='Optional device override, e.g. "cuda:0" or "cpu".')
    parser.add_argument("--json", default="", help="Optional path to save the JSON result.")
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=10,
        help="Repeat pure forward timing this many times and report the mean.",
    )
    parser.add_argument(
        "--timing-warmup",
        type=int,
        default=3,
        help="Warm up both paths this many times before timing.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full JSON result to stdout.",
    )
    args = parser.parse_args()

    state_file = _resolve_checkpoint_state_file(args.checkpoint)
    checkpoint_dir = state_file.parent
    sidecar = _load_sidecar_eval_metadata(checkpoint_dir)
    metadata = sidecar.get("metadata", {})

    dataset_choice = args.dataset_choice or metadata.get("dataset_choice")
    if dataset_choice is None:
        raise ValueError(
            "Could not infer dataset_choice from eval_metrics.json. "
            "Please provide --dataset-choice explicitly."
        )
    dataset_choice = int(dataset_choice)

    config.DATASET_CHOICE = dataset_choice
    _, df_test = read_dataset(dataset_choice, test_mode=False)
    df_test = preprocess_and_filter(df_test)

    state_dict = _load_state_dict(state_file)
    arch_config = _infer_architecture_config(state_dict, k_value=args.k_value)

    tokenizer = load_tokenizer(args.model_name)
    model = load_bert_model(
        model_name=args.model_name,
        num_labels=config.NUM_LABELS,
        use_dual_layer=True,
        **arch_config,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if args.device:
        device = torch.device(args.device)
        model.to(device)
    else:
        model, device = move_model_to_device(model)

    test_dataset = _build_test_dataset(tokenizer, df_test, arch_config, max_length=int(args.max_length))
    items_cpu = _build_grouped_eval_items(df_test, test_dataset)
    items = _move_items_to_device(items_cpu, device)
    _warmup_paths(model, items, args.timing_warmup)

    non_reuse = _evaluate_path(model, items, reuse=False, timing_repeats=args.timing_repeats)
    reuse = _evaluate_path(model, items, reuse=True, timing_repeats=args.timing_repeats)
    by_aspect_count = _build_aspect_count_breakdown(items, model, args.timing_repeats, aspect_counts=[2, 3, 4])

    result = {
        "checkpoint": str(checkpoint_dir),
        "state_file": str(state_file),
        "dataset_choice": dataset_choice,
        "dataset_name": metadata.get("dataset_name", checkpoint_dir.parent.name),
        "device": str(device),
        "model_name": args.model_name,
        "architecture_config": arch_config,
        "timing_warmup": int(args.timing_warmup),
        "timing_repeats": int(args.timing_repeats),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "sidecar_metrics": sidecar.get("metrics", {}),
        "non_reuse": non_reuse,
        "reuse": reuse,
        "delta": {
            "accuracy": float(reuse["accuracy"] - non_reuse["accuracy"]),
            "macro_f1": float(reuse["macro_f1"] - non_reuse["macro_f1"]),
            "seconds": float(reuse["seconds"] - non_reuse["seconds"]),
            "speedup_x": float(non_reuse["seconds"] / reuse["seconds"]) if reuse["seconds"] > 0 else None,
        },
        "by_aspect_count": by_aspect_count,
    }

    _pretty_print(result)
    if args.print_json:
        print("\nJSON")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.json:
        out_path = Path(args.json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
