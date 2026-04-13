#!/usr/bin/env python3
"""
Run reuse-path inference for checkpoints under outputs/full_model.

This script reconstructs the current full-model architecture, loads each
`model.safetensors`, evaluates on the official test split, saves
`eval_metrics.json` and `predictions.json` under each seed directory, and
writes a summary CSV under the full_model root.
"""

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from scripts.ablation_configs import PHASE_A_CONFIGS
from scripts.train import run_reuse_sentence_grouped_eval
from src import config
from src.core.data import preprocess_and_filter, read_dataset
from src.core.model import load_bert_model, load_tokenizer, move_model_to_device


DATASET_ID_TO_NAME = {
    2: "Laptop-14",
    3: "Restaurant-14",
    4: "Restaurant-15",
    5: "Restaurant-16",
}
DATASET_NAME_TO_ID = {v: k for k, v in DATASET_ID_TO_NAME.items()}


def _resolve_state_file(seed_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"):
        candidate = seed_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model state file found under: {seed_dir}")


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


def _parse_seed(seed_dir_name: str) -> int:
    if not seed_dir_name.startswith("seed_"):
        raise ValueError(f"Invalid seed directory name: {seed_dir_name}")
    return int(seed_dir_name.split("_", 1)[1])


def _discover_models(root: Path, dataset_id: int | None, seed: int | None) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for dataset_name in sorted(DATASET_NAME_TO_ID.keys()):
        dataset_dir = root / dataset_name
        if not dataset_dir.exists():
            continue
        current_dataset_id = DATASET_NAME_TO_ID[dataset_name]
        if dataset_id is not None and current_dataset_id != dataset_id:
            continue
        for seed_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            current_seed = _parse_seed(seed_dir.name)
            if seed is not None and current_seed != seed:
                continue
            try:
                state_file = _resolve_state_file(seed_dir)
            except FileNotFoundError:
                continue
            entries.append(
                {
                    "dataset_id": current_dataset_id,
                    "dataset_name": dataset_name,
                    "seed": current_seed,
                    "seed_dir": seed_dir,
                    "state_file": state_file,
                }
            )
    return entries


def _build_architecture_config(k_value: int) -> Dict[str, Any]:
    raw = dict(PHASE_A_CONFIGS["full_model"])
    raw["k_value"] = int(k_value)
    return raw


def _clear_previous_outputs(seed_dir: Path) -> None:
    for name in ("eval_metrics.json", "predictions.json"):
        path = seed_dir / name
        if path.exists():
            path.unlink()


def _save_metrics_and_predictions(
    seed_dir: Path,
    dataset_id: int,
    dataset_name: str,
    seed: int,
    model_name: str,
    state_file: Path,
    arch_config: Dict[str, Any],
    metrics: Dict[str, Any],
    df_test,
    y_true,
    y_pred,
    missing_keys: List[str],
    unexpected_keys: List[str],
    device: str,
    max_length: int,
) -> Dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
        labels=list(range(config.NUM_LABELS)),
    )
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[2]

    eval_results = {
        "metadata": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_choice": dataset_id,
            "seed": seed,
            "state_file": str(state_file),
            "device": device,
            "max_length": int(max_length),
            "architecture_config": arch_config,
            "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
            "total_samples": int(len(y_true)),
        },
        "metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
        },
        "per_class_metrics": {
            config.CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(config.CLASS_NAMES))
        },
    }

    test_rows = df_test.reset_index(drop=True)
    predictions = []
    for idx, row in test_rows.iterrows():
        true_label = int(y_true[idx])
        pred_label = int(y_pred[idx])
        predictions.append(
            {
                "sample_id": int(idx),
                "text": row["text"],
                "aspect_term": row["aspect_term"],
                "from": None if row.get("from") is None else int(row["from"]),
                "to": None if row.get("to") is None else int(row["to"]),
                "true_label": true_label,
                "predicted_label": pred_label,
                "true_class": config.CLASS_NAMES[true_label],
                "predicted_class": config.CLASS_NAMES[pred_label],
                "correct": bool(true_label == pred_label),
            }
        )

    predictions_results = {
        "metadata": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_choice": dataset_id,
            "seed": seed,
            "state_file": str(state_file),
            "total_samples": int(len(y_true)),
            "class_names": config.CLASS_NAMES,
        },
        "predictions": predictions,
    }

    metrics_path = seed_dir / "eval_metrics.json"
    predictions_path = seed_dir / "predictions.json"
    metrics_path.write_text(json.dumps(eval_results, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return eval_results


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_detail(detail_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset_id",
        "dataset_name",
        "seed",
        "model_path",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "eval_accuracy",
        "eval_mf1",
        "total_samples",
        "status",
        "error",
    ]
    _write_csv(detail_path, rows, fieldnames)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _build_dataset_summary_rows(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, str], List[Dict[str, Any]]] = {}
    for row in detail_rows:
        key = (int(row["dataset_id"]), str(row["dataset_name"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    metric_keys = ["accuracy", "macro_f1", "weighted_f1", "eval_accuracy", "eval_mf1"]
    for (dataset_id, dataset_name), rows in sorted(grouped.items(), key=lambda x: x[0][0]):
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        aggregated = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "num_seeds": len(ok_rows),
            "seed_list": ",".join(str(row["seed"]) for row in sorted(ok_rows, key=lambda r: int(r["seed"]))),
            "status": "ok" if len(ok_rows) == len(rows) else "partial_error",
            "error": "" if len(ok_rows) == len(rows) else "; ".join(
                f"seed_{row['seed']}={row['error']}" for row in rows if row.get("status") != "ok"
            ),
        }
        for key in metric_keys:
            values = [float(row[key]) for row in ok_rows if _is_number(row.get(key))]
            aggregated[f"mean_{key}"] = (sum(values) / len(values)) if values else ""
        summary_rows.append(aggregated)
    return summary_rows


def _write_summary(summary_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset_id",
        "dataset_name",
        "num_seeds",
        "seed_list",
        "mean_accuracy",
        "mean_macro_f1",
        "mean_weighted_f1",
        "mean_eval_accuracy",
        "mean_eval_mf1",
        "status",
        "error",
    ]
    _write_csv(summary_path, rows, fieldnames)


def _build_error_row(entry: Dict[str, Any], error: str) -> Dict[str, Any]:
    return {
        "dataset_id": entry["dataset_id"],
        "dataset_name": entry["dataset_name"],
        "seed": entry["seed"],
        "model_path": str(entry["state_file"]),
        "accuracy": "",
        "macro_f1": "",
        "weighted_f1": "",
        "eval_accuracy": "",
        "eval_mf1": "",
        "total_samples": "",
        "status": "error",
        "error": error,
    }


def _build_summary_row_from_saved_files(entry: Dict[str, Any]) -> Dict[str, Any]:
    metrics_path = entry["seed_dir"] / "eval_metrics.json"
    if not metrics_path.exists():
        return _build_error_row(entry, "Missing eval_metrics.json after inference.")

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _build_error_row(entry, f"Failed to read eval_metrics.json: {exc}")

    metrics = data.get("metrics", {})
    metadata = data.get("metadata", {})
    return {
        "dataset_id": entry["dataset_id"],
        "dataset_name": entry["dataset_name"],
        "seed": entry["seed"],
        "model_path": str(entry["state_file"]),
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "eval_accuracy": metrics.get("eval_accuracy", ""),
        "eval_mf1": metrics.get("eval_mf1", ""),
        "total_samples": metadata.get("total_samples", ""),
        "status": "ok",
        "error": "",
    }


def _run_single_inference(entry: Dict[str, Any], model_name: str, max_length: int, k_value: int, device_override: str) -> Dict[str, Any]:
    dataset_id = entry["dataset_id"]
    dataset_name = entry["dataset_name"]
    seed = entry["seed"]
    seed_dir = entry["seed_dir"]
    state_file = entry["state_file"]

    print(f"[RUN] dataset={dataset_name} seed={seed} model={state_file}")
    _clear_previous_outputs(seed_dir)

    _, df_test = read_dataset(dataset_id, test_mode=False)
    df_test = preprocess_and_filter(df_test).reset_index(drop=True)

    arch_config = _build_architecture_config(k_value=k_value)
    use_dual_layer = bool(arch_config.pop("use_dual_layer", True))

    tokenizer = load_tokenizer(model_name)
    model = load_bert_model(
        model_name=model_name,
        num_labels=config.NUM_LABELS,
        use_dual_layer=use_dual_layer,
        **arch_config,
    )

    state_dict = _load_state_dict(state_file)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if device_override:
        device = torch.device(device_override)
        model.to(device)
    else:
        model, device = move_model_to_device(model)

    metrics, y_true, y_pred = run_reuse_sentence_grouped_eval(
        model=model,
        tokenizer=tokenizer,
        eval_df=df_test,
        max_length=max_length,
    )

    saved_metrics = _save_metrics_and_predictions(
        seed_dir=seed_dir,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        seed=seed,
        model_name=model_name,
        state_file=state_file,
        arch_config={**arch_config, "use_dual_layer": use_dual_layer},
        metrics=metrics,
        df_test=df_test,
        y_true=y_true,
        y_pred=y_pred,
        missing_keys=list(missing_keys),
        unexpected_keys=list(unexpected_keys),
        device=str(device),
        max_length=max_length,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "seed": seed,
        "model_path": str(state_file),
        "accuracy": saved_metrics["metrics"]["accuracy"],
        "macro_f1": saved_metrics["metrics"]["macro_f1"],
        "weighted_f1": saved_metrics["metrics"]["weighted_f1"],
        "eval_accuracy": saved_metrics["metrics"].get("eval_accuracy"),
        "eval_mf1": saved_metrics["metrics"].get("eval_mf1"),
        "total_samples": saved_metrics["metadata"]["total_samples"],
        "status": "ok",
        "error": "",
    }


def _run_worker_subprocess(entry: Dict[str, Any], args, script_path: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(script_path),
        "--root",
        str(args.root),
        "--dataset-id",
        str(entry["dataset_id"]),
        "--seed",
        str(entry["seed"]),
        "--model-name",
        args.model_name,
        "--max-length",
        str(args.max_length),
        "--k-value",
        str(args.k_value),
        "--worker",
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "entry": entry,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference for all models under outputs/full_model and save summary.csv."
    )
    parser.add_argument(
        "--root",
        default=str(Path(PROJECT_ROOT) / "outputs" / "full_model"),
        help="Root directory containing dataset/seed/model.safetensors layout.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        choices=sorted(DATASET_ID_TO_NAME.keys()),
        default=None,
        help="Optional dataset id filter: 2/3/4/5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed filter, e.g. 42.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", config.MODEL_NAME),
        help="Backbone model name used to reconstruct the model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer max length for inference.",
    )
    parser.add_argument(
        "--k-value",
        type=int,
        default=6,
        help="k_value used to reconstruct the current full model.",
    )
    parser.add_argument(
        "--device",
        default="",
        help='Optional device override, e.g. "cuda:0" or "cpu".',
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum number of inference workers. Default: 3.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Rebuild detail.csv and summary.csv from existing eval_metrics.json files without rerunning inference.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    entries = _discover_models(root=root, dataset_id=args.dataset_id, seed=args.seed)
    if not entries:
        raise SystemExit("No matching models found.")

    if args.rebuild_only and not args.worker:
        detail_rows = [_build_summary_row_from_saved_files(entry) for entry in entries]
        detail_path = root / "detail.csv"
        summary_path = root / "summary.csv"
        _write_detail(detail_path, detail_rows)
        _write_summary(summary_path, _build_dataset_summary_rows(detail_rows))
        print(f"[DONE] detail={detail_path}")
        print(f"[DONE] summary={summary_path}")
        return

    summary_rows: List[Dict[str, Any]] = []
    had_error = False

    if args.worker or args.max_parallel <= 1 or len(entries) == 1:
        for entry in entries:
            try:
                row = _run_single_inference(
                    entry=entry,
                    model_name=args.model_name,
                    max_length=int(args.max_length),
                    k_value=int(args.k_value),
                    device_override=args.device,
                )
            except Exception as exc:
                had_error = True
                row = _build_error_row(entry, str(exc))
                print(f"[ERROR] dataset={entry['dataset_name']} seed={entry['seed']} error={exc}")
            summary_rows.append(row)
        if not args.worker:
            detail_path = root / "detail.csv"
            summary_path = root / "summary.csv"
            _write_detail(detail_path, summary_rows)
            _write_summary(summary_path, _build_dataset_summary_rows(summary_rows))
            print(f"[DONE] detail={detail_path}")
            print(f"[DONE] summary={summary_path}")
        if had_error:
            raise SystemExit(1)
        return

    print(f"[INFO] Running inference with max_parallel={args.max_parallel}")
    script_path = Path(__file__).resolve()
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.max_parallel)) as executor:
        future_to_entry = {
            executor.submit(_run_worker_subprocess, entry, args, script_path): entry for entry in entries
        }
        for future in concurrent.futures.as_completed(future_to_entry):
            result = future.result()
            entry = result["entry"]
            if result["stdout"].strip():
                print(result["stdout"].strip())
            if result["returncode"] != 0:
                had_error = True
                err_text = result["stderr"].strip() or f"Worker failed with returncode={result['returncode']}"
                print(f"[ERROR] dataset={entry['dataset_name']} seed={entry['seed']} error={err_text}")
            elif result["stderr"].strip():
                print(result["stderr"].strip())

    for entry in entries:
        row = _build_summary_row_from_saved_files(entry)
        if row["status"] == "error":
            had_error = True
        summary_rows.append(row)

    detail_path = root / "detail.csv"
    summary_path = root / "summary.csv"
    _write_detail(detail_path, summary_rows)
    _write_summary(summary_path, _build_dataset_summary_rows(summary_rows))
    print(f"[DONE] detail={detail_path}")
    print(f"[DONE] summary={summary_path}")
    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
