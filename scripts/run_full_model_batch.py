#!/usr/bin/env python3
"""
Batch runner for the true DABS full-model configuration.

Default behavior:
- datasets: Laptop-14, Restaurant-14, Restaurant-15, Restaurant-16
- seeds: 42, 123, 456
- model: microsoft/deberta-v3-base

This script always uses PHASE_A_CONFIGS['full_model'] and calls:
    python scripts/train.py --dual-layer
with the matching ABLATION_CONFIG payload.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.ablation_configs import DATASETS, PHASE_A_CONFIGS


DEFAULT_DATASETS = [2, 3, 4, 5]
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_MAX_PARALLEL = 3
CONFIG_KEY = "full_model"


@dataclass
class RunResult:
    dataset_id: int
    dataset_name: str
    seed: int
    success: bool
    return_code: int
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    checkpoint_path: Optional[str] = None
    stdout_log: Optional[str] = None
    stderr_log: Optional[str] = None
    error: Optional[str] = None


class SimpleLogger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M:%S UTC]")
        line = f"{timestamp} {message}"
        print(line)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_full_model_config() -> Dict[str, Any]:
    raw = dict(PHASE_A_CONFIGS[CONFIG_KEY])
    for key in ("name", "description", "expected_improvement"):
        raw.pop(key, None)
    return raw


def dataset_name_from_id(dataset_id: int) -> str:
    if dataset_id not in DATASETS:
        raise ValueError(
            f"Unsupported dataset id {dataset_id}. "
            f"Available ids in ablation_configs.py: {sorted(DATASETS.keys())}"
        )
    return DATASETS[dataset_id]


def parse_training_output(stdout: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for line in stdout.splitlines():
        if line.startswith("ABLATION_RESULTS_JSON:"):
            payload = line.split("ABLATION_RESULTS_JSON:", 1)[1].strip()
            try:
                parsed.update(json.loads(payload))
            except json.JSONDecodeError:
                pass
        elif "Best model saved to" in line:
            parsed["checkpoint_path"] = line.split("Best model saved to", 1)[1].strip()
    return parsed


def run_one(
    dataset_id: int,
    seed: int,
    model_name: str,
    full_model_cfg: Dict[str, Any],
    results_dir: Path,
    use_test_mode: bool,
    logger: SimpleLogger,
) -> RunResult:
    dataset_name = dataset_name_from_id(dataset_id)
    cmd = [sys.executable, "scripts/train.py", "--dual-layer"]

    env = os.environ.copy()
    env["DATASET_CHOICE"] = str(dataset_id)
    env["MODEL_NAME"] = model_name
    env["RANDOM_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    env["OUTPUT_REL_DIR"] = f"full_model/{dataset_name}"
    env["ABLATION_CONFIG"] = json.dumps(full_model_cfg, ensure_ascii=False)
    if use_test_mode:
        env["USE_TEST_MODE"] = "1"

    logger.log(
        f"Running dataset={dataset_name} (id={dataset_id}) seed={seed} "
        f"model={model_name}"
    )
    logger.log(f"Command: {' '.join(cmd)}")

    process = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    run_logs_dir = results_dir / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = run_logs_dir / f"{dataset_name}_seed_{seed}.stdout.log"
    stderr_log = run_logs_dir / f"{dataset_name}_seed_{seed}.stderr.log"
    stdout_log.write_text(process.stdout, encoding="utf-8")
    stderr_log.write_text(process.stderr, encoding="utf-8")

    parsed = parse_training_output(process.stdout)
    success = process.returncode == 0 and parsed.get("accuracy") is not None

    result = RunResult(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        seed=seed,
        success=success,
        return_code=process.returncode,
        accuracy=parsed.get("accuracy"),
        macro_f1=parsed.get("macro_f1", parsed.get("mf1_score")),
        checkpoint_path=parsed.get("checkpoint_path"),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        error=None if success else "Training failed or result JSON missing.",
    )

    if result.success:
        logger.log(
            f"SUCCESS dataset={dataset_name} seed={seed} "
            f"acc={result.accuracy:.4f} mf1={result.macro_f1:.4f}"
        )
        if result.checkpoint_path:
            logger.log(f"Checkpoint: {result.checkpoint_path}")
    else:
        logger.log(
            f"FAILED dataset={dataset_name} seed={seed} return_code={process.returncode}"
        )
        if process.stderr.strip():
            logger.log(f"stderr saved to {stderr_log}")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-run only the true DABS full-model configuration."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        default=DEFAULT_DATASETS,
        help=f"Dataset ids to run. Default: {DEFAULT_DATASETS}",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Random seeds to run. Default: {DEFAULT_SEEDS}",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Backbone model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable USE_TEST_MODE=1 for quick smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without executing them.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help=f"Concurrent jobs. Default: {DEFAULT_MAX_PARALLEL}",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    full_model_cfg = build_full_model_config()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / f"full_model_batch_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(results_dir / "full_model_batch.log")

    logger.log("Starting full-model batch run")
    logger.log(f"Datasets: {args.datasets}")
    logger.log(f"Seeds: {args.seeds}")
    logger.log(f"Model: {args.model_name}")
    logger.log(f"Full-model config: {json.dumps(full_model_cfg, ensure_ascii=False)}")

    planned_runs = [
        (dataset_id, dataset_name_from_id(dataset_id), seed)
        for dataset_id in args.datasets
        for seed in args.seeds
    ]

    if args.dry_run:
        for dataset_id, dataset_name, seed in planned_runs:
            logger.log(
                f"DRY RUN dataset={dataset_name} (id={dataset_id}) seed={seed}"
            )
        logger.log(f"Total planned runs: {len(planned_runs)}")
        return

    results: List[RunResult] = []
    total_runs = len(planned_runs)
    idx = 0
    future_to_run = {}
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        while planned_runs or future_to_run:
            while planned_runs and len(future_to_run) < args.max_parallel:
                dataset_id, _, seed = planned_runs.pop(0)
                idx += 1
                logger.log(f"Progress {idx}/{total_runs}")
                future = executor.submit(
                    run_one,
                    dataset_id=dataset_id,
                    seed=seed,
                    model_name=args.model_name,
                    full_model_cfg=full_model_cfg,
                    results_dir=results_dir,
                    use_test_mode=args.test,
                    logger=logger,
                )
                future_to_run[future] = (dataset_id, seed)

            done, _ = wait(future_to_run, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results.append(result)
                future_to_run.pop(future, None)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_key": CONFIG_KEY,
        "full_model_config": full_model_cfg,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "model_name": args.model_name,
        "success_count": sum(1 for r in results if r.success),
        "failure_count": sum(1 for r in results if not r.success),
        "results": [r.__dict__ for r in results],
    }

    summary_path = results_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.log(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
