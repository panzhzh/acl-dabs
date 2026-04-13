#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Qwen training-mode experiments:
1) freeze_backbone (head-only / DABS-only trainable)
2) qlora (4-bit + LoRA adapters)

This script orchestrates multiple calls to scripts/train.py and collects:
- accuracy / macro_f1
- latency p50/p95
- total/trainable params

Example:
  python scripts/run_qwen_training_modes.py \
    --model-names Qwen/Qwen3-8B,Qwen/Qwen3-14B,Qwen/Qwen3-32B \
    --datasets 2,3,4,5,6,7,8 --seeds 42 \
    --architectures full_model,encoder_only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.ablation_configs import PHASE_A_CONFIGS  # noqa: E402


DATASET_ID_TO_TAG = {
    1: "ACL-14",
    2: "Laptop-14",
    3: "Restaurant-14",
    4: "Restaurant-15",
    5: "Restaurant-16",
    6: "Restaurant-16_FR",
    7: "Restaurant-16_RU",
    8: "Restaurant-16_ES",
    9: "Restaurant-16_DU",
    10: "Restaurant-16_TU",
    11: "Phone-16_DU",
}

ARCH_FULL_MODEL = "full_model"
ARCH_ENCODER_ONLY = "encoder_only"
VALID_ARCHITECTURES = (ARCH_FULL_MODEL, ARCH_ENCODER_ONLY)


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sanitize(s: str) -> str:
    s = s.strip().replace("/", "_")
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)


def _model_suffix(name: str) -> str:
    return _sanitize(name.split("/")[-1] if "/" in name else name)


def _parse_int_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_model_names(model_name: str, model_names_csv: str) -> List[str]:
    raw = _parse_str_csv(model_names_csv) if str(model_names_csv or "").strip() else [str(model_name).strip()]
    out: List[str] = []
    for name in raw:
        if name and name not in out:
            out.append(name)
    return out


def _normalize_architecture(arch: str) -> str:
    key = arch.strip().lower()
    alias = {
        "full": ARCH_FULL_MODEL,
        "dual": ARCH_FULL_MODEL,
        "dual_layer": ARCH_FULL_MODEL,
        "full_model": ARCH_FULL_MODEL,
        "encoder": ARCH_ENCODER_ONLY,
        "encoder_only": ARCH_ENCODER_ONLY,
        "non_reuse": ARCH_ENCODER_ONLY,
    }
    if key in alias:
        return alias[key]
    raise ValueError(f"Unsupported architecture: {arch!r}. Supported={list(VALID_ARCHITECTURES)}")


def _parse_architectures(s: str) -> List[str]:
    out: List[str] = []
    for raw in _parse_str_csv(s):
        normalized = _normalize_architecture(raw)
        if normalized not in out:
            out.append(normalized)
    return out


def _fixed_batch_policy(model_name: str) -> Dict[str, int]:
    name = str(model_name or "").lower()
    if "32b" in name:
        train_bs = 1
    elif "14b" in name:
        train_bs = 1
    elif "8b" in name:
        train_bs = 16
    elif "7b" in name:
        train_bs = 4
    else:
        train_bs = 16
    eval_bs = train_bs
    target_eff = 32
    grad_acc = max(1, (target_eff + train_bs - 1) // train_bs)
    return {
        "train_batch_size": int(train_bs),
        "eval_batch_size": int(eval_bs),
        "target_effective_batch": int(target_eff),
        "grad_acc_steps": int(grad_acc),
    }


def _extract_results_json(stdout: str) -> Optional[Dict[str, Any]]:
    out = None
    for line in stdout.splitlines():
        if line.startswith("ABLATION_RESULTS_JSON:"):
            payload = line.split("ABLATION_RESULTS_JSON:", 1)[1].strip()
            try:
                out = json.loads(payload)
            except Exception:
                continue
    return out


def _read_eval_loss_from_checkpoint(checkpoint_path: str) -> Optional[float]:
    """
    Try to read eval_loss from eval_metrics.json produced by results_saver.
    Returns float (possibly nan) or None if file not found/parse fails.
    """
    if not checkpoint_path:
        return None
    p = Path(checkpoint_path)
    if p.is_file():
        p = p.parent
    candidates = []
    direct = p / "eval_metrics.json"
    if direct.exists():
        candidates.append(direct)
    else:
        # Fallback for older layouts (e.g., checkpoint-acc*/eval_metrics.json)
        try:
            candidates.extend(list(p.rglob("eval_metrics.json"))[:3])
        except Exception:
            candidates = []
    for c in candidates:
        try:
            data = json.loads(c.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
            loss = metrics.get("eval_loss")
            if loss is None:
                continue
            return float(loss)
        except Exception:
            continue
    return None


def _ablation_env_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    drop = {"name", "description", "expected_improvement"}
    return {k: v for k, v in cfg.items() if k not in drop}


@dataclass(frozen=True)
class ExpSpec:
    dataset_choice: int
    dataset_tag: str
    seed: int
    mode: str
    architecture: str
    model_name: str
    use_dual_layer: bool
    config_name_env: str
    ablation_config: Optional[Dict[str, Any]]


class Logger:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.log_file = out_dir / "qwen_training_modes.log.txt"

    def log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M:%S UTC]")
        line = f"{ts} {msg}"
        print(line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _should_live_log(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    live_markers = [
        "🚀 Training started",
        "📍 Epoch",
        "Evaluation Results",
        "New best accuracy",
        "Best model saved to",
        "===== Evaluation Complete =====",
        "Accuracy:",
        "Macro F1:",
    ]
    return any(m in s for m in live_markers)


def _run_and_stream(
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    logger: Logger,
    *,
    heartbeat_sec: int = 120,
) -> Dict[str, Any]:
    """
    Run subprocess and stream merged stdout/stderr in real time.
    Returns dict with returncode/stdout/stderr (stderr empty because merged).
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: List[str] = []
    last_heartbeat = time.time()
    start = time.time()

    with open(stdout_path, "w", encoding="utf-8") as f_out:
        # Merge stderr into stdout to avoid blocking and provide live progress.
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            all_lines.append(line)
            f_out.write(line)
            f_out.flush()

            if _should_live_log(line):
                logger.log(f"  [LIVE] {line.strip()}")

            now = time.time()
            if now - last_heartbeat >= heartbeat_sec:
                elapsed = now - start
                logger.log(f"  [LIVE] running... elapsed={elapsed/60.0:.1f} min")
                last_heartbeat = now

        returncode = proc.wait()

    # Keep stderr file for compatibility; merged stream is in stdout.
    stderr_path.write_text("", encoding="utf-8")
    return {
        "returncode": int(returncode),
        "stdout": "".join(all_lines),
        "stderr": "",
    }


def run_one(
    spec: ExpSpec,
    logger: Logger,
    out_dir: Path,
    *,
    config_profile: str,
    force: bool,
    test_mode: bool,
    robust_latency: bool,
    warmup: int,
    runs: int,
    qlora_r: int,
    qlora_alpha: int,
    qlora_dropout: float,
    qlora_compute_dtype: str,
    qlora_quant_type: str,
    qlora_double_quant: bool,
    qlora_target_modules: List[str],
) -> Dict[str, Any]:
    model_sfx = _model_suffix(spec.model_name)
    run_id = (
        f"qwen_mode__ds{spec.dataset_choice}_{_sanitize(spec.dataset_tag)}__"
        f"{model_sfx}__{_sanitize(spec.config_name_env)}__seed{spec.seed}"
    )
    out_json = out_dir / f"{run_id}.json"
    out_stdout = out_dir / "stdout" / f"{run_id}.stdout.txt"
    out_stderr = out_dir / "stderr" / f"{run_id}.stderr.txt"
    out_stdout.parent.mkdir(parents=True, exist_ok=True)
    out_stderr.parent.mkdir(parents=True, exist_ok=True)

    if out_json.exists() and not force:
        logger.log(f"SKIP (exists): {out_json.name}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    cmd = ["python", "scripts/train.py"]
    if spec.use_dual_layer:
        cmd.append("--dual-layer")

    env = os.environ.copy()
    env["DATASET_CHOICE"] = str(spec.dataset_choice)
    env["MODEL_NAME"] = spec.model_name
    env["RANDOM_SEED"] = str(spec.seed)
    env["PYTHONHASHSEED"] = str(spec.seed)
    env["LANG_CODE"] = _sanitize(f"qwen_mode_{spec.dataset_tag}__{model_sfx}")
    env["CONFIG_NAME"] = spec.config_name_env
    env["FINETUNE_MODE"] = spec.mode
    env["TRAIN_CONFIG_PROFILE"] = config_profile
    env["QWEN_SAVE_CHECKPOINTS"] = "1"
    env["QWEN_LOAD_BEST_MODEL"] = "1"
    # Keep training setup deterministic and independent from shell exports.
    batch_cfg = _fixed_batch_policy(spec.model_name)
    env["TRAIN_BATCH_SIZE"] = str(batch_cfg["train_batch_size"])
    env["EVAL_BATCH_SIZE"] = str(batch_cfg["eval_batch_size"])
    env["TARGET_EFFECTIVE_BATCH"] = str(batch_cfg["target_effective_batch"])
    env["GRAD_ACC_STEPS"] = str(batch_cfg["grad_acc_steps"])
    env["BF16"] = "1"
    env["FP16"] = "0"

    if test_mode:
        env["USE_TEST_MODE"] = "1"
    if robust_latency:
        env["WARMUP_COUNT"] = str(int(warmup))
        env["ROBUST_INFERENCE_COUNT"] = str(int(runs))
    if spec.use_dual_layer and spec.ablation_config is not None:
        env["ABLATION_CONFIG"] = json.dumps(spec.ablation_config, ensure_ascii=False)

    if spec.mode == "qlora":
        env["QLORA_R"] = str(int(qlora_r))
        env["QLORA_ALPHA"] = str(int(qlora_alpha))
        env["QLORA_DROPOUT"] = str(float(qlora_dropout))
        env["QLORA_COMPUTE_DTYPE"] = qlora_compute_dtype
        env["QLORA_QUANT_TYPE"] = qlora_quant_type
        env["QLORA_DOUBLE_QUANT"] = "1" if qlora_double_quant else "0"
        if qlora_target_modules:
            env["QLORA_TARGET_MODULES"] = ",".join(qlora_target_modules)

    logger.log(f"RUN {run_id}")
    logger.log(f"  CMD: {' '.join(cmd)}")
    logger.log(
        f"  ENV: DATASET_CHOICE={spec.dataset_choice} MODEL_NAME={spec.model_name} "
        f"SEED={spec.seed} FINETUNE_MODE={spec.mode} ARCH={spec.architecture} CONFIG_NAME={spec.config_name_env} "
        f"TRAIN_CONFIG_PROFILE={config_profile}"
    )
    logger.log(
        f"  OPT: TRAIN_BATCH_SIZE={env.get('TRAIN_BATCH_SIZE')} EVAL_BATCH_SIZE={env.get('EVAL_BATCH_SIZE')} "
        f"GRAD_ACC_STEPS={env.get('GRAD_ACC_STEPS')} TARGET_EFFECTIVE_BATCH={env.get('TARGET_EFFECTIVE_BATCH')} "
        f"BF16={env.get('BF16')} FP16={env.get('FP16')}"
    )
    if spec.mode == "qlora":
        logger.log(
            f"  QLORA: r={qlora_r} alpha={qlora_alpha} dropout={qlora_dropout} "
            f"dtype={qlora_compute_dtype} quant={qlora_quant_type} double_quant={qlora_double_quant}"
        )

    run_ret = _run_and_stream(
        cmd=cmd,
        env=env,
        cwd=PROJECT_ROOT,
        stdout_path=out_stdout,
        stderr_path=out_stderr,
        logger=logger,
    )

    parsed = _extract_results_json(run_ret["stdout"] or "")
    ok = (run_ret["returncode"] == 0) and (parsed is not None)

    experiment = asdict(spec)
    if spec.mode == "qlora":
        experiment["qlora_hparams"] = {
            "r": int(qlora_r),
            "alpha": int(qlora_alpha),
            "dropout": float(qlora_dropout),
            "compute_dtype": str(qlora_compute_dtype),
            "quant_type": str(qlora_quant_type),
            "double_quant": bool(qlora_double_quant),
            "target_modules": list(qlora_target_modules) if qlora_target_modules else [],
        }

    result: Dict[str, Any] = {
        "ok": bool(ok),
        "returncode": int(run_ret["returncode"]),
        "run_id": run_id,
        "experiment": experiment,
        "stdout_path": str(out_stdout),
        "stderr_path": str(out_stderr),
        "parsed_results": parsed,
    }
    if parsed:
        result["accuracy"] = parsed.get("accuracy")
        result["macro_f1"] = parsed.get("macro_f1")
        result["finetune_mode"] = parsed.get("finetune_mode")
        result["config_profile"] = parsed.get("config_profile")
        perf = parsed.get("performance_metrics", {}) or {}
        result["latency_p50_ms"] = perf.get("latency_p50_ms")
        result["latency_p95_ms"] = perf.get("latency_p95_ms")
        result["training_time_s"] = perf.get("training_time_s")
        mm = parsed.get("model_metrics", {}) or {}
        result["total_params"] = mm.get("total_params")
        result["trainable_params"] = mm.get("trainable_params")
        if mm.get("total_params"):
            result["trainable_ratio"] = float(mm.get("trainable_params", 0)) / float(mm["total_params"])
        result["checkpoint_path"] = parsed.get("checkpoint_path")

    # Validate training via eval_loss when available.
    if result.get("checkpoint_path"):
        eval_loss = _read_eval_loss_from_checkpoint(str(result["checkpoint_path"]))
        result["eval_loss"] = eval_loss
        if eval_loss is not None and (not math.isfinite(float(eval_loss))):
            # Treat as invalid training run even if process exit code was 0.
            result["ok"] = False
            result["error"] = "Non-finite eval_loss (NaN/Inf) detected in eval_metrics.json"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if ok:
        logger.log(
            f"  OK: mode={spec.mode} arch={spec.architecture} MF1={result.get('macro_f1'):.4f} "
            f"Acc={result.get('accuracy'):.4f} train_s={result.get('training_time_s'):.1f}"
        )
    else:
        extra = ""
        if result.get("error"):
            extra = f" error={result['error']}"
        logger.log(f"  FAIL: returncode={run_ret['returncode']} parsed_json={parsed is not None}{extra}")
    return result


def build_specs(
    *,
    model_names: List[str],
    dataset_choices: List[int],
    seeds: List[int],
    modes: List[str],
    architectures: List[str],
    force_dual_layer: bool,
) -> List[ExpSpec]:
    if not model_names:
        raise ValueError("No model names provided.")
    selected_architectures = [ARCH_FULL_MODEL] if force_dual_layer else list(architectures)
    if not selected_architectures:
        raise ValueError("No architectures provided.")
    full_cfg = _ablation_env_from_config(PHASE_A_CONFIGS["full_model"])

    specs: List[ExpSpec] = []
    for model_name in model_names:
        for ds in dataset_choices:
            tag = DATASET_ID_TO_TAG.get(ds, f"Dataset-{ds}")
            for seed in seeds:
                for mode in modes:
                    for arch in selected_architectures:
                        if arch == ARCH_FULL_MODEL:
                            use_dual_layer = True
                            ablation = dict(full_cfg)
                        elif arch == ARCH_ENCODER_ONLY:
                            use_dual_layer = False
                            ablation = None
                        else:
                            raise ValueError(f"Unsupported architecture: {arch}")
                        cfg_name = f"{mode}__{arch}"
                        specs.append(
                            ExpSpec(
                                dataset_choice=ds,
                                dataset_tag=tag,
                                seed=seed,
                                mode=mode,
                                architecture=arch,
                                model_name=model_name,
                                use_dual_layer=use_dual_layer,
                                config_name_env=cfg_name,
                                ablation_config=ablation,
                            )
                        )
    return specs


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        out = float(v)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _extract_path_after_marker(text: str, marker: str) -> Optional[str]:
    out = None
    for line in text.splitlines():
        if marker in line:
            out = line.split(marker, 1)[1].strip()
    return out


def _build_encoder_only_non_reuse_linear_records(
    *,
    spec: ExpSpec,
    run_result: Dict[str, Any],
    max_aspects: int,
) -> List[Dict[str, Any]]:
    base_p50 = _to_float_or_none(run_result.get("latency_p50_ms"))
    base_p95 = _to_float_or_none(run_result.get("latency_p95_ms"))
    if base_p50 is None or base_p95 is None:
        return []
    rows: List[Dict[str, Any]] = []
    for m in range(1, int(max_aspects) + 1):
        rows.append(
            {
                "run_id": run_result.get("run_id"),
                "dataset_choice": spec.dataset_choice,
                "dataset_tag": spec.dataset_tag,
                "seed": spec.seed,
                "mode": spec.mode,
                "architecture": spec.architecture,
                "model_name": spec.model_name,
                "num_aspects": int(m),
                "reuse_p50_ms": None,
                "reuse_p95_ms": None,
                "non_reuse_p50_ms": float(base_p50) * float(m),
                "non_reuse_p95_ms": float(base_p95) * float(m),
                "p50_speedup": None,
                "p95_speedup": None,
                "benchmark_source": "encoder_only_non_reuse_linear_estimate",
                "benchmark_source_csv": None,
                "estimate_note": "non_reuse(M)=M*single_aspect_latency",
            }
        )
    return rows


def _build_encoder_only_synthetic_sentences(
    *,
    seed: int,
    num_sentences: int,
    max_aspects: int,
) -> List[Dict[str, Any]]:
    import random

    rng = random.Random(int(seed))
    products = ["phone", "laptop", "restaurant", "hotel", "camera", "headphones", "tablet", "car"]
    aspects = [
        "battery", "screen", "performance", "design", "price", "service", "quality", "sound",
        "comfort", "speed", "display", "build", "features", "software", "hardware", "keyboard",
        "touchpad", "menu", "staff", "location",
    ]
    sentiments = ["great", "poor", "excellent", "terrible", "outstanding", "disappointing", "perfect", "awful"]
    if max_aspects > len(aspects):
        raise ValueError(f"max_aspects={max_aspects} exceeds aspect pool size={len(aspects)}")

    def one_record(aspect_count: int) -> Dict[str, Any]:
        product = rng.choice(products)
        chosen_aspects = rng.sample(aspects, int(aspect_count))
        chosen_sentiments = rng.choices(sentiments, k=int(aspect_count))
        clauses = [f"the {a} is {s}" for a, s in zip(chosen_aspects, chosen_sentiments)]
        if len(clauses) == 1:
            sent = f"For this {product}, {clauses[0]}."
        else:
            sent = f"For this {product}, " + ", ".join(clauses[:-1]) + f", and {clauses[-1]}."
        return {"sentence": sent, "aspect_terms": chosen_aspects}

    rows: List[Dict[str, Any]] = []
    per_m = int(num_sentences) // int(max_aspects)
    for m in range(1, int(max_aspects) + 1):
        for _ in range(per_m):
            rows.append(one_record(m))
    remaining = int(num_sentences) - len(rows)
    for _ in range(max(0, remaining)):
        rows.append(one_record(rng.randint(1, int(max_aspects))))
    return rows


def _sample_encoder_only_pool(
    *,
    records: List[Dict[str, Any]],
    pool_size: int,
    fixed_pool: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    import random

    if not records:
        return []
    if len(records) <= int(pool_size):
        return records
    if fixed_pool:
        rng = random.Random(int(seed))
        return rng.sample(records, int(pool_size))
    return random.sample(records, int(pool_size))


def _encoder_only_aspect_text(sentence: str, aspect_term: str) -> str:
    return f"{sentence} [SEP] aspect: {aspect_term}"


def _run_encoder_only_non_reuse_measured(
    *,
    spec: ExpSpec,
    out_dir: Path,
    logger: Logger,
    force: bool,
    max_aspects: int,
    num_sentences: int,
    num_runs: int,
    repeat_per_sentence: int,
    fixed_pool: bool,
    max_length: int,
    device: str,
    qlora_r: int,
    qlora_alpha: int,
    qlora_dropout: float,
    qlora_compute_dtype: str,
    qlora_quant_type: str,
    qlora_double_quant: bool,
    qlora_target_modules: List[str],
) -> List[Dict[str, Any]]:
    raw_dir = out_dir / "encoder_only_non_reuse_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{_sanitize(f'{spec.mode}_{spec.dataset_tag}_seed{spec.seed}')}_m1to{int(max_aspects)}.csv"
    if cache_path.exists() and not force:
        try:
            cached = pd.read_csv(cache_path)
            out_rows: List[Dict[str, Any]] = []
            for _, r in cached.iterrows():
                rec = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
                out_rows.append(rec)
            if out_rows:
                logger.log(f"  [LATENCY] reuse cached encoder_only measured: {cache_path.name} rows={len(out_rows)}")
                return out_rows
        except Exception as e:
            logger.log(f"  [LATENCY] cache read failed ({cache_path.name}): {e}")

    try:
        import numpy as np
        import torch
        from src.core.model import load_tokenizer, load_bert_model, move_model_to_device
    except Exception as e:
        logger.log(f"  [LATENCY] encoder_only measured unavailable (import error): {e}")
        return []

    logger.log(
        f"  [LATENCY] RUN encoder_only measured M=1..{int(max_aspects)} mode={spec.mode} "
        f"ds{spec.dataset_choice} seed={spec.seed}"
    )

    try:
        tokenizer = load_tokenizer(spec.model_name)
        model = load_bert_model(
            model_name=spec.model_name,
            num_labels=3,
            use_dual_layer=False,
            finetune_mode=spec.mode,
            qlora_compute_dtype=qlora_compute_dtype,
            qlora_quant_type=qlora_quant_type,
            qlora_use_double_quant=qlora_double_quant,
            qlora_r=qlora_r,
            qlora_alpha=qlora_alpha,
            qlora_dropout=qlora_dropout,
            qlora_target_modules=qlora_target_modules if qlora_target_modules else None,
        )
        model, model_device = move_model_to_device(model)
        model.eval()
        if device:
            try:
                forced_device = torch.device(device)
                model = model.to(forced_device)
                model_device = forced_device
            except Exception as e:
                logger.log(f"  [LATENCY] ignore --reuse-device={device!r} for encoder_only benchmark: {e}")

        samples = _build_encoder_only_synthetic_sentences(
            seed=spec.seed,
            num_sentences=int(num_sentences),
            max_aspects=int(max_aspects),
        )

        measured_rows: List[Dict[str, Any]] = []
        for m in range(1, int(max_aspects) + 1):
            eligible = [r for r in samples if len(r["aspect_terms"]) >= m]
            subset = _sample_encoder_only_pool(
                records=eligible,
                pool_size=int(num_runs),
                fixed_pool=bool(fixed_pool),
                seed=spec.seed + 10007 * m,
            )
            if not subset:
                continue

            all_times_s: List[float] = []
            for sent in subset:
                sentence = str(sent["sentence"])
                aspect_terms = list(sent["aspect_terms"])[:m]
                encoded_inputs: List[Dict[str, Any]] = []
                for aspect_term in aspect_terms:
                    text = _encoder_only_aspect_text(sentence, str(aspect_term))
                    enc = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=int(max_length),
                        padding=True,
                    )
                    allowed = ("input_ids", "attention_mask", "token_type_ids")
                    model_inputs = {
                        key: value.to(model_device)
                        for key, value in enc.items()
                        if key in allowed
                    }
                    encoded_inputs.append(model_inputs)

                for rep in range(int(repeat_per_sentence)):
                    do_warmup = rep == 0
                    if do_warmup:
                        with torch.no_grad():
                            for model_inputs in encoded_inputs:
                                _ = model(**model_inputs)
                        if torch.cuda.is_available() and model_device.type == "cuda":
                            torch.cuda.synchronize(model_device)

                    if torch.cuda.is_available() and model_device.type == "cuda":
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(model_device)
                        start_event.record()
                        with torch.no_grad():
                            for model_inputs in encoded_inputs:
                                _ = model(**model_inputs)
                        end_event.record()
                        torch.cuda.synchronize(model_device)
                        elapsed_s = float(start_event.elapsed_time(end_event)) / 1000.0
                    else:
                        start = time.perf_counter()
                        with torch.no_grad():
                            for model_inputs in encoded_inputs:
                                _ = model(**model_inputs)
                        elapsed_s = time.perf_counter() - start

                    all_times_s.append(float(elapsed_s))

            if not all_times_s:
                continue
            p50_ms = float(np.percentile(all_times_s, 50) * 1000.0)
            p95_ms = float(np.percentile(all_times_s, 95) * 1000.0)
            measured_rows.append(
                {
                    "run_id": f"encoder_only__{_sanitize(spec.dataset_tag)}__{_sanitize(spec.mode)}__seed{spec.seed}",
                    "dataset_choice": spec.dataset_choice,
                    "dataset_tag": spec.dataset_tag,
                    "seed": spec.seed,
                    "mode": spec.mode,
                    "architecture": spec.architecture,
                    "model_name": spec.model_name,
                    "num_aspects": int(m),
                    "reuse_p50_ms": None,
                    "reuse_p95_ms": None,
                    "non_reuse_p50_ms": p50_ms,
                    "non_reuse_p95_ms": p95_ms,
                    "p50_speedup": None,
                    "p95_speedup": None,
                    "benchmark_source": "encoder_only_non_reuse_measured",
                    "benchmark_source_csv": str(cache_path),
                    "estimate_note": None,
                }
            )

        if measured_rows:
            pd.DataFrame(measured_rows).to_csv(cache_path, index=False)
            logger.log(f"  [LATENCY] encoder_only measured saved: {cache_path}")
        return measured_rows
    except Exception as e:
        logger.log(f"  [LATENCY] encoder_only measured failed: {e}")
        return []
    finally:
        try:
            del model
        except Exception:
            pass
        try:
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def run_reuse_latency_benchmark(
    *,
    spec: ExpSpec,
    run_result: Dict[str, Any],
    out_dir: Path,
    logger: Logger,
    force: bool,
    max_aspects: int,
    num_sentences: int,
    num_runs: int,
    repeat_per_sentence: int,
    bootstrap_b: int,
    ci_level: float,
    fixed_pool: bool,
    device: str,
    torch_dtype: str,
    max_length: int,
    qlora_r: int,
    qlora_alpha: int,
    qlora_dropout: float,
    qlora_compute_dtype: str,
    qlora_quant_type: str,
    qlora_double_quant: bool,
    qlora_target_modules: List[str],
) -> List[Dict[str, Any]]:
    if not run_result.get("ok"):
        return []
    if spec.architecture == ARCH_ENCODER_ONLY:
        linear_rows = _build_encoder_only_non_reuse_linear_records(
            spec=spec,
            run_result=run_result,
            max_aspects=max_aspects,
        )
        measured_rows = _run_encoder_only_non_reuse_measured(
            spec=spec,
            out_dir=out_dir,
            logger=logger,
            force=force,
            max_aspects=max_aspects,
            num_sentences=num_sentences,
            num_runs=num_runs,
            repeat_per_sentence=repeat_per_sentence,
            fixed_pool=fixed_pool,
            max_length=max_length,
            device=device,
            qlora_r=qlora_r,
            qlora_alpha=qlora_alpha,
            qlora_dropout=qlora_dropout,
            qlora_compute_dtype=qlora_compute_dtype,
            qlora_quant_type=qlora_quant_type,
            qlora_double_quant=qlora_double_quant,
            qlora_target_modules=qlora_target_modules,
        )
        if not linear_rows:
            logger.log(
                f"  [LATENCY] skip encoder_only non-reuse estimate (missing p50/p95): {run_result.get('run_id')}"
            )
        return measured_rows + linear_rows

    if spec.architecture != ARCH_FULL_MODEL:
        return []

    run_id = str(run_result.get("run_id") or f"ds{spec.dataset_choice}_seed{spec.seed}_{spec.mode}")
    bench_dir = out_dir / "reuse_latency_raw"
    bench_dir.mkdir(parents=True, exist_ok=True)
    output_tag = _sanitize(f"{run_id}__m1to{max_aspects}")
    existing = sorted(
        bench_dir.glob(f"dora_reuse_benchmark_*_{output_tag}.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if existing and not force:
        csv_path = existing[-1]
    else:
        cmd = [
            "python",
            "scripts/multi_aspect_reuse_benchmark.py",
            "--model-name",
            str(spec.model_name),
            "--seed",
            str(spec.seed),
            "--max-aspects",
            str(int(max_aspects)),
            "--num-sentences",
            str(int(num_sentences)),
            "--num-runs",
            str(int(num_runs)),
            "--repeat-per-sentence",
            str(int(repeat_per_sentence)),
            "--bootstrap-b",
            str(int(bootstrap_b)),
            "--ci-level",
            str(float(ci_level)),
            "--max-length",
            str(int(max_length)),
            "--results-dir",
            str(bench_dir),
            "--output-tag",
            output_tag,
        ]
        if spec.ablation_config and spec.ablation_config.get("k_value") is not None:
            cmd.extend(["--k-value", str(int(spec.ablation_config["k_value"]))])
        if not fixed_pool:
            cmd.append("--no-fixed-pool")
        if device:
            cmd.extend(["--device", str(device)])
        if torch_dtype:
            cmd.extend(["--torch-dtype", str(torch_dtype)])

        out_stdout = out_dir / "reuse_latency_stdout" / f"{run_id}.stdout.txt"
        out_stderr = out_dir / "reuse_latency_stderr" / f"{run_id}.stderr.txt"
        logger.log(f"  [LATENCY] RUN {' '.join(cmd)}")
        ret = _run_and_stream(
            cmd=cmd,
            env=os.environ.copy(),
            cwd=PROJECT_ROOT,
            stdout_path=out_stdout,
            stderr_path=out_stderr,
            logger=logger,
        )
        if ret["returncode"] != 0:
            logger.log(f"  [LATENCY] FAIL returncode={ret['returncode']} run_id={run_id}")
            return []
        csv_hint = _extract_path_after_marker(ret.get("stdout", ""), "CSV data saved to:")
        csv_path = Path(csv_hint) if csv_hint else None
        if csv_path is None or (not csv_path.exists()):
            refreshed = sorted(
                bench_dir.glob(f"dora_reuse_benchmark_*_{output_tag}.csv"),
                key=lambda p: p.stat().st_mtime,
            )
            if not refreshed:
                logger.log(f"  [LATENCY] FAIL missing CSV for run_id={run_id}")
                return []
            csv_path = refreshed[-1]

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.log(f"  [LATENCY] FAIL read_csv={csv_path}: {e}")
        return []

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        m = int(row.get("num_aspects", 0))
        if m < 1 or m > int(max_aspects):
            continue
        rows.append(
            {
                "run_id": run_id,
                "dataset_choice": spec.dataset_choice,
                "dataset_tag": spec.dataset_tag,
                "seed": spec.seed,
                "mode": spec.mode,
                "architecture": spec.architecture,
                "model_name": spec.model_name,
                "num_aspects": int(m),
                "reuse_p50_ms": _to_float_or_none(row.get("reuse_p50_latency_ms")),
                "reuse_p95_ms": _to_float_or_none(row.get("reuse_p95_latency_ms")),
                "non_reuse_p50_ms": _to_float_or_none(row.get("non_reuse_p50_latency_ms")),
                "non_reuse_p95_ms": _to_float_or_none(row.get("non_reuse_p95_latency_ms")),
                "p50_speedup": _to_float_or_none(row.get("p50_speedup")),
                "p95_speedup": _to_float_or_none(row.get("p95_speedup")),
                "benchmark_source": "multi_aspect_reuse_benchmark",
                "benchmark_source_csv": str(csv_path),
                "estimate_note": None,
            }
        )
    if rows:
        logger.log(
            f"  [LATENCY] OK run_id={run_id} rows={len(rows)} source={csv_path.name}"
        )
    return rows


def _fmt_cell(v: Any, ndigits: int = 2) -> str:
    fv = _to_float_or_none(v)
    if fv is None:
        return "-"
    return f"{fv:.{ndigits}f}"


def write_latency_tables(
    *,
    latency_rows: List[Dict[str, Any]],
    out_dir: Path,
    max_aspects: int,
) -> Dict[str, Path]:
    if not latency_rows:
        return {}
    df = pd.DataFrame(latency_rows)
    records_csv = out_dir / "qwen_reuse_vs_non_reuse_records.csv"
    df.to_csv(records_csv, index=False)

    table_rows: List[Dict[str, Any]] = []
    md_lines: List[str] = []
    md_lines.append(f"# Reuse vs Non-reuse Latency (M=1..{int(max_aspects)})")
    md_lines.append("")

    group_cols = ["model_name", "dataset_choice", "dataset_tag", "mode", "seed"]
    grouped = df.groupby(group_cols, dropna=False)
    for (model_name, ds_choice, ds_tag, mode, seed), g in grouped:
        md_lines.append(f"## model={model_name} | ds{int(ds_choice)} {ds_tag} | mode={mode} | seed={int(seed)}")
        md_lines.append("")
        md_lines.append(
            "| M | Full Reuse p50 | Full Non-reuse p50 | Full p50 Speedup | Encoder-only Non-reuse p50 (measured) | Encoder-only Non-reuse p50 (linear) | Full Reuse p95 | Full Non-reuse p95 | Encoder-only Non-reuse p95 (measured) | Encoder-only Non-reuse p95 (linear) |"
        )
        md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in range(1, int(max_aspects) + 1):
            full = g[
                (g["architecture"] == ARCH_FULL_MODEL)
                & (g["num_aspects"] == m)
                & (g["benchmark_source"] == "multi_aspect_reuse_benchmark")
            ]
            enc_measured = g[
                (g["architecture"] == ARCH_ENCODER_ONLY)
                & (g["num_aspects"] == m)
                & (g["benchmark_source"] == "encoder_only_non_reuse_measured")
            ]
            enc_linear = g[
                (g["architecture"] == ARCH_ENCODER_ONLY)
                & (g["num_aspects"] == m)
                & (g["benchmark_source"] == "encoder_only_non_reuse_linear_estimate")
            ]
            full_row = full.iloc[0].to_dict() if not full.empty else {}
            enc_measured_row = enc_measured.iloc[0].to_dict() if not enc_measured.empty else {}
            enc_linear_row = enc_linear.iloc[0].to_dict() if not enc_linear.empty else {}
            table_rows.append(
                {
                    "model_name": model_name,
                    "dataset_choice": int(ds_choice),
                    "dataset_tag": ds_tag,
                    "mode": mode,
                    "seed": int(seed),
                    "num_aspects": int(m),
                    "full_reuse_p50_ms": _to_float_or_none(full_row.get("reuse_p50_ms")),
                    "full_non_reuse_p50_ms": _to_float_or_none(full_row.get("non_reuse_p50_ms")),
                    "full_p50_speedup": _to_float_or_none(full_row.get("p50_speedup")),
                    "encoder_only_non_reuse_p50_ms_measured": _to_float_or_none(
                        enc_measured_row.get("non_reuse_p50_ms")
                    ),
                    "encoder_only_non_reuse_p50_ms_linear": _to_float_or_none(
                        enc_linear_row.get("non_reuse_p50_ms")
                    ),
                    "full_reuse_p95_ms": _to_float_or_none(full_row.get("reuse_p95_ms")),
                    "full_non_reuse_p95_ms": _to_float_or_none(full_row.get("non_reuse_p95_ms")),
                    "encoder_only_non_reuse_p95_ms_measured": _to_float_or_none(
                        enc_measured_row.get("non_reuse_p95_ms")
                    ),
                    "encoder_only_non_reuse_p95_ms_linear": _to_float_or_none(
                        enc_linear_row.get("non_reuse_p95_ms")
                    ),
                }
            )
            md_lines.append(
                "| "
                f"{m} | "
                f"{_fmt_cell(full_row.get('reuse_p50_ms'))} | "
                f"{_fmt_cell(full_row.get('non_reuse_p50_ms'))} | "
                f"{_fmt_cell(full_row.get('p50_speedup'))}x | "
                f"{_fmt_cell(enc_measured_row.get('non_reuse_p50_ms'))} | "
                f"{_fmt_cell(enc_linear_row.get('non_reuse_p50_ms'))} | "
                f"{_fmt_cell(full_row.get('reuse_p95_ms'))} | "
                f"{_fmt_cell(full_row.get('non_reuse_p95_ms'))} | "
                f"{_fmt_cell(enc_measured_row.get('non_reuse_p95_ms'))} | "
                f"{_fmt_cell(enc_linear_row.get('non_reuse_p95_ms'))} |"
            )
        md_lines.append("")

    table_csv = out_dir / "qwen_reuse_vs_non_reuse_table.csv"
    pd.DataFrame(table_rows).to_csv(table_csv, index=False)
    table_md = out_dir / "qwen_reuse_vs_non_reuse_table.md"
    table_md.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    return {
        "records_csv": records_csv,
        "table_csv": table_csv,
        "table_md": table_md,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run freeze_backbone and QLoRA experiments for Qwen.")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B", help="Backbone model name.")
    p.add_argument(
        "--model-names",
        default="",
        help="Optional comma-separated backbone model names. When set, overrides --model-name.",
    )
    p.add_argument("--datasets", default="2,3,4,5,6,7,8", help="Comma-separated DATASET_CHOICE ids.")
    p.add_argument("--seeds", default="42", help="Comma-separated seeds.")
    p.add_argument("--modes", default="freeze_backbone,qlora", help="Comma-separated modes.")
    p.add_argument(
        "--architectures",
        default="full_model,encoder_only",
        help="Comma-separated architectures: full_model, encoder_only.",
    )
    p.add_argument("--dual-layer", action="store_true", help="Force only full_model architecture.")
    p.add_argument("--results-dir", default="", help="Optional explicit output directory.")
    p.add_argument(
        "--config-profile",
        default="qwen",
        help="Config profile name passed via TRAIN_CONFIG_PROFILE (default: qwen).",
    )
    p.add_argument("--force", action="store_true", help="Re-run existing per-run JSON.")
    p.add_argument("--test", action="store_true", help="Enable USE_TEST_MODE=1.")
    p.add_argument("--robust-latency", action="store_true", help="Set robust latency env vars.")
    p.add_argument("--warmup", type=int, default=20, help="Warmup runs for latency.")
    p.add_argument("--runs", type=int, default=200, help="Measured runs for latency.")
    p.add_argument(
        "--reuse-latency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run reuse vs non-reuse (M=1..4) latency recording.",
    )
    p.add_argument("--reuse-max-aspects", type=int, default=4, help="Largest M for reuse latency table.")
    p.add_argument("--reuse-num-sentences", type=int, default=120, help="Synthetic sentence count for reuse benchmark.")
    p.add_argument("--reuse-num-runs", type=int, default=30, help="Sentence pool size per M.")
    p.add_argument("--reuse-repeat-per-sentence", type=int, default=1, help="Repeat count per sentence in reuse benchmark.")
    p.add_argument("--reuse-bootstrap-b", type=int, default=300, help="Bootstrap rounds for reuse benchmark CIs.")
    p.add_argument("--reuse-ci-level", type=float, default=0.95, help="CI level for reuse benchmark.")
    p.add_argument(
        "--reuse-fixed-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic sentence pool in reuse benchmark.",
    )
    p.add_argument("--reuse-device", default="", help='Device for reuse benchmark, e.g. "cuda:0".')
    p.add_argument(
        "--reuse-torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for reuse benchmark.",
    )
    p.add_argument("--reuse-max-length", type=int, default=128, help="Tokenizer max length for reuse benchmark.")
    p.add_argument("--dry-run", action="store_true", help="Print planned runs and exit.")
    # QLoRA options
    p.add_argument("--qlora-r", type=int, default=16)
    p.add_argument("--qlora-alpha", type=int, default=32)
    p.add_argument("--qlora-dropout", type=float, default=0.05)
    p.add_argument("--qlora-compute-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--qlora-quant-type", default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--no-qlora-double-quant", action="store_true")
    p.add_argument("--qlora-target-modules", default="", help="Optional comma-separated LoRA target modules.")
    args = p.parse_args()

    model_names = _parse_model_names(args.model_name, args.model_names)
    dataset_choices = _parse_int_csv(args.datasets)
    seeds = _parse_int_csv(args.seeds)
    modes = _parse_str_csv(args.modes)
    architectures = _parse_architectures(args.architectures)
    qlora_target_modules = _parse_str_csv(args.qlora_target_modules)

    if not model_names:
        raise ValueError("No model names provided.")
    if not dataset_choices:
        raise ValueError("No datasets provided.")
    if not seeds:
        raise ValueError("No seeds provided.")
    if not modes:
        raise ValueError("No modes provided.")
    if not architectures and not args.dual_layer:
        raise ValueError("No architectures provided.")
    if int(args.reuse_max_aspects) < 1:
        raise ValueError("--reuse-max-aspects must be >= 1")
    if int(args.reuse_num_sentences) < 1:
        raise ValueError("--reuse-num-sentences must be >= 1")
    if int(args.reuse_num_runs) < 1:
        raise ValueError("--reuse-num-runs must be >= 1")
    if int(args.reuse_repeat_per_sentence) < 1:
        raise ValueError("--reuse-repeat-per-sentence must be >= 1")
    if int(args.reuse_bootstrap_b) < 0:
        raise ValueError("--reuse-bootstrap-b must be >= 0")
    if not (0.0 < float(args.reuse_ci_level) < 1.0):
        raise ValueError("--reuse-ci-level must be in (0,1)")
    if int(args.reuse_max_length) < 8:
        raise ValueError("--reuse-max-length must be >= 8")

    out_dir = Path(args.results_dir) if args.results_dir else (PROJECT_ROOT / "results" / f"qwen_train_modes_{_utc_ts()}")
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)

    specs = build_specs(
        model_names=model_names,
        dataset_choices=dataset_choices,
        seeds=seeds,
        modes=modes,
        architectures=architectures,
        force_dual_layer=bool(args.dual_layer),
    )

    logger.log("===== Qwen Training Modes =====")
    logger.log(f"results_dir: {out_dir}")
    if len(model_names) == 1:
        logger.log(f"model_name: {model_names[0]}")
    else:
        logger.log(f"model_names: {model_names}")
    logger.log(f"datasets: {dataset_choices}")
    logger.log(f"seeds: {seeds}")
    logger.log(f"modes: {modes}")
    logger.log(f"architectures: {[ARCH_FULL_MODEL] if args.dual_layer else architectures}")
    logger.log(f"dual_layer: {args.dual_layer}")
    logger.log(f"reuse_latency: {args.reuse_latency}")
    logger.log(f"config_profile: {args.config_profile}")
    logger.log(f"total_runs: {len(specs)}")
    logger.log("==============================")

    if args.dry_run:
        for s in specs:
            logger.log(f"DRY: {s}")
        return

    all_results: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs, 1):
        logger.log(f"\n--- [{idx}/{len(specs)}] ---")
        res = run_one(
            spec=spec,
            logger=logger,
            out_dir=out_dir,
            config_profile=str(args.config_profile),
            force=bool(args.force),
            test_mode=bool(args.test),
            robust_latency=bool(args.robust_latency),
            warmup=int(args.warmup),
            runs=int(args.runs),
            qlora_r=int(args.qlora_r),
            qlora_alpha=int(args.qlora_alpha),
            qlora_dropout=float(args.qlora_dropout),
            qlora_compute_dtype=args.qlora_compute_dtype,
            qlora_quant_type=args.qlora_quant_type,
            qlora_double_quant=not bool(args.no_qlora_double_quant),
            qlora_target_modules=qlora_target_modules,
        )
        all_results.append(res)
        if args.reuse_latency:
            bench_rows = run_reuse_latency_benchmark(
                spec=spec,
                run_result=res,
                out_dir=out_dir,
                logger=logger,
                force=bool(args.force),
                max_aspects=int(args.reuse_max_aspects),
                num_sentences=int(args.reuse_num_sentences),
                num_runs=int(args.reuse_num_runs),
                repeat_per_sentence=int(args.reuse_repeat_per_sentence),
                bootstrap_b=int(args.reuse_bootstrap_b),
                ci_level=float(args.reuse_ci_level),
                fixed_pool=bool(args.reuse_fixed_pool),
                device=str(args.reuse_device or "").strip(),
                torch_dtype=str(args.reuse_torch_dtype),
                max_length=int(args.reuse_max_length),
                qlora_r=int(args.qlora_r),
                qlora_alpha=int(args.qlora_alpha),
                qlora_dropout=float(args.qlora_dropout),
                qlora_compute_dtype=str(args.qlora_compute_dtype),
                qlora_quant_type=str(args.qlora_quant_type),
                qlora_double_quant=not bool(args.no_qlora_double_quant),
                qlora_target_modules=qlora_target_modules,
            )
            latency_rows.extend(bench_rows)

    latency_artifacts = write_latency_tables(
        latency_rows=latency_rows,
        out_dir=out_dir,
        max_aspects=int(args.reuse_max_aspects),
    )

    summary = {
        "schema_version": "1.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "total_runs": len(all_results),
        "ok_runs": int(sum(1 for r in all_results if r.get("ok"))),
        "failed_runs": int(sum(1 for r in all_results if not r.get("ok"))),
        "latency_rows": int(len(latency_rows)),
        "latency_artifacts": {k: str(v) for k, v in latency_artifacts.items()},
        "results": all_results,
    }
    summary_path = out_dir / "qwen_training_modes_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"Summary saved: {summary_path}")

    rows: List[Dict[str, Any]] = []
    for r in all_results:
        exp = r.get("experiment", {})
        rows.append(
            {
                "ok": r.get("ok"),
                "dataset_choice": exp.get("dataset_choice"),
                "dataset_tag": exp.get("dataset_tag"),
                "seed": exp.get("seed"),
                "mode": exp.get("mode"),
                "architecture": exp.get("architecture"),
                "model_name": exp.get("model_name"),
                "dual_layer": exp.get("use_dual_layer"),
                "macro_f1": r.get("macro_f1"),
                "accuracy": r.get("accuracy"),
                "training_time_s": r.get("training_time_s"),
                "latency_p50_ms": r.get("latency_p50_ms"),
                "latency_p95_ms": r.get("latency_p95_ms"),
                "total_params": r.get("total_params"),
                "trainable_params": r.get("trainable_params"),
                "trainable_ratio": r.get("trainable_ratio"),
                "checkpoint_path": r.get("checkpoint_path"),
                "run_id": r.get("run_id"),
            }
        )
    compact = pd.DataFrame(rows)
    compact_path = out_dir / "qwen_training_modes_compact.csv"
    compact.to_csv(compact_path, index=False)
    logger.log(f"Compact CSV saved: {compact_path}")
    if latency_artifacts:
        for key, path in latency_artifacts.items():
            logger.log(f"Latency artifact [{key}]: {path}")


if __name__ == "__main__":
    main()
