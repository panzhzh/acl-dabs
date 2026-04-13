#!/usr/bin/env python3
# scripts/train.py
# -*- coding: utf-8 -*-

"""
Single experiment training script for ABSA.
Run a single training experiment with specified configuration.
"""

import os
import sys
import torch
import random
import numpy as np
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from src package
from src import config
from src.core.data import (
    read_dataset,
    preprocess_and_filter,
    tokenize_texts,
    tokenize_texts_with_aspect_positions,
    convert_token_positions_to_tokenizer_positions,
    TweetsDataset,
)
from src.core.model import load_tokenizer, load_bert_model, move_model_to_device
from src.core.trainer import compute_metrics, create_training_args, create_trainer, run_training_with_results_saving


def measure_inference_latency(model, sample, device, n_warmup: int = 10, n_runs: int = 50):
    """Measure single-sample inference latency (ms) with fixed seq_len and bs=1"""
    import time
    model.eval()
    
    # Determine which input keys are supported by the current model type.
    is_dual_layer = hasattr(model, 'dual_layer_config') or hasattr(model, 'compute_gate_statistics')
    
    # Base keys supported by all models.
    basic_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    
    # Extra keys supported by the dual-layer model.
    dual_layer_keys = ['aspect_token_positions', 'aspect_mask']
    
    # Final set of keys to keep.
    if is_dual_layer:
        allowed_keys = basic_keys + dual_layer_keys
    else:
        allowed_keys = basic_keys
    
    # Ensure tensors on device and bs=1
    inputs = {}
    for k, v in sample.items():
        if k in ('labels',):  # Skip labels.
            continue
        if k not in allowed_keys:  # Keep only supported keys.
            continue
        if hasattr(v, 'unsqueeze'):
            t = v.unsqueeze(0) if v.dim() == 1 else v
            inputs[k] = t.to(device)
    # Warm-up
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000.0)
    times_sorted = sorted(times)
    mean_ms = sum(times) / len(times)
    
    # Compute simple percentile estimates from the sorted latency list.
    n = len(times_sorted)
    p50_idx = max(0, min(n-1, n//2))
    p95_idx = max(0, min(n-1, int(n*0.95)))
    
    p50 = times_sorted[p50_idx]
    p95 = times_sorted[p95_idx]
    
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    
    print(f"    Latency stats ({n_runs} runs): Mean={mean_ms:.2f}ms, P50={p50:.2f}ms, P95={p95:.2f}ms")
    
    return {
        'inference_latency_ms': mean_ms,
        'latency_p50_ms': p50,
        'latency_p95_ms': p95,
        'throughput_sps': throughput
    }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_contiguous_sentence_groups(df):
    """
    Group consecutive aspect rows that originate from the same source sentence.
    This preserves official aspect-instance order while enabling sentence-level reuse.
    """
    groups = []
    current_group = []
    previous_key = None

    for _, row in df.reset_index(drop=True).iterrows():
        token_field = row['token']
        token_key = tuple(token_field) if isinstance(token_field, list) else token_field
        current_key = (row['clean_text'], token_key)
        if previous_key is None or current_key != previous_key:
            if current_group:
                groups.append(current_group)
            current_group = []
            previous_key = current_key
        current_group.append(row)

    if current_group:
        groups.append(current_group)

    return groups


def run_reuse_sentence_grouped_eval(model, tokenizer, eval_df, max_length=128):
    """
    Final evaluation only: reuse one shared sentence encoding for all aspects in that sentence.
    """
    if not hasattr(model, 'encode_shared') or not hasattr(model, 'forward_aspect_from_shared'):
        raise AttributeError("Model does not implement reuse evaluation helpers.")

    device = next(model.parameters()).device
    grouped_rows = _build_contiguous_sentence_groups(eval_df)
    logits_all = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for sentence_rows in grouped_rows:
            sentence_text = sentence_rows[0]['clean_text']
            encoding = tokenizer(
                [sentence_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            shared_inputs = {k: v.to(device) for k, v in encoding.items()}
            shared_output = model.encode_shared(**shared_inputs)
            seq_len = int(shared_inputs['input_ids'].shape[1])

            for row in sentence_rows:
                converted_pos = None
                if (
                    row.get('from') is not None
                    and row.get('to') is not None
                    and isinstance(row.get('token'), list)
                    and len(row['token']) > 0
                ):
                    converted_pos = convert_token_positions_to_tokenizer_positions(
                        tokenizer,
                        row['token'],
                        row['from'],
                        row['to'],
                        sentence_text,
                    )

                aspect_positions = None
                aspect_mask = torch.zeros(1, seq_len, device=device, dtype=torch.long)
                if converted_pos is not None:
                    start, end = converted_pos
                    start = max(0, min(int(start), seq_len))
                    end = max(start, min(int(end), seq_len))
                    if end > start:
                        aspect_positions = [(start, end)]
                        aspect_mask[0, start:end] = 1

                outputs = model.forward_aspect_from_shared(
                    shared_output,
                    aspect_token_positions=aspect_positions,
                    aspect_mask=aspect_mask,
                )
                logits_all.append(outputs['logits'].detach().cpu())
                y_true.append(int(row['label']))

    logits_np = torch.cat(logits_all, dim=0).numpy()
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.argmax(logits_np, axis=1)

    raw_metrics = compute_metrics(
        SimpleNamespace(predictions=logits_np, label_ids=y_true_np)
    )
    eval_results = {f"eval_{k}": float(v) for k, v in raw_metrics.items()}
    eval_results["eval_samples"] = int(len(y_true_np))
    eval_results["eval_mode_reuse"] = 1.0

    return eval_results, y_true_np, y_pred_np

def main(best_hp: dict | None = None, use_dual_layer=False):
    # Get the seed from the environment or fall back to the default.
    seed = int(os.environ.get('RANDOM_SEED', 42))
    set_seed(seed)
    print(f"🎲 Random seed: {seed}")
    print(f"🧩 Config profile: {getattr(config, 'ACTIVE_CONFIG_PROFILE', 'default')}")
    print(f"🗂️ Dataset config: {getattr(config, 'ACTIVE_DATASET_CONFIG', 'default')}")
    
    # Check whether test mode is enabled.
    use_test_mode = os.environ.get('USE_TEST_MODE') == '1'
    if use_test_mode:
        print("⚡️ [TEST MODE] Training and evaluation will use 1% of the dataset.")
        
    # ========== 1. Read and preprocess data ==========
    df_train, df_test = read_dataset(config.DATASET_CHOICE, test_mode=use_test_mode)
    
    # Preprocess the datasets.
    df_train = preprocess_and_filter(df_train)
    df_test = preprocess_and_filter(df_test)

    # Print basic dataset previews.
    print("===== Training Samples =====")
    print(df_train.head())
    print("\n===== Test Samples =====")
    print(df_test.head())

    X_train = df_train['clean_text']
    y_train = df_train['label']
    X_test = df_test['clean_text']
    y_test = df_test['label']

    X_test_aspect = df_test['aspect_term']
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    class_counts = np.bincount(y_train, minlength=config.NUM_LABELS)
    inv_freq     = 1.0 / class_counts
    alpha = (inv_freq / inv_freq.sum()).tolist()
    config.FOCAL_ALPHA = alpha

    counter = Counter(y_train)
    print("\n[Class Counts in the Training Set]")
    for lbl in range(config.NUM_LABELS):
        print(f"  label {lbl:>2} -> {counter[lbl]} samples")

    print(f"\nalpha (Focal Loss) = {alpha}")      

    # ========== 2. Load tokenizer and model ==========
    # The model name only needs to be swapped here.
    tokenizer = load_tokenizer(
        config.MODEL_NAME
        # No extra special tokens are used in the simplified setup.
    )
    # Check for ablation configuration from environment
    ablation_config = {}
    if use_dual_layer and 'ABLATION_CONFIG' in os.environ:
        try:
            ablation_config = json.loads(os.environ['ABLATION_CONFIG'])
            print(f"🔬 Using ablation config: {ablation_config}")
        except json.JSONDecodeError:
            print("⚠️  Failed to parse ablation config; using defaults")
    
    # Remove use_dual_layer from ablation_config because the CLI flag already controls it.
    if 'use_dual_layer' in ablation_config:
        del ablation_config['use_dual_layer']

    # Fine-tuning mode for large backbones:
    #   full / freeze_backbone / qlora
    # Prefer config-profile defaults (e.g. qwen) when FINETUNE_MODE is not explicitly set.
    finetune_mode = os.environ.get(
        'FINETUNE_MODE',
        getattr(config, 'FINETUNE_MODE', 'full')
    ).strip().lower()
    qlora_compute_dtype = os.environ.get('QLORA_COMPUTE_DTYPE', 'bfloat16')
    qlora_quant_type = os.environ.get('QLORA_QUANT_TYPE', 'nf4')
    qlora_use_double_quant = os.environ.get('QLORA_DOUBLE_QUANT', '1') not in ('0', 'false', 'False')
    qlora_r = int(os.environ.get('QLORA_R', '16'))
    qlora_alpha = int(os.environ.get('QLORA_ALPHA', '32'))
    qlora_dropout = float(os.environ.get('QLORA_DROPOUT', '0.05'))
    qlora_target_modules_env = os.environ.get('QLORA_TARGET_MODULES', '').strip()
    qlora_target_modules = None
    if qlora_target_modules_env:
        qlora_target_modules = [x.strip() for x in qlora_target_modules_env.split(',') if x.strip()]

    print(f"🎛️ Fine-tuning mode: {finetune_mode}")
    if finetune_mode == 'qlora':
        print(
            f"   QLoRA: r={qlora_r}, alpha={qlora_alpha}, dropout={qlora_dropout}, "
            f"compute_dtype={qlora_compute_dtype}, quant={qlora_quant_type}, double_quant={qlora_use_double_quant}"
        )
        if qlora_target_modules:
            print(f"   QLoRA target modules: {qlora_target_modules}")

    model = load_bert_model(
        config.MODEL_NAME, 
        num_labels=config.NUM_LABELS,
        use_dual_layer=use_dual_layer,
        finetune_mode=finetune_mode,
        qlora_compute_dtype=qlora_compute_dtype,
        qlora_quant_type=qlora_quant_type,
        qlora_use_double_quant=qlora_use_double_quant,
        qlora_r=qlora_r,
        qlora_alpha=qlora_alpha,
        qlora_dropout=qlora_dropout,
        qlora_target_modules=qlora_target_modules,
        **ablation_config
    )

    # Qwen-like models require a valid pad_token_id when batch size > 1.
    if tokenizer.pad_token_id is not None:
        try:
            if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = int(tokenizer.pad_token_id)
            if hasattr(model, "backbone") and hasattr(model.backbone, "config"):
                if getattr(model.backbone.config, "pad_token_id", None) is None:
                    model.backbone.config.pad_token_id = int(tokenizer.pad_token_id)
            if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
                if getattr(model.base_model.config, "pad_token_id", None) is None:
                    model.base_model.config.pad_token_id = int(tokenizer.pad_token_id)
        except Exception as e:
            print(f"⚠️ Failed to set pad_token_id: {e}")

    # No resize step is needed because there are no extra tokens.

    # ========== 3. Build datasets and trainer ==========
    if use_dual_layer:
        # Detect pure DORA-only settings with no ACBS path, which do not depend on aspect inputs.
        is_pure_dora = (
            ablation_config.get('enable_multi_scale', False) or ablation_config.get('enable_inter_gru', False)
        ) and not (
            ablation_config.get('enable_aspect_attention', False) or
            ablation_config.get('enable_context_importance', False) or
            ablation_config.get('enable_layer_attention', False)
        )

        # For dual-layer model, extract aspect positions unless pure DORA-only
        print("📍 Extracting aspect token positions for the dual-layer architecture...")
        if is_pure_dora:
            train_encodings = tokenize_texts(tokenizer, X_train)
            test_encodings = tokenize_texts(tokenizer, X_test)
            train_dataset = TweetsDataset(train_encodings, y_train.tolist())
            test_dataset  = TweetsDataset(test_encodings, y_test.tolist())
        else:
            train_encodings, train_aspect_positions = tokenize_texts_with_aspect_positions(
                tokenizer, X_train, df_train
            )
            test_encodings, test_aspect_positions = tokenize_texts_with_aspect_positions(
                tokenizer, X_test, df_test
            )

            # Log how many aspect positions were extracted successfully.
            valid_train_positions = sum(1 for pos in train_aspect_positions if pos is not None)
            valid_test_positions = sum(1 for pos in test_aspect_positions if pos is not None)
            print(f"   Train set: {valid_train_positions}/{len(train_aspect_positions)} samples with extracted positions")
            print(f"   Test set: {valid_test_positions}/{len(test_aspect_positions)} samples with extracted positions")

            train_dataset = TweetsDataset(train_encodings, y_train.tolist(), train_aspect_positions)
            test_dataset  = TweetsDataset(test_encodings, y_test.tolist(), test_aspect_positions)
    else:
        # For standard model, use simple tokenization
        train_encodings = tokenize_texts(tokenizer, X_train)
        test_encodings = tokenize_texts(tokenizer, X_test)
        
        train_dataset = TweetsDataset(train_encodings, y_train.tolist())
        test_dataset  = TweetsDataset(test_encodings, y_test.tolist())

    # --- Build a unique output directory ---
    output_rel_dir = os.environ.get('OUTPUT_REL_DIR', '').strip()
    if output_rel_dir:
        # Custom relative output path, for example: fullmodel/Restaurant-15
        output_dir = config.OUTPUT_DIR / Path(output_rel_dir) / f"seed_{seed}"
    else:
        lang_code = os.environ.get('LANG_CODE', 'unknown')
        config_name_from_env = os.environ.get('CONFIG_NAME', 'default')
        
        # Format: outputs/multilingual/{lang}/{config_name}/seed_{seed}
        output_dir = (
            config.OUTPUT_DIR / 
            "multilingual" /
            lang_code /
            config_name_from_env /
            f"seed_{seed}"
        )
    
    training_args = create_training_args(output_dir=output_dir, seed=seed)
    model, device = move_model_to_device(model)

    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics_fn=compute_metrics
    )

    # ========== 4. Train and evaluate ==========
    print("\n===== Training Started =====")
    
    # Record training start time
    import time
    training_start_time = time.time()
    
    # Use the new training wrapper to automatically save detailed outputs.
    dataset_name = f"Dataset-{config.DATASET_CHOICE}"
    model_name = config.MODEL_NAME.split('/')[-1]
    
    custom_eval_fn = None
    if use_dual_layer and not is_pure_dora:
        custom_eval_fn = lambda model, _eval_dataset: run_reuse_sentence_grouped_eval(
            model=model,
            tokenizer=tokenizer,
            eval_df=df_test,
        )
        print("🔁 Final evaluation will use the sentence-level reuse path (training and early stopping stay unchanged)")

    trainer, metrics, checkpoint_path = run_training_with_results_saving(
        trainer=trainer,
        eval_dataset=test_dataset,
        model_name=model_name,
        dataset_name=dataset_name,
        custom_eval_fn=custom_eval_fn,
    )
    
    # Record training end time
    training_end_time = time.time()
    actual_training_time = training_end_time - training_start_time
    
    # Print the best-model path so upstream scripts can parse it.
    if not checkpoint_path:
        checkpoint_path = trainer.args.output_dir
    print(f"Best model saved to {checkpoint_path}")
    
    print(f"\n===== Evaluation Complete =====")
    print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Macro F1: {metrics['eval_mf1']:.4f}")
    
    # Collect model parameters and computational metrics
    model_metrics = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    # Calculate added parameters (difference from base model)
    base_params = 183245312  # Approximate base DeBERTa parameters
    model_metrics["added_params"] = max(0, model_metrics["total_params"] - base_params)
    
    # Real FLOPs measurement using thop
    try:
        from thop import profile
        
        # Get a sample from test dataset for FLOPs measurement
        sample_input = test_dataset[0]
        model.eval()
        
        # Prepare inputs for thop (uses tuple format)
        input_ids = sample_input['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample_input['attention_mask'].unsqueeze(0).to(device)
        
        # Add aspect positions if dual-layer model
        if hasattr(model, 'dual_layer_config') or hasattr(model, 'compute_gate_statistics'):
            if 'aspect_positions' in sample_input:
                aspect_positions = sample_input['aspect_positions'].unsqueeze(0).to(device)
                input_tuple = (input_ids, attention_mask, aspect_positions)
            else:
                input_tuple = (input_ids, attention_mask)
        else:
            input_tuple = (input_ids, attention_mask)
        
        # Measure actual FLOPs
        with torch.no_grad():
            flops, params = profile(model, inputs=input_tuple, verbose=False)
        
        model_metrics["flops_G_per_sample"] = flops / 1e9
        model_metrics["flops_context"] = {
            "measurement_method": "thop_actual",
            "input_shape": str(input_ids.shape),
            "device": str(device)
        }
        
    except Exception as e:
        print(f"⚠️ FLOPs measurement failed: {e}")
        model_metrics["flops_G_per_sample"] = None
        model_metrics["flops_context"] = {"error": str(e)}
    
    # Real parameter breakdown by module
    try:
        encoder_params = 0
        head_params = 0
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            # Classify parameters by module name
            if any(keyword in name.lower() for keyword in ['deberta', 'bert', 'encoder', 'embeddings', 'attention', 'intermediate', 'output']):
                # Base model parameters (encoder)
                if not any(dual_keyword in name.lower() for dual_keyword in ['dual', 'gate', 'aspect', 'context', 'layer_attention']):
                    encoder_params += param_count
                else:
                    # Dual-layer specific parameters
                    head_params += param_count
            else:
                # Classification head and other components
                head_params += param_count
        
        model_metrics["encoder_params"] = encoder_params
        model_metrics["head_params"] = head_params
        
    except Exception as e:
        print(f"⚠️ Parameter breakdown failed: {e}")
        model_metrics["encoder_params"] = None
        model_metrics["head_params"] = None
    
    # Calculate efficiency metrics
    training_time = actual_training_time
    # Real inference latency on a single sample
    n_warmup = int(os.environ.get('WARMUP_COUNT', 10))
    n_runs = int(os.environ.get('ROBUST_INFERENCE_COUNT', 50))
    try:
        sample0 = test_dataset[0]
        # Read inference settings from environment variables.
        perf = measure_inference_latency(model, sample0, device, n_warmup=n_warmup, n_runs=n_runs)
        latency_ms = perf['inference_latency_ms']
        p50_ms = perf['latency_p50_ms']
        p95_ms = perf['latency_p95_ms']
        throughput_sps = perf['throughput_sps']
    except Exception as e:
        print(f"⚠️ Inference latency measurement failed: {e}")
        latency_ms, p50_ms, p95_ms, throughput_sps = None, None, None, None
    
    # Pull statistics returned by the trainer (already collected on the best model).
    gate_statistics = metrics.get('gate_statistics')
    token_statistics = metrics.get('token_statistics')
    epoch_logs = metrics.get('epoch_logs', [])

    finetune_info = {"mode": finetune_mode}
    if finetune_mode == 'qlora':
        finetune_info["qlora"] = {
            "r": qlora_r,
            "alpha": qlora_alpha,
            "dropout": qlora_dropout,
            "compute_dtype": qlora_compute_dtype,
            "quant_type": qlora_quant_type,
            "double_quant": qlora_use_double_quant,
            "target_modules": qlora_target_modules,
        }

    # Output results in JSON format for ablation script parsing
    result_json = {
        "accuracy": metrics['eval_accuracy'],
        "macro_f1": metrics['eval_mf1'],
        "checkpoint_path": checkpoint_path,
        "finetune_mode": finetune_mode,
        "finetune": finetune_info,
        "config_profile": getattr(config, 'ACTIVE_CONFIG_PROFILE', 'default'),
        "epoch_logs": epoch_logs,
        "model_metrics": model_metrics,
        "performance_metrics": {
            "training_time_s": training_time,
            "inference_latency_ms": latency_ms,
            "latency_p50_ms": p50_ms,
            "latency_p95_ms": p95_ms,
            "throughput_sps": throughput_sps
        },
        "latency_settings": {"warmup": n_warmup, "runs": n_runs},
    }
    
    # Add gate statistics if available
    if gate_statistics:
        result_json["gate_statistics"] = gate_statistics
    if token_statistics:
        result_json["token_statistics"] = token_statistics
    print(f"ABLATION_RESULTS_JSON: {json.dumps(result_json)}")
    
    print("\n=== Program Finished ===")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ABSA model training')
    parser.add_argument('--dual-layer', action='store_true', 
                       help='Use the dual-layer architecture (Dual-Layer Aspect-Aware Architecture)')
    
    args = parser.parse_args()
    
    main(use_dual_layer=args.dual_layer)
