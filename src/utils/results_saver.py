#!/usr/bin/env python3
# src/utils/results_saver.py
# -*- coding: utf-8 -*-

"""
Results saving utilities.
Save experiment results and predictions to JSON files.
"""

import json
import datetime
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import Dict, List, Any, Optional

from .. import config


def save_experiment_results(output_dir: str, 
                          metrics: Dict[str, Any],
                          y_true: List[int],
                          y_pred: List[int],
                          model_name: str = "DeBERTa",
                          dataset_name: str = "ABSA",
                          additional_info: Optional[Dict] = None) -> None:
    """
    Save experiment artifacts: evaluation metrics JSON plus predictions JSON.

    Args:
        output_dir: Output directory, typically the checkpoint directory.
        metrics: Evaluation metrics dictionary.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Model name.
        dataset_name: Dataset name.
        additional_info: Optional extra metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Compute detailed evaluation metrics.
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average=None, 
        zero_division=0,
        labels=list(range(config.NUM_LABELS))  # Ensure every class is present in the report.
    )
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )[2]
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[2]
    
    # 1. Save evaluation metrics JSON.
    metrics_results = {
        "metadata": {
            "timestamp": timestamp,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_choice": config.DATASET_CHOICE,
            "dataset_config": getattr(config, "ACTIVE_DATASET_CONFIG", "default"),
            "num_labels": config.NUM_LABELS,
            "train_batch_size": config.TRAIN_BATCH_SIZE,
            "eval_batch_size": config.EVAL_BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "warmup_ratio": config.WARMUP_RATIO,
            "max_grad_norm": config.MAX_GRAD_NORM,
            "num_epochs": config.NUM_EPOCHS,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "use_focal_loss": config.USE_FOCAL_LOSS,
            "total_samples": len(y_true)
        },
        "metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
        },
        "per_class_metrics": {
            config.CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }
            for i in range(len(config.CLASS_NAMES))
        }
    }
    
    if additional_info:
        metrics_results["metadata"].update(additional_info)
    
    # 2. Save prediction results JSON.
    predictions_results = {
        "metadata": {
            "timestamp": timestamp,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "total_samples": len(y_true),
            "class_names": config.CLASS_NAMES
        },
        "predictions": [
            {
                "sample_id": i,
                "true_label": int(y_true[i]),
                "predicted_label": int(y_pred[i]),
                "true_class": config.CLASS_NAMES[y_true[i]],
                "predicted_class": config.CLASS_NAMES[y_pred[i]],
                "correct": bool(y_true[i] == y_pred[i])
            }
            for i in range(len(y_true))
        ]
    }
    
    # Persist the output files.
    metrics_file = output_path / "eval_metrics.json"
    predictions_file = output_path / "predictions.json"
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_results, f, ensure_ascii=False, indent=2)
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 Results saved:")
    print(f"   📄 Metrics: {metrics_file}")
    print(f"   📄 Predictions: {predictions_file}")
