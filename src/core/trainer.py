#!/usr/bin/env python3
# src/core/trainer.py
# -*- coding: utf-8 -*-

"""
Training and evaluation module.
Contains training logic, metrics computation, and custom loss functions.
"""

import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from transformers.trainer_callback import ProgressCallback
from copy import deepcopy
import torch
import torch.nn.functional as F

# Silence sklearn warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from .. import config
import logging
from ..utils import CleanProgressCallback, QuietProgressCallback
from ..utils.results_saver import save_experiment_results

logger = logging.getLogger(__name__)


class EpochMetricsCallback(TrainerCallback):
    """Record training and validation metrics for each epoch."""
    def __init__(self):
        self.epoch_logs = []

    def on_epoch_end(self, args, state, control, **kwargs):
        # Record the current epoch summary.
        epoch_log = {
            "epoch": int(state.epoch),
            "train_loss": state.log_history[-1].get("loss") if state.log_history else None,
        }
        self.epoch_logs.append(epoch_log)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Attach validation metrics to the latest epoch entry.
        if self.epoch_logs and metrics:
            self.epoch_logs[-1].update({
                "dev_loss": metrics.get("eval_loss"),
                "dev_acc": metrics.get("eval_accuracy"),
                "dev_f1": metrics.get("eval_mf1"),
            })

class FocalLossTrainer(Trainer):
    def __init__(self, gamma=2.0, alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        if alpha is not None:
            # alpha may be a list/ndarray/torch.Tensor.
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        logger.info(f"⚡  Using FocalLossTrainer | gamma={gamma} | alpha={alpha}")
    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **unused          # Capture future Trainer kwargs such as num_items_in_batch.
    ):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        ce = F.cross_entropy(
            logits,
            labels,
            reduction="none",
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
        )
        p_t  = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma * ce).mean()

        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    """
    Metric function used by Trainer.evaluate().
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    # Use zero_division to avoid warnings.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    # Also compute macro-F1.
    _, _, mf1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'mf1': mf1
    }

def create_training_args(output_dir=None, seed=None):
    """
    Build HuggingFace TrainingArguments.
    Trainer-behavior options are sourced from config.TRAINER_ARGS.
    """
    trainer_kwargs = deepcopy(config.TRAINER_ARGS)

    # Override the output directory when one is provided.
    # This allows external scripts to control the output path dynamically.
    if output_dir:
        trainer_kwargs['output_dir'] = output_dir

    return TrainingArguments(
        num_train_epochs            = config.NUM_EPOCHS,
        per_device_train_batch_size = config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = config.EVAL_BATCH_SIZE,
        weight_decay                = config.WEIGHT_DECAY,
        seed                        = seed if seed is not None else 42,
        report_to                   = [] if not config.USE_WANDB else ["wandb"],
        **trainer_kwargs            # Unpack the remaining settings, including logging_steps.
    )

def create_trainer(model, training_args, train_dataset, eval_dataset, compute_metrics_fn):
    """
    Create the HuggingFace Trainer instance.
    """
    TrainerClass = FocalLossTrainer if config.USE_FOCAL_LOSS else Trainer

    # ---------- Build the callback list ----------
    callbacks = [CleanProgressCallback() if config.VERBOSE_TRAINING else QuietProgressCallback()]

    # Add the epoch-metrics callback.
    callbacks.append(EpochMetricsCallback())

    # Add early stopping when enabled.
    if config.USE_EARLY_STOPPING:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD
        )
        callbacks.append(early_stopping_callback)
        logger.info(f"⏹️  Early stopping enabled: patience={config.EARLY_STOPPING_PATIENCE}, threshold={config.EARLY_STOPPING_THRESHOLD}")

    # ---------- Build kwargs ----------
    kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    if config.USE_FOCAL_LOSS:          # Only pass these when focal loss is enabled.
        kwargs.update(gamma=config.FOCAL_GAMMA,
                      alpha=config.FOCAL_ALPHA)
    # ---------------------------------

    trainer = TrainerClass(**kwargs)

    # Remove the plain built-in ProgressCallback so only our custom postfix-aware
    # progress callback controls the training bar.
    for callback in list(trainer.callback_handler.callbacks):
        if type(callback) is ProgressCallback:
            trainer.remove_callback(callback)

    return trainer

def run_training_with_results_saving(
    trainer,
    eval_dataset,
    model_name="DeBERTa",
    dataset_name="ABSA",
    custom_eval_fn=None,
):
    """
    Run training and automatically save detailed outputs to the final checkpoint directory.
    """
    # Locate the epoch-metrics callback.
    epoch_callback = None
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, EpochMetricsCallback):
            epoch_callback = callback
            break

    # Train the model.
    trainer.train()

    # load_best_model_at_end=True already restores the best model automatically.
    # No manual checkpoint reload is required.

    # Evaluate and collect predictions from the best model.
    if custom_eval_fn is None:
        eval_results = trainer.evaluate()
        predictions = trainer.predict(eval_dataset)
        y_true = predictions.label_ids
        y_pred = np.argmax(predictions.predictions, axis=1)
    else:
        eval_results, y_true, y_pred = custom_eval_fn(trainer.model, eval_dataset)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

    # Attach epoch-level logs.
    if epoch_callback:
        eval_results['epoch_logs'] = epoch_callback.epoch_logs
    
    # Collect gate/token statistics on the best model when supported.
    gate_statistics = None
    token_statistics = None
    if hasattr(trainer.model, 'compute_gate_statistics'):
        try:
            # Run a small sample through the model to collect statistics.
            import torch
            eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
            sample_inputs = next(iter(eval_dataloader))
            
            # Move inputs to the target device.
            device = next(trainer.model.parameters()).device
            sample_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in sample_inputs.items()}
            
            # Run a forward pass and capture statistics.
            trainer.model.eval()
            with torch.no_grad():
                outputs = trainer.model(**sample_inputs)
                if 'gate_statistics' in outputs:
                    gate_statistics = outputs['gate_statistics']
                if 'token_statistics' in outputs:
                    token_statistics = outputs['token_statistics']
                    print(f"📊 Collected token statistics: {token_statistics}")
        except Exception as e:
            print(f"⚠️ Failed to collect gating statistics: {e}")
    
    # Add the collected statistics to eval_results for downstream use.
    if gate_statistics:
        eval_results['gate_statistics'] = gate_statistics
    if token_statistics:
        eval_results['token_statistics'] = token_statistics
    
    # Read the final renamed output directory from the callback.
    final_dir = None
    for callback in trainer.callback_handler.callbacks:
        if hasattr(callback, 'final_output_dir') and callback.final_output_dir:
            final_dir = callback.final_output_dir
            break
    
    if final_dir is None:
        final_dir = trainer.args.output_dir
    
    # Save results into the final output directory.
    save_experiment_results(
        output_dir=final_dir,
        metrics=eval_results,
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        model_name=model_name,
        dataset_name=dataset_name,
        additional_info={
            "total_steps": trainer.state.global_step,
            "total_epochs": trainer.state.epoch
        }
    )
    
    return trainer, eval_results, final_dir

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
