#!/usr/bin/env python3
# src/core/callbacks.py
# -*- coding: utf-8 -*-

"""
Custom training callbacks for better logging and progress display.
"""

import json
import time
import os
import shutil
from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def _metric_key(metric_name):
    if not metric_name:
        return "eval_mf1"
    return metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"


def _metric_tag(metric_name):
    metric = _metric_key(metric_name).replace("eval_", "")
    if metric == "accuracy":
        return "acc"
    return metric


def _metric_label(metric_name):
    tag = _metric_tag(metric_name)
    if tag == "mf1":
        return "MF1"
    if tag == "acc":
        return "accuracy"
    return tag


def _metric_value(logs, metric_name):
    if logs is None:
        return None
    key = _metric_key(metric_name)
    if key in logs:
        return logs[key]
    fallback = key.replace("eval_", "")
    return logs.get(fallback)


def _rename_best_checkpoint(output_dir, metric_name, best_metric_value):
    checkpoint_dirs = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoint_dirs:
        return None

    old_checkpoint = checkpoint_dirs[0]
    metric_tag = _metric_tag(metric_name)
    new_checkpoint = output_dir / f"checkpoint-{metric_tag}-{best_metric_value:.4f}"

    if old_checkpoint == new_checkpoint:
        return new_checkpoint

    if new_checkpoint.exists():
        shutil.rmtree(str(new_checkpoint))

    shutil.move(str(old_checkpoint), str(new_checkpoint))
    return new_checkpoint


def _format_metric(value):
    return f"{value:.4f}" if value is not None else "n/a"


class _BaseProgressCallback(ProgressCallback):
    """Progress callback that keeps eval metrics on the tqdm postfix."""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.best_eval_accuracy = 0.0
        self.best_eval_metric = 0.0
        self.original_output_dir = None
        self.final_output_dir = None
        self.current_epoch = 0
        self.total_epochs = 0

    def _update_postfix(self, args, acc=None, f1=None, improved=False):
        if self.training_bar is None:
            return
        monitored_metric = getattr(args, "metric_for_best_model", None)
        label = _metric_tag(monitored_metric)
        parts = []
        if self.current_epoch and self.total_epochs:
            parts.append(f"epoch={self.current_epoch}/{self.total_epochs}")
        if acc is not None:
            parts.append(f"acc={acc:.4f}")
        if f1 is not None:
            parts.append(f"mf1={f1:.4f}")
        if self.best_eval_metric > 0:
            parts.append(f"best_{label}={self.best_eval_metric:.4f}")
        if improved:
            parts.append("new_best")
        self.training_bar.set_postfix_str(" | ".join(parts), refresh=False)
        self.training_bar.refresh()

    def _finalize_output_dir(self, args, monitored_metric):
        if self.original_output_dir and self.best_eval_metric > 0:
            try:
                output_dir = Path(args.output_dir)
                if output_dir.exists():
                    new_checkpoint = _rename_best_checkpoint(
                        output_dir,
                        monitored_metric,
                        self.best_eval_metric,
                    )
                    if new_checkpoint is not None:
                        self.final_output_dir = str(new_checkpoint)
                        return
                    self.final_output_dir = str(output_dir)
                    return
                self.final_output_dir = args.output_dir
                return
            except Exception as e:
                print(f"⚠️ Could not rename directory: {e}")
                self.final_output_dir = args.output_dir
                return
        self.final_output_dir = args.output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.original_output_dir = args.output_dir
        self.total_epochs = int(args.num_train_epochs)
        super().on_train_begin(args, state, control, **kwargs)
        print("\n🚀 Training started...")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch) + 1
        self._update_postfix(args)

    def on_log(self, args, state, control, logs=None, **kwargs):
        return

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if not logs:
            return

        acc = _metric_value(logs, "eval_accuracy")
        f1 = _metric_value(logs, "eval_mf1")
        if acc is not None:
            self.best_eval_accuracy = max(self.best_eval_accuracy, acc)

        monitored_metric = getattr(args, "metric_for_best_model", None)
        current_metric = _metric_value(logs, monitored_metric)
        improved = False
        if current_metric is not None and current_metric > self.best_eval_metric:
            self.best_eval_metric = current_metric
            improved = True

        self._update_postfix(args, acc=acc, f1=f1, improved=improved)

    def on_save(self, args, state, control, **kwargs):
        return


class CleanProgressCallback(_BaseProgressCallback):
    """Verbose progress callback with postfix-based eval metrics."""

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        print("=" * 60)

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        total_time_str = f"{total_time/60:.1f} minutes" if total_time > 60 else f"{total_time:.0f} seconds"
        monitored_metric = getattr(args, "metric_for_best_model", None)
        best_metric = getattr(state, "best_metric", None)
        if best_metric is not None:
            self.best_eval_metric = best_metric

        self._finalize_output_dir(args, monitored_metric)
        super().on_train_end(args, state, control, **kwargs)

        print("\n" + "=" * 60)
        print("✅ Training completed!")
        print(f"   📈 Total steps: {state.global_step}")
        print(f"   ⏱️  Total time: {total_time_str}")
        print(f"   🏆 Best {_metric_label(monitored_metric)}: {self.best_eval_metric:.4f}")
        if self.best_eval_accuracy > 0:
            print(f"   📏 Best observed accuracy: {self.best_eval_accuracy:.4f}")
        if self.final_output_dir:
            print(f"   📂 Results saved to: {self.final_output_dir}")
        print("=" * 60)


class QuietProgressCallback(_BaseProgressCallback):
    """Quieter progress callback with postfix-based eval metrics."""

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        monitored_metric = getattr(args, "metric_for_best_model", None)
        best_metric = getattr(state, "best_metric", None)
        if best_metric is not None:
            self.best_eval_metric = best_metric

        self._finalize_output_dir(args, monitored_metric)
        super().on_train_end(args, state, control, **kwargs)

        print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
        print(f"🏆 Best {_metric_label(monitored_metric)}: {self.best_eval_metric:.4f}")
        if self.best_eval_accuracy > 0:
            print(f"📏 Best observed accuracy: {self.best_eval_accuracy:.4f}")
        if self.final_output_dir:
            print(f"📂 Results saved to: {self.final_output_dir}")
