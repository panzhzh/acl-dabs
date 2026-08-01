#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the complete DORA--QCBS model for end-to-end ASTE."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOURNAL_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from journal.dabs_structured.aste.data import (  # noqa: E402
    extract_spans_from_bio,
    read_aste_split,
    score_aste_examples,
)
from journal.dabs_structured.aste.decode import (  # noqa: E402
    decode_aste_examples_batched,
    predictions_at_pair_threshold,
)
from journal.dabs_structured.aste.dataset import (  # noqa: E402
    ASTECollator,
    ASTETrainingDataset,
    IGNORE_INDEX,
    load_aste_tokenizer,
)
from journal.dabs_structured.checkpoint import atomic_torch_save  # noqa: E402
from journal.dabs_structured.model import DABSStructuredModel  # noqa: E402


DEFAULT_ASTE_ROOT = JOURNAL_ROOT / "data" / "aste"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def should_stop_early(
    *,
    early_stop_patience: int,
    min_epochs: int,
    epoch: int,
    stale_epochs: int,
    best_dev_score: float,
    require_positive_selection: bool,
) -> bool:
    """Apply the configured early-stop rule without treating zero F1 as convergence."""

    if (
        early_stop_patience <= 0
        or epoch < min_epochs
        or stale_epochs < early_stop_patience
    ):
        return False
    if require_positive_selection and best_dev_score <= 0.0:
        return False
    return True


def _tensor_batch(batch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    mask = labels != IGNORE_INDEX
    if not bool(mask.any()):
        return 0, 0
    preds = logits.argmax(dim=-1)
    correct = int((preds[mask] == labels[mask]).sum().item())
    total = int(mask.sum().item())
    return correct, total


def _pair_accuracy(logits: torch.Tensor, labels: torch.Tensor, pair_mask: torch.Tensor) -> tuple[int, int]:
    mask = pair_mask & (labels != IGNORE_INDEX)
    if not bool(mask.any()):
        return 0, 0
    preds = logits.argmax(dim=-1)
    correct = int((preds[mask] == labels[mask]).sum().item())
    total = int(mask.sum().item())
    return correct, total


def _pair_positive_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pair_mask: torch.Tensor,
) -> tuple[int, int, int]:
    valid = pair_mask.bool() & (labels != IGNORE_INDEX)
    if not bool(valid.any()):
        return 0, 0, 0
    predictions = logits.argmax(dim=-1)
    gold_positive = (labels != 0) & valid
    predicted_positive = (predictions != 0) & valid
    correct = int(
        ((predictions == labels) & gold_positive & predicted_positive).sum().item()
    )
    return (
        correct,
        int(predicted_positive.sum().item()),
        int(gold_positive.sum().item()),
    )


def _span_proposal_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
    label_index: int,
) -> tuple[int, int, int]:
    valid = mask.bool()
    if not bool(valid.any()):
        return 0, 0, 0
    probs = logits[..., label_index].sigmoid()
    gold = labels > 0.5
    predicted = probs >= float(threshold)
    correct = int((predicted & gold & valid).sum().item())
    predicted_total = int((predicted & valid).sum().item())
    gold_total = int((gold & valid).sum().item())
    return correct, predicted_total, gold_total


def _span_counts_from_bio(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, int]:
    predictions = logits.argmax(dim=-1).detach().cpu()
    labels = labels.detach().cpu()
    correct = 0
    predicted_total = 0
    gold_total = 0
    for pred_row, label_row in zip(predictions, labels):
        valid = label_row != IGNORE_INDEX
        if not bool(valid.any()):
            continue
        pred_spans = {
            span.as_tuple()
            for span in extract_spans_from_bio(pred_row[valid].tolist())
        }
        gold_spans = {
            span.as_tuple()
            for span in extract_spans_from_bio(label_row[valid].tolist())
        }
        correct += len(pred_spans & gold_spans)
        predicted_total += len(pred_spans)
        gold_total += len(gold_spans)
    return correct, predicted_total, gold_total


def _prf(correct: int, predicted: int, gold: int) -> dict[str, float]:
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    *,
    span_proposal_threshold: float,
    precision: str,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss_sum": 0.0,
        "steps": 0,
        "aspect_correct": 0,
        "aspect_total": 0,
        "opinion_correct": 0,
        "opinion_total": 0,
        "pair_correct": 0,
        "pair_total": 0,
        "pair_positive_correct": 0,
        "pair_positive_predicted": 0,
        "pair_positive_gold": 0,
        "span_aspect_correct": 0,
        "span_aspect_predicted": 0,
        "span_aspect_gold": 0,
        "span_opinion_correct": 0,
        "span_opinion_predicted": 0,
        "span_opinion_gold": 0,
        "aspect_span_correct": 0,
        "aspect_span_predicted": 0,
        "aspect_span_gold": 0,
        "opinion_span_correct": 0,
        "opinion_span_predicted": 0,
        "opinion_span_gold": 0,
    }
    use_bf16 = precision == "bf16" and device.type == "cuda"
    for batch in loader:
        tensor_batch = _tensor_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            outputs = model(**tensor_batch)
        totals["loss_sum"] += float(outputs["loss"].detach().cpu())
        totals["steps"] += 1

        correct, total = _token_accuracy(
            outputs["aspect_bio_logits"],
            tensor_batch["aspect_bio_labels"],
        )
        totals["aspect_correct"] += correct
        totals["aspect_total"] += total
        correct, total = _token_accuracy(
            outputs["opinion_bio_logits"],
            tensor_batch["opinion_bio_labels"],
        )
        totals["opinion_correct"] += correct
        totals["opinion_total"] += total
        correct, predicted, gold = _span_counts_from_bio(
            outputs["aspect_bio_logits"],
            tensor_batch["aspect_bio_labels"],
        )
        totals["aspect_span_correct"] += correct
        totals["aspect_span_predicted"] += predicted
        totals["aspect_span_gold"] += gold
        correct, predicted, gold = _span_counts_from_bio(
            outputs["opinion_bio_logits"],
            tensor_batch["opinion_bio_labels"],
        )
        totals["opinion_span_correct"] += correct
        totals["opinion_span_predicted"] += predicted
        totals["opinion_span_gold"] += gold
        correct, total = _pair_accuracy(
            outputs["pair_logits"],
            tensor_batch["pair_labels"],
            tensor_batch["pair_mask"],
        )
        totals["pair_correct"] += correct
        totals["pair_total"] += total
        correct, predicted, gold = _pair_positive_counts(
            outputs["pair_logits"],
            tensor_batch["pair_labels"],
            tensor_batch["pair_mask"],
        )
        totals["pair_positive_correct"] += correct
        totals["pair_positive_predicted"] += predicted
        totals["pair_positive_gold"] += gold
        correct, predicted, gold = _span_proposal_counts(
            outputs["span_proposal_logits"],
            tensor_batch["span_aspect_labels"],
            tensor_batch["span_proposal_mask"],
            threshold=span_proposal_threshold,
            label_index=0,
        )
        totals["span_aspect_correct"] += correct
        totals["span_aspect_predicted"] += predicted
        totals["span_aspect_gold"] += gold
        correct, predicted, gold = _span_proposal_counts(
            outputs["span_proposal_logits"],
            tensor_batch["span_opinion_labels"],
            tensor_batch["span_proposal_mask"],
            threshold=span_proposal_threshold,
            label_index=1,
        )
        totals["span_opinion_correct"] += correct
        totals["span_opinion_predicted"] += predicted
        totals["span_opinion_gold"] += gold

    steps = max(1, totals["steps"])
    aspect_span = _prf(
        totals["aspect_span_correct"],
        totals["aspect_span_predicted"],
        totals["aspect_span_gold"],
    )
    opinion_span = _prf(
        totals["opinion_span_correct"],
        totals["opinion_span_predicted"],
        totals["opinion_span_gold"],
    )
    span_aspect = _prf(
        totals["span_aspect_correct"],
        totals["span_aspect_predicted"],
        totals["span_aspect_gold"],
    )
    span_opinion = _prf(
        totals["span_opinion_correct"],
        totals["span_opinion_predicted"],
        totals["span_opinion_gold"],
    )
    pair_positive = _prf(
        totals["pair_positive_correct"],
        totals["pair_positive_predicted"],
        totals["pair_positive_gold"],
    )
    return {
        "loss": totals["loss_sum"] / steps,
        "aspect_token_acc": totals["aspect_correct"] / totals["aspect_total"]
        if totals["aspect_total"]
        else 0.0,
        "aspect_span_f1": aspect_span["f1"],
        "aspect_span_precision": aspect_span["precision"],
        "aspect_span_recall": aspect_span["recall"],
        "opinion_token_acc": totals["opinion_correct"] / totals["opinion_total"]
        if totals["opinion_total"]
        else 0.0,
        "opinion_span_f1": opinion_span["f1"],
        "opinion_span_precision": opinion_span["precision"],
        "opinion_span_recall": opinion_span["recall"],
        "span_proposal_aspect_f1": span_aspect["f1"],
        "span_proposal_aspect_precision": span_aspect["precision"],
        "span_proposal_aspect_recall": span_aspect["recall"],
        "span_proposal_opinion_f1": span_opinion["f1"],
        "span_proposal_opinion_precision": span_opinion["precision"],
        "span_proposal_opinion_recall": span_opinion["recall"],
        "pair_acc": totals["pair_correct"] / totals["pair_total"]
        if totals["pair_total"]
        else 0.0,
        "pair_positive_f1": pair_positive["f1"],
        "pair_positive_precision": pair_positive["precision"],
        "pair_positive_recall": pair_positive["recall"],
        "pair_total": float(totals["pair_total"]),
    }


def _selection_score(metrics: dict[str, float], *, use_span_proposal: bool) -> float:
    if use_span_proposal:
        aspect_f1 = metrics["span_proposal_aspect_f1"]
        opinion_f1 = metrics["span_proposal_opinion_f1"]
    else:
        aspect_f1 = metrics["aspect_span_f1"]
        opinion_f1 = metrics["opinion_span_f1"]
    return (aspect_f1 + opinion_f1 + metrics["pair_positive_f1"]) / 3.0


@torch.no_grad()
def _end_to_end_selection(
    model,
    tokenizer,
    examples,
    device: torch.device,
    args,
) -> dict[str, float]:
    started = time.time()
    decoded = decode_aste_examples_batched(
        model,
        tokenizer,
        examples,
        batch_size=args.selection_batch_size,
        device=device,
        max_length=args.max_length,
        max_pairs=args.selection_max_pairs,
        pair_confidence_threshold=0.0,
        proposal_source=args.selection_proposal_source,
        max_proposal_span_len=max(1, args.span_proposal_max_len),
        span_proposal_threshold=args.span_proposal_threshold,
        span_proposal_top_k=args.selection_span_proposal_top_k,
        include_bio_proposals=args.selection_include_bio_proposals,
        confidence_mode=args.selection_confidence_mode,
        precision=args.precision,
    )
    rows = []
    for threshold in args.selection_thresholds:
        predictions = predictions_at_pair_threshold(
            examples,
            decoded.diagnostics,
            float(threshold),
        )
        metrics = score_aste_examples(predictions, examples).as_dict()
        rows.append({"threshold": float(threshold), **metrics})
    best = max(
        rows,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            -float(row["threshold"]),
        ),
    )
    return {
        "f1": float(best["f1"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "threshold": float(best["threshold"]),
        "elapsed_s": time.time() - started,
        "pair_count_mean": (
            sum(decoded.pair_counts) / len(decoded.pair_counts)
            if decoded.pair_counts
            else 0.0
        ),
    }


def _optimizer_groups(model: DABSStructuredModel, args) -> list[dict[str, object]]:
    encoder_parameters = []
    head_parameters = []
    classifier_parameters = []
    classifier_markers = (
        "classifier",
        "pair_selector",
        "proposal_depth_output",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            encoder_parameters.append(parameter)
        elif any(marker in name for marker in classifier_markers):
            classifier_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    return [
        {"params": encoder_parameters, "lr": args.encoder_learning_rate},
        {"params": head_parameters, "lr": args.head_learning_rate},
        {"params": classifier_parameters, "lr": args.classifier_learning_rate},
    ]


def train(args) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device.index)
        torch.set_float32_matmul_precision("high")

    train_examples = read_aste_split(args.root, args.dataset, "train")
    dev_examples = read_aste_split(args.root, args.dataset, "dev")
    if args.max_train_examples is not None:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_eval_examples is not None:
        dev_examples = dev_examples[: args.max_eval_examples]

    tokenizer = load_aste_tokenizer(args.model_name)

    train_dataset = ASTETrainingDataset(
        train_examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_pairs=args.max_pairs,
        extra_negative_spans=args.extra_negative_spans,
        span_proposal_max_len=args.span_proposal_max_len,
        max_span_proposals=args.max_span_proposals,
        pair_candidate_source=args.pair_candidate_source,
        pair_candidate_max_span_len=args.pair_candidate_max_span_len,
        pair_negative_strategy=args.pair_negative_strategy,
    )
    dev_dataset = ASTETrainingDataset(
        dev_examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_pairs=args.max_pairs,
        extra_negative_spans=args.extra_negative_spans,
        span_proposal_max_len=args.span_proposal_max_len,
        max_span_proposals=args.max_span_proposals,
        pair_candidate_source=args.pair_candidate_source,
        pair_candidate_max_span_len=args.pair_candidate_max_span_len,
        pair_negative_strategy=args.pair_negative_strategy,
    )
    collator = ASTECollator(
        tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of if device.type == "cuda" else None,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        **loader_kwargs,
    )

    model = DABSStructuredModel(
        model_name=args.model_name,
        k_value=args.k_value,
        bio_loss_weights=(args.bio_o_weight, args.bio_b_weight, args.bio_i_weight),
        pair_loss_weights=(
            args.pair_none_weight,
            args.pair_neg_weight,
            args.pair_neu_weight,
            args.pair_pos_weight,
        ),
        span_proposal_loss_weight=args.span_proposal_loss_weight,
        span_proposal_pos_weights=(
            args.span_aspect_pos_weight,
            args.span_opinion_pos_weight,
        ),
        span_proposal_ranking_loss_weight=args.span_proposal_ranking_loss_weight,
        span_proposal_ranking_margin=args.span_proposal_ranking_margin,
        span_proposal_ranking_negatives=args.span_proposal_ranking_negatives,
        pair_head_type=args.pair_head_type,
        pair_relation_loss_weight=args.pair_relation_loss_weight,
        pair_polarity_loss_weight=args.pair_polarity_loss_weight,
        pair_focal_gamma=args.pair_focal_gamma,
        pair_selection_loss_weight=args.pair_selection_loss_weight,
        pair_selection_pos_weight=args.pair_selection_pos_weight,
        pair_contrastive_loss_weight=args.pair_contrastive_loss_weight,
        pair_contrastive_temperature=args.pair_contrastive_temperature,
        pair_distance_embedding_dim=args.pair_distance_embedding_dim,
        pair_distance_max=args.pair_distance_max,
    ).to(device=device, dtype=torch.float32)

    optimizer_kwargs = {
        "params": _optimizer_groups(model, args),
        "lr": args.encoder_learning_rate,
        "weight_decay": args.weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(**optimizer_kwargs)
    except (RuntimeError, TypeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(**optimizer_kwargs)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    history = []
    best_dev_score = -1.0
    best_dev_threshold = None
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    use_bf16 = args.precision == "bf16" and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    started = time.perf_counter()
    checkpoint_path = (
        args.save_model.expanduser().resolve() if args.save_model else None
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0
        synchronize()
        epoch_started = time.perf_counter()
        for batch in train_loader:
            tensor_batch = _tensor_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                outputs = model(**tensor_batch)
                loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.detach().cpu())
            train_steps += 1

        synchronize()
        train_step_s = time.perf_counter() - epoch_started
        proxy_started = time.perf_counter()
        dev_metrics = evaluate(
            model,
            dev_loader,
            device,
            span_proposal_threshold=args.span_proposal_threshold,
            precision=args.precision,
        )
        synchronize()
        dev_proxy_s = time.perf_counter() - proxy_started
        proxy_score = _selection_score(
            dev_metrics,
            use_span_proposal=args.span_proposal_max_len > 0,
        )
        end_to_end_metrics = None
        dev_end_to_end_s = 0.0
        if args.checkpoint_selection == "end_to_end":
            end_to_end_started = time.perf_counter()
            end_to_end_metrics = _end_to_end_selection(
                model,
                tokenizer,
                dev_examples,
                device,
                args,
            )
            synchronize()
            dev_end_to_end_s = time.perf_counter() - end_to_end_started
            dev_score = end_to_end_metrics["f1"]
        else:
            dev_score = proxy_score
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_steps),
            "dev_proxy_score": proxy_score,
            "checkpoint_selection": args.checkpoint_selection,
            "dev_selection_score": dev_score,
            "train_step_s": train_step_s,
            "dev_proxy_s": dev_proxy_s,
            "dev_end_to_end_s": dev_end_to_end_s,
            "epoch_compute_s": train_step_s + dev_proxy_s + dev_end_to_end_s,
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
        }
        if end_to_end_metrics is not None:
            row.update(
                {
                    f"dev_end_to_end_{key}": value
                    for key, value in end_to_end_metrics.items()
                }
            )
        history.append(row)
        improved = dev_score > best_dev_score
        if improved:
            best_dev_score = dev_score
            best_dev_threshold = (
                end_to_end_metrics["threshold"]
                if end_to_end_metrics is not None
                else None
            )
            best_epoch = epoch
            stale_epochs = 0
            if checkpoint_path is not None:
                atomic_torch_save(
                    checkpoint_path,
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "model_name": args.model_name,
                            "k_value": args.k_value,
                            "max_length": args.max_length,
                            "decode_max_pairs": args.selection_max_pairs,
                            "span_proposal_max_len": args.span_proposal_max_len,
                            "span_proposal_threshold": args.span_proposal_threshold,
                            "span_proposal_top_k": args.selection_span_proposal_top_k,
                            "include_bio_proposals": args.selection_include_bio_proposals,
                            "confidence_mode": args.selection_confidence_mode,
                            "proposal_depth_evidence": "span",
                            "pair_head_type": args.pair_head_type,
                            "pair_depth_evidence": "span",
                            "pair_contrastive_loss_weight": args.pair_contrastive_loss_weight,
                            "pair_contrastive_temperature": args.pair_contrastive_temperature,
                            "pair_distance_embedding_dim": args.pair_distance_embedding_dim,
                            "pair_distance_max": args.pair_distance_max,
                            "precision": args.precision,
                            "encoder_learning_rate": args.encoder_learning_rate,
                            "head_learning_rate": args.head_learning_rate,
                            "classifier_learning_rate": args.classifier_learning_rate,
                            "seed": args.seed,
                            "checkpoint_selection": args.checkpoint_selection,
                            "require_positive_selection_before_early_stop": (
                                args.require_positive_selection_before_early_stop
                            ),
                        },
                        "best_epoch": best_epoch,
                        "best_dev_score": best_dev_score,
                        "best_dev_threshold": best_dev_threshold,
                        "history": history,
                    },
                )
        else:
            stale_epochs += 1
        row["best_epoch"] = best_epoch
        row["best_dev_score"] = best_dev_score
        row["stale_epochs"] = stale_epochs
        synchronize()
        row["elapsed_s"] = time.perf_counter() - started
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if should_stop_early(
            early_stop_patience=args.early_stop_patience,
            min_epochs=args.min_epochs,
            epoch=epoch,
            stale_epochs=stale_epochs,
            best_dev_score=best_dev_score,
            require_positive_selection=(
                args.require_positive_selection_before_early_stop
            ),
        ):
            stopped_early = True
            break

    result = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "seed": args.seed,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "epochs": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_dev_score": best_dev_score,
        "best_dev_threshold": best_dev_threshold,
        "time_to_best_dev_s": (
            float(history[best_epoch - 1]["elapsed_s"]) if best_epoch > 0 else None
        ),
        "checkpoint_selection": args.checkpoint_selection,
        "require_positive_selection_before_early_stop": (
            args.require_positive_selection_before_early_stop
        ),
        "stopped_early": stopped_early,
        "elapsed_s": time.perf_counter() - started,
        "precision": args.precision,
        "gpu_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "gpu_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "extra_negative_spans": args.extra_negative_spans,
        "span_proposal_max_len": args.span_proposal_max_len,
        "max_span_proposals": args.max_span_proposals,
        "span_proposal_loss_weight": args.span_proposal_loss_weight,
        "span_proposal_ranking_loss_weight": args.span_proposal_ranking_loss_weight,
        "span_proposal_ranking_margin": args.span_proposal_ranking_margin,
        "span_proposal_ranking_negatives": args.span_proposal_ranking_negatives,
        "span_proposal_threshold": args.span_proposal_threshold,
        "proposal_depth_evidence": "span",
        "pair_candidate_source": args.pair_candidate_source,
        "pair_candidate_max_span_len": args.pair_candidate_max_span_len,
        "pair_negative_strategy": args.pair_negative_strategy,
        "pair_head_type": args.pair_head_type,
        "pair_depth_evidence": "span",
        "pair_contrastive_loss_weight": args.pair_contrastive_loss_weight,
        "pair_contrastive_temperature": args.pair_contrastive_temperature,
        "pair_distance_embedding_dim": args.pair_distance_embedding_dim,
        "pair_distance_max": args.pair_distance_max,
        "pair_relation_loss_weight": args.pair_relation_loss_weight,
        "pair_polarity_loss_weight": args.pair_polarity_loss_weight,
        "pair_focal_gamma": args.pair_focal_gamma,
        "pair_selection_loss_weight": args.pair_selection_loss_weight,
        "pair_selection_pos_weight": args.pair_selection_pos_weight,
        "selection_max_pairs": args.selection_max_pairs,
        "selection_batch_size": args.selection_batch_size,
        "selection_proposal_source": args.selection_proposal_source,
        "selection_span_proposal_top_k": args.selection_span_proposal_top_k,
        "selection_include_bio_proposals": args.selection_include_bio_proposals,
        "selection_confidence_mode": args.selection_confidence_mode,
        "selection_thresholds": args.selection_thresholds,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "history": history,
    }
    if args.output_json:
        out_path = args.output_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=Path, default=None)
    known_config, _ = config_probe.parse_known_args()

    parser = argparse.ArgumentParser(description="Train the Full DABS ASTE model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file containing official training defaults.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ASTE_ROOT)
    parser.add_argument(
        "--dataset",
        default="14lap",
        help="Dataset directory name under --root (for example 14lap, ca, or eu).",
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument(
        "--require-positive-selection-before-early-stop",
        action="store_true",
        help=(
            "Do not apply patience while the best checkpoint-selection score "
            "is still zero. Intended for low-resource structured extraction."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--extra-negative-spans", type=int, default=8)
    parser.add_argument("--pair-candidate-source", choices=["gold_extra", "enumerated"], default="enumerated")
    parser.add_argument("--pair-candidate-max-span-len", type=int, default=2)
    parser.add_argument("--pair-negative-strategy", choices=["first", "structured"], default="structured")
    parser.add_argument("--span-proposal-max-len", type=int, default=3)
    parser.add_argument("--max-span-proposals", type=int, default=None)
    parser.add_argument("--span-proposal-loss-weight", type=float, default=1.0)
    parser.add_argument("--span-proposal-ranking-loss-weight", type=float, default=0.5)
    parser.add_argument("--span-proposal-ranking-margin", type=float, default=1.0)
    parser.add_argument("--span-proposal-ranking-negatives", type=int, default=16)
    parser.add_argument("--span-proposal-threshold", type=float, default=0.7)
    parser.add_argument("--span-aspect-pos-weight", type=float, default=20.0)
    parser.add_argument("--span-opinion-pos-weight", type=float, default=20.0)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Legacy shared LR; fills any unspecified parameter-group LR.",
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=None)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--classifier-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--k-value", type=int, default=12)
    parser.add_argument("--bio-o-weight", type=float, default=0.2)
    parser.add_argument("--bio-b-weight", type=float, default=1.0)
    parser.add_argument("--bio-i-weight", type=float, default=1.0)
    parser.add_argument("--pair-none-weight", type=float, default=1.0)
    parser.add_argument("--pair-neg-weight", type=float, default=1.0)
    parser.add_argument("--pair-neu-weight", type=float, default=1.0)
    parser.add_argument("--pair-pos-weight", type=float, default=1.0)
    parser.add_argument("--pair-focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--pair-head-type",
        choices=["joint", "factorized"],
        default="joint",
    )
    parser.add_argument("--pair-contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--pair-contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--pair-distance-embedding-dim", type=int, default=64)
    parser.add_argument("--pair-distance-max", type=int, default=32)
    parser.add_argument("--pair-relation-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-polarity-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-selection-loss-weight", type=float, default=0.0)
    parser.add_argument("--pair-selection-pos-weight", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["proxy", "end_to_end"],
        default="end_to_end",
    )
    parser.add_argument("--selection-max-pairs", type=int, default=256)
    parser.add_argument("--selection-batch-size", type=int, default=1)
    parser.add_argument(
        "--selection-proposal-source",
        choices=["bio", "gold", "enumerated", "span_proposal"],
        default="span_proposal",
    )
    parser.add_argument("--selection-span-proposal-top-k", type=int, default=4)
    parser.add_argument(
        "--selection-include-bio-proposals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selection-confidence-mode",
        choices=["pair", "joint"],
        default="joint",
    )
    parser.add_argument(
        "--selection-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny model and tiny subset for a fast end-to-end check.",
    )
    if known_config.config is not None:
        config_path = known_config.config.expanduser().resolve()
        config_defaults = json.loads(config_path.read_text(encoding="utf-8"))
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = set(config_defaults) - valid_keys
        if unknown_keys:
            parser.error(
                f"Unknown keys in {config_path}: {sorted(unknown_keys)}"
            )
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    shared_lr = args.learning_rate
    if args.encoder_learning_rate is None:
        args.encoder_learning_rate = shared_lr if shared_lr is not None else 2e-5
    if args.head_learning_rate is None:
        args.head_learning_rate = shared_lr if shared_lr is not None else 3e-4
    if args.classifier_learning_rate is None:
        args.classifier_learning_rate = shared_lr if shared_lr is not None else 1e-3

    if args.save_model is None:
        run_dir = JOURNAL_ROOT / "outputs" / "aste" / args.dataset / f"seed_{args.seed}"
        args.save_model = run_dir / "best.pt"
    if args.output_json is None:
        args.output_json = args.save_model.parent / "summary.json"

    if args.smoke:
        args.model_name = "sshleifer/tiny-distilroberta-base"
        args.epochs = 1
        args.min_epochs = 1
        args.early_stop_patience = 0
        args.batch_size = 2
        args.eval_batch_size = 2
        args.max_train_examples = 8
        args.max_eval_examples = 4
        args.max_length = min(args.max_length, 96)
        args.max_pairs = min(args.max_pairs, 64)
        args.extra_negative_spans = min(args.extra_negative_spans, 4)
        if args.span_proposal_max_len <= 0:
            args.span_proposal_max_len = 2
        if args.max_span_proposals is not None:
            args.max_span_proposals = min(args.max_span_proposals, 64)
        args.k_value = min(args.k_value, 2)
        args.device = "cpu"
        args.precision = "fp32"

    result = train(args)
    print("DABS ASTE result")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
