#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end decoding utilities for the DABS ASTE release.

The training dataset can supervise pair-query ACBS on gold span candidates.
This module performs the real pipeline: BIO proposal -> aspect/opinion pair
enumeration -> pair-query polarity classification -> ASTE triplets. It also
supports a gold-span proposal mode for pair/polarity diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .data import (
    ASTE_PAIR_ID_TO_LABEL,
    BIO_ID_TO_TAG,
    ASTESentence,
    ASTESpan,
    ASTETriplet,
    extract_spans_from_bio,
    gold_spans_for_example,
)


@dataclass
class ASTEDecodeOutput:
    predictions: list[ASTESentence]
    aspect_spans: list[tuple[ASTESpan, ...]]
    opinion_spans: list[tuple[ASTESpan, ...]]
    pair_counts: list[int]
    diagnostics: list[dict[str, Any]]


def _word_to_token_spans(word_ids: Sequence[int | None]) -> dict[int, tuple[int, int]]:
    spans: dict[int, list[int]] = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        spans.setdefault(int(word_idx), []).append(int(token_idx))
    return {
        word_idx: (indices[0], indices[-1] + 1)
        for word_idx, indices in spans.items()
    }


def _first_subword_indices(word_ids: Sequence[int | None]) -> dict[int, int]:
    first: dict[int, int] = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        first.setdefault(int(word_idx), int(token_idx))
    return first


def _span_to_token_span(
    span: ASTESpan,
    word_to_token: dict[int, tuple[int, int]],
) -> tuple[int, int] | None:
    token_ranges = []
    for word_idx in span.indices:
        mapped = word_to_token.get(int(word_idx))
        if mapped is None:
            return None
        token_ranges.append(mapped)
    return token_ranges[0][0], token_ranges[-1][1]


def _decode_bio_logits_to_word_spans(
    logits: torch.Tensor,
    word_ids: Sequence[int | None],
    word_count: int,
) -> tuple[tuple[ASTESpan, ...], tuple[str, ...]]:
    predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    first_subwords = _first_subword_indices(word_ids)
    tags = ["O"] * int(word_count)

    for word_idx, token_idx in first_subwords.items():
        if word_idx >= word_count:
            continue
        tags[word_idx] = BIO_ID_TO_TAG[int(predicted_ids[token_idx])]

    return extract_spans_from_bio(tags), tuple(tags)


def _enumerate_contiguous_word_spans(
    word_count: int,
    *,
    max_span_len: int,
) -> tuple[ASTESpan, ...]:
    spans = []
    for start in range(int(word_count)):
        upper = min(int(word_count), start + int(max_span_len))
        for end_exclusive in range(start + 1, upper + 1):
            spans.append(ASTESpan(indices=tuple(range(start, end_exclusive))))
    return tuple(spans)


def _build_span_proposal_candidates(
    word_spans: Sequence[ASTESpan],
    word_to_token: dict[int, tuple[int, int]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[ASTESpan]]:
    span_rows: list[list[int]] = []
    span_specs: list[ASTESpan] = []
    for span in word_spans:
        token_span = _span_to_token_span(span, word_to_token)
        if token_span is None:
            continue
        span_rows.append([int(token_span[0]), int(token_span[1])])
        span_specs.append(span)

    if not span_rows:
        return (
            torch.zeros((1, 0, 2), dtype=torch.long, device=device),
            torch.zeros((1, 0), dtype=torch.bool, device=device),
            span_specs,
        )

    spans = torch.tensor([span_rows], dtype=torch.long, device=device)
    mask = torch.ones((1, len(span_rows)), dtype=torch.bool, device=device)
    return spans, mask, span_specs


def _select_scored_spans(
    spans: Sequence[ASTESpan],
    scores: Sequence[float],
    *,
    threshold: float,
    top_k: int | None,
) -> tuple[ASTESpan, ...]:
    ranked = [
        (span, float(score))
        for span, score in zip(spans, scores)
        if float(score) >= float(threshold)
    ]
    ranked.sort(key=lambda item: (item[1], -item[0].length, -item[0].start), reverse=True)
    if top_k is not None:
        ranked = ranked[: max(0, int(top_k))]
    return tuple(span for span, _ in ranked)


def _merge_unique_spans(
    primary_spans: Sequence[ASTESpan],
    extra_spans: Sequence[ASTESpan],
) -> tuple[ASTESpan, ...]:
    merged: list[ASTESpan] = []
    seen: set[tuple[int, ...]] = set()
    for span in list(primary_spans) + list(extra_spans):
        key = span.as_tuple()
        if key in seen:
            continue
        seen.add(key)
        merged.append(span)
    return tuple(merged)


def _span_center(span: ASTESpan) -> float:
    return (float(span.start) + float(span.end_exclusive - 1)) / 2.0


def _pair_distance(aspect: ASTESpan, opinion: ASTESpan) -> float:
    return abs(_span_center(aspect) - _span_center(opinion))


def _build_pair_candidates(
    aspect_spans: Sequence[ASTESpan],
    opinion_spans: Sequence[ASTESpan],
    word_to_token: dict[int, tuple[int, int]],
    *,
    max_pairs: int | None,
    aspect_scores: dict[tuple[int, ...], float] | None = None,
    opinion_scores: dict[tuple[int, ...], float] | None = None,
    pair_pruning_mode: str = "sequential",
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[ASTESpan, ASTESpan]]]:
    pair_records: list[dict[str, object]] = []
    pruning_key = pair_pruning_mode.strip().lower()
    if pruning_key not in {"sequential", "proposal_score", "selector"}:
        raise ValueError(
            f"Unsupported pair_pruning_mode={pair_pruning_mode!r}; "
            "use 'sequential', 'proposal_score', or 'selector'."
        )
    aspect_scores = aspect_scores or {}
    opinion_scores = opinion_scores or {}

    for aspect in aspect_spans:
        aspect_token_span = _span_to_token_span(aspect, word_to_token)
        if aspect_token_span is None:
            continue
        for opinion in opinion_spans:
            opinion_token_span = _span_to_token_span(opinion, word_to_token)
            if opinion_token_span is None:
                continue
            aspect_score = float(aspect_scores.get(aspect.as_tuple(), 1.0))
            opinion_score = float(opinion_scores.get(opinion.as_tuple(), 1.0))
            pair_records.append(
                {
                    "row": [
                        int(aspect_token_span[0]),
                        int(aspect_token_span[1]),
                        int(opinion_token_span[0]),
                        int(opinion_token_span[1]),
                    ],
                    "spec": (aspect, opinion),
                    "proposal_score": aspect_score * opinion_score,
                    "distance": _pair_distance(aspect, opinion),
                }
            )

    if max_pairs is not None and len(pair_records) > int(max_pairs):
        if pruning_key == "proposal_score":
            pair_records.sort(
                key=lambda item: (
                    float(item["proposal_score"]),
                    -float(item["distance"]),
                    -int(item["spec"][0].length),  # type: ignore[index]
                    -int(item["spec"][1].length),  # type: ignore[index]
                    -int(item["spec"][0].start),  # type: ignore[index]
                    -int(item["spec"][1].start),  # type: ignore[index]
                ),
                reverse=True,
            )
        if pruning_key != "selector":
            pair_records = pair_records[: int(max_pairs)]

    pair_rows = [
        [
            int(value)
            for value in record["row"]  # type: ignore[union-attr]
        ]
        for record in pair_records
    ]
    pair_specs = [
        record["spec"]  # type: ignore[misc]
        for record in pair_records
    ]

    if not pair_rows:
        return (
            torch.zeros((1, 0, 4), dtype=torch.long, device=device),
            torch.zeros((1, 0), dtype=torch.bool, device=device),
            pair_specs,
        )

    pair_spans = torch.tensor([pair_rows], dtype=torch.long, device=device)
    pair_mask = torch.ones((1, len(pair_rows)), dtype=torch.bool, device=device)
    return pair_spans, pair_mask, pair_specs


def _pad_candidate_tensors(
    tensors: Sequence[torch.Tensor],
    *,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    counts = [int(tensor.shape[1]) for tensor in tensors]
    max_count = max(counts, default=0)
    padded = torch.zeros(
        (len(tensors), max_count, width),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros(
        (len(tensors), max_count),
        dtype=torch.bool,
        device=device,
    )
    for batch_idx, (tensor, count) in enumerate(zip(tensors, counts)):
        if count == 0:
            continue
        padded[batch_idx, :count] = tensor[0]
        mask[batch_idx, :count] = True
    return padded, mask, counts


def _minimum_bio_probability_margin(
    logits: torch.Tensor,
    word_ids: Sequence[int | None],
) -> float:
    first_subwords = _first_subword_indices(word_ids)
    if not first_subwords:
        return 1.0
    indices = torch.tensor(
        list(first_subwords.values()),
        dtype=torch.long,
        device=logits.device,
    )
    probabilities = logits.index_select(0, indices).softmax(dim=-1).float()
    top_two = probabilities.topk(k=2, dim=-1).values
    return float((top_two[:, 0] - top_two[:, 1]).min().detach().cpu())


def _span_selection_is_sensitive(
    scores: Sequence[float],
    *,
    threshold: float,
    top_k: int | None,
    margin: float,
) -> bool:
    if margin <= 0.0:
        return False
    values = [float(score) for score in scores]
    if any(abs(score - float(threshold)) <= margin for score in values):
        return True
    eligible = sorted(
        (score for score in values if score >= float(threshold)),
        reverse=True,
    )
    if top_k is None or top_k <= 0 or len(eligible) <= int(top_k):
        return False
    return abs(eligible[int(top_k) - 1] - eligible[int(top_k)]) <= margin


def _has_overlapping_spans(spans: Sequence[ASTESpan]) -> bool:
    span_sets = [set(span.indices) for span in spans]
    return any(
        bool(left & right)
        for left_idx, left in enumerate(span_sets)
        for right in span_sets[left_idx + 1 :]
    )


def decode_aste_example(
    model,
    tokenizer,
    example: ASTESentence,
    *,
    device: torch.device | str | None = None,
    max_length: int = 128,
    max_pairs: int | None = 256,
    pair_confidence_threshold: float = 0.0,
    proposal_source: str = "bio",
    max_proposal_span_len: int = 3,
    span_proposal_threshold: float = 0.5,
    span_proposal_top_k: int | None = 20,
    span_proposal_aspect_top_k: int | None = None,
    span_proposal_opinion_top_k: int | None = None,
    include_bio_proposals: bool = False,
    confidence_mode: str = "pair",
    pair_pruning_mode: str = "sequential",
    return_intermediates: bool = False,
) -> tuple[ASTESentence, dict[str, Any]]:
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    encoding = tokenizer(
        list(example.tokens),
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    word_ids = encoding.word_ids()
    word_to_token = _word_to_token_spans(word_ids)
    confidence_key = confidence_mode.strip().lower()
    pruning_key = pair_pruning_mode.strip().lower()
    if confidence_key not in {"pair", "joint"}:
        raise ValueError(
            f"Unsupported confidence_mode={confidence_mode!r}; use 'pair' or 'joint'."
        )
    input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [encoding["attention_mask"]],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        shared = model.encode_shared(input_ids=input_ids, attention_mask=attention_mask)
        context = shared["context_features"]
        aspect_logits = model.aspect_bio_classifier(context)[0]
        opinion_logits = model.opinion_bio_classifier(context)[0]

        bio_aspect_spans, aspect_tags = _decode_bio_logits_to_word_spans(
            aspect_logits,
            word_ids,
            len(example.tokens),
        )
        bio_opinion_spans, opinion_tags = _decode_bio_logits_to_word_spans(
            opinion_logits,
            word_ids,
            len(example.tokens),
        )
        proposal_key = proposal_source.strip().lower()
        span_proposal_candidate_count = 0
        span_proposal_scores: list[dict[str, object]] = []
        aspect_score_by_span: dict[tuple[int, ...], float] = {}
        opinion_score_by_span: dict[tuple[int, ...], float] = {}
        bio_aspect_added_count = 0
        bio_opinion_added_count = 0
        if proposal_key == "bio":
            aspect_spans = bio_aspect_spans
            opinion_spans = bio_opinion_spans
        elif proposal_key == "gold":
            aspect_spans = gold_spans_for_example(example, "aspect")
            opinion_spans = gold_spans_for_example(example, "opinion")
        elif proposal_key == "enumerated":
            aspect_spans = _enumerate_contiguous_word_spans(
                len(example.tokens),
                max_span_len=max_proposal_span_len,
            )
            opinion_spans = aspect_spans
        elif proposal_key == "span_proposal":
            candidate_word_spans = _enumerate_contiguous_word_spans(
                len(example.tokens),
                max_span_len=max_proposal_span_len,
            )
            span_proposal_spans, span_proposal_mask, span_specs = (
                _build_span_proposal_candidates(
                    candidate_word_spans,
                    word_to_token,
                    device=device,
                )
            )
            span_proposal_candidate_count = len(span_specs)
            if span_specs:
                span_logits = model.span_proposal_readout(
                    shared,
                    span_proposal_spans=span_proposal_spans,
                    span_proposal_mask=span_proposal_mask,
                )
                span_probs = span_logits[0].sigmoid().detach().cpu().float()
                aspect_scores = span_probs[:, 0].tolist()
                opinion_scores = span_probs[:, 1].tolist()
                aspect_score_by_span = {
                    span.as_tuple(): float(score)
                    for span, score in zip(span_specs, aspect_scores)
                }
                opinion_score_by_span = {
                    span.as_tuple(): float(score)
                    for span, score in zip(span_specs, opinion_scores)
                }
                aspect_top_k = (
                    span_proposal_top_k
                    if span_proposal_aspect_top_k is None
                    else span_proposal_aspect_top_k
                )
                opinion_top_k = (
                    span_proposal_top_k
                    if span_proposal_opinion_top_k is None
                    else span_proposal_opinion_top_k
                )
                aspect_spans = _select_scored_spans(
                    span_specs,
                    aspect_scores,
                    threshold=span_proposal_threshold,
                    top_k=aspect_top_k,
                )
                opinion_spans = _select_scored_spans(
                    span_specs,
                    opinion_scores,
                    threshold=span_proposal_threshold,
                    top_k=opinion_top_k,
                )
                span_proposal_scores = [
                    {
                        "span": span.as_tuple(),
                        "aspect_score": float(aspect_score),
                        "opinion_score": float(opinion_score),
                    }
                    for span, aspect_score, opinion_score in zip(
                        span_specs,
                        aspect_scores,
                        opinion_scores,
                    )
                ]
            else:
                aspect_spans = tuple()
                opinion_spans = tuple()
        else:
            raise ValueError(
                f"Unsupported ASTE proposal_source={proposal_source!r}; "
                "use 'bio', 'gold', 'enumerated', or 'span_proposal'."
            )
        if proposal_key == "span_proposal" and include_bio_proposals:
            aspect_count = len(aspect_spans)
            opinion_count = len(opinion_spans)
            aspect_spans = _merge_unique_spans(aspect_spans, bio_aspect_spans)
            opinion_spans = _merge_unique_spans(opinion_spans, bio_opinion_spans)
            bio_aspect_added_count = len(aspect_spans) - aspect_count
            bio_opinion_added_count = len(opinion_spans) - opinion_count
        pair_candidate_count_before_pruning = len(aspect_spans) * len(opinion_spans)
        pair_spans, pair_mask, pair_specs = _build_pair_candidates(
            aspect_spans,
            opinion_spans,
            word_to_token,
            max_pairs=None if pruning_key == "selector" else max_pairs,
            aspect_scores=aspect_score_by_span,
            opinion_scores=opinion_score_by_span,
            pair_pruning_mode=pair_pruning_mode,
            device=device,
        )

        pair_predictions: list[int] = []
        pair_confidences: list[float] = []
        pair_selection_scores: list[float] = []
        intermediates: dict[str, torch.Tensor] = {}
        if pair_specs:
            pair_logits, intermediates = model.pair_query_readout(
                shared,
                pair_spans=pair_spans,
                pair_mask=pair_mask,
                return_intermediates=return_intermediates or pruning_key == "selector",
            )
            pair_probs = pair_logits[0].softmax(dim=-1)
            pair_predictions = pair_probs.argmax(dim=-1).detach().cpu().tolist()
            pair_confidences = (
                pair_probs.max(dim=-1).values.detach().cpu().float().tolist()
            )
            if "pair_selection_logits" in intermediates:
                pair_selection_scores = (
                    intermediates["pair_selection_logits"][0]
                    .sigmoid()
                    .detach()
                    .cpu()
                    .float()
                    .tolist()
                )
            else:
                pair_selection_scores = [1.0 for _ in pair_specs]

            if (
                pruning_key == "selector"
                and max_pairs is not None
                and len(pair_specs) > int(max_pairs)
            ):
                keep_indices = sorted(
                    range(len(pair_specs)),
                    key=lambda idx: (
                        float(pair_selection_scores[idx])
                        * float(
                            aspect_score_by_span.get(
                                pair_specs[idx][0].as_tuple(),
                                1.0,
                            )
                        )
                        * float(
                            opinion_score_by_span.get(
                                pair_specs[idx][1].as_tuple(),
                                1.0,
                            )
                        ),
                        -_pair_distance(pair_specs[idx][0], pair_specs[idx][1]),
                    ),
                    reverse=True,
                )[: int(max_pairs)]
                pair_specs = [pair_specs[idx] for idx in keep_indices]
                pair_predictions = [pair_predictions[idx] for idx in keep_indices]
                pair_confidences = [pair_confidences[idx] for idx in keep_indices]
                pair_selection_scores = [pair_selection_scores[idx] for idx in keep_indices]

    triplets: list[ASTETriplet] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...], str]] = set()
    pair_label_counts: dict[str, int] = {}
    candidate_pairs: list[dict[str, object]] = []
    threshold = float(pair_confidence_threshold)
    if not pair_selection_scores:
        pair_selection_scores = [1.0 for _ in pair_specs]
    for (aspect, opinion), label_id, confidence, selector_score in zip(
        pair_specs,
        pair_predictions,
        pair_confidences,
        pair_selection_scores,
    ):
        label = ASTE_PAIR_ID_TO_LABEL[int(label_id)]
        pair_label_counts[label] = pair_label_counts.get(label, 0) + 1
        pair_confidence = float(confidence)
        aspect_score = float(aspect_score_by_span.get(aspect.as_tuple(), 1.0))
        opinion_score = float(opinion_score_by_span.get(opinion.as_tuple(), 1.0))
        joint_confidence = pair_confidence * aspect_score * opinion_score
        filter_confidence = (
            joint_confidence if confidence_key == "joint" else pair_confidence
        )
        candidate_pairs.append(
            {
                "aspect": aspect.as_tuple(),
                "opinion": opinion.as_tuple(),
                "label": label,
                "confidence": float(filter_confidence),
                "pair_confidence": pair_confidence,
                "aspect_score": aspect_score,
                "opinion_score": opinion_score,
                "joint_confidence": joint_confidence,
                "pair_selection_score": float(selector_score),
            }
        )
        if label == "NONE" or filter_confidence < threshold:
            continue
        triplet = ASTETriplet(aspect=aspect, opinion=opinion, sentiment=label)
        key = triplet.as_tuple()
        if key not in seen:
            seen.add(key)
            triplets.append(triplet)

    prediction = ASTESentence(
        text=example.text,
        tokens=example.tokens,
        triplets=tuple(triplets),
        source_path=example.source_path,
        line_no=example.line_no,
    )
    diagnostics: dict[str, Any] = {
        "aspect_spans": aspect_spans,
        "opinion_spans": opinion_spans,
        "bio_aspect_spans": bio_aspect_spans,
        "bio_opinion_spans": bio_opinion_spans,
        "aspect_tags": aspect_tags,
        "opinion_tags": opinion_tags,
        "pair_count": len(pair_specs),
        "pair_candidate_count_before_pruning": pair_candidate_count_before_pruning,
        "pair_pruning_mode": pair_pruning_mode,
        "pair_label_counts": pair_label_counts,
        "candidate_pairs": candidate_pairs,
        "pair_confidence_threshold": threshold,
        "confidence_mode": confidence_key,
        "proposal_source": proposal_source,
        "max_proposal_span_len": int(max_proposal_span_len),
        "span_proposal_threshold": float(span_proposal_threshold),
        "span_proposal_top_k": span_proposal_top_k,
        "span_proposal_aspect_top_k": span_proposal_aspect_top_k,
        "span_proposal_opinion_top_k": span_proposal_opinion_top_k,
        "include_bio_proposals": bool(include_bio_proposals),
        "bio_aspect_added_count": bio_aspect_added_count,
        "bio_opinion_added_count": bio_opinion_added_count,
        "span_proposal_candidate_count": span_proposal_candidate_count,
        "span_proposal_scores": span_proposal_scores,
    }
    if return_intermediates:
        diagnostics["intermediates"] = {
            key: value.detach().cpu() for key, value in intermediates.items()
        }
    return prediction, diagnostics


def decode_aste_examples(
    model,
    tokenizer,
    examples: Sequence[ASTESentence],
    *,
    device: torch.device | str | None = None,
    max_length: int = 128,
    max_pairs: int | None = 256,
    pair_confidence_threshold: float = 0.0,
    proposal_source: str = "bio",
    max_proposal_span_len: int = 3,
    span_proposal_threshold: float = 0.5,
    span_proposal_top_k: int | None = 20,
    span_proposal_aspect_top_k: int | None = None,
    span_proposal_opinion_top_k: int | None = None,
    include_bio_proposals: bool = False,
    confidence_mode: str = "pair",
    pair_pruning_mode: str = "sequential",
    return_intermediates: bool = False,
    precision: str = "fp32",
) -> ASTEDecodeOutput:
    was_training = model.training
    model.eval()
    predictions: list[ASTESentence] = []
    aspect_spans: list[tuple[ASTESpan, ...]] = []
    opinion_spans: list[tuple[ASTESpan, ...]] = []
    pair_counts: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    use_bf16 = precision == "bf16" and model_device.type == "cuda"

    try:
        for example in examples:
            with torch.autocast(
                device_type=model_device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                prediction, diagnostic = decode_aste_example(
                    model,
                    tokenizer,
                    example,
                    device=model_device,
                    max_length=max_length,
                    max_pairs=max_pairs,
                    pair_confidence_threshold=pair_confidence_threshold,
                    proposal_source=proposal_source,
                    max_proposal_span_len=max_proposal_span_len,
                    span_proposal_threshold=span_proposal_threshold,
                    span_proposal_top_k=span_proposal_top_k,
                    span_proposal_aspect_top_k=span_proposal_aspect_top_k,
                    span_proposal_opinion_top_k=span_proposal_opinion_top_k,
                    include_bio_proposals=include_bio_proposals,
                    confidence_mode=confidence_mode,
                    pair_pruning_mode=pair_pruning_mode,
                    return_intermediates=return_intermediates,
                )
            predictions.append(prediction)
            aspect_spans.append(diagnostic["aspect_spans"])
            opinion_spans.append(diagnostic["opinion_spans"])
            pair_counts.append(int(diagnostic["pair_count"]))
            diagnostics.append(diagnostic)
    finally:
        if was_training:
            model.train()

    return ASTEDecodeOutput(
        predictions=predictions,
        aspect_spans=aspect_spans,
        opinion_spans=opinion_spans,
        pair_counts=pair_counts,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def _decode_aste_batch(
    model,
    tokenizer,
    examples: Sequence[ASTESentence],
    *,
    device: torch.device,
    max_length: int,
    max_pairs: int | None,
    pair_confidence_threshold: float,
    proposal_source: str,
    max_proposal_span_len: int,
    span_proposal_threshold: float,
    span_proposal_top_k: int | None,
    span_proposal_aspect_top_k: int | None,
    span_proposal_opinion_top_k: int | None,
    include_bio_proposals: bool,
    confidence_mode: str,
    pair_pruning_mode: str,
    return_intermediates: bool,
    numerical_fallback_margin: float,
) -> ASTEDecodeOutput:
    if not examples:
        return ASTEDecodeOutput([], [], [], [], [])

    confidence_key = confidence_mode.strip().lower()
    pruning_key = pair_pruning_mode.strip().lower()
    proposal_key = proposal_source.strip().lower()
    if confidence_key not in {"pair", "joint"}:
        raise ValueError(
            f"Unsupported confidence_mode={confidence_mode!r}; use 'pair' or 'joint'."
        )
    if pruning_key not in {"sequential", "proposal_score", "selector"}:
        raise ValueError(
            f"Unsupported pair_pruning_mode={pair_pruning_mode!r}; "
            "use 'sequential', 'proposal_score', or 'selector'."
        )
    if proposal_key not in {"bio", "gold", "enumerated", "span_proposal"}:
        raise ValueError(
            f"Unsupported ASTE proposal_source={proposal_source!r}; "
            "use 'bio', 'gold', 'enumerated', or 'span_proposal'."
        )

    encoding = tokenizer(
        [list(example.tokens) for example in examples],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    shared = model.encode_shared(input_ids=input_ids, attention_mask=attention_mask)
    aspect_bio_logits = model.aspect_bio_classifier(shared["context_features"])
    opinion_bio_logits = model.opinion_bio_classifier(shared["context_features"])

    records: list[dict[str, Any]] = []
    proposal_tensors: list[torch.Tensor] = []
    for batch_idx, example in enumerate(examples):
        word_ids = tuple(encoding.word_ids(batch_index=batch_idx))
        word_to_token = _word_to_token_spans(word_ids)
        bio_aspect_spans, aspect_tags = _decode_bio_logits_to_word_spans(
            aspect_bio_logits[batch_idx],
            word_ids,
            len(example.tokens),
        )
        bio_opinion_spans, opinion_tags = _decode_bio_logits_to_word_spans(
            opinion_bio_logits[batch_idx],
            word_ids,
            len(example.tokens),
        )
        record: dict[str, Any] = {
            "example": example,
            "word_ids": word_ids,
            "word_to_token": word_to_token,
            "bio_aspect_spans": bio_aspect_spans,
            "bio_opinion_spans": bio_opinion_spans,
            "aspect_tags": aspect_tags,
            "opinion_tags": opinion_tags,
            "aspect_score_by_span": {},
            "opinion_score_by_span": {},
            "span_proposal_scores": [],
            "span_proposal_candidate_count": 0,
            "numerically_sensitive": min(
                _minimum_bio_probability_margin(
                    aspect_bio_logits[batch_idx], word_ids
                ),
                _minimum_bio_probability_margin(
                    opinion_bio_logits[batch_idx], word_ids
                ),
            )
            <= numerical_fallback_margin,
        }
        if proposal_key == "span_proposal":
            candidate_word_spans = _enumerate_contiguous_word_spans(
                len(example.tokens),
                max_span_len=max_proposal_span_len,
            )
            span_tensor, _, span_specs = _build_span_proposal_candidates(
                candidate_word_spans,
                word_to_token,
                device=device,
            )
            record["span_specs"] = span_specs
            record["span_proposal_candidate_count"] = len(span_specs)
            proposal_tensors.append(span_tensor)
        else:
            proposal_tensors.append(
                torch.zeros((1, 0, 2), dtype=torch.long, device=device)
            )
        records.append(record)

    if proposal_key == "span_proposal":
        proposal_spans, proposal_mask, proposal_counts = _pad_candidate_tensors(
            proposal_tensors,
            width=2,
            device=device,
        )
        proposal_logits = model.span_proposal_readout(
            shared,
            span_proposal_spans=proposal_spans,
            span_proposal_mask=proposal_mask,
        )
        proposal_probs = proposal_logits.sigmoid().detach().cpu().float()
        for batch_idx, (record, count) in enumerate(zip(records, proposal_counts)):
            span_specs = record["span_specs"]
            aspect_scores = proposal_probs[batch_idx, :count, 0].tolist()
            opinion_scores = proposal_probs[batch_idx, :count, 1].tolist()
            record["aspect_score_by_span"] = {
                span.as_tuple(): float(score)
                for span, score in zip(span_specs, aspect_scores)
            }
            record["opinion_score_by_span"] = {
                span.as_tuple(): float(score)
                for span, score in zip(span_specs, opinion_scores)
            }
            aspect_top_k = (
                span_proposal_top_k
                if span_proposal_aspect_top_k is None
                else span_proposal_aspect_top_k
            )
            opinion_top_k = (
                span_proposal_top_k
                if span_proposal_opinion_top_k is None
                else span_proposal_opinion_top_k
            )
            record["numerically_sensitive"] = bool(
                record["numerically_sensitive"]
                or _span_selection_is_sensitive(
                    aspect_scores,
                    threshold=span_proposal_threshold,
                    top_k=aspect_top_k,
                    margin=numerical_fallback_margin,
                )
                or _span_selection_is_sensitive(
                    opinion_scores,
                    threshold=span_proposal_threshold,
                    top_k=opinion_top_k,
                    margin=numerical_fallback_margin,
                )
            )
            record["aspect_spans"] = _select_scored_spans(
                span_specs,
                aspect_scores,
                threshold=span_proposal_threshold,
                top_k=aspect_top_k,
            )
            record["opinion_spans"] = _select_scored_spans(
                span_specs,
                opinion_scores,
                threshold=span_proposal_threshold,
                top_k=opinion_top_k,
            )
            record["span_proposal_scores"] = [
                {
                    "span": span.as_tuple(),
                    "aspect_score": float(aspect_score),
                    "opinion_score": float(opinion_score),
                }
                for span, aspect_score, opinion_score in zip(
                    span_specs,
                    aspect_scores,
                    opinion_scores,
                )
            ]
    else:
        for record in records:
            example = record["example"]
            if proposal_key == "bio":
                record["aspect_spans"] = record["bio_aspect_spans"]
                record["opinion_spans"] = record["bio_opinion_spans"]
            elif proposal_key == "gold":
                record["aspect_spans"] = gold_spans_for_example(example, "aspect")
                record["opinion_spans"] = gold_spans_for_example(example, "opinion")
            else:
                spans = _enumerate_contiguous_word_spans(
                    len(example.tokens),
                    max_span_len=max_proposal_span_len,
                )
                record["aspect_spans"] = spans
                record["opinion_spans"] = spans

    pair_tensors: list[torch.Tensor] = []
    for record in records:
        aspect_spans = record["aspect_spans"]
        opinion_spans = record["opinion_spans"]
        record["bio_aspect_added_count"] = 0
        record["bio_opinion_added_count"] = 0
        if proposal_key == "span_proposal" and include_bio_proposals:
            aspect_count = len(aspect_spans)
            opinion_count = len(opinion_spans)
            aspect_spans = _merge_unique_spans(
                aspect_spans,
                record["bio_aspect_spans"],
            )
            opinion_spans = _merge_unique_spans(
                opinion_spans,
                record["bio_opinion_spans"],
            )
            record["bio_aspect_added_count"] = len(aspect_spans) - aspect_count
            record["bio_opinion_added_count"] = len(opinion_spans) - opinion_count
            record["aspect_spans"] = aspect_spans
            record["opinion_spans"] = opinion_spans

        record["pair_candidate_count_before_pruning"] = (
            len(aspect_spans) * len(opinion_spans)
        )
        record["has_overlapping_spans"] = bool(
            _has_overlapping_spans(aspect_spans)
            or _has_overlapping_spans(opinion_spans)
        )
        pair_tensor, _, pair_specs = _build_pair_candidates(
            aspect_spans,
            opinion_spans,
            record["word_to_token"],
            max_pairs=None if pruning_key == "selector" else max_pairs,
            aspect_scores=record["aspect_score_by_span"],
            opinion_scores=record["opinion_score_by_span"],
            pair_pruning_mode=pair_pruning_mode,
            device=device,
        )
        record["pair_specs"] = pair_specs
        pair_tensors.append(pair_tensor)

    pair_spans, pair_mask, pair_counts = _pad_candidate_tensors(
        pair_tensors,
        width=4,
        device=device,
    )
    need_intermediates = return_intermediates or pruning_key == "selector"
    if pair_spans.shape[1] > 0:
        pair_logits, pair_intermediates = model.pair_query_readout(
            shared,
            pair_spans=pair_spans,
            pair_mask=pair_mask,
            return_intermediates=need_intermediates,
        )
        pair_probabilities = pair_logits.softmax(dim=-1).detach().cpu().float()
    else:
        pair_probabilities = torch.zeros(
            (len(records), 0, len(ASTE_PAIR_ID_TO_LABEL)),
            dtype=torch.float32,
        )
        pair_intermediates = {}

    predictions: list[ASTESentence] = []
    decoded_aspects: list[tuple[ASTESpan, ...]] = []
    decoded_opinions: list[tuple[ASTESpan, ...]] = []
    decoded_pair_counts: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    threshold = float(pair_confidence_threshold)
    for batch_idx, (record, original_pair_count) in enumerate(
        zip(records, pair_counts)
    ):
        pair_specs = list(record["pair_specs"])
        row_probs = pair_probabilities[batch_idx, :original_pair_count]
        if row_probs.shape[0] > 0:
            top_two = row_probs.topk(k=2, dim=-1).values
            minimum_pair_margin = float((top_two[:, 0] - top_two[:, 1]).min())
            record["numerically_sensitive"] = bool(
                record["numerically_sensitive"]
                or minimum_pair_margin <= numerical_fallback_margin
                or (
                    record["has_overlapping_spans"]
                    and minimum_pair_margin
                    <= max(0.3, numerical_fallback_margin * 6.0)
                )
            )
        pair_predictions = row_probs.argmax(dim=-1).tolist()
        pair_confidences = row_probs.max(dim=-1).values.tolist()
        if "pair_selection_logits" in pair_intermediates:
            pair_selection_scores = (
                pair_intermediates["pair_selection_logits"][
                    batch_idx, :original_pair_count
                ]
                .sigmoid()
                .detach()
                .cpu()
                .float()
                .tolist()
            )
        else:
            pair_selection_scores = [1.0 for _ in pair_specs]

        if (
            pruning_key == "selector"
            and max_pairs is not None
            and len(pair_specs) > int(max_pairs)
        ):
            keep_indices = sorted(
                range(len(pair_specs)),
                key=lambda idx: (
                    float(pair_selection_scores[idx])
                    * float(
                        record["aspect_score_by_span"].get(
                            pair_specs[idx][0].as_tuple(),
                            1.0,
                        )
                    )
                    * float(
                        record["opinion_score_by_span"].get(
                            pair_specs[idx][1].as_tuple(),
                            1.0,
                        )
                    ),
                    -_pair_distance(pair_specs[idx][0], pair_specs[idx][1]),
                ),
                reverse=True,
            )[: int(max_pairs)]
            pair_specs = [pair_specs[idx] for idx in keep_indices]
            pair_predictions = [pair_predictions[idx] for idx in keep_indices]
            pair_confidences = [pair_confidences[idx] for idx in keep_indices]
            pair_selection_scores = [
                pair_selection_scores[idx] for idx in keep_indices
            ]

        triplets: list[ASTETriplet] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...], str]] = set()
        pair_label_counts: dict[str, int] = {}
        candidate_pairs: list[dict[str, object]] = []
        for (aspect, opinion), label_id, confidence, selector_score in zip(
            pair_specs,
            pair_predictions,
            pair_confidences,
            pair_selection_scores,
        ):
            label = ASTE_PAIR_ID_TO_LABEL[int(label_id)]
            pair_label_counts[label] = pair_label_counts.get(label, 0) + 1
            pair_confidence = float(confidence)
            aspect_score = float(
                record["aspect_score_by_span"].get(aspect.as_tuple(), 1.0)
            )
            opinion_score = float(
                record["opinion_score_by_span"].get(opinion.as_tuple(), 1.0)
            )
            joint_confidence = pair_confidence * aspect_score * opinion_score
            filter_confidence = (
                joint_confidence if confidence_key == "joint" else pair_confidence
            )
            if label != "NONE" and (
                abs(filter_confidence - threshold) <= numerical_fallback_margin
                or (
                    record["has_overlapping_spans"]
                    and abs(filter_confidence - threshold)
                    <= max(0.3, numerical_fallback_margin * 6.0)
                )
            ):
                record["numerically_sensitive"] = True
            candidate_pairs.append(
                {
                    "aspect": aspect.as_tuple(),
                    "opinion": opinion.as_tuple(),
                    "label": label,
                    "confidence": float(filter_confidence),
                    "pair_confidence": pair_confidence,
                    "aspect_score": aspect_score,
                    "opinion_score": opinion_score,
                    "joint_confidence": joint_confidence,
                    "pair_selection_score": float(selector_score),
                }
            )
            if label == "NONE" or filter_confidence < threshold:
                continue
            triplet = ASTETriplet(
                aspect=aspect,
                opinion=opinion,
                sentiment=label,
            )
            key = triplet.as_tuple()
            if key not in seen:
                seen.add(key)
                triplets.append(triplet)

        example = record["example"]
        predictions.append(
            ASTESentence(
                text=example.text,
                tokens=example.tokens,
                triplets=tuple(triplets),
                source_path=example.source_path,
                line_no=example.line_no,
            )
        )
        decoded_aspects.append(record["aspect_spans"])
        decoded_opinions.append(record["opinion_spans"])
        decoded_pair_counts.append(len(pair_specs))
        diagnostic: dict[str, Any] = {
            "aspect_spans": record["aspect_spans"],
            "opinion_spans": record["opinion_spans"],
            "bio_aspect_spans": record["bio_aspect_spans"],
            "bio_opinion_spans": record["bio_opinion_spans"],
            "aspect_tags": record["aspect_tags"],
            "opinion_tags": record["opinion_tags"],
            "pair_count": len(pair_specs),
            "pair_candidate_count_before_pruning": record[
                "pair_candidate_count_before_pruning"
            ],
            "pair_pruning_mode": pair_pruning_mode,
            "pair_label_counts": pair_label_counts,
            "candidate_pairs": candidate_pairs,
            "pair_confidence_threshold": threshold,
            "confidence_mode": confidence_key,
            "proposal_source": proposal_source,
            "max_proposal_span_len": int(max_proposal_span_len),
            "span_proposal_threshold": float(span_proposal_threshold),
            "span_proposal_top_k": span_proposal_top_k,
            "span_proposal_aspect_top_k": span_proposal_aspect_top_k,
            "span_proposal_opinion_top_k": span_proposal_opinion_top_k,
            "include_bio_proposals": bool(include_bio_proposals),
            "bio_aspect_added_count": record["bio_aspect_added_count"],
            "bio_opinion_added_count": record["bio_opinion_added_count"],
            "span_proposal_candidate_count": record[
                "span_proposal_candidate_count"
            ],
            "span_proposal_scores": record["span_proposal_scores"],
            "numerically_sensitive": bool(record["numerically_sensitive"]),
        }
        if return_intermediates:
            diagnostic["intermediates"] = {
                key: value[
                    batch_idx : batch_idx + 1,
                    :original_pair_count,
                ]
                .detach()
                .cpu()
                for key, value in pair_intermediates.items()
            }
        diagnostics.append(diagnostic)

    return ASTEDecodeOutput(
        predictions=predictions,
        aspect_spans=decoded_aspects,
        opinion_spans=decoded_opinions,
        pair_counts=decoded_pair_counts,
        diagnostics=diagnostics,
    )


def decode_aste_examples_batched(
    model,
    tokenizer,
    examples: Sequence[ASTESentence],
    *,
    batch_size: int = 8,
    device: torch.device | str | None = None,
    max_length: int = 128,
    max_pairs: int | None = 256,
    pair_confidence_threshold: float = 0.0,
    proposal_source: str = "bio",
    max_proposal_span_len: int = 3,
    span_proposal_threshold: float = 0.5,
    span_proposal_top_k: int | None = 20,
    span_proposal_aspect_top_k: int | None = None,
    span_proposal_opinion_top_k: int | None = None,
    include_bio_proposals: bool = False,
    confidence_mode: str = "pair",
    pair_pruning_mode: str = "sequential",
    return_intermediates: bool = False,
    precision: str = "fp32",
    bucket_by_length: bool = True,
    strict_numerical_equivalence: bool = True,
    numerical_fallback_margin: float = 0.04,
) -> ASTEDecodeOutput:
    """Decode multiple sentences per encoder call while keeping sparse queries.

    Exact-length bucketing avoids padding-induced bf16 boundary changes while
    preserving the caller's original example order.
    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if numerical_fallback_margin < 0.0:
        raise ValueError("numerical_fallback_margin must be non-negative")
    if int(batch_size) == 1:
        return decode_aste_examples(
            model,
            tokenizer,
            examples,
            device=device,
            max_length=max_length,
            max_pairs=max_pairs,
            pair_confidence_threshold=pair_confidence_threshold,
            proposal_source=proposal_source,
            max_proposal_span_len=max_proposal_span_len,
            span_proposal_threshold=span_proposal_threshold,
            span_proposal_top_k=span_proposal_top_k,
            span_proposal_aspect_top_k=span_proposal_aspect_top_k,
            span_proposal_opinion_top_k=span_proposal_opinion_top_k,
            include_bio_proposals=include_bio_proposals,
            confidence_mode=confidence_mode,
            pair_pruning_mode=pair_pruning_mode,
            return_intermediates=return_intermediates,
            precision=precision,
        )

    was_training = model.training
    model.eval()
    model_device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    use_bf16 = precision == "bf16" and model_device.type == "cuda"
    if not examples:
        return ASTEDecodeOutput([], [], [], [], [])

    if bucket_by_length:
        length_groups: dict[int, list[int]] = {}
        for index, example in enumerate(examples):
            encoded = tokenizer(
                list(example.tokens),
                is_split_into_words=True,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            length_groups.setdefault(len(encoded["input_ids"]), []).append(index)
        work_batches = [
            indices[start : start + int(batch_size)]
            for length in sorted(length_groups)
            for indices in [length_groups[length]]
            for start in range(0, len(indices), int(batch_size))
        ]
    else:
        work_batches = [
            list(range(start, min(len(examples), start + int(batch_size))))
            for start in range(0, len(examples), int(batch_size))
        ]

    ordered_predictions: list[ASTESentence | None] = [None] * len(examples)
    ordered_aspects: list[tuple[ASTESpan, ...] | None] = [None] * len(examples)
    ordered_opinions: list[tuple[ASTESpan, ...] | None] = [None] * len(examples)
    ordered_pair_counts: list[int | None] = [None] * len(examples)
    ordered_diagnostics: list[dict[str, Any] | None] = [None] * len(examples)
    sensitive_indices: set[int] = set()
    try:
        for indices in work_batches:
            batch_examples = [examples[index] for index in indices]
            if len(batch_examples) == 1:
                batch_output = decode_aste_examples(
                    model,
                    tokenizer,
                    batch_examples,
                    device=model_device,
                    max_length=max_length,
                    max_pairs=max_pairs,
                    pair_confidence_threshold=pair_confidence_threshold,
                    proposal_source=proposal_source,
                    max_proposal_span_len=max_proposal_span_len,
                    span_proposal_threshold=span_proposal_threshold,
                    span_proposal_top_k=span_proposal_top_k,
                    span_proposal_aspect_top_k=span_proposal_aspect_top_k,
                    span_proposal_opinion_top_k=span_proposal_opinion_top_k,
                    include_bio_proposals=include_bio_proposals,
                    confidence_mode=confidence_mode,
                    pair_pruning_mode=pair_pruning_mode,
                    return_intermediates=return_intermediates,
                    precision=precision,
                )
            else:
                with torch.autocast(
                    device_type=model_device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    batch_output = _decode_aste_batch(
                        model,
                        tokenizer,
                        batch_examples,
                        device=model_device,
                        max_length=max_length,
                        max_pairs=max_pairs,
                        pair_confidence_threshold=pair_confidence_threshold,
                        proposal_source=proposal_source,
                        max_proposal_span_len=max_proposal_span_len,
                        span_proposal_threshold=span_proposal_threshold,
                        span_proposal_top_k=span_proposal_top_k,
                        span_proposal_aspect_top_k=span_proposal_aspect_top_k,
                        span_proposal_opinion_top_k=span_proposal_opinion_top_k,
                        include_bio_proposals=include_bio_proposals,
                        confidence_mode=confidence_mode,
                        pair_pruning_mode=pair_pruning_mode,
                        return_intermediates=return_intermediates,
                        numerical_fallback_margin=numerical_fallback_margin,
                    )
            for batch_idx, original_idx in enumerate(indices):
                ordered_predictions[original_idx] = batch_output.predictions[batch_idx]
                ordered_aspects[original_idx] = batch_output.aspect_spans[batch_idx]
                ordered_opinions[original_idx] = batch_output.opinion_spans[batch_idx]
                ordered_pair_counts[original_idx] = batch_output.pair_counts[batch_idx]
                diagnostic = batch_output.diagnostics[batch_idx]
                if diagnostic.pop("numerically_sensitive", False):
                    sensitive_indices.add(original_idx)
                diagnostic["batched_numerical_fallback"] = False
                ordered_diagnostics[original_idx] = diagnostic

        if strict_numerical_equivalence and sensitive_indices:
            fallback_indices = sorted(sensitive_indices)
            fallback_output = decode_aste_examples(
                model,
                tokenizer,
                [examples[index] for index in fallback_indices],
                device=model_device,
                max_length=max_length,
                max_pairs=max_pairs,
                pair_confidence_threshold=pair_confidence_threshold,
                proposal_source=proposal_source,
                max_proposal_span_len=max_proposal_span_len,
                span_proposal_threshold=span_proposal_threshold,
                span_proposal_top_k=span_proposal_top_k,
                span_proposal_aspect_top_k=span_proposal_aspect_top_k,
                span_proposal_opinion_top_k=span_proposal_opinion_top_k,
                include_bio_proposals=include_bio_proposals,
                confidence_mode=confidence_mode,
                pair_pruning_mode=pair_pruning_mode,
                return_intermediates=return_intermediates,
                precision=precision,
            )
            for fallback_idx, original_idx in enumerate(fallback_indices):
                ordered_predictions[original_idx] = fallback_output.predictions[
                    fallback_idx
                ]
                ordered_aspects[original_idx] = fallback_output.aspect_spans[
                    fallback_idx
                ]
                ordered_opinions[original_idx] = fallback_output.opinion_spans[
                    fallback_idx
                ]
                ordered_pair_counts[original_idx] = fallback_output.pair_counts[
                    fallback_idx
                ]
                diagnostic = fallback_output.diagnostics[fallback_idx]
                diagnostic["batched_numerical_fallback"] = True
                ordered_diagnostics[original_idx] = diagnostic
    finally:
        if was_training:
            model.train()
    if any(item is None for item in ordered_predictions):
        raise RuntimeError("Batched ASTE decoding did not populate every example.")
    return ASTEDecodeOutput(
        predictions=[item for item in ordered_predictions if item is not None],
        aspect_spans=[item for item in ordered_aspects if item is not None],
        opinion_spans=[item for item in ordered_opinions if item is not None],
        pair_counts=[item for item in ordered_pair_counts if item is not None],
        diagnostics=[item for item in ordered_diagnostics if item is not None],
    )


def predictions_at_pair_threshold(
    examples: Sequence[ASTESentence],
    diagnostics: Sequence[dict[str, Any]],
    threshold: float,
) -> list[ASTESentence]:
    """Rebuild triplet predictions from one decode pass at a new threshold."""
    predictions: list[ASTESentence] = []
    for example, diagnostic in zip(examples, diagnostics):
        triplets: list[ASTETriplet] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...], str]] = set()
        for candidate in diagnostic.get("candidate_pairs", []):
            label = str(candidate["label"])
            confidence = float(candidate["confidence"])
            if label == "NONE" or confidence < float(threshold):
                continue
            triplet = ASTETriplet(
                aspect=ASTESpan(
                    indices=tuple(int(index) for index in candidate["aspect"])
                ),
                opinion=ASTESpan(
                    indices=tuple(int(index) for index in candidate["opinion"])
                ),
                sentiment=label,
            )
            key = triplet.as_tuple()
            if key in seen:
                continue
            seen.add(key)
            triplets.append(triplet)
        predictions.append(
            ASTESentence(
                text=example.text,
                tokens=example.tokens,
                triplets=tuple(triplets),
                source_path=example.source_path,
                line_no=example.line_no,
            )
        )
    return predictions
