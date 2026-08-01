#!/usr/bin/env python3
"""Sparse end-to-end decoding and exact surface metrics for Rest quadruples."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .data import (
    DEFAULT_REST_MAX_LENGTH,
    NUM_REST_CATEGORIES,
    REST_CATEGORY_VOCAB,
    REST_PAIR_ID_TO_SENTIMENT,
    RestCharSpan,
    RestQuadExample,
    RestSurfaceQuad,
)


@dataclass
class RestQuadDecodeOutput:
    predictions: list[list[RestSurfaceQuad]]
    diagnostics: list[dict[str, Any]]


def _active_indices(offsets, attention_mask, text: str) -> list[int]:
    return [
        index
        for index, ((start, end), active) in enumerate(zip(offsets, attention_mask))
        if active and end > start and text[start:end].strip()
    ]


def _token_to_char(
    token_span: tuple[int, int],
    offsets: Sequence[tuple[int, int]],
    text: str,
) -> RestCharSpan:
    start, end = token_span
    char_start = int(offsets[start][0])
    char_end = int(offsets[end - 1][1])
    while char_start < char_end and text[char_start].isspace():
        char_start += 1
    while char_end > char_start and text[char_end - 1].isspace():
        char_end -= 1
    if char_end <= char_start:
        raise ValueError(f"Token span {token_span} has no lexical characters")
    return RestCharSpan(char_start, char_end)


def _enumerate_spans(indices: Sequence[int], max_len: int) -> list[tuple[int, int]]:
    spans = []
    for position, start in enumerate(indices):
        for width in range(1, max_len + 1):
            final_position = position + width - 1
            if final_position >= len(indices):
                break
            final = indices[final_position]
            # Some DeBERTa-v3 fast-tokenizer records represent only a space
            # (notably around separated curly apostrophes).  Such records are
            # excluded as standalone candidates but remain inside the
            # half-open token span between two lexical endpoints.
            spans.append((start, final + 1))
    return spans


def _bio_spans_with_scores(
    logits: torch.Tensor,
    active_indices: Sequence[int],
) -> list[tuple[tuple[int, int], float]]:
    probabilities = logits.float().softmax(dim=-1).detach().cpu()
    labels = probabilities.argmax(dim=-1).tolist()
    spans = []
    current = []

    def flush():
        if not current:
            return
        indices = [row[0] for row in current]
        token_probabilities = torch.tensor([row[1] for row in current]).clamp_min(
            1e-8
        )
        # Keep confidence comparable across span lengths.  A raw product would
        # mechanically suppress longer opinion expressions.
        score = float(token_probabilities.log().mean().exp())
        spans.append(((indices[0], indices[-1] + 1), score))

    for index in active_indices:
        label = int(labels[index])
        if label == 1:
            if current:
                flush()
            current = [(index, float(probabilities[index, label]))]
        elif label == 2:
            row = (index, float(probabilities[index, label]))
            if current:
                current.append(row)
            else:
                current = [row]
        else:
            if current:
                flush()
                current = []
    if current:
        flush()
    return spans


def _bio_spans(logits: torch.Tensor, active_indices: Sequence[int]):
    return [span for span, _ in _bio_spans_with_scores(logits, active_indices)]


def _fuse_bio_score(proposal_score: float, bio_score: float, mode: str) -> float:
    if mode == "proposal":
        return float(proposal_score)
    if mode == "max":
        return max(float(proposal_score), float(bio_score))
    raise ValueError(f"Unsupported BIO score fusion: {mode!r}")


def _unique_candidates(candidates):
    result = []
    positions = {}
    for char_span, token_span, score in candidates:
        key = char_span.as_tuple()
        if key in positions:
            position = positions[key]
            prior = result[position]
            if float(score) > prior[2]:
                result[position] = (char_span, token_span, float(score))
            continue
        positions[key] = len(result)
        result.append((char_span, token_span, float(score)))
    return result


def _select(candidates, *, threshold: float, top_k: int):
    retained = [candidate for candidate in candidates if candidate[2] >= threshold]
    retained.sort(
        key=lambda row: (
            row[2],
            -((row[0].end or 0) - (row[0].start or 0)),
            -(row[0].start or 0),
        ),
        reverse=True,
    )
    return retained[:top_k]


def _from_lattice(
    pair_records,
    relation_probabilities,
    category_probabilities,
    *,
    threshold: float,
    factorized: bool = True,
):
    predictions = []
    span_label_seen = set()
    for record, relation_row, category_row in zip(
        pair_records,
        relation_probabilities,
        category_probabilities,
    ):
        if factorized:
            # ``relation_row`` is [1-p(exists), p(exists)*p(polarity)].
            # Selecting over all four entries would add an unintended gate:
            # the winning polarity product would also have to exceed
            # 1-p(exists).  For a factorized head, choose polarity within its
            # three-way conditional distribution and let the development-set
            # joint threshold decide whether the pair is emitted.
            relation_label = int(relation_row[1:].argmax().item()) + 1
        else:
            relation_label = int(relation_row.argmax().item())
            if relation_label == 0:
                continue
        sentiment = REST_PAIR_ID_TO_SENTIMENT[relation_label]
        relation_probability = float(relation_row[relation_label])
        for category_index, category_probability in enumerate(
            category_row.tolist()
        ):
            joint_score = (
                relation_probability
                * float(category_probability)
                * float(record["proposal_score"])
            )
            if joint_score < threshold:
                continue
            if category_index >= len(REST_CATEGORY_VOCAB):
                continue
            key = (
                record["aspect"].as_tuple(),
                record["opinion"].as_tuple(),
                category_index,
                sentiment,
            )
            if key in span_label_seen:
                continue
            span_label_seen.add(key)
            predictions.append(
                RestSurfaceQuad(
                    aspect=record["aspect_surface"],
                    category=REST_CATEGORY_VOCAB[category_index],
                    sentiment=sentiment,
                    opinion=record["opinion_surface"],
                )
            )
    return predictions


def decode_rest_quad_example(
    model,
    tokenizer,
    example: RestQuadExample,
    *,
    device: torch.device,
    max_length: int = DEFAULT_REST_MAX_LENGTH,
    max_pairs: int = 256,
    max_proposal_span_len: int = 3,
    span_proposal_threshold: float = 0.5,
    span_proposal_top_k: int = 8,
    quad_threshold: float = 0.5,
    return_score_lattice: bool = False,
    bio_score_fusion: str = "proposal",
) -> tuple[list[RestSurfaceQuad], dict[str, Any]]:
    encoded = tokenizer(
        example.text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    attention = [int(value) for value in encoded["attention_mask"]]
    active = _active_indices(offsets, attention, example.text)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention], dtype=torch.long, device=device)
    with torch.no_grad():
        shared = model.encode_shared(input_ids, attention_mask=attention_mask)
        context = shared["context_features"]
        aspect_bio_scored = _bio_spans_with_scores(
            model.aspect_bio_classifier(context)[0], active
        )
        opinion_bio_scored = _bio_spans_with_scores(
            model.opinion_bio_classifier(context)[0], active
        )
        aspect_bio = [span for span, _ in aspect_bio_scored]
        opinion_bio = [span for span, _ in opinion_bio_scored]
        token_spans = _enumerate_spans(active, max_proposal_span_len)
        scored_token_spans = list(token_spans)
        for bio_span in (*aspect_bio, *opinion_bio):
            if bio_span not in scored_token_spans:
                scored_token_spans.append(bio_span)
        if scored_token_spans:
            span_tensor = torch.tensor(
                [[list(span) for span in scored_token_spans]],
                dtype=torch.long,
                device=device,
            )
            span_mask = torch.ones(
                (1, len(scored_token_spans)), dtype=torch.bool, device=device
            )
            proposal_probability = model.span_proposal_readout(
                shared,
                span_proposal_spans=span_tensor,
                span_proposal_mask=span_mask,
            )[0].sigmoid().float().cpu()
        else:
            proposal_probability = torch.zeros((0, 2))

        aspect_candidates = []
        opinion_candidates = []
        scores_by_span = {
            span: scores
            for span, scores in zip(scored_token_spans, proposal_probability)
        }
        for token_span in token_spans:
            scores = scores_by_span[token_span]
            char_span = _token_to_char(token_span, offsets, example.text)
            aspect_candidates.append((char_span, token_span, float(scores[0])))
            opinion_candidates.append((char_span, token_span, float(scores[1])))
        proposal_aspect_candidates = _select(
            aspect_candidates,
            threshold=span_proposal_threshold,
            top_k=span_proposal_top_k,
        )
        proposal_opinion_candidates = _select(
            opinion_candidates,
            threshold=span_proposal_threshold,
            top_k=span_proposal_top_k,
        )
        bio_aspect_candidates = [
            (
                _token_to_char(span, offsets, example.text),
                span,
                _fuse_bio_score(
                    float(scores_by_span[span][0]),
                    bio_score,
                    bio_score_fusion,
                ),
            )
            for span, bio_score in aspect_bio_scored
        ]
        bio_opinion_candidates = [
            (
                _token_to_char(span, offsets, example.text),
                span,
                _fuse_bio_score(
                    float(scores_by_span[span][1]),
                    bio_score,
                    bio_score_fusion,
                ),
            )
            for span, bio_score in opinion_bio_scored
        ]
        aspect_candidates = _unique_candidates(
            proposal_aspect_candidates + bio_aspect_candidates
        )
        opinion_candidates = _unique_candidates(
            proposal_opinion_candidates + bio_opinion_candidates
        )
        # NULL is aspect-only by benchmark definition.
        aspect_candidates.append((RestCharSpan.null(), (-1, -1), 1.0))

        pair_records = []
        for aspect, aspect_tokens, aspect_score in aspect_candidates:
            for opinion, opinion_tokens, opinion_score in opinion_candidates:
                pair_records.append(
                    {
                        "aspect": aspect,
                        "opinion": opinion,
                        "aspect_surface": aspect.surface(example.text),
                        "opinion_surface": opinion.surface(example.text),
                        "row": [*aspect_tokens, *opinion_tokens],
                        "aspect_proposal_score": aspect_score,
                        "opinion_proposal_score": opinion_score,
                        "proposal_score": aspect_score * opinion_score,
                    }
                )
        pair_records.sort(
            key=lambda row: (
                float(row["proposal_score"]),
                not row["aspect"].is_null,
            ),
            reverse=True,
        )
        pair_records_before_cap = tuple(pair_records)
        pair_count_before_cap = len(pair_records)
        if len(pair_records) > max_pairs:
            null_records = [
                record for record in pair_records if record["aspect"].is_null
            ][:max_pairs]
            explicit_records = [
                record for record in pair_records if not record["aspect"].is_null
            ]
            pair_records = (
                null_records
                + explicit_records[: max_pairs - len(null_records)]
            )
            pair_records.sort(
                key=lambda row: float(row["proposal_score"]), reverse=True
            )
        if pair_records:
            pair_spans = torch.tensor(
                [[record["row"] for record in pair_records]],
                dtype=torch.long,
                device=device,
            )
            pair_mask = torch.ones(
                (1, len(pair_records)), dtype=torch.bool, device=device
            )
            pair_logits, intermediates = model.pair_query_readout(
                shared,
                pair_spans=pair_spans,
                pair_mask=pair_mask,
                return_intermediates=True,
            )
            if model.pair_head_type == "factorized":
                relation_probabilities = pair_logits[0].exp().float().cpu()
            else:
                relation_probabilities = pair_logits[0].softmax(dim=-1).float().cpu()
            category_probabilities = model.category_classifier(
                intermediates["pair_final"]
            )[0].sigmoid().float().cpu()
        else:
            relation_probabilities = torch.zeros((0, 4))
            category_probabilities = torch.zeros((0, NUM_REST_CATEGORIES))
    predictions = _from_lattice(
        pair_records,
        relation_probabilities,
        category_probabilities,
        threshold=quad_threshold,
        factorized=model.pair_head_type == "factorized",
    )
    gold_pairs = {
        (
            annotation.aspect_span.as_tuple(),
            annotation.opinion_span.as_tuple(),
        )
        for annotation in example.representable_annotations
        if annotation.aspect_span is not None and annotation.opinion_span is not None
    }
    candidate_pairs_before_cap = {
        (record["aspect"].as_tuple(), record["opinion"].as_tuple())
        for record in pair_records_before_cap
    }
    candidate_pairs_after_cap = {
        (record["aspect"].as_tuple(), record["opinion"].as_tuple())
        for record in pair_records
    }
    gold_explicit_pairs = {pair for pair in gold_pairs if pair[0] is not None}
    gold_null_pairs = {pair for pair in gold_pairs if pair[0] is None}
    gold_explicit_aspects = {
        pair[0] for pair in gold_explicit_pairs if pair[0] is not None
    }
    gold_opinions = {pair[1] for pair in gold_pairs}

    def char_keys(candidates):
        return {candidate[0].as_tuple() for candidate in candidates}

    proposal_aspects = char_keys(proposal_aspect_candidates)
    proposal_opinions = char_keys(proposal_opinion_candidates)
    bio_aspects = char_keys(bio_aspect_candidates)
    bio_opinions = char_keys(bio_opinion_candidates)
    union_aspects = char_keys(aspect_candidates)
    union_opinions = char_keys(opinion_candidates)
    proposal_pair_lattice = {
        (aspect, opinion)
        for aspect in proposal_aspects | {None}
        for opinion in proposal_opinions
    }
    bio_pair_lattice = {
        (aspect, opinion)
        for aspect in bio_aspects | {None}
        for opinion in bio_opinions
    }
    union_pair_lattice = {
        (aspect, opinion)
        for aspect in union_aspects
        for opinion in union_opinions
    }
    diagnostics = {
        "proposal_aspect_candidates": len(proposal_aspect_candidates),
        "proposal_opinion_candidates": len(proposal_opinion_candidates),
        "bio_aspect_candidates": len(bio_aspect_candidates),
        "bio_opinion_candidates": len(bio_opinion_candidates),
        "aspect_candidates": len(aspect_candidates),
        "opinion_candidates": len(opinion_candidates),
        "pair_count_before_cap": pair_count_before_cap,
        "pair_count": len(pair_records),
        "cap_applied": pair_count_before_cap > len(pair_records),
        "pairs_removed_by_cap": pair_count_before_cap - len(pair_records),
        "gold_representable_pairs": len(gold_pairs),
        "gold_explicit_pairs": len(gold_explicit_pairs),
        "gold_null_pairs": len(gold_null_pairs),
        "covered_gold_pairs_proposal_only": len(
            gold_pairs & proposal_pair_lattice
        ),
        "covered_gold_pairs_bio_only": len(gold_pairs & bio_pair_lattice),
        "covered_gold_pairs_union_before_cap": len(
            gold_pairs & union_pair_lattice
        ),
        "covered_gold_pairs_before_cap": len(
            gold_pairs & candidate_pairs_before_cap
        ),
        "covered_gold_pairs": len(gold_pairs & candidate_pairs_after_cap),
        "covered_gold_explicit_pairs_before_cap": len(
            gold_explicit_pairs & candidate_pairs_before_cap
        ),
        "covered_gold_explicit_pairs": len(
            gold_explicit_pairs & candidate_pairs_after_cap
        ),
        "covered_gold_null_pairs_before_cap": len(
            gold_null_pairs & candidate_pairs_before_cap
        ),
        "covered_gold_null_pairs": len(
            gold_null_pairs & candidate_pairs_after_cap
        ),
        "gold_pairs_removed_by_cap": len(
            (gold_pairs & candidate_pairs_before_cap)
            - candidate_pairs_after_cap
        ),
        "gold_explicit_aspects": len(gold_explicit_aspects),
        "gold_opinions": len(gold_opinions),
        "covered_gold_aspects_proposal": len(
            gold_explicit_aspects & proposal_aspects
        ),
        "covered_gold_aspects_bio": len(gold_explicit_aspects & bio_aspects),
        "covered_gold_aspects_union": len(gold_explicit_aspects & union_aspects),
        "covered_gold_opinions_proposal": len(
            gold_opinions & proposal_opinions
        ),
        "covered_gold_opinions_bio": len(gold_opinions & bio_opinions),
        "covered_gold_opinions_union": len(gold_opinions & union_opinions),
        "bio_score_fusion": bio_score_fusion,
        "quad_threshold": float(quad_threshold),
    }
    if return_score_lattice:
        diagnostics["_score_lattice"] = {
            "pair_records": pair_records,
            "relation_probabilities": relation_probabilities,
            "category_probabilities": category_probabilities,
            "factorized": model.pair_head_type == "factorized",
        }
    return predictions, diagnostics


def decode_rest_quad_examples(
    model,
    tokenizer,
    examples: Sequence[RestQuadExample],
    *,
    precision: str = "bf16",
    **kwargs,
) -> RestQuadDecodeOutput:
    device = torch.device(kwargs["device"])
    was_training = model.training
    model.eval()
    predictions = []
    diagnostics = []
    try:
        for example in examples:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                predicted, diagnostic = decode_rest_quad_example(
                    model,
                    tokenizer,
                    example,
                    **kwargs,
                )
            predictions.append(predicted)
            diagnostics.append(diagnostic)
    finally:
        if was_training:
            model.train()
    return RestQuadDecodeOutput(predictions, diagnostics)


def predictions_at_threshold(examples, diagnostics, threshold: float):
    predictions = []
    for example, diagnostic in zip(examples, diagnostics):
        del example
        lattice = diagnostic.get("_score_lattice")
        if lattice is None:
            raise ValueError("Decode with return_score_lattice=True")
        predictions.append(
            _from_lattice(
                lattice["pair_records"],
                lattice["relation_probabilities"],
                lattice["category_probabilities"],
                threshold=threshold,
                factorized=lattice["factorized"],
            )
        )
    return predictions


def _multiset_counts(predicted_rows, gold_rows):
    true_positive = 0
    predicted = 0
    gold = 0
    for predicted_items, gold_items in zip(predicted_rows, gold_rows):
        predicted_counter = Counter(predicted_items)
        gold_counter = Counter(gold_items)
        true_positive += sum(
            min(count, gold_counter[item])
            for item, count in predicted_counter.items()
        )
        predicted += sum(predicted_counter.values())
        gold += sum(gold_counter.values())
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": true_positive,
        "predicted": predicted,
        "gold": gold,
    }


def _official_membership_counts(predicted_rows, gold_rows):
    """Reproduce the benchmark evaluator used by published ASQP systems.

    Its membership test does not consume a matched gold item, so repeated
    predictions can each count as correct.  We expose this only to make paper
    comparisons protocol-faithful and retain strict multiset scoring alongside
    it for auditing.
    """
    true_positive = 0
    predicted = 0
    gold = 0
    for predicted_items, gold_items in zip(predicted_rows, gold_rows):
        predicted += len(predicted_items)
        gold += len(gold_items)
        true_positive += sum(item in gold_items for item in predicted_items)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": true_positive,
        "predicted": predicted,
        "gold": gold,
    }


def score_rest_quad_predictions_official(predictions, examples):
    gold = [
        [annotation.raw.as_tuple() for annotation in example.annotations]
        for example in examples
    ]
    predicted = [
        [quad.as_tuple() for quad in example_predictions]
        for example_predictions in predictions
    ]
    return {"quadruple": _official_membership_counts(predicted, gold)}


def score_rest_quad_predictions(predictions, examples, *, representable_only=False):
    gold = []
    for example in examples:
        annotations = (
            example.representable_annotations
            if representable_only
            else example.annotations
        )
        gold.append([annotation.raw.as_tuple() for annotation in annotations])
    predicted = [
        [quad.as_tuple() for quad in example_predictions]
        for example_predictions in predictions
    ]

    def project(rows, mode):
        output = []
        for row in rows:
            projected = []
            for aspect, category, sentiment, opinion in row:
                if mode == "aos":
                    projected.append((aspect, opinion, sentiment))
                elif mode == "explicit":
                    if aspect != "NULL":
                        projected.append((aspect, category, sentiment, opinion))
                elif mode == "null":
                    if aspect == "NULL":
                        projected.append((aspect, category, sentiment, opinion))
            output.append(projected)
        return output

    return {
        "quadruple": _multiset_counts(predicted, gold),
        "aos_projection": _multiset_counts(
            project(predicted, "aos"), project(gold, "aos")
        ),
        "explicit_aspect_quadruple": _multiset_counts(
            project(predicted, "explicit"), project(gold, "explicit")
        ),
        "null_aspect_quadruple": _multiset_counts(
            project(predicted, "null"), project(gold, "null")
        ),
    }
