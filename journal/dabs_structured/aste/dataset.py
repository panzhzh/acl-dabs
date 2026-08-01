#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch dataset and collator for the DABS ASTE release.

The first model stage uses BIO proposal heads for aspect/opinion spans. The
pair-query stage is trained on gold aspect/opinion span candidates so we can
separate proposal recall from pair/polarity reasoning in early diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .data import (
    ASTE_PAIR_LABEL_TO_ID,
    ASTESentence,
    ASTESpan,
    bio_tags_for_spans,
    gold_spans_for_example,
)


IGNORE_INDEX = -100


def load_aste_tokenizer(model_name: str, **kwargs):
    """Load a fast tokenizer with the word-boundary contract used by ASTE.

    ASTE examples are passed to the tokenizer as pre-split words. RoBERTa-family
    tokenizers must therefore add the leading-space marker to every supplied
    word; otherwise all non-initial words are encoded as if they occurred at
    the beginning of a string and no longer match the pretrained distribution.
    """

    options = {"use_fast": True, **kwargs}
    if "roberta" in model_name.lower():
        options.setdefault("add_prefix_space", True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "DABS-ASTE requires a fast tokenizer with word_ids() support."
        )
    return tokenizer


@dataclass
class ASTEFeature:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    aspect_bio_labels: torch.Tensor
    opinion_bio_labels: torch.Tensor
    pair_spans: torch.Tensor
    pair_labels: torch.Tensor
    pair_mask: torch.Tensor
    span_proposal_spans: torch.Tensor
    span_aspect_labels: torch.Tensor
    span_opinion_labels: torch.Tensor
    span_proposal_mask: torch.Tensor
    tokens: tuple[str, ...]
    text: str


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


def _first_subword_mask(word_ids: Sequence[int | None]) -> list[bool]:
    seen: set[int] = set()
    mask = []
    for word_idx in word_ids:
        if word_idx is None:
            mask.append(False)
            continue
        word_idx = int(word_idx)
        if word_idx in seen:
            mask.append(False)
        else:
            seen.add(word_idx)
            mask.append(True)
    return mask


def _bio_word_labels(example: ASTESentence, kind: str) -> list[int]:
    tags = bio_tags_for_spans(
        len(example.tokens),
        gold_spans_for_example(example, kind),
        allow_overlap=True,
    )
    tag_to_id = {"O": 0, "B": 1, "I": 2}
    return [tag_to_id[tag] for tag in tags]


def _align_word_labels_to_tokens(
    word_labels: Sequence[int],
    word_ids: Sequence[int | None],
) -> torch.Tensor:
    first_mask = _first_subword_mask(word_ids)
    labels = []
    for word_idx, is_first in zip(word_ids, first_mask):
        if word_idx is None or not is_first:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(int(word_labels[int(word_idx)]))
    return torch.tensor(labels, dtype=torch.long)


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


def _enumerate_contiguous_word_spans(
    token_count: int,
    max_span_len: int,
) -> tuple[ASTESpan, ...]:
    spans = []
    for start in range(int(token_count)):
        upper = min(int(token_count), start + int(max_span_len))
        for end_exclusive in range(start + 1, upper + 1):
            spans.append(ASTESpan(indices=tuple(range(start, end_exclusive))))
    return tuple(spans)


def _spans_overlap(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return not (a[-1] < b[0] or b[-1] < a[0])


def _span_gap(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    if _spans_overlap(a, b):
        return 0
    if a[-1] < b[0]:
        return int(b[0] - a[-1])
    return int(a[0] - b[-1])


def _overlaps_any(span_key: tuple[int, ...], gold_keys: set[tuple[int, ...]]) -> bool:
    return any(_spans_overlap(span_key, gold_key) for gold_key in gold_keys)


def _negative_pair_hardness(
    aspect_key: tuple[int, ...],
    opinion_key: tuple[int, ...],
    *,
    gold_aspect_keys: set[tuple[int, ...]],
    gold_opinion_keys: set[tuple[int, ...]],
) -> tuple[int, int, int, int, int, int]:
    aspect_exact = aspect_key in gold_aspect_keys
    opinion_exact = opinion_key in gold_opinion_keys
    aspect_overlap = _overlaps_any(aspect_key, gold_aspect_keys)
    opinion_overlap = _overlaps_any(opinion_key, gold_opinion_keys)
    endpoint_score = int(aspect_exact) + int(opinion_exact)
    overlap_score = int(aspect_overlap) + int(opinion_overlap)
    distance = _span_gap(aspect_key, opinion_key)
    length_sum = len(aspect_key) + len(opinion_key)
    # Larger tuple is harder. Closer, shorter spans get priority after endpoint
    # match/overlap because they resemble plausible extracted pairs.
    return (
        endpoint_score,
        overlap_score,
        -distance,
        -length_sum,
        -aspect_key[0],
        -opinion_key[0],
    )


def _build_span_proposal_supervision(
    example: ASTESentence,
    word_to_token: dict[int, tuple[int, int]],
    max_span_len: int,
    max_span_proposals: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_span_len <= 0:
        return (
            torch.zeros((0, 2), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.bool),
        )

    gold_aspects = {
        span.as_tuple() for span in gold_spans_for_example(example, "aspect")
    }
    gold_opinions = {
        span.as_tuple() for span in gold_spans_for_example(example, "opinion")
    }
    span_rows: list[list[int]] = []
    aspect_labels: list[float] = []
    opinion_labels: list[float] = []
    for span in _enumerate_contiguous_word_spans(len(example.tokens), max_span_len):
        token_span = _span_to_token_span(span, word_to_token)
        if token_span is None:
            continue
        span_key = span.as_tuple()
        span_rows.append([int(token_span[0]), int(token_span[1])])
        aspect_labels.append(1.0 if span_key in gold_aspects else 0.0)
        opinion_labels.append(1.0 if span_key in gold_opinions else 0.0)

    if max_span_proposals is not None and len(span_rows) > int(max_span_proposals):
        positive_indices = [
            idx
            for idx, (aspect_label, opinion_label) in enumerate(
                zip(aspect_labels, opinion_labels)
            )
            if aspect_label > 0.0 or opinion_label > 0.0
        ]
        negative_indices = [
            idx
            for idx, (aspect_label, opinion_label) in enumerate(
                zip(aspect_labels, opinion_labels)
            )
            if aspect_label <= 0.0 and opinion_label <= 0.0
        ]
        keep = positive_indices + negative_indices[
            : max(0, int(max_span_proposals) - len(positive_indices))
        ]
        keep = keep[: int(max_span_proposals)]
        span_rows = [span_rows[idx] for idx in keep]
        aspect_labels = [aspect_labels[idx] for idx in keep]
        opinion_labels = [opinion_labels[idx] for idx in keep]

    if not span_rows:
        return (
            torch.zeros((0, 2), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.bool),
        )

    return (
        torch.tensor(span_rows, dtype=torch.long),
        torch.tensor(aspect_labels, dtype=torch.float32),
        torch.tensor(opinion_labels, dtype=torch.float32),
        torch.ones((len(span_rows),), dtype=torch.bool),
    )


def _build_pair_supervision(
    example: ASTESentence,
    word_to_token: dict[int, tuple[int, int]],
    max_pairs: int | None,
    extra_negative_spans: int = 0,
    pair_candidate_source: str = "gold_extra",
    pair_candidate_max_span_len: int = 2,
    pair_negative_strategy: str = "first",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aspect_word_spans = gold_spans_for_example(example, "aspect")
    opinion_word_spans = gold_spans_for_example(example, "opinion")

    def add_extra_spans(gold_spans: tuple[ASTESpan, ...]) -> tuple[ASTESpan, ...]:
        if extra_negative_spans <= 0:
            return gold_spans
        existing = {span.as_tuple() for span in gold_spans}
        extras = []
        for word_idx in range(len(example.tokens)):
            span = ASTESpan.from_positions(word_idx)
            if span.as_tuple() in existing:
                continue
            extras.append(span)
            if len(extras) >= int(extra_negative_spans):
                break
        return tuple(gold_spans) + tuple(extras)

    source_key = pair_candidate_source.strip().lower()
    if source_key == "gold_extra":
        candidate_aspect_word_spans = add_extra_spans(aspect_word_spans)
        candidate_opinion_word_spans = add_extra_spans(opinion_word_spans)
    elif source_key == "enumerated":
        enumerated = _enumerate_contiguous_word_spans(
            len(example.tokens),
            pair_candidate_max_span_len,
        )

        def include_gold(enumerated_spans: tuple[ASTESpan, ...], gold_spans: tuple[ASTESpan, ...]) -> tuple[ASTESpan, ...]:
            seen = {span.as_tuple() for span in enumerated_spans}
            merged = list(enumerated_spans)
            for span in gold_spans:
                if span.as_tuple() not in seen:
                    merged.append(span)
                    seen.add(span.as_tuple())
            return tuple(merged)

        candidate_aspect_word_spans = include_gold(enumerated, aspect_word_spans)
        candidate_opinion_word_spans = include_gold(enumerated, opinion_word_spans)
    else:
        raise ValueError(
            f"Unsupported pair_candidate_source={pair_candidate_source!r}; "
            "use 'gold_extra' or 'enumerated'."
        )

    aspect_token_spans = {
        span.as_tuple(): _span_to_token_span(span, word_to_token)
        for span in candidate_aspect_word_spans
    }
    opinion_token_spans = {
        span.as_tuple(): _span_to_token_span(span, word_to_token)
        for span in candidate_opinion_word_spans
    }
    aspect_token_spans = {
        key: value for key, value in aspect_token_spans.items() if value is not None
    }
    opinion_token_spans = {
        key: value for key, value in opinion_token_spans.items() if value is not None
    }

    gold_pair_labels: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    for triplet in example.triplets:
        aspect_key = triplet.aspect.as_tuple()
        opinion_key = triplet.opinion.as_tuple()
        if aspect_key not in aspect_token_spans or opinion_key not in opinion_token_spans:
            continue
        gold_pair_labels[(aspect_key, opinion_key)] = ASTE_PAIR_LABEL_TO_ID[
            triplet.sentiment
        ]

    pair_spans: list[list[int]] = []
    pair_labels: list[int] = []
    pair_keys: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for aspect_key, aspect_token_span in aspect_token_spans.items():
        for opinion_key, opinion_token_span in opinion_token_spans.items():
            pair_spans.append(
                [
                    int(aspect_token_span[0]),
                    int(aspect_token_span[1]),
                    int(opinion_token_span[0]),
                    int(opinion_token_span[1]),
                ]
            )
            pair_keys.append((aspect_key, opinion_key))
            pair_labels.append(
                gold_pair_labels.get((aspect_key, opinion_key), ASTE_PAIR_LABEL_TO_ID["NONE"])
            )

    if max_pairs is not None and len(pair_spans) > int(max_pairs):
        strategy_key = pair_negative_strategy.strip().lower()
        if strategy_key not in {"first", "structured"}:
            raise ValueError(
                f"Unsupported pair_negative_strategy={pair_negative_strategy!r}; "
                "use 'first' or 'structured'."
            )
        positive_indices = [
            idx for idx, label in enumerate(pair_labels) if label != ASTE_PAIR_LABEL_TO_ID["NONE"]
        ]
        negative_indices = [
            idx for idx, label in enumerate(pair_labels) if label == ASTE_PAIR_LABEL_TO_ID["NONE"]
        ]
        if strategy_key == "structured":
            gold_aspect_keys = {span.as_tuple() for span in aspect_word_spans}
            gold_opinion_keys = {span.as_tuple() for span in opinion_word_spans}
            negative_indices = sorted(
                negative_indices,
                key=lambda idx: _negative_pair_hardness(
                    pair_keys[idx][0],
                    pair_keys[idx][1],
                    gold_aspect_keys=gold_aspect_keys,
                    gold_opinion_keys=gold_opinion_keys,
                ),
                reverse=True,
            )
        keep = positive_indices + negative_indices[: max(0, int(max_pairs) - len(positive_indices))]
        keep = keep[: int(max_pairs)]
        pair_spans = [pair_spans[idx] for idx in keep]
        pair_labels = [pair_labels[idx] for idx in keep]

    if not pair_spans:
        return (
            torch.zeros((0, 4), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.bool),
        )

    return (
        torch.tensor(pair_spans, dtype=torch.long),
        torch.tensor(pair_labels, dtype=torch.long),
        torch.ones((len(pair_spans),), dtype=torch.bool),
    )


class ASTETrainingDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[ASTESentence],
        tokenizer,
        max_length: int = 128,
        max_pairs: int | None = 128,
        extra_negative_spans: int = 0,
        span_proposal_max_len: int = 0,
        max_span_proposals: int | None = None,
        pair_candidate_source: str = "gold_extra",
        pair_candidate_max_span_len: int = 2,
        pair_negative_strategy: str = "first",
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_pairs = max_pairs
        self.extra_negative_spans = int(extra_negative_spans)
        self.span_proposal_max_len = int(span_proposal_max_len)
        self.max_span_proposals = max_span_proposals
        self.pair_candidate_source = pair_candidate_source
        self.pair_candidate_max_span_len = int(pair_candidate_max_span_len)
        self.pair_negative_strategy = pair_negative_strategy

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ASTEFeature:
        example = self.examples[idx]
        encoding = self.tokenizer(
            list(example.tokens),
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        word_ids = encoding.word_ids()
        word_to_token = _word_to_token_spans(word_ids)

        aspect_labels = _align_word_labels_to_tokens(
            _bio_word_labels(example, "aspect"),
            word_ids,
        )
        opinion_labels = _align_word_labels_to_tokens(
            _bio_word_labels(example, "opinion"),
            word_ids,
        )
        pair_spans, pair_labels, pair_mask = _build_pair_supervision(
            example,
            word_to_token,
            self.max_pairs,
            extra_negative_spans=self.extra_negative_spans,
            pair_candidate_source=self.pair_candidate_source,
            pair_candidate_max_span_len=self.pair_candidate_max_span_len,
            pair_negative_strategy=self.pair_negative_strategy,
        )
        (
            span_proposal_spans,
            span_aspect_labels,
            span_opinion_labels,
            span_proposal_mask,
        ) = _build_span_proposal_supervision(
            example,
            word_to_token,
            self.span_proposal_max_len,
            max_span_proposals=self.max_span_proposals,
        )

        return ASTEFeature(
            input_ids=torch.tensor(encoding["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(encoding["attention_mask"], dtype=torch.long),
            aspect_bio_labels=aspect_labels,
            opinion_bio_labels=opinion_labels,
            pair_spans=pair_spans,
            pair_labels=pair_labels,
            pair_mask=pair_mask,
            span_proposal_spans=span_proposal_spans,
            span_aspect_labels=span_aspect_labels,
            span_opinion_labels=span_opinion_labels,
            span_proposal_mask=span_proposal_mask,
            tokens=example.tokens,
            text=example.text,
        )


class ASTECollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[ASTEFeature]) -> dict[str, Any]:
        encoded = self.tokenizer.pad(
            [
                {
                    "input_ids": feature.input_ids,
                    "attention_mask": feature.attention_mask,
                }
                for feature in features
            ],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch_size, seq_len = encoded["input_ids"].shape
        aspect_labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
        opinion_labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
        max_pairs = max((feature.pair_spans.shape[0] for feature in features), default=0)
        pair_spans = torch.zeros((batch_size, max_pairs, 4), dtype=torch.long)
        pair_labels = torch.full((batch_size, max_pairs), IGNORE_INDEX, dtype=torch.long)
        pair_mask = torch.zeros((batch_size, max_pairs), dtype=torch.bool)
        max_span_proposals = max(
            (feature.span_proposal_spans.shape[0] for feature in features),
            default=0,
        )
        span_proposal_spans = torch.zeros(
            (batch_size, max_span_proposals, 2),
            dtype=torch.long,
        )
        span_aspect_labels = torch.zeros(
            (batch_size, max_span_proposals),
            dtype=torch.float32,
        )
        span_opinion_labels = torch.zeros(
            (batch_size, max_span_proposals),
            dtype=torch.float32,
        )
        span_proposal_mask = torch.zeros(
            (batch_size, max_span_proposals),
            dtype=torch.bool,
        )

        for i, feature in enumerate(features):
            length = feature.input_ids.shape[0]
            aspect_labels[i, :length] = feature.aspect_bio_labels
            opinion_labels[i, :length] = feature.opinion_bio_labels

            num_pairs = feature.pair_spans.shape[0]
            if num_pairs > 0:
                pair_spans[i, :num_pairs] = feature.pair_spans
                pair_labels[i, :num_pairs] = feature.pair_labels
                pair_mask[i, :num_pairs] = feature.pair_mask

            num_span_proposals = feature.span_proposal_spans.shape[0]
            if num_span_proposals > 0:
                span_proposal_spans[i, :num_span_proposals] = feature.span_proposal_spans
                span_aspect_labels[i, :num_span_proposals] = feature.span_aspect_labels
                span_opinion_labels[i, :num_span_proposals] = feature.span_opinion_labels
                span_proposal_mask[i, :num_span_proposals] = feature.span_proposal_mask

        encoded.update(
            {
                "aspect_bio_labels": aspect_labels,
                "opinion_bio_labels": opinion_labels,
                "pair_spans": pair_spans,
                "pair_labels": pair_labels,
                "pair_mask": pair_mask,
                "span_proposal_spans": span_proposal_spans,
                "span_aspect_labels": span_aspect_labels,
                "span_opinion_labels": span_opinion_labels,
                "span_proposal_mask": span_proposal_mask,
                "tokens": [feature.tokens for feature in features],
                "texts": [feature.text for feature in features],
            }
        )
        return encoded
