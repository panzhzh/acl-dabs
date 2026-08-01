#!/usr/bin/env python3
"""Token-aligned sparse training features for ABSA-QUAD Rest15/Rest16."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TypeVar

import torch
from torch.utils.data import Dataset

from .data import (
    DEFAULT_REST_MAX_LENGTH,
    NUM_REST_CATEGORIES,
    REST_CATEGORY_TO_ID,
    REST_PAIR_LABEL_TO_ID,
    RestCharSpan,
    RestQuadExample,
)


IGNORE_INDEX = -100
_T = TypeVar("_T")


@dataclass(frozen=True)
class _RoleCandidate:
    token_span: tuple[int, int]
    is_null: bool = False


@dataclass(frozen=True)
class _TokenAlignment:
    token_span: tuple[int, int] | None
    method: str
    reason: str | None = None


@dataclass(frozen=True)
class RestQuadHardNegative:
    """A teacher-mined non-gold pair in tokenizer coordinates.

    ``pair_span`` is ``(aspect_start, aspect_end, opinion_start, opinion_end)``.
    A NULL aspect is encoded as ``(-1, -1)``; NULL opinions are invalid for the
    official Rest15/Rest16 task.  Rows are consumed in descending teacher-score
    order, so the score is retained for auditing even though it is not used as
    a loss weight.
    """

    pair_span: tuple[int, int, int, int]
    score: float

    def __post_init__(self) -> None:
        if len(self.pair_span) != 4:
            raise ValueError("A hard-negative pair span must contain four offsets")
        if not torch.isfinite(torch.tensor(float(self.score))):
            raise ValueError("A hard-negative teacher score must be finite")


@dataclass
class RestQuadFeature:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    aspect_bio_labels: torch.Tensor
    opinion_bio_labels: torch.Tensor
    pair_spans: torch.Tensor
    pair_null_mask: torch.Tensor
    pair_labels: torch.Tensor
    category_targets: torch.Tensor
    category_target_mask: torch.Tensor
    pair_mask: torch.Tensor
    span_proposal_spans: torch.Tensor
    span_aspect_labels: torch.Tensor
    span_opinion_labels: torch.Tensor
    span_proposal_mask: torch.Tensor
    example: RestQuadExample


@dataclass
class RestTokenAlignmentStats:
    examples_seen: int = 0
    raw_quads_seen: int = 0
    surface_representable_quads: int = 0
    surface_unrepresentable_quads: int = 0
    explicit_spans_seen: int = 0
    exact_token_spans: int = 0
    boundary_adjusted_token_spans: int = 0
    truncated_token_spans: int = 0
    unaligned_token_spans: int = 0
    quads_dropped_for_token_alignment: int = 0
    trainable_quads: int = 0
    positive_pairs: int = 0
    negative_pairs: int = 0
    multi_label_positive_pairs: int = 0
    pair_candidates_before_cap: int = 0
    pair_candidates_after_cap: int = 0
    span_proposals_before_cap: int = 0
    span_proposals_after_cap: int = 0
    hard_negative_pairs_requested: int = 0
    hard_negative_pairs_retained: int = 0
    hard_negative_pairs_dropped_by_cap: int = 0
    hard_negative_pairs_already_required: int = 0
    hard_negative_pairs_duplicate: int = 0
    surface_equivalent_negative_pairs_excluded: int = 0

    @property
    def final_representable_quads(self) -> int:
        return self.trainable_quads

    @property
    def raw_gold_recall_ceiling(self) -> float:
        return (
            self.trainable_quads / self.raw_quads_seen
            if self.raw_quads_seen
            else 1.0
        )

    def as_dict(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = {
            field_name: int(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }
        payload["final_representable_quads"] = self.final_representable_quads
        payload["raw_gold_recall_ceiling"] = self.raw_gold_recall_ceiling
        return payload


@dataclass(frozen=True)
class RestTokenAlignmentIssue:
    example_index: int
    line_no: int | None
    role: str
    char_span: tuple[int, int]
    surface: str
    reason: str


def _append_unique(items: list[_T], values: Iterable[_T]) -> None:
    seen = set(items)
    for value in values:
        if value not in seen:
            items.append(value)
            seen.add(value)


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    return [int(item) for item in value]


def _as_offsets(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    return [(int(start), int(end)) for start, end in value]


def _align_char_span(
    span: RestCharSpan,
    *,
    text: str,
    offsets: Sequence[tuple[int, int]],
    attention_mask: Sequence[int],
) -> _TokenAlignment:
    if span.is_null:
        raise ValueError("NULL spans use the explicit sentinel path")
    assert span.start is not None and span.end is not None
    indices = [
        index
        for index, ((start, end), active) in enumerate(zip(offsets, attention_mask))
        if active and end > start and end > span.start and start < span.end
    ]
    retained_end = max(
        (
            end
            for (start, end), active in zip(offsets, attention_mask)
            if active and end > start
        ),
        default=0,
    )
    if not indices:
        method = "truncated" if span.end > retained_end else "unaligned"
        return _TokenAlignment(None, method, "no overlapping retained token")
    first, last = indices[0], indices[-1]
    covered_start = offsets[first][0]
    covered_end = offsets[last][1]
    if covered_start == span.start and covered_end == span.end:
        return _TokenAlignment((first, last + 1), "exact")
    gold_surface = text[span.start : span.end]
    covered_surface = text[covered_start:covered_end]
    if covered_surface.strip() == gold_surface:
        return _TokenAlignment((first, last + 1), "boundary_adjusted")
    method = "truncated" if span.end > retained_end else "unaligned"
    return _TokenAlignment(
        None,
        method,
        f"token coverage {covered_surface!r} differs from {gold_surface!r}",
    )


def _content_token_indices(
    offsets: Sequence[tuple[int, int]],
    attention_mask: Sequence[int],
    text: str,
) -> list[int]:
    return [
        index
        for index, ((start, end), active) in enumerate(zip(offsets, attention_mask))
        if active and end > start and text[start:end].strip()
    ]


def _enumerate_token_spans(
    offsets: Sequence[tuple[int, int]],
    attention_mask: Sequence[int],
    max_span_len: int,
    text: str,
) -> list[tuple[int, int]]:
    indices = _content_token_indices(offsets, attention_mask, text)
    spans = []
    for position, start in enumerate(indices):
        for width in range(1, max_span_len + 1):
            end_position = position + width - 1
            if end_position >= len(indices):
                break
            end_token = indices[end_position]
            # Whitespace-only tokenizer records are not candidates in their
            # own right, but a multi-token span may bridge them.
            spans.append((start, end_token + 1))
    return spans


def _bio_labels(
    length: int,
    offsets: Sequence[tuple[int, int]],
    attention_mask: Sequence[int],
    spans: Sequence[tuple[int, int]],
    text: str,
) -> torch.Tensor:
    labels = torch.full((length,), IGNORE_INDEX, dtype=torch.long)
    for index, ((start, end), active) in enumerate(zip(offsets, attention_mask)):
        if active and end > start and text[start:end].strip():
            labels[index] = 0
    for start, end in spans:
        for index in range(start, end):
            if labels[index] != IGNORE_INDEX:
                labels[index] = 1 if index == start else 2
    return labels


def _span_proposals(
    *,
    offsets: Sequence[tuple[int, int]],
    attention_mask: Sequence[int],
    aspect_spans: Sequence[tuple[int, int]],
    opinion_spans: Sequence[tuple[int, int]],
    max_span_len: int,
    max_proposals: int,
    text: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    spans = _enumerate_token_spans(
        offsets,
        attention_mask,
        max_span_len,
        text,
    )
    _append_unique(spans, aspect_spans)
    _append_unique(spans, opinion_spans)
    before = len(spans)
    aspect_set = set(aspect_spans)
    opinion_set = set(opinion_spans)
    positives = [
        span for span in spans if span in aspect_set or span in opinion_set
    ]
    negatives = [
        span for span in spans if span not in aspect_set and span not in opinion_set
    ]
    if len(positives) > max_proposals:
        raise ValueError("Positive spans exceed max_span_proposals")
    spans = positives + negatives[: max_proposals - len(positives)]
    if not spans:
        return (
            torch.zeros((0, 2), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.bool),
            before,
        )
    return (
        torch.tensor(spans, dtype=torch.long),
        torch.tensor([float(span in aspect_set) for span in spans]),
        torch.tensor([float(span in opinion_set) for span in spans]),
        torch.ones((len(spans),), dtype=torch.bool),
        before,
    )


class RestQuadTrainingDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[RestQuadExample],
        tokenizer,
        *,
        max_length: int = DEFAULT_REST_MAX_LENGTH,
        max_pairs: int = 32,
        pair_candidate_max_span_len: int = 3,
        span_proposal_max_len: int = 3,
        max_span_proposals: int = 384,
        candidate_seed: int = 0,
        hard_negative_pairs_by_example: Sequence[
            Sequence[RestQuadHardNegative]
        ]
        | None = None,
    ):
        if not bool(getattr(tokenizer, "is_fast", False)):
            raise ValueError("Rest surface alignment requires a fast tokenizer")
        self.examples = tuple(examples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_pairs = int(max_pairs)
        self.pair_candidate_max_span_len = int(pair_candidate_max_span_len)
        self.span_proposal_max_len = int(span_proposal_max_len)
        self.max_span_proposals = int(max_span_proposals)
        self.candidate_seed = int(candidate_seed)
        if hard_negative_pairs_by_example is None:
            self.hard_negative_pairs_by_example = tuple(
                () for _ in self.examples
            )
        else:
            if len(hard_negative_pairs_by_example) != len(self.examples):
                raise ValueError(
                    "hard_negative_pairs_by_example must align one-to-one with "
                    "the training examples"
                )
            self.hard_negative_pairs_by_example = tuple(
                tuple(rows) for rows in hard_negative_pairs_by_example
            )
        self.alignment_stats = RestTokenAlignmentStats()
        self.alignment_issues: list[RestTokenAlignmentIssue] = []
        self.features = [
            self._build_feature(example, index)
            for index, example in enumerate(self.examples)
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> RestQuadFeature:
        return self.features[index]

    def _build_feature(
        self, example: RestQuadExample, example_index: int
    ) -> RestQuadFeature:
        stats = self.alignment_stats
        stats.examples_seen += 1
        stats.raw_quads_seen += len(example.annotations)
        stats.surface_representable_quads += len(example.representable_annotations)
        stats.surface_unrepresentable_quads += (
            len(example.annotations) - len(example.representable_annotations)
        )
        encoded = self.tokenizer(
            example.text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        input_ids = _as_int_list(encoded["input_ids"])
        attention_mask = _as_int_list(encoded["attention_mask"])
        offsets = _as_offsets(encoded["offset_mapping"])
        if len(input_ids) != self.max_length:
            raise ValueError("Tokenizer did not max-length pad Rest input")

        alignment_cache: dict[RestCharSpan, _TokenAlignment] = {}
        for annotation in example.representable_annotations:
            assert annotation.aspect_span is not None
            assert annotation.opinion_span is not None
            for role, span in (
                ("aspect", annotation.aspect_span),
                ("opinion", annotation.opinion_span),
            ):
                if span.is_null or span in alignment_cache:
                    continue
                stats.explicit_spans_seen += 1
                aligned = _align_char_span(
                    span,
                    text=example.text,
                    offsets=offsets,
                    attention_mask=attention_mask,
                )
                alignment_cache[span] = aligned
                if aligned.method == "exact":
                    stats.exact_token_spans += 1
                elif aligned.method == "boundary_adjusted":
                    stats.boundary_adjusted_token_spans += 1
                elif aligned.method == "truncated":
                    stats.truncated_token_spans += 1
                else:
                    stats.unaligned_token_spans += 1
                if aligned.token_span is None:
                    assert span.start is not None and span.end is not None
                    self.alignment_issues.append(
                        RestTokenAlignmentIssue(
                            example_index,
                            example.line_no,
                            role,
                            (span.start, span.end),
                            span.surface(example.text),
                            aligned.reason or aligned.method,
                        )
                    )

        null_candidate = _RoleCandidate((0, 1), is_null=True)
        gold_aspects: list[_RoleCandidate] = []
        gold_opinions: list[_RoleCandidate] = []
        gold_by_pair: dict[
            tuple[_RoleCandidate, _RoleCandidate], set[tuple[int, int]]
        ] = {}
        for annotation in example.representable_annotations:
            assert annotation.aspect_span is not None
            assert annotation.opinion_span is not None
            if annotation.aspect_span.is_null:
                aspect = null_candidate
            else:
                aspect_alignment = alignment_cache[annotation.aspect_span]
                if aspect_alignment.token_span is None:
                    stats.quads_dropped_for_token_alignment += 1
                    continue
                aspect = _RoleCandidate(aspect_alignment.token_span)
            if annotation.opinion_span.is_null:
                raise ValueError("Rest benchmark does not support NULL opinions")
            opinion_alignment = alignment_cache[annotation.opinion_span]
            if opinion_alignment.token_span is None:
                stats.quads_dropped_for_token_alignment += 1
                continue
            opinion = _RoleCandidate(opinion_alignment.token_span)
            _append_unique(gold_aspects, [aspect])
            _append_unique(gold_opinions, [opinion])
            label = (
                REST_CATEGORY_TO_ID[annotation.raw.category],
                REST_PAIR_LABEL_TO_ID[annotation.raw.sentiment],
            )
            gold_by_pair.setdefault((aspect, opinion), set()).add(label)
            stats.trainable_quads += 1

        # NULL is an aspect-only state in the official Rest benchmark.
        role_aspects = list(gold_aspects)
        role_opinions = list(gold_opinions)
        _append_unique(role_aspects, [null_candidate])
        enumerated = [
            _RoleCandidate(span)
            for span in _enumerate_token_spans(
                offsets,
                attention_mask,
                self.pair_candidate_max_span_len,
                example.text,
            )
        ]
        _append_unique(role_aspects, enumerated)
        _append_unique(role_opinions, enumerated)

        total_candidates = len(role_aspects) * len(role_opinions)
        positive_pairs = list(gold_by_pair)
        if len(positive_pairs) > self.max_pairs:
            raise ValueError("Positive Rest pairs exceed max_pairs")

        def candidate_surface(candidate: _RoleCandidate) -> str:
            if candidate.is_null:
                return "NULL"
            start, end = candidate.token_span
            char_start = int(offsets[start][0])
            char_end = int(offsets[end - 1][1])
            return example.text[char_start:char_end].strip()

        gold_surface_pairs = {
            (
                annotation.raw.aspect.strip().casefold(),
                annotation.raw.opinion.strip().casefold(),
            )
            for annotation in example.annotations
        }

        def is_surface_equivalent_gold(
            pair: tuple[_RoleCandidate, _RoleCandidate]
        ) -> bool:
            aspect, opinion = pair
            return (
                candidate_surface(aspect).casefold(),
                candidate_surface(opinion).casefold(),
            ) in gold_surface_pairs

        required_null_negatives = []
        for opinion in gold_opinions:
            pair = (null_candidate, opinion)
            if pair not in gold_by_pair:
                if is_surface_equivalent_gold(pair):
                    stats.surface_equivalent_negative_pairs_excluded += 1
                    continue
                _append_unique(required_null_negatives, [pair])
        if len(positive_pairs) + len(required_null_negatives) > self.max_pairs:
            raise ValueError("Positive and required NULL pairs exceed max_pairs")

        content_indices = set(
            _content_token_indices(offsets, attention_mask, example.text)
        )

        def validate_explicit_span(
            span: tuple[int, int], *, role: str
        ) -> None:
            start, end = span
            if (
                start < 0
                or end <= start
                or start not in content_indices
                or (end - 1) not in content_indices
            ):
                raise ValueError(
                    f"Invalid mined {role} token span {span} for example "
                    f"{example_index}"
                )

        mined_negatives = []
        mined_seen = set()
        mined_rows = self.hard_negative_pairs_by_example[example_index]
        stats.hard_negative_pairs_requested += len(mined_rows)
        for mined in mined_rows:
            a_start, a_end, o_start, o_end = tuple(
                int(value) for value in mined.pair_span
            )
            if (a_start, a_end) == (-1, -1):
                aspect = null_candidate
            else:
                validate_explicit_span((a_start, a_end), role="aspect")
                aspect = _RoleCandidate((a_start, a_end))
            if (o_start, o_end) == (-1, -1):
                raise ValueError("Mined NULL opinions are invalid for Rest data")
            validate_explicit_span((o_start, o_end), role="opinion")
            opinion = _RoleCandidate((o_start, o_end))
            pair = (aspect, opinion)
            if pair in gold_by_pair:
                raise ValueError(
                    "Hard-negative artifact contains a gold aspect-opinion pair "
                    f"for example {example_index}: {mined.pair_span}"
                )
            if is_surface_equivalent_gold(pair):
                raise ValueError(
                    "Hard-negative artifact contains a surface-equivalent gold "
                    f"pair for example {example_index}: {mined.pair_span}"
                )
            if pair in required_null_negatives:
                stats.hard_negative_pairs_already_required += 1
                continue
            if pair in mined_seen:
                stats.hard_negative_pairs_duplicate += 1
                continue
            mined_negatives.append(pair)
            mined_seen.add(pair)

        remaining_after_required = (
            self.max_pairs - len(positive_pairs) - len(required_null_negatives)
        )
        retained_mined_negatives = mined_negatives[:remaining_after_required]
        stats.hard_negative_pairs_retained += len(retained_mined_negatives)
        stats.hard_negative_pairs_dropped_by_cap += (
            len(mined_negatives) - len(retained_mined_negatives)
        )
        other_negatives = []
        required_set = set(required_null_negatives)
        mined_set = set(retained_mined_negatives)
        for aspect in role_aspects:
            for opinion in role_opinions:
                pair = (aspect, opinion)
                if (
                    pair in gold_by_pair
                    or pair in required_set
                    or pair in mined_set
                ):
                    continue
                if is_surface_equivalent_gold(pair):
                    stats.surface_equivalent_negative_pairs_excluded += 1
                    continue
                other_negatives.append(pair)
        # A deterministic left-to-right prefix exposes the pair head to only a
        # tiny and position-biased slice of the inference lattice.  Sample the
        # remaining negatives reproducibly per example while always retaining
        # every positive and the required NULL contrasts.
        random.Random(self.candidate_seed + example_index).shuffle(other_negatives)
        remaining = (
            self.max_pairs
            - len(positive_pairs)
            - len(required_null_negatives)
            - len(retained_mined_negatives)
        )
        other_negatives = other_negatives[:remaining]
        pairs = (
            positive_pairs
            + required_null_negatives
            + retained_mined_negatives
            + other_negatives
        )
        stats.pair_candidates_before_cap += total_candidates
        stats.pair_candidates_after_cap += len(pairs)

        pair_rows = []
        null_rows = []
        pair_labels = []
        category_targets = []
        category_target_mask = []
        for aspect, opinion in pairs:
            pair_rows.append([*aspect.token_span, *opinion.token_span])
            null_rows.append([aspect.is_null, opinion.is_null])
            category_target = torch.zeros(
                NUM_REST_CATEGORIES, dtype=torch.float32
            )
            labels = gold_by_pair.get((aspect, opinion), set())
            sentiments = {sentiment for _, sentiment in labels}
            if len(sentiments) > 1:
                raise ValueError(
                    "A Rest aspect-opinion pair has conflicting sentiment labels"
                )
            pair_labels.append(
                next(iter(sentiments))
                if sentiments
                else REST_PAIR_LABEL_TO_ID["NONE"]
            )
            for category, _ in labels:
                category_target[category] = 1.0
            category_targets.append(category_target)
            category_target_mask.append(bool(labels))
            if labels:
                stats.positive_pairs += 1
                stats.multi_label_positive_pairs += int(len(labels) > 1)
            else:
                stats.negative_pairs += 1

        explicit_aspects = [
            candidate.token_span for candidate in gold_aspects if not candidate.is_null
        ]
        explicit_opinions = [
            candidate.token_span for candidate in gold_opinions if not candidate.is_null
        ]
        (
            proposal_spans,
            proposal_aspect,
            proposal_opinion,
            proposal_mask,
            proposals_before,
        ) = _span_proposals(
            offsets=offsets,
            attention_mask=attention_mask,
            aspect_spans=explicit_aspects,
            opinion_spans=explicit_opinions,
            max_span_len=self.span_proposal_max_len,
            max_proposals=self.max_span_proposals,
            text=example.text,
        )
        stats.span_proposals_before_cap += proposals_before
        stats.span_proposals_after_cap += len(proposal_spans)

        return RestQuadFeature(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            aspect_bio_labels=_bio_labels(
                len(input_ids),
                offsets,
                attention_mask,
                explicit_aspects,
                example.text,
            ),
            opinion_bio_labels=_bio_labels(
                len(input_ids),
                offsets,
                attention_mask,
                explicit_opinions,
                example.text,
            ),
            pair_spans=(
                torch.tensor(pair_rows, dtype=torch.long)
                if pair_rows
                else torch.zeros((0, 4), dtype=torch.long)
            ),
            pair_null_mask=(
                torch.tensor(null_rows, dtype=torch.bool)
                if null_rows
                else torch.zeros((0, 2), dtype=torch.bool)
            ),
            pair_labels=torch.tensor(pair_labels, dtype=torch.long),
            category_targets=(
                torch.stack(category_targets)
                if category_targets
                else torch.zeros((0, NUM_REST_CATEGORIES), dtype=torch.float32)
            ),
            category_target_mask=torch.tensor(
                category_target_mask, dtype=torch.bool
            ),
            pair_mask=torch.ones((len(pairs),), dtype=torch.bool),
            span_proposal_spans=proposal_spans,
            span_aspect_labels=proposal_aspect,
            span_opinion_labels=proposal_opinion,
            span_proposal_mask=proposal_mask,
            example=example,
        )


class RestQuadCollator:
    def __call__(self, features: Sequence[RestQuadFeature]) -> dict[str, Any]:
        batch_size = len(features)
        max_pairs = max(len(feature.pair_spans) for feature in features)
        max_proposals = max(len(feature.span_proposal_spans) for feature in features)
        pair_spans = torch.zeros((batch_size, max_pairs, 4), dtype=torch.long)
        pair_null_mask = torch.zeros((batch_size, max_pairs, 2), dtype=torch.bool)
        pair_mask = torch.zeros((batch_size, max_pairs), dtype=torch.bool)
        pair_labels = torch.full(
            (batch_size, max_pairs), IGNORE_INDEX, dtype=torch.long
        )
        category_targets = torch.zeros(
            (batch_size, max_pairs, NUM_REST_CATEGORIES), dtype=torch.float32
        )
        category_target_mask = torch.zeros(
            (batch_size, max_pairs), dtype=torch.bool
        )
        proposal_spans = torch.zeros(
            (batch_size, max_proposals, 2), dtype=torch.long
        )
        proposal_aspect = torch.zeros((batch_size, max_proposals))
        proposal_opinion = torch.zeros((batch_size, max_proposals))
        proposal_mask = torch.zeros((batch_size, max_proposals), dtype=torch.bool)
        for row, feature in enumerate(features):
            pairs = len(feature.pair_spans)
            proposals = len(feature.span_proposal_spans)
            pair_spans[row, :pairs] = feature.pair_spans
            pair_null_mask[row, :pairs] = feature.pair_null_mask
            pair_mask[row, :pairs] = feature.pair_mask
            pair_labels[row, :pairs] = feature.pair_labels
            category_targets[row, :pairs] = feature.category_targets
            category_target_mask[row, :pairs] = feature.category_target_mask
            proposal_spans[row, :proposals] = feature.span_proposal_spans
            proposal_aspect[row, :proposals] = feature.span_aspect_labels
            proposal_opinion[row, :proposals] = feature.span_opinion_labels
            proposal_mask[row, :proposals] = feature.span_proposal_mask
        return {
            "input_ids": torch.stack([feature.input_ids for feature in features]),
            "attention_mask": torch.stack(
                [feature.attention_mask for feature in features]
            ),
            "aspect_bio_labels": torch.stack(
                [feature.aspect_bio_labels for feature in features]
            ),
            "opinion_bio_labels": torch.stack(
                [feature.opinion_bio_labels for feature in features]
            ),
            "pair_spans": pair_spans,
            "pair_null_mask": pair_null_mask,
            "pair_mask": pair_mask,
            "pair_labels": pair_labels,
            "category_targets": category_targets,
            "category_target_mask": category_target_mask,
            "span_proposal_spans": proposal_spans,
            "span_aspect_labels": proposal_aspect,
            "span_opinion_labels": proposal_opinion,
            "span_proposal_mask": proposal_mask,
            "examples": [feature.example for feature in features],
        }
