#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for ASTE-Data-V2 style Aspect Sentiment Triplet Extraction data.

Supported line formats:
  sentence####[(target position, opinion position, sentiment)]
  sentence#### #### ####[(target position, opinion position, sentiment)]

Triplets are normalized to token-index tuples so exact-match evaluation is
stable across the common "all indices" and "[start, end]" span conventions.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ASTE_DATASET_NAMES = {
    "14lap": "Laptop-14",
    "14res": "Restaurant-14",
    "15res": "Restaurant-15",
    "16res": "Restaurant-16",
}

ASTE_SPLIT_FILENAMES = {
    "train": ("train_triplets.txt", "train.txt"),
    "dev": ("dev_triplets.txt", "dev.txt"),
    "test": ("test_triplets.txt", "test.txt"),
}

SENTIMENT_ALIASES = {
    "POS": "POS",
    "POSITIVE": "POS",
    "1": "POS",
    "NEU": "NEU",
    "NEUTRAL": "NEU",
    "0": "NEU",
    "NEG": "NEG",
    "NEGATIVE": "NEG",
    "-1": "NEG",
}

ASTE_PAIR_LABEL_TO_ID = {"NONE": 0, "NEG": 1, "NEU": 2, "POS": 3}
ASTE_PAIR_ID_TO_LABEL = {v: k for k, v in ASTE_PAIR_LABEL_TO_ID.items()}

BIO_TAG_TO_ID = {"O": 0, "B": 1, "I": 2}
BIO_ID_TO_TAG = {v: k for k, v in BIO_TAG_TO_ID.items()}


@dataclass(frozen=True)
class ASTESpan:
    """A contiguous token span represented by exact token indices."""

    indices: tuple[int, ...]

    @classmethod
    def from_positions(cls, positions: Sequence[int] | int) -> "ASTESpan":
        if isinstance(positions, int):
            raw = [positions]
        else:
            raw = [int(x) for x in positions]

        if not raw:
            raise ValueError("ASTE span positions cannot be empty")
        if any(x < 0 for x in raw):
            raise ValueError(f"ASTE span positions must be non-negative: {raw}")

        if len(raw) == 1:
            normalized = tuple(raw)
        elif len(raw) == 2 and raw[1] >= raw[0]:
            # Span-ASTE documents multi-word spans as [start, end]. ASTE-Data-V2
            # often lists every index. Expanding two-point spans makes both exact.
            normalized = tuple(range(raw[0], raw[1] + 1))
        else:
            normalized = tuple(sorted(set(raw)))

        return cls(indices=normalized)

    @property
    def start(self) -> int:
        return self.indices[0]

    @property
    def end_exclusive(self) -> int:
        return self.indices[-1] + 1

    @property
    def length(self) -> int:
        return len(self.indices)

    def text(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens[i] for i in self.indices)

    def as_tuple(self) -> tuple[int, ...]:
        return self.indices


@dataclass(frozen=True)
class ASTETriplet:
    """One ASTE target-opinion-sentiment triplet."""

    aspect: ASTESpan
    opinion: ASTESpan
    sentiment: str

    def as_tuple(self) -> tuple[tuple[int, ...], tuple[int, ...], str]:
        return (self.aspect.as_tuple(), self.opinion.as_tuple(), self.sentiment)


@dataclass(frozen=True)
class ASTESentence:
    """A tokenized sentence plus zero or more ASTE triplets."""

    text: str
    tokens: tuple[str, ...]
    triplets: tuple[ASTETriplet, ...]
    source_path: str | None = None
    line_no: int | None = None

    def triplet_set(self) -> set[tuple[tuple[int, ...], tuple[int, ...], str]]:
        return {triplet.as_tuple() for triplet in self.triplets}


@dataclass(frozen=True)
class ASTETripletMetrics:
    precision: float
    recall: float
    f1: float
    correct: int
    predicted: int
    gold: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "correct": self.correct,
            "predicted": self.predicted,
            "gold": self.gold,
        }


@dataclass(frozen=True)
class ASTESpanMetrics:
    precision: float
    recall: float
    f1: float
    correct: int
    predicted: int
    gold: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "correct": self.correct,
            "predicted": self.predicted,
            "gold": self.gold,
        }


def normalize_sentiment(value: object) -> str:
    key = str(value).strip().upper()
    if key not in SENTIMENT_ALIASES:
        raise ValueError(f"Unsupported ASTE sentiment label: {value!r}")
    return SENTIMENT_ALIASES[key]


def _error_context(path: str | Path | None, line_no: int | None) -> str:
    parts = []
    if path is not None:
        parts.append(str(path))
    if line_no is not None:
        parts.append(f"line {line_no}")
    return " ".join(parts) if parts else "ASTE line"


def parse_aste_line(
    line: str,
    *,
    source_path: str | Path | None = None,
    line_no: int | None = None,
) -> ASTESentence:
    raw_line = line.rstrip("\n")
    if not raw_line.strip():
        raise ValueError(f"{_error_context(source_path, line_no)} is empty")

    parts = raw_line.split("####")
    if len(parts) < 2:
        raise ValueError(
            f"{_error_context(source_path, line_no)} must contain '####': {raw_line!r}"
        )

    sentence = parts[0].strip()
    triplet_literal = parts[-1].strip() or "[]"
    tokens = tuple(sentence.split())

    try:
        raw_triplets = ast.literal_eval(triplet_literal)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"{_error_context(source_path, line_no)} has invalid triplet literal: "
            f"{triplet_literal!r}"
        ) from exc

    if not isinstance(raw_triplets, (list, tuple)):
        raise ValueError(
            f"{_error_context(source_path, line_no)} triplets must be a list/tuple"
        )

    triplets: list[ASTETriplet] = []
    for raw_triplet in raw_triplets:
        if not isinstance(raw_triplet, (list, tuple)) or len(raw_triplet) != 3:
            raise ValueError(
                f"{_error_context(source_path, line_no)} has malformed triplet: "
                f"{raw_triplet!r}"
            )

        aspect_pos, opinion_pos, sentiment_value = raw_triplet
        aspect = ASTESpan.from_positions(aspect_pos)
        opinion = ASTESpan.from_positions(opinion_pos)
        sentiment = normalize_sentiment(sentiment_value)

        max_index = len(tokens) - 1
        if aspect.indices[-1] > max_index or opinion.indices[-1] > max_index:
            raise ValueError(
                f"{_error_context(source_path, line_no)} span exceeds token count "
                f"({len(tokens)}): {raw_triplet!r}"
            )

        triplets.append(
            ASTETriplet(aspect=aspect, opinion=opinion, sentiment=sentiment)
        )

    return ASTESentence(
        text=sentence,
        tokens=tokens,
        triplets=tuple(triplets),
        source_path=str(source_path) if source_path is not None else None,
        line_no=line_no,
    )


def read_aste_file(path: str | Path) -> list[ASTESentence]:
    data_path = Path(path)
    examples: list[ASTESentence] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            examples.append(
                parse_aste_line(line, source_path=data_path, line_no=line_no)
            )
    return examples


def resolve_aste_split_path(root: str | Path, dataset: str, split: str) -> Path:
    dataset_key = dataset.strip()
    split_key = split.strip().lower()
    if split_key not in ASTE_SPLIT_FILENAMES:
        raise ValueError(f"Unsupported ASTE split={split!r}")

    dataset_dir = Path(root) / dataset_key
    for filename in ASTE_SPLIT_FILENAMES[split_key]:
        candidate = dataset_dir / filename
        if candidate.exists():
            return candidate

    expected = ", ".join(ASTE_SPLIT_FILENAMES[split_key])
    raise FileNotFoundError(
        f"Could not find ASTE {dataset_key}/{split_key} under {dataset_dir}. "
        f"Expected one of: {expected}"
    )


def read_aste_split(root: str | Path, dataset: str, split: str) -> list[ASTESentence]:
    return read_aste_file(resolve_aste_split_path(root, dataset, split))


def score_aste_examples(
    predicted: Sequence[ASTESentence],
    gold: Sequence[ASTESentence],
    *,
    check_text: bool = True,
) -> ASTETripletMetrics:
    if len(predicted) != len(gold):
        raise ValueError(
            f"Prediction/gold length mismatch: {len(predicted)} vs {len(gold)}"
        )

    correct = 0
    predicted_total = 0
    gold_total = 0

    for i, (pred_item, gold_item) in enumerate(zip(predicted, gold)):
        if check_text and pred_item.tokens != gold_item.tokens:
            raise ValueError(
                f"Prediction/gold token mismatch at example {i}: "
                f"{pred_item.text!r} vs {gold_item.text!r}"
            )
        pred_set = pred_item.triplet_set()
        gold_set = gold_item.triplet_set()
        correct += len(pred_set & gold_set)
        predicted_total += len(pred_set)
        gold_total += len(gold_set)

    precision = correct / predicted_total if predicted_total else 0.0
    recall = correct / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return ASTETripletMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        correct=correct,
        predicted=predicted_total,
        gold=gold_total,
    )


def score_aste_files(
    pred_path: str | Path,
    gold_path: str | Path,
    *,
    check_text: bool = True,
) -> ASTETripletMetrics:
    return score_aste_examples(
        read_aste_file(pred_path),
        read_aste_file(gold_path),
        check_text=check_text,
    )


def gold_spans_for_example(example: ASTESentence, kind: str) -> tuple[ASTESpan, ...]:
    kind_key = kind.strip().lower()
    if kind_key not in {"aspect", "opinion"}:
        raise ValueError(f"Unsupported ASTE span kind={kind!r}; use aspect/opinion")

    spans = {
        triplet.aspect if kind_key == "aspect" else triplet.opinion
        for triplet in example.triplets
    }
    return tuple(sorted(spans, key=lambda span: (span.start, span.end_exclusive)))


def bio_tags_for_spans(
    token_count: int,
    spans: Iterable[ASTESpan],
    *,
    allow_overlap: bool = False,
) -> list[str]:
    tags = ["O"] * int(token_count)
    for span in sorted(spans, key=lambda item: (item.start, item.end_exclusive)):
        if span.end_exclusive > token_count:
            raise ValueError(
                f"Span {span.indices} exceeds token_count={token_count}"
            )
        for offset, token_idx in enumerate(span.indices):
            next_tag = "B" if offset == 0 else "I"
            if tags[token_idx] != "O" and not allow_overlap:
                raise ValueError(
                    f"Overlapping spans cannot be represented as BIO tags: "
                    f"token={token_idx}, current={tags[token_idx]}, new={next_tag}"
                )
            if tags[token_idx] == "O" or allow_overlap:
                tags[token_idx] = next_tag
    return tags


def bio_tag_ids_for_spans(
    token_count: int,
    spans: Iterable[ASTESpan],
    *,
    allow_overlap: bool = False,
) -> list[int]:
    return [
        BIO_TAG_TO_ID[tag]
        for tag in bio_tags_for_spans(
            token_count, spans, allow_overlap=allow_overlap
        )
    ]


def extract_spans_from_bio(tags: Sequence[str | int]) -> tuple[ASTESpan, ...]:
    normalized_tags: list[str] = []
    for tag in tags:
        if isinstance(tag, int):
            normalized_tags.append(BIO_ID_TO_TAG[int(tag)])
        else:
            tag_key = str(tag).strip().upper()
            if tag_key not in BIO_TAG_TO_ID:
                raise ValueError(f"Unsupported BIO tag: {tag!r}")
            normalized_tags.append(tag_key)

    spans: list[ASTESpan] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush_current() -> None:
        nonlocal current_start, current_end
        if current_start is not None and current_end is not None:
            spans.append(ASTESpan.from_positions([current_start, current_end]))
        current_start = None
        current_end = None

    for idx, tag in enumerate(normalized_tags):
        if tag == "B":
            flush_current()
            current_start = idx
            current_end = idx
        elif tag == "I":
            if current_start is None:
                # Robustly treat stray I as a new span. This mirrors common BIO
                # decoding behavior and avoids losing recoverable predictions.
                current_start = idx
            current_end = idx
        else:
            flush_current()

    flush_current()
    return tuple(spans)


def bio_tags_for_example(example: ASTESentence, kind: str) -> list[str]:
    return bio_tags_for_spans(
        len(example.tokens),
        gold_spans_for_example(example, kind),
    )


def score_span_predictions(
    predicted: Sequence[Iterable[ASTESpan]],
    gold: Sequence[Iterable[ASTESpan]],
) -> ASTESpanMetrics:
    if len(predicted) != len(gold):
        raise ValueError(
            f"Prediction/gold length mismatch: {len(predicted)} vs {len(gold)}"
        )

    correct = 0
    predicted_total = 0
    gold_total = 0
    for pred_spans, gold_spans in zip(predicted, gold):
        pred_set = {span.as_tuple() for span in pred_spans}
        gold_set = {span.as_tuple() for span in gold_spans}
        correct += len(pred_set & gold_set)
        predicted_total += len(pred_set)
        gold_total += len(gold_set)

    precision = correct / predicted_total if predicted_total else 0.0
    recall = correct / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return ASTESpanMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        correct=correct,
        predicted=predicted_total,
        gold=gold_total,
    )


def compute_aste_statistics(examples: Sequence[ASTESentence]) -> dict[str, object]:
    sentiment_counts: Counter[str] = Counter()
    triplets_per_sentence: list[int] = []
    token_lengths: list[int] = []
    aspect_lengths: list[int] = []
    opinion_lengths: list[int] = []
    unique_aspect_counts: list[int] = []
    unique_opinion_counts: list[int] = []
    overlap_triplets = 0

    for example in examples:
        token_lengths.append(len(example.tokens))
        triplets_per_sentence.append(len(example.triplets))

        aspect_set = set()
        opinion_set = set()
        for triplet in example.triplets:
            sentiment_counts[triplet.sentiment] += 1
            aspect_set.add(triplet.aspect.as_tuple())
            opinion_set.add(triplet.opinion.as_tuple())
            aspect_lengths.append(triplet.aspect.length)
            opinion_lengths.append(triplet.opinion.length)
            if set(triplet.aspect.indices) & set(triplet.opinion.indices):
                overlap_triplets += 1

        unique_aspect_counts.append(len(aspect_set))
        unique_opinion_counts.append(len(opinion_set))

    sentence_count = len(examples)
    triplet_count = sum(triplets_per_sentence)

    def mean(values: Sequence[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def rate(count: int) -> float:
        return float(count / sentence_count) if sentence_count else 0.0

    return {
        "sentences": sentence_count,
        "triplets": triplet_count,
        "sentiment_POS": int(sentiment_counts["POS"]),
        "sentiment_NEU": int(sentiment_counts["NEU"]),
        "sentiment_NEG": int(sentiment_counts["NEG"]),
        "empty_sentence_count": sum(1 for n in triplets_per_sentence if n == 0),
        "empty_sentence_rate": rate(sum(1 for n in triplets_per_sentence if n == 0)),
        "multi_triplet_sentence_count": sum(1 for n in triplets_per_sentence if n > 1),
        "multi_triplet_sentence_rate": rate(
            sum(1 for n in triplets_per_sentence if n > 1)
        ),
        "multi_aspect_sentence_count": sum(1 for n in unique_aspect_counts if n > 1),
        "multi_aspect_sentence_rate": rate(sum(1 for n in unique_aspect_counts if n > 1)),
        "multi_opinion_sentence_count": sum(1 for n in unique_opinion_counts if n > 1),
        "multi_opinion_sentence_rate": rate(
            sum(1 for n in unique_opinion_counts if n > 1)
        ),
        "avg_tokens_per_sentence": mean(token_lengths),
        "avg_triplets_per_sentence": mean(triplets_per_sentence),
        "avg_aspects_per_sentence": mean(unique_aspect_counts),
        "avg_opinions_per_sentence": mean(unique_opinion_counts),
        "avg_aspect_span_len": mean(aspect_lengths),
        "avg_opinion_span_len": mean(opinion_lengths),
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "max_triplets_per_sentence": max(triplets_per_sentence)
        if triplets_per_sentence
        else 0,
        "overlap_triplet_count": overlap_triplets,
    }


def iter_aste_splits(
    root: str | Path,
    datasets: Iterable[str] = ASTE_DATASET_NAMES.keys(),
    splits: Iterable[str] = ASTE_SPLIT_FILENAMES.keys(),
) -> Iterable[tuple[str, str, Path, list[ASTESentence]]]:
    for dataset in datasets:
        for split in splits:
            path = resolve_aste_split_path(root, dataset, split)
            yield dataset, split, path, read_aste_file(path)


def rows_to_markdown(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
